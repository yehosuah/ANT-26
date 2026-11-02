"""Masked telemetry learning with scoped normalization and prefix-only prediction.

The reconstruction mask covers usable source-derived grid values, including
permitted short-gap interpolation. It never treats padded/missing values as
targets. This is a compact adapted architecture, not a PatchTST reproduction.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ant26.domain.protocol import protocol_hash
from ant26.evaluation.splits import validate_scope
from ant26.models.baselines import NUMERIC_FEATURES, feature_matrix, valid_examples

CHANNELS = ("speed", "throttle", "brake", "rpm", "n_gear")
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


@dataclass(frozen=True)
class EncoderConfig:
    width: int = 128
    layers: int = 4
    heads: int = 4
    patch_size: int = 8
    max_steps: int = 128
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("width", "layers", "heads", "patch_size", "max_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must lie in [0, 1)")


def _sequences(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("No examples supplied")
    for row in examples:
        if tuple(row["features"].get("sequence_channels", ())) != CHANNELS:
            raise ValueError("Sequence channel order does not match the artifact schema")
    values = np.asarray([row["features"]["sequence"] for row in examples], dtype=np.float32)
    mask = np.asarray([row["features"]["sequence_mask"] for row in examples], dtype=bool)
    if values.ndim != 3 or values.shape[-1] != len(CHANNELS) or values.shape != mask.shape:
        raise ValueError("Sequences and masks must have shape [batch, time, five channels]")
    mask &= np.isfinite(values)
    return np.where(mask, values, 0.0), mask


class MaskedPatchEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        if config.width % config.heads or config.max_steps % config.patch_size:
            raise ValueError(
                "Width must be divisible by heads and patch size must divide max steps"
            )
        self.config = config
        self.projection = nn.Linear(config.patch_size * len(CHANNELS) * 2, config.width)
        self.position = nn.Parameter(
            torch.zeros(1, config.max_steps // config.patch_size, config.width)
        )
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.width * 2,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.continuous_decoder = nn.Linear(config.width, config.patch_size * len(CHANNELS))
        self.brake_decoder = nn.Linear(config.width, config.patch_size * 2)
        self.gear_decoder = nn.Linear(config.width, config.patch_size * 9)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels = values.shape
        patch = self.config.patch_size
        if steps != self.config.max_steps:
            raise ValueError(f"Expected {self.config.max_steps} time steps")
        packed = torch.cat((values * mask, mask.to(values.dtype)), dim=-1)
        packed = packed.reshape(batch, steps // patch, patch * channels * 2)
        hidden = self.norm(self.transformer(self.projection(packed) + self.position))
        patch_valid = mask.reshape(batch, steps // patch, patch * channels).any(dim=-1)
        pooled = (hidden * patch_valid[..., None]).sum(dim=1) / patch_valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return hidden, pooled

    def masked_loss(
        self,
        normalized: torch.Tensor,
        original: torch.Tensor,
        valid: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        patch = self.config.patch_size
        batch, steps, _ = normalized.shape
        masked_steps = patch_mask.repeat_interleave(patch, dim=1)[..., None]
        keep = valid & ~masked_steps
        hidden, _ = self(normalized.masked_fill(masked_steps, 0), keep)
        selected = valid & masked_steps
        reconstruction = self.continuous_decoder(hidden).reshape(batch, steps, len(CHANNELS))
        continuous_indices = [0, 1, 3]
        continuous_mask = selected[..., continuous_indices]
        continuous = F.smooth_l1_loss(
            reconstruction[..., continuous_indices],
            normalized[..., continuous_indices],
            reduction="none",
        )
        loss = (continuous * continuous_mask).sum() / continuous_mask.sum().clamp_min(1)
        for channel, decoder, classes, labels in (
            (2, self.brake_decoder, 2, (original[..., 2] > 0).long()),
            (4, self.gear_decoder, 9, original[..., 4].long().clamp(0, 8)),
        ):
            logits = decoder(hidden).reshape(batch, steps, classes)
            discrete = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
            mask = selected[..., channel]
            loss = loss + (discrete * mask).sum() / mask.sum().clamp_min(1)
        return loss


class QuantileHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width + len(NUMERIC_FEATURES), 64), nn.GELU(), nn.Linear(64, 5)
        )
        nn.init.normal_(self.network[-1].weight, std=0.01)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([0.0, -3.0, -3.0, -3.0, -3.0]))

    def forward(self, embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        raw = self.network(torch.cat((embedding, context), dim=1))
        median = raw[:, 0]
        delta = F.softplus(raw[:, 1:])
        return torch.stack(
            (
                median - delta[:, 0] - delta[:, 1],
                median - delta[:, 0],
                median,
                median + delta[:, 2],
                median + delta[:, 2] + delta[:, 3],
            ),
            dim=1,
        )


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = prediction.new_tensor(QUANTILES)
    residual = target[:, None] - prediction
    return torch.maximum(probabilities * residual, (probabilities - 1) * residual).mean()


class RepresentationBundle:
    def __init__(self, encoder: MaskedPatchEncoder, head: QuantileHead, metadata: dict):
        self.encoder, self.head, self.metadata = encoder, head, metadata

    def _inputs(self, examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, mask = _sequences(examples)
        mean = np.asarray(self.metadata["sequence_mean"], dtype=np.float32)
        scale = np.asarray(self.metadata["sequence_scale"], dtype=np.float32)
        normalized = np.where(mask, (values - mean) / scale, 0).astype(np.float32)
        context = feature_matrix(examples)
        context_mean = np.asarray(self.metadata["context_mean"])
        context_scale = np.asarray(self.metadata["context_scale"])
        context = (
            (np.where(np.isfinite(context), context, context_mean) - context_mean) / context_scale
        ).astype(np.float32)
        return torch.from_numpy(normalized), torch.from_numpy(mask), torch.from_numpy(context)

    def encode(self, examples: list[dict]) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            values, mask, _ = self._inputs(examples)
            return self.encoder(values, mask)[1].cpu().numpy()

    def _raw_predict(self, examples: list[dict]) -> list[list[float]]:
        if not examples:
            return []
        if any(not row.get("eligible") for row in examples):
            raise ValueError("Predict only eligible rows; emit abstentions separately")
        references = np.asarray([row["reference_seconds"] for row in examples], dtype=float)
        if not np.isfinite(references).all() or (references <= 0).any():
            raise ValueError("Reference durations must be positive and finite")
        self.encoder.eval()
        self.head.eval()
        with torch.no_grad():
            values, mask, context = self._inputs(examples)
            encoded = self.encoder(values, mask)[1]
            log_ratio = self.head(encoded, context).clamp(-8, 8)
            return (np.exp(log_ratio.cpu().numpy()) * references[:, None]).tolist()

    def predict(self, examples: list[dict]) -> list[list[float]]:
        if not examples:
            return []
        quantiles = np.asarray(self._raw_predict(examples))
        offset80, offset90 = self.metadata.get("calibration_offsets_seconds", [0.0, 0.0])
        quantiles[:, 1] -= offset80
        quantiles[:, 3] += offset80
        quantiles[:, 0] -= offset90
        quantiles[:, 4] += offset90
        quantiles[:, 0] = np.minimum(quantiles[:, 0], quantiles[:, 1])
        quantiles[:, 4] = np.maximum(quantiles[:, 4], quantiles[:, 3])
        return np.maximum(quantiles, 0.001).tolist()

    def calibrate(
        self, examples: list[dict], *, calibration_sessions: set[int]
    ) -> RepresentationBundle:
        """Adjust intervals using a designated preceding calibration partition.

        Serial dependence and track shift prevent an independent-exchangeable
        coverage guarantee. This leaves encoder, head, and normalization fixed.
        """
        calibration_sessions = set(calibration_sessions)
        learned_sessions = set(self.metadata["fit_sessions"]) | set(
            self.metadata.get("tuning_sessions", [])
        )
        if learned_sessions.intersection(calibration_sessions):
            raise ValueError("Calibration must be disjoint from fitting and tuning sessions")
        validate_scope(examples, calibration_sessions, role="representation calibration")
        rows = valid_examples(examples)
        if len(rows) < 10:
            raise ValueError("Calibration requires at least ten valid examples")
        raw = np.asarray(self._raw_predict(rows))
        targets = np.asarray([row["target_seconds"] for row in rows])
        offsets = []
        for alpha, lo, hi in ((0.2, 1, 3), (0.1, 0, 4)):
            scores = np.maximum(raw[:, lo] - targets, targets - raw[:, hi])
            rank = min(len(rows), math.ceil((len(rows) + 1) * (1 - alpha)))
            offsets.append(max(0.0, float(np.sort(scores)[rank - 1])))
        self.metadata.update(
            calibrated=True,
            calibration_sessions=sorted(calibration_sessions),
            calibration_example_count=len(rows),
            calibration_offsets_seconds=offsets,
            calibration_method="empirical_split_conformal_widening_no_exchangeability_guarantee",
            calibration_examples_sha256=hashlib.sha256(
                json.dumps(rows, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        )
        return self

    def save(self, directory: str | Path) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "model.pt"
        metadata_path = directory / "metadata.json"
        if checkpoint.exists() or metadata_path.exists():
            raise FileExistsError("Model artifacts are immutable; choose a new artifact directory")
        torch.save(
            {"encoder": self.encoder.state_dict(), "head": self.head.state_dict()}, checkpoint
        )
        metadata = {
            **self.metadata,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        return {"checkpoint": str(checkpoint), "metadata": str(metadata_path)}

    @classmethod
    def load(cls, directory: str | Path) -> RepresentationBundle:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("schema_version") != "ant26-representation-v1":
            raise ValueError("Unsupported representation artifact schema")
        if metadata.get("normalization_policy") != "constant_unit_scale_v2":
            raise ValueError(
                "Unsupported legacy normalization policy; retrain this checkpoint before prediction"
            )
        checkpoint = directory / "model.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
            raise ValueError("Model checkpoint checksum mismatch")
        encoder = MaskedPatchEncoder(EncoderConfig(**metadata["encoder_config"]))
        head = QuantileHead(encoder.config.width)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state["encoder"])
        head.load_state_dict(state["head"])
        return cls(encoder, head, metadata)


def train_representation(
    examples: list[dict],
    fit_sessions: set[int],
    *,
    protocol: dict | None = None,
    tuning_examples: list[dict] | None = None,
    tuning_sessions: set[int] | None = None,
    mode: str = "finetune",
    config: EncoderConfig | None = None,
    pretrain_epochs: int = 3,
    forecast_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 17,
    data_kind: str = "unspecified",
) -> RepresentationBundle:
    """Fit a CPU candidate, explicitly separating tuning and fitting scopes.

    Tuning examples are validated but never added to normalization, pretraining,
    or fitting. Their forecast loss is logged after fitting. For this first
    scaffold, epoch counts are caller-selected; no automatic early stopping or
    calibration is claimed.
    """
    started = time.perf_counter()
    validate_scope(examples, set(fit_sessions), role="representation fitting")
    if mode not in ("finetune", "frozen", "scratch"):
        raise ValueError("mode must be finetune, frozen or scratch")
    if tuning_examples is not None:
        tuning_sessions = set(tuning_sessions or ())
        if set(fit_sessions).intersection(tuning_sessions):
            raise ValueError("Tuning and fitting sessions must be disjoint")
        validate_scope(tuning_examples, tuning_sessions, role="representation tuning")
    if pretrain_epochs < 0 or forecast_epochs < 1 or batch_size < 1:
        raise ValueError("Invalid training budget")
    budget = (protocol or {}).get("experiments", {})
    if pretrain_epochs > budget.get(
        "maximum_pretraining_epochs", 30
    ) or forecast_epochs > budget.get("maximum_fine_tuning_epochs", 50):
        raise ValueError("Training epochs exceed the protocol budget")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Learning rate must be positive and finite")
    rows = valid_examples(examples)
    if len(rows) < 2:
        raise ValueError("At least two eligible labeled examples are required")
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = config or EncoderConfig()
    values, mask = _sequences(rows)
    if values.shape[1] != config.max_steps:
        raise ValueError("Sequence length must match encoder configuration")
    channel_count = mask.sum(axis=(0, 1))
    if (channel_count == 0).any():
        raise ValueError("Every sequence channel needs usable fitting observations")
    mean = (values * mask).sum(axis=(0, 1), dtype=np.float64) / channel_count
    variance = (((values - mean) ** 2) * mask).sum(axis=(0, 1)) / channel_count
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    context = feature_matrix(rows)
    observed_context = np.isfinite(context)
    context_mean = np.where(observed_context, context, 0).sum(axis=0) / observed_context.sum(
        axis=0
    ).clip(1)
    clean = np.where(observed_context, context, context_mean)
    context_scale = clean.std(axis=0)
    context_scale[context_scale < 1e-6] = 1.0
    encoder, head = MaskedPatchEncoder(config), QuantileHead(config.width)
    parameters = sum(item.numel() for item in encoder.parameters())
    maximum = (protocol or {}).get("experiments", {}).get("maximum_encoder_parameters", 2_000_000)
    if parameters > maximum:
        raise ValueError("Encoder exceeds the protocol parameter budget")
    metadata = {
        "schema_version": "ant26-representation-v1",
        "normalization_policy": "constant_unit_scale_v2",
        "model_type": "masked_patch_encoder_quantile_head",
        "encoder_config": asdict(config),
        "encoder_parameters": parameters,
        "fit_sessions": sorted(fit_sessions),
        "tuning_sessions": sorted(tuning_sessions or ()),
        "mode": mode,
        "seed": seed,
        "data_kind": data_kind,
        "synthetic": data_kind.startswith("synthetic"),
        "quantiles": list(QUANTILES),
        "sequence_channels": list(CHANNELS),
        "numeric_features": list(NUMERIC_FEATURES),
        "sequence_mean": mean.tolist(),
        "sequence_scale": scale.tolist(),
        "context_mean": context_mean.tolist(),
        "context_scale": context_scale.tolist(),
        "reconstruction_targets": "usable_grid_values_including_permitted_short_gap_interpolation",
        "calibrated": False,
        "device": "cpu",
        "fitting_example_count": len(rows),
        "fitting_examples_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "protocol_sha256": protocol_hash(protocol or {}),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "pretrain_epochs": 0 if mode == "scratch" else pretrain_epochs,
        "forecast_epochs": forecast_epochs,
    }
    bundle = RepresentationBundle(encoder, head, metadata)
    normalized, valid, context_tensor = bundle._inputs(rows)
    original = torch.from_numpy(values)
    target = torch.tensor(
        [math.log(row["target_seconds"] / row["reference_seconds"]) for row in rows],
        dtype=torch.float32,
    )
    fixed_mask = torch.rand((len(rows), config.max_steps // config.patch_size)) < 0.3
    fixed_mask[:, 0] = True
    encoder.eval()
    with torch.no_grad():
        before = float(encoder.masked_loss(normalized, original, valid, fixed_mask))
    reconstruction_losses = []
    if mode != "scratch":
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
        for _ in range(pretrain_epochs):
            encoder.train()
            losses = []
            for indices in torch.randperm(len(rows)).split(batch_size):
                selection = torch.rand((len(indices), config.max_steps // config.patch_size)) < 0.3
                selection[:, 0] = True
                loss = encoder.masked_loss(
                    normalized[indices], original[indices], valid[indices], selection
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            reconstruction_losses.append(float(np.mean(losses)))
    encoder.eval()
    with torch.no_grad():
        after = float(encoder.masked_loss(normalized, original, valid, fixed_mask))
    if mode == "frozen":
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    parameters_to_fit = [
        parameter
        for parameter in list(encoder.parameters()) + list(head.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters_to_fit, lr=learning_rate)
    forecast_losses = []
    for _ in range(forecast_epochs):
        encoder.train(mode != "frozen")
        head.train()
        losses = []
        for indices in torch.randperm(len(rows)).split(batch_size):
            embedding = encoder(normalized[indices], valid[indices])[1]
            loss = quantile_loss(head(embedding, context_tensor[indices]), target[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters_to_fit, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        forecast_losses.append(float(np.mean(losses)))
    metadata.update(
        reconstruction_loss_before=before,
        reconstruction_loss_after=after,
        reconstruction_loss_history=reconstruction_losses,
        forecast_loss_history=forecast_losses,
    )
    if tuning_examples is not None:
        tuning_rows = valid_examples(tuning_examples)
        if tuning_rows:
            predictions = np.asarray(bundle.predict(tuning_rows))[:, 2]
            metadata["tuning_mae_seconds"] = float(
                np.mean(
                    np.abs(predictions - np.asarray([row["target_seconds"] for row in tuning_rows]))
                )
            )
    encoder.eval()
    head.eval()
    metadata["training_duration_seconds"] = time.perf_counter() - started
    return bundle
