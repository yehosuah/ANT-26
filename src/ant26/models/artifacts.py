"""Checksummed baseline candidates without Python object deserialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ant26.domain.protocol import protocol_hash
from ant26.evaluation.splits import rolling_splits, validate_scope
from ant26.models.baselines import NUMERIC_FEATURES, QUANTILES, Baseline, valid_examples

SCHEMA = "ant26-baseline-bundle-v1"


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_code_sha() -> str:
    return _file_sha(Path(__file__).with_name("baselines.py"))


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name} array")
    return array


class BaselineBundle:
    """A fitted baseline plus a small immutable, inspectable research manifest."""

    def __init__(self, model: Baseline, metadata: dict | None = None):
        if not model.fitted:
            raise ValueError("Only fitted baselines can become artifacts")
        if model.fit_sessions.intersection(model.calibration_sessions):
            raise ValueError("Fitting and calibration scopes overlap")
        validate_scope([], set(model.fit_sessions), role="baseline artifact fitting scope")
        if model.calibration_sessions:
            validate_scope(
                [], set(model.calibration_sessions), role="baseline artifact calibration scope"
            )
        self.model = model
        self.metadata = {
            **(metadata or {}),
            "schema_version": SCHEMA,
            "kind": model.kind,
            "numeric_features": list(NUMERIC_FEATURES),
            "quantiles": list(QUANTILES),
            "fit_sessions": sorted(model.fit_sessions),
            "calibration_sessions": sorted(model.calibration_sessions),
            "calibrated": bool(model.calibration_sessions),
            "baseline_code_sha256": _baseline_code_sha(),
        }

    def predict(self, examples: list[dict]) -> list[list[float]]:
        for row in examples:
            reference = row.get("reference_seconds")
            if (
                not isinstance(reference, (int, float))
                or not math.isfinite(reference)
                or reference <= 0
            ):
                raise ValueError("Reference duration must be positive and finite")
        return self.model.predict(examples)

    def save(self, directory: str | Path) -> dict:
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Baseline artifacts are immutable; choose a new directory")
        directory.mkdir(parents=True, exist_ok=True)
        residuals = _array(self.model.residual_quantiles, (5,), "residual quantile")
        offsets = _array(self.model.calibration_offsets, (2,), "calibration offset")
        if np.any(offsets < 0) or np.any(np.diff(residuals) < 0):
            raise ValueError("Invalid calibration offsets or crossing residual quantiles")
        state = {
            "kind": self.model.kind,
            "ridge_alpha": self.model.ridge_alpha,
            "residual_quantiles": residuals.tolist(),
            "calibration_offsets": offsets.tolist(),
            "fit_sessions": sorted(self.model.fit_sessions),
            "calibration_sessions": sorted(self.model.calibration_sessions),
        }
        filenames = ["state.json"]
        if self.model.kind == "regularized_linear_residual":
            state["location"] = _array(
                self.model.location, (len(NUMERIC_FEATURES),), "location"
            ).tolist()
            scale = _array(self.model.scale, (len(NUMERIC_FEATURES),), "scale")
            if np.any(scale <= 0):
                raise ValueError("Linear scales must be positive")
            state["scale"] = scale.tolist()
            state["weights"] = _array(
                self.model.weights, (len(NUMERIC_FEATURES) + 1,), "weights"
            ).tolist()
        if self.model.kind == "catboost_quantile":
            if len(self.model.trees) != len(QUANTILES):
                raise ValueError("Expected one native CatBoost model per quantile")
            for index, tree in enumerate(self.model.trees):
                name = f"quantile-{index}.cbm"
                tree.save_model(str(directory / name), format="cbm")
                filenames.append(name)
        with (directory / "state.json").open("x") as handle:
            json.dump(state, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        metadata = {
            **self.metadata,
            "artifact_files": {name: _file_sha(directory / name) for name in filenames},
            "serialization": "validated_json_and_native_catboost_no_pickle",
        }
        metadata["metadata_sha256"] = _sha(metadata)
        with (directory / "metadata.json").open("x") as handle:
            json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        return {
            "directory": str(directory),
            "metadata": str(directory / "metadata.json"),
            "metadata_sha256": metadata["metadata_sha256"],
        }

    @classmethod
    def load(cls, directory: str | Path) -> BaselineBundle:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        expected_metadata_sha = metadata.pop("metadata_sha256", None)
        if _sha(metadata) != expected_metadata_sha:
            raise ValueError("Baseline metadata checksum mismatch")
        if metadata.get("schema_version") != SCHEMA:
            raise ValueError("Unsupported baseline artifact schema")
        if metadata.get("numeric_features") != list(NUMERIC_FEATURES) or metadata.get(
            "quantiles"
        ) != list(QUANTILES):
            raise ValueError("Baseline artifact feature or quantile schema differs from runtime")
        if metadata.get("baseline_code_sha256") != _baseline_code_sha():
            raise ValueError(
                "Baseline runtime code differs from the artifact; retrain or use its recorded source"
            )
        files = metadata.get("artifact_files", {})
        expected_names = {"state.json"}
        if metadata["kind"] == "catboost_quantile":
            expected_names.update(f"quantile-{index}.cbm" for index in range(5))
        if set(files) != expected_names:
            raise ValueError("Unexpected or missing baseline artifact files")
        for name, checksum in files.items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or _file_sha(path) != checksum:
                raise ValueError(f"Baseline artifact checksum mismatch: {name}")
        state = json.loads((directory / "state.json").read_text())
        for name in ("kind", "fit_sessions", "calibration_sessions"):
            if state.get(name) != metadata.get(name):
                raise ValueError(f"Baseline state and manifest disagree on {name}")
        model = Baseline(state["kind"], ridge_alpha=float(state["ridge_alpha"]))
        model.fit_sessions = set(state["fit_sessions"])
        model.calibration_sessions = set(state["calibration_sessions"])
        model.residual_quantiles = _array(state["residual_quantiles"], (5,), "residual quantile")
        model.calibration_offsets = tuple(
            _array(state["calibration_offsets"], (2,), "calibration offset")
        )
        if np.any(np.diff(model.residual_quantiles) < 0) or min(model.calibration_offsets) < 0:
            raise ValueError("Invalid residual quantiles or calibration offsets")
        if model.kind == "regularized_linear_residual":
            model.location = _array(state["location"], (len(NUMERIC_FEATURES),), "location")
            model.scale = _array(state["scale"], (len(NUMERIC_FEATURES),), "scale")
            model.weights = _array(state["weights"], (len(NUMERIC_FEATURES) + 1,), "weights")
            if np.any(model.scale <= 0):
                raise ValueError("Linear scales must be positive")
        elif model.kind == "catboost_quantile":
            from catboost import CatBoostRegressor

            model.trees = []
            for index in range(5):
                tree = CatBoostRegressor()
                tree.load_model(str(directory / f"quantile-{index}.cbm"), format="cbm")
                model.trees.append(tree)
        model.fitted = True
        metadata["metadata_sha256"] = expected_metadata_sha
        bundle = cls(model, metadata)
        for key in ("fit_origin_latest", "calibration_origin_latest"):
            if metadata.get(key):
                scope = model.fit_sessions if key.startswith("fit_") else model.calibration_sessions
                validate_scope(
                    [{"session_key": min(scope), "origin": metadata[key]}],
                    set(scope),
                    role=f"loaded {key}",
                )
        return bundle


def final_candidate_partitions(catalog: dict, protocol: dict, *, expected_races: int = 18) -> dict:
    """Resolve final two calibration weekends after fixed-hyperparameter refitting."""
    folds = rolling_splits(catalog, protocol)
    if not folds:
        raise ValueError("A verified pre-Mexico development catalog is required")
    last = folds[-1]
    meetings = [
        meeting
        for role in ("fit", "tune", "calibration", "test")
        for meeting in last[f"{role}_meetings"]
    ]
    if len(meetings) != expected_races or len(set(meetings)) != len(meetings):
        raise ValueError(f"Expected exactly {expected_races} verified pre-Mexico race weekends")
    allowed_sessions = {
        session
        for role in ("fit", "tune", "calibration", "test")
        for session in last[f"{role}_sessions"]
    }
    by_meeting = {
        meeting: [
            row["session_key"]
            for row in catalog["development"]
            if row["meeting_key"] == meeting and row["session_key"] in allowed_sessions
        ]
        for meeting in meetings
    }
    fitting_meetings, calibration_meetings = meetings[:-2], meetings[-2:]
    return {
        "fit_meetings": fitting_meetings,
        "tuning_meetings_included_in_refit": fitting_meetings[-2:],
        "calibration_meetings": calibration_meetings,
        "fit_sessions": [
            session for meeting in fitting_meetings for session in by_meeting[meeting]
        ],
        "calibration_sessions": [
            session for meeting in calibration_meetings for session in by_meeting[meeting]
        ],
    }


def fit_final_candidate(
    examples: list[dict],
    catalog: dict,
    protocol: dict,
    *,
    development_report: dict | str | Path | None = None,
    data_kind: str = "unspecified",
    source_envelope_sha256: str | None = None,
) -> BaselineBundle:
    """Prepare the WIS-selected baseline; this does not freeze or evaluate Mexico."""
    partitions = final_candidate_partitions(catalog, protocol)
    fit_scope, calibration_scope = (
        set(partitions["fit_sessions"]),
        set(partitions["calibration_sessions"]),
    )
    validate_scope(examples, fit_scope | calibration_scope, role="final candidate source")
    fitting = [row for row in examples if row["session_key"] in fit_scope]
    calibration = [row for row in examples if row["session_key"] in calibration_scope]
    if {row["session_key"] for row in valid_examples(fitting)} != fit_scope or {
        row["session_key"] for row in valid_examples(calibration)
    } != calibration_scope:
        raise ValueError("Every declared fitting and calibration session needs valid labels")
    if isinstance(development_report, (str, Path)):
        development_report = json.loads(Path(development_report).read_text())
    selection = {
        "criterion": "weighted_interval_score",
        "kind": "catboost_quantile",
        "evidence_status": "configured_without_development_report",
    }
    if development_report is not None:
        if development_report["input_sha256"] != _sha(examples):
            raise ValueError("Candidate and development report data hashes differ")
        scores = {
            kind: result["antonelli"]["macro"]
            for kind, result in development_report["summary"].items()
        }
        winner = min(scores, key=lambda kind: scores[kind]["weighted_interval_score"])
        if winner != "catboost_quantile":
            raise ValueError("Configured candidate differs from the development WIS winner")
        best_point = min(scores, key=lambda kind: scores[kind]["mae_seconds"])
        selection.update(
            evidence_status="development_report_verified",
            development_report_sha256=_sha(development_report),
            development_metrics=scores[winner],
            best_point_model=best_point,
            best_point_mae_seconds=scores[best_point]["mae_seconds"],
            point_mae_tradeoff_fraction=scores[winner]["mae_seconds"]
            / scores[best_point]["mae_seconds"]
            - 1,
        )
    model = (
        Baseline("catboost_quantile")
        .fit(fitting, allowed_sessions=fit_scope)
        .calibrate(calibration, allowed_sessions=calibration_scope)
    )
    metadata = {
        "status": "candidate_not_frozen_for_mexico",
        "mexico_evaluated": False,
        "seed": 17,
        "protocol_sha256": protocol_hash(protocol),
        "data_sha256": _sha(examples),
        "source_envelope_sha256": source_envelope_sha256,
        "catalog_sha256": _sha(catalog),
        "artifact_code_sha256": _file_sha(Path(__file__)),
        "data_kind": data_kind,
        "cohort_driver_numbers": sorted({row["driver_number"] for row in examples}),
        "training_driver_counts": dict(
            Counter(str(row["driver_number"]) for row in valid_examples(fitting))
        ),
        "fitting_examples_sha256": _sha(valid_examples(fitting)),
        "calibration_examples_sha256": _sha(valid_examples(calibration)),
        "fit_origin_latest": max(row["origin"] for row in valid_examples(fitting)),
        "calibration_origin_latest": max(row["origin"] for row in valid_examples(calibration)),
        "partitions": partitions,
        "selection": selection,
        "cohort_limitation": "Observed cohort only; no full-field or car-independent driver-style claim",
    }
    return BaselineBundle(model, metadata)
