"""Local CPU measurements and warning-only input drift diagnostics.

This module never fits a predictor or changes its abstention policy. Timing
starts from already materialized past-only examples, not raw source events.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from ant26.evaluation.splits import validate_scope
from ant26.models.baselines import NUMERIC_FEATURES, feature_matrix, valid_examples


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def measure_calls(
    operation, *, repetitions: int = 200, warmup: int = 20, clock=perf_counter_ns
) -> dict:
    """Time synchronous calls; include output materialization inside operation."""
    if not isinstance(repetitions, int) or not 100 <= repetitions <= 10000:
        raise ValueError("Use between 100 and 10000 measured repetitions")
    if not isinstance(warmup, int) or not 1 <= warmup <= 1000:
        raise ValueError("Use between 1 and 1000 warmup calls")
    for index in range(warmup):
        operation(index)
    milliseconds = []
    for index in range(repetitions):
        start = clock()
        operation(index)
        elapsed = (clock() - start) / 1_000_000
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Clock must yield finite, nonnegative elapsed durations")
        milliseconds.append(elapsed)
    return {
        "repetitions": repetitions,
        "warmup_calls": warmup,
        "batch_size": 1,
        "p50_ms": float(np.quantile(milliseconds, 0.5)),
        "p95_ms": float(np.quantile(milliseconds, 0.95)),
        "p99_ms": float(np.quantile(milliseconds, 0.99)),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "mean_ms": float(np.mean(milliseconds)),
        "quantile_method": "numpy_linear_empirical",
        "clock": "perf_counter_ns",
    }


def latency_gate(measurement: dict, *, threshold_ms: float = 200.0) -> dict:
    p95 = measurement["p95_ms"]
    if not math.isfinite(p95) or p95 < 0 or not math.isfinite(threshold_ms) or threshold_ms <= 0:
        raise ValueError("Latency and threshold must be finite and valid")
    return {
        "p95_ms": p95,
        "threshold_ms": threshold_ms,
        "passed": p95 < threshold_ms,
        "comparison": "strictly_less_than",
        "scope": "local_warm_batch_one_materialized_input_not_service_slo",
    }


def input_drift_report(
    reference_examples: list[dict],
    observed_examples: list[dict],
    *,
    fit_sessions: set[int],
    observed_sessions: set[int],
    context_mean: list[float] | None = None,
    context_scale: list[float] | None = None,
    z_warning: float = 10.0,
) -> dict:
    """Compare origin inputs only, without filtering on target availability.

    Range exceedances and z-score warnings are descriptive and uncalibrated.
    They do not silently turn into new forecasting or retrieval exclusions.
    """
    validate_scope(reference_examples, fit_sessions, role="operations drift fitting reference")
    validate_scope(observed_examples, observed_sessions, role="operations drift observation")
    if fit_sessions & observed_sessions:
        raise ValueError("Drift observations must be disjoint from fitting races")
    reference = [row for row in reference_examples if row.get("eligible", False)]
    observed = [row for row in observed_examples if row.get("eligible", False)]
    if not reference or not observed:
        raise ValueError("Drift measurement requires eligible reference and observed inputs")
    if not math.isfinite(z_warning) or z_warning <= 0:
        raise ValueError("z_warning must be finite and positive")
    fitting, current = feature_matrix(reference), feature_matrix(observed)
    if (context_mean is None) != (context_scale is None):
        raise ValueError("Context mean and scale must be supplied together")
    mean = scale = None
    if context_mean is not None:
        mean, scale = np.asarray(context_mean, dtype=float), np.asarray(context_scale, dtype=float)
        if (
            mean.shape != (len(NUMERIC_FEATURES),)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("Context statistics must be finite and scales positive")
    features = {}
    any_range, any_z = np.zeros(len(observed), dtype=bool), np.zeros(len(observed), dtype=bool)
    for column, name in enumerate(NUMERIC_FEATURES):
        fit_values, observed_values = fitting[:, column], current[:, column]
        finite_fit, finite_observed = np.isfinite(fit_values), np.isfinite(observed_values)
        fit_missing, current_missing = 1 - finite_fit.mean(), 1 - finite_observed.mean()
        entry = {
            "fit_missing_fraction": float(fit_missing),
            "observed_missing_fraction": float(current_missing),
            "missing_fraction_change": float(current_missing - fit_missing),
            "observed_finite_count": int(finite_observed.sum()),
            "outside_fitted_range_count": None,
        }
        if finite_fit.any():
            lower, upper = np.min(fit_values[finite_fit]), np.max(fit_values[finite_fit])
            outside = finite_observed & ((observed_values < lower) | (observed_values > upper))
            any_range |= outside
            entry["outside_fitted_range_count"] = int(outside.sum())
        if mean is not None:
            z = np.abs((observed_values[finite_observed] - mean[column]) / scale[column])
            warnings = np.zeros(len(observed), dtype=bool)
            warnings[finite_observed] = z > z_warning
            any_z |= warnings
            entry.update(
                max_abs_encoder_context_z=float(z.max()) if len(z) else None,
                encoder_context_z_warning_count=int(warnings.sum()),
            )
        features[name] = entry
    fit_drivers = {row["driver_number"] for row in reference}
    unknown = sum(row["driver_number"] not in fit_drivers for row in observed)
    return {
        "status": "descriptive_warning_only_no_runtime_policy_change",
        "reference_scope": "eligible_origin_inputs_from_explicit_fitting_sessions_only",
        "target_fields_used": False,
        "fit_sessions": sorted(fit_sessions),
        "observed_sessions": sorted(observed_sessions),
        "reference_origin_count": len(reference),
        "observed_origin_count": len(observed),
        "outside_any_fitted_range_count": int(any_range.sum()),
        "unknown_driver_origin_count": unknown,
        "encoder_context_z_warning_threshold": z_warning,
        "encoder_context_z_warning_count": int(any_z.sum()) if mean is not None else None,
        "warning_threshold_validation": "engineering_diagnostic_not_a_validated_ood_detector",
        "features": features,
    }


def _hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "torch", "catboost")
        },
    }


def benchmark_candidates(
    dataset_path: str | Path,
    baseline_path: str | Path,
    encoder_path: str | Path,
    *,
    repetitions: int = 200,
    warmup: int = 20,
) -> dict:
    """Measure existing CPU artifacts without fitting, HTTP or source access."""
    import torch

    from ant26.models.artifacts import BaselineBundle
    from ant26.models.representation import RepresentationBundle

    torch.set_num_threads(1)
    serialized = Path(dataset_path).read_bytes()
    dataset = json.loads(serialized)
    examples = dataset["examples"]
    baseline, encoder = BaselineBundle.load(baseline_path), RepresentationBundle.load(encoder_path)
    if baseline.model.kind != "catboost_quantile":
        raise ValueError("This operations benchmark requires the selected CatBoost candidate")
    if any(parameter.device.type != "cpu" for parameter in encoder.encoder.parameters()):
        raise ValueError("CPU artifacts required")
    data_sha = _digest(examples)
    envelope_sha = hashlib.sha256(serialized).hexdigest()
    for bundle in (baseline, encoder):
        if bundle.metadata.get("data_sha256") != data_sha:
            raise ValueError("Candidate and exact dataset examples differ")
        if bundle.metadata.get("source_envelope_sha256") != envelope_sha:
            raise ValueError("Candidate and exact dataset envelope differ")
        if bundle.metadata.get("protocol_sha256") != dataset.get("protocol_sha256"):
            raise ValueError("Candidate and dataset protocol differ")
    fit_sessions = set(baseline.metadata["fit_sessions"])
    calibration_sessions = set(baseline.metadata["calibration_sessions"])
    if fit_sessions != set(encoder.metadata["fit_sessions"]) or calibration_sessions != set(
        encoder.metadata["calibration_sessions"]
    ):
        raise ValueError("Candidate scopes differ")
    validate_scope(examples, fit_sessions | calibration_sessions, role="operations dataset")
    pool = sorted(
        valid_examples([row for row in examples if row["session_key"] in calibration_sessions]),
        key=lambda row: (row["session_key"], row["driver_number"], row["example_id"]),
    )
    if len(pool) < 100:
        raise ValueError("At least 100 calibration examples required for diverse timing inputs")
    # Evenly span the calibration pool; use labels for selecting valid archived
    # timing inputs only. Neither prediction call consumes those labels.
    rows = [pool[int(index)] for index in np.linspace(0, len(pool) - 1, 100)]
    batches = [[row] for row in rows]
    matrices = [feature_matrix(batch) for batch in batches]
    prepared = [encoder._inputs(batch) for batch in batches]
    encoder.encoder.eval()

    def native_trees(index):
        matrix = matrices[index % len(rows)]
        return np.column_stack(
            [tree.predict(matrix, thread_count=1) for tree in baseline.model.trees]
        )

    def encoder_forward(index):
        values, mask, _ = prepared[index % len(rows)]
        with torch.no_grad():
            return encoder.encoder(values, mask)[1].cpu().numpy()

    operations = {
        "baseline_feature_conversion": lambda i: feature_matrix(batches[i % len(rows)]),
        "baseline_native_five_trees": native_trees,
        "baseline_predict_total": lambda i: baseline.predict(batches[i % len(rows)]),
        "encoder_input_conversion": lambda i: encoder._inputs(batches[i % len(rows)]),
        "encoder_forward": encoder_forward,
        "encoder_encode_total": lambda i: encoder.encode(batches[i % len(rows)]),
    }
    measurements = {
        name: measure_calls(operation, repetitions=repetitions, warmup=warmup)
        for name, operation in operations.items()
    }
    # Sanity-check outputs but do not retain predictions or embedding vectors.
    predictions = np.asarray(baseline.predict(batches[0]))
    embeddings = encoder.encode(batches[0])
    if not np.isfinite(predictions).all() or not np.isfinite(embeddings).all():
        raise ValueError("Candidate returned nonfinite output")
    reference = valid_examples([row for row in examples if row["session_key"] in fit_sessions])
    for bundle in (baseline, encoder):
        if bundle.metadata.get("fitting_examples_sha256") != _digest(reference):
            raise ValueError("Candidate and exact fitting reference differ")
    observed = [row for row in examples if row["session_key"] in calibration_sessions]
    return {
        "schema_version": "ant26-operations-v1",
        "aggregate_only": True,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_runtime": {
            **_hardware(),
            "torch_intraop_threads": torch.get_num_threads(),
            "catboost_native_phase_threads": 1,
            "baseline_total_thread_policy": "unchanged_saved_bundle_predict_defaults",
        },
        "input": {
            "source_envelope_sha256": envelope_sha,
            "examples_sha256": data_sha,
            "protocol_sha256": dataset["protocol_sha256"],
            "timed_example_count": len(rows),
            "timed_example_set_sha256": _digest(rows),
            "timed_driver_counts": dict(Counter(str(row["driver_number"]) for row in rows)),
            "timed_sessions": sorted({row["session_key"] for row in rows}),
            "selection": "100_evenly_spaced_sorted_scoreable_calibration_examples",
            "mexico_included": False,
        },
        "artifacts": {
            "baseline_metadata_sha256": baseline.metadata["metadata_sha256"],
            "encoder_metadata_file_sha256": hashlib.sha256(
                (Path(encoder_path) / "metadata.json").read_bytes()
            ).hexdigest(),
            "encoder_checkpoint_sha256": encoder.metadata["checkpoint_sha256"],
            "operations_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "measurements": measurements,
        "latency_gates": {
            name: latency_gate(measurements[name])
            for name in ("baseline_predict_total", "encoder_encode_total")
        },
        "timing_limits": [
            "Warm synchronous CPU calls in one local process; no concurrency or load test",
            "Input is materialized; excludes source alignment, network, HTTP, disk and model loading",
            "Phase timings are independent; full baseline currently converts features for every tree",
            "Encoder timing measures continuous embeddings, not failed quantile head or index search",
            "Calibration inputs measure runtime only; no new held-out quality claim",
        ],
        "input_drift": input_drift_report(
            reference,
            observed,
            fit_sessions=fit_sessions,
            observed_sessions=calibration_sessions,
            context_mean=encoder.metadata["context_mean"],
            context_scale=encoder.metadata["context_scale"],
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Operations reports are immutable; choose a new output file")
    report = benchmark_candidates(
        args.dataset,
        args.baseline,
        args.encoder,
        repetitions=args.repetitions,
        warmup=args.warmup,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"report": str(output), "latency_gates": report["latency_gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
