"""Chronological, scoped representation experiments with immutable evidence.

This runner uses fixed model hyperparameters. It refits on fitting plus tuning
races only when the protocol allows it, then calibrates on the designated next
partition. Future test windows never become training examples.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ant26.domain.protocol import protocol_hash
from ant26.evaluation.benchmark import _write_immutable
from ant26.evaluation.metrics import macro_metrics, paired_race_bootstrap
from ant26.evaluation.splits import rolling_splits, validate_scope
from ant26.models.baselines import valid_examples
from ant26.models.representation import EncoderConfig, RepresentationBundle, train_representation


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _seed_summary(records: list[dict], driver_number: int | None) -> dict:
    seeds = sorted({row["seed"] for row in records})
    per_seed = {
        str(seed): macro_metrics([row for row in records if row["seed"] == seed], driver_number)
        for seed in seeds
    }
    races = (
        set().union(*(set(result["per_race"]) for result in per_seed.values()))
        if per_seed
        else set()
    )
    if any(set(result["per_race"]) != races for result in per_seed.values()):
        raise ValueError("Every seed must be evaluated on the same race set")
    per_race = {}
    for race in sorted(races):
        rows = [result["per_race"][race] for result in per_seed.values()]
        if len({row["count"] for row in rows}) != 1:
            raise ValueError("Every seed must be evaluated on identical example counts")
        per_race[race] = {
            name: rows[0][name] if name == "count" else float(np.mean([row[name] for row in rows]))
            for name in rows[0]
        }
    macro = {
        name: float(np.mean([row[name] for row in per_race.values()]))
        for name in next(iter(per_race.values()), {})
        if name != "count"
    }
    return {
        "driver_number": driver_number,
        "seed_count": len(seeds),
        "race_count": len(races),
        "forecast_count_per_seed": sum(row["count"] for row in per_race.values()),
        "aggregation": "mean_seed_error_within_race_then_equal_race_weight",
        "macro": macro,
        "per_race": per_race,
        "per_seed": per_seed,
    }


def _baseline_comparisons(
    predictions: list[dict],
    summaries: dict,
    baseline_report: dict,
    input_sha256: str,
    *,
    resamples: int,
    seed: int,
) -> dict:
    if baseline_report.get("input_sha256") != input_sha256:
        raise ValueError("Baseline and representation datasets must have identical input hashes")
    baseline_rows = baseline_report.get("predictions", [])
    comparisons = {}
    for mode, summary in summaries.items():
        candidate = [
            row for row in predictions if row["mode"] == mode and row["driver_number"] == 12
        ]
        expected = {(row["fold_id"], row["example_id"]) for row in candidate}
        if not expected:
            comparisons[mode] = {"status": "no_antonelli_forecasts"}
            continue
        valid_baselines = {}
        for model_name in sorted({row["model"] for row in baseline_rows}):
            rows = [
                row
                for row in baseline_rows
                if row["model"] == model_name
                and row["driver_number"] == 12
                and (row["fold_id"], row["example_id"]) in expected
            ]
            identities = {(row["fold_id"], row["example_id"]) for row in rows}
            if identities != expected or len(identities) != len(rows):
                raise ValueError(
                    f"Baseline {model_name} does not cover identical held-out examples"
                )
            valid_baselines[model_name] = macro_metrics(rows, 12)
        if not valid_baselines:
            comparisons[mode] = {"status": "no_comparable_baselines"}
            continue
        metrics = {}
        for metric in ("mae_seconds", "weighted_interval_score"):
            strongest = min(
                valid_baselines, key=lambda name: valid_baselines[name]["macro"][metric]
            )
            if valid_baselines[strongest]["macro"][metric] == 0:
                metrics[metric] = {
                    "baseline_model": strongest,
                    "status": "zero_baseline_error_no_relative_gain_possible",
                    "relative_improvement": None,
                    "confidence_interval_95": [None, None],
                }
                continue
            metrics[metric] = {
                "baseline_model": strongest,
                **paired_race_bootstrap(
                    valid_baselines[strongest]["per_race"],
                    summary["antonelli"]["per_race"],
                    metric,
                    resamples=resamples,
                    seed=seed,
                ),
            }
        comparisons[mode] = {"status": "paired", "metrics": metrics}
    return comparisons


def run_representation_benchmark(
    examples: list[dict],
    catalog: dict,
    protocol: dict,
    output_dir: str | Path,
    seeds: list[int] | None = None,
    max_folds: int | None = None,
    pretrain_epochs: int = 3,
    forecast_epochs: int = 5,
    *,
    encoder_config: EncoderConfig | None = None,
    modes: tuple[str, ...] = ("finetune",),
    baseline_report: dict | str | Path | None = None,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    cpu_threads: int = 1,
) -> dict:
    """Run fixed-budget candidates; reduced budgets remain explicitly exploratory.

    A caller may restrict folds/seeds for smoke tests. No such run earns an
    improvement claim. Baseline comparison additionally requires matching input
    hashes and the same example identities for every paired race.
    """
    started = time.perf_counter()
    if examples:
        validate_scope(
            examples,
            {row["session_key"] for row in examples},
            role="representation benchmark input",
        )
    identifiers = [row["example_id"] for row in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate representation benchmark example identities")
    protocol_seeds = protocol.get("experiments", {}).get("seeds", [17, 29, 43, 71, 101])
    selected_seeds = list(protocol_seeds if seeds is None else seeds)
    if (
        not selected_seeds
        or len(selected_seeds) != len(set(selected_seeds))
        or any(not isinstance(seed, int) for seed in selected_seeds)
    ):
        raise ValueError("Provide distinct nonempty integer seeds")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"scratch", "frozen", "finetune"}:
        raise ValueError("Provide distinct supported representation modes")
    if max_folds is not None and (not isinstance(max_folds, int) or max_folds < 1):
        raise ValueError("max_folds must be a positive integer")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    folds = rolling_splits(catalog, protocol)
    selected_folds = folds if max_folds is None else folds[:max_folds]
    config = encoder_config or EncoderConfig()
    if isinstance(baseline_report, (str, Path)):
        baseline_report = json.loads(Path(baseline_report).read_text())
    input_sha256 = _digest(examples)
    if baseline_report is not None and baseline_report.get("input_sha256") != input_sha256:
        raise ValueError("Baseline and representation datasets must have identical input hashes")
    source_files = [
        Path(__file__),
        Path(__file__).with_name("splits.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).parents[1] / "models" / "representation.py",
    ]
    source_sha256 = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
    spec = {
        "schema_version": "ant26-representation-benchmark-v1",
        "input_sha256": input_sha256,
        "catalog_sha256": _digest(catalog),
        "protocol_sha256": protocol_hash(protocol),
        "source_sha256": source_sha256,
        "encoder_config": asdict(config),
        "seeds": selected_seeds,
        "modes": list(modes),
        "fold_ids": [fold["fold_id"] for fold in selected_folds],
        "pretrain_epochs": pretrain_epochs,
        "forecast_epochs": forecast_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "cpu_threads": cpu_threads,
        "baseline_report_sha256": _digest(baseline_report) if baseline_report is not None else None,
    }
    run_dir = Path(output_dir) / f"representation-{_digest(spec)[:16]}"
    report_file = run_dir / "report.json"
    if report_file.exists():
        return json.loads(report_file.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_immutable(run_dir, "run-spec", spec)
    predictions, fold_results = [], []
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        for fold in selected_folds:
            rows = {
                role: [row for row in examples if row["session_key"] in fold[f"{role}_sessions"]]
                for role in ("fit", "tune", "calibration", "test")
            }
            counts = {role: len(valid_examples(values)) for role, values in rows.items()}
            entry = {**fold, "scorable_counts": counts, "runs": []}
            if (
                counts["fit"] < 2
                or counts["tune"] < 1
                or counts["calibration"] < 10
                or counts["test"] < 1
            ):
                entry.update(
                    status="skipped", reason="insufficient_eligible_targets_in_declared_roles"
                )
                fold_results.append(entry)
                continue
            fit_rows = rows["fit"]
            fit_sessions = set(fold["fit_sessions"])
            if protocol["splits"].get("refit_fit_plus_tuning_before_calibration", True):
                fit_rows = fit_rows + rows["tune"]
                fit_sessions.update(fold["tune_sessions"])
            calibration_sessions = set(fold["calibration_sessions"])
            test_rows = valid_examples(rows["test"])
            for mode in modes:
                for seed in selected_seeds:
                    checkpoint_dir = run_dir / "models" / mode / fold["fold_id"] / f"seed-{seed}"
                    if (checkpoint_dir / "metadata.json").exists():
                        bundle = RepresentationBundle.load(checkpoint_dir)
                        if bundle.metadata["fitting_examples_sha256"] != _digest(
                            valid_examples(fit_rows)
                        ) or bundle.metadata.get("calibration_sessions") != sorted(
                            calibration_sessions
                        ):
                            raise ValueError(
                                "Cached checkpoint scope differs from the declared fold"
                            )
                        cache_hit = True
                    else:
                        bundle = train_representation(
                            fit_rows,
                            fit_sessions,
                            protocol=protocol,
                            mode=mode,
                            config=config,
                            pretrain_epochs=pretrain_epochs,
                            forecast_epochs=forecast_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            seed=seed,
                            data_kind=catalog.get("data_kind", "unspecified"),
                        )
                        bundle.calibrate(
                            rows["calibration"], calibration_sessions=calibration_sessions
                        )
                        bundle.save(checkpoint_dir)
                        cache_hit = False
                    quantiles = bundle.predict(test_rows)
                    records = [
                        {
                            "example_id": row["example_id"],
                            "session_key": row["session_key"],
                            "driver_number": row["driver_number"],
                            "fold_id": fold["fold_id"],
                            "model": f"representation_{mode}",
                            "mode": mode,
                            "seed": seed,
                            "origin": row.get("origin"),
                            "cutoff": row.get("cutoff"),
                            "target_seconds": row["target_seconds"],
                            "reference_seconds": row["reference_seconds"],
                            "quantiles": q,
                            "source_provenance": row.get("provenance", {}),
                        }
                        for row, q in zip(test_rows, quantiles, strict=True)
                    ]
                    predictions.extend(records)
                    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
                    entry["runs"].append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "checkpoint_dir": str(checkpoint_dir),
                            "checkpoint_sha256": metadata["checkpoint_sha256"],
                            "fit_sessions": metadata["fit_sessions"],
                            "calibration_sessions": metadata["calibration_sessions"],
                            "encoder_parameters": metadata["encoder_parameters"],
                            "training_driver_counts": dict(
                                Counter(
                                    str(row["driver_number"]) for row in valid_examples(fit_rows)
                                )
                            ),
                            "training_duration_seconds": metadata["training_duration_seconds"],
                            "cache_hit": cache_hit,
                            "antonelli": macro_metrics(records, 12),
                            "all_drivers": macro_metrics(records, None),
                        }
                    )
            entry["status"] = "evaluated"
            fold_results.append(entry)
    finally:
        torch.set_num_threads(prior_threads)
    summaries = {
        mode: {
            "antonelli": _seed_summary([row for row in predictions if row["mode"] == mode], 12),
            "all_drivers": _seed_summary([row for row in predictions if row["mode"] == mode], None),
        }
        for mode in modes
    }
    race_count = max((value["antonelli"]["race_count"] for value in summaries.values()), default=0)
    evaluated = sum(row["status"] == "evaluated" for row in fold_results)
    synthetic = catalog.get("data_kind", "").startswith("synthetic")
    real = catalog.get("data_kind", "") in (
        "real_openf1",
        "real_openf1_full",
        "real_openf1_antonelli",
    )
    observed_drivers = sorted({row["driver_number"] for row in examples})
    training_drivers = sorted(
        {
            int(driver)
            for fold in fold_results
            for run in fold["runs"]
            for driver in run["training_driver_counts"]
        }
    )
    minimum_races = protocol["splits"].get("minimum_usable_development_test_races", 6)
    full_seed_budget = set(selected_seeds) == set(protocol_seeds)
    full_fold_budget = len(selected_folds) == len(folds)
    evaluated_test_sessions = {
        key
        for fold in fold_results
        if fold["status"] == "evaluated"
        for key in fold["test_sessions"]
    }
    retention = {}
    for session in sorted(evaluated_test_sessions):
        nominal = [
            row
            for row in examples
            if row["session_key"] == session
            and row["driver_number"] == 12
            and row.get("features", {}).get("context", {}).get("track_state") == "GREEN"
            and row.get("features", {}).get("context", {}).get("in_pit_lane") is False
        ]
        eligible = sum(bool(row.get("eligible")) for row in nominal)
        retention[str(session)] = {
            "nominal_green_opportunities": len(nominal),
            "origin_eligible": eligible,
            "retention": eligible / len(nominal) if nominal else 0.0,
        }
    retention_threshold = (
        protocol["metrics"]
        .get("engineering_targets", {})
        .get("minimum_nominal_green_sector_retention", 0.7)
    )
    retention_gate = (
        sum(row["retention"] >= retention_threshold for row in retention.values()) >= minimum_races
    )
    complete_benchmark = (
        real
        and not synthetic
        and race_count >= minimum_races
        and full_seed_budget
        and full_fold_budget
        and retention_gate
    )
    comparisons = (
        _baseline_comparisons(
            predictions,
            summaries,
            baseline_report,
            input_sha256,
            resamples=protocol["metrics"]["bootstrap"].get("resamples", 1000),
            seed=17,
        )
        if baseline_report is not None
        else {}
    )
    claims = {}
    for mode in modes:
        paired = comparisons.get(mode, {})
        gates = protocol["metrics"].get("improvement_claim_gates", {})
        supported = paired.get("status") == "paired"
        if supported:
            for metric, threshold in (
                ("mae_seconds", gates.get("minimum_relative_mae_improvement", 0.05)),
                ("weighted_interval_score", gates.get("minimum_relative_wis_improvement", 0.05)),
            ):
                result = paired["metrics"][metric]
                if result["relative_improvement"] is None:
                    supported = False
                    continue
                supported &= (
                    result["relative_improvement"] >= threshold
                    and result["confidence_interval_95"][0] > 0
                )
        claims[mode] = {
            "improvement_claim_allowed": bool(complete_benchmark and supported),
            "reason": "requires_full_real_data_retention_complete_multiseed_benchmark_and_both_paired_metric_gates",
        }
    report = {
        "project": "ANT-26",
        "schema_version": "ant26-representation-benchmark-v1",
        "protocol_version": protocol.get("protocol_version"),
        "protocol_sha256": protocol_hash(protocol),
        "input_sha256": input_sha256,
        "data_kind": catalog.get("data_kind", "unspecified"),
        "observed_driver_numbers": observed_drivers,
        "training_driver_count": len(training_drivers),
        "training_driver_numbers": training_drivers,
        "cohort_limitation": "Antonelli-only cohort; no pooled multi-driver representation claim"
        if observed_drivers == [12]
        else "Metrics describe observed dataset drivers, not a verified complete field",
        "synthetic": synthetic,
        "status": "evaluated"
        if complete_benchmark
        else "exploratory"
        if evaluated
        else "insufficient_data",
        "evaluated_fold_count": evaluated,
        "antonelli_test_race_count": race_count,
        "input_example_count": len(examples),
        "origin_eligible_example_count": sum(bool(row.get("eligible")) for row in examples),
        "scorable_example_count": len(valid_examples(examples)),
        "seed_budget_complete": full_seed_budget,
        "fold_budget_complete": full_fold_budget,
        "retention_gate_passed": retention_gate,
        "retention_by_test_race": retention,
        "modes": list(modes),
        "seeds": selected_seeds,
        "hyperparameters": "fixed_no_search_refit_per_protocol",
        "encoder_config": asdict(config),
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {
            "device": "cpu",
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "threads": cpu_threads,
        },
        "run_spec": spec,
        "folds": fold_results,
        "summary": summaries,
        "comparisons": comparisons,
        "claims": claims,
        "predictions": predictions,
        "report_path": str(report_file),
    }
    report["predictions_path"] = _write_immutable(run_dir, "predictions", predictions)
    _write_immutable(run_dir, "benchmark", report)
    with report_file.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return report
