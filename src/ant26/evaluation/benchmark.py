"""Offline benchmark orchestration with immutable result files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ant26.domain.protocol import protocol_hash
from ant26.evaluation.metrics import macro_metrics
from ant26.evaluation.splits import is_mexico, rolling_splits
from ant26.models.baselines import train_baselines, valid_examples


def _write_immutable(directory: Path, name: str, value: dict | list) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    path = directory / f"{name}-{digest[:16]}.json"
    try:
        with path.open("x") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_text() != payload:
            raise ValueError(f"Immutable artifact collision: {path}") from None
    return str(path)


def run_benchmark(
    examples: list[dict],
    catalog: dict | list[dict],
    protocol: dict | None = None,
    output_dir: str | Path | None = None,
    *,
    include_catboost: bool = False,
) -> dict:
    """Evaluate fixed baselines on fold-specific fit/refit and calibration scopes.

    Data availability never silently changes the split plan: incomplete folds are
    retained with a skip reason. Mexico rows are rejected even if ineligible.
    """
    protocol = protocol or {}
    if any(is_mexico(row) for row in examples):
        raise ValueError("Development benchmark input must exclude the Mexico weekend")
    identities = [row["example_id"] for row in examples]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate forecast example identities")
    folds = rolling_splits(catalog, protocol)
    results, predictions = [], []
    for fold in folds:
        role_rows = {
            role: [row for row in examples if row["session_key"] in fold[f"{role}_sessions"]]
            for role in ("fit", "tune", "calibration", "test")
        }
        sizes = {role: len(valid_examples(rows)) for role, rows in role_rows.items()}
        entry = {**fold, "eligible_counts": sizes}
        if min(sizes["fit"], sizes["tune"], sizes["test"]) < 1 or sizes["calibration"] < 10:
            entry.update(
                status="skipped", reason="insufficient_eligible_examples_in_declared_roles"
            )
            results.append(entry)
            continue
        fit_sessions = set(fold["fit_sessions"])
        fit_rows = role_rows["fit"]
        if protocol.get("splits", {}).get("refit_fit_plus_tuning_before_calibration", True):
            fit_sessions.update(fold["tune_sessions"])
            fit_rows = fit_rows + role_rows["tune"]
        models = train_baselines(
            fit_rows,
            role_rows["calibration"],
            fit_sessions=fit_sessions,
            calibration_sessions=set(fold["calibration_sessions"]),
            include_catboost=include_catboost,
        )
        test_rows = valid_examples(role_rows["test"])
        entry.update(status="evaluated", models={})
        for name, model in models.items():
            quantiles = model.predict(test_rows)
            records = [
                {
                    "example_id": row["example_id"],
                    "fold_id": fold["fold_id"],
                    "model": name,
                    "session_key": row["session_key"],
                    "driver_number": row["driver_number"],
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
            entry["models"][name] = {
                "antonelli": macro_metrics(records, 12),
                "all_drivers": macro_metrics(records, None),
                "fit_sessions": sorted(model.fit_sessions),
                "calibration_sessions": sorted(model.calibration_sessions),
            }
        results.append(entry)
    summary = {}
    for name in sorted({row["model"] for row in predictions}):
        records = [row for row in predictions if row["model"] == name]
        summary[name] = {
            "antonelli": macro_metrics(records),
            "all_drivers": macro_metrics(records, None),
        }
    evaluated = sum(row["status"] == "evaluated" for row in results)
    primary_races = max((value["antonelli"]["race_count"] for value in summary.values()), default=0)
    minimum = protocol.get("splits", {}).get("minimum_usable_development_test_races", 6)
    report = {
        "project": "ANT-26",
        "protocol_version": protocol.get("protocol_version", "defaults"),
        "protocol_sha256": protocol_hash(protocol),
        "input_sha256": hashlib.sha256(
            json.dumps(examples, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "data_kind": catalog.get("data_kind", "unspecified")
        if isinstance(catalog, dict)
        else "unspecified",
        "observed_driver_numbers": sorted({row["driver_number"] for row in examples}),
        "training_driver_count": len(
            {
                row["driver_number"]
                for fold in results
                if fold["status"] == "evaluated"
                for row in valid_examples(examples)
                if row["session_key"] in fold["fit_sessions"] + fold["tune_sessions"]
            }
        ),
        "cohort_limitation": "Antonelli-only cohort; no pooled multi-driver representation claim"
        if {row["driver_number"] for row in examples} == {12}
        else "Metrics describe observed dataset drivers, not a verified complete field",
        "synthetic": catalog.get("data_kind", "").startswith("synthetic")
        if isinstance(catalog, dict)
        else False,
        "status": "evaluated"
        if primary_races >= minimum
        else "exploratory"
        if evaluated
        else "insufficient_data",
        "minimum_usable_test_races": minimum,
        "evaluated_fold_count": evaluated,
        "antonelli_test_race_count": primary_races,
        "input_example_count": len(examples),
        "origin_eligible_example_count": sum(bool(row.get("eligible")) for row in examples),
        "eligible_example_count": len(valid_examples(examples)),
        "baseline_hyperparameters": "fixed_in_code_no_search",
        "catboost_included": include_catboost,
        "folds": results,
        "summary": summary,
        "predictions": predictions,
    }
    if output_dir is not None:
        report["predictions_path"] = _write_immutable(Path(output_dir), "predictions", predictions)
        report_path = _write_immutable(Path(output_dir), "benchmark", report)
        report["report_path"] = report_path
    return report
