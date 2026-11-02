"""Descriptive error and abstention slices over immutable forecast examples.

Realized target conditions are label-side diagnostics. This module neither creates
model inputs nor changes origin eligibility after observing the target.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from ant26.evaluation.metrics import macro_metrics

BINS = {
    "sector": ("1", "2", "3", "unknown"),
    "tyre_age_laps": ("[0,6)", "[6,16)", "[16,31)", "[31,inf)", "unknown"),
    "traffic_interval_seconds": ("(0,1]", "(1,3]", "(3,inf)", "unknown"),
    "telemetry_coverage": ("[0,0.8)", "[0.8,0.95)", "[0.95,1]", "unknown"),
    "realized_target_state": ("uninterrupted_green", "not_uninterrupted_green", "unknown"),
}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _scoreable(example: dict) -> bool:
    target = _finite(example.get("target_seconds"))
    reference = _finite(example.get("reference_seconds"))
    return bool(
        example.get("eligible", False)
        and example.get("target_valid", True)
        and target is not None
        and target > 0
        and reference is not None
        and reference > 0
    )


def _labels(example: dict) -> dict[str, str]:
    features = example.get("features") or {}
    sector = str(example.get("sector_index", "unknown"))
    age = _finite(features.get("tyre_age_laps"))
    interval = _finite(features.get("interval"))
    quality = _finite(features.get("telemetry_coverage"))
    return {
        "sector": sector if sector in {"1", "2", "3"} else "unknown",
        "tyre_age_laps": "unknown"
        if age is None or age < 0
        else "[0,6)"
        if age < 6
        else "[6,16)"
        if age < 16
        else "[16,31)"
        if age < 31
        else "[31,inf)",
        "traffic_interval_seconds": "unknown"
        if interval is None or interval <= 0
        else "(0,1]"
        if interval <= 1
        else "(1,3]"
        if interval <= 3
        else "(3,inf)",
        "telemetry_coverage": "unknown"
        if quality is None or not 0 <= quality <= 1
        else "[0,0.8)"
        if quality < 0.8
        else "[0.8,0.95)"
        if quality < 0.95
        else "[0.95,1]",
        "realized_target_state": "uninterrupted_green"
        if example.get("uninterrupted_green_target") is True
        else "not_uninterrupted_green"
        if example.get("uninterrupted_green_target") is False
        else "unknown",
    }


def _summary(records: list[dict], minimum_races: int, minimum_forecasts: int) -> dict:
    result = macro_metrics(records, driver_number=None)
    unique = len({row["example_id"] for row in records})
    seeds = {row["_seed"] for row in records}
    reasons = []
    if result["race_count"] < minimum_races:
        reasons.append("fewer_than_minimum_races")
    if unique < minimum_forecasts:
        reasons.append("fewer_than_minimum_unique_forecasts")
    return result | {
        "unique_example_count": unique,
        "prediction_count": len(records),
        "seed_count": len(seeds),
        "aggregation": "mean_seed_within_race_then_equal_race_weight"
        if len(seeds) > 1
        else "equal_race_weight",
        "exploratory": bool(reasons),
        "small_sample_reasons": reasons,
    }


def error_slices(
    predictions: list[dict],
    examples: list[dict],
    driver_number: int | None = 12,
    *,
    minimum_races: int = 6,
    minimum_forecasts: int = 30,
    evaluation_sessions: list[int] | set[int] | None = None,
) -> dict:
    """Join by identity, validate labels, and summarize fixed descriptive slices.

    Without an explicit evaluation scope, only sessions represented by predictions
    are used for coverage. Unpredicted fitting races are not called abstentions.
    Forecasts on censored targets remain visible but do not enter numeric metrics.
    Multiple seeds must cover identical scoreable examples within each model.
    """
    if minimum_races < 1 or minimum_forecasts < 1:
        raise ValueError("minimum sample sizes must be positive")
    lookup = {}
    for example in examples:
        identity = example.get("example_id")
        if identity is None or identity == "" or identity in lookup:
            raise ValueError("Forecast examples require unique nonempty example IDs")
        lookup[identity] = example
    scope = (
        set(evaluation_sessions)
        if evaluation_sessions is not None
        else {
            lookup[row["example_id"]]["session_key"]
            for row in predictions
            if row.get("example_id") in lookup
        }
    )
    models: dict[str, list[dict]] = defaultdict(list)
    censored = Counter()
    seen = set()
    model_seeds: dict[str, set[str]] = defaultdict(set)
    for prediction in predictions:
        identity = prediction.get("example_id")
        if identity not in lookup:
            raise ValueError(f"Prediction has a foreign example ID: {identity}")
        example = lookup[identity]
        for field in ("session_key", "driver_number", "lap_number", "sector_index"):
            if field in prediction and prediction[field] != example.get(field):
                raise ValueError(f"Prediction {field} does not match its example")
        for field in ("target_seconds", "reference_seconds"):
            if field in prediction:
                left, right = _finite(prediction[field]), _finite(example.get(field))
                if left != right and (
                    left is None
                    or right is None
                    or not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
                ):
                    raise ValueError(f"Prediction {field} does not match its example")
        if example["session_key"] not in scope:
            raise ValueError("Prediction session is outside the declared evaluation scope")
        name = prediction.get("model")
        if not isinstance(name, str) or not name:
            raise ValueError("Every prediction requires a model name")
        seed = str(prediction["seed"]) if prediction.get("seed") is not None else "unseeded"
        prediction_key = (name, seed, identity)
        if prediction_key in seen:
            raise ValueError("Duplicate prediction for the same model, seed and example")
        seen.add(prediction_key)
        if driver_number is not None and example["driver_number"] != driver_number:
            continue
        if not example.get("eligible", False):
            raise ValueError("A prediction was issued for an origin marked ineligible")
        model_seeds[name].add(seed)
        if not _scoreable(example):
            censored[name] += 1
            models.setdefault(name, [])
            continue
        models[name].append(
            prediction
            | {
                "session_key": example["session_key"],
                "driver_number": example["driver_number"],
                "target_seconds": example["target_seconds"],
                "reference_seconds": example["reference_seconds"],
                "_seed": seed,
                "_labels": _labels(example),
            }
        )
    selected = [
        row
        for row in examples
        if row["session_key"] in scope
        and (driver_number is None or row["driver_number"] == driver_number)
    ]
    scoreable_ids = {row["example_id"] for row in selected if _scoreable(row)}
    abstained = [row for row in selected if not row.get("eligible", False)]
    nominal = [
        row
        for row in selected
        if (row.get("features", {}).get("context") or {}).get("track_state") == "GREEN"
        and (row.get("features", {}).get("context") or {}).get("in_pit_lane") is False
    ]
    reason_counts = Counter(
        reason
        for row in abstained
        for reason in set(row.get("exclusion_reasons") or ["unspecified_origin_abstention"])
    )
    model_results = {}
    for name, records in sorted(models.items()):
        seeds = sorted(model_seeds[name])
        populations = [
            {row["example_id"] for row in records if row["_seed"] == seed} for seed in seeds
        ]
        if populations and any(population != populations[0] for population in populations[1:]):
            raise ValueError("Model seeds must cover identical scoreable example populations")
        grouped = {}
        for dimension, bins in BINS.items():
            grouped[dimension] = {
                label: _summary(
                    [row for row in records if row["_labels"][dimension] == label],
                    minimum_races,
                    minimum_forecasts,
                )
                for label in bins
            }
        model_results[name] = {
            "overall": _summary(records, minimum_races, minimum_forecasts),
            "slices": grouped,
            "per_seed": {
                seed: _summary(
                    [row for row in records if row["_seed"] == seed],
                    minimum_races,
                    minimum_forecasts,
                )
                for seed in seeds
            },
            "missing_scoreable_predictions": len(
                scoreable_ids - {row["example_id"] for row in records}
            ),
            "censored_prediction_count": censored[name],
        }
    return {
        "schema_version": "ant26-error-slices-v1",
        "driver_number": driver_number,
        "minimum_races": minimum_races,
        "minimum_unique_forecasts": minimum_forecasts,
        "models": model_results,
        "coverage": {
            "evaluation_session_keys": sorted(scope),
            "input_example_count": len(examples),
            "evaluated_example_count": len(selected),
            "origin_eligible_count": sum(bool(row.get("eligible")) for row in selected),
            "scoreable_example_count": len(scoreable_ids),
            "origin_abstention_count": len(abstained),
            "nominal_green_opportunities": len(nominal),
            "nominal_green_retention": sum(bool(row.get("eligible")) for row in nominal)
            / len(nominal)
            if nominal
            else None,
            "eligible_but_unscoreable_count": sum(
                bool(row.get("eligible")) and not _scoreable(row) for row in selected
            ),
            "examples_outside_evaluation_scope_count": sum(
                row["session_key"] not in scope for row in examples
            ),
            "other_driver_examples_in_evaluation_scope_count": sum(
                row["session_key"] in scope
                and driver_number is not None
                and row["driver_number"] != driver_number
                for row in examples
            ),
            "scope_status": "explicit_sessions"
            if evaluation_sessions is not None
            else "prediction_sessions"
            if scope
            else "no_prediction_scope",
        },
        "abstentions": {
            "example_count": len(abstained),
            "reason_counts": dict(sorted(reason_counts.items())),
            "reasons_may_overlap": True,
            "uses_origin_eligibility_only": True,
        },
        "slice_semantics": {
            "traffic": "positive interval to the car ahead; zero/leader/missing/nonfinite values are unknown",
            "realized_target_state": "label_side_diagnostic_only_never_an_origin_feature_or_eligibility_rule",
            "small_samples": "descriptive exploratory slices; no causal or psychological interpretation",
            "seed_count": "repeat model fits, not independent races or forecast opportunities",
        },
    }
