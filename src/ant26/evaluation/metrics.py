"""Metrics calculated on shared rows and uncertainty resampled by whole race."""

from __future__ import annotations

import math

import numpy as np

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def weighted_interval_score(target: float, quantiles: list[float]) -> float:
    q = np.asarray(quantiles, dtype=float)
    if q.shape != (5,) or not np.isfinite(q).all() or np.any(np.diff(q) < 0):
        raise ValueError("Expected five finite non-crossing quantiles")
    if not math.isfinite(target):
        raise ValueError("Target must be finite")
    total = 0.5 * abs(target - q[2])
    for alpha, low, high in ((0.2, q[1], q[3]), (0.1, q[0], q[4])):
        score = high - low + 2 / alpha * max(low - target, 0) + 2 / alpha * max(target - high, 0)
        total += alpha / 2 * score
    return float(total / 2.5)


def macro_metrics(records: list[dict], driver_number: int | None = 12) -> dict:
    by_race: dict[int, list[dict]] = {}
    for row in records:
        if driver_number is None or row["driver_number"] == driver_number:
            by_race.setdefault(row["session_key"], []).append(row)
    per_race = {}
    for key, rows in sorted(by_race.items()):
        errors, normalized, wis, coverage80, coverage90, widths80, widths90 = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for row in rows:
            target, reference = float(row["target_seconds"]), float(row["reference_seconds"])
            q = row["quantiles"]
            if not math.isfinite(reference) or reference <= 0:
                raise ValueError("Reference duration must be positive and finite")
            wis.append(weighted_interval_score(target, q))
            errors.append(abs(target - q[2]))
            normalized.append(abs(target - q[2]) / reference)
            coverage80.append(q[1] <= target <= q[3])
            coverage90.append(q[0] <= target <= q[4])
            widths80.append(q[3] - q[1])
            widths90.append(q[4] - q[0])
        per_race[str(key)] = {
            "count": len(rows),
            "mae_seconds": float(np.mean(errors)),
            "normalized_absolute_error": float(np.mean(normalized)),
            "weighted_interval_score": float(np.mean(wis)),
            "coverage_80": float(np.mean(coverage80)),
            "coverage_90": float(np.mean(coverage90)),
            "width_80_seconds": float(np.mean(widths80)),
            "width_90_seconds": float(np.mean(widths90)),
        }
    metric_names = [
        "mae_seconds",
        "normalized_absolute_error",
        "weighted_interval_score",
        "coverage_80",
        "coverage_90",
        "width_80_seconds",
        "width_90_seconds",
    ]
    macro = (
        {name: float(np.mean([row[name] for row in per_race.values()])) for name in metric_names}
        if per_race
        else {}
    )
    return {
        "driver_number": driver_number,
        "race_count": len(per_race),
        "forecast_count": sum(row["count"] for row in per_race.values()),
        "aggregation": "equal_race_weight",
        "macro": macro,
        "per_race": per_race,
    }


def paired_race_bootstrap(
    reference: dict,
    candidate: dict,
    metric: str = "mae_seconds",
    *,
    resamples: int = 1000,
    seed: int = 17,
) -> dict:
    """Inputs are `per_race` dictionaries; rows and seeds are never bootstrap units."""
    if set(reference) != set(candidate) or not reference:
        raise ValueError("Paired comparisons require identical nonempty race sets")
    keys = sorted(reference)
    left = np.array([reference[key][metric] for key in keys], dtype=float)
    right = np.array([candidate[key][metric] for key in keys], dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all() or left.mean() <= 0:
        raise ValueError("Invalid paired race scores")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(resamples, len(keys)))
    denominator = left[indices].mean(axis=1)
    improvements = np.divide(
        (left[indices] - right[indices]).mean(axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    low, high = np.quantile(improvements, [0.025, 0.975])
    return {
        "unit": "complete_race",
        "race_count": len(keys),
        "resamples": resamples,
        "relative_improvement": float((left.mean() - right.mean()) / left.mean()),
        "confidence_interval_95": [float(low), float(high)],
        "exploratory": len(keys) < 6,
    }
