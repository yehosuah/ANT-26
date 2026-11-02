"""Race-block comparisons of matched retrospective pedal-use observations.

No training or Mexico matching occurs here. Native samples form sector summaries;
sectors form matched cells; cells form races. Only races are bootstrap units.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ant26.evaluation.splits import validate_scope

METRICS = ("full_throttle_fraction", "brake_fraction", "coast_fraction")
DRIVERS = (12, 63)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _write(path, value):
    if path is None:
        return
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError("Refusing to replace immutable behavior analysis")
        return
    with path.open("x") as handle:
        handle.write(payload)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _rows(value):
    if isinstance(value, list):
        return value
    for key in ("development_observations", "observations", "rows"):
        if key in value:
            return value[key]
    raise ValueError("An observation list or development_observations envelope is required")


def _identity(row):
    return f"{row['session_key']}:{row['driver_number']}:{row['lap_number']}:{row['sector_index']}"


def _metric(row, name):
    return (
        row.get("brake_on_fraction")
        if name == "brake_fraction" and "brake_fraction" not in row
        else row.get(name)
    )


def _bin(value, width, *, fraction=False):
    index = int(math.floor(value / width + 1e-9))
    return min(index, int(round(1 / width)) - 1) if fraction else index


def _prepare(rows, *, minimum_coverage, minimum_gap, dry_only):
    retained, reasons = [], Counter()
    for row in rows:
        exclusions = []
        if row.get("eligible") is not True:
            exclusions += row.get("exclusion_reasons") or ["observation_ineligible"]
        if not isinstance(row.get("compound"), str) or row["compound"].upper() in {
            "",
            "UNKNOWN",
            "NONE",
        }:
            exclusions.append("missing_compound")
        if not _finite(row.get("tyre_age_laps")) or row["tyre_age_laps"] < 0:
            exclusions.append("missing_or_invalid_tyre_age")
        if not _finite(row.get("lap_fraction")) or not 0 <= row["lap_fraction"] <= 1:
            exclusions.append("missing_or_invalid_lap_fraction")
        if (
            not _finite(row.get("valid_coverage"))
            or not minimum_coverage <= row["valid_coverage"] <= 1
        ):
            exclusions.append("insufficient_native_coverage")
        if any(
            not _finite(_metric(row, name)) or not 0 <= _metric(row, name) <= 1 for name in METRICS
        ):
            exclusions.append("invalid_native_metric")
        if minimum_gap is not None and (
            not _finite(row.get("interval_gap")) or row["interval_gap"] < minimum_gap
        ):
            exclusions.append("traffic_restricted_or_missing_interval")
        if dry_only and (not _finite(row.get("rainfall")) or row["rainfall"] != 0):
            exclusions.append("rainfall_nonzero_or_missing")
        if exclusions:
            reasons.update(set(exclusions))
        else:
            retained.append(row)
    return retained, dict(sorted(reasons.items()))


def race_bootstrap(race_vectors, *, seed=17, resamples=10000):
    """Percentile intervals for three jointly resampled complete-race contrasts."""
    values = np.asarray(race_vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(METRICS) or not np.isfinite(values).all():
        raise ValueError("Expected a finite race-by-three-metric matrix")
    if not isinstance(resamples, int) or resamples < 100:
        raise ValueError("Use at least 100 whole-race bootstrap resamples")
    if len(values) < 2:
        return {
            name: {"ci95_percentage_points": None, "bonferroni98_333_percentage_points": None}
            for name in METRICS
        }
    draws = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1) * 100
    tail = 0.05 / (2 * len(METRICS))
    return {
        name: {
            "ci95_percentage_points": np.quantile(means[:, column], [0.025, 0.975]).tolist(),
            "bonferroni98_333_percentage_points": np.quantile(
                means[:, column], [tail, 1 - tail]
            ).tolist(),
        }
        for column, name in enumerate(METRICS)
    }


def matched_comparison(
    rows,
    *,
    tyre_age_bin=3,
    lap_fraction_bin=0.1,
    minimum_coverage=0.9,
    minimum_gap=None,
    dry_only=False,
    seed=17,
    resamples=10000,
):
    """Estimate ANT minus RUS using equal cells within races, then equal races."""
    if (
        tyre_age_bin <= 0
        or lap_fraction_bin <= 0
        or not math.isclose(1 / lap_fraction_bin, round(1 / lap_fraction_bin))
    ):
        raise ValueError("Positive tyre bins and a lap-fraction bin dividing one are required")
    if not 0 <= minimum_coverage <= 1 or (minimum_gap is not None and minimum_gap < 0):
        raise ValueError("Invalid coverage or traffic restriction")
    rows = list(rows)
    validate_scope(
        rows, {row["session_key"] for row in rows}, role="retrospective development matching"
    )
    identities = [_identity(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate retrospective sector observations")
    if any(
        row["driver_number"] not in DRIVERS or row["sector_index"] not in (1, 2, 3) for row in rows
    ):
        raise ValueError("Only declared Mercedes drivers and valid sector indices can be matched")
    retained, reasons = _prepare(
        rows, minimum_coverage=minimum_coverage, minimum_gap=minimum_gap, dry_only=dry_only
    )
    cells = defaultdict(lambda: defaultdict(list))
    for row in retained:
        key = (
            row["session_key"],
            row["sector_index"],
            row["compound"].upper(),
            _bin(row["tyre_age_laps"], tyre_age_bin),
            _bin(row["lap_fraction"], lap_fraction_bin, fraction=True),
        )
        cells[key][row["driver_number"]].append(row)
    matched, race_cells, matched_ids = [], defaultdict(list), set()
    for key, drivers in sorted(cells.items()):
        if set(drivers) != set(DRIVERS):
            continue
        driver_means = {
            driver: np.mean(
                [[_metric(row, name) for name in METRICS] for row in drivers[driver]], axis=0
            )
            for driver in DRIVERS
        }
        difference = driver_means[12] - driver_means[63]
        race_cells[key[0]].append((driver_means[12], driver_means[63]))
        row_ids = {
            str(driver): sorted(_identity(row) for row in drivers[driver]) for driver in DRIVERS
        }
        matched_ids.update(identity for ids in row_ids.values() for identity in ids)
        matched.append(
            {
                "session_key": key[0],
                "sector_index": key[1],
                "compound": key[2],
                "tyre_age_bin_index": key[3],
                "lap_fraction_bin_index": key[4],
                "observation_ids": row_ids,
                "driver_means": {
                    str(driver): dict(zip(METRICS, driver_means[driver].tolist(), strict=True))
                    for driver in DRIVERS
                },
                "difference": dict(zip(METRICS, difference.tolist(), strict=True)),
            }
        )
    races = []
    for session, cell_values in sorted(race_cells.items()):
        a = np.mean([cell[0] for cell in cell_values], axis=0)
        b = np.mean([cell[1] for cell in cell_values], axis=0)
        races.append(
            {
                "session_key": session,
                "matched_cell_count": len(cell_values),
                "antonelli": dict(zip(METRICS, a.tolist(), strict=True)),
                "russell": dict(zip(METRICS, b.tolist(), strict=True)),
                "difference": dict(zip(METRICS, (a - b).tolist(), strict=True)),
            }
        )
    vectors = np.array([[race["difference"][name] for name in METRICS] for race in races]).reshape(
        -1, 3
    )
    intervals = race_bootstrap(vectors, seed=seed, resamples=resamples)
    metrics = {}
    for column, name in enumerate(METRICS):
        effects = vectors[:, column]
        mean = float(effects.mean()) if len(effects) else None
        omitted = (
            [
                {
                    "excluded_session_key": race["session_key"],
                    "difference_percentage_points": float(np.delete(effects, i).mean() * 100),
                }
                for i, race in enumerate(races)
            ]
            if len(effects) > 1
            else []
        )
        sign = np.sign(mean) if mean is not None else 0
        metrics[name] = {
            "antonelli_mean": float(np.mean([race["antonelli"][name] for race in races]))
            if races
            else None,
            "russell_mean": float(np.mean([race["russell"][name] for race in races]))
            if races
            else None,
            "difference_percentage_points": mean * 100 if mean is not None else None,
            **intervals[name],
            "race_sign_counts": {
                "positive": int(np.sum(effects > 0)),
                "negative": int(np.sum(effects < 0)),
                "zero": int(np.sum(effects == 0)),
            },
            "leave_one_race_out": {
                "estimates": omitted,
                "all_same_sign_as_full_effect": bool(omitted)
                and all(np.sign(row["difference_percentage_points"]) == sign for row in omitted),
                "minimum_percentage_points": min(
                    (row["difference_percentage_points"] for row in omitted), default=None
                ),
                "maximum_percentage_points": max(
                    (row["difference_percentage_points"] for row in omitted), default=None
                ),
            },
        }
    race_coverage = []
    for session in sorted({row["session_key"] for row in rows}):
        race_coverage.append(
            {
                "session_key": session,
                "matched_cell_count": len(race_cells[session]),
                **{
                    str(driver): {
                        "observed": sum(
                            row["session_key"] == session and row["driver_number"] == driver
                            for row in rows
                        ),
                        "retained": sum(
                            row["session_key"] == session and row["driver_number"] == driver
                            for row in retained
                        ),
                        "matched": sum(
                            row["session_key"] == session
                            and row["driver_number"] == driver
                            and _identity(row) in matched_ids
                            for row in retained
                        ),
                    }
                    for driver in DRIVERS
                },
            }
        )
    return {
        "status": "evaluated" if len(races) >= 2 else "insufficient_independent_races",
        "matching": {
            "fields": [
                "session_key",
                "sector_index",
                "compound",
                "tyre_age_bin",
                "lap_fraction_bin",
            ],
            "tyre_age_bin_laps": tyre_age_bin,
            "lap_fraction_bin": lap_fraction_bin,
            "minimum_native_coverage": minimum_coverage,
            "minimum_interval_gap_seconds": minimum_gap,
            "dry_only": dry_only,
            "aggregation": "driver_mean_within_cell_then_equal_cells_within_race_then_equal_races",
        },
        "coverage": {
            "observed": len(rows),
            "retained_before_matching": len(retained),
            "matched_observations": len(matched_ids),
            "unmatched_retained_observations": len(retained) - len(matched_ids),
            "exclusion_reason_counts": reasons,
            "by_driver": {
                str(driver): {
                    "retained": sum(row["driver_number"] == driver for row in retained),
                    "matched": sum(
                        row["driver_number"] == driver and _identity(row) in matched_ids
                        for row in retained
                    ),
                }
                for driver in DRIVERS
            },
            "by_race": race_coverage,
        },
        "race_count": len(races),
        "matched_cell_count": len(matched),
        "metrics": metrics,
        "per_race": races,
        "matched_cells": matched,
    }


def run_behavior_comparison(
    observations, *, output_path=None, public_output_path=None, seed=17, resamples=10000
):
    """Run fixed primary/sensitivity analyses and emit a safe aggregate report."""
    envelope = _read(observations)
    rows = _rows(envelope)
    primary = matched_comparison(rows, seed=seed, resamples=resamples)
    configurations = {
        "tyre_age_1lap": {"tyre_age_bin": 1},
        "lap_fraction_005": {"lap_fraction_bin": 0.05},
        "coverage_095": {"minimum_coverage": 0.95},
        "traffic_gap_2s": {"minimum_gap": 2.0},
    }
    if any(_finite(row.get("rainfall")) for row in rows):
        configurations["dry_only"] = {"dry_only": True}
    sensitivities = {
        name: matched_comparison(rows, seed=seed, resamples=resamples, **config)
        for name, config in configurations.items()
    }
    findings = {}
    for metric in METRICS:
        point = primary["metrics"][metric]["difference_percentage_points"]
        interval = primary["metrics"][metric]["bonferroni98_333_percentage_points"]
        excludes_zero = interval is not None and (interval[0] > 0 or interval[1] < 0)
        available = {
            name: result for name, result in sensitivities.items() if result["race_count"] >= 2
        }
        sign_survives = (
            point is not None
            and bool(available)
            and all(
                np.sign(result["metrics"][metric]["difference_percentage_points"]) == np.sign(point)
                for result in available.values()
            )
        )
        loo = primary["metrics"][metric]["leave_one_race_out"]["all_same_sign_as_full_effect"]
        missing = sorted(set(sensitivities) - set(available))
        limited = (
            missing
            or primary["race_count"] < 6
            or sensitivities["traffic_gap_2s"]["race_count"] < 6
        )
        allowed = bool(excludes_zero and sign_survives and loo)
        level = (
            (
                "qualified_observational_signal"
                if limited
                else "robust_matched_observational_difference"
            )
            if allowed
            else ("sensitivity_dependent" if excludes_zero else "no_clear_difference")
        )
        if not excludes_zero:
            reason = "The Bonferroni-adjusted interval includes zero or cannot be estimated; a direction is not established."
        elif not sign_survives:
            reason = "The direction changes in a usable sensitivity or sensitivity support is unavailable."
        elif not loo:
            reason = "Leaving out one race changes the effect's sign."
        elif limited:
            reason = "The direction survives available checks, but missing or limited traffic support restricts it to a qualified observational signal."
        else:
            reason = "The family-wise interval excludes zero and the direction survives the declared available sensitivities and every leave-one-race-out estimate."
        point_direction = (
            "Antonelli_higher"
            if point is not None and point > 0
            else "Antonelli_lower"
            if point is not None and point < 0
            else "no_direction"
        )
        findings[metric] = {
            "direction_claim_allowed": allowed,
            "evidence_level": level,
            "direction": point_direction if allowed else "not_established",
            "point_estimate_direction": point_direction,
            "reason": reason,
            "primary_familywise_interval_excludes_zero": excludes_zero,
            "sign_survives_available_sensitivities": sign_survives,
            "leave_one_race_out_sign_stable": loo,
            "unavailable_sensitivities": missing,
            "traffic_matched_race_count": sensitivities["traffic_gap_2s"]["race_count"],
            "stable_driving_style_claim_allowed": False,
            "interpretation": "Observed matched-cohort pedal use; not intention, causality, pace superiority or a car-independent driver signature.",
        }
    report = {
        "schema_version": "ant26-driver-behavior-v1",
        "aggregate_only": False,
        "analysis_kind": "retrospective_observational_two_driver_matched_comparison",
        "drivers": {"antonelli": 12, "russell": 63},
        "mexico_included_in_matching": False,
        "metric_definitions": {
            "full_throttle_fraction": "Native valid-time fraction with throttle>=98 percent",
            "brake_fraction": "Native valid-time fraction with binary brake applied",
            "coast_fraction": "Native valid-time fraction with throttle<=5 percent,brake off,speed>=80 km/h",
        },
        "uncertainty": {
            "unit": "complete_race",
            "resamples": resamples,
            "seed": seed,
            "ordinary_confidence": 0.95,
            "primary_family_size": 3,
            "bonferroni_individual_confidence": 1 - 0.05 / 3,
            "nominal_familywise_confidence": 0.95,
            "method": "empirical_percentile_bootstrap_of_complete_matched_race_contrasts",
            "sensitivity_intervals": "descriptive_not_jointly_adjusted_across_sensitivity_analyses",
        },
        "primary": primary,
        "sensitivities": sensitivities,
        "findings": findings,
        "provenance": {
            "observations_sha256": hashlib.sha256(Path(observations).read_bytes()).hexdigest()
            if isinstance(observations, (str, Path))
            else _sha(envelope),
            "comparison_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "observation_schema_version": envelope.get("schema_version")
            if isinstance(envelope, dict)
            else None,
            "observation_metadata": copy.deepcopy(envelope.get("metadata", {}))
            if isinstance(envelope, dict)
            else {},
        },
        "limitations": [
            "Race matching controls only recorded context; fuel,setup,weather,traffic,instructions and within-bin differences can remain.",
            "Equal-cell and equal-race weights describe common support,not the time fraction over the whole season.",
            "Bootstrap races are observational blocks,not randomized assignments; few races and selective source coverage limit inference.",
            "The three primary intervals use Bonferroni correction; sensitivities are robustness checks rather than searches for significance.",
            "No human atlas validation,no driver intention or causal championship explanation is established.",
        ],
    }
    public = copy.deepcopy(report)
    public["aggregate_only"] = True
    for result in [public["primary"], *public["sensitivities"].values()]:
        result.pop("matched_cells")
    _write(output_path, report)
    _write(public_output_path, public)
    return report
