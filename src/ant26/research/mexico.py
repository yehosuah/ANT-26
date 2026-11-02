"""Safe descriptive phase summaries of completed Mexico telemetry sectors."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

PHASES = (
    ("opening", "Opening stint", 1, 42),
    ("interruption", "Pit stop and interruption", 43, 46),
    ("restart", "Restarted phase and late Safety Car", 47, 62),
    ("final", "Final run", 63, 71),
)
METRICS = {
    "full_throttle_fraction": "full_throttle_fraction",
    "brake_fraction": "brake_on_fraction",
    "coast_fraction": "coast_fraction",
}


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _aggregate(rows):
    eligible = [row for row in rows if row.get("eligible") is True]
    valid = math.fsum(row["valid_seconds"] for row in eligible)
    elapsed = math.fsum(row["duration_seconds"] for row in eligible)
    reasons = Counter(
        reason
        for row in rows
        if row.get("eligible") is not True
        for reason in set(row.get("exclusion_reasons") or ["unspecified_exclusion"])
    )
    result = {
        "observed_sectors": len(rows),
        "retained_sectors": len(eligible),
        "excluded_sectors": len(rows) - len(eligible),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "covered_seconds": valid,
        "sector_seconds": elapsed,
        "coverage_fraction": min(1.0, valid / elapsed) if elapsed else None,
        "mean_sector_seconds": statistics.fmean(row["duration_seconds"] for row in eligible)
        if eligible
        else None,
        "median_sector_seconds": statistics.median(row["duration_seconds"] for row in eligible)
        if eligible
        else None,
    }
    for name, source in METRICS.items():
        result[name] = (
            math.fsum(row[source] * row["valid_seconds"] for row in eligible) / valid
            if valid
            else None
        )
    ages = [row["tyre_age_laps"] for row in eligible if _finite(row.get("tyre_age_laps"))]
    compounds = Counter(str(row.get("compound") or "unknown") for row in eligible)
    stints = Counter(
        str(row["stint_number"]) for row in eligible if row.get("stint_number") is not None
    )
    result["tyre_context"] = {
        "compounds": dict(sorted(compounds.items())),
        "stint_numbers": dict(sorted(stints.items())),
        "stint_number_unknown_count": sum(row.get("stint_number") is None for row in eligible),
        "age_known_sectors": len(ages),
        "age_unknown_sectors": len(eligible) - len(ages),
        "age_min_laps": min(ages) if ages else None,
        "age_median_laps": statistics.median(ages) if ages else None,
        "age_max_laps": max(ages) if ages else None,
    }
    return result


def summarize_mexico(observations: dict | list[dict]) -> dict:
    """Pool native covered time within fixed phases and timing sectors.

    Source rows remain private. This is a retrospective description with no
    inference about adaptation, intention, or causal driver development.
    """
    if isinstance(observations, list):
        rows = observations
    elif isinstance(observations, dict) and isinstance(
        observations.get("mexico_observations"), list
    ):
        rows = observations["mexico_observations"]
    else:
        raise ValueError("Provide a Mexico observation list or mexico_observations envelope")
    seen = set()
    for row in rows:
        if row.get("session_key") != 11404 or row.get("driver_number") != 12:
            raise ValueError("Only the declared Mexico Antonelli observations are accepted")
        lap, sector = row.get("lap_number"), row.get("sector_index")
        if (
            not isinstance(lap, int)
            or isinstance(lap, bool)
            or not 1 <= lap <= 71
            or sector not in (1, 2, 3)
        ):
            raise ValueError("Invalid Mexico lap/sector identity")
        key = (lap, sector)
        if key in seen:
            raise ValueError("Duplicate Mexico sector identity")
        seen.add(key)
        if row.get("eligible") is True:
            duration, covered = row.get("duration_seconds"), row.get("valid_seconds")
            if not _finite(duration) or not _finite(covered) or not 0 < covered <= duration + 1e-6:
                raise ValueError("Retained sectors require valid covered and elapsed time")
            if not _finite(row.get("valid_coverage")) or not math.isclose(
                covered / duration, row["valid_coverage"], abs_tol=1e-6
            ):
                raise ValueError("Reported native coverage disagrees with elapsed-time support")
            for name in METRICS.values():
                if not _finite(row.get(name)) or not 0 <= row[name] <= 1:
                    raise ValueError("Retained control fractions must be finite and bounded")
    phases = []
    for phase_id, title, low, high in PHASES:
        selected = [row for row in rows if low <= row["lap_number"] <= high]
        phase = {
            "phase_id": phase_id,
            "title": title,
            "lap_start": low,
            "lap_end": high,
            "expected_sectors": (high - low + 1) * 3,
            "missing_sectors": (high - low + 1) * 3 - len(selected),
            **_aggregate(selected),
        }
        phase["sectors"] = [
            {
                "sector_index": sector,
                **_aggregate([row for row in selected if row["sector_index"] == sector]),
            }
            for sector in (1, 2, 3)
        ]
        phases.append(phase)
    canonical = json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    return {
        "schema_version": "ant26-mexico-driving-behavior-v1",
        "aggregate_only": True,
        "scope": {
            "session_key": 11404,
            "driver_number": 12,
            "official_laps": 71,
            "analysis": "retrospective_completed_sector_native_telemetry",
        },
        "source_observations_sha256": hashlib.sha256(canonical).hexdigest(),
        "summary_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_measurement_provenance": {
            key: observations.get("metadata", {}).get(key)
            for key in ("aggregate_source_sha256", "builder_sha256", "imported_helper_sha256")
        }
        if isinstance(observations, dict)
        else {},
        "methodology": {
            "control_aggregation": "sum_fraction_times_jointly_valid_seconds_divided_by_sum_jointly_valid_seconds",
            "control_denominator": "retained_jointly_observed_sector_time_not_unobserved_time",
            "phase_pace": "per_timing_sector_duration_mean_and_median; pooled_phase_pace_mixes_sector_geometry",
            "source_control_measurement": "native_zero_order_hold_no_interpolation; source_observation_builder_defines_gap_and_quality_gates",
            "full_throttle": "throttle_at_least_98_percent",
            "brake": "binary_brake_on_not_brake_pressure",
            "coast": "throttle_at_most_5_percent_brake_off_speed_at_least_80_kmh",
            "tyre_age": "source_tyre_age_at_stint_start_plus_laps_since_stint_start",
            "flags": "global_green_and_no_uncleared_local_yellow; veto_is_source_uncertainty_not_verified_physical_flag_state",
            "pit_adjacency": "retrospectively_exclude_pit_and_pit_out_laps_plus_one_neighbor_on_each_side",
            "exclusion_reason_counts": "one_sector_can_have_multiple_reasons",
        },
        "accounting": {
            "expected_sectors": 213,
            "missing_sectors": 213 - len(rows),
            **_aggregate(rows),
        },
        "phases": phases,
        "limitations": [
            "Phase comparisons are descriptive and selective; they do not establish adaptation, intention or causal driver development.",
            "Opening fresh soft tyres, used soft tyres after the restart, and older final-run tyres differ in age and conditions.",
            "Track evolution, fuel, traffic, neutralizations and geometry can explain control or pace differences.",
            "Conservative local-yellow persistence may reflect a missing CLEAR; it is not ground truth that the track remained yellow.",
            "Pace comparisons should use the same timing sector; overall phase control fractions also mix sector exposure.",
            "Native telemetry is sparse; unobserved intervals and invalid samples do not become inferred driving actions.",
            "Completed-sector observations are retrospective and never the inputs for their own forecast.",
        ],
    }
