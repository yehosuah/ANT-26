"""Retrospective, native-sample pedal-use measurements over completed sectors.

These observations are not forecast inputs. Sector anchors reuse the documented
lap-start-plus-sector-duration reconstruction; any complete-lap timing defect
excludes that lap. No end-of-lap backfilling is used for missing sector anchors.

Elapsed-time fractions use zero-order holds of validated native speed, throttle
and binary brake, for at most one second. There is no interpolation or tensor
grid. All three channels must be valid for a duration to enter the common
denominator. Fractions describe valid covered time; coverage separately divides
that time by the complete sector duration. A native gap exceeding one second
that overlaps the sector excludes the sector even if coverage exceeds 90%.

Require global green throughout, and conservatively veto ANY active marshal-
sector yellow because marshal sectors cannot be mapped to the three timing
sectors without geometry. Driver blue flags do not establish a persistent track
state. Pit records and pit-out laps both center conservative +/- one-lap
exclusions; no asserted physical pit-loop timestamps are used. Context matching
requires a known compound and nonnegative, finite reported tyre age.

Race progress uses the same maximum completed valid-timed lap across the declared
two-driver source cohort, excluding partial tails. It is not official/full-grid
race distance. Mexico uses its declared 71-lap denominator and remains separate.
Traffic and weather use the latest source observation at sector start, with
explicit age and missingness. No source publication-latency claim is made.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ant26.datasets.features import (
    _car_records,
    _events,
    _flags,
    _identity,
    _iso,
    _sector_records,
    _state_at,
)
from ant26.datasets.materialize import _snapshot

CHANNELS = ("speed", "throttle", "brake")
COMPOUNDS = frozenset(("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"))
DENOMINATOR_KIND = "max_completed_valid_timed_lap_across_declared_race_cohort"
POLICY = {
    "analysis": "retrospective_completed_sector_observation",
    "time_weighting": "native_zero_order_hold_joint_valid_speed_throttle_brake",
    "maximum_hold_seconds": 1.0,
    "maximum_native_gap_seconds": 1.0,
    "minimum_valid_coverage": 0.9,
    "full_throttle_minimum_percent": 98,
    "coast_throttle_maximum_percent": 5,
    "coast_minimum_speed_kmh": 80,
    "pit_exclusion": "recorded_pit_and_pit_out_lap_plus_or_minus_one_lap",
    "local_flags": "any_active_marshal_sector_non_green_veto_without_geometry",
    "maximum_traffic_observation_age_seconds": 15.0,
    "maximum_weather_observation_age_seconds": 120.0,
    "live_publication_availability": "unknown",
}


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _read(value: dict | str | Path) -> dict:
    result = value if isinstance(value, dict) else json.loads(Path(value).read_text())
    if not isinstance(result, dict):
        raise ValueError("source envelope must be an object")
    return result


def _local_flags(rows: list[dict], session: int) -> dict[Any, list]:
    grouped = defaultdict(list)
    for time, row in _events(rows, session):
        if str(row.get("scope", "")).upper() != "SECTOR":
            continue
        flag = str(row.get("flag") or "").upper()
        if flag in {"GREEN", "CLEAR", "YELLOW", "DOUBLE YELLOW", "RED", "SC", "VSC"}:
            grouped[row.get("sector")].append((time, "GREEN" if flag == "CLEAR" else flag))
    return grouped


def _green_reasons(global_flags: list, local_flags: dict, start: float, end: float) -> list[str]:
    reasons = []
    if _state_at(global_flags, start) != "GREEN" or any(
        start < time <= end and flag != "GREEN" for time, flag in global_flags
    ):
        reasons.append("not_continuously_global_green")
    if any(
        _state_at(events, start) not in (None, "GREEN")
        or any(start < time <= end and flag != "GREEN" for time, flag in events)
        for events in local_flags.values()
    ):
        reasons.append("local_marshal_sector_flag")
    return reasons


def _native_metrics(car: dict, start: float, end: float) -> dict:
    times, samples = car["times"], car["samples"]
    left, right = bisect.bisect_left(times, start), bisect.bisect_left(times, end)
    valid_seconds, full, brake, coast = 0.0, 0.0, 0.0, 0.0
    channel_seconds = dict.fromkeys(CHANNELS, 0.0)
    maximum_gap = max(0.0, min(end, times[0]) - start) if times else end - start
    for index in range(max(0, left - 1), right):
        time, values = samples[index]
        following = times[index + 1] if index + 1 < len(times) else end
        if following > start and time < end:
            maximum_gap = max(maximum_gap, following - time)
        low, high = max(start, time), min(end, following, time + 1.0)
        if high <= low:
            continue
        weight = high - low
        for channel in CHANNELS:
            if values[channel] is not None:
                channel_seconds[channel] += weight
        if any(values[channel] is None for channel in CHANNELS):
            continue
        valid_seconds += weight
        full += weight * (values["throttle"] >= 98)
        brake += weight * (values["brake"] > 0)
        coast += weight * (
            values["throttle"] <= 5 and values["brake"] == 0 and values["speed"] >= 80
        )
    duration = end - start
    return {
        "native_sample_count": right - left,
        "valid_seconds": valid_seconds,
        "valid_coverage": min(1.0, valid_seconds / duration),
        "channel_coverage": {
            key: min(1.0, value / duration) for key, value in channel_seconds.items()
        },
        "max_gap_seconds": maximum_gap,
        "full_throttle_fraction": full / valid_seconds if valid_seconds else None,
        "brake_on_fraction": brake / valid_seconds if valid_seconds else None,
        "coast_fraction": coast / valid_seconds if valid_seconds else None,
    }


def _last(events: list, start: float, maximum_age: float) -> tuple[dict, float | None]:
    times = [time for time, _ in events]
    index = bisect.bisect_right(times, start) - 1
    if index < 0:
        return {}, None
    time, value = events[index]
    age = start - time
    return (value if age <= maximum_age else {}), age


def measure_sectors(
    laps: list[dict],
    car_data: list[dict],
    *,
    race_control: list[dict],
    pit: list[dict],
    stints: list[dict],
    intervals: list[dict],
    weather: list[dict] | None = None,
    session_lap_count: int,
    lap_fraction_denominator_kind: str = DENOMINATOR_KIND,
) -> list[dict]:
    """Measure all observed sector records and preserve every exclusion reason."""
    if isinstance(session_lap_count, bool) or session_lap_count <= 0:
        raise ValueError("positive shared race-lap denominator required")
    sectors, cars = _sector_records(laps), _car_records(car_data)
    identities = {_identity(row) for row in laps}
    prepared = {}
    for identity in identities:
        session, driver = identity
        pit_centers = {
            row["lap_number"]
            for row in pit
            if _identity(row) == identity and isinstance(row.get("lap_number"), int)
        } | {
            row["lap_number"]
            for row in laps
            if _identity(row) == identity and row.get("is_pit_out_lap") is True
        }
        excluded_laps = {lap + delta for lap in pit_centers for delta in (-1, 0, 1)}
        driver_stints = sorted(
            [
                row
                for row in stints
                if _identity(row) == identity and _finite(row.get("lap_start")) is not None
            ],
            key=lambda row: float(row["lap_start"]),
        )
        prepared[identity] = {
            "global": _flags(race_control, session),
            "local": _local_flags(race_control, session),
            "pits": excluded_laps,
            "stints": driver_stints,
            "intervals": [
                (time, row)
                for time, row in _events(intervals, session)
                if row.get("driver_number") == driver
            ],
            "weather": _events(weather or [], session),
        }
    result = []
    for sector in sectors:
        lap = sector["lap"]
        identity, number = _identity(lap), lap["lap_number"]
        context = prepared[identity]
        start, end = sector["start"], sector["end"]
        reasons = list(sector["timing_reasons"])
        if number == 1:
            reasons.append("race_start_lap")
        if number in context["pits"]:
            reasons.append("pit_or_adjacent_lap")
        if lap.get("is_pit_out_lap"):
            reasons.append("pit_out_lap")
        stint = next(
            (
                row
                for row in reversed(context["stints"])
                if float(row["lap_start"]) <= number
                and (_finite(row.get("lap_end")) is None or number <= float(row["lap_end"]))
            ),
            {},
        )
        compound = str(stint.get("compound") or "").upper()
        compound = compound if compound in COMPOUNDS else None
        age = _finite(stint.get("tyre_age_at_start"))
        age = age + number - float(stint["lap_start"]) if age is not None and age >= 0 else None
        if compound is None:
            reasons.append("unknown_compound")
        if age is None or age < 0:
            reasons.append("invalid_tyre_age")
        row = {
            "example_id": f"{identity[0]}:{identity[1]}:{number}:{sector['sector_index']}",
            "session_key": identity[0],
            "meeting_key": lap.get("meeting_key"),
            "driver_number": identity[1],
            "lap_number": number,
            "sector_index": sector["sector_index"],
            "origin": _iso(start) if start is not None else None,
            "end": _iso(end) if end is not None else None,
            "duration_seconds": sector["duration"],
            "compound": compound,
            "stint_number": stint.get("stint_number"),
            "tyre_age_laps": age,
            "lap_fraction": min(1.0, max(0.0, (number - 1) / session_lap_count)),
            "lap_fraction_denominator": session_lap_count,
            "lap_fraction_denominator_kind": lap_fraction_denominator_kind,
            "interval_gap": None,
            "interval_sample_age_seconds": None,
            "weather_rainfall": None,
            "rainfall": None,
            "weather_sample_age_seconds": None,
            "native_sample_count": 0,
            "valid_seconds": 0.0,
            "valid_coverage": 0.0,
            "channel_coverage": dict.fromkeys(CHANNELS, 0.0),
            "max_gap_seconds": None,
            "full_throttle_fraction": None,
            "brake_on_fraction": None,
            "coast_fraction": None,
        }
        if start is None or end is None or end <= start:
            reasons.append("invalid_sector_anchor")
        else:
            reasons.extend(_green_reasons(context["global"], context["local"], start, end))
            car = cars.get(identity, {"times": [], "samples": []})
            row.update(_native_metrics(car, start, end))
            if row["valid_coverage"] < 0.9:
                reasons.append("insufficient_native_coverage")
            if row["max_gap_seconds"] > 1.0:
                reasons.append("native_gap_exceeds_one_second")
            traffic, age_traffic = _last(context["intervals"], start, 15)
            row["interval_gap"] = _finite(traffic.get("interval"))
            row["interval_sample_age_seconds"] = age_traffic
            weather_row, age_weather = _last(context["weather"], start, 120)
            rainfall = _finite(weather_row.get("rainfall"))
            # Provider rainfall is frequently a JSON boolean, not a channel.
            if isinstance(weather_row.get("rainfall"), bool):
                rainfall = float(weather_row["rainfall"])
            row["weather_rainfall"] = row["rainfall"] = rainfall if rainfall in (0, 1) else None
            row["weather_sample_age_seconds"] = age_weather
        row["exclusion_reasons"] = sorted(set(reasons))
        row["eligible"] = not reasons
        result.append(row)
    return result


def _quality(rows: list[dict]) -> dict:
    return {
        "observed_sector_records": len(rows),
        "eligible_sector_records": sum(row["eligible"] for row in rows),
        "exclusion_reasons": dict(
            Counter(reason for row in rows for reason in row["exclusion_reasons"])
        ),
        "by_driver_race": [
            {
                "session_key": session,
                "driver_number": driver,
                "observed": len(group),
                "eligible": sum(row["eligible"] for row in group),
            }
            for (session, driver), group in sorted(_groups(rows).items())
        ],
    }


def _groups(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["session_key"], row["driver_number"])].append(row)
    return grouped


def _denominator(laps: list[dict]) -> int:
    completed = [
        record["lap"]["lap_number"]
        for record in _sector_records(laps)
        if not record["timing_reasons"] and record["lap_completed_at"] is not None
    ]
    return max(completed, default=0)


def load_observations(
    progress: dict | str | Path,
    catalog: dict | str | Path,
    mexico_envelope: dict | str | Path | None = None,
) -> dict:
    """Load exact cached snapshots; no provider/network client is constructed."""
    progress, catalog = _read(progress), _read(catalog)
    source_manifests, development, mexico = [], [], []
    endpoints = ("laps", "race_control", "pit", "stints", "intervals", "weather")
    for race in sorted(catalog["development"], key=lambda value: value["date_start"]):
        session = race["session_key"]
        if race.get("meeting_key") == 1298 or session == 11404:
            raise ValueError("Mexico cannot enter development observations")
        if race.get("verified_identity") is not True or race.get("is_cancelled"):
            continue
        sources = []
        for driver in (12, 63):
            entry = progress.get("sessions", {}).get(f"{session}:{driver}", {})
            if entry.get("status") != "complete":
                continue
            manifests = [entry["context_sources"][name] for name in endpoints]
            data = {name: _snapshot(entry["context_sources"][name]) for name in endpoints}
            telemetry = []
            for _, chunk in sorted(entry.get("chunks", {}).items()):
                manifests.append(chunk["manifest"])
                telemetry.extend(_snapshot(chunk["manifest"]))
            if any(_identity(row) != (session, driver) for row in data["laps"] + telemetry):
                raise ValueError("cached source escapes declared driver/race")
            source_manifests.extend(manifests)
            sources.append((data, telemetry))
        denominator = _denominator([lap for data, _ in sources for lap in data["laps"]])
        if not denominator:
            continue
        for data, telemetry in sources:
            development.extend(
                measure_sectors(data.pop("laps"), telemetry, session_lap_count=denominator, **data)
            )
    if mexico_envelope is not None:
        envelope = _read(mexico_envelope)
        if envelope.get("session_key") != 11404 or envelope.get("driver_number") != 12:
            raise ValueError("Mexico observations require explicit Antonelli race envelope")
        manifests = envelope["source_manifests"]
        grouped = defaultdict(list)
        for manifest in manifests:
            if manifest["endpoint"] in {*endpoints, "car_data"}:
                grouped[manifest["endpoint"]].extend(_snapshot(manifest))
                source_manifests.append(manifest)
        if any(_identity(row) != (11404, 12) for row in grouped["laps"] + grouped["car_data"]):
            raise ValueError("Mexico snapshots escape declared subject")
        mexico = measure_sectors(
            grouped.pop("laps"),
            grouped.pop("car_data"),
            session_lap_count=envelope.get("expected_laps", 71),
            lap_fraction_denominator_kind="declared_mexico_race_laps",
            **grouped,
        )
    manifests = list(
        {
            (value["endpoint"], value["query_key"], value["sha256"]): value
            for value in source_manifests
        }.values()
    )
    manifests.sort(key=lambda value: (value["endpoint"], value["query_key"], value["sha256"]))
    identity = [
        {
            key: value[key]
            for key in ("endpoint", "query_key", "sha256", "raw_response_sha256")
            if key in value
        }
        for value in manifests
    ]
    return {
        "schema_version": "ant26-native-sector-observations-v1",
        "data_kind": "real_openf1_retrospective_two_driver_observations",
        "development_observations": development,
        "mexico_observations": mexico,
        "quality": {"development": _quality(development), "mexico": _quality(mexico)},
        "metadata": {
            "aggregate_source_sha256": hashlib.sha256(_bytes(identity)).hexdigest(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "imported_helper_sha256": {
                name: hashlib.sha256(
                    (Path(__file__).parents[1] / "datasets" / name).read_bytes()
                ).hexdigest()
                for name in ("features.py", "materialize.py")
            },
            "source_cohort_driver_numbers": [12, 63],
            "mexico_driver_numbers": [12] if mexico else [],
            "lap_fraction_denominator_kind": DENOMINATOR_KIND,
            "mexico_separate_from_development": True,
            "measurement_policy": POLICY,
        },
        "private_provenance": {"source_manifests": manifests, "input_manifest_hashes": identity},
    }
