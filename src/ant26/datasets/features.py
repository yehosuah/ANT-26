"""Build auditable sector examples without exposing labels to feature transforms.

Source publication latency is unavailable. Event time is a documented replay
assumption, not evidence that an API consumer could observe every field live.
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHANNELS = ("speed", "throttle", "brake", "rpm", "n_gear")
CONTINUOUS = frozenset(("speed", "throttle", "rpm"))
DEFAULTS = {
    "feature_guard_seconds": 1.0,
    "window_seconds": 32,
    "tensor_grid_hz": 4,
    "max_interpolation_gap_seconds": 1.0,
    "minimum_required_channel_coverage": 0.8,
    "required_channels": list(CHANNELS),
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _time(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _identity(row: dict) -> tuple[Any, Any]:
    return row.get("session_key"), row.get("driver_number")


def _valid_channel(channel: str, value: Any) -> float | None:
    value = _number(value)
    if value is None:
        return None
    bounds = {"speed": (0, 500), "throttle": (0, 100), "rpm": (0, 25000), "n_gear": (0, 8)}
    if channel == "brake":
        return value / 100 if value in (0, 100) else value if value in (0, 1) else None
    low, high = bounds.get(channel, (-math.inf, math.inf))
    if not low <= value <= high:
        return None
    if channel == "n_gear" and not value.is_integer():
        return None
    return value


def _flag_code(row: dict) -> str | None:
    scope = str(row.get("scope") or "Track").upper()
    if scope not in ("TRACK", "GLOBAL"):
        return None
    flag = str(row.get("flag") or "").upper()
    if flag in ("GREEN", "CLEAR"):
        return "GREEN"
    if flag in ("RED", "YELLOW", "DOUBLE YELLOW", "CHEQUERED", "SC", "VSC"):
        return flag
    message = str(row.get("message") or "").upper()
    if "VIRTUAL SAFETY CAR" in message and ("DEPLOYED" in message or "ENDING" in message):
        return "VSC"
    if "SAFETY CAR" in message and ("DEPLOYED" in message or "IN THIS LAP" in message):
        return "SC"
    return None


def _events(rows: list[dict], session: Any) -> list[tuple[float, dict]]:
    events = []
    for row in rows:
        if row.get("session_key", session) != session:
            continue
        timestamp = _time(row.get("date"))
        if timestamp is not None:
            events.append((timestamp, row))
    return sorted(events, key=lambda x: (x[0], json.dumps(x[1], sort_keys=True)))


def _flags(rows: list[dict], session: Any) -> list[tuple[float, str]]:
    return [(time, flag) for time, row in _events(rows, session) if (flag := _flag_code(row))]


def _state_at(flags: list[tuple[float, str]], cutoff: float) -> str | None:
    state = None
    for time, flag in flags:
        if time > cutoff:
            break
        state = flag
    return state


def _sector_records(laps: list[dict]) -> list[dict]:
    seen = set()
    records = []
    by_driver = defaultdict(list)
    for lap in laps:
        key = (*_identity(lap), lap.get("lap_number"))
        if key in seen:
            raise ValueError(f"duplicate lap key: {key}")
        seen.add(key)
        start = _time(lap.get("date_start"))
        durations = [_number(lap.get(f"duration_sector_{i}")) for i in range(1, 4)]
        lap_duration = _number(lap.get("lap_duration"))
        reasons = []
        if start is None:
            reasons.append("invalid_lap_anchor")
        if any(d is None or d <= 0 for d in durations):
            reasons.append("invalid_sector_timing")
        elif lap_duration is not None and abs(sum(durations) - lap_duration) > 0.1:
            reasons.append("sector_sum_mismatch")
        by_driver[_identity(lap)].append((lap, start, durations, reasons))
    for group in by_driver.values():
        group.sort(key=lambda x: (x[1] is None, x[1] or 0, x[0].get("lap_number", 0)))
        for index, (lap, start, durations, reasons) in enumerate(group):
            lap_reasons = list(reasons)
            reasons = list(reasons)
            successor_reasons = []
            successor_anchor_at = None
            if start is not None and all(d is not None and d > 0 for d in durations):
                if index + 1 < len(group):
                    next_lap, next_start, _, _ = group[index + 1]
                    consecutive = next_lap.get("lap_number", 0) == lap.get("lap_number", 0) + 1
                    if consecutive:
                        successor_anchor_at = next_start
                    if (
                        consecutive
                        and next_start is not None
                        and abs(start + sum(durations) - next_start) > 2
                    ):
                        reasons.append("inconsistent_successive_lap_anchor")
                        successor_reasons.append("inconsistent_successive_lap_anchor")
            origin = start
            for sector, duration in enumerate(durations, 1):
                records.append(
                    {
                        "lap": lap,
                        "sector_index": sector,
                        "start": origin,
                        "end": origin + duration
                        if origin is not None and duration is not None and duration > 0
                        else None,
                        "duration": duration,
                        "lap_completed_at": start + sum(durations)
                        if start is not None and all(d is not None and d > 0 for d in durations)
                        else None,
                        "timing_reasons": list(reasons),
                        "lap_timing_reasons": lap_reasons,
                        "successor_timing_reasons": successor_reasons,
                        "successor_anchor_at": successor_anchor_at,
                    }
                )
                origin = (
                    origin + duration
                    if origin is not None and duration is not None and duration > 0
                    else None
                )
    return sorted(
        records,
        key=lambda r: (
            r["start"] is None,
            r["start"] or 0,
            str(_identity(r["lap"])),
            r["sector_index"],
        ),
    )


def _historical_timing_valid(record: dict, cutoff: float) -> bool:
    """Apply quality evidence only after that evidence exists in event time."""
    if record["start"] is None or record["end"] is None:
        return False
    if record["end"] > cutoff or record["duration"] is None or record["duration"] <= 0:
        return False
    lap_end = record["lap_completed_at"]
    if lap_end is not None and lap_end <= cutoff and record["lap_timing_reasons"]:
        return False
    successor_at = record["successor_anchor_at"]
    if successor_at is not None and successor_at <= cutoff and record["successor_timing_reasons"]:
        return False
    return True


def _car_records(car_data: list[dict]) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in car_data:
        time = _time(row.get("date"))
        if time is not None:
            grouped[_identity(row)][time].append(row)
    result = {}
    for identity, times in grouped.items():
        samples = []
        duplicates = 0
        for time, rows in sorted(times.items()):
            values = {}
            duplicates += len(rows) - 1
            for channel in CHANNELS:
                observed = {_valid_channel(channel, row.get(channel)) for row in rows}
                values[channel] = observed.pop() if len(observed) == 1 else None
            samples.append((time, values))
        result[identity] = {
            "times": [s[0] for s in samples],
            "samples": samples,
            "duplicates": duplicates,
        }
    return result


def _window(car: dict, cutoff: float, flags: list, policy: dict) -> dict:
    channels = policy["required_channels"]
    hz, seconds = policy["tensor_grid_hz"], policy["window_seconds"]
    count = int(hz * seconds)
    max_gap = policy["max_interpolation_gap_seconds"]
    grid = [cutoff - (count - i - 1) / hz for i in range(count)]
    all_times = car["times"]
    end = bisect.bisect_right(all_times, cutoff)
    times, samples = all_times[:end], car["samples"][:end]
    last_red = max(
        (time for time, flag in flags if flag == "RED" and time <= cutoff), default=-math.inf
    )
    values, usable, observed, ages = [], [], [], []
    for point in grid:
        right = bisect.bisect_left(times, point)
        exact = right < len(times) and abs(times[right] - point) < 1e-6
        left = right if exact else right - 1
        vector, mask, native, age_vector = [], [], [], []
        for channel in channels:
            value, age = None, None
            if left >= 0 and point > last_red:
                left_time, left_values = samples[left]
                age = point - left_time
                a = left_values.get(channel)
                if left_time > last_red and a is not None and 0 <= age <= max_gap:
                    if exact or channel not in CONTINUOUS or right == len(times):
                        value = a
                    else:
                        right_time, right_values = samples[right]
                        b = right_values.get(channel)
                        if b is not None and right_time - left_time <= max_gap:
                            value = a + (b - a) * age / (right_time - left_time)
            valid = value is not None
            vector.append(value if valid else 0.0)
            mask.append(valid)
            native.append(bool(exact and valid))
            age_vector.append(age if valid else None)
        values.append(vector)
        usable.append(mask)
        observed.append(native)
        ages.append(age_vector)
    coverage = {
        channel: sum(row[i] for row in usable) / count for i, channel in enumerate(channels)
    }
    native_fraction = {
        channel: sum(row[i] for row in observed) / count for i, channel in enumerate(channels)
    }
    result = {
        "sequence": values,
        "sequence_mask": usable,
        "sequence_observed_mask": observed,
        "sequence_sample_age": ages,
        "sequence_channels": list(channels),
        "telemetry_coverage": min(coverage.values()),
        "channel_coverage": coverage,
        "native_grid_fraction": native_fraction,
        "native_sample_count": sum(grid[0] <= t <= cutoff for t in times),
    }
    for i, channel in enumerate(channels):
        channel_values = [row[i] for row, mask in zip(values, usable) if mask[i]]
        for suffix in ("mean", "std", "min", "max", "last"):
            result[f"{channel}_{suffix}"] = None
        if channel_values:
            result.update(
                {
                    f"{channel}_mean": statistics.mean(channel_values),
                    f"{channel}_std": statistics.pstdev(channel_values),
                    f"{channel}_min": min(channel_values),
                    f"{channel}_max": max(channel_values),
                    f"{channel}_last": channel_values[-1],
                }
            )
    return result


def _pit_intervals(pit: list[dict], identity: tuple) -> list[tuple[float, float | None]]:
    result = []
    for time, row in _events(pit, identity[0]):
        if row.get("driver_number") != identity[1]:
            continue
        duration = _number(row.get("lane_duration", row.get("pit_duration")))
        result.append((time, time + duration if duration is not None and duration >= 0 else None))
    return result


def _pit_state(intervals: list, cutoff: float, previous: list) -> bool | None:
    started = [(start, end) for start, end in intervals if start <= cutoff]
    if started:
        start, end = started[-1]
        if end is not None:
            return cutoff < end
        # A later completed running sector is evidence of exit under event-time replay.
        if any(r["end"] is not None and start < r["end"] <= cutoff for r in previous):
            return False
        return None
    return False if previous else None


def _context(
    cutoff: float, lap: dict, stints: list, weather: list, intervals: list, laps_by_key: dict
) -> dict:
    session, driver = _identity(lap)
    result = {
        "tyre_age_laps": None,
        "context": {"compound": None},
        **dict.fromkeys(
            (
                "air_temperature",
                "track_temperature",
                "humidity",
                "rainfall",
                "wind_speed",
                "gap_to_leader",
                "interval",
            )
        ),
    }
    known_stints = []
    for row in stints:
        if _identity(row) != (session, driver):
            continue
        start_lap = row.get("lap_start")
        anchor = laps_by_key.get((session, driver, start_lap))
        if anchor is not None and anchor <= cutoff:
            known_stints.append((anchor, row))
    if known_stints:
        _, stint = max(known_stints, key=lambda x: x[0])
        result["context"]["compound"] = stint.get("compound")
        age = _number(stint.get("tyre_age_at_start"))
        if age is not None:
            result["tyre_age_laps"] = age + max(0, lap["lap_number"] - stint["lap_start"])
    weather_history = [(t, r) for t, r in _events(weather, session) if t <= cutoff]
    if weather_history:
        time, row = weather_history[-1]
        # Weather older than five minutes is stale, rather than forward-filled indefinitely.
        if cutoff - time <= 300:
            for name in (
                "air_temperature",
                "track_temperature",
                "humidity",
                "rainfall",
                "wind_speed",
            ):
                result[name] = _number(row.get(name))
    traffic = [
        (t, r)
        for t, r in _events(intervals, session)
        if t <= cutoff and r.get("driver_number") == driver
    ]
    if traffic:
        time, row = traffic[-1]
        if cutoff - time <= 15:
            result["gap_to_leader"] = _number(row.get("gap_to_leader"))
            result["interval"] = _number(row.get("interval"))
    return result


def build_examples(
    laps: list[dict],
    car_data: list[dict],
    *,
    race_control: list[dict],
    pit: list[dict],
    stints: list[dict] | None = None,
    weather: list[dict] | None = None,
    intervals: list[dict] | None = None,
    protocol: dict | None = None,
) -> list[dict]:
    """Construct independent features, origin eligibility, and target validity.

    Label-only exclusions never change features or the decision to issue a
    forecast. A surprise red flag censors scoring, not historical eligibility.
    Mexico can be transformed for replay; fitting exclusion belongs to split
    and training functions rather than this stateless transformation.
    """
    policy = {**DEFAULTS, **((protocol or {}).get("prediction", {}))}
    if (
        policy["feature_guard_seconds"] <= 0
        or policy["tensor_grid_hz"] <= 0
        or policy["window_seconds"] <= 0
    ):
        raise ValueError("positive feature guard, grid frequency and window required")
    if set(policy["required_channels"]) - set(CHANNELS):
        raise ValueError("unsupported required telemetry channel")
    sectors = _sector_records(laps)
    cars = _car_records(car_data)
    grouped = defaultdict(list)
    for record in sectors:
        grouped[_identity(record["lap"])].append(record)
    lap_anchors = {
        (*_identity(row), row.get("lap_number")): _time(row.get("date_start")) for row in laps
    }
    result = []
    for identity, driver_sectors in grouped.items():
        flags = _flags(race_control, identity[0])
        pits = _pit_intervals(pit, identity)
        car = cars.get(identity, {"times": [], "samples": [], "duplicates": 0})
        for record in driver_sectors:
            lap, sector = record["lap"], record["sector_index"]
            origin = record["start"]
            key = f"{identity[0]}:{identity[1]}:{lap.get('lap_number')}:{sector}"
            row = {
                "example_id": key,
                "session_key": identity[0],
                "meeting_key": lap.get("meeting_key"),
                "driver_number": identity[1],
                "lap_number": lap.get("lap_number"),
                "sector_index": sector,
                "origin": _iso(origin) if origin is not None else None,
                "cutoff": None,
                "reference_seconds": None,
                "target_seconds": record["duration"],
                "features": {},
                "eligible": False,
                "exclusion_reasons": [],
                "target_valid": False,
                "target_exclusion_reasons": list(record["timing_reasons"]),
                "provenance": {
                    "anchor": "approximate_lap_start_plus_completed_sector_durations",
                    "availability": "event_time_not_verified_live_publication_time",
                    "pit_state_assumption": "bounded_pit_records_and_completed_running_sectors",
                    "source_identity": {"session_key": identity[0], "driver_number": identity[1]},
                    "grid_adds_information": False,
                },
            }
            if origin is None:
                row["exclusion_reasons"].append("invalid_sector_anchor")
                result.append(row)
                continue
            cutoff = origin - policy["feature_guard_seconds"]
            row["cutoff"] = _iso(cutoff)
            prior = [r for r in driver_sectors if _historical_timing_valid(r, cutoff)]
            history = []
            for previous in prior:
                if previous["sector_index"] != sector or previous["lap"].get("is_pit_out_lap"):
                    continue
                start, end = previous["start"], previous["end"]
                was_green = _state_at(flags, start) == "GREEN" and not any(
                    start < t <= end and f != "GREEN" for t, f in flags
                )
                was_pitting = any(
                    start < (pit_end if pit_end is not None else math.inf) and end > pit_start
                    for pit_start, pit_end in pits
                )
                if was_green and not was_pitting:
                    history.append(previous)
            history.sort(key=lambda r: r["end"])
            recent = [r["duration"] for r in history[-3:]]
            reference = statistics.median(recent) if recent else None
            state = _state_at(flags, cutoff)
            # Full-lap validity is retrospective until the complete lap ends.
            # Incomplete-lap sectors must not establish an inferred pit exit:
            # their validity could depend on the future target duration.
            completed_lap_evidence = [
                r
                for r in prior
                if r["lap_completed_at"] is not None and r["lap_completed_at"] <= cutoff
            ]
            pitting = _pit_state(pits, cutoff, completed_lap_evidence)
            features = _window(car, cutoff, flags, policy)
            features.update(
                _context(cutoff, lap, stints or [], weather or [], intervals or [], lap_anchors)
            )
            features.update(
                {
                    "lap_number": lap.get("lap_number"),
                    "sector_index": sector,
                    "driver_number": identity[1],
                    "reference_seconds": reference,
                    "last_same_sector_seconds": recent[-1] if recent else None,
                    "same_sector_history_count": len(history),
                }
            )
            features["context"].update({"track_state": state, "in_pit_lane": pitting})
            row["features"], row["reference_seconds"] = features, reference
            if state is None:
                row["exclusion_reasons"].append("unknown_track_state")
            elif state != "GREEN":
                row["exclusion_reasons"].append("non_green_origin")
            if pitting is None:
                row["exclusion_reasons"].append("unknown_pit_state")
            elif pitting:
                row["exclusion_reasons"].append("pit_lane_origin")
            if len(history) < 2:
                row["exclusion_reasons"].append("insufficient_history")
            if features["telemetry_coverage"] < policy["minimum_required_channel_coverage"]:
                row["exclusion_reasons"].append("insufficient_telemetry_coverage")
            row["eligible"] = not row["exclusion_reasons"]
            end = record["end"]
            if end is not None and any(origin <= t <= end and flag == "RED" for t, flag in flags):
                row["target_exclusion_reasons"].append("red_flag_target")
            if _state_at(flags, origin) == "RED":
                row["target_exclusion_reasons"].append("red_flag_target")
            row["target_exclusion_reasons"] = sorted(set(row["target_exclusion_reasons"]))
            row["target_valid"] = not row["target_exclusion_reasons"] and end is not None
            row["uninterrupted_green_target"] = bool(
                end is not None
                and _state_at(flags, origin) == "GREEN"
                and not any(origin < t <= end and flag != "GREEN" for t, flag in flags)
            )
            # This is label-side metadata, never an input to eligibility.
            row["provenance"]["target_end"] = _iso(end) if end is not None else None
            row["provenance"]["maximum_feature_timestamp"] = _iso(cutoff)
            result.append(row)
    return sorted(
        result,
        key=lambda row: (
            str(row["session_key"]),
            str(row["driver_number"]),
            row["lap_number"],
            row["sector_index"],
        ),
    )


def quality_report(examples: list[dict]) -> dict:
    """Report retention without silently treating invalid targets as forecasts."""
    reasons = Counter(reason for row in examples for reason in row["exclusion_reasons"])
    target_reasons = Counter(
        reason for row in examples for reason in row.get("target_exclusion_reasons", [])
    )
    nominal = [
        r
        for r in examples
        if r.get("features", {}).get("context", {}).get("track_state") == "GREEN"
        and r.get("features", {}).get("context", {}).get("in_pit_lane") is False
    ]
    eligible = sum(row["eligible"] for row in examples)
    groups = defaultdict(list)
    for row in examples:
        groups[f"{row['session_key']}:{row['driver_number']}"].append(row)
    return {
        "total_examples": len(examples),
        "nominal_green_opportunities": len(nominal),
        "eligible_examples": eligible,
        "scoreable_examples": sum(
            row["eligible"] and row.get("target_valid", True) for row in examples
        ),
        "retention": eligible / len(nominal) if nominal else None,
        "origin_exclusion_counts": dict(sorted(reasons.items())),
        "target_exclusion_counts": dict(sorted(target_reasons.items())),
        "by_session_driver": {
            key: {"total": len(rows), "eligible": sum(r["eligible"] for r in rows)}
            for key, rows in sorted(groups.items())
        },
        "timing_assumption": "reconstructed_event_time_not_measured_live_latency",
    }


def write_examples(examples: list[dict], path: str | Path) -> None:
    """Write JSON or Parquet examples; require explicit extension selection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(examples, allow_nan=False, indent=2) + "\n")
    elif path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(examples), path)
    else:
        raise ValueError("examples output must use .json or .parquet")
