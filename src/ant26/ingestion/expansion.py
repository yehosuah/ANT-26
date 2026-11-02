"""Resumable bounded telemetry expansion after the source-feasibility audit."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ant26.ingestion.client import OpenF1Client, ProviderError, _atomic_json


def _date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source dates must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def expand_full_traces(
    client: OpenF1Client,
    *,
    session_keys: tuple[int, ...] = (11396, 11388, 11731),
    driver_numbers: tuple[int, ...] = (12, 63),
    progress_path: Path | str = Path("data/audit/full-trace-progress.json"),
    chunk_seconds: int = 60,
    on_session_complete: Any = None,
) -> dict[str, Any]:
    """Expand driver-first, checkpointing source manifests after every chunk.

    Successful chunks remain in immutable cache; interrupted runs reuse them.
    Failed chunks are recorded and subsequent chunks continue, with the final
    status partial rather than falsely complete.
    """
    if 11404 in session_keys:
        raise ValueError("Mexico is quarantined")
    if chunk_seconds <= 0:
        raise ValueError("chunk size must be positive")
    progress_path = Path(progress_path)
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "schema_version": "ant26-full-trace-progress-v1",
            "sessions": {},
            "mexico_ingested": False,
        }
    )
    for driver in driver_numbers:
        for session_key in session_keys:
            laps = client.fetch("laps", {"session_key": session_key, "driver_number": driver})
            valid = [row for row in laps if row.get("date_start") and row.get("lap_duration")]
            if not valid:
                raise ProviderError(f"no timed laps for {session_key}/{driver}")
            if any(row.get("meeting_key") == 1298 for row in laps):
                raise ValueError("Mexico meeting is quarantined")
            start = min(_date(row["date_start"]) for row in valid)
            stop = max(
                _date(row["date_start"]) + timedelta(seconds=row["lap_duration"]) for row in valid
            )
            identity = f"{session_key}:{driver}"
            entry = progress["sessions"].setdefault(
                identity,
                {
                    "session_key": session_key,
                    "driver_number": driver,
                    "start": start.isoformat(),
                    "end": stop.isoformat(),
                    "chunks": {},
                    "failed_bounds": [],
                },
            )
            if entry["start"] != start.isoformat() or entry["end"] != stop.isoformat():
                raise ProviderError(
                    "lap-source bounds changed; version the expansion progress file"
                )
            entry["status"] = "running"
            entry["failed_bounds"] = []
            left = start
            while left < stop:
                right = min(left + timedelta(seconds=chunk_seconds), stop)
                key = f"{left.isoformat()}/{right.isoformat()}"
                try:
                    rows = client.fetch_time_range(
                        "car_data",
                        {"session_key": session_key, "driver_number": driver},
                        left,
                        right,
                        chunk_seconds=chunk_seconds,
                    )
                    entry["chunks"][key] = {
                        "start": left.isoformat(),
                        "end": right.isoformat(),
                        "rows": len(rows),
                        "manifest": client.last_window_manifests[0],
                    }
                except ProviderError as exc:
                    entry["failed_bounds"].append(
                        {"start": left.isoformat(), "end": right.isoformat(), "error": str(exc)}
                    )
                entry["completed_chunks"] = len(entry["chunks"])
                entry["rows"] = sum(chunk["rows"] for chunk in entry["chunks"].values())
                progress["updated_at"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(progress_path, progress)
                left = right
            entry["status"] = "partial" if entry["failed_bounds"] else "complete"
            _atomic_json(progress_path, progress)
            if on_session_complete:
                on_session_complete(dict(entry))
    return progress
