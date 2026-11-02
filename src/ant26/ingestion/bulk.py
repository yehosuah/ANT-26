"""Acquire complete development telemetry efficiently through bounded CSV snapshots."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ant26.ingestion.client import OpenF1Client, ProviderError, _atomic_json


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source time must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def ingest_development(
    client: OpenF1Client,
    catalog: dict[str, Any],
    *,
    progress_path: Path | str = Path("data/audit/bulk-progress.json"),
    driver_numbers: tuple[int, ...] = (12,),
    session_keys: tuple[int, ...] | None = None,
    on_session_complete: Any = None,
) -> dict[str, Any]:
    """Persist full race-bounded telemetry and context, checkpointing each query.

    The progress schema matches bounded expansion: each completed session has
    chunks containing normalized-JSON manifests. CSV source bytes are preserved
    separately and both representations have verified checksums.
    """
    path = Path(progress_path)
    progress = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "schema_version": "ant26-full-trace-progress-v1",
            "data_kind": "real_openf1",
            "acquisition": "race_bounded_csv",
            "sessions": {},
            "mexico_ingested": False,
        }
    )
    races = [
        row
        for row in catalog["development"]
        if session_keys is None or row["session_key"] in session_keys
    ]
    if any(row.get("meeting_key") == 1298 or row.get("session_key") == 11404 for row in races):
        raise ValueError("Mexico is quarantined from development ingestion")
    for driver in driver_numbers:
        for race in races:
            key = race["session_key"]
            identity = f"{key}:{driver}"
            entry = progress["sessions"].setdefault(
                identity,
                {
                    "session_key": key,
                    "driver_number": driver,
                    "chunks": {},
                    "context_sources": {},
                },
            )
            entry.update({"status": "running", "failed_bounds": []})
            try:
                sources = {}
                for endpoint in (
                    "laps",
                    "race_control",
                    "weather",
                    "drivers",
                    "stints",
                    "intervals",
                    "pit",
                ):
                    params = {"session_key": key}
                    if endpoint not in {"race_control", "weather", "drivers"}:
                        params["driver_number"] = driver
                    if endpoint == "pit":
                        sources[endpoint] = client.fetch(endpoint, params, allow_no_results=True)
                    else:
                        sources[endpoint] = client.fetch(endpoint, params)
                    entry["context_sources"][endpoint] = dict(client.last_manifest or {})
                    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _atomic_json(path, progress)
                valid = [
                    row
                    for row in sources["laps"]
                    if row.get("date_start") and row.get("lap_duration")
                ]
                if not valid:
                    entry.update(
                        {"status": "unscorable", "reason": "no_valid_timed_laps", "rows": 0}
                    )
                    _atomic_json(path, progress)
                    if on_session_complete:
                        on_session_complete(dict(entry))
                    continue
                start = min(_utc(row["date_start"]) for row in valid)
                end = max(
                    _utc(row["date_start"]) + timedelta(seconds=row["lap_duration"])
                    for row in valid
                )
                params = {
                    "session_key": key,
                    "driver_number": driver,
                    "date>": (start - timedelta(microseconds=1)).replace(tzinfo=None).isoformat(),
                    "date<": end.replace(tzinfo=None).isoformat(),
                }
                rows = client.fetch("car_data", params, response_format="csv")
                if not rows:
                    raise ProviderError("no telemetry returned for a timed race")
                for row in rows:
                    if row.get("session_key") != key or row.get("driver_number") != driver:
                        raise ProviderError("provider ignored telemetry identity filters")
                    if not start <= _utc(row["date"]) < end:
                        raise ProviderError("provider ignored telemetry time filters")
                source = dict(client.last_manifest or {})
                entry.update(
                    {
                        "status": "complete",
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "rows": len(rows),
                        "completed_chunks": 1,
                        "chunks": {
                            "full_race_csv": {
                                "start": start.isoformat(),
                                "end": end.isoformat(),
                                "rows": len(rows),
                                "manifest": source,
                            }
                        },
                    }
                )
            except (ProviderError, ValueError, KeyError, TypeError) as exc:
                entry["status"] = "partial"
                entry["failed_bounds"].append({"error": str(exc)})
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(path, progress)
            if on_session_complete:
                on_session_complete(dict(entry))
    return progress
