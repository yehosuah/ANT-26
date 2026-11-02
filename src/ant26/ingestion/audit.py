"""Inspect source quality without turning completeness into a modeling claim."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ant26.ingestion.catalog import build_catalog
from ant26.ingestion.client import OpenF1Client, ProviderError


def audit_rows(endpoint: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({field for row in rows for field in row})
    report: dict[str, Any] = {
        "endpoint": endpoint,
        "rows": len(rows),
        "fields": fields,
        "null_counts": {field: sum(row.get(field) is None for row in rows) for field in fields},
        "exact_duplicate_rows": len(rows) - len({json.dumps(row, sort_keys=True) for row in rows}),
    }
    if endpoint == "car_data":
        dates = sorted(datetime.fromisoformat(row["date"].replace("Z", "+00:00")) for row in rows)
        gaps = [(b - a).total_seconds() for a, b in zip(dates, dates[1:])]
        report.update(
            {
                "first_timestamp": dates[0].isoformat() if dates else None,
                "last_timestamp": dates[-1].isoformat() if dates else None,
                "duplicate_timestamps": sum(gap == 0 for gap in gaps),
                "median_gap_seconds": statistics.median(gaps) if gaps else None,
                "max_gap_seconds": max(gaps) if gaps else None,
                "gaps_over_one_second": sum(gap > 1 for gap in gaps),
            }
        )
    if endpoint == "laps":
        numbers = sorted({int(row["lap_number"]) for row in rows})
        inconsistent = []
        for row in rows:
            sector_values = [row.get(f"duration_sector_{n}") for n in range(1, 4)]
            if row.get("lap_duration") is not None and all(
                value is not None for value in sector_values
            ):
                if abs(sum(sector_values) - row["lap_duration"]) > 0.01:
                    inconsistent.append(row["lap_number"])
        report.update(
            {
                "lap_numbers": numbers,
                "internal_missing_laps": sorted(set(range(1, max(numbers) + 1)) - set(numbers))
                if numbers
                else [],
                "duplicate_lap_numbers": [
                    key
                    for key, count in Counter(row["lap_number"] for row in rows).items()
                    if count > 1
                ],
                "sector_sum_mismatch_laps": inconsistent,
                "completeness_denominator": "max_observed_lap_only_not_official_completed_laps",
            }
        )
    return report


def audit_session(
    client: OpenF1Client,
    session: dict[str, Any],
    *,
    driver_numbers: tuple[int, ...] = (12, 63),
    telemetry_window_seconds: int = 180,
) -> dict[str, Any]:
    """Fetch an audit's source inputs with Mexico rejected at the boundary."""
    if session.get("meeting_key") == 1298 or session.get("session_key") == 11404:
        raise ValueError("Mexico is quarantined from development ingestion")
    session_key = session["session_key"]
    report: dict[str, Any] = {"session": session, "sources": [], "errors": []}
    queries = [
        (endpoint, {"session_key": session_key})
        for endpoint in ("race_control", "weather", "drivers")
    ]
    queries += [
        (endpoint, {"session_key": session_key, "driver_number": driver})
        for driver in driver_numbers
        for endpoint in ("laps", "car_data", "stints", "intervals", "pit")
    ]
    for endpoint, params in queries:
        try:
            if endpoint == "car_data" and telemetry_window_seconds:
                lap_rows = client.fetch("laps", params)
                anchor = next((row for row in lap_rows if row.get("lap_number") == 10), None)
                anchor = anchor or next((row for row in lap_rows if row.get("date_start")), None)
                if anchor:
                    start = datetime.fromisoformat(anchor["date_start"].replace("Z", "+00:00"))
                    params = params | {
                        "date>": start.astimezone(timezone.utc).replace(tzinfo=None).isoformat(),
                        "date<": (start + timedelta(seconds=telemetry_window_seconds))
                        .astimezone(timezone.utc)
                        .replace(tzinfo=None)
                        .isoformat(),
                    }
            rows = client.fetch(endpoint, params)
            report["sources"].append(
                {
                    "query": params,
                    "quality": audit_rows(endpoint, rows),
                    "manifest": dict(client.last_manifest or {}),
                }
            )
        except ProviderError as exc:
            report["errors"].append({"endpoint": endpoint, "query": params, "error": str(exc)})
    return report


def run_feasibility_audit(
    cache_root: Path | str = Path("data/raw"),
    output_root: Path | str = Path("data/audit"),
    *,
    overrides_path: Path | str = Path("configs/catalog-overrides.json"),
    driver_numbers: tuple[int, ...] = (12, 63),
    sample_races: int = 3,
    offline: bool = False,
    telemetry_window_seconds: int = 180,
) -> dict[str, Any]:
    client = OpenF1Client(cache_root, offline=offline)
    sessions = client.fetch("sessions", {"year": 2026, "session_name": "Race"})
    session_manifest = dict(client.last_manifest or {})
    meetings = client.fetch("meetings", {"year": 2026})
    overrides = (
        json.loads(Path(overrides_path).read_text()) if Path(overrides_path).exists() else {}
    )
    catalog = build_catalog(sessions, meetings, overrides=overrides)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "catalog.json").write_text(json.dumps(catalog, indent=2))
    chosen = catalog["development"][-sample_races:]
    reports = []
    for session in chosen:
        reports.append(
            audit_session(
                client,
                session,
                driver_numbers=driver_numbers,
                telemetry_window_seconds=telemetry_window_seconds,
            )
        )
        result = {
            "schema_version": "ant26-audit-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "catalog_source": session_manifest,
            "development_races": len(catalog["development"]),
            "sample_driver_numbers": list(driver_numbers),
            "sample_races": reports,
            "telemetry_window_seconds_per_driver": telemetry_window_seconds,
            "mexico_ingested": False,
            "scope": "source_feasibility_not_full_season_retention_or_model_validation",
        }
        (output_root / "audit.json").write_text(json.dumps(result, indent=2))
    if not reports:
        raise ProviderError("catalog contains no eligible development races")
    return result
