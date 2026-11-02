"""Acquire the final race only after a verified, immutable release decision."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ant26.datasets.features import build_examples, quality_report
from ant26.domain.protocol import protocol_hash
from ant26.ingestion.client import OpenF1Client, ProviderError, _atomic_json


def _write(path: Path, payload: dict) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != serialized:
            raise ValueError("Case-study artifact already exists with different bytes")
    else:
        with path.open("xb") as handle:
            handle.write(serialized)
    return hashlib.sha256(serialized).hexdigest()


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Case-study source time must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def ingest_case_study(
    client: OpenF1Client,
    freeze_path: str | Path,
    protocol: dict,
    *,
    output_root: str | Path = "data/case-study/mexico",
    on_endpoint=None,
) -> dict:
    """Fetch exact Mexico snapshots without relaxing development quarantine.

    The release is verified before the first network/cache query. Metadata and
    source completeness are retained separately from past-only model inputs.
    Resuming uses the provider client's immutable cache, never rewritten data.
    """
    from ant26.release import verify_release

    release = verify_release(freeze_path, protocol)
    freeze_sha = hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest()
    case = protocol["final_case_study"]
    session, meeting = case["race_session_key"], case["meeting_key"]
    if (session, meeting) != (11404, 1298):
        raise ValueError("Only the predeclared Mexico case study is supported")
    root = Path(output_root)
    progress_path = root / "progress.json"
    progress = {"freeze_sha256": freeze_sha, "session_key": session, "endpoints": {}}
    manifests, sources = [], {}

    def fetch(endpoint, params, **kwargs):
        rows = client.fetch(endpoint, params, **kwargs)
        manifest = dict(client.last_manifest or {})
        if not manifest.get("sha256"):
            raise ProviderError("Case-study query did not produce a pinned snapshot")
        manifests.append({key: value for key, value in manifest.items() if key != "cache_hit"})
        progress["endpoints"][endpoint] = {"row_count": len(rows), "sha256": manifest["sha256"]}
        _atomic_json(progress_path, progress)
        if on_endpoint:
            on_endpoint(endpoint, len(rows))
        return rows

    identity = fetch("sessions", {"session_key": session})
    if len(identity) != 1 or identity[0].get("meeting_key") != meeting:
        raise ProviderError("Case-study session identity does not match the frozen target")
    for endpoint in (
        "laps",
        "race_control",
        "weather",
        "stints",
        "intervals",
        "pit",
        "session_result",
    ):
        params = {"session_key": session}
        if endpoint not in {"race_control", "weather"}:
            params["driver_number"] = 12
        sources[endpoint] = fetch(endpoint, params, allow_no_results=endpoint == "pit")
    laps = sources["laps"]
    if any(row.get("session_key") != session or row.get("driver_number") != 12 for row in laps):
        raise ProviderError("Case-study lap query escaped the frozen subject")
    timed = [row for row in laps if row.get("date_start") and row.get("lap_duration")]
    if not timed:
        raise ProviderError("Case-study source contains no complete timed laps")
    start = min(_time(row["date_start"]) for row in timed)
    end = max(_time(row["date_start"]) + timedelta(seconds=row["lap_duration"]) for row in timed)
    car = fetch(
        "car_data",
        {
            "session_key": session,
            "driver_number": 12,
            "date>": (start - timedelta(microseconds=1)).replace(tzinfo=None).isoformat(),
            "date<": end.replace(tzinfo=None).isoformat(),
        },
        response_format="csv",
    )
    if not car or any(
        row.get("session_key") != session
        or row.get("driver_number") != 12
        or not start <= _time(row["date"]) < end
        for row in car
    ):
        raise ProviderError("Case-study telemetry identity or bounds are invalid")
    examples = build_examples(
        laps,
        car,
        protocol=protocol,
        **{
            name: sources[name]
            for name in ("stints", "intervals", "pit", "race_control", "weather")
        },
    )
    source_identity = [{"endpoint": row["endpoint"], "sha256": row["sha256"]} for row in manifests]
    payload = {
        "data_kind": "real_openf1_mexico_case_study",
        "protocol_sha256": protocol_hash(protocol),
        "freeze_sha256": freeze_sha,
        "freeze_manifest": str(Path(freeze_path)),
        "feature_builder_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "datasets" / "features.py").read_bytes()
        ).hexdigest(),
        "session_key": session,
        "meeting_key": meeting,
        "driver_number": 12,
        "expected_laps": case["expected_race_laps"],
        "observed_laps": sorted({row["lap_number"] for row in laps}),
        "telemetry_rows": len(car),
        "source_manifests": manifests,
        "source_identity": source_identity,
        "source_session": identity[0],
        "session_result": sources["session_result"],
        "race_control": sources["race_control"],
        "stints": sources["stints"],
        "pit": sources["pit"],
        "examples": examples,
        "quality": quality_report(examples),
        "replay_assumption": "event_time_not_verified_live_publication_time",
        "release_verified": bool(release),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    path = root / f"examples-{digest[:16]}.json"
    _write(path, payload)
    progress.update({"status": "complete", "dataset_path": str(path), "dataset_sha256": digest})
    _atomic_json(progress_path, progress)
    return {
        "path": str(path),
        "sha256": digest,
        "telemetry_rows": len(car),
        "quality": payload["quality"],
    }
