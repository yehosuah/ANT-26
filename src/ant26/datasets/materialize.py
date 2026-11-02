"""Incrementally materialize exact completed source snapshots without network IO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ant26.datasets.features import build_examples, quality_report
from ant26.domain.protocol import protocol_hash

ENDPOINTS = ("laps", "stints", "intervals", "pit", "race_control", "weather")


def _feature_code_sha256() -> str:
    """Invalidate cached examples when any feature helper changes."""
    return hashlib.sha256(Path(__file__).with_name("features.py").read_bytes()).hexdigest()


def _data_kind(driver_number: int) -> str:
    return {12: "real_openf1_antonelli", 63: "real_openf1_russell"}.get(
        driver_number, "real_openf1_single_driver"
    )


def _bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(
                "immutable materialization artifact has different existing bytes"
            ) from None


def _snapshot(manifest: dict) -> list[dict]:
    directory = Path(manifest["snapshot_path"])
    payload = (directory / "response.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("source snapshot checksum mismatch during materialization")
    if manifest.get("raw_response_filename"):
        original = (directory / manifest["raw_response_filename"]).read_bytes()
        if hashlib.sha256(original).hexdigest() != manifest["raw_response_sha256"]:
            raise ValueError("original response checksum mismatch during materialization")
    rows = json.loads(payload)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("snapshot must contain a list of source objects")
    if len(rows) != manifest["row_count"]:
        raise ValueError("source row count disagrees with manifest")
    return rows


def _manifest(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != "cache_hit"}


def _source_manifests(entry: dict) -> list[dict]:
    contexts = entry.get("context_sources", {})
    missing = set(ENDPOINTS) - set(contexts)
    if missing:
        raise ValueError(f"completed entry lacks required contexts: {sorted(missing)}")
    chunks = entry.get("chunks", {})
    if not chunks:
        raise ValueError("completed entry has no telemetry snapshots")
    return [_manifest(contexts[name]) for name in ENDPOINTS] + [
        _manifest(chunk["manifest"]) for _, chunk in sorted(chunks.items())
    ]


def materialize_available_sessions(
    progress_path: str | Path,
    catalog_path: str | Path,
    protocol: dict,
    *,
    output_root: str | Path = "data/processed/materialized",
    driver_number: int = 12,
) -> dict:
    """Reuse completed session artifacts and merge the currently complete cohort.

    The mutable acquisition-progress file is read once. Every source subsequently
    loaded is pinned to that read's exact immutable snapshot, never a latest-cache
    pointer. The merged artifact explicitly lists incomplete or absent sessions.
    """
    progress = json.loads(Path(progress_path).read_text())
    catalog = json.loads(Path(catalog_path).read_text())
    output_root = Path(output_root)
    active_hash = protocol_hash(protocol)
    feature_hash = _feature_code_sha256()
    examples, all_manifests, completed, skipped, session_paths = [], [], [], [], []
    for race in sorted(catalog["development"], key=lambda row: row["date_start"]):
        session = race["session_key"]
        if race.get("meeting_key") == 1298 or session in (11397, 11398, 11399, 11400, 11404):
            raise ValueError("Mexico cannot enter a development materialization")
        if race.get("is_cancelled"):
            skipped.append({"session_key": session, "reason": "canceled"})
            continue
        if race.get("verified_identity") is not True:
            skipped.append({"session_key": session, "reason": "unverified_identity"})
            continue
        entry = progress.get("sessions", {}).get(f"{session}:{driver_number}", {})
        if entry.get("status") != "complete":
            skipped.append(
                {
                    "session_key": session,
                    "driver_number": driver_number,
                    "reason": entry.get("reason", "source_not_complete"),
                    **(
                        {
                            "source_manifests": [
                                _manifest(m) for m in entry.get("context_sources", {}).values()
                            ]
                        }
                        if entry.get("status") == "unscorable"
                        else {}
                    ),
                }
            )
            continue
        manifests = _source_manifests(entry)
        identity = {
            "protocol_sha256": active_hash,
            "feature_builder_sha256": feature_hash,
            "session_key": session,
            "driver_number": driver_number,
            "sources": [
                {
                    "endpoint": row["endpoint"],
                    "query_key": row["query_key"],
                    "sha256": row["sha256"],
                }
                for row in manifests
            ],
        }
        source_hash = hashlib.sha256(_bytes(identity)).hexdigest()
        directory = output_root / f"session-{session}"
        pointer_path = directory / f"sources-{source_hash}.json"
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text())
            path = directory / pointer["filename"]
            serialized = path.read_bytes()
            if hashlib.sha256(serialized).hexdigest() != pointer["artifact_sha256"]:
                raise ValueError("cached session materialization checksum mismatch")
            payload = json.loads(serialized)
        else:
            contexts = {name: _snapshot(entry["context_sources"][name]) for name in ENDPOINTS}
            car = [
                row
                for _, chunk in sorted(entry["chunks"].items())
                for row in _snapshot(chunk["manifest"])
            ]
            for row in contexts["laps"] + car:
                if row.get("session_key") != session or row.get("driver_number") != driver_number:
                    raise ValueError("source row escapes declared session/driver cohort")
            session_examples = build_examples(
                contexts.pop("laps"), car, protocol=protocol, **contexts
            )
            payload = {
                "data_kind": _data_kind(driver_number),
                "protocol_sha256": active_hash,
                "feature_builder_sha256": feature_hash,
                "input_sources_sha256": source_hash,
                "cohort_driver_numbers": [driver_number],
                "session_key": session,
                "examples": session_examples,
                "quality": quality_report(session_examples),
                "source_manifests": manifests,
            }
            serialized = _bytes(payload)
            digest = hashlib.sha256(serialized).hexdigest()
            path = directory / f"examples-{digest[:16]}.json"
            _write(path, serialized)
            _write(pointer_path, _bytes({"filename": path.name, "artifact_sha256": digest}))
        if (
            payload.get("protocol_sha256") != active_hash
            or payload.get("feature_builder_sha256") != feature_hash
            or payload.get("input_sources_sha256") != source_hash
        ):
            raise ValueError("cached session protocol or source identity mismatch")
        if any(
            row.get("driver_number") != driver_number or row.get("session_key") != session
            for row in payload["examples"]
        ):
            raise ValueError("cached materialization escapes declared cohort")
        examples.extend(payload["examples"])
        all_manifests.extend(payload["source_manifests"])
        completed.append(session)
        session_paths.append(str(path))
    if not completed:
        raise ValueError("no verified completed driver sessions available")
    merged = {
        "data_kind": _data_kind(driver_number),
        "protocol_sha256": active_hash,
        "feature_builder_sha256": feature_hash,
        "cohort_driver_numbers": [driver_number],
        "examples": examples,
        "quality": quality_report(examples),
        "source_manifests": all_manifests,
        "completed_sessions": completed,
        "skipped": skipped,
    }
    serialized = _bytes(merged)
    digest = hashlib.sha256(serialized).hexdigest()
    path = output_root / f"merged-{digest[:16]}.json"
    _write(path, serialized)
    return {
        "path": str(path),
        "artifact_sha256": digest,
        "completed_sessions": completed,
        "session_artifacts": session_paths,
        "skipped": skipped,
        "quality": merged["quality"],
    }


def merge_materialized_cohorts(
    dataset_paths: list[str | Path],
    protocol: dict,
    *,
    output_root: str | Path = "data/processed/mercedes",
) -> dict:
    """Merge disjoint driver cohorts, recording each immutable dataset checksum."""
    if not dataset_paths:
        raise ValueError("at least one materialized cohort is required")
    active_hash, feature_hash = protocol_hash(protocol), _feature_code_sha256()
    examples, source_files, manifests, completed, skipped, pairs = [], [], [], set(), [], []
    drivers, identifiers = set(), set()
    for dataset_path in sorted(map(Path, dataset_paths), key=str):
        serialized = dataset_path.read_bytes()
        payload = json.loads(serialized)
        if (
            payload.get("protocol_sha256") != active_hash
            or payload.get("feature_builder_sha256") != feature_hash
        ):
            raise ValueError(
                "cohort protocol or feature-code checksum differs from the active build"
            )
        cohort = set(payload.get("cohort_driver_numbers", []))
        if not cohort or drivers & cohort:
            raise ValueError("cohort driver identities are missing or overlap")
        drivers.update(cohort)
        for row in payload["examples"]:
            if row.get("driver_number") not in cohort:
                raise ValueError("example escapes declared input cohort")
            if row.get("meeting_key") == 1298 or row.get("session_key") in (
                11397,
                11398,
                11399,
                11400,
                11404,
            ):
                raise ValueError("Mexico cannot enter a development cohort merge")
            if row["example_id"] in identifiers:
                raise ValueError("duplicate example identity across cohorts")
            identifiers.add(row["example_id"])
        examples.extend(payload["examples"])
        manifests.extend(payload.get("source_manifests", []))
        completed.update(payload.get("completed_sessions", []))
        skipped.extend(payload.get("skipped", []))
        pairs.extend(
            f"{session}:{driver}"
            for session in payload.get("completed_sessions", [])
            for driver in sorted(cohort)
        )
        source_files.append(
            {
                "path": str(dataset_path),
                "sha256": hashlib.sha256(serialized).hexdigest(),
                "cohort_driver_numbers": sorted(cohort),
            }
        )
    examples.sort(
        key=lambda row: (
            row["session_key"],
            row["driver_number"],
            row["lap_number"],
            row["sector_index"],
        )
    )
    distinct_manifests = {(m["endpoint"], m["query_key"], m["sha256"]): m for m in manifests}
    merged = {
        "data_kind": "real_openf1_mercedes"
        if drivers == {12, 63}
        else "real_openf1_declared_cohorts",
        "protocol_sha256": active_hash,
        "feature_builder_sha256": feature_hash,
        "cohort_driver_numbers": sorted(drivers),
        "source_datasets": source_files,
        "examples": examples,
        "quality": quality_report(examples),
        "source_manifests": [m for _, m in sorted(distinct_manifests.items())],
        "completed_sessions": sorted(completed),
        "completed_session_driver_pairs": sorted(pairs),
        "skipped": skipped,
    }
    serialized = _bytes(merged)
    digest = hashlib.sha256(serialized).hexdigest()
    path = Path(output_root) / f"merged-{digest[:16]}.json"
    _write(path, serialized)
    return {
        "path": str(path),
        "artifact_sha256": digest,
        "cohort_driver_numbers": sorted(drivers),
        "completed_session_driver_pairs": sorted(pairs),
        "skipped": skipped,
        "quality": merged["quality"],
    }
