"""Export derived case-study aggregates without redistributing source records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ant26.evaluation.metrics import macro_metrics


def export_case_report(run_path, freeze_path, output_path, *, chapters=(), sources=()) -> dict:
    """Create a public companion report, leaving immutable scored files intact."""
    root = Path(run_path)
    report = json.loads((root / "report.json").read_text())
    timeline = json.loads((root / "timeline.json").read_text())
    predictions = json.loads((root / "predictions.json").read_text())
    freeze = json.loads(Path(freeze_path).read_text())
    if report.get("synthetic") or report.get("status") != "case_study_scored":
        raise ValueError("Only a scored real case can become a public case report")
    freeze_sha = hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest()
    if report["provenance"]["freeze_sha256"] != freeze_sha:
        raise ValueError("Case report and freeze file disagree")
    for filename, value in (("predictions.json", predictions), ("timeline.json", timeline)):
        digest = hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if report["output_files"].get(filename) != digest:
            raise ValueError("Case output checksum differs from the scored report")
    if sorted(row["lap_number"] for row in timeline) != list(range(1, 72)):
        raise ValueError("Public accounting requires exactly the full 71-lap timeline")
    laps = []
    for row in timeline:
        sectors = row["sectors"]
        counts = Counter(sector["status"] for sector in sectors)
        reasons = Counter(reason for sector in sectors for reason in sector.get("reasons", []))
        laps.append(
            {
                "lap_number": row["lap_number"],
                "expected_sectors": 3,
                "observed_sectors": 3 - counts["missing_record"],
                "eligible_forecasts": counts["scored"] + counts["censored"],
                "scoreable_forecasts": counts["scored"],
                "abstentions": counts["abstained"],
                "censored_targets": counts["censored"],
                "missing_records": counts["missing_record"],
                "status": "missing_record"
                if counts["missing_record"] == 3
                else "scored"
                if counts["scored"] == 3
                else "mixed",
                "exclusion_reasons": dict(sorted(reasons.items())),
            }
        )
    if (
        sum(row["scoreable_forecasts"] for row in laps)
        != report["accounting"]["scoreable_forecasts"]
    ):
        raise ValueError("Timeline and reported scored counts disagree")
    if (
        sum(
            row["abstentions"]
            + row["censored_targets"]
            + row["missing_records"]
            + row["scoreable_forecasts"]
            for row in laps
        )
        != 213
    ):
        raise ValueError("Lap accounting does not reconcile to 213 sector opportunities")
    scored = [row for row in predictions if row.get("scoreable")]
    chapter_results = []
    covered = []
    for chapter in chapters:
        first, last = chapter["lap_start"], chapter["lap_end"]
        if not 1 <= first <= last <= 71:
            raise ValueError("Chapter boundaries must lie within the official race")
        covered.extend(range(first, last + 1))
        subset = [row for row in scored if first <= row["lap_number"] <= last]
        counts = [row for row in laps if first <= row["lap_number"] <= last]
        chapter_results.append(
            {
                **chapter,
                "metrics": macro_metrics(subset, 12)["macro"],
                "scoreable_forecasts": len(subset),
                "abstentions": sum(row["abstentions"] for row in counts),
                "censored_targets": sum(row["censored_targets"] for row in counts),
                "missing_sector_records": sum(row["missing_records"] for row in counts),
            }
        )
    if chapters and sorted(covered) != list(range(1, 72)):
        raise ValueError("Narrative chapters must partition all 71 laps without overlaps")
    errors = [abs(row["target_seconds"] - row["quantiles"][2]) for row in scored]
    payload = {
        "schema_version": "ant26-public-case-study-v1",
        "aggregate_only": True,
        "synthetic": False,
        "status": report["status"],
        "case_kind": report["provenance"]["case_claim"],
        "protocol_version": freeze.get("protocol_version"),
        "race": report["race"],
        "accounting": report["accounting"],
        "metrics": report["metrics"],
        "lap_accounting": laps,
        "chapters": chapter_results,
        "absolute_error_summary": {
            "count": len(errors),
            "minimum_seconds": min(errors) if errors else None,
            "maximum_seconds": max(errors) if errors else None,
        },
        "atlas_decision": report["atlas_decision"],
        "sources": list(sources),
        "provenance": {
            key: report["provenance"][key]
            for key in (
                "freeze_sha256",
                "release_manifest_sha256",
                "code_revision",
                "source_files",
                "protocol_sha256",
                "case_examples_sha256",
                "case_envelope_sha256",
                "source_manifests_sha256",
                "case_claim",
                "replay_assumption",
            )
        },
        "release": {
            "frozen_at_utc": freeze["frozen_at_utc"],
            "manifest_sha256": freeze["manifest_sha256"],
            "artifact_files": freeze["artifact_files"],
            "index_hash": freeze["index_hash"],
            "training_dataset_sha256": freeze["training_dataset"]["sha256"],
            "fit_sessions": freeze["fit_sessions"],
            "calibration_sessions": freeze["calibration_sessions"],
        },
        "local_output_checksums": report["output_files"],
        "limits": [
            "One race is a descriptive external case study, not a generalization estimate.",
            "Mexico had prior exploratory exposure; it was excluded from fitting and selection.",
            "Reconstructed event time does not establish original live data availability.",
            "Retrieval is experimental with zero human ratings; no driver-intention or championship-causation claim.",
            "Public data contain derived counts and metrics; source telemetry, individual forecasts and weights remain local.",
        ],
    }
    if correction := freeze.get("implementation_correction"):
        payload["implementation_correction"] = {
            key: value for key, value in correction.items() if key != "record_path"
        }
        payload["limits"].insert(
            0,
            "This is a post-inspection diagnostic replay after a preserved zero-forecast first attempt; it is not a new independent test.",
        )
    if revision := freeze.get("feature_policy_revision"):
        payload["feature_policy_revision"] = {
            key: value for key, value in revision.items() if key != "record_path"
        }
        payload["limits"].insert(
            0,
            "This post-inspection feature-policy replay uses refitted artifacts and revised development data. Added coverage and forecast-quality changes are distinct outcomes.",
        )
    output = Path(output_path)
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if output.exists() and output.read_text() != serialized:
        raise FileExistsError("Public case report already exists; choose a new version")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write(serialized)
    return payload
