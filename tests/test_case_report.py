import hashlib
import json

import pytest

from ant26.evaluation.case_report import export_case_report


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def case(tmp_path):
    freeze = {
        "frozen_at_utc": "2026-11-02T00:00:00+00:00",
        "manifest_sha256": "freeze-id",
        "artifact_files": {},
        "index_hash": "index-id",
        "training_dataset": {"sha256": "dataset-id"},
        "fit_sessions": [1],
        "calibration_sessions": [2],
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    timeline = [
        {
            "lap_number": lap,
            "sectors": [
                {
                    "sector_index": sector,
                    "status": "missing_record",
                    "reasons": ["missing_source_sector_record"],
                }
                for sector in range(1, 4)
            ],
        }
        for lap in range(1, 72)
    ]
    predictions = []
    report = {
        "synthetic": False,
        "status": "case_study_scored",
        "race": {"session_key": 11404, "lap_count": 71},
        "accounting": {"scoreable_forecasts": 0},
        "metrics": {},
        "atlas_decision": {"human_rating_count": 0},
        "provenance": {
            key: None
            for key in (
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
        "output_files": {"predictions.json": _sha(predictions), "timeline.json": _sha(timeline)},
    }
    report["provenance"]["freeze_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, value in [("report", report), ("timeline", timeline), ("predictions", predictions)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    return path


def test_public_export_preserves_full_missingness_without_source_records(tmp_path):
    freeze = case(tmp_path)
    result = export_case_report(tmp_path, freeze, tmp_path / "public.json")
    assert len(result["lap_accounting"]) == 71
    assert sum(row["missing_records"] for row in result["lap_accounting"]) == 213
    assert result["absolute_error_summary"]["count"] == 0
    assert "representative_predictions" not in result
    text = (tmp_path / "public.json").read_text()
    assert str(tmp_path) not in text
    assert '"predictions"' not in text
    assert '"target_seconds"' not in text
    assert result["aggregate_only"]


def test_public_export_rejects_changed_predictions_and_incomplete_chapters(tmp_path):
    freeze = case(tmp_path)
    with pytest.raises(ValueError, match="partition"):
        export_case_report(
            tmp_path, freeze, tmp_path / "public.json", chapters=[{"lap_start": 1, "lap_end": 70}]
        )
    (tmp_path / "predictions.json").write_text('[{"target_seconds": 40}]')
    with pytest.raises(ValueError, match="checksum"):
        export_case_report(tmp_path, freeze, tmp_path / "public.json")
