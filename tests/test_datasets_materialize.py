import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ant26.datasets.materialize import materialize_available_sessions, merge_materialized_cohorts
from ant26.domain.protocol import load_protocol


def save_snapshot(root, endpoint, rows):
    payload = json.dumps(rows).encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory = root / endpoint / digest
    directory.mkdir(parents=True)
    (directory / "response.json").write_bytes(payload)
    return {
        "endpoint": endpoint,
        "query_key": "fixture",
        "sha256": digest,
        "snapshot_path": str(directory),
        "row_count": len(rows),
    }


def inputs(tmp_path, driver_number=12):
    catalog = {
        "development": [
            {
                "session_key": 100,
                "meeting_key": 1,
                "date_start": "2026-03-08T04:00:00+00:00",
                "verified_identity": True,
            },
            {
                "session_key": 101,
                "meeting_key": 2,
                "date_start": "2026-03-15T04:00:00+00:00",
                "verified_identity": True,
            },
        ]
    }
    lap = {
        "session_key": 100,
        "driver_number": driver_number,
        "meeting_key": 1,
        "lap_number": 1,
        "date_start": "2026-03-08T04:00:00+00:00",
        "duration_sector_1": 10,
        "duration_sector_2": 10,
        "duration_sector_3": 10,
        "lap_duration": 30,
    }
    contexts = {
        name: save_snapshot(tmp_path / "raw", name, [lap] if name == "laps" else [])
        for name in ["laps", "stints", "intervals", "pit", "race_control", "weather"]
    }
    car = save_snapshot(
        tmp_path / "raw",
        "car_data",
        [
            {
                "session_key": 100,
                "driver_number": driver_number,
                "date": "2026-03-08T04:00:00+00:00",
                "speed": 200,
                "throttle": 90,
                "brake": 0,
                "rpm": 10000,
                "n_gear": 6,
            }
        ],
    )
    progress = {
        "sessions": {
            f"100:{driver_number}": {
                "status": "complete",
                "context_sources": contexts,
                "chunks": {"csv": {"manifest": car}},
            },
            f"101:{driver_number}": {"status": "running"},
        }
    }
    catalog_path, progress_path = tmp_path / "catalog.json", tmp_path / "progress.json"
    catalog_path.write_text(json.dumps(catalog))
    progress_path.write_text(json.dumps(progress))
    return catalog_path, progress_path


def test_materializer_pins_sources_reuses_sessions_and_reports_partial_cohort(tmp_path):
    catalog, progress = inputs(tmp_path)
    output = tmp_path / "processed"
    result = materialize_available_sessions(progress, catalog, load_protocol(), output_root=output)
    payload = json.loads(Path(result["path"]).read_text())
    assert payload["completed_sessions"] == [100]
    assert payload["cohort_driver_numbers"] == [12]
    assert payload["data_kind"] == "real_openf1_antonelli"
    assert payload["skipped"] == [
        {"session_key": 101, "driver_number": 12, "reason": "source_not_complete"}
    ]
    assert len(payload["examples"]) == 3
    assert (
        hashlib.sha256(Path(result["path"]).read_bytes()).hexdigest() == result["artifact_sha256"]
    )
    # A cache pointer changes independently, but exact immutable sources remain pinned.
    (tmp_path / "raw" / "latest.json").write_text("not a valid pointer")
    with patch(
        "ant26.datasets.materialize.build_examples", side_effect=AssertionError("must reuse")
    ):
        repeated = materialize_available_sessions(
            progress, catalog, load_protocol(), output_root=output
        )
    assert repeated["path"] == result["path"]


def test_materializer_detects_source_and_cached_artifact_corruption(tmp_path):
    catalog, progress = inputs(tmp_path)
    output = tmp_path / "processed"
    result = materialize_available_sessions(progress, catalog, load_protocol(), output_root=output)
    Path(result["session_artifacts"][0]).write_text("corrupt")
    with pytest.raises(ValueError, match="cached session.*checksum"):
        materialize_available_sessions(progress, catalog, load_protocol(), output_root=output)
    source = json.loads(progress.read_text())["sessions"]["100:12"]["context_sources"]["laps"]
    (Path(source["snapshot_path"]) / "response.json").write_text("[]")
    with pytest.raises(ValueError, match="source snapshot checksum"):
        materialize_available_sessions(
            progress, catalog, load_protocol(), output_root=tmp_path / "different"
        )


def test_materializer_rejects_mexico_before_transforming(tmp_path):
    catalog, progress = inputs(tmp_path)
    catalog.write_text(
        json.dumps(
            {
                "development": [
                    {
                        "session_key": 11404,
                        "meeting_key": 1298,
                        "date_start": "2026-11-01T20:00:00+00:00",
                        "verified_identity": True,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Mexico"):
        materialize_available_sessions(
            progress, catalog, load_protocol(), output_root=tmp_path / "output"
        )


def test_feature_code_change_invalidates_cached_examples(tmp_path):
    catalog, progress = inputs(tmp_path)
    output = tmp_path / "processed"
    protocol = load_protocol()
    first = materialize_available_sessions(progress, catalog, protocol, output_root=output)
    with patch("ant26.datasets.materialize._feature_code_sha256", return_value="a" * 64):
        second = materialize_available_sessions(progress, catalog, protocol, output_root=output)
    before, after = [json.loads(Path(result["path"]).read_text()) for result in [first, second]]
    assert first["path"] != second["path"]
    assert before["feature_builder_sha256"] != after["feature_builder_sha256"]
    assert before["examples"] == after["examples"]
    assert first["session_artifacts"] != second["session_artifacts"]


def test_merge_preserves_explicit_driver_cohorts_and_source_file_checksums(tmp_path):
    protocol = load_protocol()
    paths = []
    for driver in [12, 63]:
        catalog, progress = inputs(tmp_path / str(driver), driver_number=driver)
        result = materialize_available_sessions(
            progress,
            catalog,
            protocol,
            output_root=tmp_path / "processed" / str(driver),
            driver_number=driver,
        )
        paths.append(result["path"])
    result = merge_materialized_cohorts(paths, protocol, output_root=tmp_path / "merged")
    payload = json.loads(Path(result["path"]).read_text())
    assert payload["cohort_driver_numbers"] == [12, 63]
    assert payload["data_kind"] == "real_openf1_mercedes"
    assert payload["completed_session_driver_pairs"] == ["100:12", "100:63"]
    assert {row["driver_number"] for row in payload["examples"]} == {12, 63}
    for source in payload["source_datasets"]:
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    with pytest.raises(ValueError, match="overlap"):
        merge_materialized_cohorts([paths[0], paths[0]], protocol, output_root=tmp_path / "bad")


def test_unscorable_source_is_explicitly_justified_in_skip_record(tmp_path):
    catalog, progress = inputs(tmp_path)
    state = json.loads(progress.read_text())
    state["sessions"]["101:12"] = {
        "status": "unscorable",
        "reason": "no_valid_timed_laps",
        "context_sources": {"laps": state["sessions"]["100:12"]["context_sources"]["laps"]},
    }
    progress.write_text(json.dumps(state))
    result = materialize_available_sessions(
        progress, catalog, load_protocol(), output_root=tmp_path / "processed"
    )
    assert result["skipped"][0]["reason"] == "no_valid_timed_laps"
    assert result["skipped"][0]["source_manifests"][0]["endpoint"] == "laps"
