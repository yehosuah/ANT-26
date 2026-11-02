import json

import pytest

from ant26.ingestion import ingest_development


class Client:
    last_manifest = None
    bad_filter = False

    def fetch(self, endpoint, params, **kwargs):
        self.last_manifest = {"endpoint": endpoint, "params": params, "sha256": "example"}
        if endpoint == "laps":
            return [{"date_start": "2026-10-25T20:00:00Z", "lap_duration": 60}]
        if endpoint == "car_data":
            assert kwargs["response_format"] == "csv"
            return [
                {
                    "date": "2026-10-25T20:00:00.100000+00:00",
                    "session_key": params["session_key"],
                    "driver_number": 63 if self.bad_filter else params["driver_number"],
                }
            ]
        return []


CATALOG = {"development": [{"session_key": 11396, "meeting_key": 1297}]}


def test_bulk_emits_compatible_progress_and_all_context_sources(tmp_path):
    path = tmp_path / "bulk.json"
    progress = ingest_development(Client(), CATALOG, progress_path=path)
    entry = progress["sessions"]["11396:12"]
    assert entry["status"] == "complete"
    assert entry["rows"] == 1
    assert len(entry["context_sources"]) == 7
    assert entry["chunks"]["full_race_csv"]["manifest"]["endpoint"] == "car_data"
    assert json.loads(path.read_text())["data_kind"] == "real_openf1"
    assert not progress["mexico_ingested"]


def test_bulk_rejects_ignored_filters_and_mexico(tmp_path):
    client = Client()
    client.bad_filter = True
    progress = ingest_development(client, CATALOG, progress_path=tmp_path / "bad.json")
    entry = progress["sessions"]["11396:12"]
    assert entry["status"] == "partial"
    assert "identity filters" in entry["failed_bounds"][0]["error"]
    with pytest.raises(ValueError, match="Mexico"):
        ingest_development(client, {"development": [{"session_key": 11404, "meeting_key": 1298}]})


def test_zero_timed_laps_is_visible_unscorable_status(tmp_path):
    class RetiredClient(Client):
        def fetch(self, endpoint, params, **kwargs):
            if endpoint == "laps":
                self.last_manifest = {"endpoint": "laps", "params": params}
                return [{"date_start": "2026-10-25T20:00:00Z", "lap_duration": None}]
            assert endpoint != "car_data"
            return super().fetch(endpoint, params, **kwargs)

    progress = ingest_development(RetiredClient(), CATALOG, progress_path=tmp_path / "retired.json")
    entry = progress["sessions"]["11396:12"]
    assert entry["status"] == "unscorable"
    assert entry["reason"] == "no_valid_timed_laps"
    assert entry["rows"] == 0
