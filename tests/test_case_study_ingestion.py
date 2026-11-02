"""The dedicated final-race path must not weaken the development quarantine."""

import hashlib
import json
from unittest.mock import Mock

import pytest

from ant26.domain.protocol import load_protocol
from ant26.ingestion.case_study import ingest_case_study
from ant26.ingestion.client import ProviderError


def test_unverified_release_prevents_every_provider_query(tmp_path):
    client = Mock()
    with pytest.raises((ValueError, FileNotFoundError)):
        ingest_case_study(client, tmp_path / "missing-freeze.json", load_protocol())
    client.fetch.assert_not_called()


def test_wrong_session_identity_stops_before_telemetry(monkeypatch, tmp_path):
    monkeypatch.setattr("ant26.release.verify_release", lambda *args: {"status": "frozen"})
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    client = Mock()
    client.fetch.return_value = [{"session_key": 11404, "meeting_key": 999}]
    client.last_manifest = {"sha256": "a" * 64, "endpoint": "sessions"}
    with pytest.raises(ProviderError, match="identity"):
        ingest_case_study(client, freeze, load_protocol(), output_root=tmp_path / "case")
    assert client.fetch.call_count == 1


def test_case_materialization_preserves_missing_lap_and_source_hashes(monkeypatch, tmp_path):
    monkeypatch.setattr("ant26.release.verify_release", lambda *args: {"status": "frozen"})
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")

    class Client:
        def fetch(self, endpoint, params, **kwargs):
            if endpoint == "sessions":
                rows = [{"session_key": 11404, "meeting_key": 1298}]
            elif endpoint == "laps":
                rows = [
                    {
                        "session_key": 11404,
                        "meeting_key": 1298,
                        "driver_number": 12,
                        "lap_number": lap,
                        "date_start": f"2026-11-01T20:0{lap}:00+00:00",
                        "lap_duration": 60,
                        "duration_sector_1": 20,
                        "duration_sector_2": 20,
                        "duration_sector_3": 20,
                        "is_pit_out_lap": False,
                    }
                    for lap in [1, 3]
                ]
            elif endpoint == "car_data":
                rows = [
                    {
                        "session_key": 11404,
                        "driver_number": 12,
                        "date": "2026-11-01T20:01:01+00:00",
                        "speed": 200,
                    }
                ]
            else:
                rows = []
            self.last_manifest = {
                "endpoint": endpoint,
                "params": params,
                "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            }
            return rows

    result = ingest_case_study(Client(), freeze, load_protocol(), output_root=tmp_path / "case")
    payload = json.loads(open(result["path"]).read())
    assert payload["observed_laps"] == [1, 3]
    assert payload["expected_laps"] == 71
    assert len(payload["examples"]) == 6
    assert len(payload["source_manifests"]) == 9
    assert all(not row["eligible"] for row in payload["examples"])
    assert all("insufficient_history" in row["exclusion_reasons"] for row in payload["examples"])
