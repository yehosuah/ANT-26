import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from ant26.ingestion import (
    CacheIntegrityError,
    OpenF1Client,
    ProviderError,
    audit_rows,
    audit_session,
    build_catalog,
    expand_full_traces,
)


def fake_response(payload):
    return lambda *_args, **_kwargs: io.BytesIO(json.dumps(payload).encode())


def test_cache_is_immutable_and_offline_read_is_verified(tmp_path):
    client = OpenF1Client(tmp_path, opener=fake_response([{"a": 1}]), rate_limit_per_minute=0)
    assert client.fetch("laps", {"session_key": 12}) == [{"a": 1}]
    original = Path(client.last_manifest["snapshot_path"])
    manifest = (original / "manifest.json").read_bytes()
    client._opener = fake_response([{"a": 2}])
    assert client.fetch("laps", {"session_key": 12}, refresh=True) == [{"a": 2}]
    assert (original / "manifest.json").read_bytes() == manifest
    assert json.loads((original / "response.json").read_text()) == [{"a": 1}]
    offline = OpenF1Client(tmp_path, offline=True)
    assert offline.fetch("laps", {"session_key": 12}) == [{"a": 2}]
    assert offline.last_manifest["cache_hit"]
    latest = Path(offline.last_manifest["snapshot_path"])
    (latest / "response.json").write_text("[]")
    with pytest.raises(CacheIntegrityError):
        offline.fetch("laps", {"session_key": 12})


def test_missing_offline_snapshot_and_invalid_payload_fail(tmp_path):
    with pytest.raises(ProviderError, match="offline cache miss"):
        OpenF1Client(tmp_path, offline=True).fetch("laps")
    client = OpenF1Client(
        tmp_path, opener=fake_response({"error": "limit"}), rate_limit_per_minute=0
    )
    with pytest.raises(ProviderError, match="list of objects"):
        client.fetch("laps")
    assert not list(tmp_path.glob("**/manifest.json"))


def test_retries_transient_error_without_caching_error(tmp_path):
    calls = []
    sleeps = []

    def opener(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError("url", 429, "slow down", {"Retry-After": "2"}, None)
        return io.BytesIO(b"[]")

    client = OpenF1Client(tmp_path, opener=opener, sleeper=sleeps.append, rate_limit_per_minute=0)
    assert client.fetch("laps") == []
    assert len(calls) == 2
    assert sleeps == [2]
    assert len(list(tmp_path.glob("**/manifest.json"))) == 1


def test_authentication_error_is_not_retried(tmp_path):
    def opener(*args, **kwargs):
        raise HTTPError("url", 401, "auth", {}, None)

    with pytest.raises(ProviderError, match="401"):
        OpenF1Client(tmp_path, opener=opener, rate_limit_per_minute=0).fetch("laps")


def session(key, meeting, date="2026-10-04T07:00:00+00:00", **extra):
    return {
        "session_key": key,
        "meeting_key": meeting,
        "session_name": "Race",
        "year": 2026,
        "date_start": date,
        "country_name": "Bahrain",
        "circuit_short_name": "Kuala Lumpur",
        **extra,
    }


def test_catalog_quarantine_override_and_final_meeting():
    rows = [
        session(11731, 1308),
        session(11404, 1298),
        session(1, 1, is_cancelled=True),
        session(2, 2, date="2026-12-01T00:00:00+00:00"),
    ]
    catalog = build_catalog(rows)
    assert not catalog["development"]
    assert [row["session_key"] for row in catalog["final_case_study"]] == [11404]
    reasons = {row["session_key"]: row["exclusion_reasons"] for row in catalog["excluded"]}
    assert reasons[11731] == ["unresolved_event_identity"]
    assert reasons[1] == ["cancelled"]
    assert reasons[2] == ["future_session"]
    overrides = {
        "sessions": {
            "11731": {
                "verified_identity": True,
                "sources": ["official"],
                "host_country": "Malaysia",
                "circuit": "Sepang",
            }
        }
    }
    resolved = build_catalog(rows, overrides=overrides)
    assert resolved["development"][0]["canonical_host_country"] == "Malaysia"
    assert rows[0]["country_name"] == "Bahrain"
    assert "canonical_host_country" not in rows[0]


def test_session_meeting_mismatch_and_naive_time_quarantined():
    rows = [session(3, 3, circuit_key=20), session(4, 4, date="2026-10-04T07:00:00")]
    catalog = build_catalog(rows, [{"meeting_key": 3, "circuit_key": 21}])
    assert not catalog["development"]
    assert set(reason for row in catalog["excluded"] for reason in row["exclusion_reasons"]) == {
        "session_meeting_circuit_mismatch",
        "invalid_date_start",
    }


def test_audit_reports_gaps_missing_laps_and_inconsistent_sums():
    laps = [
        {
            "lap_number": 1,
            "lap_duration": 90.0,
            "duration_sector_1": 30.0,
            "duration_sector_2": 30.0,
            "duration_sector_3": 30.0,
        },
        {
            "lap_number": 3,
            "lap_duration": 90.0,
            "duration_sector_1": 30.0,
            "duration_sector_2": 30.0,
            "duration_sector_3": 31.0,
        },
    ]
    report = audit_rows("laps", laps)
    assert report["internal_missing_laps"] == [2]
    assert report["sector_sum_mismatch_laps"] == [3]
    report = audit_rows(
        "car_data",
        [
            {"date": "2026-01-01T00:00:00Z", "speed": 50},
            {"date": "2026-01-01T00:00:02Z", "speed": None},
        ],
    )
    assert report["max_gap_seconds"] == 2
    assert report["null_counts"]["speed"] == 1


def test_mexico_development_ingestion_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match="quarantined"):
        audit_session(OpenF1Client(tmp_path, offline=True), session(11404, 1298))


def test_path_traversal_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match="endpoint"):
        OpenF1Client(tmp_path).fetch("../../laps")


def test_bounded_time_ranges_are_half_open_and_keep_manifests(tmp_path):
    client = OpenF1Client(tmp_path, rate_limit_per_minute=0)
    queries = []

    def fetch(endpoint, params):
        queries.append(params)
        date = "2026-10-04T07:00:00+00:00" if len(queries) == 1 else "2026-10-04T07:01:00+00:00"
        client.last_manifest = {"query": params}
        return [{"date": date}]

    client.fetch = fetch
    rows = client.fetch_time_range("car_data", {}, "2026-10-04T07:00:00Z", "2026-10-04T07:02:00Z")
    assert len(rows) == len(client.last_window_manifests) == 2
    assert queries[0]["date>"] == "2026-10-04T06:59:59.999999"
    assert queries[0]["date<"] == "2026-10-04T07:01:00"
    assert queries[1]["date>"] == "2026-10-04T07:00:59.999999"
    with pytest.raises(ValueError, match="timezone-aware"):
        client.fetch_time_range("car_data", {}, "2026-10-04T07:00:00", "2026-10-04T07:01:00Z")


def test_time_range_rejects_ignored_provider_filters(tmp_path):
    client = OpenF1Client(tmp_path, rate_limit_per_minute=0)
    client.fetch = lambda *args: [{"date": "2026-10-04T08:00:00Z"}]
    with pytest.raises(ProviderError, match="outside"):
        client.fetch_time_range("car_data", {}, "2026-10-04T07:00:00Z", "2026-10-04T07:01:00Z")


def test_expansion_checkpoints_partial_failure_and_recovers(tmp_path):
    class Client:
        fail = True
        last_window_manifests = []

        def fetch(self, endpoint, params):
            return [
                {"meeting_key": 1297, "date_start": "2026-10-25T20:00:00Z", "lap_duration": 120}
            ]

        def fetch_time_range(self, endpoint, params, start, end, **kwargs):
            if start.minute == 1 and self.fail:
                raise ProviderError("temporary")
            self.last_window_manifests = [{"sha256": str(start), "row_count": 1}]
            return [{"date": start.isoformat()}]

    client = Client()
    path = tmp_path / "progress.json"
    first = expand_full_traces(
        client, session_keys=(11396,), driver_numbers=(12,), progress_path=path
    )
    assert first["sessions"]["11396:12"]["status"] == "partial"
    assert len(first["sessions"]["11396:12"]["failed_bounds"]) == 1
    assert json.loads(path.read_text())["sessions"]["11396:12"]["completed_chunks"] == 1
    client.fail = False
    second = expand_full_traces(
        client, session_keys=(11396,), driver_numbers=(12,), progress_path=path
    )
    assert second["sessions"]["11396:12"]["status"] == "complete"
    assert second["sessions"]["11396:12"]["completed_chunks"] == 2
    assert second["sessions"]["11396:12"]["rows"] == 2
    with pytest.raises(ValueError, match="quarantined"):
        expand_full_traces(client, session_keys=(11404,), progress_path=path)
