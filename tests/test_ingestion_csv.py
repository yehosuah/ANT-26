import io
from pathlib import Path

import pytest

from ant26.ingestion import CacheIntegrityError, OpenF1Client, ProviderError

CSV = b"brake,date,driver_number,drs,meeting_key,n_gear,rpm,session_key,speed,throttle\r\n0,2026-10-25 20:00:00.100000+00:00,12,,1297,6,11000,11396,270.5,100\r\n"


def test_csv_preserves_source_and_normalizes_numeric_null_utc(tmp_path):
    client = OpenF1Client(
        tmp_path, opener=lambda *a, **kw: io.BytesIO(CSV), rate_limit_per_minute=0
    )
    rows = client.fetch("car_data", {"driver_number": 12}, response_format="csv")
    assert rows[0]["date"] == "2026-10-25T20:00:00.100000+00:00"
    assert rows[0]["drs"] is None
    assert rows[0]["speed"] == 270.5
    assert rows[0]["brake"] == 0
    assert rows[0]["session_key"] == 11396
    path = Path(client.last_manifest["snapshot_path"])
    assert (path / "response.csv").read_bytes() == CSV
    assert client.last_manifest["raw_response_sha256"] != client.last_manifest["sha256"]
    offline = OpenF1Client(tmp_path, offline=True)
    assert offline.fetch("car_data", {"driver_number": 12}, response_format="csv") == rows
    (path / "response.csv").write_bytes(b"corrupted")
    with pytest.raises(CacheIntegrityError, match="original CSV"):
        offline.fetch("car_data", {"driver_number": 12}, response_format="csv")


@pytest.mark.parametrize("payload", [b"error,detail\n429,throttle\n", CSV.replace(b"+00:00", b"")])
def test_invalid_csv_schema_or_naive_timestamp_is_not_cached(tmp_path, payload):
    client = OpenF1Client(
        tmp_path, opener=lambda *a, **kw: io.BytesIO(payload), rate_limit_per_minute=0
    )
    with pytest.raises(ProviderError, match="invalid telemetry CSV"):
        client.fetch("car_data", response_format="csv")
    assert not list(tmp_path.rglob("manifest.json"))
