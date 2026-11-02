"""Cross-layer release checks for artifact identity and honest fixture labeling."""

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from ant26.cli import main
from ant26.serving.api import create_app


def test_smoke_artifacts_match_saved_bytes_and_api_response(tmp_path):
    run = tmp_path / "smoke"
    assert main(["smoke", "--output", str(run)]) == 0
    report = json.loads((run / "report.json").read_text())
    assert report["synthetic"] is True
    assert report["data_kind"] == "synthetic_integration_fixture"
    assert report["status"] == "exploratory"
    predictions_path = Path(report["predictions_path"])
    payload = predictions_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest()[:16] in predictions_path.name
    assert json.loads(payload) == json.loads((run / "predictions.json").read_text())
    client = TestClient(create_app(tmp_path))
    served_report = client.get("/runs/smoke/report").json()
    assert served_report["input_sha256"] == report["input_sha256"]
    served_predictions = client.get("/runs/smoke/predictions?limit=5000").json()["predictions"]
    assert served_predictions == [row for row in json.loads(payload) if row["driver_number"] == 12]
    assert all(row["source_provenance"] for row in served_predictions)
    assert all(row["cutoff"] < row["origin"] for row in served_predictions)
