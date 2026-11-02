import json

from fastapi.testclient import TestClient

from ant26.cli import main
from ant26.serving.api import create_app


def test_offline_pipeline_reaches_read_only_api(tmp_path):
    output = tmp_path / "smoke"
    assert main(["smoke", "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["synthetic"] is True
    assert report["evaluated_fold_count"] == 2
    assert report["status"] == "exploratory"
    assert report["predictions"]
    client = TestClient(create_app(tmp_path))
    response = client.get("/runs/smoke/predictions?driver_number=12")
    assert response.status_code == 200
    assert response.json()["count"] > 0
    assert main(["smoke", "--output", str(output)]) == 0
