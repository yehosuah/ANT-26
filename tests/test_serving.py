import json

from fastapi.testclient import TestClient

from ant26.serving.api import create_app


def test_api_exposes_only_existing_artifacts_and_filters_driver(tmp_path):
    run = tmp_path / "smoke"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"synthetic": True}))
    (run / "predictions.json").write_text(
        json.dumps(
            [
                {"driver_number": 12, "session_key": 1, "quantiles": [1, 2, 3, 4, 5]},
                {"driver_number": 63, "session_key": 1, "quantiles": [1, 2, 3, 4, 5]},
            ]
        )
    )
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/runs").json() == {"runs": ["smoke"]}
    assert client.get("/runs/smoke/report").json()["synthetic"]
    result = client.get("/runs/smoke/predictions").json()
    assert result["count"] == 1
    assert result["predictions"][0]["driver_number"] == 12
    assert client.get("/runs/smoke/atlas").status_code == 404
    assert client.get("/runs/missing/report").status_code == 404


def test_artifact_directory_symlinks_cannot_escape_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    other = tmp_path / "outside"
    other.mkdir()
    (other / "report.json").write_text('{"private": true}')
    (root / "escape").symlink_to(other, target_is_directory=True)
    client = TestClient(create_app(root))
    assert client.get("/runs/escape/report").status_code == 404


def test_prediction_index_and_pagination_preserve_late_races(tmp_path):
    run = tmp_path / "large"
    run.mkdir()
    rows = [{"driver_number": 12, "session_key": 1, "model": "reference"} for _ in range(5001)]
    rows.append({"driver_number": 12, "session_key": 2, "model": "ridge"})
    (run / "predictions.json").write_text(json.dumps(rows))
    client = TestClient(create_app(tmp_path))
    index = client.get("/runs/large/prediction-index").json()["entries"]
    assert {entry["session_key"] for entry in index} == {1, 2}
    page = client.get("/runs/large/predictions?offset=5000&limit=1").json()
    assert page["count"] == 5002
    assert len(page["predictions"]) == 1
    assert client.get("/runs/large/predictions?session_key=2&model=ridge").json()["count"] == 1


def test_invalidated_experiments_are_quarantined_from_replay(tmp_path):
    run = tmp_path / "superseded"
    run.mkdir()
    (run / "report.json").write_text('{"status":"evaluated"}')
    (run / "normalization-defect.json").write_text('{"invalidated":true}')
    client = TestClient(create_app(tmp_path))
    assert client.get("/runs").json()["runs"] == []
    assert client.get("/runs/superseded/report").status_code == 410
