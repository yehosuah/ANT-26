import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "review.py"


def blind_fixture(tmp_path):
    def window(identifier, lap):
        return {
            "example_id": identifier,
            "session_key": 100,
            "driver_number": 12,
            "lap_number": lap,
            "sector_index": 1,
            "origin": "2026-01-01T00:01:01+00:00",
            "cutoff": "2026-01-01T00:01:00+00:00",
            "sequence_channels": ["speed", "throttle", "brake"],
            "sequence": [[100, 80, 0], [90, 0, 1]],
            "sequence_mask": [[True, True, True], [True, True, True]],
            "context": {"compound": "SOFT", "track_state": "GREEN"},
        }

    neighbor = {
        "rank": 1,
        "observed": window("reference", 1),
        "observed_summary": {"speed_mean": 95},
    }
    payload = {
        "schema_version": "ant26-blind-atlas-review-v1",
        "data_kinds": ["synthetic_integration_fixture"],
        "queries": [
            {
                "query_id": "query-001",
                "query": window("query", 3),
                "candidates": {"A": [neighbor], "B": [neighbor]},
            }
        ],
    }
    payload["review_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    path = tmp_path / "blind-review.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_review_app_writes_no_implicit_rating_and_saves_only_on_click(tmp_path, monkeypatch):
    path, payload = blind_fixture(tmp_path)
    answers = tmp_path / "answers"
    monkeypatch.setenv("ANT26_REVIEW_PATH", str(path))
    monkeypatch.setenv("ANT26_REVIEW_ANSWER_ROOT", str(answers))
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    assert app.radio[0].value == "Unrated"
    assert len(app.get("vega_lite_chart")) + len(app.get("arrow_vega_lite_chart")) == 6
    assert not answers.exists()
    app.text_input[0].set_value("fixture-reviewer").run()
    app.radio[0].set_value("A").run()
    assert not answers.exists()
    app.button[0].click().run()
    answer = json.loads((answers / payload["review_id"] / "fixture-reviewer.json").read_text())
    assert answer["responses"] == [{"query_id": "query-001", "choice": "A", "notes": ""}]
    assert not app.exception
    app.text_input[0].set_value("another-reviewer").run()
    assert app.radio[0].value == "Unrated"
    assert not (answers / payload["review_id"] / "another-reviewer.json").exists()


def test_review_app_rejects_private_key_without_reading_it(tmp_path, monkeypatch):
    path = tmp_path / "private" / "answer-key.json"
    monkeypatch.setenv("ANT26_REVIEW_PATH", str(path))
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    assert len(app.error) == 1
    assert "private assignment key" in app.error[0].value
    assert not app.radio
