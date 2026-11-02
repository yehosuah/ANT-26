from copy import deepcopy

import numpy as np
import pytest

from ant26.evaluation.benchmark import run_benchmark
from ant26.models.baselines import Baseline, feature_matrix, train_baselines, valid_examples


def rows(session, count=20):
    return [
        {
            "example_id": f"{session}-{i}",
            "session_key": session,
            "driver_number": 12,
            "lap_number": i + 3,
            "sector_index": 1,
            "origin": "2026-01-01T12:00:00+00:00",
            "cutoff": "2026-01-01T11:59:59+00:00",
            "target_seconds": 30 + (i % 5) / 10,
            "reference_seconds": 30,
            "features": {"last_same_sector_seconds": 30, "speed_mean": 150 + i},
            "eligible": True,
            "exclusion_reasons": [],
            "provenance": {},
        }
        for i in range(count)
    ]


def test_numeric_target_aliases_are_not_features():
    data = rows(1)
    injected = deepcopy(data)
    for row in injected:
        row["features"].update(target_seconds=99999, lap_duration=-123, duration_sector_1=888)
    np.testing.assert_equal(feature_matrix(data), feature_matrix(injected))


def test_origin_eligibility_does_not_make_censored_targets_scorable():
    data = rows(1, 2)
    data[0]["target_valid"] = False
    data[0]["target_exclusion_reasons"] = ["red_flag_overlap"]
    assert valid_examples(data) == [data[1]]


@pytest.mark.parametrize("field", ["origin", "cutoff"])
def test_future_unknown_session_is_rejected_before_baseline_fit(field):
    data = rows(9999)
    data[0][field] = "2026-10-30T00:00:00+00:00"
    with pytest.raises(ValueError, match="later timestamps"):
        Baseline("historical_reference").fit(data, allowed_sessions={9999})


def test_scope_rejects_mexico_and_calibration_overlap():
    with pytest.raises(ValueError, match="Mexico"):
        Baseline("historical_reference").fit(rows(11404), allowed_sessions={11404})
    unknown_mexico = rows(1)
    unknown_mexico[0]["provenance"]["meeting_key"] = 1298
    with pytest.raises(ValueError, match="Mexico"):
        Baseline("historical_reference").fit(unknown_mexico, allowed_sessions={1})
    model = Baseline("historical_reference").fit(rows(1), allowed_sessions={1})
    with pytest.raises(ValueError, match="disjoint"):
        model.calibrate(rows(1), allowed_sessions={1})
    with pytest.raises(ValueError, match="Unexpected session"):
        model.calibrate(rows(2), allowed_sessions={3})


def test_models_calibrate_predict_unknown_context_without_mutation():
    train, cal = rows(1), rows(2)
    original = deepcopy(train)
    models = train_baselines(train, cal, fit_sessions={1}, calibration_sessions={2})
    test = rows(3)
    for row in test:
        row["features"]["circuit_key"] = "never_seen"
    for model in models.values():
        predictions = np.array(model.predict(test))
        assert predictions.shape == (20, 5)
        assert np.all(np.diff(predictions, axis=1) >= 0)
        assert np.isfinite(predictions).all()
        assert model.fit_sessions == {1}
        assert model.calibration_sessions == {2}
    assert train == original


def test_offline_benchmark_records_skips_and_immutable_artifacts(tmp_path):
    catalog = {
        "data_kind": "synthetic_test_fixture",
        "development": [
            {
                "session_key": i,
                "meeting_key": i + 100,
                "verified_identity": True,
                "date_start": f"2026-{(i - 1) // 3 + 1:02d}-{(i - 1) % 3 + 1:02d}T12:00:00+00:00",
            }
            for i in range(1, 13)
        ],
    }
    examples = [row for i in range(1, 12) for row in rows(i)]
    report = run_benchmark(examples, catalog, output_dir=tmp_path)
    assert report["status"] == "exploratory"
    assert report["evaluated_fold_count"] == 1
    assert report["folds"][1]["status"] == "skipped"
    assert report["summary"]["historical_reference"]["antonelli"]["forecast_count"] == 20
    assert (
        run_benchmark(examples, catalog, output_dir=tmp_path)["report_path"]
        == report["report_path"]
    )
    with pytest.raises(ValueError, match="Mexico"):
        run_benchmark(examples + rows(11404), catalog)
