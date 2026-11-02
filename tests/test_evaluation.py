from copy import deepcopy

import pytest

from ant26.evaluation.metrics import macro_metrics, paired_race_bootstrap, weighted_interval_score
from ant26.evaluation.splits import rolling_splits


def catalog(count=12):
    return {
        "development": [
            {
                "session_key": 100 + i,
                "meeting_key": 200 + i,
                "date_start": f"2026-{i // 3 + 1:02d}-{i % 3 + 1:02d}T12:00:00+00:00",
                "session_name": "Race",
                "verified_identity": True,
            }
            for i in range(count)
        ]
    }


def test_hand_calculated_wis():
    assert weighted_interval_score(10, [8, 9, 10, 11, 12]) == pytest.approx(0.16)
    assert weighted_interval_score(13, [8, 9, 10, 11, 12]) == pytest.approx(1.96)
    with pytest.raises(ValueError):
        weighted_interval_score(10, [8, 10, 9, 11, 12])


def test_split_roles_and_weekend_grouping():
    data = catalog()
    first = data["development"][0]
    data["development"].append({**first, "session_key": 999, "session_name": "Practice 1"})
    folds = rolling_splits(data)
    assert len(folds) == 2
    fold = folds[0]
    assert len(fold["fit_meetings"]) == 6
    assert len(fold["tune_meetings"]) == 2
    assert len(fold["calibration_meetings"]) == 2
    assert fold["test_sessions"] == [110]
    assert 999 in fold["fit_sessions"]
    roles = [set(fold[f"{role}_meetings"]) for role in ("fit", "tune", "calibration", "test")]
    assert all(
        not left.intersection(right) for i, left in enumerate(roles) for right in roles[i + 1 :]
    )


def test_mexico_cancelled_and_future_excluded():
    data = catalog()
    for key, meeting, date, canceled in [
        (11404, 1298, "2026-11-01", False),
        (9999, 1298, "2026-10-30", False),
        (8888, 7777, "2026-12-01", False),
        (7777, 6666, "2026-01-01", True),
    ]:
        data["development"].append(
            {
                "session_key": key,
                "meeting_key": meeting,
                "date_start": date + "T12:00:00+00:00",
                "is_cancelled": canceled,
            }
        )
    assert len(rolling_splits(data)) == 2
    for fold in rolling_splits(data):
        for role in ("fit", "tune", "calibration", "test"):
            assert not set(fold[f"{role}_sessions"]).intersection({11404, 9999, 8888, 7777})


def test_macro_does_not_weight_large_race_more():
    one = {
        "session_key": 1,
        "driver_number": 12,
        "target_seconds": 10,
        "reference_seconds": 10,
        "quantiles": [8, 9, 10, 11, 12],
    }
    two = {**one, "session_key": 2, "target_seconds": 14}
    metrics = macro_metrics([one] * 100 + [two])
    assert metrics["macro"]["mae_seconds"] == 2
    assert metrics["forecast_count"] == 101


def test_bootstrap_resamples_paired_races_only():
    reference = {str(i): {"mae_seconds": 2} for i in range(6)}
    candidate = {str(i): {"mae_seconds": 1} for i in range(6)}
    result = paired_race_bootstrap(reference, candidate, resamples=50)
    assert result["relative_improvement"] == 0.5
    assert result["confidence_interval_95"] == [0.5, 0.5]
    assert result["unit"] == "complete_race"
    assert not result["exploratory"]
    bad = deepcopy(candidate)
    del bad["0"]
    with pytest.raises(ValueError):
        paired_race_bootstrap(reference, bad)
