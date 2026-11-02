from copy import deepcopy

import pytest

from ant26.evaluation.diagnostics import error_slices


def example(identity, session=1, **changes):
    row = {
        "example_id": identity,
        "session_key": session,
        "driver_number": 12,
        "lap_number": 4,
        "sector_index": 1,
        "target_seconds": 10,
        "reference_seconds": 10,
        "eligible": True,
        "target_valid": True,
        "uninterrupted_green_target": True,
        "exclusion_reasons": [],
        "features": {
            "tyre_age_laps": 4,
            "interval": 0.5,
            "telemetry_coverage": 1.0,
            "context": {"track_state": "GREEN", "in_pit_lane": False},
        },
    }
    return row | changes


def prediction(identity, median=10, model="baseline", **changes):
    return {
        "example_id": identity,
        "model": model,
        "quantiles": [median - 2, median - 1, median, median + 1, median + 2],
    } | changes


def test_hand_computed_macro_slices_and_coverage_scope():
    rows = [
        example("a"),
        example("b"),
        example("c", session=2, target_seconds=14),
        example("training", session=99, eligible=False, exclusion_reasons=["cold_start"]),
        example("abstain", eligible=False, exclusion_reasons=["cold_start", "poor_coverage"]),
    ]
    result = error_slices([prediction("a"), prediction("b"), prediction("c")], rows)
    model = result["models"]["baseline"]
    assert model["overall"]["macro"]["mae_seconds"] == 2
    assert model["overall"]["forecast_count"] == 3
    assert model["overall"]["unique_example_count"] == 3
    assert model["overall"]["race_count"] == 2
    assert model["overall"]["exploratory"]
    assert model["slices"]["sector"]["1"]["macro"]["mae_seconds"] == 2
    assert model["slices"]["tyre_age_laps"]["[0,6)"]["unique_example_count"] == 3
    assert model["slices"]["traffic_interval_seconds"]["(0,1]"]["unique_example_count"] == 3
    assert result["abstentions"]["reason_counts"] == {"cold_start": 1, "poor_coverage": 1}
    assert result["coverage"]["examples_outside_evaluation_scope_count"] == 1
    assert result["coverage"]["origin_abstention_count"] == 1
    assert result["coverage"]["evaluation_session_keys"] == [1, 2]
    assert result["coverage"]["nominal_green_opportunities"] == 4
    assert result["coverage"]["nominal_green_retention"] == 0.75


def test_target_condition_only_changes_realized_diagnostic_and_does_not_mutate_inputs():
    rows = [example("a"), example("b", target_seconds=20, uninterrupted_green_target=False)]
    predictions = [prediction("a"), prediction("b")]
    before = deepcopy((rows, predictions))
    first = error_slices(predictions, rows)
    changed = deepcopy(rows)
    changed[1]["uninterrupted_green_target"] = True
    second = error_slices(predictions, changed)
    assert first["coverage"] == second["coverage"]
    assert first["abstentions"] == second["abstentions"]
    assert first["models"]["baseline"]["overall"] == second["models"]["baseline"]["overall"]
    assert (
        first["models"]["baseline"]["slices"]["realized_target_state"]["not_uninterrupted_green"][
            "unique_example_count"
        ]
        == 1
    )
    assert (
        first["models"]["baseline"]["slices"]["realized_target_state"]["uninterrupted_green"][
            "unique_example_count"
        ]
        == 1
    )
    assert (rows, predictions) == before


@pytest.mark.parametrize(
    "bad",
    [
        prediction("foreign"),
        prediction("a", session_key=99),
        prediction("a", driver_number=63),
        prediction("a", target_seconds=99),
    ],
)
def test_foreign_or_mismatched_predictions_rejected(bad):
    with pytest.raises(ValueError):
        error_slices([bad], [example("a")])


def test_unknown_values_explicit_and_zero_interval_not_labeled_close_traffic():
    rows = [
        example(
            "a",
            sector_index=None,
            features={"interval": 0, "telemetry_coverage": float("nan"), "tyre_age_laps": -1},
        )
    ]
    result = error_slices([prediction("a")], rows)["models"]["baseline"]["slices"]
    for dimension in ["sector", "tyre_age_laps", "traffic_interval_seconds", "telemetry_coverage"]:
        assert result[dimension]["unknown"]["unique_example_count"] == 1


def test_seeds_not_counted_as_independent_forecasts_or_races():
    rows = [example("a"), example("b", session=2)]
    predictions = [
        prediction(identity, median, model="encoder", seed=seed)
        for identity in ["a", "b"]
        for seed, median in [(17, 10), (29, 12)]
    ]
    model = error_slices(predictions, rows)["models"]["encoder"]
    assert model["overall"]["race_count"] == 2
    assert model["overall"]["unique_example_count"] == 2
    assert model["overall"]["prediction_count"] == 4
    assert model["overall"]["seed_count"] == 2
    assert model["overall"]["macro"]["mae_seconds"] == 1
    assert model["per_seed"]["17"]["macro"]["mae_seconds"] == 0
    assert model["per_seed"]["29"]["macro"]["mae_seconds"] == 2
    with pytest.raises(ValueError, match="identical scoreable"):
        error_slices(predictions[:-1], rows)


def test_duplicate_records_rejected_and_model_missing_predictions_visible():
    rows = [example("a"), example("b")]
    with pytest.raises(ValueError, match="Duplicate prediction"):
        error_slices([prediction("a"), prediction("a")], rows)
    result = error_slices([prediction("a")], rows)
    assert result["models"]["baseline"]["missing_scoreable_predictions"] == 1
    with pytest.raises(ValueError, match="unique"):
        error_slices([], [example("a"), example("a")])


def test_censored_target_kept_separate_from_origin_abstention():
    rows = [
        example("a"),
        example("censored", target_valid=False, target_seconds=None),
        example("abstain", eligible=False, exclusion_reasons=["poor_coverage"]),
    ]
    result = error_slices([prediction("a"), prediction("censored")], rows)
    assert result["models"]["baseline"]["overall"]["unique_example_count"] == 1
    assert result["models"]["baseline"]["censored_prediction_count"] == 1
    assert result["coverage"]["eligible_but_unscoreable_count"] == 1
    assert result["abstentions"]["example_count"] == 1
    with pytest.raises(ValueError, match="ineligible"):
        error_slices([prediction("abstain")], rows)


def test_no_predictions_does_not_invent_evaluation_scope():
    rows = [example("a", eligible=False)]
    assert error_slices([], rows)["coverage"]["scope_status"] == "no_prediction_scope"
    explicit = error_slices([], rows, evaluation_sessions=[1])
    assert explicit["coverage"]["scope_status"] == "explicit_sessions"
    assert explicit["abstentions"]["example_count"] == 1
