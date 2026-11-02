"""Race-level matching and uncertainty tests with hand-calculated contrasts."""

import copy
import json

import numpy as np
import pytest

from ant26.research.comparison import matched_comparison, race_bootstrap, run_behavior_comparison


def observation(
    session,
    driver,
    *,
    lap=4,
    sector=1,
    full=0.5,
    brake=0.2,
    coast=0.1,
    age=4,
    progress=0.2,
    coverage=0.98,
    gap=3,
    compound="MEDIUM",
):
    return {
        "session_key": session,
        "driver_number": driver,
        "lap_number": lap,
        "sector_index": sector,
        "origin": "2026-01-01T00:00:00Z",
        "compound": compound,
        "tyre_age_laps": age,
        "lap_fraction": progress,
        "interval_gap": gap,
        "valid_coverage": coverage,
        "rainfall": 0,
        "eligible": True,
        "exclusion_reasons": [],
        "full_throttle_fraction": full,
        "brake_on_fraction": brake,
        "coast_fraction": coast,
    }


def paired(session, **kwargs):
    return [
        observation(session, 12, full=0.6, **kwargs),
        observation(session, 63, full=0.4, **kwargs),
    ]


def test_equal_cells_then_equal_races_not_native_samples_or_rows():
    # Race1 cell1 contrast.2 (unequal row count), cell2 contrast.6 =>race1.4.
    # Race2 contrast-.2 =>equal-race mean.1, not equal-cell mean.2.
    rows = paired(100)
    rows += [observation(100, 12, lap=5, full=0.6)]
    rows += [observation(100, 12, sector=2, full=0.8), observation(100, 63, sector=2, full=0.2)]
    rows += [observation(101, 12, full=0.3), observation(101, 63, full=0.5)]
    result = matched_comparison(rows, resamples=100)
    assert result["matched_cell_count"] == 3
    assert result["race_count"] == 2
    assert result["metrics"]["full_throttle_fraction"][
        "difference_percentage_points"
    ] == pytest.approx(10)
    assert result["per_race"][0]["difference"]["full_throttle_fraction"] == pytest.approx(0.4)
    assert result["coverage"]["matched_observations"] == 7
    assert result["metrics"]["full_throttle_fraction"]["race_sign_counts"] == {
        "positive": 1,
        "negative": 1,
        "zero": 0,
    }
    loo = result["metrics"]["full_throttle_fraction"]["leave_one_race_out"]
    assert loo["all_same_sign_as_full_effect"] is False
    assert loo["minimum_percentage_points"] == pytest.approx(-20)
    assert loo["maximum_percentage_points"] == pytest.approx(40)


def test_exact_context_bins_unmatched_losses_and_endpoint_boundaries():
    rows = [observation(100, 12, age=3, progress=0.3), observation(100, 63, age=5, progress=0.399)]
    rows += [observation(100, 12, lap=6, age=6, progress=0.3)]
    rows += [observation(100, 63, lap=7, compound="HARD", age=3, progress=0.3)]
    result = matched_comparison(rows, resamples=100)
    assert result["matched_cell_count"] == 1
    assert result["coverage"]["unmatched_retained_observations"] == 2
    assert result["matched_cells"][0]["lap_fraction_bin_index"] == 3
    endpoint = matched_comparison(paired(100, progress=1), resamples=100)
    assert endpoint["matched_cells"][0]["lap_fraction_bin_index"] == 9
    tight = matched_comparison(rows, tyre_age_bin=1, resamples=100)
    assert tight["matched_cell_count"] == 0


def test_whole_race_bootstrap_reproducible_and_bonferroni_contains_ordinary():
    values = [[-0.1, 0.02, -0.03], [0.2, -0.01, 0.04], [0.5, 0.03, 0.08], [-0.4, -0.05, -0.01]]
    first = race_bootstrap(values, seed=17, resamples=10000)
    assert first == race_bootstrap(values, seed=17, resamples=10000)
    for result in first.values():
        normal = result["ci95_percentage_points"]
        family = result["bonferroni98_333_percentage_points"]
        assert family[0] <= normal[0] <= normal[1] <= family[1]
    # There are only four race blocks, despite many possible sector samples.
    assert first["full_throttle_fraction"]["ci95_percentage_points"][0] < 0
    assert first["full_throttle_fraction"]["ci95_percentage_points"][1] > 0
    one = race_bootstrap([[0.1, 0.2, 0.3]], resamples=100)
    assert all(result["ci95_percentage_points"] is None for result in one.values())


def test_known_null_and_driver_swap_are_reported_honestly():
    rows = [row for session in range(100, 107) for row in paired(session)]
    report = run_behavior_comparison(rows, resamples=100)
    assert report["findings"]["full_throttle_fraction"]["direction_claim_allowed"] is True
    assert report["findings"]["brake_fraction"]["direction_claim_allowed"] is False
    assert report["findings"]["coast_fraction"]["evidence_level"] == "no_clear_difference"
    original = report["primary"]["metrics"]["full_throttle_fraction"][
        "difference_percentage_points"
    ]
    swapped = copy.deepcopy(rows)
    for row in swapped:
        row["driver_number"] = 63 if row["driver_number"] == 12 else 12
    reversed_report = run_behavior_comparison(swapped, resamples=100)
    assert reversed_report["primary"]["metrics"]["full_throttle_fraction"][
        "difference_percentage_points"
    ] == pytest.approx(-original)


def test_quality_traffic_and_weather_losses_are_visible():
    rows = paired(100) + paired(101)
    rows[0]["valid_coverage"] = 0.94
    rows[2]["interval_gap"] = None
    result = run_behavior_comparison(rows, resamples=100)
    quality = result["sensitivities"]["coverage_095"]
    traffic = result["sensitivities"]["traffic_gap_2s"]
    assert quality["coverage"]["exclusion_reason_counts"]["insufficient_native_coverage"] == 1
    assert (
        traffic["coverage"]["exclusion_reason_counts"]["traffic_restricted_or_missing_interval"]
        == 1
    )
    assert quality["race_count"] == traffic["race_count"] == 1
    assert (
        result["findings"]["full_throttle_fraction"]["evidence_level"]
        == "qualified_observational_signal"
    )
    assert (
        result["findings"]["full_throttle_fraction"]["stable_driving_style_claim_allowed"] is False
    )
    wet = copy.deepcopy(rows)
    for row in wet:
        row["rainfall"] = 1
    assert (
        run_behavior_comparison(wet, resamples=100)["sensitivities"]["dry_only"]["race_count"] == 0
    )


def test_no_common_support_has_counts_without_invented_estimates():
    rows = [observation(100, 12), observation(101, 63)]
    result = run_behavior_comparison(rows, resamples=100)
    assert result["primary"]["race_count"] == 0
    assert result["primary"]["coverage"]["unmatched_retained_observations"] == 2
    assert (
        result["primary"]["metrics"]["full_throttle_fraction"]["difference_percentage_points"]
        is None
    )
    assert result["findings"]["full_throttle_fraction"]["direction_claim_allowed"] is False


def test_mexico_and_duplicate_observations_cannot_enter_matching():
    with pytest.raises(ValueError, match="Mexico"):
        run_behavior_comparison(paired(11404), resamples=100)
    rows = paired(100)
    rows[0]["meeting_key"] = 1298
    with pytest.raises(ValueError, match="Mexico"):
        run_behavior_comparison(rows, resamples=100)
    with pytest.raises(ValueError, match="Duplicate"):
        run_behavior_comparison(paired(100) + paired(100), resamples=100)


def test_public_artifact_contains_race_aggregates_without_sector_identities(tmp_path):
    rows = [row for session in range(100, 107) for row in paired(session)]
    private, public = tmp_path / "private.json", tmp_path / "public.json"
    metadata = {
        "aggregate_source_sha256": "source-hash",
        "dependency_code_sha256": {"features.py": "helper-hash"},
    }
    envelope = {
        "development_observations": rows,
        "mexico_observations": paired(11404),
        "metadata": metadata,
        "private_provenance": {"snapshot_path": "/private/raw"},
    }
    result = run_behavior_comparison(
        envelope,
        output_path=private,
        public_output_path=public,
        resamples=100,
    )
    safe = json.loads(public.read_text())
    assert safe["aggregate_only"] is True
    assert len(safe["primary"]["per_race"]) == 7
    assert "matched_cells" not in safe["primary"]
    assert "observation_ids" not in public.read_text()
    assert "/private/raw" not in public.read_text()
    assert safe["provenance"]["observation_metadata"] == metadata
    assert json.loads(private.read_text()) == result
    assert result == run_behavior_comparison(
        envelope,
        output_path=private,
        public_output_path=public,
        resamples=100,
    )
    with pytest.raises(ValueError, match="immutable"):
        run_behavior_comparison(rows, output_path=private, resamples=100)


def test_nonfinite_metrics_are_excluded_not_silently_averaged():
    rows = paired(100) + paired(101)
    rows[0]["full_throttle_fraction"] = np.inf
    result = matched_comparison(rows, resamples=100)
    assert result["race_count"] == 1
    assert result["coverage"]["exclusion_reason_counts"]["invalid_native_metric"] == 1
