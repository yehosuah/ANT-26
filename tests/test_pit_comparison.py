"""Cohort changes must never masquerade as paired forecasting improvements."""

import copy
import json

import pytest

from ant26.evaluation.metrics import macro_metrics
from ant26.evaluation.pit_comparison import (
    _digest,
    compare_development_examples,
    compare_pit_replays,
)


def replay(scored, *, censored=(), protocol="old-policy"):
    predictions, timeline = [], []
    for lap in range(1, 72):
        sectors = []
        for sector in range(1, 4):
            key = (lap, sector)
            identity = f"11404:12:{lap}:{sector}"
            status = "missing_record" if lap == 5 else "abstained"
            if key in scored or key in censored:
                status = "scored" if key in scored else "censored"
                median = 10 + scored.get(key, 0)
                predictions.append(
                    {
                        "example_id": identity,
                        "session_key": 11404,
                        "driver_number": 12,
                        "lap_number": lap,
                        "sector_index": sector,
                        "target_seconds": 10,
                        "reference_seconds": 10,
                        "origin": f"lap-{lap}-sector-{sector}",
                        "cutoff": f"cutoff-{lap}-{sector}",
                        "scoreable": status == "scored",
                        "target_valid": status == "scored",
                        "quantiles": [median - 2, median - 1, median, median + 1, median + 2],
                    }
                )
            entry = {"sector_index": sector, "status": status}
            if status != "missing_record":
                entry["example_id"] = identity
            sectors.append(entry)
        timeline.append({"lap_number": lap, "sectors": sectors})
    count = len(scored)
    report = {
        "synthetic": True,
        "race": {"session_key": 11404, "driver_number": 12, "lap_count": 71},
        "accounting": {
            "official_laps": 71,
            "expected_sector_records": 213,
            "observed_sector_records": 210,
            "missing_sector_records": 3,
            "eligible_forecasts": count + len(censored),
            "scoreable_forecasts": count,
            "censored_targets": len(censored),
            "abstentions": 210 - count - len(censored),
        },
        "metrics": macro_metrics([row for row in predictions if row["scoreable"]], 12),
        "provenance": {"protocol_sha256": protocol, "case_claim": "post_inspection_diagnostic"},
    }
    return report, predictions, timeline


def test_common_added_and_lost_cohorts_are_separate_with_all_213_slots():
    prior = replay({(1, 1): 4, (2, 1): 2})
    new = replay({(1, 1): 1, (47, 1): 8}, censored={(48, 1)}, protocol="new-policy")
    report = compare_pit_replays(*prior, *new)
    common = report["common_scoreable"]
    assert common["count"] == 1
    assert common["prior_metrics"]["macro"]["mae_seconds"] == 4
    assert common["new_metrics"]["macro"]["mae_seconds"] == 1
    assert common["delta_new_minus_prior"]["mae_seconds"] == -3
    assert report["newly_scoreable"]["count"] == 1
    assert report["newly_scoreable"]["metrics"]["macro"]["mae_seconds"] == 8
    assert report["lost_scoreable"]["count"] == 1
    assert report["lost_scoreable"]["metrics"]["macro"]["mae_seconds"] == 2
    assert len(report["coverage"]["newly_eligible"]) == 2
    assert len(report["coverage"]["lost_eligible"]) == 1
    whole = report["coverage"]["all_213_slots"]
    assert whole["prior"]["expected_slots"] == 213
    assert (
        whole["new"]["scored"]
        + whole["new"]["censored"]
        + whole["new"]["abstained"]
        + whole["new"]["missing_record"]
        == 213
    )
    late = report["coverage"]["late_75_slots_laps_47_to_71"]
    assert late["prior"]["expected_slots"] == 75
    assert late["prior"]["eligible"] == 0
    assert late["new"]["scored"] == 1 and late["new"]["censored"] == 1
    assert report["implementation_and_model_changes"]["protocol_changed"] is True
    assert len(report["coverage"]["by_lap"]) == 71
    assert report["coverage"]["by_lap"][46]["new"]["scored"] == 1
    assert report["implementation_and_model_changes"]["baseline_artifacts_changed"] is None
    assert report["interpretation"]["improvement_claim_allowed"] is False
    assert report["interpretation"]["whole_population_metrics_have_different_denominators"] is True


def test_no_common_scores_does_not_invent_paired_metrics():
    report = compare_pit_replays(*replay({}), *replay({(47, 1): 2}))
    assert report["common_scoreable"]["count"] == 0
    assert report["common_scoreable"]["delta_new_minus_prior"] == {}
    assert report["newly_scoreable"]["count"] == 1


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("label", "target labels changed"),
        ("cutoff", "cutoff changed"),
        ("missing_slot", "213"),
        ("duplicate", "duplicated"),
        ("accounting", "accounting"),
        ("metric", "metrics"),
        ("missing_prediction", "exactly one"),
        ("censored_as_scoreable", "scoreability"),
    ],
)
def test_malformed_or_incomparable_saved_artifacts_are_rejected(mutation, message):
    prior = replay({(1, 1): 4})
    new = replay({(1, 1): 1})
    if mutation == "label":
        new[1][0]["target_seconds"] = 11
        new[0]["metrics"] = macro_metrics(new[1], 12)
    elif mutation == "cutoff":
        new[1][0]["cutoff"] = "different"
    elif mutation == "missing_slot":
        new[2][-1]["sectors"].pop()
    elif mutation == "duplicate":
        new[1].append(copy.deepcopy(new[1][0]))
    elif mutation == "accounting":
        new[0]["accounting"]["abstentions"] += 1
    elif mutation == "metric":
        new[0]["metrics"]["macro"]["mae_seconds"] += 1
    elif mutation == "missing_prediction":
        new[1].clear()
    elif mutation == "censored_as_scoreable":
        new[1][0]["scoreable"] = False
    with pytest.raises(ValueError, match=message):
        compare_pit_replays(*prior, *new)


def attach_freeze(replay_data, *, model_hash, data_hash, protocol):
    freeze = {
        "protocol_sha256": protocol,
        "artifact_files": {
            "baseline/state.json": model_hash,
            "encoder/model.pt": model_hash,
            "index.json": model_hash,
        },
        "training_dataset": {"examples_sha256": data_hash},
        "index_hash": model_hash,
        "fit_sessions": [10],
        "calibration_sessions": [11],
    }
    freeze["manifest_sha256"] = _digest(freeze)
    replay_data[0]["provenance"].update(
        protocol_sha256=protocol,
        release_manifest_sha256=freeze["manifest_sha256"],
    )
    return freeze


def test_changed_model_and_development_data_are_explicit_not_a_policy_attribution():
    prior, new = replay({(1, 1): 4}), replay({(1, 1): 1})
    before = attach_freeze(prior, model_hash="old-model", data_hash="old-data", protocol="old")
    after = attach_freeze(new, model_hash="new-model", data_hash="new-data", protocol="new")
    report = compare_pit_replays(*prior, *new, prior_freeze=before, new_freeze=after)
    changes = report["implementation_and_model_changes"]
    assert changes["baseline_artifacts_changed"] is True
    assert changes["encoder_artifacts_changed"] is True
    assert changes["index_changed"] is True
    assert changes["development_examples_changed"] is True
    assert changes["fit_or_calibration_sessions_changed"] is False
    assert (
        report["interpretation"][
            "timestamp_policy_effect_is_not_isolated_from_model_or_data_changes"
        ]
        is True
    )
    after["artifact_files"]["baseline/state.json"] = "corrupted"
    with pytest.raises(ValueError, match="checksum"):
        compare_pit_replays(*prior, *new, prior_freeze=before, new_freeze=after)


def test_path_inputs_pin_exact_bytes_and_comparison_is_immutable(tmp_path):
    prior, new = replay({(1, 1): 4}), replay({(47, 1): 1})
    paths = []
    for index, value in enumerate((*prior, *new)):
        path = tmp_path / f"input-{index}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    output = tmp_path / "comparison.json"
    result = compare_pit_replays(*paths, output_path=output)
    assert result == compare_pit_replays(*paths, output_path=output)
    assert result["provenance"]["input_sha256"]["prior_report"] == _digest(paths[0])
    paths[3].write_text(json.dumps(new[0], indent=2))
    with pytest.raises(ValueError, match="immutable"):
        compare_pit_replays(*paths, output_path=output)


def development_row(session=10, *, identity="dev"):
    return {
        "example_id": identity,
        "session_key": session,
        "driver_number": 12,
        "lap_number": 4,
        "sector_index": 1,
        "origin": "2026-01-01T00:00:00Z",
        "eligible": True,
        "target_valid": True,
        "target_seconds": 20,
        "reference_seconds": 20,
        "features": {"speed_mean": 200},
    }


def test_development_policy_changes_are_counted_by_scope_and_driver_race():
    prior = [development_row(), development_row(11, identity="cal")]
    new = copy.deepcopy(prior)
    new[0]["features"]["speed_mean"] = 190
    new[1]["eligible"] = False
    result = compare_development_examples(prior, new, fit_sessions=[10], calibration_sessions=[11])
    assert result["changed_example_count"] == 2
    assert result["changed_top_level_fields"] == {"eligible": 1, "features": 1}
    assert result["change_impact"]["forecast_input_changed_examples"] == 1
    assert result["change_impact"]["eligibility_changed_examples"] == 1
    assert result["change_impact"]["metadata_only_changed_examples"] == 0
    assert result["change_impact_by_scope"]["final_fit"]["forecast_input_changed_examples"] == 1
    assert result["fit_scope_changed"] and result["calibration_scope_changed"]
    assert result["learned_artifact_reuse_allowed"] is False
    assert result["prior_counts"]["scoreable"] == 2
    assert result["new_counts"]["scoreable"] == 1
    assert len(result["by_driver_race"]) == 2
    equal = compare_development_examples(
        prior, copy.deepcopy(prior), fit_sessions=[10], calibration_sessions=[11]
    )
    assert equal["learned_artifact_reuse_allowed"] is True
    reordered = compare_development_examples(prior, list(reversed(prior)))
    assert reordered["learned_artifact_reuse_allowed"] is False


def test_metadata_only_changes_are_not_misreported_as_feature_changes():
    prior = [development_row()]
    new = copy.deepcopy(prior)
    new[0]["provenance"] = {"pit_timestamp_policy": "completion_report_v2"}
    result = compare_development_examples(prior, new, fit_sessions=[10], test_sessions=[10])
    impact = result["change_impact"]
    assert impact["metadata_only_changed_examples"] == 1
    assert impact["forecast_input_changed_examples"] == 0
    assert impact["substantive_changed_examples"] == 0
    assert result["change_impact_by_scope"]["rolling_test"]["metadata_only_changed_examples"] == 1


def test_development_comparison_never_accepts_mexico_or_duplicate_ids():
    with pytest.raises(ValueError, match="Mexico"):
        compare_development_examples([development_row(11404)], [development_row(11404)])
    with pytest.raises(ValueError, match="duplicate"):
        compare_development_examples([development_row(), development_row()], [])
