"""User-visible portfolio contracts without network or private experiment artifacts."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "portfolio.py"


@pytest.fixture
def offline_environment(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("ANT26_REPORT_ROOT", str(reports))
    monkeypatch.setenv("ANT26_PORTFOLIO_ARTIFACT_ROOT", str(artifacts))
    # The application has no data-service client; an accidental request is a regression.
    import httpx

    def forbidden(*args, **kwargs):
        raise AssertionError("The portfolio must not request network data")

    monkeypatch.setattr(httpx, "get", forbidden)
    return reports, artifacts


def test_fresh_checkout_all_pages_use_committed_aggregates(offline_environment):
    reports, artifacts = offline_environment
    for filename in ("development-results.json", "operations.json"):
        shutil.copyfile(ROOT / "reports" / filename, reports / filename)
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric} == {
        "Antonelli race traces": "18",
        "Native telemetry rows": "760,694",
        "Completed encoder fits": "160",
        "Held-out races": "8",
    }
    assert any("kept the baseline" in node.value for node in app.markdown)
    app.sidebar.radio[0].set_value("Development Results").run()
    assert not app.exception
    table = app.dataframe[0].value
    persistence = table.loc[table["Model"] == "Same-sector persistence"].iloc[0]
    assert persistence["MAE (s)"] == pytest.approx(0.9455166584)
    app.selectbox[0].set_value("Interval score").run()
    assert not app.exception
    chart = (list(app.get("vega_lite_chart")) + list(app.get("arrow_vega_lite_chart")))[0]
    spec = json.loads(chart.proto.spec)
    assert spec["encoding"]["x"]["stack"] is None
    assert spec["encoding"]["x"]["scale"]["domain"][0] == 0
    assert spec["encoding"]["color"]["type"] == "nominal"
    import pyarrow as pa

    plotted = pa.ipc.open_stream(chart.proto.data.data).read_all().to_pylist()
    assert {row["Model"]: row["Value"] for row in plotted} == dict(
        zip(table["Model"], table["Interval score"], strict=True)
    )
    app.radio[0].set_value("mercedes_control").run()
    assert not app.exception
    assert app.radio[0].value == "mercedes_control"
    for page in ("Mexico Case Study", "Atlas", "Methods"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    assert {m.label: m.value for m in app.metric}["CatBoost p95"] == "0.710 ms"
    assert not artifacts.exists()


def test_missing_and_malformed_reports_degrade_without_fake_data(offline_environment):
    reports, _ = offline_environment
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert not app.metric
    (reports / "development-results.json").write_text("{broken")
    (reports / "mexico-case-study.json").write_text("{broken")
    app.run()
    assert not app.exception
    assert any("could not be read" in warning.value for warning in app.warning)
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    assert not app.dataframe
    assert any("not been bundled" in node.value for node in app.info)
    app.sidebar.radio[0].set_value("Atlas").run()
    assert not app.exception
    assert any("No local case-retrieval" in node.value for node in app.info)
    app.sidebar.radio[0].set_value("Driving Findings").run()
    assert not app.exception
    assert any("analysis report has not been bundled" in node.value for node in app.info)


def test_observed_driving_works_without_private_source_artifacts(offline_environment):
    reports, artifacts = offline_environment
    for filename in (
        "driver-behavior.json",
        "mexico-driving-behavior.json",
        "driver-behavior-audit.json",
    ):
        shutil.copyfile(ROOT / "reports" / filename, reports / filename)
    source = json.loads((reports / "driver-behavior.json").read_text())
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.sidebar.radio[0].set_value("Driving Findings").run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric}["Matched races"] == str(
        source["primary"]["race_count"]
    )
    table = app.dataframe[0].value
    assert len(table) == 3
    assert table.iloc[0]["Antonelli minus Russell (pp)"] == pytest.approx(
        source["primary"]["metrics"]["full_throttle_fraction"]["difference_percentage_points"]
    )
    assert len(app.dataframe[1].value) == 4
    assert any("Additional fixed 98% coverage audit" in node.value for node in app.caption)
    assert not artifacts.exists()


def test_public_lap_accounting_retains_all_71_rows_and_zero_counts(offline_environment):
    reports, _ = offline_environment
    # Explicit synthetic accounting fixture, with no invented telemetry or durations.
    (reports / "mexico-case-study.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "race": {"lap_count": 71},
                "accounting": {
                    "official_laps": 71,
                    "observed_lap_records": 70,
                    "missing_lap_numbers": [5],
                    "scoreable_forecasts": 0,
                    "abstentions": 0,
                },
                "lap_accounting": [
                    {
                        "lap_number": lap,
                        "observed_sectors": 3,
                        "eligible_forecasts": 0,
                        "scoreable_forecasts": 0,
                        "abstentions": 0,
                        "censored_targets": 0,
                        "exclusion_reasons": {"fixture_only": 3},
                    }
                    for lap in range(1, 72)
                    if lap != 5
                ],
            }
        )
    )
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    assert any("Synthetic demonstration" in warning.value for warning in app.warning)
    assert {m.label: m.value for m in app.metric}["Scoreable forecasts"] == "0"
    accounting = app.dataframe[0].value
    assert accounting["Lap"].tolist() == list(range(1, 72))
    assert accounting.loc[accounting["Lap"] == 5, "Status"].item() == "Missing source timing"
    assert len(app.get("vega_lite_chart")) + len(app.get("arrow_vega_lite_chart")) == 1


def test_scorer_nested_timeline_censored_prediction_and_atlas_queries(offline_environment):
    reports, artifacts = offline_environment
    case = artifacts / "mexico-case-study"
    case.mkdir(parents=True)
    (reports / "mexico-case-study.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "race": {"lap_count": 71},
                "accounting": {
                    "official_laps": 71,
                    "observed_lap_records": 1,
                    "scoreable_forecasts": 1,
                    "abstentions": 1,
                },
            }
        )
    )
    (case / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "lap_number": 1,
                    "source_lap_record_present": True,
                    "sectors": [
                        {
                            "sector_index": 1,
                            "status": "scored",
                            "track_state": "GREEN",
                            "reasons": [],
                        },
                        {
                            "sector_index": 2,
                            "status": "censored",
                            "track_state": "RED",
                            "reasons": ["red_flag_target"],
                        },
                        {
                            "sector_index": 3,
                            "status": "abstained",
                            "track_state": "UNKNOWN",
                            "reasons": ["unknown_flag"],
                        },
                    ],
                },
                {
                    "lap_number": 2,
                    "source_lap_record_present": False,
                    "sectors": [
                        {
                            "sector_index": sector,
                            "status": "missing_record",
                            "reasons": ["missing_source_sector_record"],
                        }
                        for sector in (1, 2, 3)
                    ],
                },
            ]
        )
    )
    (case / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "model": "catboost_quantile",
                    "lap_number": lap,
                    "sector_index": 1,
                    "quantiles": [10, 11, 12, 13, 14],
                    "target_seconds": 12,
                    "scoreable": scoreable,
                }
                for lap, scoreable in ((1, True), (2, False))
            ]
        )
    )
    (case / "atlas.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "index_hash": "fixture",
                "retrospective": True,
                "validation_status": "exploratory_unvalidated",
                "human_rating_count": 0,
                "queries": [
                    {
                        "query_id": f"11404:12:{lap}:1",
                        "distance_metric": "euclidean",
                        "neighbors": [
                            {
                                "example_id": f"11396:63:{lap}:2",
                                "session_key": 11396,
                                "driver_number": 63,
                                "lap_number": lap,
                                "sector_index": 2,
                                "distance": 0.5,
                                "observed_summary": {"speed_mean": 123},
                                "context": {"track_state": "GREEN", "compound": "MEDIUM"},
                            }
                        ],
                    }
                    for lap in (1, 2)
                ],
            }
        )
    )
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    row = app.dataframe[0].value.iloc[0]
    assert (row["Eligible"], row["Scoreable"], row["Censored"], row["Abstentions"]) == (2, 1, 1, 1)
    assert app.dataframe[0].value.iloc[1]["Status"] == "Missing source timing"
    chart = (list(app.get("vega_lite_chart")) + list(app.get("arrow_vega_lite_chart")))[1]
    import pyarrow as pa

    values = pa.ipc.open_stream(chart.proto.data.data).read_all().to_pylist()
    assert [row["Lap"] for row in values] == [1]
    assert values[0]["Lower"] == 10 and values[0]["Upper"] == 14
    app.sidebar.radio[0].set_value("Atlas").run()
    assert not app.exception
    assert app.selectbox[0].value == 0
    app.selectbox[0].set_value(1).run()
    assert not app.exception
    assert any("11404:12:2:1" in caption.value for caption in app.caption)
    assert any("Human ratings: 0" in caption.value for caption in app.caption)
    assert any("63" in caption.value and "Lap 2" in caption.value for caption in app.caption)


@pytest.mark.parametrize("claim_location", ["case_kind", "provenance"])
def test_corrected_mexico_replay_prominently_discloses_original_failure(
    offline_environment, claim_location
):
    reports, _ = offline_environment
    # Preserve actual public first-attempt accounting; no invented performance metrics.
    report = json.loads((ROOT / "reports" / "mexico-first-attempt.json").read_text())
    claim = "post_inspection_implementation_correction_diagnostic_replay"
    if claim_location == "case_kind":
        report["case_kind"] = claim
    else:
        report["provenance"]["case_claim"] = claim
    report["implementation_correction"] = {
        "first_attempt_eligible_forecasts": 0,
        "first_attempt_scoreable_forecasts": 0,
        "forecast_artifacts_and_training_examples_unchanged": True,
    }
    (reports / "mexico-case-study.json").write_text(json.dumps(report))
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    banner = next(w.value for w in app.warning if "original frozen Mexico run" in w.value)
    assert "0 forecasts" in banner and "not an independent blind test" in banner
    assert any("mexico-first-attempt.json" in caption.value for caption in app.caption)
    assert any("training examples are unchanged" in caption.value for caption in app.caption)
    assert app.dataframe[0].value["Lap"].tolist() == list(range(1, 72))


def test_real_public_mexico_report_shows_selective_metrics_without_private_artifacts(
    offline_environment,
):
    reports, artifacts = offline_environment
    shutil.copyfile(ROOT / "reports" / "mexico-case-study.json", reports / "mexico-case-study.json")
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Official race laps"] == "71"
    assert values["Scoreable forecasts"] == "115"
    assert values["Abstentions"] == "95"
    assert values["Case MAE"] == "0.724 s"
    assert values["Case interval score"] == "0.505"
    assert values["Case 90% coverage"] == "93.0%"
    assert values["Case 90% width"] == "2.298 s"
    table = app.dataframe[0].value
    assert len(table) == 71
    assert table["Scoreable"].sum() == 115
    assert table["Abstentions"].sum() == 95
    assert table.loc[table["Lap"] > 46, "Abstentions"].sum() == 75
    chart = (list(app.get("vega_lite_chart")) + list(app.get("arrow_vega_lite_chart")))[0]
    spec = json.loads(chart.proto.spec)
    assert spec["height"] == 75
    assert spec["autosize"]["type"] == "fit-x"
    assert spec["encoding"]["y"] == {"field": "Band", "type": "nominal", "axis": None}
    assert spec["encoding"]["color"]["type"] == "nominal"
    assert any("not proof that the driver stayed" in notice.value for notice in app.info)
    assert not artifacts.exists()


def test_case_version_defaults_to_revised_and_matches_atlas_without_legacy_claims(
    offline_environment,
):
    reports, artifacts = offline_environment
    shutil.copyfile(ROOT / "reports" / "mexico-case-study.json", reports / "mexico-case-study.json")
    revised = {
        "synthetic": True,
        "case_kind": "post_inspection_feature_policy_revision_diagnostic_replay",
        "race": {"lap_count": 2},
        "accounting": {"official_laps": 2, "scoreable_forecasts": 3, "abstentions": 3},
        "lap_accounting": [
            {"lap_number": 1, "scoreable_forecasts": 3, "eligible_forecasts": 3, "abstentions": 0},
            {"lap_number": 2, "scoreable_forecasts": 0, "eligible_forecasts": 0, "abstentions": 3},
        ],
        # Even a misleading invariant flag cannot carry the legacy claim into a revision.
        "implementation_correction": {"forecast_artifacts_and_training_examples_unchanged": True},
        "chapters": [
            {
                "title": "Late fixture",
                "lap_start": 47,
                "lap_end": 71,
                "scoreable_forecasts": 0,
                "abstentions": 75,
            }
        ],
    }
    (reports / "mexico-case-study-pit-v0.2.0.json").write_text(json.dumps(revised))
    comparison = {
        "synthetic": True,
        "common_scoreable": {"count": 1},
        "newly_scoreable": {"count": 2},
        "lost_scoreable": {"count": 0},
        "coverage": {
            "late_75_slots_laps_47_to_71": {
                "prior": {"expected_slots": 3, "scored": 0, "abstained": 3},
                "new": {"expected_slots": 3, "scored": 2, "abstained": 1},
            }
        },
    }
    (reports / "pit-policy-comparison.json").write_text(json.dumps(comparison))
    for directory, identifier in (
        ("mexico-case-study", "old-query"),
        ("mexico-case-study-pit-v0.2.0", "revised-query"),
    ):
        path = artifacts / directory
        path.mkdir(parents=True)
        (path / "atlas.json").write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "human_rating_count": 0,
                    "queries": [{"query_id": identifier, "neighbors": []}],
                }
            )
        )
    app = AppTest.from_file(APP_PATH).run()
    assert app.sidebar.selectbox[0].value == "pit-v0.2.0"
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    assert any("revised feature policy" in warning.value for warning in app.warning)
    assert not any("training examples are unchanged" in caption.value for caption in app.caption)
    assert not any("All 75 sector origins" in notice.value for notice in app.info)
    assert {m.label: m.value for m in app.metric}["Common scored origins"] == "1"
    assert {m.label: m.value for m in app.metric}["Newly scored origins"] == "2"
    app.sidebar.radio[0].set_value("Atlas").run()
    assert not app.exception
    assert any("revised-query" in caption.value for caption in app.caption)
    app.sidebar.selectbox[0].set_value("published-v0.1").run()
    assert not app.exception
    assert any("old-query" in caption.value for caption in app.caption)
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric}["Scoreable forecasts"] == "115"
    assert any("All 75 sector origins" in notice.value for notice in app.info)


def test_revised_baseline_summary_is_separate_from_historical_neural_fits(offline_environment):
    reports, _ = offline_environment
    shutil.copyfile(
        ROOT / "reports" / "development-results.json", reports / "development-results.json"
    )
    (reports / "pit-policy-development.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "protocol_version": "0.2.0",
                "cohort_driver_numbers": [12, 63],
                "input_example_count": 6,
                "scoreable_example_count": 5,
                "evaluated_fold_count": 2,
                "summary": {
                    "catboost_quantile": {
                        "antonelli": {
                            "macro": {
                                "mae_seconds": 1.5,
                                "weighted_interval_score": 0.8,
                                "coverage_90": 0.9,
                            },
                            "forecast_count": 5,
                        }
                    }
                },
            }
        )
    )
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Development Results").run()
    assert not app.exception
    assert any("160 neural fits remain attached" in caption.value for caption in app.caption)
    assert {m.label: m.value for m in app.metric}["Revised-policy scoreable examples"] == "5"
    assert app.dataframe[-1].value.iloc[0]["MAE (s)"] == 1.5


def test_operations_report_tracks_the_selected_case_weights(offline_environment):
    reports, _ = offline_environment
    for filename in (
        "mexico-case-study.json",
        "mexico-case-study-pit-v0.2.0.json",
        "operations.json",
        "operations-pit-v0.2.0.json",
    ):
        shutil.copyfile(ROOT / "reports" / filename, reports / filename)
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Methods").run()
    assert not app.exception
    values = {metric.label: metric.value for metric in app.metric}
    assert (values["CatBoost p95"], values["Encoder p95"]) == ("0.716 ms", "0.372 ms")
    assert any("reports/operations-pit-v0.2.0.json" in caption.value for caption in app.caption)
    app.sidebar.selectbox[0].set_value("published-v0.1").run()
    assert not app.exception
    values = {metric.label: metric.value for metric in app.metric}
    assert (values["CatBoost p95"], values["Encoder p95"]) == ("0.710 ms", "0.360 ms")
    assert any("reports/operations.json" in caption.value for caption in app.caption)
    # A missing revised report cannot silently show timings from older weights.
    (reports / "operations-pit-v0.2.0.json").unlink()
    app.sidebar.selectbox[0].set_value("pit-v0.2.0").run()
    assert not app.exception
    assert not app.metric


def test_real_revised_case_reconciles_censoring_comparison_and_legacy_switch(offline_environment):
    reports, _ = offline_environment
    for filename in (
        "mexico-case-study.json",
        "mexico-case-study-pit-v0.2.0.json",
        "pit-policy-comparison.json",
    ):
        shutil.copyfile(ROOT / "reports" / filename, reports / filename)
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Mexico Case Study").run()
    assert not app.exception
    assert app.sidebar.selectbox[0].value == "pit-v0.2.0"
    values = {metric.label: metric.value for metric in app.metric}
    assert (
        values["Scoreable forecasts"],
        values["Abstentions"],
        values["Common scored origins"],
        values["Newly scored origins"],
        values["Lost scored origins"],
    ) == ("177", "32", "115", "62", "0")
    assert (values["Case MAE"], values["Case interval score"], values["Case 90% coverage"]) == (
        "0.876 s",
        "0.610",
        "92.7%",
    )
    common = next(
        table.value for table in app.dataframe if "Common-origin subset" in table.value.columns
    )
    assert common.loc[common["Common-origin subset"] == "Prior", "MAE (s)"].item() == pytest.approx(
        0.7236416399782453
    )
    assert common.loc[
        common["Common-origin subset"] == "Revised", "MAE (s)"
    ].item() == pytest.approx(0.7204364570100978)
    assert common.loc[
        common["Common-origin subset"] == "Revised", "Interval score"
    ].item() == pytest.approx(0.5093054728564964)
    accounting = next(table.value for table in app.dataframe if "Lap" in table.value.columns)
    assert len(accounting) == 71 and accounting["Scoreable"].sum() == 177
    assert accounting["Censored"].sum() == 1
    assert accounting.loc[accounting["Lap"] == 65, "Status"].item() == "Censored"
    assert not any("All 75 sector origins" in notice.value for notice in app.info)
    assert not any("training examples are unchanged" in caption.value for caption in app.caption)
    coverage = app.dataframe[0].value
    assert len(coverage) == 4  # two real scopes x prior/new, excluding status-transition maps
    late = coverage.loc[coverage["Scope"] == "late_75_slots_laps_47_to_71"]
    assert late.loc[late["Version"] == "new", "scored"].item() == 62
    app.sidebar.selectbox[0].set_value("published-v0.1").run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Scoreable forecasts"] == "115"
    assert any("All 75 sector origins" in notice.value for notice in app.info)
