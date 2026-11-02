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
