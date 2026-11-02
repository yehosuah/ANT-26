"""Offline-first ANT-26 portfolio over safe aggregates and optional local replay."""

from __future__ import annotations

import html
import json
import math
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path(os.environ.get("ANT26_REPORT_ROOT", str(ROOT / "reports")))
ARTIFACT_ROOT = Path(os.environ.get("ANT26_PORTFOLIO_ARTIFACT_ROOT", str(ROOT / "artifacts")))
PAGES = ["Overview", "Development Results", "Mexico Case Study", "Atlas", "Methods"]
MODEL_NAMES = {
    "same_sector_persistence": "Same-sector persistence",
    "historical_reference": "Recent-sector median",
    "catboost_quantile": "CatBoost quantiles",
    "regularized_linear_residual": "Linear residual",
    "finetune": "Masked encoder · fine-tuned",
    "frozen": "Masked encoder · frozen",
    "scratch": "Encoder · from scratch",
}
RACE_NAMES = {
    11342: "Hungary",
    11353: "Netherlands",
    11361: "Italy",
    11369: "Madrid",
    11377: "Azerbaijan",
    11731: "Malaysia",
    11388: "Singapore",
    11396: "United States",
}

st.set_page_config(page_title="ANT-26 · F1 behavior research", page_icon="↗", layout="wide")
st.markdown(
    """
<style>
.stApp {background:#f5f2ea;}
[data-testid="stHeader"] {background:rgba(245,242,234,.95);}
[data-testid="stSidebar"] {background:#20272c;border-right:1px solid #394147;}
[data-testid="stSidebar"] * {color:#eeeae1;}
[data-testid="stSidebar"] hr {border-color:#485056;}
.block-container {max-width:1430px;padding:2.1rem 2.8rem 4rem;}
h1,h2,h3 {letter-spacing:-.035em;color:#242b30;}
h1 {font-weight:650 !important;}
[data-testid="stMetric"] {padding:1rem 1.15rem;background:#fcfaf5;border:1px solid #dcd7cb;border-radius:10px;}
[data-testid="stMetricLabel"] {font-size:.78rem;letter-spacing:.02em;}
[data-testid="stMetricValue"] {font-size:2rem;letter-spacing:-.055em;}
.ant-hero {background:#20272c;color:#f7f4ed;border-radius:14px;padding:2rem 2.2rem;margin-bottom:1.45rem;}
.ant-hero h1 {color:#f7f4ed;font-size:3.7rem;line-height:1.05;margin:.45rem 0 .7rem;}
.ant-hero p {color:#d3d6d2;font-size:1.02rem;max-width:840px;margin:0;line-height:1.65;}
.ant-kicker {font-family:monospace;font-size:.74rem;letter-spacing:.16em;color:#dfb77a;text-transform:uppercase;}
.ant-tag {display:inline-block;border:1px solid #59615f;border-radius:20px;padding:.28rem .65rem;font-size:.72rem;color:#e8d3ae;margin-top:1rem;}
.ant-card {border-top:2px solid #ad8a52;padding:.8rem .1rem .5rem;margin:.5rem 0;}
.ant-card h3 {font-size:1.12rem;margin:0 0 .45rem;letter-spacing:-.025em;}
.ant-card p {font-size:.92rem;line-height:1.65;color:#626760;}
.ant-verdict {border-left:3px solid #a77f43;padding:.7rem 1rem;background:#eae4d6;border-radius:0 8px 8px 0;line-height:1.6;margin:1rem 0 1.5rem;}
.ant-pipeline {display:grid;grid-template-columns:repeat(5,1fr);gap:.65rem;margin:1rem 0 1.4rem;}
.ant-pipeline div {background:#fcfaf5;border:1px solid #d8d1c4;border-radius:9px;padding:1rem;font-size:.9rem;}
.ant-pipeline small {display:block;color:#8f723f;font-family:monospace;margin-bottom:.4rem;}
@media(max-width:800px) {.block-container{padding:1.4rem 1rem 3rem}.ant-hero{padding:1.5rem}.ant-hero h1{font-size:2.8rem}.ant-pipeline{grid-template-columns:1fr 1fr}}
</style>
""",
    unsafe_allow_html=True,
)


def load_json(path: Path):
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, (dict, list)):
            return None, f"{path.name} has an unsupported structure."
        return payload, None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError):
        return None, f"{path.name} could not be read. Its charts are unavailable."


def hero(title, subtitle, *, kicker="ANT-26 / RESEARCH PORTFOLIO", tag=None):
    badge = f'<span class="ant-tag">{html.escape(tag)}</span>' if tag else ""
    st.markdown(
        f'<div class="ant-hero"><div class="ant-kicker">{html.escape(kicker)}</div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{badge}</div>',
        unsafe_allow_html=True,
    )


def card(title, text):
    st.markdown(
        f'<div class="ant-card"><h3>{html.escape(title)}</h3><p>{html.escape(text)}</p></div>',
        unsafe_allow_html=True,
    )


def metrics(items):
    for column, (label, value, help_text) in zip(st.columns(len(items)), items, strict=True):
        with column:
            st.metric(label, value, help=help_text)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    return next((mapping[key] for key in keys if finite(mapping.get(key))), None)


def percent(value):
    return f"{value:.1%}" if finite(value) else "—"


def source_footer(filename):
    st.caption(
        f"Evidence: {filename}. Aggregate results are bundled with the repository; raw source snapshots and weights remain local."
    )


def result_table(experiment):
    primary = experiment.get("primary_antonelli", {})
    rows = []
    for group, family in (("baselines", "Baseline"), ("representations", "Encoder")):
        for name, result in primary.get(group, {}).items():
            macro = result.get("macro", {})
            rows.append(
                {
                    "Model": MODEL_NAMES.get(name, name),
                    "Family": family,
                    "MAE (s)": macro.get("mae_seconds"),
                    "Interval score": macro.get("weighted_interval_score"),
                    "90% coverage": macro.get("coverage_90"),
                    "90% width (s)": macro.get("width_90_seconds"),
                }
            )
    return rows


def rows_from(payload, *keys):
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = next((payload[key] for key in keys if isinstance(payload.get(key), list)), [])
        return [row for row in rows if isinstance(row, dict)]
    return []


def displayed_number(mapping, *keys):
    value = number(mapping, *keys)
    return value if value is not None else "—"


def synthetic_notice(payload):
    if isinstance(payload, dict) and (
        payload.get("synthetic") is True
        or str(payload.get("data_kind", "")).startswith("synthetic")
    ):
        st.warning("Synthetic demonstration output — these are not observed race data.")


def lap_summary(row):
    """Adapt the scorer's sector statuses without treating censored labels as scored."""
    if row is None or not isinstance(row.get("sectors"), list):
        return row
    sectors = rows_from(row["sectors"])
    statuses = Counter(sector.get("status") for sector in sectors)
    return {
        **row,
        "observed_sectors": sum(sector.get("status") != "missing_record" for sector in sectors),
        "scoreable_forecasts": statuses["scored"],
        "eligible_forecasts": statuses["scored"] + statuses["censored"],
        "abstentions": statuses["abstained"],
        "censored_targets": statuses["censored"],
        "exclusion_reasons": dict(
            Counter(reason for sector in sectors for reason in sector.get("reasons", []))
        ),
    }


development, development_error = load_json(REPORT_ROOT / "development-results.json")
if not isinstance(development, dict):
    development = None
mexico, mexico_error = load_json(REPORT_ROOT / "mexico-case-study.json")
if not isinstance(mexico, dict):
    mexico = None

with st.sidebar:
    st.markdown("### ANT-26")
    st.caption("F1 TELEMETRY / ML RESEARCH")
    st.divider()
    page = st.radio("Explore", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Kimi Antonelli · 2026")
    st.caption("Reproducible evidence. Explicit uncertainty.")
    st.caption("Offline portfolio · no data-service connection required")

if development_error:
    st.warning(development_error)
if mexico_error and page == "Mexico Case Study":
    st.warning(mexico_error)

if page == "Overview":
    hero(
        "ANT-26",
        "A reproducible study of Kimi Antonelli's 2026 season: learn driving representations, forecast the next sector, and inspect the evidence behind both.",
        tag="Behavioral atlas + probabilistic forecasting",
    )
    if development:
        facts = development.get("source_acquisition_facts", {})
        experiments = development.get("experiments", {})
        ant = experiments.get("antonelli_only", {})
        metrics(
            [
                (
                    "Antonelli race traces",
                    facts.get("complete_timed_driver_race_traces_by_driver", {}).get("12", "—"),
                    "Verified pre-Mexico timed race traces.",
                ),
                (
                    "Native telemetry rows",
                    f"{facts['native_telemetry_rows_total']:,}"
                    if finite(facts.get("native_telemetry_rows_total"))
                    else "—",
                    "Combined Antonelli and Russell source rows; this is a two-driver cohort.",
                ),
                (
                    "Completed encoder fits",
                    development.get("total_completed_representation_fits", "—"),
                    "Fixed-budget fits across chronological folds, modes and seeds.",
                ),
                (
                    "Held-out races",
                    ant.get("budget", {}).get("fold_count", "—"),
                    "Whole-race chronological test partitions.",
                ),
            ]
        )
        st.markdown(
            '<div class="ant-verdict"><strong>The benchmark kept the baseline.</strong><br>No learned-model improvement gate passed under the declared fixed budgets. The neural experiments, uncertainty checks and failures remain part of the result.</div>',
            unsafe_allow_html=True,
        )
        columns = st.columns([1.2, 1])
        with columns[0]:
            st.subheader("A question with a measurable answer")
            st.write(
                "Does a self-supervised telemetry representation improve next-sector prediction over strong statistical and tree baselines? The study tests that question on future races and uses a separate review protocol for retrieval quality."
            )
            table = result_table(ant)
            baseline = [r for r in table if r["Family"] == "Baseline" and finite(r["MAE (s)"])]
            candidate = next((r for r in table if r["Model"] == MODEL_NAMES["finetune"]), None)
            if baseline and candidate:
                best = min(baseline, key=lambda row: row["MAE (s)"])
                st.caption(
                    f"Antonelli-only development: best observed baseline MAE {best['MAE (s)']:.3f} s; fine-tuned encoder MAE {candidate['MAE (s)']:.3f} s. Equal race weights across five encoder seeds."
                )
        with columns[1]:
            card(
                "Source integrity",
                "Immutable snapshots, schema checks and explicit masks preserve the distinction between native observations and reconstructed tensors.",
            )
            card(
                "Scientific restraint",
                "Forecast validation and representation usefulness are separate questions. No latent intention or exact racing-line claim is made.",
            )
    else:
        st.info(
            "Development aggregates are not available in this checkout. Methods remain accessible; no metrics have been invented."
        )
    st.subheader("From source to an inspectable result")
    st.markdown(
        '<div class="ant-pipeline"><div><small>01 / DATA</small>Pinned source snapshots</div><div><small>02 / CONTRACTS</small>Past-only sector windows</div><div><small>03 / MODELS</small>Baselines + masked encoder</div><div><small>04 / EVALUATION</small>Race holdouts + calibration</div><div><small>05 / PRODUCT</small>Replay + evidence retrieval</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        card(
            "Mexico as the final case",
            "The final case page uses frozen-run evidence when the case report is bundled. Development results remain separate from that held-out race.",
        )
    with right:
        card(
            "A system engineers can inspect",
            "A small Python package separates provider adapters, domain rules, feature transforms, model fitting, evaluation and artifact serving.",
        )
    if development:
        source_footer("reports/development-results.json")

elif page == "Development Results":
    hero(
        "What the benchmark found",
        "Chronological evaluation of sector timing, probabilistic uncertainty, and representation learning. Antonelli remains the primary test subject.",
        kicker="ANT-26 / DEVELOPMENT RESULTS",
    )
    if not development:
        st.info(
            "No development result report is available. This page does not substitute synthetic results."
        )
    else:
        available = development.get("experiments", {})
        labels = {"antonelli_only": "Antonelli only", "mercedes_control": "Antonelli + Russell"}
        keys = [key for key in labels if key in available]
        if not keys:
            st.warning("The development report contains no supported experiment summaries.")
        else:
            selected = st.radio("Training cohort", keys, format_func=labels.get, horizontal=True)
            experiment = available[selected]
            budget, cohort = experiment.get("budget", {}), experiment.get("cohort", {})
            metrics(
                [
                    (
                        "Held-out races",
                        budget.get("fold_count", "—"),
                        "Whole weekends remain grouped.",
                    ),
                    (
                        "Training seeds",
                        len(budget.get("seeds", [])),
                        "Every declared seed is retained.",
                    ),
                    (
                        "Scoreable source examples",
                        f"{cohort['scoreable_example_count']:,}"
                        if finite(cohort.get("scoreable_example_count"))
                        else "—",
                        "All cohort examples before test-fold selection.",
                    ),
                    (
                        "Encoder parameters",
                        f"{budget.get('encoder_parameters', [0])[0]:,}"
                        if budget.get("encoder_parameters")
                        else "—",
                        "Fixed architecture, with no configuration search in these experiments.",
                    ),
                ]
            )
            st.info(
                "No learned-model improvement claim passed. Lower observed errors are descriptive; the report does not establish a blanket model-capacity conclusion."
            )
            table = result_table(experiment)
            metric = st.selectbox("Prediction metric", ["MAE (s)", "Interval score"])
            chart_rows = [
                {"Model": row["Model"], "Family": row["Family"], "Value": row[metric]}
                for row in table
                if finite(row.get(metric))
            ]
            if chart_rows:
                st.vega_lite_chart(
                    {
                        "data": {"values": chart_rows},
                        "height": 300,
                        "mark": {"type": "bar", "cornerRadiusEnd": 3},
                        "encoding": {
                            "y": {
                                "field": "Model",
                                "type": "nominal",
                                "sort": {"field": "Value", "order": "ascending"},
                                "title": None,
                            },
                            "x": {
                                "field": "Value",
                                "type": "quantitative",
                                "title": f"Race-macro {metric} · lower is better",
                            },
                            "color": {
                                "field": "Family",
                                "scale": {
                                    "domain": ["Baseline", "Encoder"],
                                    "range": ["#526b68", "#bd945d"],
                                },
                                "legend": {"orient": "top"},
                            },
                            "tooltip": [{"field": "Model"}, {"field": "Value", "format": ".4f"}],
                        },
                    },
                    width="stretch",
                )
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            st.caption(
                "Forecast metrics use equal race weights. Encoder summaries average seed errors within each race. Source-row counts are not independent race samples."
            )
            with st.expander("Five-seed variation and paired uncertainty"):
                for mode, result in (
                    experiment.get("primary_antonelli", {}).get("representations", {}).items()
                ):
                    st.markdown(f"**{MODEL_NAMES.get(mode, mode)}**")
                    seeds = [
                        {
                            "Seed": seed,
                            "MAE (s)": values.get("mae_seconds"),
                            "Interval score": values.get("weighted_interval_score"),
                            "90% coverage": percent(values.get("coverage_90")),
                        }
                        for seed, values in result.get("per_seed_macro", {}).items()
                    ]
                    if seeds:
                        st.dataframe(pd.DataFrame(seeds), hide_index=True, width="stretch")
                gains = []
                for mode, result in (
                    experiment.get("primary_antonelli", {}).get("paired_gains", {}).items()
                ):
                    for name, value in result.get("metrics", {}).items():
                        interval = value.get("confidence_interval_95", [None, None])
                        gains.append(
                            {
                                "Encoder": mode,
                                "Metric": name,
                                "Baseline": MODEL_NAMES.get(
                                    value.get("baseline_model"), value.get("baseline_model")
                                ),
                                "Relative gain": percent(value.get("relative_improvement")),
                                "95% interval": f"[{percent(interval[0])}, {percent(interval[1])}]",
                            }
                        )
                if gains:
                    st.dataframe(pd.DataFrame(gains), hide_index=True, width="stretch")
                st.caption(
                    "Positive gains favor the encoder. Confidence intervals resample complete races, retaining within-race dependence."
                )
            st.subheader("Coverage changes what is evaluated")
            retained = [
                {
                    "Race": RACE_NAMES.get(int(key), str(key)),
                    "Known-green origins": values["nominal_green_opportunities"],
                    "Eligible origins": values["origin_eligible"],
                    "Retention": percent(values["retention"]),
                }
                for key, values in experiment.get("retention_by_test_race", {}).items()
            ]
            if retained:
                st.dataframe(pd.DataFrame(retained), hide_index=True, width="stretch")
            st.caption(
                "Retention uses observed origins with known green track state and outside-pit state. It is not coverage of every official race opportunity. Hungary's selective missingness remains visible."
            )
            source_footer("reports/development-results.json")

elif page == "Mexico Case Study":
    hero(
        "Mexico 2026",
        "The final race case study: forecasts, interruptions and coverage under a frozen development protocol.",
        kicker="ANT-26 / FINAL CASE STUDY",
        tag="Development and final-case evidence stay separate",
    )
    if not mexico:
        st.info(
            "The final Mexico case report has not been bundled yet. No race replay or case metrics are displayed without source-backed output."
        )
        card(
            "What this page will account for",
            "Every race lap, including missing source timing, eligible forecasts, abstentions, censored targets and neutralized periods.",
        )
    else:
        synthetic_notice(mexico)
        race = mexico.get("race", {})
        accounting = mexico.get("accounting", mexico.get("coverage", {}))
        laps = number(accounting, "official_laps", "expected_laps") or number(
            race, "lap_count", "laps"
        )
        metrics(
            [
                (
                    "Official race laps",
                    int(laps) if finite(laps) else "—",
                    "From the final case report.",
                ),
                (
                    "Observed lap records",
                    displayed_number(accounting, "observed_lap_records", "observed_laps"),
                    "Missing source records remain accounted for.",
                ),
                (
                    "Scoreable forecasts",
                    displayed_number(
                        accounting, "scoreable_forecasts", "scored_forecasts", "scoreable_examples"
                    ),
                    "Valid labels and eligible past-only inputs.",
                ),
                (
                    "Abstentions",
                    number(accounting, "abstentions", "abstained_forecasts")
                    if number(accounting, "abstentions", "abstained_forecasts") is not None
                    else "—",
                    "Forecasts withheld under the declared input rules.",
                ),
            ]
        )
        if mexico.get("status"):
            st.caption(f"Case status: {mexico['status']}")
        for chapter in mexico.get("chapters", []):
            if isinstance(chapter, dict):
                card(
                    chapter.get("title", "Race phase"),
                    chapter.get("summary", chapter.get("description", "")),
                )
        case_root = ARTIFACT_ROOT / "mexico-case-study"
        local_timeline, timeline_error = load_json(case_root / "timeline.json")
        synthetic_notice(local_timeline)
        if timeline_error:
            st.warning(timeline_error)
        timeline = (
            rows_from(local_timeline, "laps", "timeline")
            or rows_from(mexico.get("lap_accounting"), "laps")
            or rows_from(mexico.get("timeline"), "laps", "timeline")
        )
        if timeline and finite(laps):
            lookup = {int(row.get("lap_number", row.get("lap", 0))): row for row in timeline}
            missing = set(accounting.get("missing_lap_numbers", []))
            complete = []
            for lap in range(1, int(laps) + 1):
                row = lap_summary(lookup.get(lap))
                scored = number(
                    row, "scoreable_forecasts", "scored_forecasts", "scoreable", "scored"
                )
                eligible = number(row, "eligible_forecasts", "origin_eligible", "eligible")
                abstained = number(row, "abstentions", "abstained")
                status = (
                    "Missing source timing"
                    if lap in missing
                    or (row is not None and row.get("source_lap_record_present") is False)
                    else "No timeline record"
                    if row is None
                    else "Scored"
                    if scored == 3
                    else "Partial"
                    if scored is not None and scored > 0
                    else "Abstained"
                    if abstained is not None and abstained > 0
                    else "Censored"
                    if number(row, "censored_targets")
                    else "Observed"
                )
                complete.append(
                    {
                        "Lap": lap,
                        "Status": status,
                        "Eligible": eligible,
                        "Scoreable": scored,
                        "Abstentions": abstained,
                        "Censored": number(row, "censored_targets"),
                        "Observed sectors": number(row, "observed_sectors"),
                        "Exclusion reasons": json.dumps(row.get("exclusion_reasons", {}))
                        if row is not None
                        else "",
                    }
                )
            st.subheader("Complete lap accounting")
            st.vega_lite_chart(
                {
                    "data": {"values": complete},
                    "height": 75,
                    "mark": "rect",
                    "encoding": {
                        "x": {
                            "field": "Lap",
                            "type": "ordinal",
                            "title": "Race lap",
                            "axis": {"labelAngle": 0, "labelOverlap": True},
                        },
                        "color": {
                            "field": "Status",
                            "scale": {
                                "domain": [
                                    "Scored",
                                    "Partial",
                                    "Abstained",
                                    "Missing source timing",
                                    "No timeline record",
                                    "Observed",
                                    "Censored",
                                ],
                                "range": [
                                    "#547970",
                                    "#c4a066",
                                    "#77818b",
                                    "#a26051",
                                    "#dad4c9",
                                    "#b9bdb7",
                                    "#9b819c",
                                ],
                            },
                            "legend": {"orient": "bottom"},
                        },
                        "tooltip": [
                            {"field": "Lap"},
                            {"field": "Status"},
                            {"field": "Scoreable"},
                            {"field": "Abstentions"},
                        ],
                    },
                },
                width="stretch",
            )
            st.dataframe(pd.DataFrame(complete), hide_index=True, width="stretch")
            sector_rows = [
                {
                    "Lap": lap.get("lap_number"),
                    "Sector": sector.get("sector_index"),
                    "Status": sector.get("status"),
                    "Track state": sector.get("track_state"),
                    "Reasons": ", ".join(sector.get("reasons", [])),
                }
                for lap in timeline
                for sector in rows_from(lap.get("sectors"))
            ]
            if sector_rows:
                with st.expander("Sector status and exclusion details"):
                    st.dataframe(pd.DataFrame(sector_rows), hide_index=True, width="stretch")
        else:
            st.caption(
                "Detailed lap accounting is available when the local timeline artifact is present."
            )
        local_predictions, predictions_error = load_json(case_root / "predictions.json")
        synthetic_notice(local_predictions)
        if predictions_error:
            st.warning(predictions_error)
        predictions = rows_from(local_predictions, "predictions")
        if predictions:
            st.subheader("Forecast replay")
            model_names = sorted({str(row.get("model", "frozen model")) for row in predictions})
            model = st.selectbox("Frozen forecast model", model_names)
            sector = st.selectbox("Sector", [1, 2, 3])
            plotted = []
            for row in predictions:
                parts = str(row.get("example_id", "")).split(":")
                lap = row.get(
                    "lap_number",
                    int(parts[-2]) if len(parts) == 4 and parts[-2].isdigit() else None,
                )
                row_sector = row.get(
                    "sector_index",
                    int(parts[-1]) if len(parts) == 4 and parts[-1].isdigit() else None,
                )
                quantiles = row.get("quantiles", [])
                if (
                    str(row.get("model", "frozen model")) == model
                    and row_sector == sector
                    and row.get("scoreable") is True
                    and isinstance(quantiles, list)
                    and len(quantiles) == 5
                    and finite(lap)
                    and finite(row.get("target_seconds"))
                    and all(finite(q) for q in quantiles)
                ):
                    plotted.append(
                        {
                            "Lap": lap,
                            "Actual": row["target_seconds"],
                            "Median": quantiles[2],
                            "Lower": quantiles[0],
                            "Upper": quantiles[4],
                        }
                    )
            if plotted:
                st.vega_lite_chart(
                    {
                        "data": {"values": plotted},
                        "height": 320,
                        "encoding": {
                            "x": {"field": "Lap", "type": "quantitative", "title": "Race lap"}
                        },
                        "layer": [
                            {
                                "mark": {"type": "area", "color": "#c6a363", "opacity": 0.22},
                                "encoding": {
                                    "y": {
                                        "field": "Lower",
                                        "type": "quantitative",
                                        "title": "Sector duration (s)",
                                        "scale": {"zero": False},
                                    },
                                    "y2": {"field": "Upper"},
                                },
                            },
                            {
                                "mark": {"type": "line", "color": "#a57b39"},
                                "encoding": {"y": {"field": "Median", "type": "quantitative"}},
                            },
                            {
                                "mark": {"type": "point", "color": "#293a43", "filled": True},
                                "encoding": {"y": {"field": "Actual", "type": "quantitative"}},
                            },
                        ],
                    },
                    width="stretch",
                )
                st.caption(
                    "Dots are valid realized durations; the line is the forecast median and the band spans q05–q95 (90%). Stored quantiles are q05, q10, q50, q90 and q95. Only scoreable rows appear here; censored targets and abstentions remain in the accounting above."
                )
            else:
                st.info("No scored forecast rows are available for this model and sector.")
        with st.expander("Case metrics and freeze provenance"):
            st.json(
                {
                    key: mexico[key]
                    for key in ("metrics", "provenance", "freeze", "limitations")
                    if key in mexico
                }
            )
        source_footer("reports/mexico-case-study.json")

elif page == "Atlas":
    hero(
        "Behavior through comparison",
        "Retrieve earlier windows with similar observed execution. Compare the evidence without assigning hidden intentions to the driver.",
        kicker="ANT-26 / BEHAVIORAL ATLAS",
    )
    left, right = st.columns(2)
    with left:
        card(
            "Continuous similarity",
            "The encoder maps past-only telemetry windows into a learned representation. A separate feature/PCA index is the comparison baseline.",
        )
    with right:
        card(
            "State claims withheld",
            "The current evidence does not validate discrete driving states. Human retrieval judgments remain separate from forecast-head performance.",
        )
    atlas, error = load_json(ARTIFACT_ROOT / "mexico-case-study" / "atlas.json")
    synthetic_notice(atlas)
    if error:
        st.warning(error)
    queries = rows_from(atlas, "queries")
    if not queries and isinstance(atlas, dict) and atlas.get("neighbors"):
        queries = [atlas]
    if queries:
        st.subheader("Local case retrieval")
        st.warning(
            "Exploratory continuous retrieval. Human validation has not been established; these similarities do not identify driving intent."
        )
        if isinstance(atlas, dict):
            st.caption(
                f"Retrieval mode: {'retrospective' if atlas.get('retrospective') else 'prefix'} · Human ratings: {atlas.get('human_rating_count', 'not recorded')} · Validation: {atlas.get('validation_status', 'not established')}"
            )
        query_index = st.selectbox(
            "Case query",
            range(len(queries)),
            format_func=lambda index: str(queries[index].get("query_id", f"Query {index + 1}")),
        )
        query = queries[query_index]
        st.caption(
            f"Query source ID: {query.get('query_id', '—')} · Distance metric: {query.get('distance_metric', '—')}"
        )
        neighbors = rows_from(query, "neighbors")
        if neighbors:
            neighbor_index = st.selectbox(
                "Historical neighbor",
                range(len(neighbors)),
                format_func=lambda index: (
                    f"{index + 1}. {neighbors[index].get('example_id', 'Reference')}"
                ),
            )
            neighbor = neighbors[neighbor_index]
            with st.container(border=True):
                st.markdown(f"**{neighbor.get('example_id', 'Historical reference')}**")
                st.caption(
                    f"Session {neighbor.get('session_key', '—')} · Driver {neighbor.get('driver_number', '—')} · Lap {neighbor.get('lap_number', '—')} · Sector {neighbor.get('sector_index', '—')}"
                )
                summary, conditions = st.columns(2)
                with summary:
                    st.markdown("**Observed execution**")
                    st.json(neighbor.get("observed_summary", {}))
                with conditions:
                    st.markdown("**Conditions at cutoff**")
                    st.json(neighbor.get("context", {}))
                st.caption(
                    f"Distance: {neighbor.get('distance', '—')} · Observation cutoff: {neighbor.get('cutoff', '—')}"
                )
                with st.expander("Reference provenance"):
                    st.json(neighbor.get("provenance", {}))
        else:
            st.info("This query has no eligible historical neighbors.")
        st.caption(
            "Distances compare observed execution under this representation. They are not calibrated probabilities of shared intention."
        )
    else:
        st.info(
            "No local case-retrieval artifact is bundled. The portfolio does not fabricate neighbors or telemetry traces."
        )
    with st.expander("How the blinded review works"):
        st.write(
            "The fixed review protocol compares 60 queries across eight chronological held-out races. Each fold uses its own frozen encoder and exact fitting references. Reviewers see A/B observations and conditions, with methods and distances hidden. No preferences are filled automatically."
        )
        st.code(
            "uv run --extra app streamlit run app/review.py --server.port 8502", language="bash"
        )
        st.caption(
            "The local review form requires its generated blinded pack. Raw source data and private answer keys are not part of the public portfolio."
        )

elif page == "Methods":
    hero(
        "Small interfaces. Inspectable evidence.",
        "Source contracts, temporal boundaries and reproducible artifacts define the research system.",
        kicker="ANT-26 / METHODS & ARCHITECTURE",
    )
    st.subheader("The forecasting contract")
    st.write(
        "At reconstructed sector entry, forecast the upcoming sector's duration using a telemetry window that ends one second earlier. A 32-second, 4 Hz tensor includes values, usable masks and sample ages. Its grid does not increase the source's measurement frequency."
    )
    st.markdown(
        '<div class="ant-pipeline"><div><small>INGESTION</small>Read-only adapters<br>Checksummed snapshots</div><div><small>DATASETS</small>UTC alignment<br>Past-only transforms</div><div><small>MODELING</small>Four baselines<br>Masked patch encoder</div><div><small>EVALUATION</small>Whole-race splits<br>Calibration + bootstrap</div><div><small>SERVING</small>Immutable artifacts<br>Offline portfolio + API</div></div>',
        unsafe_allow_html=True,
    )
    first, second = st.columns(2)
    with first:
        card(
            "Learning",
            "Masked reconstruction combines continuous-channel regression with discrete brake and gear losses. Scratch, frozen and fine-tuned modes test the contribution of pretraining. Ordered quantile offsets prevent crossing.",
        )
        card(
            "Temporal isolation",
            "Fitting, tuning, calibration and testing use separate chronological race partitions. Future sector labels and whole-lap quality evidence cannot enter an earlier forecast.",
        )
    with second:
        card(
            "Operational accounting",
            "Origin eligibility and label validity remain separate. Surprise neutralizations stay in operational scoring when durations are valid; censored targets and abstentions remain counted.",
        )
        card(
            "Reproducibility",
            "Source checksums, feature-code hashes, split scopes, fixed seeds and saved model metadata make the experiment traceable. The public report contains safe aggregates.",
        )
    st.subheader("What the data can support")
    st.write(
        "The public telemetry supports sampled speed and control-channel comparisons. Position feeds approximate progress around the circuit and do not establish accurate lateral racing lines. Fuel, setup, tyre temperature and other unobserved conditions limit behavioral interpretation."
    )
    operations, operations_error = load_json(REPORT_ROOT / "operations.json")
    if operations_error:
        st.warning(operations_error)
    if isinstance(operations, dict):
        st.subheader("Measured local runtime")
        measured = operations.get("measurements", {})
        runtime = operations.get("hardware_runtime", {})
        baseline_ms = number(measured.get("baseline_predict_total"), "p95_ms")
        encoder_ms = number(measured.get("encoder_encode_total"), "p95_ms")
        metrics(
            [
                (
                    "CatBoost p95",
                    f"{baseline_ms:.3f} ms" if baseline_ms is not None else "—",
                    "Warm materialized batch-one prediction.",
                ),
                (
                    "Encoder p95",
                    f"{encoder_ms:.3f} ms" if encoder_ms is not None else "—",
                    "Continuous embedding only; excludes index search.",
                ),
                (
                    "Runtime CPU",
                    runtime.get("cpu", "—"),
                    "Local measurement hardware; not a service SLO.",
                ),
            ]
        )
        st.caption(
            "Warm synchronous calls with materialized inputs. Source alignment, network, HTTP, disk and model loading are excluded; this is not a service load test."
        )
        source_footer("reports/operations.json")
    st.subheader("Run the portfolio from a fresh checkout")
    st.code(
        "uv sync --extra app\nuv run --extra app streamlit run app/portfolio.py", language="bash"
    )
    st.caption(
        "The overview and development results need only the committed aggregate report. Local immutable case artifacts add detailed replay; the separate API-backed viewer remains in app/replay.py."
    )
    if development:
        with st.expander("Report provenance and declared limitations"):
            st.json(
                {
                    key: development[key]
                    for key in (
                        "schema_version",
                        "sources",
                        "exclusions_and_limits",
                        "interpretation",
                        "wall_time_note",
                    )
                    if key in development
                }
            )
        st.download_button(
            "Download aggregate development results",
            json.dumps(development, sort_keys=True, indent=2),
            file_name="ant26-development-results.json",
            mime="application/json",
        )

st.divider()
st.caption(
    "ANT-26 · Kimi Antonelli, 2026 · Research software and evidence, with uncertainty kept visible."
)
