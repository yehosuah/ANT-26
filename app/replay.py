"""Thin UI over saved artifacts; no training or provider access occurs here."""

import os

import httpx
import streamlit as st

st.set_page_config(page_title="ANT-26", layout="wide")
st.title("ANT-26")
st.write("Inspect sector forecasts and the evidence behind them.")
api = os.environ.get("ANT26_API_URL", "http://127.0.0.1:8000")


def get(path: str):
    response = httpx.get(f"{api}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


try:
    runs = get("/runs")["runs"]
except httpx.HTTPError:
    st.error("Start the local ANT-26 API to inspect saved runs.")
    st.stop()

if not runs:
    st.info("No saved runs. Run the offline smoke pipeline or a development benchmark first.")
    st.stop()

run = st.selectbox("Saved run", runs)
report = get(f"/runs/{run}/report")
if report.get("synthetic"):
    st.warning("Synthetic integration fixture. These results are not evidence about Antonelli.")
st.caption(f"Evaluation status: {report.get('status', 'unknown')}")
if report.get("claims") and not any(
    row.get("improvement_claim_allowed", False) for row in report["claims"].values()
):
    st.info(
        "Improvement gates failed. The baseline is retained for forecasting; these candidate results remain available for research inspection."
    )
entries = get(f"/runs/{run}/prediction-index?driver_number=12")["entries"]
if not entries:
    st.info("This run has no eligible Antonelli forecasts. Inspect the quality report.")
    st.json(report)
    st.stop()

model = st.selectbox("Model", sorted({r["model"] for r in entries}))
session = st.selectbox(
    "Race session", sorted({r["session_key"] for r in entries if r["model"] == model})
)
seeds = sorted(
    {
        r["seed"]
        for r in entries
        if r["model"] == model and r["session_key"] == session and r.get("seed") is not None
    }
)
seed = st.selectbox("Training seed", seeds) if seeds else None
seed_query = f"&seed={seed}" if seed is not None else ""
selected = []
while True:
    page = get(
        f"/runs/{run}/predictions?driver_number=12&session_key={session}&model={model}{seed_query}&offset={len(selected)}&limit=5000"
    )
    selected.extend(page["predictions"])
    if len(selected) >= page["count"] or not page["predictions"]:
        break
selected.sort(key=lambda r: r["origin"])
st.caption(
    "Scored forecasts only. Coverage diagnostics include abstentions and censored targets; individual training seeds are viewed separately."
)
st.line_chart(
    {
        "actual_seconds": [r["target_seconds"] for r in selected],
        "median_seconds": [r["quantiles"][2] for r in selected],
        "lower_90": [r["quantiles"][0] for r in selected],
        "upper_90": [r["quantiles"][4] for r in selected],
    }
)
choice = st.slider("Forecast", 0, len(selected) - 1, 0) if len(selected) > 1 else 0
st.json(selected[choice])
with st.expander("Evaluation and provenance"):
    st.json({key: value for key, value in report.items() if key != "predictions"})
with st.expander("Behavioral atlas"):
    try:
        result = get(f"/runs/{run}/atlas")
        st.caption(
            "Similarity under the saved representation; patterns do not establish driver intention."
        )
        st.json(result)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            st.info("No atlas query has been saved for this run.")
        else:
            raise
with st.expander("Error and coverage slices"):
    try:
        st.json(get(f"/runs/{run}/diagnostics"))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            st.info("No diagnostics have been saved for this run.")
        else:
            raise
st.caption(
    "Historical event-time replay; source publication latency was not observed. Mexico remains reserved until model freeze."
)
