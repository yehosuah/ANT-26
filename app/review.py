"""Blinded human retrieval review. This application never reads the answer key."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ANT-26 atlas review", layout="wide")
st.title("ANT-26 atlas review")
st.write(
    "Compare observed driving execution in the query with sets A and B. Account for the conditions shown. Choose A, B, tie or uncertain; no rating is generated automatically."
)

path = Path(
    os.environ.get("ANT26_REVIEW_PATH", "artifacts/atlas-review-ant26-v2-seed17/blind-review.json")
)
if path.parent.name == "private" or path.name == "answer-key.json":
    st.error("Select the blinded review pack, not its private assignment key.")
    st.stop()
try:
    blind = json.loads(path.read_text())
except (OSError, ValueError):
    st.info("The corrected blinded review pack is not available yet.")
    st.stop()
if blind.get("schema_version") != "ant26-blind-atlas-review-v1":
    st.error("This file is not a blinded atlas review pack.")
    st.stop()
digest = hashlib.sha256(
    json.dumps(
        {key: value for key, value in blind.items() if key != "review_id"},
        sort_keys=True,
        allow_nan=False,
    ).encode()
).hexdigest()
if digest != blind.get("review_id"):
    st.error("The review pack no longer matches its recorded checksum.")
    st.stop()
queries = blind.get("queries", [])
if not queries:
    st.info("This review contains no queries.")
    st.stop()

reviewer = st.text_input("Reviewer identifier", placeholder="Use a short alias")
valid_reviewer = bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", reviewer))
answer_root = Path(os.environ.get("ANT26_REVIEW_ANSWER_ROOT", "artifacts/atlas-review-answers"))
answer_path = answer_root / blind["review_id"] / f"{reviewer}.json" if valid_reviewer else None
responses = {
    "review_id": blind["review_id"],
    "reviewer_id": reviewer if valid_reviewer else None,
    "responses": [{"query_id": row["query_id"], "choice": None, "notes": ""} for row in queries],
}
if answer_path is not None and answer_path.exists():
    try:
        saved = json.loads(answer_path.read_text())
        if saved.get("review_id") != blind["review_id"] or saved.get("reviewer_id") != reviewer:
            raise ValueError("response identity mismatch")
        known = {row["query_id"]: row for row in saved.get("responses", [])}
        for row in responses["responses"]:
            if row["query_id"] in known:
                existing = known[row["query_id"]]
                if existing.get("choice") not in (None, "A", "B", "tie", "uncertain", "neither"):
                    raise ValueError("invalid saved choice")
                row.update(choice=existing.get("choice"), notes=existing.get("notes", ""))
    except (OSError, ValueError):
        st.error(
            "Saved responses cannot be read safely. Use a new reviewer alias or repair that response file."
        )
        st.stop()

ratings = {row["query_id"]: row for row in responses["responses"]}
rated = sum(row["choice"] is not None for row in ratings.values())
st.caption(
    f"{rated} of {len(queries)} queries rated. Ratings are saved only when you press Save response."
)
if any(str(kind).startswith("synthetic") for kind in blind.get("data_kinds", [])):
    st.warning("Synthetic review fixture; this is not evidence about Antonelli.")
query_ids = [row["query_id"] for row in queries]
selected_id = st.selectbox("Review query", query_ids)
selected = next(row for row in queries if row["query_id"] == selected_id)
query = selected["query"]


def identity(window):
    return f"Session {window['session_key']} · Driver {window['driver_number']} · Lap {window['lap_number']} · Sector {window['sector_index']}"


def trace(window, channel):
    channels = window.get("sequence_channels", [])
    if channel not in channels:
        return []
    index = channels.index(channel)
    values, masks = window.get("sequence", []), window.get("sequence_mask", [])
    return [
        row[index] if i < len(masks) and masks[i][index] else None for i, row in enumerate(values)
    ]


def conditions(window):
    return {
        "source": window.get("example_id"),
        "cutoff": window.get("cutoff"),
        **window.get("context", {}),
    }


st.subheader("Query")
st.caption(identity(query))
st.json(conditions(query), expanded=False)
st.caption(
    "Each trace is the observed window ending at its own cutoff. Gaps remain masked. These traces do not show the upcoming target sector."
)
available = min(len(selected["candidates"].get(label, [])) for label in ("A", "B"))
if available < 1:
    st.error("The query does not have two comparable candidate sets.")
    st.stop()
rank = st.selectbox("Neighbor rank", list(range(1, available + 1))) if available > 1 else 1
for column, label in zip(st.columns(2), ("A", "B"), strict=True):
    with column:
        st.subheader(f"Set {label}, neighbor {rank}")
        neighbor = selected["candidates"][label][rank - 1]["observed"]
        st.caption(identity(neighbor))
        st.json(conditions(neighbor), expanded=False)
        for channel, unit in (("speed", "km/h"), ("throttle", "%"), ("brake", "application")):
            query_trace, neighbor_trace = trace(query, channel), trace(neighbor, channel)
            st.caption(f"{channel.title()} ({unit})")
            if query_trace and neighbor_trace and len(query_trace) == len(neighbor_trace):
                st.line_chart({"Query": query_trace, f"Set {label}": neighbor_trace}, height=150)
            else:
                st.info("This channel is unavailable for comparison.")

options = ["Unrated", "A", "B", "Tie", "Uncertain"]
to_choice = {"Unrated": None, "A": "A", "B": "B", "Tie": "tie", "Uncertain": "uncertain"}
to_display = {value: key for key, value in to_choice.items()}
prior = ratings[selected_id]
prior_display = to_display.get(prior["choice"], "Uncertain")
choice = st.radio(
    "Which set better matches the query's observed execution under its conditions?",
    options,
    index=options.index(prior_display),
    key=f"rating-{blind['review_id']}-{reviewer}-{selected_id}",
)
notes = st.text_area(
    "Notes", value=prior["notes"], key=f"notes-{blind['review_id']}-{reviewer}-{selected_id}"
)
if st.button("Save response"):
    if not valid_reviewer:
        st.error(
            "Enter a reviewer alias using letters, digits, underscores or hyphens before saving."
        )
    else:
        ratings[selected_id].update(choice=to_choice[choice], notes=notes)
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(responses, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", dir=answer_path.parent, delete=False) as output:
            output.write(serialized)
            temporary = Path(output.name)
        os.replace(temporary, answer_path)
        st.success("Response saved. Unrated queries remain blank.")
