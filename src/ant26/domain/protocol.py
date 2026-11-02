"""Validate the evaluation contract before any pipeline stage."""

import hashlib
import json
from pathlib import Path
from typing import Any


def load_protocol(path: str | Path = "configs/evaluation-protocol.json") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    prediction = payload["prediction"]
    if prediction["feature_guard_seconds"] <= 0:
        raise ValueError("Prediction features require a positive timing guard")
    if prediction["quantiles"] != [0.05, 0.10, 0.50, 0.90, 0.95]:
        raise ValueError("Protocol must declare the supported five quantiles")
    if payload["final_case_study"]["meeting_key"] != 1298:
        raise ValueError("Mexico meeting 1298 must remain the final case study")
    required = {"pretraining", "model_fitting", "calibration", "retrieval_index_construction"}
    if not required.issubset(payload["final_case_study"]["exclude_from"]):
        raise ValueError("Mexico exclusion is incomplete")
    return payload


def protocol_hash(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
