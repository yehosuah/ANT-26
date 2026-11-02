"""A separately recorded, fixed 98% coverage audit of the behavior comparison.

The stricter coverage threshold was requested before observations were measured
but missed in the primary orchestration. Executing it after the primary report
is an explicit amendment, not another untouched confirmatory analysis. Existing
primary artifacts remain unchanged. No threshold search or new metric is used.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ant26.research import comparison


def run_additional_audit(observations: dict | list[dict] | str | Path) -> dict:
    """Return safe aggregates for the fixed coverage audit; never write inputs."""
    if isinstance(observations, (str, Path)):
        raw = Path(observations).read_bytes()
        envelope = json.loads(raw)
    else:
        envelope = observations
        raw = json.dumps(envelope, sort_keys=True, allow_nan=False).encode()
    rows = comparison._rows(envelope)
    result = comparison.matched_comparison(rows, minimum_coverage=0.98, seed=17, resamples=10000)
    result.pop("matched_cells")
    return {
        "schema_version": "ant26-driver-behavior-additional-audit-v1",
        "aggregate_only": True,
        "analysis_kind": "retrospective_observational_quality_sensitivity_amendment",
        "drivers": {"antonelli": 12, "russell": 63},
        "mexico_included_in_matching": False,
        "amendment": {
            "minimum_native_coverage": 0.98,
            "request_timing": "requested_before_measurements",
            "execution_timing": "executed_after_primary_analysis",
            "reason": "Previously requested stricter coverage check omitted from primary orchestration.",
            "threshold_search_performed": False,
            "primary_outputs_replaced": False,
            "matching_and_metrics_changed": False,
            "bootstrap": {"unit": "complete_race", "seed": 17, "resamples": 10000},
        },
        "coverage_098": result,
        "provenance": {
            "observations_sha256": hashlib.sha256(raw).hexdigest(),
            "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "comparison_code_sha256": hashlib.sha256(
                Path(comparison.__file__).read_bytes()
            ).hexdigest(),
            "observation_metadata": copy.deepcopy(envelope.get("metadata", {}))
            if isinstance(envelope, dict)
            else {},
        },
        "interpretation": (
            "This fixed quality check supplements the preserved primary analysis. "
            "Matched-cohort sensor fractions do not establish intent, causality, "
            "pace superiority or a car-independent driving signature."
        ),
        "limitations": [
            "The analysis is retrospective and the audit was executed after primary results were visible.",
            "The three metric intervals use complete-race blocks and Bonferroni adjustment; sensitivities are descriptive robustness checks.",
            "Selective source coverage and unrecorded fuel, setup, traffic and race instructions can remain confounders.",
            "An interval close to zero supports cautious finite-cohort wording rather than a universal driver-style claim.",
        ],
    }
