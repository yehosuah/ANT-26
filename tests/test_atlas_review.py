import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ant26.atlas.review import bootstrap_cluster_stability, build_review_set, score_review_responses
from ant26.demo import synthetic_examples
from ant26.domain.protocol import load_protocol, protocol_hash
from ant26.evaluation.splits import rolling_splits
from ant26.models.baselines import valid_examples


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class FixtureBundle:
    def __init__(self, references, fit_sessions, calibration_sessions, protocol):
        self.metadata = {
            "seed": 17,
            "data_kind": "synthetic_integration_fixture",
            "fit_sessions": sorted(fit_sessions),
            "calibration_sessions": calibration_sessions,
            "fitting_examples_sha256": digest(references),
            "protocol_sha256": protocol_hash(protocol),
            "checkpoint_sha256": "0" * 64,
            "normalization_policy": "constant_unit_scale_v2",
        }

    def encode(self, rows):
        return np.asarray(
            [[row["features"]["speed_mean"], row["features"]["throttle_mean"]] for row in rows]
        )


def fixtures():
    protocol = load_protocol()
    examples, catalog = synthetic_examples(protocol)
    specs, bundles = [], {}
    for fold in rolling_splits(catalog, protocol):
        fitting = set(fold["fit_sessions"]) | set(fold["tune_sessions"])
        references = valid_examples([row for row in examples if row["session_key"] in fitting])
        path = fold["fold_id"]
        specs.append(
            {
                "fold_id": path,
                "bundle_path": path,
                "fit_sessions": sorted(fitting),
                "test_sessions": fold["test_sessions"],
            }
        )
        bundles[path] = FixtureBundle(references, fitting, fold["calibration_sessions"], protocol)
    return protocol, examples, catalog, specs, bundles


def test_review_is_blind_reproducible_and_has_no_prefilled_human_responses(tmp_path):
    protocol, examples, catalog, specs, bundles = fixtures()
    with patch("ant26.atlas.review._load_bundle", side_effect=lambda path: bundles[path]):
        result = build_review_set(
            examples, catalog, specs, tmp_path, protocol=protocol, query_count=4, minimum_races=2
        )
        repeated = build_review_set(
            examples, catalog, specs, tmp_path, protocol=protocol, query_count=4, minimum_races=2
        )
    assert result == repeated
    assert result["human_responses"] == 0
    blind = json.loads(Path(result["paths"]["blind_review"]).read_text())
    key = json.loads(Path(result["paths"]["private_answer_key"]).read_text())
    responses = json.loads(Path(result["paths"]["response_template"]).read_text())
    assert blind["query_count"] == 4
    assert blind["race_count"] == 2
    assert blind["status"] == "awaiting_human_review"
    assert blind["data_kinds"] == ["synthetic_integration_fixture"]
    serialized = json.dumps(blind)
    assert "feature_pca" not in serialized
    assert "checkpoint_sha256" not in serialized
    assert '"distance"' not in serialized
    assert all(response["choice"] is None for response in responses["responses"])
    assert key["review_id"] == blind["review_id"]
    scored = score_review_responses(blind, key, [])
    assert scored["status"] == "awaiting_human_review"
    assert scored["learned_preference_fraction_including_ties"] is None
    assert scored["validated"] is False


def test_reference_snapshot_or_scope_mismatch_is_rejected(tmp_path):
    protocol, examples, catalog, specs, bundles = fixtures()
    changed = copy.deepcopy(examples)
    reference_id = valid_examples(examples)[0]["example_id"]
    next(row for row in changed if row["example_id"] == reference_id)["features"]["speed_mean"] = (
        499
    )
    with patch("ant26.atlas.review._load_bundle", side_effect=lambda path: bundles[path]):
        with pytest.raises(ValueError, match="snapshot"):
            build_review_set(
                changed, catalog, specs, tmp_path, protocol=protocol, query_count=4, minimum_races=2
            )
        bad_specs = copy.deepcopy(specs)
        bad_specs[0]["fit_sessions"].append(bad_specs[0]["test_sessions"][0])
        with pytest.raises(ValueError, match="scope"):
            build_review_set(
                examples,
                catalog,
                bad_specs,
                tmp_path,
                protocol=protocol,
                query_count=4,
                minimum_races=2,
            )


def test_mexico_or_insufficient_race_review_cannot_be_prepared(tmp_path):
    protocol, examples, catalog, specs, _ = fixtures()
    changed = copy.deepcopy(examples)
    changed[0]["meeting_key"] = 1298
    with pytest.raises(ValueError, match="Mexico"):
        build_review_set(changed, catalog, specs, tmp_path, protocol=protocol)
    with pytest.raises(ValueError, match="insufficient"):
        build_review_set(examples, catalog, specs, tmp_path, protocol=protocol)


def test_legacy_normalization_checkpoint_cannot_prepare_a_review(tmp_path):
    protocol, examples, catalog, specs, bundles = fixtures()
    bundles[specs[0]["bundle_path"]].metadata.pop("normalization_policy")
    with patch("ant26.atlas.review._load_bundle", side_effect=lambda path: bundles[path]):
        with pytest.raises(ValueError, match="legacy normalization"):
            build_review_set(
                examples,
                catalog,
                specs,
                tmp_path,
                protocol=protocol,
                query_count=4,
                minimum_races=2,
            )


def test_only_supplied_response_choices_are_scored_and_tampering_rejected(tmp_path):
    protocol, examples, catalog, specs, bundles = fixtures()
    with patch("ant26.atlas.review._load_bundle", side_effect=lambda path: bundles[path]):
        result = build_review_set(
            examples, catalog, specs, tmp_path, protocol=protocol, query_count=4, minimum_races=2
        )
    blind = json.loads(Path(result["paths"]["blind_review"]).read_text())
    key = json.loads(Path(result["paths"]["private_answer_key"]).read_text())
    responses = {
        "review_id": blind["review_id"],
        "reviewer_id": "fixture-reviewer",
        "responses": [
            {"query_id": "query-001", "choice": "A"},
            {"query_id": "query-002", "choice": "tie"},
        ],
    }
    scored = score_review_responses(blind, key, [responses])
    assert scored["response_count"] == 2
    assert scored["counts"][key["assignments"][0]["A"]] == 1
    assert scored["counts"]["tie"] == 1
    assert scored["validated"] is False
    assert scored["status"] == "exploratory_single_reviewer"
    changed = copy.deepcopy(blind)
    changed["query_count"] = 60
    with pytest.raises(ValueError, match="checksum"):
        score_review_responses(changed, key, [responses])
    changed_key = copy.deepcopy(key)
    changed_key["assignments"][0]["A"] = "tampered"
    with pytest.raises(ValueError, match="key content.*checksum"):
        score_review_responses(blind, changed_key, [responses])


def test_cluster_stability_requires_disjoint_scope_and_sufficient_races():
    protocol, examples, catalog, _, _ = fixtures()
    fitting = valid_examples([row for row in examples if row["session_key"] == 1000])
    probing = valid_examples([row for row in examples if row["session_key"] == 1001])

    def matrix(rows):
        return np.asarray(
            [[row["features"]["speed_mean"], row["features"]["throttle_mean"]] for row in rows]
        )

    result = bootstrap_cluster_stability(
        fitting,
        matrix(fitting),
        probing,
        matrix(probing),
        fit_sessions={1000},
        probe_sessions={1001},
    )
    assert result["status"] == "insufficient_samples"
    assert result["validated"] is False
    with pytest.raises(ValueError, match="disjoint"):
        bootstrap_cluster_stability(
            fitting,
            matrix(fitting),
            fitting,
            matrix(fitting),
            fit_sessions={1000},
            probe_sessions={1000},
        )
