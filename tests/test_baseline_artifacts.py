"""Baseline persistence is explicit data, never pickle-executable Python objects."""

import json
from pathlib import Path

import numpy as np
import pytest

from ant26.demo import synthetic_examples
from ant26.models.artifacts import BaselineBundle, final_candidate_partitions, fit_final_candidate
from ant26.models.baselines import Baseline, valid_examples


def rows(session, count=12):
    return [
        {
            "example_id": f"{session}-{i}",
            "session_key": session,
            "driver_number": 12,
            "lap_number": i + 3,
            "sector_index": 1,
            "origin": f"2026-01-{session:02d}T12:00:00+00:00",
            "cutoff": f"2026-01-{session:02d}T11:59:59+00:00",
            "target_seconds": 30 + (i % 5) / 10,
            "reference_seconds": 30,
            "features": {"last_same_sector_seconds": 30, "speed_mean": 150 + i},
            "eligible": True,
            "target_valid": True,
            "provenance": {},
        }
        for i in range(count)
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "historical_reference",
        "same_sector_persistence",
        "regularized_linear_residual",
        "catboost_quantile",
    ],
)
def test_baseline_bundle_roundtrip_without_pickle(tmp_path, kind):
    if kind == "catboost_quantile":
        pytest.importorskip("catboost")
    model = (
        Baseline(kind).fit(rows(1), allowed_sessions={1}).calibrate(rows(2), allowed_sessions={2})
    )
    bundle = BaselineBundle(model, {"status": "candidate_not_frozen_for_mexico"})
    before = bundle.predict(rows(3))
    bundle.save(tmp_path)
    loaded = BaselineBundle.load(tmp_path)
    np.testing.assert_allclose(loaded.predict(rows(3)), before)
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pickle"))
    assert loaded.metadata["serialization"] == "validated_json_and_native_catboost_no_pickle"
    with pytest.raises(FileExistsError):
        bundle.save(tmp_path)


def test_corrupt_state_and_metadata_are_rejected(tmp_path):
    model = Baseline("historical_reference").fit(rows(1), allowed_sessions={1})
    BaselineBundle(model).save(tmp_path)
    path = tmp_path / "state.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        BaselineBundle.load(tmp_path)
    metadata = tmp_path / "metadata.json"
    value = json.loads(metadata.read_text())
    value["status"] = "fake_frozen"
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata checksum"):
        BaselineBundle.load(tmp_path)


def test_corrupt_native_tree_is_rejected_before_deserialization(tmp_path):
    pytest.importorskip("catboost")
    model = (
        Baseline("catboost_quantile")
        .fit(rows(1), allowed_sessions={1})
        .calibrate(rows(2), allowed_sessions={2})
    )
    BaselineBundle(model).save(tmp_path)
    path = tmp_path / "quantile-2.cbm"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        BaselineBundle.load(tmp_path)


def test_final_candidate_scopes_are_first16_refit_last2_calibration(tmp_path):
    pytest.importorskip("catboost")
    protocol = json.loads(
        (Path(__file__).parents[1] / "configs" / "evaluation-protocol.json").read_text()
    )
    examples, catalog = synthetic_examples(protocol, race_count=18)
    partition = final_candidate_partitions(catalog, protocol)
    assert len(partition["fit_meetings"]) == 16
    assert len(partition["calibration_meetings"]) == 2
    assert not set(partition["fit_sessions"]).intersection(partition["calibration_sessions"])
    candidate = fit_final_candidate(
        examples, catalog, protocol, data_kind="synthetic_final_candidate_fixture"
    )
    assert candidate.metadata["status"] == "candidate_not_frozen_for_mexico"
    assert candidate.metadata["mexico_evaluated"] is False
    assert candidate.metadata["seed"] == 17
    assert candidate.metadata["fit_sessions"] == list(range(1000, 1016))
    assert candidate.metadata["calibration_sessions"] == [1016, 1017]
    candidate.save(tmp_path)
    sample = valid_examples(examples)[-5:]
    np.testing.assert_allclose(
        BaselineBundle.load(tmp_path).predict(sample), candidate.predict(sample)
    )
    with pytest.raises(ValueError, match="Mexico"):
        fit_final_candidate(examples + [{"session_key": 11404}], catalog, protocol)
