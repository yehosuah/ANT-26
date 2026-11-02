import copy
import hashlib
import json

import pytest

from ant26.domain.protocol import load_protocol, protocol_hash
from ant26.revisions import feature_revision_details


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def revision(tmp_path):
    protocol = load_protocol()
    new_protocol = copy.deepcopy(protocol)
    new_protocol["protocol_version"] = "0.2.0"
    new_protocol["prediction"]["pit_timestamp_policy"] = "completion_report_v2"
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    encoder = {
        "encoder_config": {"width": 128},
        "seed": 17,
        "mode": "finetune",
        "pretrain_epochs": 3,
        "forecast_epochs": 5,
        "normalization_policy": "constant_unit_scale_v2",
        "numeric_features": ["speed_mean"],
        "sequence_channels": ["speed"],
    }
    baseline = {
        "kind": "catboost_quantile",
        "seed": 17,
        "numeric_features": ["speed_mean"],
        "quantiles": [0.05, 0.1, 0.5, 0.9, 0.95],
        "baseline_code_sha256": "unchanged-baseline-code",
    }
    files = {
        "encoder/metadata.json": save(old_root / "encoder/metadata.json", encoder),
        "baseline/metadata.json": save(old_root / "baseline/metadata.json", baseline),
        "index.json": save(old_root / "index.json", {"old_index": True}),
    }
    common = {
        "status": "frozen_for_case_study",
        "fit_sessions": [1],
        "calibration_sessions": [2],
        "cohort_driver_numbers": [12, 63],
        "prediction_candidate": "catboost_quantile",
        "atlas_decision": {"human_rating_count": 0},
        "source_files": {"models/representation.py": "unchanged-training-source"},
    }
    parent = {
        **common,
        "protocol_sha256": protocol_hash(protocol),
        "candidate_root": "old",
        "artifact_files": files,
        "training_dataset": {"examples_sha256": "old-data"},
        "index_hash": "old-index",
    }
    parent["manifest_sha256"] = sha(parent)
    parent_file = tmp_path / "parent-freeze.json"
    parent_digest = save(parent_file, parent)
    report = {
        "status": "case_study_scored",
        "race": {"session_key": 11404},
        "accounting": {"scoreable_forecasts": 115},
        "provenance": {
            "freeze_sha256": parent_digest,
            "release_manifest_sha256": parent["manifest_sha256"],
        },
    }
    report_file = tmp_path / "parent-report.json"
    benchmark = {
        "input_sha256": "new-data",
        "protocol_sha256": protocol_hash(new_protocol),
        "synthetic": False,
        "evaluated_fold_count": 8,
    }
    benchmark_file = tmp_path / "benchmark.json"
    baseline["selection"] = {
        "evidence_status": "development_report_verified",
        "development_report_sha256": sha(benchmark),
    }
    save(new_root / "baseline/metadata.json", baseline)
    save(new_root / "encoder/metadata.json", encoder)
    prior_protocol_file = tmp_path / "prior-protocol.json"
    evidence = tmp_path / "evidence.md"
    evidence.write_text("Primary evidence and interval availability rule.")
    record = {"description": "Explicit pit timestamp policy change with development refitting."}
    for name, path, digest in (
        ("parent_freeze", parent_file, parent_digest),
        ("parent_report", report_file, save(report_file, report)),
        ("parent_protocol", prior_protocol_file, save(prior_protocol_file, protocol)),
        ("development_report", benchmark_file, save(benchmark_file, benchmark)),
        ("evidence", evidence, hashlib.sha256(evidence.read_bytes()).hexdigest()),
    ):
        record[f"{name}_path"] = path.name
        record[f"{name}_sha256"] = digest
    record_path = tmp_path / "revision.json"
    save(record_path, record)
    current = {
        **common,
        "protocol_sha256": protocol_hash(new_protocol),
        "candidate_root": "new",
        "artifact_files": {"new-model": "new-bytes"},
        "training_dataset": {"examples_sha256": "new-data"},
        "index_hash": "new-index",
    }
    return record_path, current, tmp_path / "new-freeze.json", new_protocol


def test_feature_revision_preserves_parent_but_reports_refitting(revision):
    result = feature_revision_details(*revision)
    assert result["parent_scoreable_forecasts"] == 115
    assert result["post_inspection"] is True
    assert result["forecast_artifacts_unchanged"] is False
    assert result["development_examples_unchanged"] is False
    assert result["model_recipe_unchanged"] is True


def test_feature_revision_rejects_unrelated_protocol_change(revision):
    path, release, output, protocol = revision
    protocol["prediction"]["feature_guard_seconds"] = 0.01
    with pytest.raises(ValueError, match="only the declared"):
        feature_revision_details(path, release, output, protocol)


def test_feature_revision_rejects_changed_parent_bytes(revision):
    path, release, output, protocol = revision
    (path.parent / "old/index.json").write_text("changed")
    with pytest.raises(ValueError, match="parent model/index bytes"):
        feature_revision_details(path, release, output, protocol)


@pytest.mark.parametrize(
    "field,value",
    [("seed", 29), ("forecast_epochs", 10), ("normalization_policy", "another-policy")],
)
def test_feature_revision_rejects_changed_training_recipe(revision, field, value):
    path, release, output, protocol = revision
    target = path.parent / "new/encoder/metadata.json"
    metadata = json.loads(target.read_text())
    metadata[field] = value
    save(target, metadata)
    with pytest.raises(ValueError, match="encoder recipe"):
        feature_revision_details(path, release, output, protocol)


def test_feature_revision_requires_new_data_benchmark(revision):
    path, release, output, protocol = revision
    release["training_dataset"]["examples_sha256"] = "wrong-data"
    with pytest.raises(ValueError, match="rebuilt development data"):
        feature_revision_details(path, release, output, protocol)


def test_feature_revision_refuses_unpinned_evidence(revision):
    path, release, output, protocol = revision
    (path.parent / "evidence.md").write_text("different evidence")
    with pytest.raises(ValueError, match="evidence checksum"):
        feature_revision_details(path, release, output, protocol)
