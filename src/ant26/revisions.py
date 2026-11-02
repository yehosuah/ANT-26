"""Evidence-bound feature-policy revisions, separate from unchanged-model repairs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ant26.domain.protocol import protocol_hash

FEATURE_REVISION_CLAIM = "post_inspection_feature_policy_revision_diagnostic_replay"


def _read(path):
    return json.loads(Path(path).read_text())


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def feature_revision_details(record_path, release, freeze_path, protocol):
    """Require a preserved parent and a separately rebuilt development recipe.

    This path permits changed development rows, normalization and weights. It
    cannot pretend to be the unchanged-artifact implementation repair used by
    the earlier release. Only the declared pit-policy/version delta is allowed.
    """
    record_path, freeze_path = Path(record_path).resolve(), Path(freeze_path).resolve()
    record = _read(record_path)
    if not isinstance(record.get("description"), str) or not record["description"].strip():
        raise ValueError("Feature revision requires an evidence-based description")
    pinned = {}
    for name in (
        "parent_freeze",
        "parent_report",
        "parent_protocol",
        "evidence",
        "development_report",
    ):
        path_value, digest = record.get(f"{name}_path"), record.get(f"{name}_sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise ValueError(f"Feature revision requires pinned {name}")
        path = (record_path.parent / path_value).resolve()
        if not path.is_file() or _file_sha(path) != digest:
            raise ValueError(f"Feature revision {name} checksum mismatch")
        pinned[name] = path
    parent, report, prior_protocol = (
        _read(pinned[name]) for name in ("parent_freeze", "parent_report", "parent_protocol")
    )
    unsigned = {key: value for key, value in parent.items() if key != "manifest_sha256"}
    if (
        parent.get("manifest_sha256") != _sha(unsigned)
        or parent.get("status") != "frozen_for_case_study"
    ):
        raise ValueError("Feature revision parent freeze is invalid")
    if (
        report.get("race", {}).get("session_key") != 11404
        or report.get("status") != "case_study_scored"
    ):
        raise ValueError("Feature revision requires its preserved Mexico parent report")
    provenance = report.get("provenance", {})
    if (
        provenance.get("freeze_sha256") != record["parent_freeze_sha256"]
        or provenance.get("release_manifest_sha256") != parent["manifest_sha256"]
    ):
        raise ValueError("Feature revision parent report is not bound to its freeze")
    if protocol_hash(prior_protocol) != parent["protocol_sha256"]:
        raise ValueError("Feature revision parent protocol differs from the frozen protocol")
    old, new = copy.deepcopy(prior_protocol), copy.deepcopy(protocol)
    old.pop("protocol_version", None)
    new.pop("protocol_version", None)
    old_policy = old["prediction"].pop("pit_timestamp_policy", "entry_at_date_v1")
    new_policy = new["prediction"].pop("pit_timestamp_policy", None)
    if old_policy != "entry_at_date_v1" or new_policy != "completion_report_v2" or old != new:
        raise ValueError(
            "Feature revision allows only the declared pit timestamp policy and version change"
        )
    for key in (
        "fit_sessions",
        "calibration_sessions",
        "cohort_driver_numbers",
        "prediction_candidate",
        "atlas_decision",
    ):
        if parent.get(key) != release.get(key):
            raise ValueError(f"Feature revision must preserve {key}")
    if parent["protocol_sha256"] == release["protocol_sha256"]:
        raise ValueError("A feature policy revision needs a new protocol identity")
    parent_candidate = (pinned["parent_freeze"].parent / parent["candidate_root"]).resolve()
    candidate = (freeze_path.parent / release["candidate_root"]).resolve()
    # Parent bytes remain available and untouched, independently of current code.
    for name, digest in parent["artifact_files"].items():
        path = parent_candidate / name
        if path.is_symlink() or not path.is_file() or _file_sha(path) != digest:
            raise ValueError("Feature revision parent model/index bytes changed")
    left, right = (
        _read(parent_candidate / "encoder/metadata.json"),
        _read(candidate / "encoder/metadata.json"),
    )
    for name in (
        "encoder_config",
        "seed",
        "mode",
        "pretrain_epochs",
        "forecast_epochs",
        "normalization_policy",
        "numeric_features",
        "sequence_channels",
    ):
        if left.get(name) != right.get(name):
            raise ValueError(f"Feature revision changed the encoder recipe: {name}")
    old_base, new_base = (
        _read(parent_candidate / "baseline/metadata.json"),
        _read(candidate / "baseline/metadata.json"),
    )
    for name in ("kind", "seed", "numeric_features", "quantiles", "baseline_code_sha256"):
        if old_base.get(name) != new_base.get(name):
            raise ValueError(f"Feature revision changed the baseline recipe: {name}")
    if (
        parent["source_files"]["models/representation.py"]
        != release["source_files"]["models/representation.py"]
    ):
        raise ValueError("Feature revision changed representation training code")
    benchmark = _read(pinned["development_report"])
    if benchmark.get("input_sha256") != release["training_dataset"]["examples_sha256"]:
        raise ValueError("Feature revision benchmark does not match the rebuilt development data")
    if (
        benchmark.get("protocol_sha256") != release["protocol_sha256"]
        or benchmark.get("synthetic") is not False
        or benchmark.get("evaluated_fold_count") != 8
    ):
        raise ValueError(
            "Feature revision requires the complete real eight-fold development benchmark"
        )
    selection = new_base.get("selection", {})
    if selection.get("evidence_status") != "development_report_verified" or selection.get(
        "development_report_sha256"
    ) != _sha(benchmark):
        raise ValueError(
            "Feature revision candidate selection does not match the new development benchmark"
        )
    return {
        "record_path": os.path.relpath(record_path, freeze_path.parent),
        "record_sha256": _file_sha(record_path),
        "description": record["description"],
        **{f"{name}_sha256": record[f"{name}_sha256"] for name in pinned},
        "post_inspection": True,
        "old_pit_policy": old_policy,
        "new_pit_policy": new_policy,
        "parent_scoreable_forecasts": report["accounting"]["scoreable_forecasts"],
        "forecast_artifacts_unchanged": parent["artifact_files"] == release["artifact_files"],
        "development_examples_unchanged": parent["training_dataset"]["examples_sha256"]
        == release["training_dataset"]["examples_sha256"],
        "new_development_benchmark_verified": True,
        "model_recipe_unchanged": True,
    }
