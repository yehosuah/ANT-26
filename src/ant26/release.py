"""Immutable release decisions and offline scoring of the predeclared case study.

This module never fetches source data or fits a model. Experimental retrieval is
explicitly unvalidated when no human preferences exist; a freeze records that
decision rather than inventing validation or changing candidate metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ant26.atlas import AtlasIndex
from ant26.domain.protocol import protocol_hash
from ant26.evaluation.metrics import macro_metrics
from ant26.evaluation.splits import MEXICO_SESSIONS, utc, validate_scope
from ant26.models.artifacts import BaselineBundle
from ant26.models.baselines import valid_examples

FREEZE_SCHEMA = "ant26-release-freeze-v1"
REVIEW_DECISION = "exploratory_unvalidated_no_human_ratings"


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(value: str | Path | dict) -> dict:
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to replace immutable release artifact: {path.name}")
        return
    with path.open("x") as handle:
        handle.write(payload)


def _code_files() -> dict[str, Path]:
    package = Path(__file__).parent
    names = (
        "release.py",
        "datasets/features.py",
        "models/baselines.py",
        "models/artifacts.py",
        "models/representation.py",
        "atlas/index.py",
        "evaluation/metrics.py",
        "evaluation/splits.py",
        "ingestion/client.py",
        "ingestion/case_study.py",
        "domain/protocol.py",
    )
    return {name: package / name for name in names}


def _artifact_files(candidate: Path) -> dict[str, Path]:
    files = {
        "baseline/metadata.json": candidate / "baseline/metadata.json",
        "baseline/state.json": candidate / "baseline/state.json",
        "encoder/metadata.json": candidate / "encoder/metadata.json",
        "encoder/model.pt": candidate / "encoder/model.pt",
        "index.json": candidate / "index.json",
    }
    baseline_metadata = _read(files["baseline/metadata.json"])
    for name in baseline_metadata["artifact_files"]:
        if Path(name).name != name:
            raise ValueError("Artifact filenames must not escape the baseline directory")
        files[f"baseline/{name}"] = candidate / "baseline" / name
    return files


def _path(reference: str, freeze_path: Path) -> Path:
    return (freeze_path.parent / reference).resolve()


def _check_models(candidate: Path, protocol: dict):
    from ant26.models.representation import RepresentationBundle

    baseline = BaselineBundle.load(candidate / "baseline")
    encoder = RepresentationBundle.load(candidate / "encoder")
    index = AtlasIndex.load(candidate / "index.json")
    left, right = baseline.metadata, encoder.metadata
    if baseline.model.kind != "catboost_quantile":
        raise ValueError("The declared probabilistic release candidate must be CatBoost")
    for name in (
        "protocol_sha256",
        "data_sha256",
        "fitting_examples_sha256",
        "calibration_examples_sha256",
        "fit_sessions",
        "calibration_sessions",
    ):
        if left.get(name) != right.get(name):
            raise ValueError(f"Baseline and encoder disagree on {name}")
    if left["protocol_sha256"] != protocol_hash(protocol):
        raise ValueError("Candidate protocol differs from the release protocol")
    fitting, calibration = set(left["fit_sessions"]), set(left["calibration_sessions"])
    if (
        fitting.intersection(calibration)
        or fitting.intersection(MEXICO_SESSIONS)
        or calibration.intersection(MEXICO_SESSIONS)
    ):
        raise ValueError("Invalid release fitting/calibration boundary")
    payload = _read(candidate / "index.json")
    if set(payload["allowed_sessions"]) != fitting:
        raise ValueError("Atlas reference allowlist differs from encoder fitting scope")
    validate_scope(index.references, fitting, role="release atlas references")
    if any(row["session_key"] in calibration for row in index.references):
        raise ValueError("Calibration observations cannot become atlas references")
    if left.get("mexico_evaluated") or right.get("mexico_evaluated"):
        raise ValueError("A pre-case release cannot reuse a Mexico-evaluated candidate")
    return baseline, encoder, index


def freeze_release(
    candidate_root,
    protocol,
    dataset_path,
    output_path,
    *,
    code_revision,
    review_decision="experimental_continuous_retrieval_unvalidated",
) -> dict:
    """Pin existing candidates and record the explicit exploratory atlas decision.

    This is a freeze for subsequent historical case-study scoring, not evidence
    that the forecast existed before the race or that retrieval is validated.
    """
    allowed_decisions = {"experimental_continuous_retrieval_unvalidated", REVIEW_DECISION}
    if review_decision not in allowed_decisions:
        raise ValueError(
            "This release supports only the explicit unvalidated continuous-atlas decision"
        )
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("A source revision must be recorded before freezing")
    candidate, dataset, output = (
        Path(candidate_root).resolve(),
        Path(dataset_path).resolve(),
        Path(output_path).resolve(),
    )
    if output.exists():
        previous = verify_release(output, protocol)
        if (
            previous["code_revision"] != code_revision
            or _path(previous["candidate_root"], output) != candidate
            or previous["training_dataset"]["sha256"] != _file_sha(dataset)
        ):
            raise ValueError("Existing release freeze differs from the requested decision")
        return previous
    baseline, encoder, index = _check_models(candidate, protocol)
    envelope = _read(dataset)
    if envelope.get("protocol_sha256") != protocol_hash(protocol):
        raise ValueError("Training dataset protocol differs from the release protocol")
    examples = envelope["examples"]
    if _sha(examples) != baseline.metadata["data_sha256"]:
        raise ValueError("Training dataset differs from the candidate data snapshot")
    fitting = set(baseline.metadata["fit_sessions"])
    calibration = set(baseline.metadata["calibration_sessions"])
    validate_scope(examples, fitting | calibration, role="release training dataset")
    reference = valid_examples([row for row in examples if row["session_key"] in fitting])
    cal_rows = valid_examples([row for row in examples if row["session_key"] in calibration])
    if (
        _sha(reference) != encoder.metadata["fitting_examples_sha256"]
        or _sha(cal_rows) != encoder.metadata["calibration_examples_sha256"]
    ):
        raise ValueError("Release training/calibration rows differ from the candidate snapshots")
    if [row["example_id"] for row in reference] != [row["example_id"] for row in index.references]:
        raise ValueError("Atlas references differ from the exact encoder fitting examples")
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vectors = np.concatenate(
            [
                encoder.encode(reference[start : start + 64])
                for start in range(0, len(reference), 64)
            ]
        )
    finally:
        torch.set_num_threads(previous_threads)
    saved_vectors = np.asarray(_read(candidate / "index.json")["vectors"])
    if vectors.shape != saved_vectors.shape or not np.allclose(
        vectors, saved_vectors, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Atlas vectors do not match the pinned encoder and fitting snapshot")
    files = _artifact_files(candidate)
    for path in files.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Release artifacts must be local regular files")
    code = {name: _file_sha(path) for name, path in _code_files().items()}
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen_for_case_study",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "protocol_sha256": protocol_hash(protocol),
        "candidate_root": os.path.relpath(candidate, output.parent),
        "training_dataset": {
            "path": os.path.relpath(dataset, output.parent),
            "sha256": _file_sha(dataset),
            "examples_sha256": _sha(examples),
        },
        "artifact_files": {name: _file_sha(path) for name, path in files.items()},
        "source_files": code,
        "feature_builder_sha256": code["datasets/features.py"],
        "scoring_source_sha256": code["release.py"],
        "fit_sessions": sorted(fitting),
        "calibration_sessions": sorted(calibration),
        "index_hash": index.index_hash,
        "cohort_driver_numbers": sorted({row["driver_number"] for row in examples}),
        "atlas_decision": {
            "status": REVIEW_DECISION,
            "requested_decision": review_decision,
            "human_rating_count": 0,
            "validated_retrieval": False,
            "discrete_style_claims": False,
            "mode": "experimental_continuous_retrieval",
        },
        "prediction_candidate": "catboost_quantile",
        "case_study": {
            "session_key": 11404,
            "meeting_key": 1298,
            "driver_number": 12,
            "expected_laps": 71,
            "claim": protocol["final_case_study"]["claim"],
        },
        "index_encoder_max_abs_difference": float(np.max(np.abs(vectors - saved_vectors))),
    }
    manifest["manifest_sha256"] = _sha(manifest)
    _write(output, manifest)
    return manifest


def verify_release(path, protocol) -> dict:
    """Fail closed if an artifact, feature builder, scorer, or protocol changed."""
    path = Path(path).resolve()
    manifest = _read(path)
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha(unsigned):
        raise ValueError("Release manifest checksum mismatch")
    if (
        manifest.get("schema_version") != FREEZE_SCHEMA
        or manifest.get("status") != "frozen_for_case_study"
    ):
        raise ValueError("A frozen release is required before final-case access")
    if manifest.get("protocol_sha256") != protocol_hash(protocol):
        raise ValueError("Release protocol has changed")
    decision = manifest.get("atlas_decision", {})
    if (
        decision.get("status") != REVIEW_DECISION
        or decision.get("human_rating_count") != 0
        or decision.get("validated_retrieval") is not False
        or decision.get("discrete_style_claims") is not False
    ):
        raise ValueError("Unvalidated atlas decision was changed or overstated")
    for name, current in _code_files().items():
        if manifest.get("source_files", {}).get(name) != _file_sha(current):
            raise ValueError(f"Frozen source changed: {name}")
    candidate = _path(manifest["candidate_root"], path)
    current_files = _artifact_files(candidate)
    if set(current_files) != set(manifest["artifact_files"]):
        raise ValueError("Frozen artifact set changed")
    for name, current in current_files.items():
        if (
            current.is_symlink()
            or not current.is_file()
            or _file_sha(current) != manifest["artifact_files"][name]
        ):
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    if (
        _file_sha(_path(manifest["training_dataset"]["path"], path))
        != manifest["training_dataset"]["sha256"]
    ):
        raise ValueError("Frozen training dataset changed")
    baseline, _, index = _check_models(candidate, protocol)
    if (
        sorted(baseline.metadata["fit_sessions"]) != manifest["fit_sessions"]
        or sorted(baseline.metadata["calibration_sessions"]) != manifest["calibration_sessions"]
        or index.index_hash != manifest["index_hash"]
    ):
        raise ValueError("Frozen model/index scopes changed")
    return manifest


def _source_checks(manifests: list[dict], *, required: bool) -> dict[str, list[dict]]:
    """Verify exact cached provider bytes before deriving any scoring inputs."""
    endpoints = {
        "sessions",
        "laps",
        "car_data",
        "stints",
        "intervals",
        "pit",
        "race_control",
        "weather",
        "session_result",
    }
    if not isinstance(manifests, list) or (required and not manifests):
        raise ValueError("Real case inputs require exact source manifests")
    responses = {}
    for manifest in manifests:
        endpoint = manifest.get("endpoint")
        if endpoint not in endpoints or endpoint in responses:
            raise ValueError("Case source manifests have unknown or duplicate endpoints")
        if not manifest.get("snapshot_path"):
            raise ValueError("Case source manifest requires an exact snapshot_path")
        snapshot = Path(manifest["snapshot_path"])
        stored = _read(snapshot / "manifest.json")
        supplied = {
            key: value
            for key, value in manifest.items()
            if key not in {"snapshot_path", "cache_hit"}
        }
        if stored != supplied or stored.get("provider") != "openf1":
            raise ValueError("Case source manifest differs from its exact snapshot")
        if _file_sha(snapshot / "response.json") != manifest["sha256"]:
            raise ValueError("Case source snapshot checksum mismatch")
        params = manifest.get("params", {})
        if params.get("session_key") != 11404 or (
            endpoint not in {"sessions", "weather", "race_control"}
            and params.get("driver_number") != 12
        ):
            raise ValueError("Case source query escaped the frozen subject")
        if manifest.get("raw_response_filename"):
            name = manifest["raw_response_filename"]
            if Path(name).name != name or _file_sha(snapshot / name) != manifest.get(
                "raw_response_sha256"
            ):
                raise ValueError("Case original response checksum mismatch")
        elif manifest.get("response_format") == "csv":
            raise ValueError("CSV source manifest requires original response checksums")
        rows = json.loads((snapshot / "response.json").read_text())
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Case source response must contain provider row objects")
        if len(rows) != manifest.get("row_count"):
            raise ValueError("Case source row count differs from its manifest")
        if any(row.get("session_key") != 11404 for row in rows) or (
            endpoint not in {"sessions", "weather", "race_control"}
            and any(row.get("driver_number") != 12 for row in rows)
        ):
            raise ValueError("Case source rows escaped the frozen subject")
        responses[endpoint] = rows
    if required and set(responses) != endpoints:
        raise ValueError("Real case inputs require all nine exact source manifests")
    return responses


def _verify_case_reconstruction(envelope: dict, responses: dict, protocol: dict) -> None:
    from ant26.datasets.features import build_examples

    sessions = responses["sessions"]
    if (
        len(sessions) != 1
        or sessions[0].get("meeting_key") != 1298
        or envelope.get("source_session") != sessions[0]
    ):
        raise ValueError("Case session identity differs from the verified source")
    for endpoint in ("session_result", "race_control", "stints", "pit"):
        if envelope.get(endpoint) != responses[endpoint]:
            raise ValueError(f"Case {endpoint} differs from the verified source")
    observed = sorted({row["lap_number"] for row in responses["laps"]})
    if envelope.get("observed_laps") != observed or envelope.get("telemetry_rows") != len(
        responses["car_data"]
    ):
        raise ValueError("Case lap/telemetry accounting differs from the verified source")
    rebuilt = build_examples(
        responses["laps"],
        responses["car_data"],
        protocol=protocol,
        **{
            name: responses[name]
            for name in ("stints", "intervals", "pit", "race_control", "weather")
        },
    )
    if _sha(rebuilt) != _sha(envelope["examples"]):
        raise ValueError("Case features or targets differ from the verified source reconstruction")


def score_case_study(
    freeze_path, protocol, examples_envelope, output_dir, *, source_manifests=None, catalog=None
) -> dict:
    """Score existing Mexico examples offline; never fetch, fit, or update an index."""
    freeze_path, output = Path(freeze_path).resolve(), Path(output_dir)
    release = verify_release(freeze_path, protocol)
    envelope = _read(examples_envelope)
    freeze_sha = _file_sha(freeze_path)
    if envelope.get("freeze_sha256") != freeze_sha:
        raise ValueError("Case input is not bound to this frozen release")
    if (
        envelope.get("protocol_sha256") != protocol_hash(protocol)
        or envelope.get("feature_builder_sha256") != release["feature_builder_sha256"]
    ):
        raise ValueError("Case protocol or feature builder differs from the release")
    if envelope.get("expected_laps", 71) != 71:
        raise ValueError("The predeclared Mexico case requires 71 official laps")
    examples = envelope["examples"]
    if any(
        row.get("session_key") != 11404 or row.get("meeting_key", 1298) not in (None, 1298)
        for row in examples
    ):
        raise ValueError("Only Mexico race session 11404 can enter the final case study")
    if not examples:
        raise ValueError("No case-study sector records supplied")
    primary = [row for row in examples if row.get("driver_number") == 12]
    if not primary:
        raise ValueError("The case study requires Antonelli sector records")
    identifiers = [row["example_id"] for row in primary]
    keys = [(row["lap_number"], row["sector_index"]) for row in primary]
    if len(set(identifiers)) != len(identifiers) or len(set(keys)) != len(keys):
        raise ValueError("Duplicate case-study sector identities")
    if any(not isinstance(lap, int) or lap < 1 or sector not in (1, 2, 3) for lap, sector in keys):
        raise ValueError("Invalid case-study lap/sector identity")
    extra_laps = sorted({row["lap_number"] for row in primary if row["lap_number"] > 71})
    primary = [row for row in primary if row["lap_number"] <= 71]
    eligible = [row for row in primary if row.get("eligible")]
    for row in eligible:
        origin, cutoff = utc(row["origin"]), utc(row["cutoff"])
        if (origin - cutoff).total_seconds() + 1e-6 < protocol["prediction"][
            "feature_guard_seconds"
        ]:
            raise ValueError("Case features violate the frozen timing guard")
        maximum = row.get("provenance", {}).get("maximum_feature_timestamp")
        if maximum is None or utc(maximum) > cutoff:
            raise ValueError("Case features do not establish a past-only cutoff")
    manifests = (
        envelope.get("source_manifests", []) if source_manifests is None else source_manifests
    )
    synthetic = envelope.get("data_kind", "").startswith("synthetic")
    responses = _source_checks(manifests, required=not synthetic)
    if not synthetic:
        _verify_case_reconstruction(envelope, responses, protocol)
    candidate = _path(release["candidate_root"], freeze_path)
    baseline, encoder, index = _check_models(candidate, protocol)
    before_hash = index.index_hash
    scorable = valid_examples(primary)
    scored_ids = {row["example_id"] for row in scorable}
    quantiles = baseline.predict(eligible)
    predictions = [
        {
            "example_id": row["example_id"],
            "session_key": 11404,
            "driver_number": 12,
            "lap_number": row["lap_number"],
            "sector_index": row["sector_index"],
            "origin": row["origin"],
            "cutoff": row["cutoff"],
            "model": "catboost_quantile",
            "reference_seconds": row["reference_seconds"],
            "target_seconds": row.get("target_seconds"),
            "quantiles": values,
            "scoreable": row["example_id"] in scored_ids,
            "target_valid": row.get("target_valid", True),
            "target_exclusion_reasons": row.get("target_exclusion_reasons", []),
            "source_provenance": row.get("provenance", {}),
            "freeze_sha256": freeze_sha,
        }
        for row, values in zip(eligible, quantiles, strict=True)
    ]
    scored_predictions = [row for row in predictions if row["scoreable"]]
    observed_values = envelope.get("observed_laps", sorted({row["lap_number"] for row in primary}))
    observed_laps = sorted(
        {value["lap_number"] if isinstance(value, dict) else value for value in observed_values}
    )
    missing = sorted(set(range(1, 72)) - set(observed_laps))
    by_key = {(row["lap_number"], row["sector_index"]): row for row in primary}
    timeline = []
    for lap in range(1, 72):
        sectors = []
        for sector in range(1, 4):
            row = by_key.get((lap, sector))
            if row is None:
                sectors.append(
                    {
                        "sector_index": sector,
                        "status": "missing_record",
                        "reasons": ["missing_source_sector_record"],
                    }
                )
            else:
                status = (
                    "scored"
                    if row["example_id"] in scored_ids
                    else "censored"
                    if row.get("eligible")
                    else "abstained"
                )
                reasons = (
                    row.get("target_exclusion_reasons", [])
                    if status == "censored"
                    else row.get("exclusion_reasons", [])
                )
                sectors.append(
                    {
                        "sector_index": sector,
                        "example_id": row["example_id"],
                        "origin": row.get("origin"),
                        "cutoff": row.get("cutoff"),
                        "status": status,
                        "reasons": reasons,
                        "track_state": row.get("features", {})
                        .get("context", {})
                        .get("track_state"),
                    }
                )
        timeline.append(
            {
                "lap_number": lap,
                "source_lap_record_present": lap in observed_laps,
                "sectors": sectors,
            }
        )
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        queries = []
        for start in range(0, len(eligible), 64):
            rows = eligible[start : start + 64]
            vectors = encoder.encode(rows)
            queries.extend(
                index.query(row, vector, retrospective=True)
                for row, vector in zip(rows, vectors, strict=True)
            )
    finally:
        torch.set_num_threads(previous_threads)
    if index.index_hash != before_hash:
        raise ValueError("Case scoring changed the frozen atlas index")
    verify_release(freeze_path, protocol)
    accounting = {
        "official_laps": 71,
        "observed_lap_records": len(set(observed_laps) & set(range(1, 72))),
        "missing_lap_numbers": missing,
        "extra_lap_numbers": sorted(set(observed_laps) - set(range(1, 72)) | set(extra_laps)),
        "expected_sector_records": 213,
        "observed_sector_records": len(primary),
        "missing_sector_records": sum(
            sector["status"] == "missing_record" for lap in timeline for sector in lap["sectors"]
        ),
        "eligible_forecasts": len(eligible),
        "scoreable_forecasts": len(scorable),
        "abstentions": len(primary) - len(eligible),
        "censored_targets": len(eligible) - len(scorable),
        "abstention_reasons": dict(
            Counter(
                reason
                for row in primary
                if not row.get("eligible")
                for reason in row.get("exclusion_reasons", [])
            )
        ),
        "target_exclusion_reasons": dict(
            Counter(
                reason
                for row in eligible
                if row["example_id"] not in scored_ids
                for reason in row.get("target_exclusion_reasons", [])
                or ["invalid_or_missing_target"]
            )
        ),
    }
    atlas = {
        "index_hash": index.index_hash,
        "retrospective": True,
        "validation_status": REVIEW_DECISION,
        "human_rating_count": 0,
        "validated_retrieval": False,
        "discrete_style_claims": False,
        "queries": queries,
    }
    report = {
        "schema_version": "ant26-case-study-v1",
        "status": "case_study_scored",
        "synthetic": synthetic,
        "data_kind": envelope.get("data_kind", "unspecified"),
        "race": {"session_key": 11404, "meeting_key": 1298, "driver_number": 12, "lap_count": 71},
        "accounting": accounting,
        "metrics": macro_metrics(scored_predictions, 12),
        "chapters": envelope.get("chapters", []),
        "predictions": predictions,
        "atlas_decision": release["atlas_decision"],
        "provenance": {
            "freeze_sha256": freeze_sha,
            "release_manifest_sha256": release["manifest_sha256"],
            "code_revision": release["code_revision"],
            "source_files": release["source_files"],
            "protocol_sha256": protocol_hash(protocol),
            "case_examples_sha256": _sha(examples),
            "case_envelope_sha256": _file_sha(Path(examples_envelope))
            if isinstance(examples_envelope, (str, Path))
            else _sha(envelope),
            "source_manifests_sha256": _sha(manifests),
            "source_reconstruction_verified": not synthetic,
            "case_claim": release["case_study"]["claim"],
            "replay_assumption": "event_time_not_verified_live_publication_time",
            "source_session": envelope.get("source_session"),
            "session_result": envelope.get("session_result"),
            "catalog_sha256": _sha(catalog) if catalog is not None else None,
        },
        "cohort_driver_numbers": release["cohort_driver_numbers"],
        "auxiliary_examples_not_scored": len(examples)
        - len(primary)
        - sum(row.get("driver_number") == 12 and row["lap_number"] > 71 for row in examples),
        "output_files": {
            name: _sha(value)
            for name, value in (
                ("predictions.json", predictions),
                ("timeline.json", timeline),
                ("atlas.json", atlas),
            )
        },
    }
    _write(output / "predictions.json", predictions)
    _write(output / "timeline.json", timeline)
    _write(output / "atlas.json", atlas)
    _write(output / "report.json", report)
    return report
