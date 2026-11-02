"""Release decisions precede offline case scoring; tests never access a provider."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("catboost")

from ant26.atlas import AtlasIndex  # noqa: E402
from ant26.demo import synthetic_examples  # noqa: E402
from ant26.domain.protocol import protocol_hash  # noqa: E402
from ant26.models.artifacts import final_candidate_partitions, fit_final_candidate  # noqa: E402
from ant26.models.baselines import valid_examples  # noqa: E402
from ant26.models.representation import EncoderConfig, train_representation  # noqa: E402
from ant26.release import freeze_release, score_case_study, verify_release  # noqa: E402


@pytest.fixture
def prepared(tmp_path):
    import torch

    protocol = json.loads(
        (Path(__file__).parents[1] / "configs/evaluation-protocol.json").read_text()
    )
    examples, catalog = synthetic_examples(protocol, race_count=18)
    data = {
        "examples": examples,
        "protocol_sha256": protocol_hash(protocol),
        "data_kind": "synthetic_release_test",
    }
    dataset = tmp_path / "development.json"
    dataset.write_text(json.dumps(data))
    root = tmp_path / "candidate"
    baseline = fit_final_candidate(examples, catalog, protocol, data_kind="synthetic_release_test")
    baseline.save(root / "baseline")
    partition = final_candidate_partitions(catalog, protocol)
    fitting = [row for row in examples if row["session_key"] in partition["fit_sessions"]]
    calibration = [
        row for row in examples if row["session_key"] in partition["calibration_sessions"]
    ]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        encoder = train_representation(
            fitting,
            set(partition["fit_sessions"]),
            protocol=protocol,
            config=EncoderConfig(width=16, layers=1, heads=2),
            pretrain_epochs=0,
            forecast_epochs=1,
            data_kind="synthetic_release_test",
        )
        encoder.calibrate(calibration, calibration_sessions=set(partition["calibration_sessions"]))
        encoder.metadata.update(
            data_sha256=baseline.metadata["data_sha256"],
            status="candidate_not_frozen_for_mexico",
            mexico_evaluated=False,
        )
        encoder.save(root / "encoder")
        reference = valid_examples(fitting)
        AtlasIndex().fit(
            reference, encoder.encode(reference), allowed_sessions=partition["fit_sessions"]
        ).save(root / "index.json")
    finally:
        torch.set_num_threads(previous)
    freeze = tmp_path / "release.json"
    release = freeze_release(root, protocol, dataset, freeze, code_revision="test-revision")
    return root, protocol, dataset, freeze, release, examples


def case_input(protocol, freeze, release, examples):
    # Canonical features are reused solely as synthetic scoring fixtures. The
    # timestamps are shifted into November; no source request occurs.
    source = [
        row
        for row in examples
        if row["driver_number"] == 12 and row["session_key"] == 1017 and row["lap_number"] != 5
    ]
    rows = copy.deepcopy(source)
    for row in rows:
        row.update(
            session_key=11404,
            meeting_key=1298,
            example_id=f"11404:12:{row['lap_number']}:{row['sector_index']}",
        )
        for key in ("origin", "cutoff"):
            if row.get(key):
                row[key] = row[key].replace("2026-01-18", "2026-11-01")
        if row.get("provenance", {}).get("maximum_feature_timestamp"):
            row["provenance"]["maximum_feature_timestamp"] = row["provenance"][
                "maximum_feature_timestamp"
            ].replace("2026-01-18", "2026-11-01")
    eligible = next(row for row in rows if row["eligible"])
    eligible["target_valid"] = False
    eligible["target_exclusion_reasons"] = ["synthetic_red_flag_censoring"]
    return {
        "examples": rows,
        "expected_laps": 71,
        "observed_laps": sorted({row["lap_number"] for row in rows}),
        "data_kind": "synthetic_case_fixture",
        "protocol_sha256": protocol_hash(protocol),
        "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
        "feature_builder_sha256": release["feature_builder_sha256"],
        "source_manifests": [],
    }


def test_freeze_records_unvalidated_decision_without_mutating_candidates(prepared):
    root, protocol, dataset, freeze, release, _ = prepared
    assert release["status"] == "frozen_for_case_study"
    assert release["atlas_decision"]["status"] == "exploratory_unvalidated_no_human_ratings"
    assert release["atlas_decision"]["human_rating_count"] == 0
    assert release["atlas_decision"]["validated_retrieval"] is False
    assert (
        json.loads((root / "baseline/metadata.json").read_text())["status"]
        == "candidate_not_frozen_for_mexico"
    )
    assert verify_release(freeze, protocol) == release
    assert freeze_release(root, protocol, dataset, freeze, code_revision="test-revision") == release
    with pytest.raises(ValueError, match="differs"):
        freeze_release(root, protocol, dataset, freeze, code_revision="changed-revision")


def test_freeze_refuses_overstated_human_validation(prepared, tmp_path):
    root, protocol, dataset, _, _, _ = prepared
    with pytest.raises(ValueError, match="unvalidated"):
        freeze_release(
            root,
            protocol,
            dataset,
            tmp_path / "bad.json",
            code_revision="test-revision",
            review_decision="human_validated",
        )


def test_changed_artifact_invalidates_freeze(prepared):
    root, protocol, _, freeze, _, _ = prepared
    path = root / "baseline/quantile-0.cbm"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_release(freeze, protocol)


def test_offline_scoring_accounts_missing_abstained_and_censored(prepared, tmp_path, monkeypatch):
    root, protocol, _, freeze, release, examples = prepared
    envelope = case_input(protocol, freeze, release, examples)
    before = hashlib.sha256((root / "index.json").read_bytes()).hexdigest()
    # Scoring must never enter training, even though it loads the frozen encoder.
    import ant26.models.representation as representation

    monkeypatch.setattr(
        representation,
        "train_representation",
        lambda *a, **kw: pytest.fail("Scoring fitted a model"),
    )
    output = tmp_path / "case"
    result = score_case_study(freeze, protocol, envelope, output)
    accounting = result["accounting"]
    assert result["synthetic"] is True
    assert result["race"]["lap_count"] == 71
    assert 5 in accounting["missing_lap_numbers"]
    assert accounting["censored_targets"] == 1
    assert (
        accounting["eligible_forecasts"]
        == accounting["scoreable_forecasts"] + accounting["censored_targets"]
    )
    assert (
        accounting["observed_sector_records"]
        == accounting["eligible_forecasts"] + accounting["abstentions"]
    )
    assert (
        accounting["expected_sector_records"]
        == accounting["observed_sector_records"] + accounting["missing_sector_records"]
    )
    assert len(json.loads((output / "timeline.json").read_text())) == 71
    atlas = json.loads((output / "atlas.json").read_text())
    assert atlas["retrospective"] is True
    assert atlas["validated_retrieval"] is False
    assert all(query["retrospective"] for query in atlas["queries"])
    assert hashlib.sha256((root / "index.json").read_bytes()).hexdigest() == before
    assert score_case_study(freeze, protocol, envelope, output) == result


def test_scoring_requires_exact_freeze_and_past_only_prefix(prepared, tmp_path):
    _, protocol, _, freeze, release, examples = prepared
    envelope = case_input(protocol, freeze, release, examples)
    changed = copy.deepcopy(envelope)
    changed["freeze_sha256"] = "wrong"
    with pytest.raises(ValueError, match="bound"):
        score_case_study(freeze, protocol, changed, tmp_path / "wrong")
    changed = copy.deepcopy(envelope)
    row = next(row for row in changed["examples"] if row["eligible"])
    row["provenance"]["maximum_feature_timestamp"] = row["origin"]
    with pytest.raises(ValueError, match="past-only"):
        score_case_study(freeze, protocol, changed, tmp_path / "future")


def test_real_inputs_cannot_omit_exact_source_manifests(prepared, tmp_path):
    _, protocol, _, freeze, release, examples = prepared
    envelope = case_input(protocol, freeze, release, examples)
    envelope["data_kind"] = "real_openf1_mexico_case_study"
    with pytest.raises(ValueError, match="exact source manifests"):
        score_case_study(freeze, protocol, envelope, tmp_path / "unbound")


def raw_case_sources(tmp_path):
    sources = {
        name: []
        for name in (
            "sessions",
            "laps",
            "car_data",
            "stints",
            "intervals",
            "pit",
            "race_control",
            "weather",
            "session_result",
        )
    }
    sources["sessions"] = [{"session_key": 11404, "meeting_key": 1298}]
    sources["laps"] = [
        {
            "session_key": 11404,
            "meeting_key": 1298,
            "driver_number": 12,
            "lap_number": 1,
            "date_start": "2026-11-01T20:00:00+00:00",
            "lap_duration": 60,
            "duration_sector_1": 20,
            "duration_sector_2": 20,
            "duration_sector_3": 20,
            "is_pit_out_lap": False,
        }
    ]
    manifests = []
    for endpoint, rows in sources.items():
        directory = tmp_path / endpoint
        directory.mkdir()
        payload = json.dumps(rows, sort_keys=True).encode()
        (directory / "response.json").write_bytes(payload)
        params = {"session_key": 11404}
        if endpoint not in {"sessions", "weather", "race_control"}:
            params["driver_number"] = 12
        stored = {
            "provider": "openf1",
            "endpoint": endpoint,
            "params": params,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "row_count": len(rows),
            "response_format": "json",
        }
        (directory / "manifest.json").write_text(json.dumps(stored))
        manifests.append(stored | {"snapshot_path": str(directory)})
    return sources, manifests


def test_real_source_manifest_integrity_and_complete_endpoint_scope(tmp_path):
    from ant26.release import _source_checks

    sources, manifests = raw_case_sources(tmp_path)
    assert _source_checks(manifests, required=True) == sources
    with pytest.raises(ValueError, match="nine"):
        _source_checks(manifests[:-1], required=True)
    with pytest.raises(ValueError, match="duplicate"):
        _source_checks(manifests + manifests[:1], required=True)
    missing = copy.deepcopy(manifests)
    del missing[0]["snapshot_path"]
    with pytest.raises(ValueError, match="snapshot_path"):
        _source_checks(missing, required=True)
    (tmp_path / "laps/response.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        _source_checks(manifests, required=True)


def test_real_features_and_targets_must_reconstruct_from_verified_sources(tmp_path):
    from ant26.datasets.features import build_examples
    from ant26.domain.protocol import load_protocol
    from ant26.release import _source_checks, _verify_case_reconstruction

    protocol = load_protocol()
    sources, manifests = raw_case_sources(tmp_path)
    responses = _source_checks(manifests, required=True)
    envelope = {
        "source_session": sources["sessions"][0],
        "observed_laps": [1],
        "telemetry_rows": 0,
        **{name: sources[name] for name in ("session_result", "race_control", "stints", "pit")},
        "examples": build_examples(sources["laps"], [], protocol=protocol, race_control=[], pit=[]),
    }
    _verify_case_reconstruction(envelope, responses, protocol)
    envelope["examples"][0]["target_seconds"] = 0.123
    with pytest.raises(ValueError, match="source reconstruction"):
        _verify_case_reconstruction(envelope, responses, protocol)


def correction_record(tmp_path, freeze, release, *, eligible=0):
    report = tmp_path / "preserved-first-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "case_study_scored",
                "race": {"session_key": 11404},
                "accounting": {"eligible_forecasts": eligible, "scoreable_forecasts": 0},
                "provenance": {
                    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
                    "release_manifest_sha256": release["manifest_sha256"],
                    "protocol_sha256": release["protocol_sha256"],
                },
            }
        )
    )
    record = tmp_path / "correction.json"
    record.write_text(
        json.dumps(
            {
                "parent_freeze_path": str(freeze),
                "parent_freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
                "parent_report_path": str(report),
                "parent_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "description": "Resolve a pre-race pit interval using only prior completed-lap evidence.",
            }
        )
    )
    return record


def test_post_inspection_correction_is_explicit_and_preserves_first_attempt(prepared, tmp_path):
    root, protocol, dataset, freeze, release, _ = prepared
    original = freeze.read_bytes()
    record = correction_record(tmp_path, freeze, release)
    corrected = tmp_path / "corrected-freeze.json"
    result = freeze_release(
        root,
        protocol,
        dataset,
        corrected,
        code_revision="correction-revision",
        correction_record=record,
    )
    assert (
        result["case_study"]["claim"]
        == "post_inspection_implementation_correction_diagnostic_replay"
    )
    assert result["implementation_correction"]["first_attempt_eligible_forecasts"] == 0
    assert result["implementation_correction"]["post_inspection"] is True
    assert result["artifact_files"] == release["artifact_files"]
    assert result["atlas_decision"]["validated_retrieval"] is False
    assert freeze.read_bytes() == original
    assert verify_release(corrected, protocol) == result
    assert (
        freeze_release(
            root,
            protocol,
            dataset,
            corrected,
            code_revision="correction-revision",
            correction_record=record,
        )
        == result
    )
    with pytest.raises(ValueError, match="differs"):
        freeze_release(root, protocol, dataset, corrected, code_revision="correction-revision")
    payload = json.loads(record.read_text())
    payload["description"] = "Changed rationale"
    record.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance changed"):
        verify_release(corrected, protocol)


def test_correction_rejects_nonzero_first_attempt_and_changed_forecast_artifacts(
    prepared, tmp_path
):
    root, protocol, dataset, freeze, release, _ = prepared
    record = correction_record(tmp_path, freeze, release, eligible=1)
    with pytest.raises(ValueError, match="zero-forecast"):
        freeze_release(
            root,
            protocol,
            dataset,
            tmp_path / "bad.json",
            code_revision="changed",
            correction_record=record,
        )
    parent = copy.deepcopy(release)
    parent["artifact_files"]["baseline/quantile-0.cbm"] = "different"
    from ant26.release import _sha

    parent["manifest_sha256"] = _sha(
        {key: value for key, value in parent.items() if key != "manifest_sha256"}
    )
    fake_parent = tmp_path / "changed-parent.json"
    fake_parent.write_text(json.dumps(parent))
    record = correction_record(tmp_path, fake_parent, parent)
    with pytest.raises(ValueError, match="preserve artifact_files"):
        freeze_release(
            root,
            protocol,
            dataset,
            tmp_path / "changed-model.json",
            code_revision="changed",
            correction_record=record,
        )
