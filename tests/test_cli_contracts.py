"""CLI boundaries must reject invalid research scopes before expensive work."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ant26.cli import main
from ant26.domain.protocol import protocol_hash


@pytest.fixture
def protocol_file(tmp_path):
    source = Path(__file__).parents[1] / "configs" / "evaluation-protocol.json"
    protocol = json.loads(source.read_text())
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path, protocol


def row(session=1, identifier="fit", cutoff="2026-01-02T12:00:00+00:00"):
    return {
        "example_id": identifier,
        "session_key": session,
        "meeting_key": 100 + session,
        "driver_number": 12,
        "lap_number": 3,
        "sector_index": 1,
        "origin": cutoff,
        "cutoff": cutoff,
        "reference_seconds": 30,
        "target_seconds": 30.2,
        "target_valid": True,
        "eligible": True,
        "features": {"speed_mean": 200},
        "provenance": {},
    }


def dataset(tmp_path, protocol, examples):
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps(
            {
                "examples": examples,
                "protocol_sha256": protocol_hash(protocol),
                "data_kind": "synthetic_cli_contract_fixture",
            }
        )
    )
    return path


def test_benchmark_rejects_dataset_protocol_mismatch_before_catalog_read(tmp_path, protocol_file):
    path, protocol = protocol_file
    payload = tmp_path / "examples.json"
    payload.write_text(json.dumps({"examples": [], "protocol_sha256": "wrong"}))
    with pytest.raises(ValueError, match="protocol hash"):
        main(
            [
                "--protocol",
                str(path),
                "benchmark",
                "--examples",
                str(payload),
                "--catalog",
                str(tmp_path / "does-not-exist.json"),
            ]
        )


def test_synthetic_dataset_cannot_be_promoted_to_real_release(tmp_path, protocol_file):
    pytest.importorskip("torch")
    path, protocol = protocol_file
    payload = dataset(tmp_path, protocol, [row(1)])
    with pytest.raises(ValueError, match="real Mercedes cohort"):
        main(
            [
                "--protocol",
                str(path),
                "prepare-release",
                "--examples",
                str(payload),
                "--catalog",
                str(tmp_path / "absent-catalog.json"),
                "--baseline-report",
                str(tmp_path / "absent-report.json"),
                "--output",
                str(tmp_path / "candidate"),
            ]
        )
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("fit,calibration", [([1], [1]), ([1], [11404]), ([11404], [])])
def test_train_invalid_scope_rejected_before_training(
    tmp_path, protocol_file, monkeypatch, fit, calibration
):
    pytest.importorskip("torch")
    import ant26.models.representation as representation

    path, protocol = protocol_file
    payload = dataset(tmp_path, protocol, [row(1), row(11404, "mexico")])

    def training_was_reached(*args, **kwargs):
        raise AssertionError("Training must not run for an invalid fitting/calibration scope")

    monkeypatch.setattr(representation, "train_representation", training_was_reached)
    command = [
        "--protocol",
        str(path),
        "train",
        "--examples",
        str(payload),
        "--fit-sessions",
        *map(str, fit),
        "--output",
        str(tmp_path / "bundle"),
    ]
    if calibration:
        command += ["--calibration-sessions", *map(str, calibration)]
    with pytest.raises(ValueError):
        main(command)


def test_train_rejects_unaccounted_fitting_sessions(tmp_path, protocol_file, monkeypatch):
    pytest.importorskip("torch")
    import ant26.models.representation as representation

    path, protocol = protocol_file
    payload = dataset(tmp_path, protocol, [row(1)])

    def forbidden(*args, **kwargs):
        raise AssertionError("A phantom fitting session reached training")

    monkeypatch.setattr(representation, "train_representation", forbidden)
    with pytest.raises(ValueError):
        main(
            [
                "--protocol",
                str(path),
                "train",
                "--examples",
                str(payload),
                "--fit-sessions",
                "1",
                "99",
                "--output",
                str(tmp_path / "bundle"),
            ]
        )


def test_train_rejects_future_unknown_session_before_training(tmp_path, protocol_file, monkeypatch):
    pytest.importorskip("torch")
    import ant26.models.representation as representation

    path, protocol = protocol_file
    future = row(9999, "future", "2026-11-08T12:00:00+00:00")
    payload = dataset(tmp_path, protocol, [future])

    def forbidden(*args, **kwargs):
        raise AssertionError("Future fitting data reached training")

    monkeypatch.setattr(representation, "train_representation", forbidden)
    with pytest.raises(ValueError, match="later timestamps"):
        main(
            [
                "--protocol",
                str(path),
                "train",
                "--examples",
                str(payload),
                "--fit-sessions",
                "9999",
                "--output",
                str(tmp_path / "bundle"),
            ]
        )


def fake_bundle(monkeypatch, protocol, fitting):
    pytest.importorskip("torch")
    from ant26.models.representation import RepresentationBundle

    class Bundle:
        metadata = {
            "protocol_sha256": protocol_hash(protocol),
            "fitting_examples_sha256": hashlib.sha256(
                json.dumps(fitting, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "fit_sessions": [1],
            "synthetic": True,
        }

        def encode(self, examples):
            raise AssertionError("Invalid atlas data reached encoding")

    monkeypatch.setattr(RepresentationBundle, "load", classmethod(lambda cls, path: Bundle()))


def test_atlas_rejects_query_before_encoder_fitting_data(tmp_path, protocol_file, monkeypatch):
    path, protocol = protocol_file
    fitting = [row(1)]
    fake_bundle(monkeypatch, protocol, fitting)
    query = row(2, "query", "2026-01-01T12:00:00+00:00")
    payload = dataset(tmp_path, protocol, fitting + [query])
    with pytest.raises(ValueError, match="follow"):
        main(
            [
                "--protocol",
                str(path),
                "atlas",
                "--examples",
                str(payload),
                "--bundle",
                "unused",
                "--query-example",
                "query",
                "--output",
                str(tmp_path / "atlas"),
            ]
        )


def test_atlas_rejects_changed_encoder_fitting_snapshot(tmp_path, protocol_file, monkeypatch):
    path, protocol = protocol_file
    fitting = [row(1)]
    fake_bundle(monkeypatch, protocol, fitting)
    changed = copy.deepcopy(fitting)
    changed[0]["features"]["speed_mean"] = 499
    query = row(2, "query", "2026-01-03T12:00:00+00:00")
    payload = dataset(tmp_path, protocol, changed + [query])
    with pytest.raises(ValueError):
        main(
            [
                "--protocol",
                str(path),
                "atlas",
                "--examples",
                str(payload),
                "--bundle",
                "unused",
                "--query-example",
                "query",
                "--output",
                str(tmp_path / "atlas"),
            ]
        )
