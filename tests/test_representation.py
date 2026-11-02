from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ant26.models.representation import (  # noqa: E402
    EncoderConfig,
    MaskedPatchEncoder,
    RepresentationBundle,
    train_representation,
)


def examples(session=1, count=12):
    result = []
    for i in range(count):
        sequence = [[100 + i + time / 4, 80, 0, 10000 + time * 10, 5] for time in range(32)]
        result.append(
            {
                "example_id": f"{session}-{i}",
                "session_key": session,
                "driver_number": 12,
                "lap_number": i + 3,
                "sector_index": 1,
                "reference_seconds": 30,
                "target_seconds": 30 + i / 100,
                "eligible": True,
                "features": {
                    "sequence": sequence,
                    "sequence_mask": [[True] * 5 for _ in sequence],
                    "sequence_channels": ["speed", "throttle", "brake", "rpm", "n_gear"],
                    "speed_mean": 100 + i,
                    "last_same_sector_seconds": 30,
                },
                "provenance": {},
            }
        )
    return result


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_default_encoder_is_under_budget():
    model = MaskedPatchEncoder(EncoderConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_mexico_or_unexpected_sessions_fail_before_missing_sequence_access():
    for row in ({"session_key": 11404}, {"session_key": 1, "meeting_key": 1298}):
        with pytest.raises(ValueError, match="Mexico"):
            train_representation([row], {row["session_key"]})
    with pytest.raises(ValueError, match="Unexpected session"):
        train_representation([{"session_key": 2}], {1})
    with pytest.raises(ValueError, match="disjoint"):
        train_representation(examples(), {1}, tuning_examples=examples(), tuning_sessions={1})
    with pytest.raises(ValueError, match="Mexico"):
        train_representation(
            examples(), {1}, tuning_examples=[{"session_key": 11404}], tuning_sessions={11404}
        )


def test_masked_loss_never_learns_missing_values():
    model = MaskedPatchEncoder(EncoderConfig(width=16, layers=1, heads=2, max_steps=32))
    mask = torch.zeros((2, 32, 5), dtype=torch.bool)
    values = torch.ones((2, 32, 5))
    selection = torch.ones((2, 4), dtype=torch.bool)
    assert model.masked_loss(values, values, mask, selection).item() == 0


def test_cpu_smoke_training_decreases_reconstruction_and_roundtrips(tmp_path):
    rows = examples()
    original = deepcopy(rows)
    bundle = train_representation(
        rows,
        {1},
        config=EncoderConfig(width=16, layers=1, heads=2, max_steps=32),
        pretrain_epochs=10,
        forecast_epochs=3,
        batch_size=12,
        seed=17,
        data_kind="synthetic_smoke_fixture",
    )
    assert np.isfinite(bundle.metadata["forecast_loss_history"]).all()
    assert (
        bundle.metadata["reconstruction_loss_after"] < bundle.metadata["reconstruction_loss_before"]
    )
    assert bundle.metadata["synthetic"] is True
    predictions = np.asarray(bundle.predict(rows))
    assert predictions.shape == (12, 5)
    assert np.all(np.diff(predictions, axis=1) >= 0)
    bundle.save(tmp_path)
    loaded = RepresentationBundle.load(tmp_path)
    np.testing.assert_allclose(loaded.predict(rows), predictions)
    assert rows == original
    with pytest.raises(FileExistsError):
        bundle.save(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        RepresentationBundle.load(tmp_path)


def test_calibration_uses_disjoint_scope_and_keeps_encoder_fixed():
    rows = examples()
    bundle = train_representation(
        rows,
        {1},
        config=EncoderConfig(width=16, layers=1, heads=2, max_steps=32),
        pretrain_epochs=0,
        forecast_epochs=1,
        data_kind="synthetic_calibration_fixture",
    )
    before = bundle.encode(rows).copy()
    with pytest.raises(ValueError, match="disjoint"):
        bundle.calibrate(rows, calibration_sessions={1})
    with pytest.raises(ValueError, match="Mexico"):
        bundle.calibrate([{"session_key": 11404}], calibration_sessions={11404})
    calibration = examples(session=2)
    for row in calibration:
        row["target_seconds"] = 40
    raw = np.asarray(bundle.predict(calibration))
    bundle.calibrate(calibration, calibration_sessions={2})
    adjusted = np.asarray(bundle.predict(calibration))
    assert bundle.metadata["calibrated"] is True
    assert bundle.metadata["calibration_sessions"] == [2]
    assert (adjusted[:, 4] >= raw[:, 4]).all()
    assert (adjusted[:, 0] <= raw[:, 0]).all()
    assert np.all(np.diff(adjusted, axis=1) >= 0)
    np.testing.assert_allclose(bundle.encode(rows), before)


def test_constant_training_features_do_not_amplify_valid_heldout_changes():
    from ant26.models.baselines import NUMERIC_FEATURES

    rows = examples()
    for row in rows:
        row["features"].update(rainfall=0.0, brake_max=1.0)
    bundle = train_representation(
        rows,
        {1},
        config=EncoderConfig(width=16, layers=1, heads=2, max_steps=32),
        pretrain_epochs=0,
        forecast_epochs=1,
        data_kind="synthetic_constant_scale_regression",
    )
    heldout = deepcopy(rows)
    for row in heldout:
        row["features"].update(rainfall=1.0, brake_max=0.0)
        for timestep in row["features"]["sequence"]:
            timestep[4] = 6.0
    sequence, _, context = bundle._inputs(heldout)
    # All are valid physical changes from columns constant in the fitting data.
    # Unit-scale fallback preserves them without turning them into +/-1e6.
    np.testing.assert_allclose(context[:, NUMERIC_FEATURES.index("rainfall")].numpy(), 1.0)
    np.testing.assert_allclose(context[:, NUMERIC_FEATURES.index("brake_max")].numpy(), -1.0)
    np.testing.assert_allclose(sequence[:, :, 4].numpy(), 1.0)
    assert bundle.metadata["normalization_policy"] == "constant_unit_scale_v2"


def test_legacy_normalization_checkpoint_requires_retraining(tmp_path):
    import json

    bundle = train_representation(
        examples(),
        {1},
        config=EncoderConfig(width=16, layers=1, heads=2, max_steps=32),
        pretrain_epochs=0,
        forecast_epochs=1,
        data_kind="synthetic_legacy_policy_regression",
    )
    bundle.save(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.pop("normalization_policy")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="retrain"):
        RepresentationBundle.load(tmp_path)
