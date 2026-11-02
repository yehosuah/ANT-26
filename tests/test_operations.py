import copy

import numpy as np
import pytest

from ant26.evaluation.operations import input_drift_report, latency_gate, main, measure_calls
from ant26.models.baselines import NUMERIC_FEATURES


def _row(session, value, *, eligible=True, driver=12):
    return {
        "session_key": session,
        "driver_number": driver,
        "eligible": eligible,
        "features": {"speed_mean": value},
        "origin": "2026-09-01T00:00:00+00:00",
    }


def test_latency_measurement_excludes_warmup_and_uses_empirical_percentiles():
    durations = np.arange(1, 101, dtype=int) * 1_000_000
    ticks = iter(tick for duration in durations for tick in (0, duration))
    calls = []
    result = measure_calls(calls.append, repetitions=100, warmup=3, clock=lambda: next(ticks))
    assert len(calls) == 103
    assert result["p50_ms"] == 50.5
    assert result["p95_ms"] == pytest.approx(95.05)
    assert result["p99_ms"] == pytest.approx(99.01)
    assert result["max_ms"] == 100


@pytest.mark.parametrize("arguments", [{"repetitions": 99}, {"warmup": 0}, {"warmup": 1001}])
def test_latency_requires_a_bounded_meaningful_sample(arguments):
    with pytest.raises(ValueError):
        measure_calls(lambda _: None, **arguments)


def test_latency_does_not_mask_operation_failure():
    def broken(_):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        measure_calls(broken)


def test_latency_gate_strictly_enforces_declared_threshold():
    assert latency_gate({"p95_ms": 199.99})["passed"]
    assert not latency_gate({"p95_ms": 200})["passed"]
    with pytest.raises(ValueError):
        latency_gate({"p95_ms": float("nan")})


def test_input_drift_counts_missing_range_and_ood_without_using_labels():
    fit = [_row(1, 10), _row(1, 20)]
    current = [_row(2, 30), _row(2, None, driver=63), _row(2, 99, eligible=False)]
    scale, mean = [1.0] * len(NUMERIC_FEATURES), [0.0] * len(NUMERIC_FEATURES)
    scale[NUMERIC_FEATURES.index("driver_number")] = 100.0
    kwargs = dict(fit_sessions={1}, observed_sessions={2}, context_mean=mean, context_scale=scale)
    result = input_drift_report(fit, current, **kwargs)
    speed = result["features"]["speed_mean"]
    assert speed["observed_missing_fraction"] == 0.5
    assert speed["outside_fitted_range_count"] == 1
    assert speed["max_abs_encoder_context_z"] == 30
    assert result["encoder_context_z_warning_count"] == 1
    assert result["unknown_driver_origin_count"] == 1
    assert result["observed_origin_count"] == 2
    changed = copy.deepcopy(current)
    for row in changed:
        row.update(target_valid=False, target_seconds=999999)
    assert input_drift_report(fit, changed, **kwargs) == result


def test_drift_all_missing_fitted_feature_is_unknown_not_zero_exceedance():
    result = input_drift_report(
        [_row(1, None)], [_row(2, 1)], fit_sessions={1}, observed_sessions={2}
    )
    assert result["features"]["speed_mean"]["outside_fitted_range_count"] is None
    assert result["features"]["speed_mean"]["fit_missing_fraction"] == 1


def test_drift_rejects_foreign_mexico_and_overlapping_scopes():
    with pytest.raises(ValueError, match="Unexpected session"):
        input_drift_report([_row(1, 1)], [_row(3, 2)], fit_sessions={1}, observed_sessions={2})
    with pytest.raises(ValueError, match="Mexico"):
        input_drift_report(
            [_row(1, 1)], [_row(11404, 2)], fit_sessions={1}, observed_sessions={11404}
        )
    with pytest.raises(ValueError, match="disjoint"):
        input_drift_report([_row(1, 1)], [_row(1, 2)], fit_sessions={1}, observed_sessions={1})


def test_drift_rejects_invalid_fitted_scales():
    with pytest.raises(ValueError, match="positive"):
        input_drift_report(
            [_row(1, 1)],
            [_row(2, 2)],
            fit_sessions={1},
            observed_sessions={2},
            context_mean=[0] * len(NUMERIC_FEATURES),
            context_scale=[0] * len(NUMERIC_FEATURES),
        )


def test_operations_report_refuses_overwrite_before_loading_any_artifact(tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"existing": true}')
    with pytest.raises(FileExistsError, match="immutable"):
        main(
            [
                "--dataset",
                "missing-dataset",
                "--baseline",
                "missing-baseline",
                "--encoder",
                "missing-encoder",
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == '{"existing": true}'
