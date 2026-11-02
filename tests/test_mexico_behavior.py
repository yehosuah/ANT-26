"""Independent time-weighting and finite-precision summary contracts."""

import pytest

from ant26.research.mexico import summarize_mexico


def observation(sector, *, duration=10.0, valid=10.0, full=0.5, brake=0.2):
    return {
        "session_key": 11404,
        "driver_number": 12,
        "lap_number": 10,
        "sector_index": sector,
        "eligible": True,
        "compound": "SOFT",
        "stint_number": 1,
        "tyre_age_laps": 9,
        "duration_seconds": duration,
        "valid_seconds": valid,
        "valid_coverage": min(1.0, valid / duration),
        "full_throttle_fraction": full,
        "brake_on_fraction": brake,
        "coast_fraction": 0.0,
    }


def test_control_summary_weights_observed_time_and_preserves_all_denominators():
    first = observation(1, duration=1.0, valid=1.0, full=1.0, brake=0.0)
    second = observation(2, duration=10.0, valid=9.0, full=0.0, brake=1.0)
    excluded = observation(3) | {
        "eligible": False,
        "exclusion_reasons": ["local_flag", "local_flag", "source_gap"],
    }
    result = summarize_mexico([first, second, excluded])
    phase = result["phases"][0]
    assert phase["full_throttle_fraction"] == pytest.approx(0.1)
    assert phase["brake_fraction"] == pytest.approx(0.9)
    assert phase["retained_sectors"] == 2
    assert phase["excluded_sectors"] == 1
    assert phase["exclusion_reasons"] == {"local_flag": 1, "source_gap": 1}
    assert result["accounting"]["missing_sectors"] == 210
    assert result["phases"][1]["full_throttle_fraction"] is None
    assert phase["tyre_context"]["age_median_laps"] == 9


def test_native_timestamp_roundoff_preserves_valid_coverage():
    row = observation(1, duration=17.795, valid=17.795000076)
    result = summarize_mexico([row])
    assert result["accounting"]["coverage_fraction"] == 1.0
    assert result["accounting"]["covered_seconds"] == row["valid_seconds"]
    with pytest.raises(ValueError, match="covered and elapsed"):
        summarize_mexico([row | {"valid_seconds": 17.796}])


def test_summary_rejects_duplicate_foreign_or_inconsistent_rows():
    row = observation(1)
    for rows in (
        [row, row],
        [row | {"session_key": 11396}],
        [row | {"valid_coverage": 0.5}],
    ):
        with pytest.raises(ValueError):
            summarize_mexico(rows)
