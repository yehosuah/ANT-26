import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ant26.research.telemetry import load_observations, measure_sectors

START = datetime(2026, 3, 8, 4, tzinfo=timezone.utc)


def stamp(seconds):
    return (START + timedelta(seconds=seconds)).isoformat()


def fixture():
    laps = [
        {
            "session_key": 11234,
            "driver_number": 12,
            "lap_number": lap,
            "date_start": stamp((lap - 1) * 30),
            "duration_sector_1": 10,
            "duration_sector_2": 10,
            "duration_sector_3": 10,
            "lap_duration": 30,
            "is_pit_out_lap": False,
        }
        for lap in range(1, 7)
    ]
    car = [
        {
            "session_key": 11234,
            "driver_number": 12,
            "date": stamp(index / 4),
            "speed": 120,
            "throttle": 50,
            "brake": 0,
            "rpm": 10000,
            "n_gear": 6,
        }
        for index in range(720)
    ]
    context = {
        "race_control": [
            {"session_key": 11234, "date": stamp(0), "scope": "Track", "flag": "GREEN"}
        ],
        "pit": [],
        "stints": [
            {
                "session_key": 11234,
                "driver_number": 12,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 6,
                "compound": "medium",
                "tyre_age_at_start": 0,
            }
        ],
        "intervals": [
            {"session_key": 11234, "driver_number": 12, "date": stamp(60), "interval": 3}
        ],
        "weather": [{"session_key": 11234, "date": stamp(60), "rainfall": False}],
        "session_lap_count": 6,
    }
    return laps, car, context


def selected(rows, lap=3, sector=1):
    return next(row for row in rows if row["lap_number"] == lap and row["sector_index"] == sector)


def test_native_elapsed_time_weights_differ_from_sample_counts_and_do_not_mutate_sources():
    laps, _, context = fixture()
    times = [60, 60.1] + [61.1 + index for index in range(9)]
    car = [
        {
            "session_key": 11234,
            "driver_number": 12,
            "date": stamp(time),
            "speed": 120,
            "throttle": 100 if time == 60 else 0,
            "brake": 0,
        }
        for time in times
    ]
    before = copy.deepcopy((laps, car, context))
    row = selected(measure_sectors(laps, car, **context))
    assert row["eligible"]
    assert row["valid_seconds"] == pytest.approx(10)
    assert row["full_throttle_fraction"] == pytest.approx(0.01, abs=1e-7)
    assert row["coast_fraction"] == pytest.approx(0.99, abs=1e-7)
    assert row["brake_on_fraction"] == 0
    assert row["compound"] == "MEDIUM" and row["tyre_age_laps"] == 2
    assert row["lap_fraction"] == pytest.approx(2 / 6)
    assert row["interval_gap"] == 3 and row["weather_rainfall"] == 0
    assert (laps, car, context) == before


def test_bad_channels_are_masked_and_long_native_gaps_never_become_observed_time():
    laps, car, context = fixture()
    for sample in car:
        if stamp(60) <= sample["date"] < stamp(61):
            sample["throttle"] = 104
    row = selected(measure_sectors(laps, car, **context))
    assert row["valid_coverage"] == pytest.approx(0.9)
    assert row["channel_coverage"]["speed"] == 1
    assert row["channel_coverage"]["throttle"] == pytest.approx(0.9)
    assert row["full_throttle_fraction"] == 0
    gapped = [sample for sample in car if not stamp(64) < sample["date"] < stamp(65.5)]
    gap = selected(measure_sectors(laps, gapped, **context))
    assert not gap["eligible"]
    assert "native_gap_exceeds_one_second" in gap["exclusion_reasons"]
    assert gap["max_gap_seconds"] == 1.5
    assert gap["valid_coverage"] < 0.9


def test_local_marshal_yellow_vetoes_any_timing_sector_until_clear_and_driver_blue_does_not():
    laps, car, context = fixture()
    context["race_control"] += [
        {
            "session_key": 11234,
            "date": stamp(59),
            "scope": "Sector",
            "sector": 13,
            "flag": "YELLOW",
        },
        {"session_key": 11234, "date": stamp(65), "scope": "Sector", "sector": 13, "flag": "CLEAR"},
        {
            "session_key": 11234,
            "date": stamp(59),
            "scope": "Driver",
            "driver_number": 12,
            "flag": "BLUE",
        },
    ]
    row = selected(measure_sectors(laps, car, **context))
    assert "local_marshal_sector_flag" in row["exclusion_reasons"]
    assert selected(measure_sectors(laps, car, **context), sector=2)["eligible"]
    context["race_control"] = [context["race_control"][0], context["race_control"][-1]]
    assert selected(measure_sectors(laps, car, **context))["eligible"]
    context["race_control"].append(
        {"session_key": 11234, "date": stamp(64), "scope": "Track", "flag": "YELLOW"}
    )
    assert (
        "not_continuously_global_green"
        in selected(measure_sectors(laps, car, **context))["exclusion_reasons"]
    )


def test_pit_neighbors_missing_anchors_and_missing_matching_context_are_explicit():
    laps, car, context = fixture()
    context["pit"] = [
        {"session_key": 11234, "driver_number": 12, "lap_number": 3, "date": stamp(92)}
    ]
    laps[3]["is_pit_out_lap"] = True
    rows = measure_sectors(laps, car, **context)
    for lap in (2, 3, 4, 5):
        assert "pit_or_adjacent_lap" in selected(rows, lap=lap)["exclusion_reasons"]
    assert selected(rows, lap=6)["eligible"]
    laps[2]["duration_sector_1"] = None
    context["stints"][0]["compound"] = "UNKNOWN"
    context["stints"][0]["tyre_age_at_start"] = "typo"
    row = selected(measure_sectors(laps, car, **context), sector=2)
    assert row["origin"] is None
    assert {"invalid_sector_anchor", "unknown_compound", "invalid_tyre_age"} <= set(
        row["exclusion_reasons"]
    )


def test_exact_loader_checks_source_hashes_and_denominator_excludes_partial_tail(tmp_path):
    laps, car, context = fixture()
    laps.append(
        {
            **laps[-1],
            "lap_number": 7,
            "date_start": stamp(180),
            "duration_sector_1": None,
            "duration_sector_2": None,
            "duration_sector_3": None,
            "lap_duration": None,
        }
    )
    sources = {
        "laps": laps,
        "car_data": car,
        **{key: value for key, value in context.items() if key != "session_lap_count"},
    }
    manifests = {}
    for endpoint, rows in sources.items():
        directory = tmp_path / endpoint
        directory.mkdir()
        payload = json.dumps(rows).encode()
        (directory / "response.json").write_bytes(payload)
        manifests[endpoint] = {
            "endpoint": endpoint,
            "query_key": endpoint,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "snapshot_path": str(directory),
            "row_count": len(rows),
        }
    progress = {
        "sessions": {
            "11234:12": {
                "status": "complete",
                "context_sources": {
                    key: value for key, value in manifests.items() if key != "car_data"
                },
                "chunks": {"trace": {"manifest": manifests["car_data"]}},
            }
        }
    }
    catalog = {
        "development": [{"session_key": 11234, "date_start": stamp(0), "verified_identity": True}]
    }
    output = load_observations(progress, catalog)
    assert not output["mexico_observations"]
    assert all(row["lap_fraction_denominator"] == 6 for row in output["development_observations"])
    assert set(output["metadata"]["imported_helper_sha256"]) == {"features.py", "materialize.py"}
    assert len(output["private_provenance"]["source_manifests"]) == 7
    json.dumps(output, allow_nan=False)
    (Path(manifests["laps"]["snapshot_path"]) / "response.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        load_observations(progress, catalog)
