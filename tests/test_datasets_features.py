import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ant26.datasets import build_examples, quality_report, write_examples

START = datetime(2026, 3, 8, 4, 0, tzinfo=timezone.utc)


def stamp(seconds):
    return (START + timedelta(seconds=seconds)).isoformat()


def fixture():
    laps = [
        {
            "session_key": 11234,
            "meeting_key": 1279,
            "driver_number": 12,
            "lap_number": n,
            "date_start": stamp((n - 1) * 30),
            "duration_sector_1": 10,
            "duration_sector_2": 10,
            "duration_sector_3": 10,
            "lap_duration": 30,
            "is_pit_out_lap": False,
        }
        for n in range(1, 7)
    ]
    car_data = [
        {
            "session_key": 11234,
            "driver_number": 12,
            "date": stamp(i / 4),
            "speed": 200 + i % 20,
            "throttle": 80,
            "brake": 0,
            "rpm": 10000,
            "n_gear": 6,
            "drs": None,
        }
        for i in range(720)
    ]
    flags = [{"session_key": 11234, "date": stamp(0), "flag": "GREEN", "scope": "Track"}]
    return laps, car_data, flags


def make_examples(laps=None, car_data=None, flags=None, **kwargs):
    default_laps, default_car, default_flags = fixture()
    return build_examples(
        default_laps if laps is None else laps,
        default_car if car_data is None else car_data,
        race_control=default_flags if flags is None else flags,
        pit=[],
        **kwargs,
    )


class DatasetTests(unittest.TestCase):
    def test_sector_examples_have_past_reference_and_valid_window(self):
        rows = make_examples()
        self.assertEqual(len(rows), 18)
        row = next(r for r in rows if r["lap_number"] == 4 and r["sector_index"] == 1)
        self.assertTrue(row["eligible"], row["exclusion_reasons"])
        self.assertEqual(row["target_seconds"], 10)
        self.assertEqual(row["reference_seconds"], 10)
        self.assertEqual(row["cutoff"], stamp(89))
        self.assertEqual(len(row["features"]["sequence"]), 128)
        self.assertEqual(row["features"]["same_sector_history_count"], 3)
        self.assertNotIn("target_seconds", row["features"])
        json.dumps(rows, allow_nan=False)

    def test_future_mutation_cannot_change_earlier_features(self):
        laps, car, flags = fixture()
        before = make_examples(laps, car, flags)
        # Lap 4 S1 starts at t90: its own duration is a target, never an input.
        mutated_laps = copy.deepcopy(laps)
        mutated_laps[3]["duration_sector_1"] = 17
        mutated_laps[3]["lap_duration"] = 37
        mutated_laps[3]["position"] = 22
        mutated_car = copy.deepcopy(car)
        for row in mutated_car:
            if row["date"] > stamp(89):
                row["speed"] = 499
                row["throttle"] = 0
        flags.append({"session_key": 11234, "date": stamp(95), "flag": "YELLOW", "scope": "Track"})
        after = make_examples(mutated_laps, mutated_car, flags)

        def key(r):
            return r["lap_number"] == 4 and r["sector_index"] == 1

        a, b = next(filter(key, before)), next(filter(key, after))
        self.assertEqual(a["features"], b["features"])
        self.assertEqual(a["origin"], b["origin"])
        self.assertNotEqual(a["target_seconds"], b["target_seconds"])

    def test_unknown_flags_and_cold_start_abstain(self):
        rows = make_examples(flags=[])
        self.assertFalse(any(row["eligible"] for row in rows))
        self.assertIn("unknown_track_state", rows[-1]["exclusion_reasons"])
        first = make_examples()[0]
        self.assertIn("insufficient_history", first["exclusion_reasons"])

    def test_gap_masks_missing_samples_instead_of_fabricating_resolution(self):
        laps, car, flags = fixture()
        car = [row for row in car if not (stamp(60) < row["date"] < stamp(80))]
        row = next(
            r
            for r in make_examples(laps, car, flags)
            if r["lap_number"] == 4 and r["sector_index"] == 1
        )
        self.assertFalse(row["eligible"])
        self.assertIn("insufficient_telemetry_coverage", row["exclusion_reasons"])
        self.assertLess(row["features"]["telemetry_coverage"], 0.8)
        self.assertTrue(any(not all(mask) for mask in row["features"]["sequence_mask"]))

    def test_missing_lap_does_not_create_history_or_new_laps(self):
        laps, car, flags = fixture()
        laps = [row for row in laps if row["lap_number"] != 2]
        rows = make_examples(laps, car, flags)
        self.assertEqual(len(rows), 15)
        row = next(r for r in rows if r["lap_number"] == 3 and r["sector_index"] == 1)
        self.assertEqual(row["features"]["same_sector_history_count"], 1)
        self.assertFalse(row["eligible"])

    def test_red_flag_censors_target_but_does_not_leak_to_forecast(self):
        laps, car, flags = fixture()
        base = make_examples(laps, car, flags)
        flags.append({"session_key": 11234, "date": stamp(95), "flag": "RED", "scope": "Track"})
        rows = make_examples(laps, car, flags)

        def key(r):
            return r["lap_number"] == 4 and r["sector_index"] == 1

        before, after = next(filter(key, base)), next(filter(key, rows))
        self.assertEqual(before["features"], after["features"])
        self.assertTrue(after["eligible"])
        self.assertFalse(after["target_valid"])
        self.assertIn("red_flag_target", after["target_exclusion_reasons"])

    def test_future_stint_metadata_and_weather_do_not_leak(self):
        laps, car, flags = fixture()
        kwargs = {
            "stints": [
                {
                    "session_key": 11234,
                    "driver_number": 12,
                    "lap_start": 1,
                    "lap_end": 6,
                    "compound": "SOFT",
                    "tyre_age_at_start": 0,
                }
            ],
            "weather": [{"session_key": 11234, "date": stamp(0), "air_temperature": 22}],
        }
        before = make_examples(laps, car, flags, **kwargs)
        kwargs["stints"][0]["lap_end"] = 999
        kwargs["stints"].append(
            {
                "session_key": 11234,
                "driver_number": 12,
                "lap_start": 5,
                "compound": "HARD",
                "tyre_age_at_start": 42,
            }
        )
        kwargs["weather"].append({"session_key": 11234, "date": stamp(100), "air_temperature": 99})
        after = make_examples(laps, car, flags, **kwargs)

        def key(r):
            return r["lap_number"] == 4 and r["sector_index"] == 1

        self.assertEqual(
            next(filter(key, before))["features"], next(filter(key, after))["features"]
        )

    def test_inputs_unchanged_and_json_export(self):
        laps, car, flags = fixture()
        original = copy.deepcopy((laps, car, flags))
        rows = make_examples(laps, car, flags)
        self.assertEqual((laps, car, flags), original)
        report = quality_report(rows)
        self.assertEqual(report["total_examples"], 18)
        self.assertGreater(report["eligible_examples"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.json"
            write_examples(rows, path)
            self.assertEqual(json.loads(path.read_text()), rows)

    def test_duplicate_lap_keys_fail_visibly(self):
        laps, car, flags = fixture()
        laps.append(copy.deepcopy(laps[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            make_examples(laps, car, flags)

    def test_surprise_safety_car_remains_in_operational_scoring(self):
        laps, car, flags = fixture()
        flags.append(
            {
                "session_key": 11234,
                "date": stamp(95),
                "category": "SafetyCar",
                "message": "SAFETY CAR DEPLOYED",
            }
        )
        row = next(
            r
            for r in make_examples(laps, car, flags)
            if r["lap_number"] == 4 and r["sector_index"] == 1
        )
        self.assertTrue(row["eligible"])
        self.assertTrue(row["target_valid"])
        self.assertFalse(row["uninterrupted_green_target"])

    def test_irregular_native_samples_are_not_marked_as_grid_observations(self):
        laps, car, flags = fixture()
        for row in car:
            row["date"] = (
                datetime.fromisoformat(row["date"]) + timedelta(seconds=0.03)
            ).isoformat()
        row = next(
            r
            for r in make_examples(laps, car, flags)
            if r["lap_number"] == 4 and r["sector_index"] == 1
        )
        self.assertTrue(row["eligible"])
        self.assertEqual(row["features"]["native_grid_fraction"]["speed"], 0)
        self.assertGreater(row["features"]["native_sample_count"], 100)
        self.assertGreater(row["features"]["telemetry_coverage"], 0.95)

    def test_driver_and_session_data_are_isolated(self):
        laps, car, flags = fixture()
        baseline = make_examples(laps, car, flags)
        polluted = copy.deepcopy(car)
        for row in car:
            other = dict(row, driver_number=63, speed=499)
            polluted.append(other)
            polluted.append(dict(row, session_key=999, speed=498))
        self.assertEqual(baseline, make_examples(laps, polluted, flags))

    def test_unknown_pit_exit_cannot_use_unfinished_lap_validity(self):
        laps, car, flags = fixture()
        pit = [{"session_key": 11234, "driver_number": 12, "date": stamp(96)}]
        before = build_examples(laps, car, race_control=flags, pit=pit)
        changed = copy.deepcopy(laps)
        changed[3]["duration_sector_3"] = 17
        changed[3]["lap_duration"] = 37
        after = build_examples(changed, car, race_control=flags, pit=pit)
        a = next(r for r in before if r["lap_number"] == 4 and r["sector_index"] == 3)
        b = next(r for r in after if r["lap_number"] == 4 and r["sector_index"] == 3)
        self.assertEqual(a["features"], b["features"])
        self.assertEqual(a["eligible"], b["eligible"])

    def test_previous_lap_future_sector_cannot_change_completed_sector_history(self):
        laps, car, flags = fixture()
        before = make_examples(laps, car, flags)
        changed = copy.deepcopy(laps)
        # At t89, lap3 S1 and S2 have completed; S3 has not. Its eventual
        # duration and the full-lap/successor consistency checks are unavailable.
        changed[2]["duration_sector_3"] = 20
        changed[2]["lap_duration"] = 40
        after = make_examples(changed, car, flags)
        a = next(r for r in before if r["lap_number"] == 4 and r["sector_index"] == 1)
        b = next(r for r in after if r["lap_number"] == 4 and r["sector_index"] == 1)
        self.assertEqual(a["features"], b["features"])
        self.assertEqual(a["eligible"], b["eligible"])

    def test_previous_lap_missing_future_sector_does_not_remove_past_history(self):
        laps, car, flags = fixture()
        before = make_examples(laps, car, flags)
        changed = copy.deepcopy(laps)
        changed[2]["duration_sector_3"] = None
        changed[2]["lap_duration"] = None
        after = make_examples(changed, car, flags)
        a = next(r for r in before if r["lap_number"] == 4 and r["sector_index"] == 1)
        b = next(r for r in after if r["lap_number"] == 4 and r["sector_index"] == 1)
        self.assertEqual(a["features"], b["features"])


if __name__ == "__main__":
    unittest.main()
