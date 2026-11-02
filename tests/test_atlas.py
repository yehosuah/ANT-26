import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ant26.atlas import AtlasIndex, FeatureAtlasIndex, fit_pattern_clusters


def example(identifier, session=100, driver=12, lap=3, second=0, meeting=1):
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    origin = f"2026-03-08T{hour:02d}:{minute:02d}:{second:02d}+00:00"
    return {
        "example_id": identifier,
        "session_key": session,
        "meeting_key": meeting,
        "driver_number": driver,
        "lap_number": lap,
        "sector_index": 1,
        "origin": origin,
        "cutoff": origin,
        "features": {"speed_mean": 200 + lap, "throttle_mean": 90, "context": {"compound": "SOFT"}},
        "provenance": {"availability": "event_time"},
    }


class AtlasTests(unittest.TestCase):
    def test_prefix_query_excludes_future_and_overlapping_same_driver_windows(self):
        rows = [
            example("past", second=10, lap=1),
            example("overlap", second=95, lap=2),
            example("future", second=120, lap=4),
        ]
        index = AtlasIndex().fit(rows, [[2, 0], [0, 0], [0, 0]], allowed_sessions=[100])
        query = example("query", second=100, lap=3)
        result = index.query(query, [0, 0])
        self.assertEqual([n["example_id"] for n in result["neighbors"]], ["past"])
        self.assertFalse(result["retrospective"])

    def test_same_lap_is_excluded_even_in_retrospective_mode(self):
        rows = [example("same", second=10, lap=3), example("other", session=101, second=120)]
        index = AtlasIndex().fit(rows, [[0, 0], [1, 0]], allowed_sessions=[100, 101])
        result = index.query(example("query", second=100, lap=3), [0, 0], retrospective=True)
        self.assertEqual([n["example_id"] for n in result["neighbors"]], ["other"])
        self.assertTrue(result["retrospective"])

    def test_mexico_and_nonallowlisted_examples_fail_at_fit(self):
        for row in (
            example("mex", session=11404),
            example("mexmeeting", meeting=1298),
            example("practice", session=11397),
        ):
            with self.assertRaisesRegex(ValueError, "Mexico"):
                AtlasIndex().fit([row], [[1, 0]], allowed_sessions=[row["session_key"]])
        with self.assertRaisesRegex(ValueError, "allowlist"):
            AtlasIndex().fit([example("x")], [[1, 0]], allowed_sessions=[999])

    def test_reference_copy_and_json_roundtrip_are_immutable(self):
        row = example("x", second=10)
        vectors = np.array([[1.0, 2.0]])
        index = AtlasIndex().fit([row], vectors, allowed_sessions=[100])
        original_hash = index.index_hash
        row["features"]["speed_mean"] = 0
        vectors[:] = 99
        refs = index.references
        refs[0]["observed_summary"]["speed_mean"] = 999
        result = index.query(example("q", session=101, second=100), [1, 2])
        self.assertEqual(result["neighbors"][0]["observed_summary"]["speed_mean"], 203)
        self.assertEqual(result["neighbors"][0]["distance"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            index.save(path)
            restored = AtlasIndex.load(path)
            self.assertEqual(restored.index_hash, original_hash)
            self.assertEqual(restored.query(example("q", session=101, second=100), [1, 2]), result)
            payload = json.loads(path.read_text())
            payload["vectors"][0][0] = 888
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash"):
                AtlasIndex.load(path)

    def test_invalid_vectors_and_duplicate_ids_fail_visibly(self):
        for vectors in ([[float("nan"), 0]], [[float("inf"), 1]], [[]]):
            with self.assertRaises(ValueError):
                AtlasIndex().fit([example("x")], vectors, allowed_sessions=[100])
        row = example("x")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            AtlasIndex().fit([row, copy.deepcopy(row)], [[1, 0], [2, 0]], allowed_sessions=[100])
        index = AtlasIndex().fit([row], [[1, 0]], allowed_sessions=[100])
        with self.assertRaisesRegex(ValueError, "dimension"):
            index.query(example("q"), [1, 2, 3])

    def test_feature_baseline_ignores_target_and_fits_scaling_on_reference_only(self):
        rows = [example(f"x{i}", second=i * 60, lap=i + 1) for i in range(4)]
        index = FeatureAtlasIndex().fit(rows, allowed_sessions=[100])
        query = example("q", session=101, second=600)
        first = index.query(query)
        query["target_seconds"] = 9999
        query["features"]["target_alias"] = 9999
        self.assertEqual(index.query(query), first)
        self.assertFalse(index.query(query)["retrospective"])

    def test_small_sample_clusters_do_not_claim_validated_states(self):
        result = fit_pattern_clusters([[1, 0], [0, 1]], seed=17)
        self.assertEqual(result["status"], "insufficient_samples")
        self.assertFalse(result["validated"])

    def test_saved_indices_are_idempotent_and_refuse_replacement(self):
        rows = [example(f"x{i}", second=i * 60, lap=i + 1) for i in range(4)]
        index = AtlasIndex().fit(rows, [[1, i] for i in range(4)], allowed_sessions=[100])
        features = FeatureAtlasIndex().fit(rows, allowed_sessions=[100])
        with tempfile.TemporaryDirectory() as tmp:
            for name, artifact in (("learned", index), ("features", features)):
                path = Path(tmp) / f"{name}.json"
                artifact.save(path)
                before = path.read_bytes()
                artifact.save(path)
                self.assertEqual(before, path.read_bytes())
                path.write_text('{"different": true}\n')
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    artifact.save(path)
                self.assertEqual(path.read_text(), '{"different": true}\n')


if __name__ == "__main__":
    unittest.main()
