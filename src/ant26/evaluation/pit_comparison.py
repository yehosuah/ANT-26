"""Compare saved pit-policy replays without fitting, fetching, or scoring models.

Coverage changes and forecast error changes are different outcomes. Only shared
scoreable identities with unchanged labels receive paired metric deltas; newly
covered and lost populations remain separate descriptive slices.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ant26.evaluation.metrics import macro_metrics, weighted_interval_score
from ant26.evaluation.splits import validate_scope
from ant26.models.baselines import valid_examples

STATUSES = ("scored", "censored", "abstained", "missing_record")
SLOTS = {(lap, sector) for lap in range(1, 72) for sector in range(1, 4)}


def _read(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _digest(value) -> str:
    payload = (
        Path(value).read_bytes()
        if isinstance(value, (str, Path))
        else json.dumps(value, sort_keys=True, allow_nan=False).encode()
    )
    return hashlib.sha256(payload).hexdigest()


def _save(path, report):
    if path is None:
        return
    path = Path(path)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError("Refusing to replace immutable pit comparison")
    else:
        with path.open("x") as handle:
            handle.write(payload)


def _close(left, right) -> bool:
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    )


def _validate_replay(report, predictions, timeline):
    race = report.get("race", {})
    if (race.get("session_key"), race.get("driver_number"), race.get("lap_count")) != (
        11404,
        12,
        71,
    ):
        raise ValueError("Pit replay comparisons require the same Mexico 71-lap Antonelli race")
    if isinstance(timeline, dict):
        timeline = timeline["laps"]
    slots, identities = {}, {}
    for lap in timeline:
        for sector in lap["sectors"]:
            key = (lap["lap_number"], sector["sector_index"])
            if key in slots or key not in SLOTS or sector.get("status") not in STATUSES:
                raise ValueError(
                    "Replay timeline contains duplicate, invalid or unknown-status slots"
                )
            slots[key] = sector
            identity = sector.get("example_id")
            if sector["status"] != "missing_record" and not isinstance(identity, str):
                raise ValueError("Observed timeline sectors require example identities")
            if identity is not None:
                if identity in identities:
                    raise ValueError("Duplicate timeline example identity")
                identities[identity] = key
    if set(slots) != SLOTS or len(timeline) != 71:
        raise ValueError("Replay timeline must account for all 213 official sector slots")
    by_id = {}
    for row in predictions:
        identity = row["example_id"]
        key = (row["lap_number"], row["sector_index"])
        if identity in by_id or identities.get(identity) != key:
            raise ValueError("Prediction identity is duplicated or disagrees with the timeline")
        if row.get("session_key") != 11404 or row.get("driver_number") != 12:
            raise ValueError("Prediction escaped the case-study subject")
        if type(row.get("scoreable")) is not bool:
            raise ValueError("Saved predictions must explicitly declare scoreable status")
        expected = "scored" if row["scoreable"] else "censored"
        if slots[key]["status"] != expected:
            raise ValueError("Prediction scoreability disagrees with the timeline")
        weighted_interval_score(0.0, row["quantiles"])
        if row["scoreable"]:
            target, reference = row["target_seconds"], row["reference_seconds"]
            if (
                not all(
                    isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                    for value in (target, reference)
                )
                or row.get("target_valid", True) is not True
            ):
                raise ValueError("Scoreable prediction has an invalid target or reference")
            weighted_interval_score(target, row["quantiles"])
        by_id[identity] = row
    eligible_ids = {
        row["example_id"] for row in slots.values() if row["status"] in ("scored", "censored")
    }
    if set(by_id) != eligible_ids:
        raise ValueError("Every eligible timeline sector requires exactly one saved prediction")
    counts = Counter(row["status"] for row in slots.values())
    accounting = report["accounting"]
    expected_counts = {
        "official_laps": 71,
        "expected_sector_records": 213,
        "observed_sector_records": 213 - counts["missing_record"],
        "missing_sector_records": counts["missing_record"],
        "eligible_forecasts": counts["scored"] + counts["censored"],
        "scoreable_forecasts": counts["scored"],
        "censored_targets": counts["censored"],
        "abstentions": counts["abstained"],
    }
    if any(accounting.get(name) != count for name, count in expected_counts.items()):
        raise ValueError("Report accounting disagrees with its complete sector timeline")
    computed = macro_metrics([row for row in predictions if row["scoreable"]], 12)
    stored = report["metrics"]
    if stored.get("forecast_count") != computed["forecast_count"] or set(
        stored.get("macro", {})
    ) != set(computed["macro"]):
        raise ValueError("Report metrics disagree with saved scoreable predictions")
    if any(not _close(value, stored["macro"][name]) for name, value in computed["macro"].items()):
        raise ValueError("Report metrics disagree with saved scoreable predictions")
    return slots, identities, by_id


def _coverage(slots, *, late=False):
    selected = {key: row for key, row in slots.items() if not late or key[0] >= 47}
    counts = Counter(row["status"] for row in selected.values())
    return {
        "expected_slots": len(selected),
        **{name: counts[name] for name in STATUSES},
        "eligible": counts["scored"] + counts["censored"],
        "eligibility_fraction_of_all_slots": (counts["scored"] + counts["censored"])
        / len(selected),
        "scoreable_fraction_of_all_slots": counts["scored"] / len(selected),
    }


def _slice(predictions, ids):
    ordered = sorted(ids)
    return {
        "count": len(ordered),
        "example_ids": ordered,
        "metrics": macro_metrics([predictions[identity] for identity in ordered], 12),
    }


def compare_pit_replays(
    prior_report,
    prior_predictions,
    prior_timeline,
    new_report,
    new_predictions,
    new_timeline,
    *,
    output_path=None,
    prior_freeze=None,
    new_freeze=None,
) -> dict:
    """Validate saved artifacts and separate common, added and lost populations.

    Deltas are new minus prior on exactly the shared scoreable IDs. No confidence
    interval is fabricated from sectors in a single race; neither coverage gain
    nor a changed-population MAE establishes improved forecasting skill.
    """
    inputs = dict(
        zip(
            (
                "prior_report",
                "prior_predictions",
                "prior_timeline",
                "new_report",
                "new_predictions",
                "new_timeline",
            ),
            (
                prior_report,
                prior_predictions,
                prior_timeline,
                new_report,
                new_predictions,
                new_timeline,
            ),
            strict=True,
        )
    )
    prior, new = _read(prior_report), _read(new_report)
    left_slots, left_ids, left = _validate_replay(
        prior, _read(prior_predictions), _read(prior_timeline)
    )
    right_slots, right_ids, right = _validate_replay(
        new, _read(new_predictions), _read(new_timeline)
    )
    if prior.get("synthetic", False) != new.get("synthetic", False):
        raise ValueError("Synthetic and real replays must not be mixed")
    for identity in left_ids.keys() & right_ids.keys():
        if left_ids[identity] != right_ids[identity]:
            raise ValueError("The same example identity changed its official sector slot")
    left_scored = {identity for identity, row in left.items() if row["scoreable"]}
    right_scored = {identity for identity, row in right.items() if row["scoreable"]}
    common = left_scored & right_scored
    for identity in common:
        if left[identity]["target_seconds"] != right[identity]["target_seconds"]:
            raise ValueError("Common-cohort target labels changed; paired metrics are invalid")
        if any(
            left[identity].get(name) != right[identity].get(name) for name in ("origin", "cutoff")
        ):
            raise ValueError("Common-cohort prediction origin or cutoff changed")
    before, after = _slice(left, common), _slice(right, common)
    delta = {
        name: after["metrics"]["macro"][name] - value
        for name, value in before["metrics"]["macro"].items()
    }
    transitions = Counter(
        f"{left_slots[key]['status']}->{right_slots[key]['status']}" for key in sorted(SLOTS)
    )
    changes = {
        "protocol_changed": prior.get("provenance", {}).get("protocol_sha256")
        != new.get("provenance", {}).get("protocol_sha256"),
        "baseline_artifacts_changed": None,
        "encoder_artifacts_changed": None,
        "index_changed": None,
        "development_examples_changed": None,
        "fit_or_calibration_sessions_changed": None,
    }
    if (prior_freeze is None) != (new_freeze is None):
        raise ValueError("Both release freezes are needed to compare model/data provenance")
    if prior_freeze is not None:
        releases = []
        for name, value, replay in (
            ("prior_freeze", prior_freeze, prior),
            ("new_freeze", new_freeze, new),
        ):
            freeze = _read(value)
            unsigned = {key: item for key, item in freeze.items() if key != "manifest_sha256"}
            if freeze.get("manifest_sha256") != _digest(unsigned):
                raise ValueError("Comparison release freeze checksum mismatch")
            provenance = replay.get("provenance", {})
            if (
                provenance.get("release_manifest_sha256") != freeze["manifest_sha256"]
                or provenance.get("protocol_sha256") != freeze["protocol_sha256"]
                or (
                    isinstance(value, (str, Path))
                    and provenance.get("freeze_sha256") != _digest(value)
                )
            ):
                raise ValueError("Comparison replay is not linked to the supplied release freeze")
            inputs[name] = value
            releases.append(freeze)
        before_release, after_release = releases
        for prefix in ("baseline", "encoder"):
            hashes = [
                {
                    key: value
                    for key, value in release["artifact_files"].items()
                    if key.startswith(prefix + "/")
                }
                for release in releases
            ]
            changes[f"{prefix}_artifacts_changed"] = hashes[0] != hashes[1]
        changes["index_changed"] = before_release["index_hash"] != after_release["index_hash"]
        changes["development_examples_changed"] = (
            before_release["training_dataset"]["examples_sha256"]
            != after_release["training_dataset"]["examples_sha256"]
        )
        changes["fit_or_calibration_sessions_changed"] = any(
            before_release[name] != after_release[name]
            for name in ("fit_sessions", "calibration_sessions")
        )
    report = {
        "schema_version": "ant26-pit-replay-comparison-v1",
        "synthetic": prior.get("synthetic", False),
        "race": {"session_key": 11404, "driver_number": 12, "lap_count": 71},
        "coverage": {
            "all_213_slots": {"prior": _coverage(left_slots), "new": _coverage(right_slots)},
            "late_75_slots_laps_47_to_71": {
                "prior": _coverage(left_slots, late=True),
                "new": _coverage(right_slots, late=True),
            },
            "status_transitions": dict(sorted(transitions.items())),
            "newly_eligible": sorted(right.keys() - left.keys()),
            "lost_eligible": sorted(left.keys() - right.keys()),
            "by_lap": [
                {
                    "lap_number": lap,
                    "prior": _coverage(
                        {key: row for key, row in left_slots.items() if key[0] == lap}
                    ),
                    "new": _coverage(
                        {key: row for key, row in right_slots.items() if key[0] == lap}
                    ),
                }
                for lap in range(1, 72)
            ],
        },
        "common_scoreable": {
            "count": len(common),
            "example_ids": sorted(common),
            "prior_metrics": before["metrics"],
            "new_metrics": after["metrics"],
            "delta_new_minus_prior": delta,
            "changed_reference_seconds_count": sum(
                left[i]["reference_seconds"] != right[i]["reference_seconds"] for i in common
            ),
            "paired_labels_identical": True,
        },
        "newly_scoreable": _slice(right, right_scored - left_scored),
        "lost_scoreable": _slice(left, left_scored - right_scored),
        "each_version_scored_population": {"prior": prior["metrics"], "new": new["metrics"]},
        "implementation_and_model_changes": changes,
        "provenance": {
            "input_sha256": {name: _digest(value) for name, value in inputs.items()},
            "prior_freeze_sha256": prior.get("provenance", {}).get("freeze_sha256"),
            "new_freeze_sha256": new.get("provenance", {}).get("freeze_sha256"),
            "prior_claim": prior.get("provenance", {}).get("case_claim"),
            "new_claim": new.get("provenance", {}).get("case_claim"),
            "prior_protocol_sha256": prior.get("provenance", {}).get("protocol_sha256"),
            "new_protocol_sha256": new.get("provenance", {}).get("protocol_sha256"),
        },
        "interpretation": {
            "improvement_claim_allowed": False,
            "coverage_recovery_is_not_forecast_skill_improvement": True,
            "whole_population_metrics_have_different_denominators": left_scored != right_scored,
            "common_cohort_delta_is_descriptive_not_causal": True,
            "timestamp_policy_effect_is_not_isolated_from_model_or_data_changes": True,
            "normalized_errors_use_each_version_reference": True,
            "uncertainty": "one_race_no_sector_bootstrap_or_population_generalization",
        },
    }
    _save(output_path, report)
    return report


def compare_development_examples(
    prior_examples,
    new_examples,
    *,
    fit_sessions=(),
    calibration_sessions=(),
    test_sessions=(),
    output_path=None,
) -> dict:
    """Audit complete development example equality and learned-scope impact."""

    def examples(value):
        value = _read(value)
        return value["examples"] if isinstance(value, dict) else value

    prior, new = examples(prior_examples), examples(new_examples)
    fitting, calibration, testing = set(fit_sessions), set(calibration_sessions), set(test_sessions)
    if fitting & calibration:
        raise ValueError("Fitting and calibration scopes must be disjoint")
    collections = []
    for rows in (prior, new):
        sessions = {row["session_key"] for row in rows}
        validate_scope(rows, sessions, role="pit development comparison")
        by_id = {row["example_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Development examples contain duplicate identities")
        collections.append(by_id)
    left, right = collections
    shared = left.keys() & right.keys()
    changed = {identity for identity in shared if left[identity] != right[identity]}
    affected = changed | (left.keys() ^ right.keys())
    for identity in shared:
        if any(
            left[identity][key] != right[identity][key]
            for key in ("session_key", "driver_number", "lap_number", "sector_index")
        ):
            raise ValueError("Development example identity changed its session/driver/sector")
    changed_fields = Counter(
        key
        for identity in changed
        for key in left[identity].keys() | right[identity].keys()
        if left[identity].get(key) != right[identity].get(key)
    )

    def impact(ids):
        changed_ids = changed & ids
        feature_fields = ("features", "reference_seconds", "origin", "cutoff")
        target_fields = ("target_seconds", "target_valid", "target_exclusion_reasons")
        input_changes = {
            i for i in changed_ids if any(left[i].get(k) != right[i].get(k) for k in feature_fields)
        }
        eligibility_changes = {
            i for i in changed_ids if left[i].get("eligible") != right[i].get("eligible")
        }
        reason_changes = {
            i
            for i in changed_ids
            if left[i].get("exclusion_reasons") != right[i].get("exclusion_reasons")
        }
        target_changes = {
            i for i in changed_ids if any(left[i].get(k) != right[i].get(k) for k in target_fields)
        }
        substantive = input_changes | eligibility_changes | reason_changes | target_changes
        shared_ids = shared & ids
        return {
            "changed_examples": len(changed_ids),
            "forecast_input_changed_examples": len(input_changes),
            "eligibility_changed_examples": len(eligibility_changes),
            "eligibility_reason_changed_examples": len(reason_changes),
            "target_changed_examples": len(target_changes),
            "metadata_only_changed_examples": len(changed_ids - substantive),
            "substantive_changed_examples": len(substantive),
            "newly_eligible_examples": sum(
                not left[i].get("eligible") and bool(right[i].get("eligible")) for i in shared_ids
            ),
            "lost_eligible_examples": sum(
                bool(left[i].get("eligible")) and not right[i].get("eligible") for i in shared_ids
            ),
            "changed_top_level_fields": dict(
                sorted(
                    Counter(
                        key
                        for i in changed_ids
                        for key in left[i].keys() | right[i].keys()
                        if left[i].get(key) != right[i].get(key)
                    ).items()
                )
            ),
        }

    scope_ids = {
        name: {
            identity
            for identity in left.keys() | right.keys()
            if (left.get(identity) or right[identity])["session_key"] in sessions
        }
        for name, sessions in (
            ("final_fit", fitting),
            ("final_calibration", calibration),
            ("rolling_test", testing),
        )
    }

    def count(rows):
        return {
            "examples": len(rows),
            "eligible": sum(bool(row.get("eligible")) for row in rows),
            "scoreable": len(valid_examples(rows)),
        }

    all_keys = sorted({(row["session_key"], row["driver_number"]) for row in prior + new})
    touched_sessions = {
        row["session_key"]
        for identity in affected
        for row in (left.get(identity), right.get(identity))
        if row is not None
    }
    report = {
        "schema_version": "ant26-pit-development-comparison-v1",
        "input_sha256": {"prior": _digest(prior_examples), "new": _digest(new_examples)},
        "canonical_examples_sha256": {"prior": _digest(prior), "new": _digest(new)},
        "all_example_values_and_order_unchanged": prior == new,
        "changed_example_count": len(changed),
        "added_example_count": len(right.keys() - left.keys()),
        "removed_example_count": len(left.keys() - right.keys()),
        "changed_top_level_fields": dict(sorted(changed_fields.items())),
        "change_impact": impact(set(shared)),
        "change_impact_by_scope": {name: impact(ids) for name, ids in scope_ids.items()},
        "scope_note": "Final fit and calibration are disjoint; rolling test races can overlap final-fit races because the final candidate is refitted after development evaluation.",
        "changed_sessions": sorted(touched_sessions),
        "fit_scope_changed": bool(touched_sessions & fitting),
        "calibration_scope_changed": bool(touched_sessions & calibration),
        "fit_sessions": sorted(fitting),
        "calibration_sessions": sorted(calibration),
        "rolling_test_sessions": sorted(testing),
        "learned_artifact_reuse_allowed": not affected and prior == new,
        "prior_counts": count(prior),
        "new_counts": count(new),
        "by_driver_race": [
            {
                "session_key": session,
                "driver_number": driver,
                "prior": count(
                    [
                        r
                        for r in prior
                        if (r["session_key"], r["driver_number"]) == (session, driver)
                    ]
                ),
                "new": count(
                    [r for r in new if (r["session_key"], r["driver_number"]) == (session, driver)]
                ),
            }
            for session, driver in all_keys
        ],
        "interpretation": "Changed fitting/calibration examples require new learned artifacts with the fixed configuration; historical experiments remain tied to their original feature policy.",
    }
    _save(output_path, report)
    return report
