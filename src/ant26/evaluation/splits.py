"""Build chronological roles without splitting any race weekend."""

from __future__ import annotations

from datetime import datetime, timezone

MEXICO_MEETING = 1298
MEXICO_SESSIONS = frozenset({11397, 11398, 11399, 11400, 11404})
MEXICO_START = datetime(2026, 11, 1, 20, tzinfo=timezone.utc)
MEXICO_WEEKEND_START = datetime(2026, 10, 30, tzinfo=timezone.utc)


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def is_mexico(row: dict) -> bool:
    meeting = row.get("meeting_key", row.get("provenance", {}).get("meeting_key"))
    return meeting == MEXICO_MEETING or row.get("session_key") in MEXICO_SESSIONS


def validate_scope(examples: list[dict], allowed_sessions: set[int], *, role: str) -> None:
    """Validate all supplied rows, including ineligible rows, before any fitting."""
    if not allowed_sessions:
        raise ValueError(f"{role} requires an explicit nonempty session allowlist")
    if allowed_sessions.intersection(MEXICO_SESSIONS):
        raise ValueError(f"Mexico is forbidden for {role}")
    for example in examples:
        if is_mexico(example):
            raise ValueError(f"Mexico is forbidden for {role}")
        if example.get("session_key") not in allowed_sessions:
            raise ValueError(f"Unexpected session {example.get('session_key')} in {role}")
        for name in ("origin", "cutoff"):
            value = example.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{role} requires timezone-aware {name}")
                if utc(value) >= MEXICO_WEEKEND_START:
                    raise ValueError(
                        f"Mexico-weekend and later timestamps are forbidden for {role}"
                    )


def rolling_splits(catalog: dict | list[dict], protocol: dict | None = None) -> list[dict]:
    """Fit 6+, tune 2, calibrate 2, test 1; group every session by meeting.

    Catalog entries must be resolved development sessions. Explicitly unverified,
    canceled, quarantined, Mexico, and future entries cannot become fitting data.
    """
    protocol = protocol or {}
    split_config = protocol.get("splits", {})
    quarantine = set(protocol.get("source", {}).get("event_identity_quarantine", [11731]))
    require_verified = protocol.get("source", {}).get("require_verified_event_identity", True)
    rows = catalog.get("development", []) if isinstance(catalog, dict) else catalog
    groups: dict[int, list[dict]] = {}
    seen: set[int] = set()
    for row in rows:
        key = row["session_key"]
        if key in seen:
            raise ValueError(f"Duplicate catalog session {key}")
        seen.add(key)
        if (
            is_mexico(row)
            or key in quarantine
            or row.get("is_cancelled", False)
            or (require_verified and row.get("verified_identity") is not True)
            or row.get("verified_identity") is False
            or utc(row["date_start"]) >= MEXICO_START
        ):
            continue
        groups.setdefault(row["meeting_key"], []).append(row)
    races = []
    for meeting, entries in groups.items():
        race_rows = [row for row in entries if row.get("session_name", "Race") == "Race"]
        if len(race_rows) != 1:
            raise ValueError(f"Meeting {meeting} must have exactly one race")
        races.append(
            {
                "meeting_key": meeting,
                "date_start": race_rows[0]["date_start"],
                "session_keys": sorted(row["session_key"] for row in entries),
            }
        )
    races.sort(key=lambda row: utc(row["date_start"]))
    tune_n = split_config.get("tuning_races_immediately_before_calibration", 2)
    cal_n = split_config.get("calibration_races_immediately_before_test", 2)
    first_test = split_config.get("first_rolling_test_race_one_based", 11) - 1
    fit_min = split_config.get("minimum_initial_fitting_races", 6)
    if tune_n < 1 or cal_n < 1 or first_test - tune_n - cal_n < fit_min:
        raise ValueError("Invalid chronological split contract")
    folds = []
    for i in range(first_test, len(races)):
        roles = {
            "fit": races[: i - tune_n - cal_n],
            "tune": races[i - tune_n - cal_n : i - cal_n],
            "calibration": races[i - cal_n : i],
            "test": races[i : i + 1],
        }
        fold = {"fold_id": f"rolling-{i + 1:02d}"}
        for role, items in roles.items():
            fold[f"{role}_sessions"] = [key for item in items for key in item["session_keys"]]
            fold[f"{role}_meetings"] = [item["meeting_key"] for item in items]
        folds.append(fold)
    return folds
