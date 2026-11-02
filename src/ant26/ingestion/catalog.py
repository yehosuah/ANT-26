"""Catalog filtering keeps the final case study out of development data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("catalog dates must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def build_catalog(
    sessions: list[dict[str, Any]],
    meetings: list[dict[str, Any]] | None = None,
    *,
    as_of: str = "2026-11-01T23:59:59+00:00",
    quarantined_session_keys: tuple[int, ...] | list[int] = (11731,),
    overrides: dict[str, Any] | None = None,
    final_meeting_key: int = 1298,
) -> dict[str, Any]:
    """Return race catalog with sourced exceptions, cancellations and future excluded.

    Source session rows are never mutated. A validated override may release an
    otherwise quarantined entry, while preserving its original country and circuit.
    """
    meeting_lookup = {row["meeting_key"]: row for row in (meetings or [])}
    override_rows = (overrides or {}).get("sessions", {})
    development, final, excluded = [], [], []
    seen_sessions: set[int] = set()
    seen_meetings: set[int] = set()
    for source in sorted(sessions, key=lambda row: row.get("date_start", "")):
        row = dict(source)
        if row.get("session_name") != "Race" or row.get("year") != 2026:
            continue
        key, meeting_key = row.get("session_key"), row.get("meeting_key")
        reasons = []
        if key in seen_sessions or meeting_key in seen_meetings:
            reasons.append("duplicate_session_or_meeting")
        seen_sessions.add(key)
        override = override_rows.get(str(key), {})
        resolved = bool(override.get("verified_identity") and override.get("sources"))
        if key in quarantined_session_keys and not resolved:
            reasons.append("unresolved_event_identity")
        if row.get("is_cancelled"):
            reasons.append("cancelled")
        try:
            if _utc(row["date_start"]) > _utc(as_of):
                reasons.append("future_session")
        except (KeyError, ValueError, TypeError):
            reasons.append("invalid_date_start")
        if key is None or meeting_key is None:
            reasons.append("missing_identity")
        row["verified_identity"] = not reasons
        row["canonical_host_country"] = override.get("host_country", row.get("country_name"))
        row["canonical_circuit"] = override.get("circuit", row.get("circuit_short_name"))
        row["identity_source"] = override.get("sources", ["openf1_session_meeting_consistency"])
        meeting = meeting_lookup.get(meeting_key)
        if meeting and meeting.get("circuit_key") != row.get("circuit_key"):
            reasons.append("session_meeting_circuit_mismatch")
            row["verified_identity"] = False
        if reasons:
            row["exclusion_reasons"] = reasons
            excluded.append(row)
        elif meeting_key == final_meeting_key:
            final.append(row)
        else:
            development.append(row)
            seen_meetings.add(meeting_key)
    return {
        "schema_version": "ant26-catalog-v1",
        "as_of": as_of,
        "development": development,
        "final_case_study": final,
        "excluded": excluded,
        "final_meeting_quarantined": final_meeting_key,
    }
