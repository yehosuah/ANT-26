"""Deterministic synthetic sources for offline integration, never research evidence."""

import math
from datetime import datetime, timedelta, timezone

from ant26.datasets import build_examples


def synthetic_examples(protocol: dict, race_count: int = 12) -> tuple[list[dict], dict]:
    examples, catalog = [], []
    for race in range(race_count):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=race)
        key = 1000 + race
        catalog.append(
            {
                "session_key": key,
                "meeting_key": 500 + race,
                "date_start": start.isoformat(),
                "session_name": "Race",
                "year": 2026,
                "verified_identity": True,
            }
        )
        laps, car = [], []
        for driver in (12, 63):
            clock = 0.0
            for lap in range(1, 8):
                sectors = [10 + 0.1 * math.sin(lap + sector + race / 3) for sector in range(3)]
                laps.append(
                    {
                        "session_key": key,
                        "meeting_key": 500 + race,
                        "driver_number": driver,
                        "lap_number": lap,
                        "date_start": (start + timedelta(seconds=clock)).isoformat(),
                        "lap_duration": sum(sectors),
                        **{f"duration_sector_{i + 1}": s for i, s in enumerate(sectors)},
                        "is_pit_out_lap": False,
                    }
                )
                clock += sum(sectors)
            for step in range(int(clock * 4)):
                t = step / 4
                car.append(
                    {
                        "session_key": key,
                        "driver_number": driver,
                        "date": (start + timedelta(seconds=t)).isoformat(),
                        "speed": 220 + 30 * math.sin(t / 4),
                        "throttle": 80 + 20 * math.cos(t / 4),
                        "brake": 100 if step % 100 < 8 else 0,
                        "rpm": 10000 + 500 * math.sin(t / 4),
                        "n_gear": 6 if step % 100 > 10 else 5,
                    }
                )
        flags = [{"session_key": key, "date": start.isoformat(), "scope": "Track", "flag": "GREEN"}]
        rows = build_examples(laps, car, race_control=flags, pit=[], protocol=protocol)
        for row in rows:
            row["provenance"]["data_kind"] = "synthetic_integration_fixture"
        examples.extend(rows)
    return examples, {"development": catalog, "data_kind": "synthetic_integration_fixture"}
