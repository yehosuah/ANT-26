"""Reproduce retrospective driving comparisons from pinned, local source snapshots.

No source acquisition, model fitting, forecast changes or Mexico index updates.
Run from the repository root with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ant26.research.audit import run_additional_audit
from ant26.research.comparison import run_behavior_comparison
from ant26.research.mexico import summarize_mexico
from ant26.research.telemetry import load_observations


def write_json(path: Path, value: dict) -> None:
    """Preserve existing bytes; a changed result requires a new output path."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing analysis differs; choose a new output path: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", default="data/audit/bulk-progress.json")
    parser.add_argument("--catalog", default="data/audit/catalog.json")
    parser.add_argument(
        "--mexico",
        default="data/case-study/mexico-pit-v0.2.0/examples-3575246c1f2b3895.json",
    )
    parser.add_argument("--output", default="artifacts/behavior-v1")
    parser.add_argument("--public-report", default="reports/driver-behavior.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    observations = load_observations(args.progress, args.catalog, args.mexico)
    write_json(output / "observations.json", observations)
    report = run_behavior_comparison(
        output / "observations.json",
        output_path=output / "comparison.json",
        public_output_path=args.public_report,
        seed=17,
        resamples=10000,
    )
    mexico_report = Path(args.public_report).with_name("mexico-driving-behavior.json")
    write_json(mexico_report, summarize_mexico(observations))
    write_json(
        Path(args.public_report).with_name("driver-behavior-audit.json"),
        run_additional_audit(output / "observations.json"),
    )
    print(
        json.dumps(
            {"matched_races": report["primary"]["race_count"], "findings": report["findings"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
