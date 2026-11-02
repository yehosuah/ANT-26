#!/usr/bin/env python3
"""Summarize private Mexico sector observations without publishing source rows."""

import argparse
import json
from pathlib import Path

from ant26.research.mexico import summarize_mexico


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new summary path; existing evidence is preserved")
    summary = summarize_mexico(json.loads(Path(args.observations).read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "accounting": summary["accounting"]}))


if __name__ == "__main__":
    main()
