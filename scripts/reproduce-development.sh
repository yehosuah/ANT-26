#!/bin/sh
# Fresh source revisions get fresh run directories. No credentials are required.
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

usage() {
    cat <<'TEXT'
Usage:
  scripts/reproduce-development.sh development --run-id NAME
  scripts/reproduce-development.sh freeze --run-id NAME --freeze PATH
  scripts/reproduce-development.sh case-study --run-id NAME --freeze PATH --include-final-case

development fetches public pre-Mexico data, materializes both cohorts, runs the
fixed 160-fit experiments, and prepares candidates. It never freezes or fetches
Mexico. Existing run directories are refused. Network calls can take minutes.

freeze records the explicit continuous-retrieval-only release decision. Read
the release limitations first. The candidate artifacts are not modified.

case-study fetches and scores Mexico only when both the opt-in flag and a
verified existing freeze file are supplied. Each case output must be new.
TEXT
}

die() { printf '%s\n' "$*" >&2; exit 1; }

if [ "$#" -eq 0 ]; then usage; exit 2; fi
stage=$1
shift
run_id=
freeze_path=
include_final_case=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id)
            [ "$#" -ge 2 ] || die '--run-id requires a value'
            run_id=$2; shift 2 ;;
        --freeze)
            [ "$#" -ge 2 ] || die '--freeze requires a value'
            freeze_path=$2; shift 2 ;;
        --include-final-case) include_final_case=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
case "$stage" in development|freeze|case-study) ;; --help|-h) usage; exit 0 ;; *) die 'Unknown stage' ;; esac
case "$run_id" in ''|*[!A-Za-z0-9_-]*) die 'Use a run ID containing only letters, digits, underscores or hyphens' ;; esac
[ "${#run_id}" -le 80 ] || die 'Run ID must be at most 80 characters'
if [ "$stage" = case-study ]; then
    [ "$include_final_case" = true ] || die 'Mexico requires explicit --include-final-case'
    [ -n "$freeze_path" ] && [ -f "$freeze_path" ] || die 'Mexico requires an existing --freeze file'
else
    [ "$include_final_case" = false ] || die '--include-final-case is only valid for case-study'
fi
if [ "$stage" = freeze ]; then
    [ -n "$freeze_path" ] || die 'freeze requires an explicit --freeze output path'
    [ ! -e "$freeze_path" ] || die 'Freeze output already exists; replacement is forbidden'
fi
if [ "$stage" = development ]; then
    [ -z "$freeze_path" ] || die 'development does not accept a freeze path'
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
command -v uv >/dev/null 2>&1 || die 'Install uv and run the locked ML dependency sync first'

data_root="data/reproduction/$run_id"
artifact_root="artifacts/reproduction/$run_id"
paths_file="$artifact_root/paths.json"
cli() { uv run --no-sync python -m ant26.cli "$@"; }

# CLI output can include progress JSON. Read the last object containing the
# required emitted key rather than guessing a content-addressed filename.
emitted_path() {
    uv run --no-sync python - "$1" <<'PY'
import json
import sys
from pathlib import Path

for line in reversed(Path(sys.argv[1]).read_text().splitlines()):
    try:
        record = json.loads(line)
    except ValueError:
        continue
    if isinstance(record, dict) and isinstance(record.get("path"), str):
        path = Path(record["path"])
        if not path.is_file():
            raise SystemExit("The emitted dataset path does not exist")
        print(path)
        break
else:
    raise SystemExit("CLI did not emit a dataset path")
PY
}

saved_path() {
    uv run --no-sync python - "$paths_file" "$1" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text())[sys.argv[2]]
if not isinstance(value, str) or not Path(value).is_file():
    raise SystemExit("Recorded dataset path is missing")
print(value)
PY
}

case "$stage" in
    development)
        [ ! -e "$data_root" ] && [ ! -e "$artifact_root" ] || die 'Run already exists; use a new run ID for a new revision'
        mkdir -p "$data_root" "$artifact_root"
        git rev-parse HEAD > "$artifact_root/code-revision.txt"
        cli protocol-check > "$artifact_root/protocol-check.json"
        cli audit --cache-root "$data_root/raw" --output "$data_root/audit" \
            > "$artifact_root/audit-result.json"
        cli ingest --all-development --drivers 12 63 --format csv \
            --cache-root "$data_root/raw" --catalog "$data_root/audit/catalog.json" \
            --progress "$data_root/bulk-progress.json" > "$artifact_root/ingest-result.json"
        cli materialize --drivers 12 63 --progress "$data_root/bulk-progress.json" \
            --catalog "$data_root/audit/catalog.json" --output "$data_root/processed" \
            > "$artifact_root/materialize-result.jsonl"
        dataset=$(emitted_path "$artifact_root/materialize-result.jsonl")
        uv run --no-sync python - "$dataset" "$paths_file" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
payload = json.loads(dataset.read_text())
matches = [row["path"] for row in payload["source_datasets"] if row["cohort_driver_numbers"] == [12]]
if len(matches) != 1 or not Path(matches[0]).is_file():
    raise SystemExit("Merged dataset must emit exactly one existing Antonelli source cohort")
result = {
    "mercedes_dataset": str(dataset),
    "antonelli_dataset": matches[0],
    "source_envelope_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    "scope": "new_source_revision_not_a_guarantee_of_original_historical_bytes",
}
with Path(sys.argv[2]).open("x") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
        antonelli_dataset=$(saved_path antonelli_dataset)
        cli benchmark --examples "$antonelli_dataset" --catalog "$data_root/audit/catalog.json" \
            --catboost --output "$artifact_root/antonelli-baselines" > "$artifact_root/antonelli-baseline-result.json"
        cli representation-benchmark --examples "$antonelli_dataset" \
            --catalog "$data_root/audit/catalog.json" --seeds 17 29 43 71 101 \
            --modes finetune frozen scratch --pretrain-epochs 3 --forecast-epochs 5 \
            --baseline-report "$artifact_root/antonelli-baselines/report.json" \
            --output "$artifact_root/antonelli-representations" > "$artifact_root/antonelli-representation-result.json"
        cli benchmark --examples "$dataset" --catalog "$data_root/audit/catalog.json" \
            --catboost --output "$artifact_root/mercedes-baselines" > "$artifact_root/mercedes-baseline-result.json"
        cli representation-benchmark --examples "$dataset" \
            --catalog "$data_root/audit/catalog.json" --seeds 17 29 43 71 101 \
            --modes finetune --pretrain-epochs 3 --forecast-epochs 5 \
            --baseline-report "$artifact_root/mercedes-baselines/report.json" \
            --output "$artifact_root/mercedes-representations" > "$artifact_root/mercedes-representation-result.json"
        cli prepare-release --examples "$dataset" --catalog "$data_root/audit/catalog.json" \
            --baseline-report "$artifact_root/mercedes-baselines/report.json" \
            --output "$artifact_root/candidate" > "$artifact_root/prepare-release-result.json"
        printf 'Development complete: %s\nCandidate remains unfrozen; Mexico was excluded.\n' "$paths_file"
        ;;
    freeze)
        [ -f "$paths_file" ] || die 'Complete the development stage first'
        dataset=$(saved_path mercedes_dataset)
        cli freeze --candidate-root "$artifact_root/candidate" --examples "$dataset" \
            --code-revision "$(git rev-parse HEAD)" --output "$freeze_path"
        ;;
    case-study)
        [ -f "$paths_file" ] || die 'Complete the development stage first'
        case_data="$data_root/mexico"
        case_artifacts="$artifact_root/mexico"
        [ ! -e "$case_data" ] && [ ! -e "$case_artifacts" ] || die 'Case outputs already exist; preserve them and resume manually from pinned progress'
        mkdir -p "$case_artifacts"
        cli case-ingest --freeze "$freeze_path" --cache-root "$data_root/raw" \
            --output "$case_data" > "$case_artifacts/ingest-result.jsonl"
        case_dataset=$(emitted_path "$case_artifacts/ingest-result.jsonl")
        cli case-score --freeze "$freeze_path" --examples "$case_dataset" \
            --output "$case_artifacts/scored" > "$case_artifacts/score-result.json"
        printf 'Case study complete: %s\n' "$case_artifacts"
        ;;
esac
