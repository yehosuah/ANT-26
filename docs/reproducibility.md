# Reproducing ANT-26

A clean checkout can reproduce the offline synthetic demonstration and, with public provider access, acquire a new real-data revision. Exact historical source bytes and their original hashes are not guaranteed to remain available. The published results describe their pinned revision; a new acquisition is a new experiment, even when the modeling settings match.

## Offline code and demonstration

Use Python 3.11 and uv. Run from the repository root:

```bash
uv sync --locked --group dev --extra app --no-editable
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
uv run --no-sync ant26 protocol-check
uv run --no-sync ant26 smoke --output artifacts/my-synthetic-check
```

The smoke run is explicitly synthetic. It validates integration and cannot reproduce Antonelli's measured performance. Choose a new smoke output path for another run. Optional ML/UI tests require their respective extras; skipped ML tests are not model validation.

## New real development revision

Install ML dependencies, then choose a unique run ID:

```bash
uv sync --locked --group dev --extra ml --extra app --no-editable
scripts/reproduce-development.sh development --run-id source-revision-001
```

This stage performs public provider requests. It records a new catalog and immutable source cache under `data/reproduction/source-revision-001/`, ingests all verified pre-Mexico races for drivers 12 and 63, and materializes both cohorts. It captures the dataset path emitted by the CLI instead of guessing its content-addressed filename. The merged dataset identifies its exact Antonelli source cohort.

The script then runs both baseline comparisons, the Antonelli experiment with fine-tuned/frozen/scratch modes (120 fits), and Mercedes fine-tuning (40 fits). Both use seeds 17, 29, 43, 71 and 101, three reconstruction epochs for pretrained modes, and five forecast epochs. Scratch skips reconstruction training. Candidate preparation retains the development WIS-selected CatBoost forecaster and a separate continuous-retrieval encoder/index. It fails if the selected forecaster changes unexpectedly; it does not silently choose another model.

Artifacts and CLI result logs are saved under `artifacts/reproduction/source-revision-001/`. `paths.json` records emitted dataset paths and the source envelope hash. A changed source revision can change availability, eligible counts, errors and even whether the declared experiment can complete. The commands never relax fixed quality or scope requirements to force the original result.

The wrapper refuses existing run directories. After an interrupted acquisition, preserve the run and inspect its log/progress files. Individual `ant26 ingest` calls can resume against that run's exact catalog, cache and progress paths. Materialization and candidate outputs retain their own integrity/immutability checks. Do not delete or replace completed artifacts to make a rerun succeed. Use a new run ID when source bytes, code or protocol change. Avoid concurrent provider acquisitions with separate caches; request-budget coordination is shared within a cache root.

## Explicit release freeze

Development completion leaves the candidate unfrozen. Inspect the reports and record the actual release choice before running:

```bash
scripts/reproduce-development.sh freeze --run-id source-revision-001 \
  --freeze artifacts/reproduction/source-revision-001/release-freeze.json
```

This command records the explicit experimental continuous-atlas decision: retrieval remains unvalidated, no human ratings are invented, and no discrete driving-style claims are enabled. Freezing permits the predeclared historical case study; it does not establish a validated product or a forecast made before the race. The freeze pins model/index bytes, code/protocol provenance, training snapshots and fitting/calibration scopes. Candidate metadata is preserved. The output path must be new.

## Opt in to the final case study

Mexico is excluded from every preceding development stage. Its acquisition and scoring require both an existing verified freeze file and explicit opt-in:

```bash
scripts/reproduce-development.sh case-study --run-id source-revision-001 \
  --freeze artifacts/reproduction/source-revision-001/release-freeze.json \
  --include-final-case
```

The wrapper calls `ant26 case-ingest`, captures its emitted dataset path despite progress output, then calls `ant26 case-score`. The release is verified before acquisition. Existing case outputs are refused. Scoring uses the frozen CatBoost model and reference index; no test-weight or index updates are authorized. Report missing laps/sectors, origin abstentions and censored targets alongside scored errors. Mexico remains a retrospective case study with the exploratory-inspection limitation declared in the protocol.

## What this does and does not reproduce

The repository publishes code, a locked dependency resolution and safe aggregate results. Source responses, processed examples, weights, raw predictions, reference indexes, blinded answer keys and reviewer responses are intentionally excluded from Git. These commands generate local research assets; they do not distribute them. There is no portable public bundle of the exact original real assets.

Deterministic seeds and artifact hashes make one revision inspectable. They do not promise bit-identical numerical results across CPU architectures, library builds or provider revisions. The original benchmark source/input/protocol hashes are recorded in `reports/development-results.json`. Compare those hashes before claiming an exact reproduction. Aggregate differences on a new source revision should be reported as new results.

Script validation uses offline command stubs for orchestration and safety guards; this documentation does not claim that a second complete real acquisition was executed. No reproduction validation requests Mexico data.
