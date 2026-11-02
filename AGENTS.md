# ANT-26 working instructions

ANT-26 is a research portfolio combining observed F1 telemetry retrieval and probabilistic next-sector forecasting. The public repository is `yehosuah/ANT-26`.

## Research boundaries

- Read `docs/execution.md` and `configs/evaluation-protocol.json` before changing model or data behavior.
- Mexico meeting 1298 is the final case study. Its entire weekend is excluded from fitting, preprocessing, calibration and retrieval references. Do not ingest or evaluate it until the model, index, thresholds and review decision are recorded as frozen.
- Current source population is Antonelli and Russell, not the full grid. Report that scope explicitly.
- Predictions use reconstructed event times; original live-publication latency is unknown. Inputs must precede each cutoff. Labels and realized flags are not inputs.
- Keep all fixed seeds and held-out races visible, including Hungary's selective coverage. An executed experiment is not evidence of superiority. Keep failed and invalidated runs separate.
- Normalization policy is `constant_unit_scale_v2`. Legacy checkpoints are refused. Remaining near-constant/OOD failures are documented; the final forecasting candidate is a baseline.
- Human atlas preferences must come from explicit ratings. Blank forms, generated neighbors and automated stability checks are not human validation.

## Commands

```bash
uv sync --locked --group dev --no-editable
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
uv run --no-sync ant26 protocol-check
uv run --no-sync ant26 smoke --output artifacts/new-smoke-run
```

Mypy currently covers domain and API boundaries. Optional ML/UI checks require the `ml` and `app` extras. Docker Compose runs the API and replay UI; GitHub CI checks their container health.

## Data and artifacts

- Raw data, trained weights, predictions, blind answer keys and reviewer responses belong under ignored `data/` or `artifacts/` paths.
- Publish code and safe aggregate results; do not add raw datasets, credentials, private review keys or personal paths to Git.
- Pin exact source snapshots, protocol hashes, feature-builder code hashes and model fitting scopes. Never replace an immutable artifact with different bytes.
- Use `ant26 materialize` to build from exact source manifests. `ant26 benchmark`, `representation-benchmark` and `diagnose` operate on versioned datasets.

## Collaboration

Keep progress updates as precise memos: what completed, what passed, and what remains blocked. Respect the user's requested parallel work, assign disjoint files, and reconcile in the main chat before publishing. Keep milestone evidence in `docs/execution.md`; distinguish completed components from unmet scientific gates.
