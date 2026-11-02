# ANT-26 execution record

User authorized implementation and parallel independent work on November 1, 2026, then explicitly requested publication to a public repository under yehosuah. The implementation plan remains the scope. No paid compute is authorized.

## Shared interfaces

- Canonical source rows are plain JSON-compatible dictionaries. Times are timezone-aware UTC ISO strings.
- Forecast examples are dictionaries with `example_id`, `session_key`, `driver_number`, `lap_number`, `sector_index`, `origin`, `cutoff`, `reference_seconds`, `target_seconds`, `features`, `eligible`, `exclusion_reasons`, and `provenance`.
- Targets never enter `features`; labels and origin construction remain separate from historical inputs.
- Every learned step excludes Mexico meeting 1298 and all its sessions. Race 11404 is the final case study.
- Module ownership prevents parallel writes. Root owns packaging, CLI, integration, and plan/status files; ingestion, datasets, and modeling have separate owners.

## Active work

- [x] M0 Lock validated source catalog and protocol
- [x] M1 Implement immutable ingestion and audit the initial Mercedes cohort
- [x] M2 Implement past-only features and leakage tests
- [x] M3 Implement rolling splits, metrics, baselines and initial cohort benchmark
- [ ] M4 Implement and validate representation training
- [ ] M5 Implement and validate atlas retrieval
- [ ] M6 Integrate API and replay application
- [ ] M7 Freeze model and run Mexico case study

Milestones are marked complete only after integration checks and the corresponding evidence exist. Component completion does not establish model quality or full-season data availability.

M1 through M3 are complete for the declared initial Mercedes cohort: 18 Antonelli races and 17 scoreable Russell races. Full-grid representation pretraining remains an extension of the original population scope; no full-grid claim is made.

## Reconciliation evidence

- M0: 18 verified pre-Mexico development race entries; canceled races excluded; Mexico meeting 1298 quarantined. Official Formula 1 page resolves session 11731 as Bahrain GP in Malaysia. Canonical override preserves original fields.
- M1 component: immutable cached ingestion, integrity checks, retry/backoff and source-quality audit implemented; 9 offline tests passed. Real three-race audit is ongoing; full-session telemetry response timeouts require bounded downloads.
- M2 component: 12 tests passed for alignment, past-only inputs, missingness, future mutation, pit assumptions and separate target censoring. Full-season retention gate remains pending.
- M3 component: 10 tests passed for splits, scope guards, metrics, baselines and censoring. Synthetic integration smoke ran two chronological test folds across 12 synthetic races; this is not real F1 model evidence.
- M4 component: 559,104-parameter default encoder; reduced synthetic CPU reconstruction loss decreased from 3.7861056 to 2.9098163 in 10 epochs. Four representation tests passed. Real five-seed predictive gains remain unmeasured.
- M6 component: read-only artifact API and source-to-feature-to-baseline-to-API integration tests passed. UI and Docker require separate runtime verification.

## Reconciled progress memo

- Three-race source feasibility audit complete: 39 session queries, zero endpoint errors, Mexico untouched. The 60-second samples yield only five scoreable sectors and are not a training dataset.
- First full source trace complete: Austin session 11396, Antonelli 12, 96 chunks and 21,252 samples. Feature build `examples-da11a019ec7aa533` produced 168 sectors, 164 nominal green opportunities, 162 eligible/scoreable sectors, and retention 0.9878048780.
- Full trace expansion continues resumably for the remaining recent races and drivers. Full-season retention and six-race evaluation gates are still pending.
- Reconciliation fixed historical-timing leakage, target censoring, protocol hash mismatch, future-date fitting, exact encoder/index snapshot matching, and API pagination.
- Synthetic CLI flow verified: baseline smoke, default encoder training, disjoint calibration, artifact save/load, and five-neighbor atlas query. All outputs explicitly labeled synthetic.
- Streamlit AppTest verified API-backed run/model/session selection, forecasts, provenance, synthetic warning and saved atlas rendering. Docker daemon is unavailable; image build remains unverified.

## Full development cohort checkpoint

- Source acquisition: 760,694 native telemetry rows, 35 scoreable driver-race traces and one explicitly unscoreable Russell Belgian race. Every source and normalized CSV has a checksum; Mexico remains excluded.
- Materialized Mercedes cohort: `data/processed/mercedes/merged-e7ed15ec6216606e.json`, 6,204 observed sectors and 5,008 scoreable examples. Feature-code checksums now invalidate stale materialization caches.
- Classification crosscheck: 36 driver-race comparisons against the provider's result endpoint; no missing completed-lap records, five extra partial tail records. Independent FIA verification remains separate.
- Real benchmark: eight held-out Antonelli races, 1,039 identical scored examples per baseline. Best point MAE is persistence at 0.94568 seconds; best weighted interval score is CatBoost at 0.62862, with 90% interval coverage 0.92996. Results describe this Mercedes cohort.
- Real diagnostic coverage: 1,040 of 1,235 observed nominal green test origins retained (84.21%); one eligible censored target. Hungary's source values of 104 are masked and its selective coverage is reported.
- A full first neural experiment exposed a constant-column normalization defect. Its checkpoints, source snapshots and invalidation record are preserved; the API excludes invalidated experiments. Regression-tested policy `constant_unit_scale_v2` uses unit scale for near-zero fitted variance and rejects old checkpoints. Fixed-budget corrected experiments are running separately.
- Atlas discrete patterns failed the first bounded stability check (median ARI 0.32123 versus 0.6 target on one selective fold); continuous retrieval remains experimental. Blind review preparation is underway; no human preferences have been fabricated.
