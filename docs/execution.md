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
- [x] M4 Run corrected fixed-budget representation experiments and record the baseline decision
- [ ] M5 Implement and validate atlas retrieval
- [x] M6 Integrate API and replay application; container health verified in GitHub CI
- [x] M7 Freeze and publish Mexico case study, with the failed first attempt and diagnostic correction disclosed

Milestones are marked complete only after integration checks and the corresponding evidence exist. Component completion does not establish model quality or full-season data availability.

M1 through M3 are complete for the declared initial Mercedes cohort: 18 Antonelli races and 17 scoreable Russell races. Full-grid representation pretraining remains an extension of the original population scope; no full-grid claim is made.

## Reconciliation evidence

The next two sections preserve early implementation checkpoints. Their pending statuses are superseded by the full-cohort and corrected-experiment checkpoints below.

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

## Corrected experiment and review checkpoint

- Completed 160 corrected fixed-budget runs: 120 Antonelli-only runs (8 races x 5 seeds x 3 modes) and 40 Mercedes-cohort fine-tuning runs (8 races x 5 seeds). Pretrained modes used 3 reconstruction epochs; all modes used 5 forecast epochs. The numerical correction changed neither seeds nor budgets.
- Antonelli-only fine-tuning: race-macro MAE 0.97827 seconds and WIS 0.68379. Best respective baselines were 0.94552 and 0.64307. Its paired improvement intervals cross zero and both 5% gates fail.
- Mercedes fine-tuning: primary Antonelli MAE 0.99748 and WIS 0.77681, also failing the improvement gates. Secondary all-driver tail failures remain visible. These results do not justify deploying the learned forecast head.
- Final candidate forecasting bundle is native CatBoost with validated JSON; seed 17, first 16 verified races fitted, Singapore and Austin calibrated. Selection optimizes WIS while accepting a measured 3.6951% point-MAE tradeoff. The separate continuous-retrieval encoder and frozen-reference candidate index are prepared with exact fitting snapshots.
- All final artifacts still say `candidate_not_frozen_for_mexico`; Mexico was not queried. Candidate preparation closes technical work but does not bypass the atlas review gate.
- Sixty corrected blind comparisons across eight held-out races are ready, with trace overlays and blank ratings. `app/review.py` is the explicit-save review UI. There are no human preferences yet; M5 stays open and M7 remains held out.
- GitHub run 41388102462 passed offline checks, API/UI Docker Compose builds, HTTP health checks, and in-container protocol validation for commit 2933388. Domain/API type checking has subsequently been added to the final verification commands.

## Portfolio release checkpoint

- The user's instruction to unblock the highest-impact work resolved the human-review release dependency by narrowing the atlas claim. Continuous retrieval remains experimental, with zero ratings and no discrete driving-state claims. Scientific human validation remains open; it does not block an explicitly unvalidated research demo. See `docs/release-decision.md`.
- Added the offline five-page portfolio, committed aggregate evidence, reproducible development orchestration and measured warm CPU inference. The fresh checkout requires neither a provider connection nor private assets. Local assets add detailed forecasts and experimental neighbors.
- First case-study freeze at `f5775cc` pinned the protocol, existing models, index, dataset and eleven source files. Mexico acquisition followed the freeze: 70 lap records, 30,924 telemetry rows, nine exact source snapshots. The original frozen attempt emitted zero eligible forecasts and is preserved in `reports/mexico-first-attempt.json` and local immutable artifacts.
- A pre-race unknown-duration pit interval exposed inconsistent historical-reference handling. The repair uses only completed valid non-pit-out lap evidence available by each cutoff. Future-mutation and cutoff-boundary tests passed. Independent offline rematerialization proved all 6,204 development examples, their order and quality results unchanged across 211 exact source manifests. Existing weights and 160 experiment results remain reusable; no fitting or hyperparameter retuning occurred.
- The second freeze at `2cb6bfb` pins the correction rationale and exact first-attempt freeze/report, enforcing identical model files, reference index, protocol and canonical development examples. Its declared claim is post-inspection implementation-correction diagnostic replay, not an independent blind test.
- Corrected scoring reused all nine source manifests with the client explicitly offline. All 213 sector slots reconcile: 115 scored, 95 abstained, three missing, zero censored. MAE 0.72364 seconds, WIS 0.50507, 90% coverage 0.93043. All 75 sectors after lap 46 abstain under the unchanged long-pit interpretation; these selective metrics do not cover the entire race or establish that the driver remained in the pit lane.
- Public reports contain derived accounting and aggregate metrics only. Source rows, individual predictions, weights and private review files remain outside Git. Independent audits verified source identity, model/index invariance and metric arithmetic.
- GitHub run 41391900921 passed 146 CPU ML/UI tests, core checks and API/UI container verification for `f5775cc`. Later correction tests and final portfolio checks are recorded by the release CI run.

## Pit timestamp revision, November 2, 2026

- User explicitly requested resolving pit-timestamp interpretation, recovering valid late Mexico coverage, preserving published runs, parallel work and memo-only communication.
- Primary upstream code, official pit-stop timing and exact cached telemetry support completion/report time rather than entry time. The evidence memo pins provider revisions and source checksums. Physical timing-loop boundaries and original API publication latency remain unverified.
- Added explicit protocol v0.2.0 with `completion_report_v2`. Reports must satisfy original `date <= cutoff` before a known duration derives backward occupancy. Future reports cannot backfill earlier inputs. Unknown durations retain conservative running-lap evidence. Legacy absent-policy semantics remain unchanged.
- Offline development rebuild retained 6,204 identities and all labels, while changing 1,638 actual inputs and adding 246 eligible origins. Scoreable examples increased from 5,008 to 5,249. A separate legacy rebuild exactly reproduced every old example and its order; published reports and artifact bytes were audited unchanged.
- Re-ran all four baselines on eight development holdouts with fixed settings, then refitted/recalibrated the chosen CatBoost and separate encoder/index. The same first16/last2 race partition, seed17 and encoder3/5epoch budget remain. New index:4,568 fitting references. Historical160 neural experiments remain v0.1 evidence; no new neural advantage is claimed.
- New freeze at `e18292c` links the preserved115-forecast parent, exact old protocol, primary evidence and new development benchmark. The feature-revision path explicitly permits rebuilt data/weights while enforcing the same recipe/scopes. It does not weaken the earlier unchanged-model correction path.
- New Mexico reconstruction/scoring used the same nine source snapshots offline after the freeze. Total213slots:177scored,32abstained,1censored,3missing. Late75slots:62scored,12abstained,1censored. Common115,new62,lost0. Late exclusions remain ten non-green origins, two invalid anchors, and one invalid-timing censored target.
- New overallMAE0.875861/WIS0.610124 summarize a larger population. On the common115, MAE0.720436 versus0.723642 andWIS0.509305 versus0.505067 are descriptive; added coverage is not isolated forecast-skill improvement. New models, data and index are explicitly versioned in comparison provenance.
- Added version-selectable case/Atlas/operations UI and safe aggregate comparison export. Original reports, models, index, source snapshots and v0.1.0 release remain intact. See `docs/pit-timestamp-correction.md` for reproducibility and remaining limits.
