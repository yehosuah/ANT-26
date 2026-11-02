# ANT-26 operational evidence

The selected CatBoost forecasting bundle and the separate continuous-retrieval encoder pass the declared **local CPU p95 < 200 ms** target. This measures warm inference from materialized past-only examples. HTTP latency, source ingestion, event alignment, model loading, retrieval-index search and concurrent traffic remain outside this measurement.

## Measured latency

Measured November 1, 2026 at 22:52:46 America/Guatemala (UTC−6) on an Apple M5 Pro, arm64, 18 logical CPUs, Darwin 27.0.0. Runtime was Python 3.11.15, NumPy 2.4.6, CatBoost 1.2.10 and PyTorch 2.14.1. PyTorch used one intra-op thread. The isolated five-tree phase explicitly used one CatBoost thread; the full forecasting call retained the saved bundle's existing prediction defaults.

Each phase had 20 warmup calls and 200 timed synchronous batch-one calls. Inputs cycle over 100 deterministically selected calibration examples, 50 Antonelli and 50 Russell, spanning Singapore and Austin. Percentiles use empirical linear interpolation. Calibration examples measure runtime here; these measurements provide no new held-out accuracy evidence.

| Phase | p50 ms | p95 ms |
|---|---:|---:|
| Baseline numerical feature conversion | 0.0033 | 0.0035 |
| Five native CatBoost trees, prepared inputs | 0.4592 | 0.4924 |
| Full baseline forecast, including conversion and calibration | 0.5188 | 0.7098 |
| Encoder input conversion and normalization | 0.0386 | 0.0415 |
| Encoder forward pass, prepared inputs | 0.2582 | 0.2830 |
| Full encoder embedding call | 0.3335 | 0.3604 |

Phase timings are independent, so their percentiles cannot be added. The current full baseline converts features for each of its five trees. The encoder measurement excludes its failed quantile forecasting head. This is a local warm-call benchmark; a service latency objective needs an HTTP load test on deployment hardware.

The aggregate report is generated at `artifacts/operations/cpu-batch1-v2/report.json`. It pins the dataset envelope and example hashes, candidate checksums, protocol, benchmark source, hardware and runtime. Raw feature rows, predictions, embeddings, personal paths and hostnames are absent.

## Reproduce with the prepared research assets

```bash
uv sync --locked --group dev --extra ml --no-editable
PYTHONPATH=src uv run --no-sync python -m ant26.evaluation.operations \
  --dataset data/processed/mercedes/merged-e7ed15ec6216606e.json \
  --baseline artifacts/final-candidate/baseline \
  --encoder artifacts/final-candidate/encoder \
  --output artifacts/operations/my-cpu-run/report.json \
  --warmup 20 --repetitions 200
uv run --no-sync pytest tests/test_operations.py
```

Use a new output path for every run. The helper refuses replacement, verifies the exact dataset bytes and model fitting rows, requires disjoint fitting/calibration scopes, and rejects Mexico or later inputs. It never fits or changes a model. Hardware contention can change the measured percentiles.

## Bounded monitoring and OOD policy

The existing feature builder abstains on insufficient channel coverage, invalid origins and inadequate history. Preserve those exclusions and report their denominators. Exact-constant normalization artifacts are refused by the checkpoint loader. Keep CatBoost as the forecaster; the neural head failed the development gates, including a severe Russell/Baku tail under the corrected normalization policy.

`input_drift_report` compares observed eligible origin inputs with explicit fitting references. In this run, 680 calibration origins were compared with the exact 4,328 model fitting rows: zero origins exceeded fitted feature ranges, but two exceeded an absolute encoder context z-score of 10 (`throttle_max`, maximum 13.12). This illustrates why a range check alone does not establish safe extrapolation. Missingness changes and previously unseen driver counts are also reported. Observed target values and target-validity flags do not enter drift monitoring.

The z-score threshold is an engineering warning, not a calibrated OOD detector or a newly enforced exclusion. Show warnings in research diagnostics and review affected retrievals. Do not claim that clipping guarantees physical forecasts, or retrain automatically from flagged test examples. No deployment policy or frozen benchmark threshold changes in this module.

After labels arrive, inspect equal-race MAE, WIS, interval coverage, eligible-origin retention and error slices by driver, sector and conditions. Compare with the declared 87–93% interval-coverage range and 70% retention target. Breaches require a visible investigation and a separately declared experiment before model changes. The observed Hungary retention failure and pooled neural tail remain visible in the development results. These are retrospective checks, not an implemented continuous alerting service or statistical drift test.

## Clean-checkout limits

Code, locked dependencies, synthetic fixtures and safe aggregate results are public. Exact real source snapshots, processed examples, weights, retrieval indexes and blinded review packs are ignored. A clean clone can run the synthetic smoke pipeline and tests without provider access; the real-asset latency command above requires acquisition/materialization and candidate preparation, or an independently supplied checksummed bundle. Re-fetching a mutable provider does not guarantee the original source bytes or identical historical hashes.

The repository includes `ant26 prepare-release` for candidate preparation. A portable bundle publication and acquisition contract for the exact historical assets remains separate work. Human retrieval ratings remain pending. The monitoring helper performs no Mexico access; Mexico scoring still requires a recorded release freeze.
