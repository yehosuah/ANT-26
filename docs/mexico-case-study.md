# Mexico: a case study that exposed an implementation failure

The corrected diagnostic replay produces **115 scored forecasts, 95 abstentions and 3 missing sector records**, reconciling all 213 opportunities across 71 laps. MAE is 0.724 seconds, weighted interval score is 0.505, and the central 90% intervals cover 93.04% of scored outcomes. These are selective one-race descriptive results. They are not evidence of neural-model superiority or performance across the whole race.

The [public report](../reports/mexico-case-study.json) contains derived counts, aggregate errors and provenance. Individual predictions, sampled telemetry, model files and reference vectors remain local. The portfolio displays all 71 lap slots from the public aggregates without requiring those private artifacts.

## Preserve the first attempt

The first release froze CatBoost, the separate experimental encoder, reference index, protocol, development data and eleven source files at commit `f5775cc`. Only then did the final workflow fetch Mexico. The source returned 70 lap records and 30,924 telemetry samples. The classified race had 71 laps; lap 5 was absent from the lap endpoint.

The frozen pipeline produced **zero eligible forecasts**. An unknown-duration pit record before the race was treated as an indefinitely open interval when selecting historical reference sectors. Current pit-state logic could infer an exit, but historical reference selection did not share that evidence. Every candidate origin consequently lacked sufficient history. The [first-attempt aggregate](../reports/mexico-first-attempt.json) and its immutable local outputs are preserved.

The correction closes an unknown pit interval only at completed, valid, non-pit-out lap evidence available by the feature cutoff. It does not backdate the inferred exit to the beginning of that lap. Known-duration intervals, target labels, anchors, model hyperparameters and learned weights remain unchanged. Regression tests cover the original failure, the exact cutoff boundary and future-record mutation.

An independent offline rebuild reused 211 exact source manifests across 35 development traces. All **6,204 example values, ordering and quality results were identical** to the original development dataset. The [audit](../reports/development-correction-impact.json) therefore supports retaining the weights and 160 experiment results.

A second release at commit `2cb6bfb` pins the correction rationale, original freeze and zero-forecast report. It requires unchanged model bytes, protocol, index, fitting/calibration scopes and canonical development examples. The corrected run reused the same nine source snapshots with the client explicitly offline. Its declared claim is `post_inspection_implementation_correction_diagnostic_replay`. Earlier exploratory work had already exposed the race outcome; neither attempt is presented as an untouched blind benchmark.

## Account for the entire race

| Phase | Laps | Scored sectors | Abstentions | Missing sectors |
|---|---:|---:|---:|---:|
| Opening stint | 1–42 | 110 | 13 | 3 |
| Pit stop and interruption | 43–46 | 5 | 7 | 0 |
| Standing restart and late Safety Car | 47–62 | 0 | 48 | 0 |
| Final run | 63–71 | 0 | 27 | 0 |
| Total | 1–71 | 115 | 95 | 3 |

Input rules retain 115 of 121 observed known-green/outside-pit origins, or 95.04%. That denominator is selective: only **53.99% of the 213 official sector opportunities are scored**. Missing timing, unknown context, cold starts, neutralizations and pit exclusions remain visible. Exclusion-reason counts can overlap.

The original timestamp interpretation of a long pit record excludes the entire period after lap 46. Those exclusions describe the pipeline's inferred context, not proof that Antonelli remained in the pit lane. The source timestamp/entry-exit interpretation needs independent validation before any later model experiment changes it. This case does not silently repair that ambiguity to increase coverage.

Official reporting establishes the early VSC, pit stop, red flag and later Safety Car. It also records Antonelli's podium and title. Those events anchor the chapters; they do not establish behavioral causes. [Formula 1 race report](https://www.formula1.com/en/latest/article/antonelli-crowned-youngest-ever-f1-champion-as-verstappen-wins-eventful-mexico-city-grand-prix.xpMMeef0KmRURs9IDs3LH).

## Inspect success, failure and abstention

The local forecast table contains a near-exact prediction at lap 26, sector 2, and a large 13.482-second error at lap 7, sector 2 during the early interruption. These are descriptive error extremes selected after scoring. The latter remains in primary scoring because an eligible origin does not retroactively abstain when a surprise neutralization occurs. A local replay can inspect both outcomes; the aggregate report publishes the error range without individual forecast records.

Lap 5 illustrates a source gap. The late-race timeline illustrates abstention under ambiguous pit context. Both matter as much as a low-error sector: an accurate score on the retained subset cannot establish coverage of excluded phases.

The atlas returns five historical neighbors for each of the 115 eligible windows using the unchanged pre-Mexico index. Queries remain retrospective inspection. Their distances are not probabilities of shared intention, and the comparisons have zero human ratings. No discrete driving-state labels or driver-independent style claims are supported.

## Reproduce the evidence

See the [reproduction guide](reproducibility.md) for source acquisition and candidate preparation. Once the corresponding immutable local assets exist:

```bash
ant26 case-score --freeze artifacts/releases/ant26-v2-freeze.json \
  --examples data/case-study/mexico-v2/examples-5840cc847a4a3f06.json \
  --output artifacts/my-new-mexico-replay
ant26 case-export --run artifacts/my-new-mexico-replay \
  --freeze artifacts/releases/ant26-v2-freeze.json \
  --chapters configs/mexico-chapters.json --output reports/my-new-case-report.json
```

The exact historical asset paths are not bundled. Fresh provider responses require new data identities and releases. The older frozen runtime is recoverable from its recorded Git revision; a changed implementation cannot verify against the original freeze.
