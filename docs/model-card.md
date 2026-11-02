# ANT-26 model card

ANT-26 is a research portfolio for probabilistic next-sector forecasting and experimental continuous telemetry retrieval. Completed development experiments did not establish a learned-model advantage. The release retains CatBoost, reports the failed neural and atlas checks, and presents Mexico as a **post-inspection diagnostic replay**. [Development results](../reports/development-results.json), [Mexico results](../reports/mexico-case-study.json).

## Intended use and inputs

Forecasts estimate upcoming sector duration from telemetry and context available before reconstructed sector entry. Inputs use a 32-second, 4 Hz masked sequence and allowlisted numeric summaries; the cutoff is one second before entry. Publication latency was not recorded, so this is event-time replay rather than verified live inference. The population is Antonelli and Russell, not the full grid. [Data card](data-card.md).

Retrieval describes similarities between observed windows. It does not establish intention, confidence, racecraft, tyre health, exact racing lines or causal explanations of championship success.

## Models and fitting boundaries

The baselines are same-sector persistence, historical reference, regularized linear residual and CatBoost quantile models. Forecasts provide quantiles at 0.05, 0.10, 0.50, 0.90 and 0.95; designated calibration races widen intervals. Dependence and circuit shift prevent a general coverage guarantee.

The adapted masked patch transformer has 559,104 parameters, width 128, four layers and four heads. It reconstructs observed continuous and discrete channels, then predicts log duration relative to historical reference through a non-crossing quantile head. Scratch, frozen and fine-tuned modes were evaluated. Normalization policy `constant_unit_scale_v2` assigns unit scale to fitted standard deviations below 1e-6. Earlier defective-normalization runs are invalidated and preserved separately; legacy checkpoints are refused.

Final seed-17 candidates fit the first 16 development races and calibrate on Singapore and Austin. The encoder uses three pretraining and five forecasting epochs; its 4,328 fitting windows form the continuous retrieval index. Mexico's entire weekend is excluded from fitting, normalization, calibration and references. Checksums, scopes and source hashes are pinned by separate release manifests. [Final candidates](final-candidate.md), [release decision](release-decision.md).

## Completed development evaluation

The real cohort contains 18 Antonelli and 17 Russell timed race traces, 760,694 native telemetry rows, 6,204 sector examples and 5,008 scoreable examples. Eight chronological whole-weekend test folds, five fixed seeds and fixed budgets produced **160 corrected representation fits**: 120 Antonelli-only fits across three modes and 40 pooled fine-tuned fits. Primary metrics average within Antonelli races and then weight races equally; uncertainty resamples complete paired races. No learned candidate passed the declared joint 5% MAE/WIS improvement gate.

| Pooled development candidate, primary Antonelli metric | Race-macro MAE (s) | WIS |
| --- | ---: | ---: |
| Same-sector persistence | 0.94568 | 0.86407 |
| CatBoost quantiles, selected forecaster | 0.98062 | 0.62862 |
| Fine-tuned transformer, five-seed mean | 0.99748 | 0.77681 |

CatBoost was selected for its lower observed WIS, accepting 3.7% higher point MAE than persistence. This does not establish blanket statistical superiority. Corrected neural models still exhibit severe near-constant-feature/out-of-distribution tails: pooled Russell secondary MAE is 31.33 seconds, with documented extreme standardized throttle/brake inputs. Finite clipping is not physical plausibility. [Exact metrics, paired intervals and failure diagnostics](../reports/development-results.json).

Discrete atlas states failed the first bounded stability check: median ARI 0.32123 against a 0.6 target, across 20 race bootstraps and 28 selectively observed Hungary probes. Discrete-state claims were dropped. Continuous retrieval has **zero human ratings**, is unvalidated, and has no established advantage over feature/PCA retrieval. [Atlas evidence](atlas-review.md).

## Mexico diagnostic replay

The original frozen implementation produced zero eligible forecasts and remains preserved. Inspection identified inconsistent handling of an unknown-duration pre-race pit interval. A prefix-safe implementation correction was tested, and all 6,204 development examples and their ordering remained canonically identical; weights, hyperparameters, protocol and index were unchanged. A separate freeze records the correction and first attempt. Its claim is `post_inspection_implementation_correction_diagnostic_replay`, not an untouched blind test. [First attempt](../reports/mexico-first-attempt.json), [development equality audit](../reports/development-correction-impact.json).

Of 213 expected sector records, 115 are scored, 95 abstain and three are missing. On the scored subset, MAE is **0.72364164 s**, WIS **0.50506749**, and empirical 90% interval coverage **0.93043478**. The largest error is 13.4823 seconds at lap 7 sector 2, including an unexpected VSC. These selective, single-race results do not establish whole-race accuracy or generalization.

All 75 sector origins after lap 46 abstain under the fixed interpretation of a long recorded pit interval around the interruption. This is a source/interpretation limitation, not evidence that the driver stayed in the pit lane. [Full accounting and limitations](../reports/mexico-case-study.json).

## Verification and deployment status

Offline checks exercise timing invariance, scope isolation, censoring, quantile ordering, immutable artifacts and source-to-feature reconstruction. The public aggregate portfolio works offline; the artifact API and replay UI have automated tests, and container builds/health checks passed GitHub CI. Local warm CPU timings are measured separately from service latency. This is not a public hosted service or production deployment. [Engineering evidence](engineering.md), [operations scope](operations.md), [reproduction](reproducibility.md).
