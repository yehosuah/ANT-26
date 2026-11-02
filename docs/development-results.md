# ANT-26 development results

ANT-26 completed **160 fixed-budget representation fits** across eight chronological race holdouts. No learned-model improvement gate passed. Retain a baseline forecaster and evaluate the encoder's atlas use separately. These results establish neither a general model ranking nor an explanation for Antonelli's championship.

The public [aggregate results](../reports/development-results.json) contain exact metrics, paired intervals, input/protocol/implementation hashes and source-report checksums. Raw predictions, features and source snapshots are excluded.

## Data and experiment scope

The Antonelli cohort contains 3,228 observed sectors and 2,619 scoreable examples across 18 pre-Mexico races. The Mercedes cohort adds Russell, yielding 6,204 sectors and 5,008 scoreable examples. Both evaluate the same 1,039 held-out Antonelli sectors per model and seed. Separately validated acquisition counts are 395,446 Antonelli and 365,248 Russell telemetry rows; this is a two-driver cohort, not the full grid.

Each result averages errors within a race, then gives races equal weight. Representation summaries first average the five fixed seeds. Paired intervals resample complete races, not telemetry rows or seeds. Hungary retains only 28 of 206 nominal origins, so its evaluated subset is selective.

## Primary Antonelli results

MAE and weighted interval score (WIS) measure error. Coverage describes central 90% intervals.

| Training cohort and model | MAE seconds | WIS | Coverage |
|---|---:|---:|---:|
| Antonelli, persistence | 0.946 | 0.872 | 91.3% |
| Antonelli, CatBoost | 1.010 | 0.643 | 92.0% |
| Antonelli, scratch encoder | 1.021 | 0.710 | 91.1% |
| Antonelli, frozen encoder | 1.047 | 0.688 | 91.4% |
| Antonelli, fine-tuned encoder | 0.978 | 0.684 | 91.2% |
| Mercedes, persistence | 0.946 | 0.864 | 91.8% |
| Mercedes, CatBoost | 0.981 | 0.629 | 93.0% |
| Mercedes, fine-tuned encoder | 0.997 | 0.777 | 91.9% |

Gains compare MAE against persistence and WIS against CatBoost within each cohort. Positive values favor the candidate. Parentheses give paired 95% intervals, expressed as percent.

| Candidate | MAE gain | WIS gain |
|---|---:|---:|
| Antonelli fine-tuned | −3.46% (−47.23, 23.32) | −6.33% (−17.45, 1.59) |
| Antonelli frozen | −10.69% (−57.26, 19.01) | −6.97% (−20.44, 2.19) |
| Antonelli scratch | −7.95% (−54.39, 21.32) | −10.40% (−24.49, −1.37) |
| Mercedes fine-tuned | −5.48% (−62.84, 25.39) | −23.57% (−87.41, 6.66) |

The fine-tuned comparisons include zero. The declared requirement of at least 5% improvement on both metrics with supporting intervals was not met.

## Observed failure and remaining evidence

Secondary pooled-driver results expose a severe tail: fine-tuning has 16.223-second MAE and 7.216 WIS; Russell alone has 31.332-second MAE. Baku sector-one diagnostics show a fitted near-constant throttle-minimum scale producing standardized values above 1,800. Output clipping prevents numerical overflow while still permitting unphysical forecasts. This observed distribution-shift failure remains visible; no test-fold retuning was performed.

Earlier exact-constant normalization runs were invalidated as an implementation defect. Their errors are excluded and provide no model-capacity evidence. The current failure occurred under the corrected policy.

The budget used one 559,104-parameter encoder configuration, three pretraining epochs for pretrained modes and five forecasting epochs. Antonelli's 120 fits reported 476.25 seconds; the pooled 40 reported 398.42 seconds, using one CPU thread per run. Run durations may overlap. Human blind atlas ratings remain pending. Mexico remains held out for the final case study.
