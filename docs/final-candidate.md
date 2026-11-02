# ANT-26 final candidate preparation

The probabilistic forecasting candidate is CatBoost, selected by development weighted interval score on the Mercedes cohort. The continuous atlas has a separate fixed-budget fine-tuned encoder candidate. Their immutable metadata retains `candidate_not_frozen_for_mexico`; separate release manifests subsequently freeze those exact bytes. The release narrowed retrieval to explicitly unvalidated continuous similarity, leaving real human ratings open. See the [release decision](release-decision.md) and [Mexico case study](mexico-case-study.md) for the preserved first attempt and diagnostic correction.

## Forecasting choice and tradeoff

Across the eight chronological development test races, evaluating Antonelli with Mercedes-cohort training, the selected CatBoost baseline achieved WIS 0.6286189 and MAE 0.9806240 seconds. Same-sector persistence had the lowest MAE, 0.9456798 seconds. The probabilistic choice therefore accepts a 3.6951% point-error increase in exchange for the best observed interval score. This is an explicit objective tradeoff, not evidence that the learned encoder beat the baselines.

The observed cohort contains Antonelli and Russell only. Source availability covers 35 driver-race pairs; Russell has no valid timed laps in the Belgian source. The artifacts do not establish full-field generalization or isolate driver behavior from car, setup, and strategy.

## Fitting and calibration

The verified pre-Mexico catalog contains 18 race weekends. Fixed-hyperparameter refitting uses the first 16 races, including the last two tuning races. Calibration uses Singapore session 11388 and Austin session 11396. The native forecasting models retain seed 17 and the existing 100-iteration, depth-four quantile configuration; no hyperparameters were changed after inspecting the pilot.

The encoder candidate uses the same first-16/last-two partition, seed 17, three reconstruction epochs, and five forecast epochs. It has 559,104 encoder parameters and uses normalization policy `constant_unit_scale_v2`. Its purpose is continuous historical retrieval; predictive superiority is not claimed. The local candidate training took approximately 10.17 seconds on CPU, and both candidate serialization round-trips preserved predictions.

## Artifact contract

The forecasting bundle is stored at `artifacts/final-candidate/baseline/`. Its state and metadata are validated JSON; its five quantile trees use CatBoost's native `.cbm` format. No Python pickle is written or loaded. The loader validates the metadata digest, every state/model-file checksum, numeric-feature and quantile schemas, fitting/calibration separation, and the baseline runtime code hash. Existing artifacts cannot be overwritten.

Metadata records source-envelope and example hashes, protocol and catalog hashes, code provenance, exact fitting/calibration sessions, latest fitting/calibration timestamps, driver counts, and development selection evidence. Corrupt state, metadata, or native trees fail before use.

The encoder bundle is stored at `artifacts/final-candidate/encoder/`, with its native checkpoint and checksummed metadata. Legacy normalization checkpoints require retraining and cannot silently become the final predictor.

```python
from ant26.models.artifacts import BaselineBundle
from ant26.models.representation import RepresentationBundle

forecast_candidate = BaselineBundle.load("artifacts/final-candidate/baseline")
atlas_candidate = RepresentationBundle.load("artifacts/final-candidate/encoder")
```

The pure preparation helpers are `final_candidate_partitions(catalog, protocol)` and `fit_final_candidate(examples, catalog, protocol, development_report=...)`. They enforce the verified 18-weekend catalog, reject Mexico or future fitting records, require valid labels in every declared session, and keep the final two calibration races separate. Their outputs remain candidates until the later release workflow records the freeze decision. No artifact metadata or earlier experiment report is rewritten to imply that decision already happened.
