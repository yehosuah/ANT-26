# ANT-26 model card

ANT-26 combines probabilistic next-sector forecasting with descriptive telemetry retrieval. The current implementation has software verification and a real-source feasibility audit. It has no established real-data predictive advantage, validated driving-style taxonomy, or Mexico performance result.

## Intended use

The project is an inspectable research portfolio for ML engineers. Given past telemetry and race context at reconstructed sector entry, a forecast estimates upcoming sector duration. An atlas retrieves similar earlier windows and displays observable feature differences. Neither output establishes driver intention, confidence, racecraft, tyre health, or a causal explanation of championship success.

The HTTP application currently replays saved benchmark artifacts. It does not train models or promise real-time F1 inference. Retrospective atlas analysis must be labeled separately from prefix-only forecast replay.

## Implemented models

The baseline family contains same-sector persistence, a recent historical reference, a regularized linear residual model and optional CatBoost quantile models. Fit, calibration and test sessions are explicit. Targets are excluded from the numeric feature allowlist.

The learned candidate uses an adapted masked patch transformer and a non-crossing quantile head. Default inputs are 128 time steps by five telemetry channels, with usable masks. Masked reconstruction uses robust regression for continuous channels and classification losses for brake application and gear. Missing or padded values do not become reconstruction targets. The implementation is motivated by masked time-series learning and is not an exact reproduction of a published model.

Normalization policy `constant_unit_scale_v2` uses unit scale for fitted standard deviations below 1e-6, for both context and sequence channels. A regression reproduces and prevents extreme held-out scaling from previously constant features. Earlier neural results using the superseded policy are invalidated and hidden from replay; their source snapshots remain available locally for audit. The loader requires the current policy.

Scratch, pretrained-frozen and fine-tuned modes enable attribution of any future gain to representation learning. The head predicts log duration relative to a historical reference and emits quantiles at 0.05, 0.10, 0.50, 0.90 and 0.95. Calibration widens intervals using designated disjoint preceding races; serial dependence and circuit shift prevent a general coverage guarantee.

The atlas accepts externally supplied embeddings and also supplies a train-only standardized-feature/PCA baseline. Its immutable reference index excludes Mexico and sessions outside the fitting allowlist. Prefix queries exclude future, same-lap and overlapping same-driver windows. Optional Gaussian-mixture patterns remain descriptive and `validated=false` pending stability checks and review.

## Evidence available

- Offline tests exercise future-record invariance, timing/censoring separation, fitting-scope guards, quantile ordering, artifact integrity, retrieval filtering and API responses.
- A saved synthetic smoke benchmark runs two chronological test folds and identifies itself as exploratory. Its numerical errors cannot substantiate F1 performance claims.
- Reduced synthetic representation tests verify finite optimization and checkpoint round-tripping. Reconstruction improvement on fitting fixtures is optimization evidence only.
- The original real-source audit confirms sampled telemetry and contextual endpoints for three development races and two drivers. A subsequent Antonelli-only cohort materializes all 18 pre-Mexico races; [the real data quality review](real-data-quality.md) records coverage and selective missingness. Data availability alone does not establish a learned-model advantage.

The exact current check commands are `uv run ruff check .` and `uv run pytest`. Re-run them against the checked-out revision; test counts can change as implementation proceeds. Experiment details and current measured fixture results live in `docs/representation-experiment.md`.

## Planned evaluation and release gates

Versioned rules live in `configs/evaluation-protocol.json`. Rolling folds hold out whole weekends across all drivers, with earlier fitting, tuning and calibration partitions. Antonelli's primary metrics average within each race and then give races equal weight. Five fixed seeds and paired race-block bootstrap intervals quantify variability without treating adjacent telemetry samples as independent races.

The proposed improvement gates require at least 5% lower race-macro MAE and weighted interval score against their respective strongest development-selected baselines, with uncertainty intervals excluding zero. Interval coverage, widths, abstention, data retention, timing-guard sensitivity and error slices are reported alongside those metrics. These targets have not been achieved or tested on the full real dataset.

Mexico is the final case study after model, preprocessing, calibration, retrieval references and protocol are frozen. Its complete timeline must show missing records, neutralizations, abstentions and failures, rather than only selected successful predictions. A one-race case study cannot replace multi-race evidence of generalization.

## Limitations and deployment status

Circuit and car conditions can dominate embeddings; teammate comparisons do not remove setup, fuel, strategy or mechanical confounding. Sector anchors are approximate, sampled telemetry cannot recover exact racing lines, and unpublished source channels remain unobserved. Cluster stability and human retrieval review are pending.

Model bundles and atlas indices carry schemas, fitting scopes and checksums. Existing artifacts refuse replacement. The local API and UI scaffold are available; container execution and public deployment are separate validation steps. No production or public-hosting claim is made.
