# ANT-26 data card

ANT-26 uses observed OpenF1 telemetry and context for Kimi Antonelli (12) and George Russell (63). The development population is **18 Antonelli and 17 Russell timed race traces**, with **760,694 native telemetry rows, 6,204 sector examples and 5,008 scoreable examples**. It is a two-driver cohort, not full-grid coverage. [Acquisition counts](../reports/development-results.json), [quality review](real-data-quality.md).

## Sources and coverage

Historical records come from the independent [OpenF1 API](https://openf1.org/docs/): laps, car telemetry, stints, intervals, pit records, race control, weather, session identity and session results. Exact queries and raw-response snapshots are retained locally. The verified development catalog separates canceled sessions and Mexico. A sourced override distinguishes the Bahrain event from its Malaysian host circuit. [Catalog overrides](../configs/catalog-overrides.json).

Antonelli contributes 395,446 native telemetry rows, 3,228 sector examples and 2,619 scoreable examples. Russell contributes 365,248 telemetry rows and 2,389 scoreable examples; his Belgian race has no valid timed source laps and is excluded explicitly. These counts describe available source records, not a guarantee of official terminal-lap completeness for every development trace.

Missingness is selective. Hungary retains only 28 of 206 Antonelli nominal green, outside-pit opportunities (13.59%), principally because invalid brake-channel values fail the fixed coverage rule. That held-out race remains in evaluation. High average retention must not hide this failure. Synthetic fixtures are separately labeled and support software checks only.

## Measurements and reconstruction

The required sequence channels are speed (km/h), throttle (provider 0–100), brake application (0/1), RPM and gear. Brake pressure, steering, deployment and accurate lateral position are not observed or reconstructed as measurements. This release does not recover exact racing lines.

Sector-entry anchors approximate lap starts plus durations of already completed sectors. Inputs stop one second before entry. Each window uses a 32-second, 4 Hz grid, which adds no measurement resolution. Original live-publication times and consumer latency were not captured: availability is reconstructed from event timestamps.

`sequence_mask` identifies usable values under the short-gap rule; `sequence_observed_mask` identifies exact native observations on the grid. Sample age, native counts and usable per-channel coverage remain separate. Interpolation uses only records before the cutoff, gaps over one second are masked, and red-flag barriers prevent filling across interruptions. Every required channel needs at least 80% usable coverage. Zero padding with a false mask is not an observation.

## Eligibility and target validity

An eligible origin requires known green conditions, an outside-pit state under the fixed source interpretation, sufficient telemetry and two prior valid same-driver, same-sector observations. The historical reference is the median of up to three recent observations. Unknown context, insufficient history and bad telemetry produce explicit abstentions.

Eligibility uses evidence available by the cutoff; it cannot depend on future timing or eventual full-lap validity. Target validity is separate. An unexpected Safety Car or VSC during a valid target remains scored; a red-flag-spanning or invalid-duration target is unscoreable with reasons. Retention, censoring and missing records accompany accuracy. [Rules](../configs/evaluation-protocol.json), [model evidence](model-card.md).

## Mexico case-study status

Mexico meeting 1298, including race session 11404, is excluded from all fitting, normalization, calibration, clustering and retrieval references. Outcomes and an exploratory source window had already been seen before the first freeze, so it was never an untouched blind test.

After the original release freeze, nine exact source snapshots yielded 30,924 telemetry rows and 70 observed lap records against 71 official laps. Lap 5 is missing: 210 observed sectors plus three missing sectors equal 213 expected records. The first frozen implementation produced zero eligible forecasts and is preserved. [First-attempt aggregate](../reports/mexico-first-attempt.json).

Inspection exposed inconsistent treatment of an unknown-duration pre-race pit record. The correction bounds an unknown exit using completed non-pit-out lap evidence available by the cutoff, consistently for current state and historical-reference validity. Prefix-safety regressions passed. Rebuilding the development dataset from 211 pinned source manifests changed **none of the 6,204 examples or their canonical ordering**; only provenance changed. No model or index refit followed. [Equality audit](../reports/development-correction-impact.json).

A new freeze explicitly labels the corrected result as a **post-inspection implementation-correction diagnostic replay**. All source bytes are checked and examples rebuilt offline before scoring. It contains 115 scoreable forecasts, 95 abstentions, three missing sectors and zero censored eligible targets. MAE 0.72364164 s, WIS 0.50506749 and 90% interval coverage 0.93043478 apply only to the scored subset. [Mexico aggregate](../reports/mexico-case-study.json).

All 75 sector origins after lap 46 abstain under the fixed interpretation of a long recorded pit interval around the interruption. The output does not establish actual continuous pit-lane occupancy or forecast coverage across the whole race. Pit-context interpretation and selective retention remain material limitations.

## Provenance and access

Raw responses retain retrieval timestamps, query identities and SHA-256 manifests, including original CSV bytes when normalized. Release manifests pin protocol, code, models, index and source-to-feature reconstruction. Refreshes create new snapshots; immutable artifacts refuse replacement. Locally stored raw data and trained artifacts are ignored by Git. Public aggregate JSON and the offline portfolio omit the raw dataset. [Reproduction](reproducibility.md), [release decision](release-decision.md).
