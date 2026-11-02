# ANT-26 real data quality review

The first complete Antonelli cohort covers 18 verified pre-Mexico 2026 races. It contains 395,446 native telemetry rows and 3,228 sector examples derived from observed lap records. Mexico is absent. This establishes a usable data pipeline; it does not establish forecast quality or represent all drivers.

The immutable merged artifact is `data/processed/materialized/merged-c97b552884729f08.json`, with SHA-256 `c97b552884729f08e711d26491500d9e0cefcd34f91216a6d467f9d10934b50d`. Its explicit cohort is driver 12. The independent aggregate evidence is saved locally as `data/audit/independent-quality-report.json`.

A subsequent metadata revision adds the complete feature-builder source checksum to cache identity. The corresponding Antonelli artifact is `data/processed/materialized/merged-d4b6a227c1bbe4ed.json`. Its canonical example bytes are identical to the reviewed artifact above; their SHA-256 is `54c81787009e9d8ddf7d543e47eaa6299bbdc396852734d073ad30b50fc71aec`. Prior experiments using the original examples remain unchanged.

## Teammate control cohort

Russell's companion cohort contains 17 materialized race sessions, 2,976 observed sector rows and 2,389 scoreable examples. His Belgian session has no valid timed laps in the source and is explicitly marked `no_valid_timed_laps`, with its lap snapshot retained as exclusion evidence. This is not silently treated as an acquisition still in progress.

The pooled artifact is `data/processed/mercedes/merged-e7ed15ec6216606e.json`, with SHA-256 `e7ed15ec6216606e462b43cbfa8388b6447b6255bf40a2fde4e10819b4180450`. It explicitly declares drivers `[12, 63]`, 35 driver-race pairs, 6,204 observed sector rows and 5,008 scoreable examples. This is a Mercedes cohort, not a complete-field dataset. It records the separate input dataset checksums and source manifests. A teammate comparison still cannot remove setup, fuel, strategy or mechanical confounding.

## Coverage and denominators

Of 2,863 observed sector origins classified as known green and outside the pit lane, 2,625 meet forecast-input eligibility, a retention rate of 91.69%. There are 2,619 eligible examples with valid scoreable targets. These are separate counts: input eligibility must not depend on future target validity.

The denominator consists of observed source sector rows under the pipeline's event-time assumptions. It excludes non-green, known-pit and unknown-state origins. It is not the number of all official race opportunities. The 18 source lap lists have no gaps between their minimum and maximum observed lap numbers; comparison of their terminal lap counts with official completed-lap records is still pending.

Origin exclusion counts can overlap. They include 225 insufficient-telemetry cases, 119 insufficient-history cases, 141 non-green cases, 136 pit-lane cases, 72 unknown-pit cases, and 17 invalid anchors. Target-side timing exclusions are counted separately. Missing or invalid input values are not silently repaired to increase retention.

## Recent race checks

| Session | Observed lap records | Observed nominal origins | Eligible and scoreable | Retention |
|---|---:|---:|---:|---:|
| 11731, Malaysia-hosted Bahrain event | 55 | 133 | 130 | 97.74% |
| 11388, Singapore | 62 | 182 | 180 | 98.90% |
| 11396, Austin | 56 | 164 | 162 | 98.78% |

These three observed lap lists start at lap 1 with no internal gaps. Direct comparisons found no eligible forecast cutoff inside a source pit interval, no origin conflicting with a known global red flag, and no target duration differing from its source sector-duration field. The checks do not validate original API publication latency or the spatial accuracy of telemetry.

Two scoreable sectors in session 11731 encounter a later Safety Car deployment despite an eligible origin. They remain in operational evaluation instead of being removed as difficult examples. Each of the three races also has one eligible final sector spanning the chequered flag; those are separately identified rather than described as surprise neutralizations.

## Independent acquisition comparison

Austin was acquired through two routes: 96 bounded JSON queries and one full-race CSV query. Both returned 21,252 unique timestamps. Every timestamp matched; speed, throttle, brake, RPM, gear and optional DRS had zero mismatches after explicit numeric and UTC normalization.

Building examples independently from each source representation produced the same 168 observed sector examples, identical features and identical eligibility decisions. This verifies that the CSV acquisition shortcut preserved the bounded-query data for this race. It is not a proof that every provider session is complete.

## Hungary exposes selective missingness

Hungary session 11342 retains only 28 of 206 observed nominal origins, or 13.59%. The original CSV contains brake value 104 in 6,551 of 22,154 rows and throttle value 104 in 2,077 rows. These out-of-range values appear in the provider response itself and are masked under the existing channel rules. No threshold was changed to improve retention.

There are 59 native timestamp gaps longer than one second; the largest is 28.32 seconds. Usable brake coverage, rather than a shortage of speed samples alone, drives most exclusions: 177 origins fail the required-channel coverage rule. This race can therefore have a selective evaluated subset even when the aggregate cohort looks healthy. Its valid held-out forecasts must remain visible, accompanied by its low retention and sample count.

## Tensor values are not native observations

Across eligible windows in the recent three races, mean usable speed coverage is approximately 98.8–99.0%, while exact native-on-grid speed fractions are approximately 0.34–0.37%. Source timestamps rarely align exactly to a 4 Hz grid. The difference reflects bounded interpolation and holding under explicit masks, not a higher-frequency measurement source. Native timestamps and counts remain available for audit.

## Reproduction

`materialize_available_sessions` in `ant26.datasets.materialize` reads a single acquisition-progress snapshot, pins every source to its exact checksum, reuses immutable per-session artifacts, and writes a deterministic merged envelope. Cache identity includes the protocol and the entire feature-builder module's checksum so future feature-code changes invalidate earlier materializations. Incomplete and unscoreable sessions are listed explicitly. It never follows mutable latest-cache pointers or contacts the provider.

Rebuild with the current protocol and catalog through that function using `data/audit/bulk-progress.json` and `data/audit/catalog.json`. A changed protocol or source snapshot creates a new materialization; it must not replace this reviewed artifact. The review provides data-quality evidence only. Predictive accuracy, interval calibration, atlas usefulness and the final Mexico case study have separate gates.
