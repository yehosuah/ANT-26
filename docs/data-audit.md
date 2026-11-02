# ANT-26 source feasibility audit

The first audit successfully fetched recent 2026 historical data for Antonelli and Russell without credentials. The audit covers three races and six one-minute telemetry windows; it establishes source feasibility, not full-race retention or predictive performance. Mexico telemetry was not fetched.

## Event identity

The catalog contains 18 non-canceled race sessions before Mexico, with Mexico meeting 1298 reserved for the final case study. Bahrain session 11731 is a legitimate event identity: the Bahrain Grand Prix was held at Sepang, Malaysia, on October 4. Raw provider fields remain unchanged, while a sourced override supplies canonical host country and circuit. [Official Formula 1 event page](https://www.formula1.com/en/racing/2026/bahrain), [official timetable](https://www.formula1.com/en/latest/article/formula-1-gulf-air-bahrain-grand-prix-2026.1IHoy7D2dfCIugb5mH78cE), [FIA press conference schedule](https://www.fia.com/news/f1-2026-bahrain-gp-malaysia-schedule-press-conferences).

The canceled April Bahrain and Saudi Arabia entries are excluded. Mexico and future races never enter development roles. The catalog retains exclusion reasons, raw source identity, and canonical identity separately.

## Observed data

Each sampled race has full available lap, stint, interval, pit, driver, weather, and race-control responses. Telemetry queries intentionally cover only 60 seconds near lap 10 for each driver.

| Session | Circuit | Driver | Lap rows | Telemetry rows | Median sample gap | Maximum sample gap |
|---|---|---|---:|---:|---:|---:|
| 11731 | Sepang | Antonelli 12 | 55 | 224 | 0.240 s | 0.720 s |
| 11731 | Sepang | Russell 63 | 50 | 223 | 0.241 s | 0.720 s |
| 11388 | Singapore | Antonelli 12 | 62 | 225 | 0.240 s | 0.840 s |
| 11388 | Singapore | Russell 63 | 62 | 229 | 0.240 s | 0.840 s |
| 11396 | Austin | Antonelli 12 | 56 | 211 | 0.280 s | 1.039 s |
| 11396 | Austin | Russell 63 | 56 | 213 | 0.280 s | 0.480 s |

The completed audit has 39 successful session-level query snapshots, zero endpoint errors, and two additional metadata snapshots. Full-session car-data requests timed out during initial exploration; bounded windows succeeded. The client provides `fetch_time_range` to expand coverage through separately cached time chunks without relying on giant responses.

All sampled telemetry rows contain speed, throttle, brake application, gear, and RPM. DRS is null in every sampled row and remains excluded from the version 1 feature contract. No exact duplicates or duplicate timestamps were observed in these six windows. Antonelli's Austin window has one gap exceeding one second, which must remain masked rather than interpolated through.

No internal lap-number gaps or sector-sum mismatches above 0.01 seconds were observed in the sampled lap responses. Antonelli's Sepang response has one missing sector-3 duration. Russell's Sepang response has one missing lap duration and four missing sector durations across its three channels. The maximum observed lap number is not an independent official completeness denominator; full-race retention still requires reconciliation against completed-lap classifications and feature eligibility.

## Provenance and reproducibility

Ignored local artifacts are `data/audit/catalog.json`, `data/audit/audit.json`, and content-addressed snapshots under `data/raw`. Each raw snapshot preserves response bytes, query, source URL, UTC retrieval time, schema version, row count, and SHA-256 checksum. Audit source entries identify their exact snapshot. A refresh creates a new content-addressed snapshot instead of overwriting the previous response.

To repeat the same feasibility scope from the repository root:

```python
from ant26.ingestion import run_feasibility_audit

report = run_feasibility_audit(telemetry_window_seconds=60)
```

Use `offline=True` to require cached responses. To expand telemetry, call `OpenF1Client.fetch_time_range(endpoint, params, start, end, chunk_seconds=60)` with timezone-aware UTC boundaries. The client's `last_window_manifests` identifies every component snapshot. Reads outside requested time intervals fail visibly, and an offline cache miss never silently performs a network request.

## Verification and remaining work

Twenty offline ingestion tests passed, covering immutable refresh, tamper detection, offline failure, schema rejection, transient retry, authentication errors, source identity overrides, canceled/future/Mexico exclusions, lap quality, half-open bounded time queries, checkpoint recovery after a failed telemetry chunk, CSV normalization, raw CSV integrity, bulk acquisition filter checks, explicit no-results responses, and zero-timed-lap retirement handling. Formatting and lint checks passed for the ingestion package and its tests.

The initial sampled audit makes no claim about model quality or complete season coverage. Subsequent full-trace acquisition is recorded separately below.

Full-trace expansion is implemented separately through `expand_full_traces`. It fetches Antonelli's Austin trace first, then Singapore and Sepang, before moving to Russell. The ignored `data/audit/full-trace-progress.json` records every acquired chunk, its immutable source manifest, row count, and failed boundaries. Acquisition status `complete` means every requested time chunk was fetched; feature retention and source gaps still require analysis.

## Full development acquisition

All 18 pre-Mexico races now have complete race-bounded telemetry and context for Antonelli: **395,446 telemetry rows**. Russell supplies 17 additional traces and **365,248 rows**. His Belgium race has no valid timed lap and is recorded as unscorable, with no invented telemetry interval. Combined acquisition contains **35 usable driver-race traces and 760,694 telemetry rows**. Bounds extend from the first timed lap's start through the last valid timed lap's end. Untimed retirement tails remain outside those bounds. This is an Antonelli-and-Russell subset, not all available drivers.

The bulk path uses a bounded per-driver CSV query, preserving original bytes in `response.csv` and numeric/UTC-normalized records in `response.json`. Manifests carry separate raw-source and normalized checksums. An offline integrity check verifies both. The shared limiter still caps acquisition at 25 requests per minute.

For Austin Antonelli, the 96 original one-minute JSON queries and a single bounded CSV query yielded exactly the same 21,252 records after sorting by timestamp. The local `data/audit/csv-equivalence.json` records equality and the canonical data hash. This reduces that telemetry query count from 96 to 1 without changing its data. A full-race all-driver CSV pilot was rejected with HTTP 422 and a response indicating the query requested too much data; production acquisition therefore remains per-driver.

`data/audit/bulk-progress.json` uses the same progress schema as bounded expansion and records all context-source manifests. `ingest_development` is resumable because every completed query is reused from verified immutable cache. Partial sessions retain visible errors and are excluded from full-trace materialization.

Russell's Canada and Belgium pit queries returned the exact provider response `404 {"detail":"No results found."}`. Explicit opt-in normalization retains that raw body, its checksum and HTTP status while exposing an empty list of available records. It does not prove that no pit event happened. Unknown 404 bodies and ordinary errors still fail visibly. Canada supplies 29 complete timed laps before an incomplete retirement lap; Belgium has zero complete timed laps. The summary is saved in `data/audit/development-source-summary.json`.

Hungary illustrates why acquisition completeness and usable feature coverage differ. Antonelli's 22,154 telemetry rows include 6,551 out-of-range brake values of 104 and 2,077 throttle values of 104, preserved from the original CSV. Under the fixed 80% required-channel coverage rule, only 28 of 206 nominal green-sector opportunities passed origin eligibility in the initial feature check. Coverage thresholds remain unchanged; the rejected sectors stay visible in the quality report. Detailed source reports are saved in `data/audit/full-trace-quality.json`.
