# Pit timestamp evidence and replay boundaries

Evidence checked November 2, 2026, using primary upstream documentation/code, the official F1 results publication, and existing checksummed source snapshots. No OpenF1 refresh, paid feed access, model fitting or published-report changes were performed.

The cached Mexico evidence strongly contradicts interpreting OpenF1 pit `date` as entry and adding `lane_duration`. It supports a completion/report-time interpretation, with source timing uncertainty. The conservative candidate is to admit a record only once its original date is at or before the feature cutoff, then interpret a known duration backward. This does not establish the original API publication time.

## What the primary sources establish

The [OpenF1 API documentation](https://openf1.org/docs/#pit) defines `date` generically as UTC time and `lane_duration` as time spent in the pit lane. It does **not** specify an entry or exit timing loop. `pit_duration` is a deprecated alias; stationary `stop_duration` is distinct. A [maintainer comment](https://github.com/br-g/openf1/issues/88#issuecomment-3763108334) confirms the addition of that stationary-time field.

The inspected upstream revision is `d5cab5c9bc673611f31f829422088f9e287f21da`. Its [pit processor](https://github.com/br-g/openf1/blob/d5cab5c9bc673611f31f829422088f9e287f21da/src/openf1/services/ingestor_livetiming/core/processing/collections/pit.py) has three timestamp paths:

| Source topic | Timestamp copied into OpenF1 `date` | Duration field |
|---|---|---|
| `PitStop` | Outer message timepoint | `PitLaneTime` |
| `PitStopSeries` | Embedded `Timestamp` | Nested `PitLaneTime` |
| `PitLaneTimeCollection` | Outer message timepoint | `Duration` |

The first route replaces a matching existing stop; the other routes are fallbacks. The code performs no entry-time subtraction. It describes the last route as less accurate. The latest change to this processor is [commit `76b9c596f7c41901045c1c8d59b0c5823af64b58`](https://github.com/br-g/openf1/commit/76b9c596f7c41901045c1c8d59b0c5823af64b58), dated May 6, 2026, adding the direct `PitStop` route. Public source inspection is not proof of the deployment revision or route used for an individual cached row. Topic/version provenance is absent from those rows; null stationary time and decimal precision cannot identify the route reliably.

The [historical ingestor](https://github.com/br-g/openf1/blob/d5cab5c9bc673611f31f829422088f9e287f21da/src/openf1/services/ingestor_livetiming/historical/main.py) derives message timepoints from archive-relative time plus an inferred session origin, estimated from telemetry/position timestamps. The [live processor](https://github.com/br-g/openf1/blob/d5cab5c9bc673611f31f829422088f9e287f21da/src/openf1/services/ingestor_livetiming/real_time/processing.py) preserves the incoming message timestamp while database insertion happens later. Neither establishes when ANT-26 could first retrieve a field from the API.

The [lap processor](https://github.com/br-g/openf1/blob/d5cab5c9bc673611f31f829422088f9e287f21da/src/openf1/services/ingestor_livetiming/core/processing/collections/laps.py) sets the pit-out flag from `TimingData.PitOut`; lap starts also derive from timing messages. These are useful corroborating signals, but archived lap flags do not record their original publication times.

## Independent official publication

The [official Mexico pit-stop summary](https://www.formula1.com/en/results/2026/races/1298/mexico/pit-stop-summary) lists Antonelli's lap-43 stop at 15:02:42 local with duration 23.738 seconds. The cached OpenF1 timestamp is 25.064 seconds later. Subtracting OpenF1's rounded 23.7-second duration yields a time 1.364 seconds after the official printed time. This aligns with an entry-side official time and a later OpenF1 completion/report timestamp, allowing rounding and feed offsets. The official page does not itself define the timing-loop semantics of its time-of-day column.

This is an independent publication of timing information, not an independent physical sensor. The red-flag-associated OpenF1 record is not listed as a separate Antonelli stop on that page, so the summary does not independently validate its entire duration.

An unauthenticated read of the [official 2026 live-timing archive index](https://livetiming.formula1.com/static/2026/Index.json) returned HTTP 403 in this investigation. No credentials or paid access were attempted. Direct raw `InPit`/`PitOut` message-time verification therefore remains unavailable here.

## Exact cached alignment

The Mexico envelope is `data/case-study/mexico-v2/examples-5840cc847a4a3f06.json`, SHA-256 `5840cc847a4a3f06dd08c9a87f86566c04ae847170b75028b8af0a057624561a`. Its nine source manifests identify the existing immutable responses. Relevant normalized-response hashes are:

| Endpoint | SHA-256 |
|---|---|
| Pit | `7d058fe056d05b1ee22ac0f3178565cdd8ab789334bda97e9cb7b1c3615b61bb` |
| Laps | `5ba0f4fc43ce25bb325d636b0d4629d4baf2b8d0a84c971487e5cb0100885776` |
| Car data | `d9cb61607c7c7a05bd70a509512cb568d38bdf47116770646437a8eea086af70` |
| Race control | `2bf76136a810bbf59416de98fcd5087af11797de52abea2d4fc54167afdd1a98` |
| Stints | `13c742f603f7054a1bd80594c65164620ca1b897f572122f00145e71bf88e132` |

All following intervals use UTC and half-open bounds. They are derived checks, not exported telemetry records.

| Diagnostic | Lap-43 normal stop | Lap-45 red-flag-associated record |
|---|---|---|
| Backward interval, `date − duration` to `date` | 21:02:43.364–21:03:07.064 | 21:06:37.195–21:45:12.495 |
| Next pit-out lap | 44 | 46 |
| Next lap starts after inferred entry | 2.709 s | 3.923 s |
| Samples in backward interval | 82 | 8,518 |
| Median / maximum speed | 79 / 80 km/h | 0 / 128 km/h |
| Samples above 150 km/h | 0 | 0 |
| Samples in the forward `date` to `date + duration` interval | 84 | 8,506 |
| Forward median / maximum speed | 171 / 327 km/h | 154 / 357 km/h |
| Forward samples above 200 km/h | 33 | 2,866 |

The cached control stream suspends the race around 21:06:02 and resumes it at 21:45:00.156. The second backward interval begins shortly after suspension and ends 12.339 seconds after resumption, consistent with a duration including the red-flag wait. Stint boundaries switch to soft tyres on lap 46, and laps 44 and 46 are flagged pit-out. These are corroborations, not exact pit-entry measurements.

Adding 2,315.3 seconds to the later timestamp extends the alleged pit occupancy to 22:23:47.795, almost the race finish. The high-speed samples and subsequent completed racing laps contradict that forward occupancy interpretation. Speed alone is not a universal pit-location classifier; the comparison here tests two incompatible timing interpretations.

Across the original 35 development traces, all 64 known-duration stop records occur after the next lap's start, and all 64 next laps are flagged pit-out. The median report-to-next-lap-start delay is 19.3835 seconds. In 63 cases the backward-derived entry precedes that anchor, with a median lead of 4.020 seconds across all 64. The exception is Australia/Antonelli lap 12: subtracting the 20.985-second duration still leaves an inferred entry 19.989 seconds after the next lap starts. This prevents claiming universally exact physical boundaries from subtraction. Development changes require a new full comparison and appropriate retraining/recalibration; the earlier unknown-duration-only invariance proof cannot be reused for this different rule.

## Rule alternatives and as-of limits

1. **Completion/report proxy with an availability gate.** At each cutoff, first select records whose original `date <= cutoff`; only then derive `[date − lane_duration, date]` for finite, nonnegative known durations. Retain that original date as the evidence-availability proxy even when the derived entry is earlier. Use newly available reports to assess historical reference validity at the current cutoff. Never rewrite features emitted before the report date using that report. Preserve uncertainty for missing durations or conflicting anchors.
2. **Earlier direct state messages.** A separately timestamped `InPit` signal could support abstention before a completed report, with `PitOut` closing the state. It requires those messages and their availability semantics. A final archived pit-out flag, tyre-stint start or later report cannot be silently treated as an earlier live entry signal.
3. **Retrospective reconstruction only.** All completed reports may reconstruct past occupancy for descriptive annotation. That reconstruction must remain separate from forecasting inputs at earlier cutoffs.

For example, the record dated 21:45:12.495 cannot affect an origin cutoff at 21:06:40 solely because subtraction places its inferred entry at 21:06:37.195. It can affect a later cutoff after the report proxy becomes available. This ordering is essential for a future-record mutation test.

Even the first alternative is an explicit event-time replay assumption. The public historical responses lack field-version arrival times, and an embedded timestamp may precede the enclosing feed update. Filtering on `date` cannot prove original live availability or remove unmeasured provider latency. Keep those limits in every new diagnostic result and preserve published v1/v2 artifacts unchanged.
