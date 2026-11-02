# ANT-26 data card

ANT-26 studies observed telemetry and sector timing for Kimi Antonelli. Its present real-data evidence establishes source feasibility. It does not establish full-season coverage or model accuracy.

## Source and current coverage

Historical records come from the independent [OpenF1 API](https://openf1.org/docs/). The audited development catalog contains 18 pre-Mexico 2026 races after canceled sessions and the final case study are separated. Raw provider metadata is preserved. An explicit sourced override distinguishes the Bahrain event's identity from its Malaysian host circuit; see `configs/catalog-overrides.json`.

The saved feasibility audit covers three recent development sessions: 11731, 11388 and 11396. It includes drivers 12 and 63, race-control messages, weather, driver metadata, lap records, stints, intervals and pit records. Car telemetry is sampled for a 60-second window per driver around lap 10, with 211–229 rows in each window. These six windows cannot establish complete-race retention. Counts and source hashes are available in the locally generated `data/audit/audit.json`.

An independent synthetic fixture exercises pipeline stages and chronological splits. Its artifacts declare `synthetic=true` and `data_kind=synthetic_integration_fixture`. Synthetic forecast errors are software-test results, not findings about Antonelli.

Mexico meeting 1298, including race session 11404 and its practice/qualifying sessions, remains excluded from model fitting, normalization, calibration, clustering and retrieval references. Prior exploratory inspection exposed its outcome and a short data window, so the eventual case study must not be described as an untouched blind test.

## Units and reconstruction

Speed is in km/h; throttle uses the provider's 0–100 channel; RPM is engine revolutions per minute; gear is discrete. Brake application is represented as 0 or 1, not brake pressure. Optional source channels such as DRS do not enter the version 1 required sequence schema. Missing deployment, steering, temperature and accurate lateral-position channels are not reconstructed as measured values.

Sector-entry anchors use approximate lap starts plus durations of already completed sectors. Forecast inputs stop one second before reconstructed entry. Labels describe the subsequent sector and remain separate from inputs. This is an event-time replay assumption: original API publication times and actual live-consumer latency were not recorded.

Each input uses a 32-second, 4 Hz tensor grid. The grid adds no measurement resolution. `sequence_mask` marks values usable under the permitted short-gap rule; `sequence_observed_mask` marks exact native observations on the grid. Sample age, native sample counts and per-channel usable coverage are separate. Exact-grid native fractions can be low because source timestamps need not align with the tensor grid.

Gaps longer than one second are masked. Interpolation uses only records before the cutoff, and red-flag barriers stop cross-interruption filling. At least 80% usable coverage is required in every required channel. Padding is zero with a false mask; it is not an observation.

## Eligibility and label validity

Origin eligibility requires a known green track state, an outside-pit state under documented source assumptions, sufficient telemetry, and two prior valid observations of the same driver and sector. The reference is the median of up to three recent observations. Missing history, unknown context and bad telemetry produce explicit abstentions.

Quality checks apply when their evidence becomes available. Completed-sector history cannot depend on a later sector's duration, eventual full-lap consistency, or an unobserved successor anchor. Regression tests mutate those future records and require earlier features and eligibility to remain identical.

Forecast eligibility and target validity are separate. An unexpected Safety Car during the target remains in operational scoring when its duration is valid. A target spanning a red flag or invalid timing is marked unscoreable with reasons. Retention and censoring counts must accompany accuracy; the pipeline must not remove large errors simply because they are difficult.

## Provenance and remaining work

Raw responses have retrieval timestamps, query identity and SHA-256 manifests. Processed examples retain source identity, timing assumptions and cutoffs; integration attaches exact source manifests. Refreshing source data creates a new snapshot. Data and generated artifacts are ignored by Git.

Pending evidence includes full telemetry extraction, all-driver coverage, complete-race retention, source alignment across all development races, and final Mexico completeness checks after model freeze. Public dataset redistribution requires checking the provider's applicable rights; the repository can publish code and fetch instructions without bundling the source data.
