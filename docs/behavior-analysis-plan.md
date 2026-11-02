# Observed driving analysis: analysis plan

Recorded November 2, 2026, before computing the comparisons in this analysis.
Earlier forecasting work and Mexico inspection are already public. This is a new
retrospective observational analysis, not a blind confirmatory study.

## Question and population

What repeatable differences in observed pedal use can be measured between Kimi
Antonelli and George Russell in the available 2026 Mercedes race snapshots?
Mexico is a separate, previously inspected Antonelli-only case study. No Mexico
observations enter development matching, thresholds, fitting or reference indexes.

Use completed sectors from native telemetry rather than the preceding 32-second
forecast windows. Three primary measures are time fractions at full throttle
(at least 98%), brake applied (binary), and coasting (throttle at most 5%, brake
off, speed at least 80 km/h). These are sensor observations, not pressure,
steering, precise racing lines, intent or a car-independent driver signature.
Thresholds define descriptive measurements; repeat with stricter quality and
context matching to assess sensitivity.

## Eligibility and comparisons

Require valid complete sector timing, continuous green conditions, valid channel
values and at least 90% native time coverage. Never bridge telemetry gaps longer
than one second. Exclude starts, pit laps, adjacent laps and pit-out laps. Record
local-yellow and weather limitations. Keep the selective Hungary source coverage
visible. Exact pit timing loops remain uncertain, so lap exclusions are preferred
to an asserted physical occupancy interval.

Match drivers within race, sector, compound, tyre-age bins of three laps and race
progress bins of one tenth. Average observations within each matched cell, cells
within each race, then races equally. Publish the retained counts and unmatched
losses. Matching controls only recorded context: fuel, setup, traffic, weather,
instructions and lap timing can still differ. Additional stricter matching,
coverage and traffic restrictions are sensitivity analyses, not searches for a
preferred result.

Use complete race blocks for bootstrap uncertainty (seed 17, 10,000 resamples).
Report ordinary 95% intervals and family-wise intervals for three measures using
Bonferroni (98.333% individual intervals). Report race-level sign consistency and
leave-one-race-out stability. A direction claim requires its family-wise interval
to exclude zero and its sign to survive every available declared sensitivity;
otherwise report the uncertainty. Do not treat sectors or native samples as
independent races. Findings describe this finite two-driver cohort.

## Mexico case

Publish native descriptive summaries for opening, post-red-flag restart and final
green running, with exclusions and counts. Tyre age, fuel, traffic and restart
conditions differ between phases. Phase contrasts cannot establish learning,
adaptation, tyre management intent or a causal effect. Do not reuse model residuals
as explanations of driver behavior. Keep the preserved prediction releases intact.

## Deliverable and checks

Publish safe aggregate results, a readable findings memo, and reproducible analysis
code. Keep individual sectors, exact source snapshots and checksums under ignored
artifacts/data. Validate time weighting, quality masks, matching, race aggregation
and bootstrap behavior with meaningful tests; independently audit source samples
and arithmetic. Record null results and failed robustness gates alongside any
supported differences. Human atlas validation remains open.
