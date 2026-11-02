# What the observed telemetry says about Antonelli

November 2, 2026. Retrospective observational analysis of the cached 2026 race
snapshots. [Analysis plan](behavior-analysis-plan.md),
[machine-readable results](../reports/driver-behavior.json),
[Mexico phase summaries](../reports/mexico-driving-behavior.json).

## Antonelli shows less coasting in the matched observations

Antonelli's matched coasting fraction was **4.29% versus Russell's 5.23%**:
**−0.94 percentage points**, with a Bonferroni-adjusted bootstrap interval of
**[−1.81, −0.06] pp**. Coasting means throttle at most 5%, brake off and speed at
least 80 km/h, measured over jointly valid native telemetry time.

This comparison contains **14 races, 455 matched context cells and 1,584 sectors**
(801 Antonelli, 783 Russell). Drivers are matched within race, timing sector,
compound, three-lap tyre-age bins and one-tenth race-progress bins. Observations
are averaged within cells, then cells within races, then races equally. These
are matched-cohort averages, not whole-season time fractions.

The effect is negative in **10 of 14 races**. Removing any one race leaves a
negative estimate, between −1.16 and −0.79 pp. All five planned sensitivity point
estimates are also negative. However, the adjusted intervals under tighter race
progress matching and 95% native coverage include zero. The primary interval's
upper endpoint is close to zero. This supports a qualified observed difference;
it does not establish a permanent driving style, intention, energy-management
strategy or pace advantage.

| Matching or quality check | Matched races / cells | Coasting difference (pp) | Adjusted interval (pp) |
|---|---:|---:|---:|
| Primary | 14 / 455 | −0.94 | [−1.81, −0.06] |
| Exact tyre-age lap | 14 / 617 | −1.04 | [−1.84, −0.32] |
| Race-progress bins of 0.05 | 14 / 456 | −0.96 | [−1.88, +0.03] |
| Native coverage at least 95% | 14 / 442 | −0.88 | [−1.77, +0.02] |
| Reported gap ahead at least 2 seconds | 6 / 43 | −1.74 | [−3.08, −0.56] |
| Dry observations only | 14 / 451 | −0.93 | [−1.79, −0.05] |

The traffic check has only 132 matched observations and excludes missing or stale
gap reports, including unavailable leader intervals. It is a restricted context
check, not proof that all traffic effects have been removed. Sensitivity
intervals account for the three measurements within each check; they are not
jointly adjusted across all sensitivity analyses.

A separate **98% coverage audit**, requested before measurements but omitted from
the first comparison configuration through a coordination error, was executed
after the primary result. It preserves the original result and yields
**−0.91 pp [−1.79, −0.02]**, with 14 races, 417 cells and 1,433 sectors. The audit
was one fixed additional check, with no threshold search. Its near-zero upper
endpoint reinforces the need for qualified wording.
[Audit and amendment provenance](../reports/driver-behavior-audit.json).

## Full-throttle and brake-use differences remain unresolved

| Measurement | Antonelli | Russell | Antonelli minus Russell (pp) | Adjusted interval (pp) |
|---|---:|---:|---:|---:|
| Full throttle, at least 98% | 47.58% | 48.07% | −0.49 | [−1.86, +0.98] |
| Binary brake applied | 20.56% | 20.95% | −0.39 | [−1.03, +0.21] |
| Coasting, as defined above | 4.29% | 5.23% | −0.94 | [−1.81, −0.06] |

The first two intervals span zero. The data do not establish either equal driving
behavior or a directional difference in these measures. Binary brake telemetry
cannot distinguish braking pressure, exact braking points or trail-braking
technique. These three fractions are not an exhaustive partition of driving time.

## Mexico: quicker retained final-sector medians, with changing context

Mexico is Antonelli-only, previously inspected diagnostic evidence. It was
excluded from the teammate comparison and from all model fitting and retrieval
references. The completed-sector analysis retains **121 of 210 observed sectors**;
three expected sectors are missing with lap 5. There are 83 opening, zero
interruption, 24 restart-phase and 14 final-phase retained sectors.

Compare the same timing sector rather than pooling different track geometries:

| Timing sector | Opening laps 1–42: median seconds (n) | Restart phase 47–62 | Final laps 63–71 |
|---|---:|---:|---:|
| S1 | 27.981 (27) | 27.660 (9) | 27.569 (4) |
| S2 | 32.576 (30) | 31.834 (8) | 31.891 (6) |
| S3 | 20.899 (26) | 20.513 (7) | 20.316 (4) |

Final-phase medians are lower than opening medians in each retained timing-sector
subset. The pattern is not uniformly improving from restart to finish: S2's final
median is slightly higher than its restart median. Sample sizes are small and
selection differs between phases.

Observed full-throttle fractions rise from opening to final within each timing
sector: **56.99% → 60.16% in S1**, **19.42% → 20.62% in S2**, and
**15.87% → 18.80% in S3**. Coasting does not move uniformly: S1 is approximately
unchanged (6.41% → 6.54%), while S2 and S3 decline (6.22% → 5.63%,
7.71% → 6.69%). These are covered-time descriptions, not inferred actions during
missing intervals.

All retained observations use soft tyres. Reported tyre ages range from 2–40 laps
in the opening subset, 6–17 in the restart subset and 21–29 in the final subset.
Fuel, track evolution, traffic and neutralizations also change. These comparisons
cannot establish adaptation, deliberate tyre preservation or why he achieved his
race result. No Mexico teammate comparison is available in this source cohort.

## What makes these findings defensible—and limited

- Native telemetry is measured over completed sectors; preceding forecast windows
  are never mislabeled as execution of their target sector. Native zero-order
  holds last at most one second. No interpolation creates additional resolution.
- Require valid timing, at least 90% joint speed/throttle/brake coverage and no
  native gap exceeding one second. Exclude starts, pit and pit-out laps plus their
  neighbors, non-green periods and any uncleared local marshal-sector warning.
  Unknown local clears are conservative source uncertainty, not physical truth.
- Of 6,204 development sectors, 3,173 pass measurement/context gates; 1,584 have
  common matching support. Belgium has no scoreable Russell race; Italy, Madrid
  and Malaysia have no common cells under the specified matching. Hungary
  contributes only 24 matched sectors across six cells, reflecting its invalid
  source values and selective coverage. This is a selected two-driver cohort.
- Bootstrap uncertainty resamples complete races 10,000 times, seed 17. The
  displayed adjusted intervals use 98.333% individual coverage for three primary
  measures (Bonferroni). Race exchangeability and percentile bootstrap calibration
  are assumptions; these are not randomized driver assignments.
- Race/context matching cannot isolate the driver from car setup, strategy,
  conditions or remaining within-bin differences. Exact racing lines, steering,
  brake pressure and live API publication latency are unavailable.

The previous forecasting releases, failed neural experiments, frozen models and
Mexico source snapshots remain unchanged. Continuous atlas retrieval still has
zero human ratings. This analysis validates measured telemetry summaries, not an
embedding-based taxonomy.

## Reproduction

The public portfolio's **Driving Findings** page works from the aggregate reports
alone. Exact reproduction additionally requires the locally cached, pinned source
manifests; no raw datasets are committed:

```bash
PYTHONPATH=src uv run --no-sync python scripts/reproduce-behavior.py
# Optional separate Mexico-only export to a new path:
PYTHONPATH=src uv run --no-sync python scripts/audit_behavior.py \
  --observations artifacts/behavior-v1/observations.json \
  --output artifacts/behavior-v1/mexico-summary-check.json
```

Different inputs or code require new output paths. Existing immutable artifacts
are retained. [Source audit](behavior-source-audit.md) records the independent
checks, exclusions and source limitations.
