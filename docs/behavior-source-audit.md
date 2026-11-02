# Native driving-observation source audit

The new observations describe completed sectors retrospectively. They are separate from the 32-second pre-origin forecast windows and cannot be used as inputs for their own sector forecasts. Native pedal-use fractions support statements about observed control use in the retained conditions. They do not identify intention, adaptation, confidence, tyre preservation, braking pressure or a causal driver effect.

## Source and measurement checks

The audited observation envelope is `artifacts/behavior-v1/observations.json`, file SHA-256 `f24e858b30ddb7fea344eed9b5c225a9aa560ff892436eb2527943e9550effd9`. It contains 6,204 development observations, of which 3,173 are retained, and a separate 210 Mexico observations, of which 121 are retained. Source manifests and rows remain private. The envelope pins its builder, imported feature/materialization helpers and measurement policy.

Whole sectors use the approximate lap-start timestamp plus cumulative reported sector durations. A complete-lap timing defect excludes that lap. Missing first-sector timing is not backfilled from a future lap end. In Mexico, laps 46 and 65 have missing first-sector timing and contribute six excluded sector records. The absent lap-5 record adds three missing slots: 210 observed plus three missing equals the official 213 sector slots.

Metrics use native speed, throttle and binary brake with elapsed-time weights. A sample is held until the next native sample, sector end or one second, whichever comes first. There is no interpolation or synthetic tensor-grid sampling. All three channels must be valid for time to enter the common denominator. Coverage must be at least 90%; any overlapping native gap above one second also excludes the sector. Thus retained fractions describe jointly observed time, while excluded and missing sectors stay visible.

Full throttle means at least 98% throttle. Coasting means throttle at most 5%, brake off and speed at least 80 km/h. Brake use is binary time on the brake, not pressure or force. Provider values of 104 are invalid, not values to clip into the valid range.

Mexico's 30,924 native samples contain 1,363 invalid throttle values and the same number of invalid brake values, all equal to 104. Speed, RPM and gear have no out-of-range values under the declared source checks. There are 85 native timestamp gaps above one second, a maximum gap of 29.640 seconds and no duplicate timestamps. The median native gap is 0.240 seconds. These are full-source counts, not counts solely inside retained sectors.

An independent integration directly over the cached Mexico samples checked all 204 anchored observations. The maximum covered-time difference was 0.0000002384 seconds and the maximum control-fraction difference was 0.00000001163, explained by microsecond ISO boundary round-trips. All 210 stint assignments and pit-adjacency decisions matched independent source calculations. Evidence is preserved at `artifacts/behavior/mexico-audit.json`; it performs no provider acquisition.

## Flags, pits and tyre attribution

Global green must hold throughout the completed sector. Driver blue flags do not become a persistent track state. Local source sector identifiers range up to 16 in Mexico, so they cannot be equated with the three timing sectors. Without track geometry, any uncleared local yellow conservatively vetoes the observation.

A global green event does not silently clear a local marshal-sector state. Missing local CLEAR messages could therefore cause conservative stale-state exclusions; a veto is source uncertainty rather than proof that a driver physically encountered a yellow flag. In Mexico, 21 observations carry a local veto, nine of them without a global non-green reason; seven have no other exclusion reason. All recorded local states are clear by the 21:45 restart and at the end. The retained late-race sample is selective, but there is no indefinitely uncleared local state blacking out the entire restarted phase in this source.

The retrospective pit exclusion centers both recorded pit laps and pit-out laps, then removes one adjacent lap on each side. In Mexico the excluded union is laps 1–2 and 42–47, totaling 24 observed sectors. This intentionally conservative rule avoids relying on disputed physical entry/exit timestamps. It uses completed-event evidence and must not be promoted into an earlier forecasting input.

Compound and age come from the source stint whose lap range contains the observation. Reported tyre age is the initial stint age plus laps since its start. Mexico has a soft stint on laps 1–43 beginning at age zero, medium tyres on 44–45, and soft tyres on 46–71 beginning at reported age four. The retained observations contain soft tyres only; the medium phase has no retained observations. Source stint boundaries and ages do not measure tyre condition or wear directly.

## Mexico descriptive results

Controls below pool jointly covered time within the phase. Pace comparisons should use the same timing sector; pooled phase pace mixes sector geometry and selective exclusions.

| Phase | Retained / observed sectors | Tyre age range, laps | Full throttle | Brake on | Coast |
|---|---:|---:|---:|---:|---:|
| Opening, laps 1–42 | 83 / 123 | 2–40 | 31.04% | 25.25% | 6.64% |
| Pit/interruption, 43–46 | 0 / 12 | No retained sample | — | — | — |
| Restarted phase, 47–62 | 24 / 48 | 6–17 | 34.14% | 23.99% | 6.68% |
| Final run, 63–71 | 14 / 27 | 21–29 | 31.59% | 25.33% | 6.12% |

The opening retained ages span most of a long stint, despite that stint beginning on new tyres. Restart and final observations belong to the later used-soft stint. Fuel load, track evolution, traffic, neutralizations and selection also differ. Higher observed restart full-throttle time cannot by itself establish adaptation or a deliberate strategy.

Same-sector median durations, in timing-sector order 1/2/3, are 27.981/32.576/20.8985 seconds in the opening sample, 27.660/31.8335/20.513 after the restart, and 27.569/31.891/20.316 in the final sample. Their retained counts differ: 27/30/26, 9/8/7 and 4/6/4. These are observed conditional pace summaries, with no phase-level causal or inferential comparison.

All 121 retained sectors have effectively complete joint native coverage under floating-point tolerance. Across the 210 observed sectors, 89 are excluded and three additional sector slots are missing. Reasons overlap: 56 have a gap above one second, 24 are pit-adjacent, 21 have a local-flag veto and 17 are not continuously globally green. These reason counts must not be added as if they were disjoint populations.

## Development comparison cross-check

The independent matched-cell reconstruction reproduced 455 cells, 1,584 matched observations and 14 races from the 3,173 retained development observations; 1,589 retained observations remain unmatched. Mexico is excluded. Matching conditions are race, timing sector, compound, three-lap tyre-age bin and tenth-race progress bin, followed by equal-cell averaging within each race and equal-race averaging.

Antonelli minus Russell contrasts independently recompute to −0.48737 percentage points for full throttle, −0.39376 for brake use and −0.94249 for coasting. The coast result is conditional on this retained matched population. Its primary Bonferroni-adjusted interval excludes zero, while throttle and brake intervals include zero. Tighter progress bins and 95% native-coverage sensitivity intervals include zero for coast, so the finding must not be described as significant under every specification or as a general driver trait.

## Reproduce the safe Mexico summary

```bash
PYTHONPATH=src uv run --no-sync python scripts/audit_behavior.py \
  --observations artifacts/behavior-v1/observations.json \
  --output artifacts/behavior/my-mexico-summary.json
uv run --no-sync pytest tests/test_mexico_behavior.py
```

The summarizer is `ant26.research.mexico.summarize_mexico`. It returns counts, covered-time aggregates, tyre context and sector-specific pace summaries without individual observations. `reports/mexico-driving-behavior.json` is its safe public export. Existing output paths are refused by the command. Tests cover time weighting, overlapping reason counts, empty groups, invalid identities/coverage, and real timestamp round-off.
