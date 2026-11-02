# ANT-26

[![Checks](https://github.com/yehosuah/ANT-26/actions/workflows/ci.yml/badge.svg)](https://github.com/yehosuah/ANT-26/actions/workflows/ci.yml)

**Can a short telemetry history improve the next-sector forecast?** ANT-26 studies Kimi Antonelli's 2026 season with masked sequence learning, calibrated quantile forecasts, and an experimental behavioral atlas.

The benchmark retained CatBoost. The project includes 160 representation fits, a complete research pipeline, an offline portfolio application, and a Mexico case study that preserves a failed first attempt and its implementation correction.

**V0.2.0 resolves the pit report-time interpretation and recovers 62 late-race scored sectors.** Both published case versions remain selectable. [Correction and evidence](docs/pit-timestamp-correction.md), [cohort comparison](reports/pit-policy-comparison.json).

![ANT-26 portfolio overview](docs/assets/overview.png)

## Try the portfolio

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). From a fresh checkout:

```bash
make install
make demo
```

Open `http://127.0.0.1:8501`. The five pages work with committed aggregate results. No API, provider credentials, downloaded telemetry, trained weights, or GPU is required. `docker compose up --build` also starts the portfolio and read-only API.

Watch the [original v0.1.0 walkthrough](docs/walkthrough.md), or browse the [v0.2.0 release](https://github.com/yehosuah/ANT-26/releases/tag/v0.2.0). The sidebar selects the revised or preserved case evidence.

The API-backed artifact viewer remains available through `streamlit run app/replay.py` after starting `ant26 serve`. Local immutable Mexico artifacts add individual forecasts and experimental neighbors to the portfolio; the public checkout shows real aggregate evidence and full lap accounting.

## What the evidence says

| Evidence | Result |
|---|---|
| Development data | 18 Antonelli races, 17 scoreable Russell races; 760,694 native telemetry rows |
| V0.2 forecast dataset | 6,204 observed sectors, 5,249 scoreable examples; Mercedes cohort only |
| Historical v0.1 neural evaluation | Eight whole-race holdouts, five fixed seeds, paired race bootstrap; 559,104-parameter masked encoder |
| V0.2 baseline evaluation | Eight chronological holdouts; fixed model settings; 1,137 Antonelli test forecasts |
| V0.2 forecast selection | CatBoost MAE 0.909 s, WIS 0.581; new-policy development population |
| Neural outcome | No v0.1 improvement gate passed; those experiments and tail failures remain versioned |
| V0.2 Mexico diagnostic replay | 177 scored, 32 abstentions, one censored, three missing; MAE 0.876 s, WIS 0.610 |
| Late-race coverage | 62 of 75 sector slots scored after lap 46; preserved v0.1 scored none |
| V0.2 local CPU inference | CatBoost p95 0.716 ms; encoder p95 0.372 ms, warm batch-one calls |

Development scores evaluate Antonelli. Historical neural results remain attached to v0.1; the new protocol changes development inputs and requires refitted artifacts. Mexico is **post-inspection diagnostic evidence**. On the common 115 sectors, refitted MAE is nearly unchanged and WIS is slightly worse; the extra 62 sectors form a separate, harder cohort. Coverage recovery is not a forecast-skill improvement claim. [Detailed comparison](docs/pit-timestamp-correction.md).

## Inspect the engineering

- [Pit-timestamp revision](docs/pit-timestamp-correction.md): source semantics, cutoff safety, refitting and recovered late coverage.
- [Preserved v0.1 Mexico case](docs/mexico-case-study.md): original accounting, source limitations and first-attempt history.
- [Engineering decisions](docs/engineering.md): architecture, learning objective, leakage boundaries, normalization failure and model selection.
- [Development results](docs/development-results.md): complete comparisons, paired intervals, cohort scope and failed gates.
- [Reproduction guide](docs/reproducibility.md): fresh source acquisition through experiments, freeze and offline case scoring.
- [Operational evidence](docs/operations.md): measured preprocessing/inference phases, hardware and warning-only drift diagnostics.
- Preserved v0.1 [data quality](docs/real-data-quality.md), [data card](docs/data-card.md), [model card](docs/model-card.md), and [protocol](configs/evaluation-protocol.json); the [v0.2 revision guide](docs/pit-timestamp-correction.md) supplies the new scope and protocol.

The [aggregate reports](reports/) contain exact metrics, hashes, 71-lap derived accounting, and the [development invariance audit](reports/development-correction-impact.json). Raw source records, individual forecasts, weights, private review assignments and ratings remain outside Git.

## Reproduce and verify

```bash
make check
make smoke
uv sync --locked --group dev --extra ml --extra app --no-editable
scripts/reproduce-development.sh development --run-id my-source-revision
```

The smoke pipeline is explicitly synthetic integration evidence. The development script reproduces the original v0.1 protocol and 160 fixed-budget fits; it can take time and does not fetch Mexico. Use the v0.2 revision guide for the new pit policy. A separate verified freeze and explicit final-case opt-in are required. Re-fetching a mutable provider can produce different source bytes and hashes; new revisions become new runs.

CI checks lint, formatting, domain/API types, offline tests, optional CPU ML/UI tests, and API/UI container builds and HTTP health. Tests cover future-record mutations, missing laps, unresolved pit exits, source reconstruction, artifact corruption, freeze boundaries, censoring and UI use without private assets.

## Research limits

OpenF1 provides sampled telemetry, not exact racing lines. A 4 Hz tensor grid adds no native resolution. Sector entries and historical availability are reconstructed; verified live-publication latency is unavailable. The observed population is Antonelli and Russell, and it cannot separate driver behavior from the car, setup and strategy.

Continuous atlas retrieval remains experimental with **zero human ratings**. Cluster stability failed, so discrete driving-state claims were removed. The [blind review protocol](docs/atlas-review.md) remains available for later human evaluation. No result attributes championship success, intent, confidence, or tyre preservation to an embedding.
