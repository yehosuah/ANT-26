# ANT-26 portfolio v0.1.0

An inspectable F1 research project combining masked sequence learning, probabilistic next-sector forecasting, and experimental continuous retrieval.

- Five-page offline portfolio: `make install && make demo`.
- 160 fixed-budget encoder fits across eight chronological race holdouts and five seeds. No improvement gate passed; CatBoost is retained for forecasting.
- Audited development cohort: 18 Antonelli and 17 scoreable Russell races, 760,694 telemetry rows.
- Mexico diagnostic replay: 115 scored sectors, 95 abstentions and three missing records across all 71 laps. The original zero-forecast attempt, implementation repair and unchanged development examples are preserved.
- Immutable source/model provenance, verified release boundaries, meaningful leakage/corruption tests, CPU inference measurements, and API/UI container checks.

The Mexico replay is post-inspection diagnostic evidence. All 75 sector origins after lap 46 abstain under the fixed pit-context interpretation. Retrieval has zero human ratings and remains experimental. Raw datasets, individual forecasts, model weights and private review files are excluded from this release.

Start with the README, engineering decisions, development results and Mexico case study. The attached silent walkthrough demonstrates the public aggregate-only application.
