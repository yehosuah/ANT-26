# ANT-26 atlas review protocol

The atlas needs evidence that its neighbors are useful to a human reviewer. Forecast accuracy, reconstruction loss and visually separated embeddings cannot supply that evidence. The implemented review preparation creates blinded comparisons; it never fills preferences or claims that a human has reviewed the results.

## Fixed comparison

Prepare 60 queries from at least six chronological held-out races, with query selection and A/B assignment fixed by seed 17. With all eight development folds available, the queries are distributed across all eight races. The intended learned candidate is the predeclared fine-tuned seed-17 checkpoint for each fold, compared with the standardized-feature/PCA reference baseline.

Both methods receive the exact same fitting examples. The helper checks their canonical example checksum against the encoder checkpoint's `fitting_examples_sha256`; a matching session list alone is insufficient. It also checks the explicit fitting/test scopes, the chronological fold, calibration scope, protocol checksum and seed. Mexico cannot enter fitting or review queries.

Only checkpoints declaring normalization policy `constant_unit_scale_v2` are accepted. Legacy normalization artifacts are excluded because a constant training context channel could previously amplify a newly observed value. A trial review generated from legacy checkpoints was quarantined before any human responses; it is not a review result or a release candidate.

The learned reference vectors are encoded in batches from the frozen fold checkpoint. The feature baseline fits its imputation, scaling and PCA only on the same reference rows. Neither index is updated from test queries. Queries use past-only windows, and retrieved references cannot be future, same-lap or overlapping same-driver windows.

## What the reviewer receives

`blind-review.json` contains each query's observed telemetry window and two candidate neighbor sets labeled A and B. It includes channels, usable masks, context, source identities and observable summaries. Numerical embedding distances, method names, model paths and index identities are withheld because their scales or names can reveal the method.

`response-template.json` has blank choices. A reviewer records a reviewer identifier, then A, B, tie or uncertain for each comparison, with optional notes. The scorer also accepts the original JSON template's `neither` choice without treating it as a preference. The question is similarity of observed execution given context. It does not ask the reviewer to infer intention, confidence or tyre preservation.

`private/answer-key.json` records the A/B assignment, checkpoint and reference checksums, and fold scope. Keep it separate from the reviewer-facing files until responses are complete. These generated artifacts belong under the ignored `artifacts/` directory. They are not automatically published or sent to anyone.

`score_review_responses` accepts explicit response files, preserves ties and missing answers, checks review identity, and reports counts. It marks a single-reviewer result exploratory. It does not turn model outputs into human preferences, and it does not claim validated driving styles from preference counts.

## Optional pattern stability

`bootstrap_cluster_stability` applies a training-only PCA with at most eight dimensions and diagonal-covariance Gaussian mixtures. The component count is selected by BIC on fitting data. Twenty race-level bootstrap resamples refit that fixed count, and adjusted Rand indices compare assignments on a fixed disjoint development probe set.

The proposed median ARI target is 0.6. A failed, degenerate or incomplete check removes discrete-state claims; continuous retrieval can still be shown with its limitations. Passing stability does not establish meaningful driver intention or substitute for human retrieval review. The output always keeps `validated=false` until the broader review criteria are assessed.

## Implementation and status

The helpers live in `src/ant26/atlas/review.py`. Supply explicit checkpoint directories and fold scopes to `build_review_set`, together with the exact canonical example dataset, event catalog and protocol. Checkpoint loading verifies model bytes; review preparation verifies fitting-row identity before computing neighbors. Existing review files refuse replacement.

The offline tests cover reproducible blinding, zero prefilled answers, exact-reference mismatch, bad scope, Mexico exclusion, legacy normalization rejection, explicit response handling and disjoint stability probes. Fixture tests do not establish real retrieval quality.

Real reviewer preferences remain pending. The corrected-checkpoint review pack and any actual bootstrap result must be recorded separately when generated. No human-retrieval or discrete-state validation claim is currently supported.

The corrected pack is now prepared at `artifacts/atlas-review-ant26-v2-seed17/blind-review.json`: 60 queries across all eight chronological held-out races, with review identifier `6a38cb3a3e407073511e81d0f89750d4654201a6dabdb45aeb0eec69d74dcc1a`. All exact fitting-row checksums and `constant_unit_scale_v2` checkpoint policies passed. The separate response template contains 60 blank choices; no real ratings were created.

Launch the local human-review form with `uv run --extra app streamlit run app/review.py --server.port 8502`. It reads the blinded pack and saved human responses, never the private assignment key. Speed, throttle and brake plots overlay the query with neighbor A or B at the selected rank, showing source identities, context and masked gaps. Enter a reviewer alias and press Save response only after a deliberate choice; navigating or changing a radio selection alone writes no file. Responses go to the ignored `artifacts/atlas-review-answers/` directory. The tests verify initial unrated state, six plotted comparisons, no implicit writes, explicit-save behavior and private-key rejection.

The first corrected-checkpoint stability check used the seed-17 fine-tuned encoder from rolling fold 11, 1,333 exact fitting windows, and 28 eligible Hungary probe windows. All 20 race bootstrap fits completed. The median adjusted Rand index was 0.32123, below the 0.6 target, so discrete-state claims are dropped. The local evidence is `artifacts/atlas-stability-corrected-rolling11.json`. This is one development fold with selectively observed Hungarian telemetry, not a general verdict on every representation or a human retrieval assessment. Continuous retrieval remains available with its own pending review.
