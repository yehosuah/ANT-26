# Release decision

The November 1, 2026 implementation checkpoint retained CatBoost for probabilistic forecasting and left the atlas human-review gate open. The user subsequently authorized unblocking the highest-impact work and continuing through a respectable portfolio release.

This release resolves that gate by narrowing the atlas claim. Continuous retrieval is available as an experimental visualization of observed telemetry similarity. It has **zero human ratings**, no validated driving-style labels, and no claim of superiority to the feature/PCA reference. The blind review pack remains available for later, explicit human evaluation. The failed stability check remains reported.

CatBoost remains the forecasting model selected on development WIS. The neural forecast head is excluded from the final case study. Its corrected experiments and distribution-shift failures remain part of the public research record. No further model selection, normalization changes, or test-fold retuning are required to make the project presentable.

Before querying Mexico through the final workflow, freeze the candidate bundles, exact reference index, protocol, feature builder, scoring code, development dataset identity, and this review decision in an immutable release manifest. Verify that manifest before source acquisition and again before offline scoring. Existing candidate metadata is preserved; the separate release manifest records the freeze.

Mexico is one external case study held out from fitting and selection. Earlier exploratory inspection exposed its outcome and a short source window, so it is not an untouched blind test. Report the complete 71-lap accounting, missing records, origin abstentions, invalid targets, calibration failures, and representative successful and difficult predictions. Case-study observations must not become changes to the frozen model.

## Implementation correction and final claim

The original frozen attempt produced zero eligible forecasts because an unknown-duration pre-race pit record indefinitely excluded historical references. That attempt is preserved. A tested, prefix-safe implementation repair bounds unknown exits only with completed valid non-pit-out lap evidence available by the cutoff. Known-duration pit interpretations and learned models remain unchanged.

All 6,204 development examples were independently rebuilt from the same snapshots and found identical, including order and quality results. The second freeze binds the first freeze/report and this rationale while enforcing identical model artifacts, index, protocol and canonical training examples. The corrected case is explicitly a **post-inspection diagnostic replay**, not a new independent test. Its 115 scored sectors do not cover the 75 sector opportunities after lap 46, all of which abstain under the fixed pit-context interpretation.
