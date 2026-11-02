"""Small statistical baselines; every learned operation requires a split allowlist."""

from __future__ import annotations

import math

import numpy as np

from ant26.evaluation.splits import validate_scope

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
NUMERIC_FEATURES = tuple(
    [
        "reference_seconds",
        "last_same_sector_seconds",
        "same_sector_history_count",
        "lap_number",
        "sector_index",
        "driver_number",
        "telemetry_coverage",
    ]
    + [
        f"{channel}_{summary}"
        for channel in ("speed", "throttle", "brake", "rpm", "n_gear")
        for summary in ("mean", "std", "min", "max", "last")
    ]
    + [
        "tyre_age_laps",
        "air_temperature",
        "track_temperature",
        "humidity",
        "rainfall",
        "wind_speed",
        "gap_to_leader",
        "interval",
    ]
)


def valid_examples(examples: list[dict]) -> list[dict]:
    return [
        row
        for row in examples
        if row.get("eligible", False)
        and row.get("target_valid", True)
        and isinstance(row.get("target_seconds"), (int, float))
        and math.isfinite(row["target_seconds"])
        and row["target_seconds"] > 0
        and isinstance(row.get("reference_seconds"), (int, float))
        and math.isfinite(row["reference_seconds"])
        and row["reference_seconds"] > 0
    ]


def feature_matrix(examples: list[dict]) -> np.ndarray:
    """Explicit feature names prevent injected numeric target aliases entering a fit."""
    rows = []
    for row in examples:
        features = row.get("features", {})
        values = []
        for key in NUMERIC_FEATURES:
            value = features.get(
                key,
                row.get(key)
                if key in ("reference_seconds", "lap_number", "sector_index", "driver_number")
                else None,
            )
            values.append(
                float(value) if isinstance(value, (int, float)) and math.isfinite(value) else np.nan
            )
        rows.append(values)
    return np.asarray(rows, dtype=float).reshape((-1, len(NUMERIC_FEATURES)))


class Baseline:
    def __init__(self, kind: str, ridge_alpha: float = 1.0):
        if kind not in (
            "same_sector_persistence",
            "historical_reference",
            "regularized_linear_residual",
            "catboost_quantile",
        ):
            raise ValueError(f"Unknown baseline {kind}")
        self.kind = kind
        self.ridge_alpha = ridge_alpha
        self.fit_sessions: set[int] = set()
        self.calibration_sessions: set[int] = set()
        self.residual_quantiles = np.zeros(5)
        self.calibration_offsets = (0.0, 0.0)
        self.fitted = False

    def fit(self, examples: list[dict], *, allowed_sessions: set[int]) -> Baseline:
        validate_scope(examples, allowed_sessions, role="fitting")
        rows = valid_examples(examples)
        if len(rows) < 2:
            raise ValueError("Fitting requires at least two eligible examples")
        self.fit_sessions = set(allowed_sessions)
        target = np.log([row["target_seconds"] / row["reference_seconds"] for row in rows])
        if self.kind == "regularized_linear_residual":
            matrix = feature_matrix(rows)
            self.location = np.nanmedian(np.where(np.isnan(matrix).all(axis=0), 0, matrix), axis=0)
            clean = np.where(np.isnan(matrix), self.location, matrix)
            self.scale = clean.std(axis=0)
            self.scale[self.scale < 1e-8] = 1.0
            design = np.column_stack([np.ones(len(rows)), (clean - self.location) / self.scale])
            penalty = np.eye(design.shape[1]) * self.ridge_alpha
            penalty[0, 0] = 0
            self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        elif self.kind == "catboost_quantile":
            from catboost import CatBoostRegressor

            self.trees = []
            for quantile in QUANTILES:
                model = CatBoostRegressor(
                    iterations=100,
                    depth=4,
                    loss_function=f"Quantile:alpha={quantile}",
                    random_seed=17,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                )
                model.fit(feature_matrix(rows), target)
                self.trees.append(model)
        self.fitted = True
        if self.kind != "catboost_quantile":
            residuals = target - self._point_log(rows)
            self.residual_quantiles = np.quantile(residuals, QUANTILES)
        return self

    def _point_log(self, examples: list[dict]) -> np.ndarray:
        if self.kind == "regularized_linear_residual":
            matrix = feature_matrix(examples)
            clean = np.where(np.isnan(matrix), self.location, matrix)
            return (
                np.column_stack([np.ones(len(examples)), (clean - self.location) / self.scale])
                @ self.weights
            )
        if self.kind == "same_sector_persistence":
            return np.array(
                [
                    math.log(
                        (
                            row.get("features", {}).get("last_same_sector_seconds")
                            or row["reference_seconds"]
                        )
                        / row["reference_seconds"]
                    )
                    for row in examples
                ]
            )
        return np.zeros(len(examples))

    def _raw(self, examples: list[dict]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Predictor is not fitted")
        if any(not row.get("eligible", False) for row in examples):
            raise ValueError("Prediction expects eligible examples; emit abstentions separately")
        reference = np.array([row["reference_seconds"] for row in examples])
        if self.kind == "catboost_quantile":
            log_quantiles = np.column_stack(
                [tree.predict(feature_matrix(examples)) for tree in self.trees]
            )
        else:
            log_quantiles = self._point_log(examples)[:, None] + self.residual_quantiles
        return np.sort(np.exp(np.clip(log_quantiles, -8, 8)) * reference[:, None], axis=1)

    def calibrate(self, examples: list[dict], *, allowed_sessions: set[int]) -> Baseline:
        if self.fit_sessions.intersection(allowed_sessions):
            raise ValueError("Calibration and fitting sessions must be disjoint")
        validate_scope(examples, allowed_sessions, role="calibration")
        rows = valid_examples(examples)
        if len(rows) < 10:
            raise ValueError("Calibration requires at least ten valid examples")
        q = self._raw(rows)
        target = np.array([row["target_seconds"] for row in rows])
        offsets = []
        for alpha, lo, hi in ((0.2, 1, 3), (0.1, 0, 4)):
            scores = np.maximum(q[:, lo] - target, target - q[:, hi])
            rank = min(len(rows), math.ceil((len(rows) + 1) * (1 - alpha)))
            offsets.append(max(0.0, float(np.sort(scores)[rank - 1])))
        self.calibration_offsets = tuple(offsets)
        self.calibration_sessions = set(allowed_sessions)
        return self

    def predict(self, examples: list[dict]) -> list[list[float]]:
        if not examples:
            return []
        q = self._raw(examples)
        q[:, 1] -= self.calibration_offsets[0]
        q[:, 3] += self.calibration_offsets[0]
        q[:, 0] -= self.calibration_offsets[1]
        q[:, 4] += self.calibration_offsets[1]
        q[:, 0] = np.minimum(q[:, 0], q[:, 1])
        q[:, 4] = np.maximum(q[:, 4], q[:, 3])
        return np.maximum(q, 0.001).tolist()


def train_baselines(
    fit_examples: list[dict],
    calibration_examples: list[dict],
    *,
    fit_sessions: set[int],
    calibration_sessions: set[int],
    include_catboost: bool = False,
) -> dict[str, Baseline]:
    kinds = ["same_sector_persistence", "historical_reference", "regularized_linear_residual"]
    if include_catboost:
        kinds.append("catboost_quantile")
    models = {}
    for kind in kinds:
        models[kind] = (
            Baseline(kind)
            .fit(fit_examples, allowed_sessions=fit_sessions)
            .calibrate(calibration_examples, allowed_sessions=calibration_sessions)
        )
    return models
