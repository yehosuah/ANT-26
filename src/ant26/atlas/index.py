"""Train-only reference indices. No index update occurs during replay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MEXICO_MEETING = 1298
MEXICO_SESSIONS = frozenset((11397, 11398, 11399, 11400, 11404))
MEXICO_WEEKEND_START = datetime(2026, 10, 30, tzinfo=timezone.utc).timestamp()
SUMMARY_KEYS = (
    "reference_seconds",
    "last_same_sector_seconds",
    "same_sector_history_count",
    "telemetry_coverage",
    "tyre_age_laps",
    "air_temperature",
    "track_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "gap_to_leader",
    "interval",
) + tuple(
    f"{channel}_{stat}"
    for channel in ("speed", "throttle", "brake", "rpm", "n_gear")
    for stat in ("mean", "std", "min", "max", "last")
)


def _timestamp(value: Any) -> float:
    if not isinstance(value, str):
        raise ValueError("timezone-aware timestamp required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid atlas timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return parsed.timestamp()


def _hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _save_immutable(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x") as output:
            output.write(serialized)
    except FileExistsError:
        if path.read_text() != serialized:
            raise ValueError("refusing to overwrite different existing atlas artifact") from None


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, (str, dict, list)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _validate_scope(examples: list[dict], allowed_sessions: set) -> None:
    seen = set()
    for row in examples:
        identifier = row.get("example_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("nonempty example_id required")
        if identifier in seen:
            raise ValueError(f"duplicate atlas source id: {identifier}")
        seen.add(identifier)
        if row.get("meeting_key") == MEXICO_MEETING or row.get("session_key") in MEXICO_SESSIONS:
            raise ValueError("Mexico examples cannot enter the reference index")
        if row.get("session_key") not in allowed_sessions:
            raise ValueError(
                f"reference session outside fitting allowlist: {row.get('session_key')}"
            )
        origin = _timestamp(row.get("origin"))
        cutoff = _timestamp(row.get("cutoff"))
        if origin >= MEXICO_WEEKEND_START:
            raise ValueError("Mexico-weekend and later examples cannot enter the reference index")
        if cutoff > origin:
            raise ValueError("reference cutoff cannot follow origin")
        if row.get("driver_number") is None or row.get("lap_number") is None:
            raise ValueError("reference driver and lap identity required")


def _metadata(row: dict) -> dict:
    features = row.get("features") or {}
    summary = {
        key: number
        for key in SUMMARY_KEYS
        if (number := _finite_number(features.get(key))) is not None
    }
    # Load already-canonical metadata without reintroducing target fields.
    if "observed_summary" in row:
        summary = {
            key: number
            for key in SUMMARY_KEYS
            if (number := _finite_number(row["observed_summary"].get(key))) is not None
        }
    result = {
        key: copy.deepcopy(row.get(key))
        for key in (
            "example_id",
            "session_key",
            "meeting_key",
            "driver_number",
            "lap_number",
            "sector_index",
            "origin",
            "cutoff",
        )
    }
    result["observed_summary"] = summary
    result["context"] = copy.deepcopy(row.get("context", features.get("context", {})))
    result["provenance"] = copy.deepcopy(row.get("provenance") or {})
    # Target end is label metadata and is not necessary for historical retrieval.
    result["provenance"].pop("target_end", None)
    return result


def _matrix(vectors: Any, rows: int | None = None) -> np.ndarray:
    try:
        values = np.array(vectors, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("finite rectangular vector matrix required") from exc
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("nonempty two-dimensional vector matrix required")
    if rows is not None and values.shape[0] != rows:
        raise ValueError("vector count differs from reference count")
    if not np.isfinite(values).all():
        raise ValueError("all atlas vectors must be finite")
    return values


class AtlasIndex:
    """Reference index over externally supplied learned or engineered vectors.

    Distances describe similarity under this representation; they are not a
    calibrated probability of shared driver intention or true driving style.
    """

    def __init__(self, *, metric: str = "euclidean", window_seconds: float = 32):
        if metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be euclidean or cosine")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("positive finite window_seconds required")
        self.metric = metric
        self.window_seconds = float(window_seconds)
        self._references: list[dict] = []
        self._vectors: np.ndarray | None = None
        self._allowed_sessions: list = []
        self.index_hash: str | None = None

    @property
    def references(self) -> list[dict]:
        return copy.deepcopy(self._references)

    @property
    def dimension(self) -> int:
        self._require_fit()
        return self._vectors.shape[1]

    def _require_fit(self) -> None:
        if self._vectors is None:
            raise ValueError("atlas reference index has not been fitted")

    def fit(
        self, examples: list[dict], vectors: Any, *, allowed_sessions: list | set
    ) -> AtlasIndex:
        allowed = set(allowed_sessions)
        _validate_scope(examples, allowed)
        values = _matrix(vectors, len(examples))
        if self.metric == "cosine" and np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError("cosine distance requires nonzero reference vectors")
        self._references = [_metadata(row) for row in examples]
        self._vectors = values
        self._vectors.setflags(write=False)
        self._allowed_sessions = sorted(allowed)
        self.index_hash = _hash(self._payload())
        return self

    def _payload(self) -> dict:
        self._require_fit()
        return {
            "schema_version": "ant26-atlas-v1",
            "metric": self.metric,
            "window_seconds": self.window_seconds,
            "allowed_sessions": self._allowed_sessions,
            "references": self._references,
            "vectors": self._vectors.tolist(),
        }

    def query(self, example: dict, vector: Any, k: int = 5, *, retrospective: bool = False) -> dict:
        self._require_fit()
        try:
            vector = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("finite query vector required") from exc
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError("query vector dimension differs from index")
        if not np.isfinite(vector).all():
            raise ValueError("query vector must be finite")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100:
            raise ValueError("k must be an integer between 1 and 100")
        if not isinstance(retrospective, bool):
            raise ValueError("retrospective must be explicit boolean")
        cutoff = _timestamp(example.get("cutoff"))
        candidates = []
        for index, ref in enumerate(self._references):
            if ref["example_id"] == example.get("example_id"):
                continue
            same_driver_session = ref["session_key"] == example.get("session_key") and ref[
                "driver_number"
            ] == example.get("driver_number")
            if same_driver_session and ref["lap_number"] == example.get("lap_number"):
                continue
            ref_cutoff = _timestamp(ref["cutoff"])
            if same_driver_session and abs(cutoff - ref_cutoff) < self.window_seconds:
                continue
            if not retrospective and _timestamp(ref["origin"]) >= cutoff:
                continue
            candidates.append(index)
        if self.metric == "cosine":
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("cosine distance requires a nonzero query vector")
            norms = np.linalg.norm(self._vectors, axis=1)
            distances = 1 - self._vectors @ vector / (norms * norm)
            distances = np.clip(distances, 0, 2)
        else:
            distances = np.linalg.norm(self._vectors - vector, axis=1)
        candidates.sort(key=lambda i: (float(distances[i]), self._references[i]["example_id"]))
        return {
            "index_hash": self.index_hash,
            "retrospective": retrospective,
            "query_id": example.get("example_id"),
            "distance_metric": self.metric,
            "neighbors": [
                {**copy.deepcopy(self._references[i]), "distance": float(distances[i])}
                for i in candidates[:k]
            ],
        }

    def save(self, path: str | Path) -> None:
        _save_immutable(path, {**self._payload(), "index_hash": self.index_hash})

    @classmethod
    def _from_payload(cls, payload: dict) -> AtlasIndex:
        payload = copy.deepcopy(payload)
        expected = payload.pop("index_hash", None)
        if payload.get("schema_version") != "ant26-atlas-v1":
            raise ValueError("unsupported atlas schema")
        if not expected or _hash(payload) != expected:
            raise ValueError("atlas index hash mismatch")
        result = cls(metric=payload["metric"], window_seconds=payload["window_seconds"])
        result.fit(
            payload["references"], payload["vectors"], allowed_sessions=payload["allowed_sessions"]
        )
        if result.index_hash != expected:
            raise ValueError("atlas canonical source metadata hash mismatch")
        return result

    @classmethod
    def load(cls, path: str | Path) -> AtlasIndex:
        return cls._from_payload(json.loads(Path(path).read_text()))


class FeatureAtlasIndex:
    """Training-only imputation, standardization and PCA retrieval baseline."""

    def __init__(self, *, components: int = 8):
        if not isinstance(components, int) or components < 1:
            raise ValueError("positive PCA component count required")
        self.components = components
        self.index: AtlasIndex | None = None
        self._median = None
        self._mean = None
        self._scale = None
        self._projection = None

    @property
    def index_hash(self) -> str | None:
        if self.index is None:
            return None
        return _hash(self._payload())

    @staticmethod
    def _features(examples: list[dict]) -> np.ndarray:
        return np.array(
            [
                [
                    value
                    if (value := _finite_number(row.get("features", {}).get(key))) is not None
                    else np.nan
                    for key in SUMMARY_KEYS
                ]
                for row in examples
            ],
            dtype=np.float64,
        )

    def fit(self, examples: list[dict], *, allowed_sessions: list | set) -> FeatureAtlasIndex:
        _validate_scope(examples, set(allowed_sessions))
        if not examples:
            raise ValueError("nonempty reference examples required")
        values = self._features(examples)
        self._median = np.array(
            [
                float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0
                for column in values.T
            ]
        )
        values = np.where(np.isnan(values), self._median, values)
        self._mean = values.mean(axis=0)
        self._scale = values.std(axis=0)
        self._scale[self._scale < 1e-12] = 1
        scaled = (values - self._mean) / self._scale
        if len(examples) >= 2:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(self.components, *scaled.shape), svd_solver="full")
            pca.fit(scaled)
            self._projection = pca.components_
        else:
            self._projection = np.eye(scaled.shape[1])[: self.components]
        vectors = scaled @ self._projection.T
        self.index = AtlasIndex().fit(examples, vectors, allowed_sessions=allowed_sessions)
        return self

    def _transform(self, examples: list[dict]) -> np.ndarray:
        if self.index is None:
            raise ValueError("feature index has not been fitted")
        values = self._features(examples)
        filled = np.where(np.isnan(values), self._median, values)
        return ((filled - self._mean) / self._scale) @ self._projection.T

    def query(self, example: dict, k: int = 5, *, retrospective: bool = False) -> dict:
        vector = self._transform([example])[0]
        result = self.index.query(example, vector, k, retrospective=retrospective)
        result["index_hash"] = self.index_hash
        result["representation"] = "train_only_standardized_features_pca"
        return result

    def _payload(self) -> dict:
        if self.index is None:
            raise ValueError("feature index has not been fitted")
        return {
            "schema_version": "ant26-feature-atlas-v1",
            "components": self.components,
            "feature_keys": list(SUMMARY_KEYS),
            "median": self._median.tolist(),
            "mean": self._mean.tolist(),
            "scale": self._scale.tolist(),
            "projection": self._projection.tolist(),
            "index": {**self.index._payload(), "index_hash": self.index.index_hash},
        }

    def save(self, path: str | Path) -> None:
        _save_immutable(path, {**self._payload(), "index_hash": self.index_hash})

    @classmethod
    def load(cls, path: str | Path) -> FeatureAtlasIndex:
        payload = json.loads(Path(path).read_text())
        expected = payload.pop("index_hash", None)
        if payload.get("schema_version") != "ant26-feature-atlas-v1" or _hash(payload) != expected:
            raise ValueError("feature atlas schema or hash mismatch")
        if payload["feature_keys"] != list(SUMMARY_KEYS):
            raise ValueError("feature atlas feature order mismatch")
        result = cls(components=payload["components"])
        result.index = AtlasIndex._from_payload(payload["index"])
        for name in ("median", "mean", "scale", "projection"):
            setattr(result, f"_{name}", np.asarray(payload[name], dtype=np.float64))
        dimension = len(SUMMARY_KEYS)
        if any(
            getattr(result, f"_{name}").shape != (dimension,)
            for name in ("median", "mean", "scale")
        ):
            raise ValueError("feature atlas preprocessing dimension mismatch")
        if result._projection.shape != (result.index.dimension, dimension):
            raise ValueError("feature atlas projection dimension mismatch")
        if not all(
            np.isfinite(getattr(result, f"_{name}")).all()
            for name in ("median", "mean", "scale", "projection")
        ) or np.any(result._scale <= 0):
            raise ValueError("invalid feature atlas preprocessing values")
        return result


def fit_pattern_clusters(
    vectors: Any,
    *,
    examples: list[dict] | None = None,
    allowed_sessions: list | set | None = None,
    seed: int = 17,
) -> dict:
    """Fit optional descriptive GMM patterns without claiming validated states.

    Metadata is mandatory before any fit. Small datasets return an explicit
    insufficient-samples status rather than inventing stable behavior labels.
    """
    values = _matrix(vectors)
    if len(values) < 12:
        return {
            "status": "insufficient_samples",
            "validated": False,
            "sample_count": len(values),
            "labels": [],
        }
    if examples is None or allowed_sessions is None:
        raise ValueError("examples and fitting-session allowlist required before cluster fitting")
    if len(examples) != len(values):
        raise ValueError("cluster vector and example counts differ")
    _validate_scope(examples, set(allowed_sessions))
    from sklearn.mixture import GaussianMixture

    models = []
    for count in range(3, min(8, len(values) // 4) + 1):
        model = GaussianMixture(n_components=count, random_state=seed, n_init=3, reg_covar=1e-5)
        model.fit(values)
        models.append((float(model.bic(values)), count, model))
    score, count, model = min(models, key=lambda entry: (entry[0], entry[1]))
    labels = model.predict(values)
    summaries = []
    for cluster in range(count):
        cluster_rows = [row for row, label in zip(examples, labels) if label == cluster]
        observed = {}
        for key in SUMMARY_KEYS:
            feature_values = [
                value
                for row in cluster_rows
                if (value := _finite_number(row.get("features", {}).get(key))) is not None
            ]
            if feature_values:
                observed[key] = float(np.mean(feature_values))
        summaries.append(
            {"cluster_id": cluster, "sample_count": len(cluster_rows), "observed_means": observed}
        )
    return {
        "status": "descriptive_only_pending_stability_and_review",
        "validated": False,
        "sample_count": len(values),
        "component_count": count,
        "bic": score,
        "candidate_bic": {str(k): bic for bic, k, _ in models},
        "labels": labels.tolist(),
        "patterns": summaries,
        "claim": "inferred_telemetry_patterns_not_driver_intention",
    }
