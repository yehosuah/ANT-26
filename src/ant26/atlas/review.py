"""Prepare blinded retrieval comparisons; preferences require human responses."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from ant26.atlas.index import AtlasIndex, FeatureAtlasIndex, _save_immutable
from ant26.domain.protocol import protocol_hash
from ant26.evaluation.splits import is_mexico, rolling_splits, validate_scope
from ant26.models.baselines import valid_examples

REVIEW_NORMALIZATION_POLICY = "constant_unit_scale_v2"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _load_bundle(path):
    from ant26.models.representation import RepresentationBundle

    return RepresentationBundle.load(path)


def _encode(bundle, examples: list[dict], batch_size: int = 64) -> np.ndarray:
    return np.concatenate(
        [
            bundle.encode(examples[start : start + batch_size])
            for start in range(0, len(examples), batch_size)
        ],
        axis=0,
    )


def _observed_window(row: dict) -> dict:
    features = row.get("features", {})
    return {
        "example_id": row["example_id"],
        "session_key": row["session_key"],
        "driver_number": row["driver_number"],
        "lap_number": row["lap_number"],
        "sector_index": row["sector_index"],
        "origin": row["origin"],
        "cutoff": row["cutoff"],
        "sequence_channels": features.get("sequence_channels", []),
        "sequence": features.get("sequence", []),
        "sequence_mask": features.get("sequence_mask", []),
        "context": features.get("context", {}),
    }


def build_review_set(
    examples: list[dict],
    catalog: dict,
    fold_bundles: list[dict],
    output_dir: str | Path,
    *,
    protocol: dict,
    query_count: int = 60,
    minimum_races: int = 6,
    neighbor_count: int = 3,
    seed: int = 17,
    driver_number: int = 12,
) -> dict:
    """Write blinded A/B neighbors and a separate private assignment key.

    Each fold specification declares fold_id, bundle_path, fit_sessions and
    test_sessions. Fitting scopes must equal the protocol's fold-specific refit
    policy; a bundle's exact normalized input-row checksum must also match.
    No model is trained, modified, calibrated or selected by this preparation.
    """
    if any(is_mexico(row) for row in examples):
        raise ValueError("Mexico cannot enter development review preparation")
    if not isinstance(query_count, int) or query_count < minimum_races or minimum_races < 1:
        raise ValueError("query_count must cover at least minimum_races")
    if not isinstance(neighbor_count, int) or neighbor_count < 1:
        raise ValueError("positive neighbor_count required")
    folds = {fold["fold_id"]: fold for fold in rolling_splits(catalog, protocol)}
    specifications = {spec["fold_id"]: spec for spec in fold_bundles}
    if len(specifications) != len(fold_bundles):
        raise ValueError("duplicate review fold identifiers")
    if set(specifications) - set(folds):
        raise ValueError("review fold is absent from the declared chronological protocol")
    if len(specifications) < minimum_races:
        raise ValueError("insufficient chronological held-out races for review")
    selected = [fold for key, fold in folds.items() if key in specifications]
    if len(selected) > query_count:
        raise ValueError("query_count must permit at least one query per declared fold")
    rng = random.Random(seed)
    per_race, extra = divmod(query_count, len(selected))
    blind_queries, assignments, provenance = [], [], []
    data_kinds, reference_drivers = set(), set()
    for fold_index, fold in enumerate(selected):
        spec = specifications[fold["fold_id"]]
        expected_fit = set(fold["fit_sessions"])
        if protocol["splits"].get("refit_fit_plus_tuning_before_calibration", True):
            expected_fit.update(fold["tune_sessions"])
        if set(spec["fit_sessions"]) != expected_fit or set(spec["test_sessions"]) != set(
            fold["test_sessions"]
        ):
            raise ValueError("explicit review scope differs from its chronological fold")
        references = valid_examples([row for row in examples if row["session_key"] in expected_fit])
        queries = valid_examples(
            [
                row
                for row in examples
                if row["session_key"] in fold["test_sessions"]
                and row["driver_number"] == driver_number
            ]
        )
        validate_scope(references, expected_fit, role="atlas review fitting")
        validate_scope(queries, set(fold["test_sessions"]), role="atlas review held-out queries")
        needed = per_race + int(fold_index < extra)
        if len(queries) < needed or not references:
            raise ValueError("fold lacks the fixed number of valid review queries or references")
        bundle = _load_bundle(spec["bundle_path"])
        metadata = bundle.metadata
        if metadata.get("normalization_policy") != REVIEW_NORMALIZATION_POLICY:
            raise ValueError("legacy normalization checkpoint cannot enter the human review set")
        data_kinds.add(metadata.get("data_kind", "unspecified"))
        reference_drivers.update(row["driver_number"] for row in references)
        if set(metadata["fit_sessions"]) != expected_fit:
            raise ValueError("checkpoint fitting scope differs from review references")
        if set(metadata.get("calibration_sessions", ())) != set(fold["calibration_sessions"]):
            raise ValueError("checkpoint calibration scope differs from its review fold")
        if set(metadata.get("tuning_sessions", ())) & set(fold["test_sessions"]):
            raise ValueError("checkpoint tuning overlaps held-out review queries")
        if metadata.get("protocol_sha256") != protocol_hash(protocol):
            raise ValueError("checkpoint and review protocol hashes differ")
        if metadata.get("fitting_examples_sha256") != _digest(references):
            raise ValueError("checkpoint fitting snapshot differs from exact review references")
        if metadata.get("seed") != seed:
            raise ValueError("review must use the fixed declared checkpoint seed")
        ordered = sorted(queries, key=lambda row: row["example_id"])
        query_rows = rng.sample(ordered, needed)
        learned = AtlasIndex().fit(
            references, _encode(bundle, references), allowed_sessions=expected_fit
        )
        feature = FeatureAtlasIndex().fit(references, allowed_sessions=expected_fit)
        vectors = _encode(bundle, query_rows)
        reference_map = {row["example_id"]: row for row in references}
        for query, vector in zip(query_rows, vectors, strict=True):
            outputs = {
                "learned": learned.query(query, vector, neighbor_count),
                "feature_pca": feature.query(query, neighbor_count),
            }
            if any(len(result["neighbors"]) != neighbor_count for result in outputs.values()):
                raise ValueError("review query lacks the fixed number of legal neighbors")
            methods = ["learned", "feature_pca"]
            rng.shuffle(methods)
            query_id = f"query-{len(blind_queries) + 1:03d}"
            candidate_sets = {}
            for label, method in zip(("A", "B"), methods, strict=True):
                # Distance scales and index identities can unblind the method.
                # Reviewers see observations and rank, never method or distance.
                candidate_sets[label] = [
                    {
                        "rank": rank,
                        "observed": _observed_window(reference_map[item["example_id"]]),
                        "observed_summary": item["observed_summary"],
                    }
                    for rank, item in enumerate(outputs[method]["neighbors"], 1)
                ]
            blind_queries.append(
                {
                    "query_id": query_id,
                    "query": _observed_window(query),
                    "candidates": candidate_sets,
                }
            )
            assignments.append(
                {
                    "query_id": query_id,
                    "fold_id": fold["fold_id"],
                    "source_example_id": query["example_id"],
                    "A": methods[0],
                    "B": methods[1],
                }
            )
        provenance.append(
            {
                "fold_id": fold["fold_id"],
                "fit_sessions": sorted(expected_fit),
                "test_sessions": sorted(fold["test_sessions"]),
                "fitting_examples_sha256": metadata["fitting_examples_sha256"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "normalization_policy": metadata["normalization_policy"],
                "learned_index_hash": learned.index_hash,
                "feature_index_hash": feature.index_hash,
                "query_count": needed,
            }
        )
    race_count = len({row["query"]["session_key"] for row in blind_queries})
    if race_count < minimum_races:
        raise ValueError("prepared review does not cover the required distinct races")
    private = {
        "schema_version": "ant26-atlas-review-key-v1",
        "review_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "protocol_sha256": protocol_hash(protocol),
        "input_examples_sha256": _digest(examples),
        "assignments": assignments,
        "fold_provenance": provenance,
    }
    blind = {
        "schema_version": "ant26-blind-atlas-review-v1",
        "data_kinds": sorted(data_kinds),
        "query_driver_number": driver_number,
        "reference_driver_numbers": sorted(reference_drivers),
        "status": "awaiting_human_review",
        "query_count": len(blind_queries),
        "race_count": race_count,
        "retrospective": False,
        "instructions": "Compare A and B for similarity in observed telemetry execution, accounting for context. Record A, B, tie, or neither. Do not infer intention. No preference is prefilled.",
        "queries": blind_queries,
        "answer_key_sha256": _digest(private),
    }
    review_id = _digest(blind)
    blind["review_id"] = review_id
    private["review_id"] = review_id
    response_template = {
        "review_id": review_id,
        "reviewer_id": None,
        "responses": [
            {"query_id": row["query_id"], "choice": None, "notes": ""} for row in blind_queries
        ],
    }
    directory = Path(output_dir)
    paths = {
        "blind_review": directory / "blind-review.json",
        "response_template": directory / "response-template.json",
        "private_answer_key": directory / "private" / "answer-key.json",
    }
    for key, payload in (
        ("blind_review", blind),
        ("response_template", response_template),
        ("private_answer_key", private),
    ):
        _save_immutable(paths[key], payload)
    return {
        "review_id": review_id,
        "status": "awaiting_human_review",
        "query_count": len(blind_queries),
        "race_count": race_count,
        "paths": {key: str(path) for key, path in paths.items()},
        "human_responses": 0,
    }


def score_review_responses(blind: dict, answer_key: dict, response_sets: list[dict]) -> dict:
    """Score only explicitly supplied human responses, preserving ties and gaps."""
    review_id = blind.get("review_id")
    if _digest({key: value for key, value in blind.items() if key != "review_id"}) != review_id:
        raise ValueError("blind review content differs from its checksum")
    if review_id != answer_key.get("review_id"):
        raise ValueError("review and assignment key identities differ")
    if _digest(
        {key: value for key, value in answer_key.items() if key != "review_id"}
    ) != blind.get("answer_key_sha256"):
        raise ValueError("assignment key content differs from its checksum")
    assignments = {row["query_id"]: row for row in answer_key["assignments"]}
    counts = {"learned": 0, "feature_pca": 0, "tie": 0, "neither": 0, "uncertain": 0}
    reviewers, answered_queries = set(), set()
    for response_set in response_sets:
        reviewer = response_set.get("reviewer_id")
        if not reviewer or reviewer in reviewers or response_set.get("review_id") != review_id:
            raise ValueError("distinct reviewer identity and matching review required")
        reviewers.add(reviewer)
        seen = set()
        for response in response_set.get("responses", []):
            query_id, choice = response["query_id"], response.get("choice")
            if query_id not in assignments or query_id in seen:
                raise ValueError("unknown or duplicate response query")
            seen.add(query_id)
            if choice is None:
                continue
            if choice not in ("A", "B", "tie", "neither", "uncertain"):
                raise ValueError("human choice must be A, B, tie, uncertain or neither")
            counts[assignments[query_id][choice] if choice in ("A", "B") else choice] += 1
            answered_queries.add(query_id)
    total = sum(counts.values())
    return {
        "review_id": review_id,
        "status": "awaiting_human_review"
        if not total
        else "exploratory_single_reviewer"
        if len(reviewers) < 2
        else "human_responses_recorded",
        "reviewer_count": len(reviewers),
        "distinct_answered_queries": len(answered_queries),
        "response_count": total,
        "counts": counts,
        "learned_preference_fraction_including_ties": counts["learned"] / total if total else None,
        "validated": False,
        "note": "No human preference or validation is inferred from model outputs. Repeated reviewer responses are not independent query observations.",
    }


def bootstrap_cluster_stability(
    reference_examples: list[dict],
    reference_vectors,
    probe_examples: list[dict],
    probe_vectors,
    *,
    fit_sessions: set[int],
    probe_sessions: set[int],
    replicates: int = 20,
    seed: int = 17,
) -> dict:
    """Bounded race-bootstrap ARI check, separate from human retrieval review.

    A train-only PCA of at most eight components bounds diagonal GMM cost. The
    component count is selected by fitting-data BIC, then held fixed across race
    resamples. No held-out labels select cluster hyperparameters.
    """
    if fit_sessions & probe_sessions:
        raise ValueError("stability probe races must be disjoint from fitting races")
    validate_scope(reference_examples, fit_sessions, role="cluster stability fitting")
    validate_scope(probe_examples, probe_sessions, role="cluster stability probe")
    reference = np.asarray(reference_vectors, dtype=float)
    probe = np.asarray(probe_vectors, dtype=float)
    if (
        reference.ndim != 2
        or probe.ndim != 2
        or reference.shape[0] != len(reference_examples)
        or probe.shape[0] != len(probe_examples)
        or reference.shape[1] != probe.shape[1]
        or reference.shape[1] == 0
        or not np.isfinite(reference).all()
        or not np.isfinite(probe).all()
    ):
        raise ValueError("finite matching embedding matrices and example counts required")
    if not isinstance(replicates, int) or not 1 <= replicates <= 100:
        raise ValueError("replicates must be between 1 and 100")
    if len(reference_examples) < 24 or len(probe_examples) < 10 or len(fit_sessions) < 6:
        return {"status": "insufficient_samples", "validated": False, "replicates_completed": 0}
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    projection = PCA(n_components=min(8, reference.shape[1], len(reference) - 1), svd_solver="full")
    fitting = projection.fit_transform(reference)
    probing = projection.transform(probe)
    candidates = []
    for count in range(3, min(8, len(reference) // 8) + 1):
        model = GaussianMixture(
            n_components=count, covariance_type="diag", random_state=seed, reg_covar=1e-4, n_init=2
        )
        model.fit(fitting)
        candidates.append((float(model.bic(fitting)), count, model))
    _, components, baseline = min(candidates, key=lambda result: (result[0], result[1]))
    baseline_labels = baseline.predict(probing)
    if len(set(baseline_labels)) < 2:
        return {
            "status": "degenerate_probe_no_state_claim",
            "validated": False,
            "replicates_completed": 0,
            "component_count": components,
        }
    groups = {
        session: [i for i, row in enumerate(reference_examples) if row["session_key"] == session]
        for session in sorted(fit_sessions)
    }
    if any(not indices for indices in groups.values()):
        raise ValueError("every declared bootstrap fitting race must contain reference windows")
    rng, scores, failures = np.random.default_rng(seed), [], []
    for replicate in range(replicates):
        sampled = rng.choice(sorted(groups), size=len(groups), replace=True)
        indices = [i for session in sampled for i in groups[int(session)]]
        try:
            model = GaussianMixture(
                n_components=components,
                covariance_type="diag",
                random_state=seed + replicate + 1,
                reg_covar=1e-4,
                n_init=2,
            )
            model.fit(fitting[indices])
            if not model.converged_:
                failures.append({"replicate": replicate, "reason": "not_converged"})
                continue
            scores.append(float(adjusted_rand_score(baseline_labels, model.predict(probing))))
        except ValueError:
            failures.append({"replicate": replicate, "reason": "fit_failed"})
    median = float(np.median(scores)) if scores else None
    target_met = (
        len(scores) == replicates and replicates >= 20 and median is not None and median >= 0.6
    )
    return {
        "status": "stability_target_met_pending_human_review"
        if target_met
        else "unstable_or_incomplete_drop_discrete_state_claims",
        "validated": False,
        "seed": seed,
        "fit_sessions": sorted(fit_sessions),
        "probe_sessions": sorted(probe_sessions),
        "reference_examples_sha256": _digest(reference_examples),
        "probe_examples_sha256": _digest(probe_examples),
        "reference_embedding_sha256": _digest(reference.tolist()),
        "probe_embedding_sha256": _digest(probe.tolist()),
        "projection_components": projection.n_components_,
        "covariance_type": "diag",
        "component_count": components,
        "replicates_requested": replicates,
        "replicates_completed": len(scores),
        "median_adjusted_rand_index": median,
        "adjusted_rand_indices": scores,
        "failures": failures,
        "human_reviews_completed": 0,
    }
