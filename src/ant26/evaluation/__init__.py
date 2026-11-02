"""Chronological evaluation with an explicit final-case-study boundary."""

from ant26.evaluation.metrics import macro_metrics, paired_race_bootstrap, weighted_interval_score
from ant26.evaluation.splits import rolling_splits


def run_benchmark(*args, **kwargs):
    from ant26.evaluation.benchmark import run_benchmark as run

    return run(*args, **kwargs)


def run_representation_benchmark(*args, **kwargs):
    from ant26.evaluation.representation_benchmark import run_representation_benchmark as run

    return run(*args, **kwargs)


__all__ = [
    "macro_metrics",
    "paired_race_bootstrap",
    "rolling_splits",
    "run_benchmark",
    "run_representation_benchmark",
    "weighted_interval_score",
]
