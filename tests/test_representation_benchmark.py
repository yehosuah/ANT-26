"""The real-data runner uses synthetic fixtures only for implementation checks."""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ant26.demo import synthetic_examples  # noqa: E402
from ant26.evaluation import run_benchmark, run_representation_benchmark  # noqa: E402
from ant26.evaluation.representation_benchmark import _seed_summary  # noqa: E402
from ant26.models.representation import EncoderConfig  # noqa: E402


@pytest.fixture
def protocol():
    return json.loads(
        (Path(__file__).parents[1] / "configs" / "evaluation-protocol.json").read_text()
    )


def test_runner_uses_scoped_calibration_and_reuses_immutable_results(tmp_path, protocol):
    examples, catalog = synthetic_examples(protocol)
    baseline = run_benchmark(examples, catalog, protocol)
    report = run_representation_benchmark(
        examples,
        catalog,
        protocol,
        tmp_path,
        seeds=[17],
        max_folds=1,
        pretrain_epochs=1,
        forecast_epochs=1,
        encoder_config=EncoderConfig(width=16, layers=1, heads=2),
        baseline_report=baseline,
    )
    assert report["status"] == "exploratory"
    assert report["synthetic"] is True
    assert report["evaluated_fold_count"] == 1
    assert report["seeds"] == [17]
    assert not report["seed_budget_complete"]
    assert not report["claims"]["finetune"]["improvement_claim_allowed"]
    run = report["folds"][0]["runs"][0]
    assert not set(run["fit_sessions"]).intersection(run["calibration_sessions"])
    assert run["encoder_parameters"] < 2_000_000
    assert len(run["checkpoint_sha256"]) == 64
    assert Path(run["checkpoint_dir"], "model.pt").is_file()
    assert report["comparisons"]["finetune"]["status"] == "paired"
    assert report["comparisons"]["finetune"]["metrics"]["mae_seconds"]["race_count"] == 1
    again = run_representation_benchmark(
        examples,
        catalog,
        protocol,
        tmp_path,
        seeds=[17],
        max_folds=1,
        pretrain_epochs=1,
        forecast_epochs=1,
        encoder_config=EncoderConfig(width=16, layers=1, heads=2),
        baseline_report=baseline,
    )
    assert again == report


def test_default_seed_budget_recorded_without_fabricating_missing_folds(tmp_path, protocol):
    examples, catalog = synthetic_examples(protocol, race_count=3)
    report = run_representation_benchmark(examples, catalog, protocol, tmp_path)
    assert report["status"] == "insufficient_data"
    assert report["seeds"] == protocol["experiments"]["seeds"]
    assert report["evaluated_fold_count"] == 0
    assert report["predictions"] == []
    assert not report["claims"]["finetune"]["improvement_claim_allowed"]


@pytest.mark.parametrize(
    "session,meeting,origin",
    [(11404, 1298, "2026-01-01T00:00:00+00:00"), (9999, 1999, "2026-11-08T00:00:00+00:00")],
)
def test_final_case_and_future_rows_rejected_before_training(
    tmp_path, protocol, monkeypatch, session, meeting, origin
):
    import ant26.evaluation.representation_benchmark as runner

    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden data reached model training")

    monkeypatch.setattr(runner, "train_representation", forbidden)
    row = {
        "example_id": "forbidden",
        "session_key": session,
        "meeting_key": meeting,
        "origin": origin,
        "cutoff": origin,
    }
    with pytest.raises(ValueError):
        run_representation_benchmark([row], {"development": []}, protocol, tmp_path)


def test_censored_target_does_not_enter_evaluation(tmp_path, protocol):
    examples, catalog = synthetic_examples(protocol)
    target = next(
        row
        for row in examples
        if row["session_key"] == 1010 and row["eligible"] and row["target_valid"]
    )
    identifier = target["example_id"]
    target["target_valid"] = False
    target["target_seconds"] = 999999
    report = run_representation_benchmark(
        examples,
        catalog,
        protocol,
        tmp_path,
        seeds=[17],
        max_folds=1,
        pretrain_epochs=0,
        forecast_epochs=1,
        encoder_config=EncoderConfig(width=16, layers=1, heads=2),
    )
    assert report["evaluated_fold_count"] == 1
    assert identifier not in {row["example_id"] for row in report["predictions"]}


def test_baseline_dataset_mismatch_rejected_before_training(tmp_path, protocol):
    examples, catalog = synthetic_examples(protocol, race_count=3)
    with pytest.raises(ValueError, match="identical input hashes"):
        run_representation_benchmark(
            examples, catalog, protocol, tmp_path, baseline_report={"input_sha256": "different"}
        )


def test_seed_summary_keeps_race_as_the_independent_unit():
    rows = []
    for seed, error in [(17, 0), (29, 2)]:
        rows.append(
            {
                "session_key": 1,
                "driver_number": 12,
                "target_seconds": 10,
                "reference_seconds": 10,
                "quantiles": [8 + error, 9 + error, 10 + error, 11 + error, 12 + error],
                "seed": seed,
            }
        )
    report = _seed_summary(copy.deepcopy(rows), 12)
    assert report["seed_count"] == 2
    assert report["race_count"] == 1
    assert report["macro"]["mae_seconds"] == 1
    assert report["forecast_count_per_seed"] == 1
