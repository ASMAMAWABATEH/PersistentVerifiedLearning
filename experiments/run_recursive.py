"""Run the VerityLoop recursive-learning pipeline smoke test."""

import json
from datetime import datetime, timezone
from pathlib import Path

from verityloop.benchmark.dataset import build_benchmark
from verityloop.benchmark.tasks import get_initial_tasks
from verityloop.evaluation.evaluate import evaluate_candidates
from verityloop.evaluation.splits import split_by_family
from verityloop.recursive_learning.acceptance import decide_acceptance
from verityloop.recursive_learning.curate import curate_candidates
from verityloop.recursive_learning.generate import generate_baseline_candidates
from verityloop.recursive_learning.verify import verify_candidates
from verityloop.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Validate the pipeline without claiming model training occurred."""
    config = load_config(ROOT / "configs" / "experiment.yaml")
    seed = config["project"]["seed"]
    benchmark_config = config["benchmark"]
    recursive_config = config["recursive_learning"]

    tasks = get_initial_tasks()
    records = build_benchmark()
    splits = split_by_family(
        records,
        train_fraction=benchmark_config["train_fraction"],
        dev_fraction=benchmark_config["dev_fraction"],
        test_fraction=benchmark_config["test_fraction"],
        seed=seed,
    )

    candidates = generate_baseline_candidates(records)
    verification_results = verify_candidates(candidates, tasks)
    curation = curate_candidates(candidates, verification_results)
    curated_candidates = list(curation.accepted)

    if not curated_candidates:
        raise RuntimeError("No candidates passed verification.")

    dev_metrics = evaluate_candidates(
        curated_candidates,
        splits["dev"],
        tasks,
    )

    baseline_metrics = evaluate_candidates(
        candidates,
        splits["dev"],
        tasks,
    )

    acceptance = decide_acceptance(
        baseline_dev_score=baseline_metrics.task_pass_rate,
        candidate_dev_score=dev_metrics.task_pass_rate,
        minimum_gain=recursive_config["minimum_pass_rate_gain"],
        require_no_regression=recursive_config["require_no_dev_regression"],
    )

    report = {
        "experiment": "recursive_learning_smoke_test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "benchmark": {
            "task_count": len(records),
            "train_task_count": len(splits["train"]),
            "dev_task_count": len(splits["dev"]),
            "test_task_count": len(splits["test"]),
            "split_unit": "task family",
        },
        "verification": {
            "candidate_count": len(candidates),
            "verified_candidate_count": sum(
                result.passed
                for result in verification_results
            ),
            "curated_candidate_count": len(curated_candidates),
        },
        
        "development_evaluation": {
            "baseline_task_pass_rate": baseline_metrics.task_pass_rate,
            "candidate_task_pass_rate": dev_metrics.task_pass_rate,
        },

        "acceptance": {
            "evaluated": True,
            "accepted": acceptance.accepted,
            "reason": acceptance.reason,
        },
        "model_training_performed": False,
        "model_updated": False,
        "final_test_evaluated": False,
        "limitations": [
            "Candidates use reference solutions; no model generates code.",
            "No model training or parameter update occurs.",
            "The final test split is deliberately not evaluated.",
            "This run validates pipeline behavior, not learning improvement.",
        ],
    }

    output_dir = ROOT / "results" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "recursive_smoke_test.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("VerityLoop recursive pipeline smoke test completed.")
    print(f"Tasks: {len(records)}")
    print(
        "Verified candidates: "
        f"{report['verification']['verified_candidate_count']}/"
        f"{report['verification']['candidate_count']}"
    )
    print(f"Curated candidates: {len(curated_candidates)}")
    print(
        "Baseline dev task pass rate: "
        f"{baseline_metrics.task_pass_rate:.3f}"
    )
    print(
        "Candidate dev task pass rate: "
        f"{dev_metrics.task_pass_rate:.3f}"
    )
    print(f"Acceptance decision: {acceptance.accepted}")
    print(f"Acceptance reason: {acceptance.reason}")
    print("Model training performed: False")
    print(f"Report: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
