"""Evaluate locally generated Ollama code on a small development-only pilot."""

import argparse
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
from verityloop.recursive_learning.ollama_generate import generate_ollama_candidates
from verityloop.recursive_learning.verify import verify_candidates
from verityloop.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2, help="Number of development tasks")
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    config = load_config(ROOT / "configs" / "experiment.yaml")
    tasks = get_initial_tasks()
    task_lookup = {task.task_id: task for task in tasks}
    records = build_benchmark()
    splits = split_by_family(
        records,
        train_fraction=config["benchmark"]["train_fraction"],
        dev_fraction=config["benchmark"]["dev_fraction"],
        test_fraction=config["benchmark"]["test_fraction"],
        seed=config["project"]["seed"],
    )

    # Only use a small subset of development tasks. Never read the final test split.
    dev_records = splits["dev"][:args.limit]
    dev_tasks = [task_lookup[record["task_id"]] for record in dev_records]

    print(f"Model: {args.model}")
    print(f"Development tasks selected: {len(dev_tasks)}")
    candidates = generate_ollama_candidates(dev_tasks, model=args.model)

    verification = verify_candidates(candidates, dev_tasks)
    curation = curate_candidates(candidates, verification)
    baseline = generate_baseline_candidates(records)

    baseline_metrics = evaluate_candidates(baseline, dev_records, tasks)
    candidate_metrics = evaluate_candidates(candidates, dev_records, tasks)

    acceptance = decide_acceptance(
        baseline_dev_score=baseline_metrics.task_pass_rate,
        candidate_dev_score=candidate_metrics.task_pass_rate,
        minimum_gain=config["recursive_learning"]["minimum_pass_rate_gain"],
        require_no_regression=config["recursive_learning"]["require_no_dev_regression"],
    )

    report = {
        "experiment": "ollama_generation_pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "benchmark": {
            "total_task_count": len(records),
            "development_tasks_evaluated": [r["task_id"] for r in dev_records],
            "final_test_evaluated": False,
        },
        "verification": {
            "candidate_count": len(candidates),
            "verified_candidate_count": sum(r.passed for r in verification),
            "curated_candidate_count": len(curation.accepted),
        },
        "development_evaluation": {
            "baseline_task_pass_rate": baseline_metrics.task_pass_rate,
            "ollama_task_pass_rate": candidate_metrics.task_pass_rate,
        },
        "acceptance": {
            "accepted": acceptance.accepted,
            "reason": acceptance.reason,
        },
        "model_generation_performed": True,
        "model_training_performed": False,
        "model_updated": False,
        "final_test_evaluated": False,
        "limitations": [
            "This pilot uses local model inference, not model training.",
            "No model parameters are updated.",
            "Only a subset of development tasks is evaluated.",
            "The final test split remains untouched.",
        ],
    }

    output = ROOT / "results" / "logs" / "ollama_pilot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n===== OLLAMA PILOT RESULTS =====")
    print(f"Verified candidates: {report['verification']['verified_candidate_count']}/{len(candidates)}")
    print(f"Curated candidates: {len(curation.accepted)}")
    print(f"Baseline dev pass rate: {baseline_metrics.task_pass_rate:.3f}")
    print(f"Ollama dev pass rate: {candidate_metrics.task_pass_rate:.3f}")
    print(f"Acceptance: {acceptance.accepted} — {acceptance.reason}")
    print("Model training performed: False")
    print("Model updated: False")
    print("Final test evaluated: False")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
