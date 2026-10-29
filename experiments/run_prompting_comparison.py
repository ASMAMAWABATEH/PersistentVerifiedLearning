"""Compare zero-shot and verified-example prompting on development tasks."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from verityloop.benchmark.dataset import build_benchmark
from verityloop.benchmark.runner import evaluate_candidate
from verityloop.benchmark.tasks import get_initial_tasks
from verityloop.evaluation.evaluate import evaluate_candidates
from verityloop.evaluation.splits import split_by_family
from verityloop.recursive_learning.curate import curate_candidates
from verityloop.recursive_learning.ollama_generate import generate_ollama_candidates
from verityloop.recursive_learning.verify import verify_candidates
from verityloop.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def assess(candidates, dev_tasks, dev_records, all_tasks):
    verification = verify_candidates(candidates, dev_tasks)
    curation = curate_candidates(candidates, verification)
    metrics = evaluate_candidates(candidates, dev_records, all_tasks)
    return {
        "verification": verification,
        "curation": curation,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    if args.limit < 1 or args.examples < 1:
        raise ValueError("--limit and --examples must both be positive")

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

    train_records = splits["train"]
    dev_records = splits["dev"][:args.limit]
    if len(dev_records) < args.limit:
        print(f"Note: only {len(dev_records)} development tasks are available.")

    train_examples = [
        task_lookup[record["task_id"]]
        for record in train_records[:args.examples]
    ]
    dev_tasks = [task_lookup[record["task_id"]] for record in dev_records]

    # Validate every reference example before placing it in the few-shot prompt.
    for example in train_examples:
        result = evaluate_candidate(example, example.reference_code)
        if not result.passed:
            raise RuntimeError(
                f"Training example failed verification: {example.task_id}"
            )

    print(f"Model: {args.model}")
    print(f"Development tasks per condition: {len(dev_tasks)}")
    print(f"Verified training examples per few-shot prompt: {len(train_examples)}")
    print("\n--- Zero-shot generation ---")
    zero_candidates = generate_ollama_candidates(
        dev_tasks, model=args.model
    )
    print("\n--- Verified-example generation ---")
    few_candidates = generate_ollama_candidates(
        dev_tasks, model=args.model, example_tasks=train_examples
    )

    zero = assess(zero_candidates, dev_tasks, dev_records, tasks)
    few = assess(few_candidates, dev_tasks, dev_records, tasks)

    zero_rate = zero["metrics"].task_pass_rate
    few_rate = few["metrics"].task_pass_rate
    gain = few_rate - zero_rate
    minimum_gain = config["recursive_learning"]["minimum_pass_rate_gain"]

    report = {
        "experiment": "zero_shot_vs_verified_example_prompting",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "development_task_ids": [r["task_id"] for r in dev_records],
        "verified_training_example_ids": [t.task_id for t in train_examples],
        "zero_shot": {
            "candidate_count": len(zero_candidates),
            "verified_count": sum(r.passed for r in zero["verification"]),
            "curated_count": len(zero["curation"].accepted),
            "development_pass_rate": zero_rate,
        },
        "verified_example_prompting": {
            "candidate_count": len(few_candidates),
            "verified_count": sum(r.passed for r in few["verification"]),
            "curated_count": len(few["curation"].accepted),
            "development_pass_rate": few_rate,
        },
        "comparison": {
            "pass_rate_gain": gain,
            "minimum_gain_required": minimum_gain,
            "improvement_threshold_met": gain >= minimum_gain,
        },
        "model_generation_performed": True,
        "model_training_performed": False,
        "model_updated": False,
        "final_test_evaluated": False,
        "limitations": [
            "This is prompting-based inference, not parameter training.",
            "Reference solutions are supplied only from the training split.",
            "Results apply to this development subset and this model.",
            "The final test split remains untouched.",
        ],
    }

    output = ROOT / "results" / "logs" / "prompting_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n===== PROMPTING COMPARISON =====")
    print(f"Zero-shot: {zero_rate:.3f} "
          f"({report['zero_shot']['verified_count']}/{len(zero_candidates)} verified)")
    print(f"Verified-example: {few_rate:.3f} "
          f"({report['verified_example_prompting']['verified_count']}/{len(few_candidates)} verified)")
    print(f"Pass-rate gain: {gain:+.3f}")
    print(f"Required gain: {minimum_gain:.3f}")
    print(f"Improvement threshold met: {gain >= minimum_gain}")
    print("Model training performed: False")
    print("Model updated: False")
    print("Final test evaluated: False")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
