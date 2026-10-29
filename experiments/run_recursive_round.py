"""One recursive round using independently verified model-generated experience."""

import json
from datetime import datetime, timezone
from pathlib import Path

from verityloop.benchmark.dataset import build_benchmark
from verityloop.benchmark.tasks import get_initial_tasks
from verityloop.evaluation.evaluate import evaluate_candidates
from verityloop.evaluation.splits import split_by_family
from verityloop.recursive_learning.curate import curate_candidates
from verityloop.recursive_learning.generate import generate_baseline_candidates
from verityloop.recursive_learning.ollama_generate import generate_ollama_candidates
from verityloop.recursive_learning.verify import verify_candidates
from verityloop.recursive_learning.acceptance import decide_acceptance
from verityloop.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config(ROOT / "configs" / "experiment.yaml")
    tasks = get_initial_tasks()
    lookup = {task.task_id: task for task in tasks}
    records = build_benchmark()
    splits = split_by_family(
        records,
        train_fraction=config["benchmark"]["train_fraction"],
        dev_fraction=config["benchmark"]["dev_fraction"],
        test_fraction=config["benchmark"]["test_fraction"],
        seed=config["project"]["seed"],
    )

    train_tasks = [lookup[r["task_id"]] for r in splits["train"][:4]]
    dev_records = splits["dev"]
    dev_tasks = [lookup[r["task_id"]] for r in dev_records]

    print("Round 1: generating candidates on training tasks...")
    generated_train = generate_ollama_candidates(train_tasks)
    train_verification = verify_candidates(generated_train, train_tasks)
    train_curation = curate_candidates(generated_train, train_verification)
    accepted = list(train_curation.accepted)

    if not accepted:
        raise RuntimeError("No model-generated training candidates passed verification.")

    accepted_by_id = {c.task_id: c for c in accepted}
    verified_experience = [
        (lookup[c.task_id], c.code)
        for c in accepted
    ]

    print(f"Verified training experience: {len(verified_experience)}/{len(train_tasks)}")
    print("Round 2: solving development tasks using verified model-generated examples...")
    recursive_candidates = generate_ollama_candidates(
        dev_tasks,
        example_solutions=verified_experience,
    )

    recursive_verification = verify_candidates(recursive_candidates, dev_tasks)
    recursive_curation = curate_candidates(recursive_candidates, recursive_verification)

    baseline = generate_baseline_candidates(records)
    baseline_metrics = evaluate_candidates(baseline, dev_records, tasks)
    recursive_metrics = evaluate_candidates(recursive_candidates, dev_records, tasks)

    decision = decide_acceptance(
        baseline_dev_score=baseline_metrics.task_pass_rate,
        candidate_dev_score=recursive_metrics.task_pass_rate,
        minimum_gain=config["recursive_learning"]["minimum_pass_rate_gain"],
        require_no_regression=config["recursive_learning"]["require_no_dev_regression"],
    )

    report = {
        "experiment": "verified_experience_recursive_round",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_task_ids": [t.task_id for t in train_tasks],
        "development_task_ids": [r["task_id"] for r in dev_records],
        "verified_experience_count": len(verified_experience),
        "development": {
            "baseline_pass_rate": baseline_metrics.task_pass_rate,
            "recursive_pass_rate": recursive_metrics.task_pass_rate,
            "verified_candidate_count": sum(r.passed for r in recursive_verification),
            "curated_candidate_count": len(recursive_curation.accepted),
        },
        "acceptance": {
            "accepted": decision.accepted,
            "reason": decision.reason,
        },
        "model_generation_performed": True,
        "model_training_performed": False,
        "model_updated": False,
        "final_test_evaluated": False,
        "limitations": [
            "Recursive learning here means learning from verified examples in context, not weight updates.",
            "Only four training tasks supply generated examples in this pilot.",
            "The final test split remains untouched.",
        ],
    }

    output = ROOT / "results" / "logs" / "recursive_round.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n===== RECURSIVE ROUND RESULTS =====")
    print(f"Verified experience: {len(verified_experience)}/{len(train_tasks)}")
    print(f"Development candidates verified: {report['development']['verified_candidate_count']}/{len(dev_tasks)}")
    print(f"Baseline dev pass rate: {baseline_metrics.task_pass_rate:.3f}")
    print(f"Recursive dev pass rate: {recursive_metrics.task_pass_rate:.3f}")
    print(f"Acceptance: {decision.accepted} — {decision.reason}")
    print("Model training performed: False")
    print("Model updated: False")
    print("Final test evaluated: False")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
