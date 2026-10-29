"""Persistent verified-memory learning with bounded rounds and rollback."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from verityloop.benchmark.dataset import build_benchmark
from verityloop.benchmark.tasks import get_initial_tasks
from verityloop.evaluation.evaluate import evaluate_candidates
from verityloop.evaluation.splits import split_by_family
from verityloop.recursive_learning.acceptance import decide_acceptance
from verityloop.recursive_learning.curate import curate_candidates
from verityloop.recursive_learning.generate import Candidate
from verityloop.recursive_learning.ollama_generate import generate_ollama_candidates
from verityloop.recursive_learning.verify import verify_candidates
from verityloop.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = ROOT / "results" / "memory"
ACTIVE = MEMORY_DIR / "verified_examples.json"
PROPOSED = MEMORY_DIR / "proposed_examples.json"
BACKUP = MEMORY_DIR / "verified_examples.previous.json"
REPORT_PATH = ROOT / "results" / "logs" / "persistent_learning.json"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_verified_memory(train_tasks):
    """Load and independently re-verify training-only active memory."""
    if not ACTIVE.exists():
        return []

    lookup = {task.task_id: task for task in train_tasks}
    raw = json.loads(ACTIVE.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Active memory must contain a JSON list.")

    verified = []
    seen = set()
    for entry in raw:
        task_id = entry.get("task_id")
        task = lookup.get(task_id)
        code = entry.get("code", "")
        if task is None or not code or task_id in seen:
            print(f"Skipping invalid/non-training memory entry: {task_id}")
            continue

        candidate = Candidate(
            task_id=task_id,
            code=code,
            round_number=entry.get("round_number", 0),
            source="persistent_verified_memory",
        )
        result = verify_candidates([candidate], [task])[0]
        if result.passed and result.total_tests > 0:
            verified.append((task, code))
            seen.add(task_id)
        else:
            print(f"Rejected stale/invalid memory entry: {task_id}")

    return verified


def memory_records(memory, round_number, source):
    """Serialize only verified entries from the supplied training memory."""
    return [
        {
            "task_id": task.task_id,
            "family": task.family,
            "prompt": task.prompt,
            "function_name": task.function_name,
            "code": code,
            "round_number": round_number,
            "source": source,
        }
        for task, code in memory
    ]


def merge_memory(active_memory, accepted_candidates, lookup):
    """Replace entries only with independently verified candidates."""
    merged = {task.task_id: (task, code) for task, code in active_memory}
    for candidate in accepted_candidates:
        task = lookup[candidate.task_id]
        merged[candidate.task_id] = (task, candidate.code)
    return list(merged.values())


def generate_training_round(train_tasks, active_memory, round_number,
                            resume_saved_proposal=False):
    """Recover a saved proposal when possible and regenerate missing entries."""
    task_ids = {task.task_id for task in train_tasks}
    candidates_by_id = {}

    if resume_saved_proposal and PROPOSED.exists():
        print("Checking saved proposal; invalid or missing tasks will be regenerated.")
        saved = json.loads(PROPOSED.read_text(encoding="utf-8"))
        if not isinstance(saved, list):
            raise ValueError("Saved proposal must contain a JSON list.")

        task_lookup = {task.task_id: task for task in train_tasks}
        for entry in saved:
            task_id = entry.get("task_id")
            task = task_lookup.get(task_id)
            code = entry.get("code", "")
            if task is None or not code or task_id in candidates_by_id:
                continue

            candidate = Candidate(
                task_id=task_id,
                code=code,
                round_number=round_number,
                source="resumed_proposal",
            )
            result = verify_candidates([candidate], [task])[0]
            if result.passed and result.total_tests > 0:
                candidates_by_id[task_id] = candidate
            else:
                print(f"Saved candidate needs regeneration: {task_id}")

    missing_tasks = [
        task for task in train_tasks if task.task_id not in candidates_by_id
    ]
    if missing_tasks:
        print(f"Generating candidates for {len(missing_tasks)} missing training task(s).")
        generated = generate_ollama_candidates(
            missing_tasks,
            example_solutions=active_memory[:4],
        )
        for candidate in generated:
            candidates_by_id[candidate.task_id] = Candidate(
                task_id=candidate.task_id,
                code=candidate.code,
                round_number=round_number,
                source=candidate.source,
            )

    candidates = [
        candidates_by_id[task.task_id]
        for task in train_tasks
        if task.task_id in candidates_by_id
    ]
    reports = verify_candidates(candidates, train_tasks)
    curation = curate_candidates(candidates, reports)
    return candidates, reports, curation


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

    train_tasks = [lookup[r["task_id"]] for r in splits["train"]]
    dev_records = splits["dev"]
    dev_tasks = [lookup[r["task_id"]] for r in dev_records]
    train_ids = {task.task_id for task in train_tasks}
    dev_ids = {task.task_id for task in dev_tasks}

    if train_ids & dev_ids:
        raise ValueError("Training and development task IDs overlap.")
    if train_ids & {r["task_id"] for r in splits["test"]}:
        raise ValueError("Training and final-test task IDs overlap.")
    if dev_ids & {r["task_id"] for r in splits["test"]}:
        raise ValueError("Development and final-test task IDs overlap.")

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    active_memory = load_verified_memory(train_tasks)
    initial_active_count = len(active_memory)
    max_rounds = int(config["recursive_learning"]["max_rounds"])
    if max_rounds < 1:
        raise ValueError("recursive_learning.max_rounds must be at least 1.")

    print(f"Active verified training-memory entries: {len(active_memory)}")
    print(f"Training tasks available: {len(train_tasks)}")
    print(f"Development tasks: {len(dev_tasks)}")
    print(f"Configured maximum rounds: {max_rounds}")
    print("Final test set: reserved and untouched")

    round_history = []
    stop_reason = "Maximum configured rounds reached."
    resume_saved_proposal = True

    for round_number in range(1, max_rounds + 1):
        print(f"\n========== LEARNING ROUND {round_number}/{max_rounds} ==========")
        print(f"Active memory before round: {len(active_memory)}")

        train_candidates, train_reports, train_curation = generate_training_round(
            train_tasks,
            active_memory,
            round_number,
            resume_saved_proposal=resume_saved_proposal,
        )
        resume_saved_proposal = False

        # Keep all valid active entries. Add only candidates that passed every test.
        proposed_memory = merge_memory(
            active_memory, train_curation.accepted, lookup
        )
        proposal_data = memory_records(
            proposed_memory,
            round_number,
            "verified_training_example_memory",
        )
        save_json(PROPOSED, proposal_data)

        print(f"Training candidates: {len(train_candidates)}/{len(train_tasks)}")
        print(f"Candidates passing all tests: {len(train_curation.accepted)}")
        print(f"Candidates rejected by curation: {len(train_curation.rejected)}")
        print(f"Proposed verified memory entries: {len(proposed_memory)}")

        print("\n--- Evaluate active memory on development tasks ---")
        current_candidates = generate_ollama_candidates(
            dev_tasks,
            example_solutions=active_memory[:4],
        )
        current_results = verify_candidates(current_candidates, dev_tasks)
        current_metrics = evaluate_candidates(current_candidates, dev_records, tasks)

        print("\n--- Evaluate proposed memory on the same development tasks ---")
        proposed_candidates = generate_ollama_candidates(
            dev_tasks,
            example_solutions=proposed_memory[:4],
        )
        proposed_results = verify_candidates(proposed_candidates, dev_tasks)
        proposed_curation = curate_candidates(proposed_candidates, proposed_results)
        proposed_metrics = evaluate_candidates(proposed_candidates, dev_records, tasks)

        decision = decide_acceptance(
            baseline_dev_score=current_metrics.task_pass_rate,
            candidate_dev_score=proposed_metrics.task_pass_rate,
            minimum_gain=config["recursive_learning"]["minimum_pass_rate_gain"],
            require_no_regression=config["recursive_learning"]["require_no_dev_regression"],
        )

        round_record = {
            "round": round_number,
            "active_memory_entries_before": len(active_memory),
            "training_candidates": len(train_candidates),
            "training_tasks_expected": len(train_tasks),
            "training_candidates_passing_all_tests": len(train_curation.accepted),
            "training_candidates_rejected": len(train_curation.rejected),
            "proposed_memory_entries": len(proposed_memory),
            "development": {
                "active_memory_pass_rate": current_metrics.task_pass_rate,
                "proposed_memory_pass_rate": proposed_metrics.task_pass_rate,
                "proposed_candidates_passing_all_tests": len(proposed_curation.accepted),
            },
            "acceptance": {
                "accepted": decision.accepted,
                "reason": decision.reason,
                "improvement": decision.improvement,
            },
            "memory_update_committed": False,
            "rollback_performed": False,
        }

        if decision.accepted:
            if ACTIVE.exists():
                shutil.copy2(ACTIVE, BACKUP)
            save_json(ACTIVE, proposal_data)
            round_record["memory_update_committed"] = True
            active_memory = proposed_memory
            print("Accepted: verified training memory committed.")
        else:
            round_record["rollback_performed"] = True
            print(f"Rejected: {decision.reason}")
            print("Rollback: active memory preserved; rejected proposal retained for audit.")

        round_history.append(round_record)

        if not decision.accepted:
            stop_reason = (
                f"Stopped after round {round_number}: development acceptance gate rejected the proposal."
            )
            break

    last_round = round_history[-1] if round_history else {}
    report = {
        "experiment": "persistent_verified_memory_learning",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configured_max_rounds": max_rounds,
        "rounds_executed": len(round_history),
        "stop_reason": stop_reason,
        "training_task_count": len(train_tasks),
        "development_task_ids": [task.task_id for task in dev_tasks],
        "training_task_ids": [task.task_id for task in train_tasks],
        "active_memory_entries_before_experiment": initial_active_count,
        "active_memory_entries_after_experiment": len(active_memory),
        "round_history": round_history,
        "final_round_development": last_round.get("development"),
        "final_round_acceptance": last_round.get("acceptance"),
        "artifacts": {
            "active_memory": str(ACTIVE.relative_to(ROOT)),
            "proposed_memory": str(PROPOSED.relative_to(ROOT)),
            "backup_memory": str(BACKUP.relative_to(ROOT)) if BACKUP.exists() else None,
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
        "model_generation_performed": True,
        "persistent_system_memory_updated": any(
            item["memory_update_committed"] for item in round_history
        ),
        "model_training_performed": False,
        "model_updated": False,
        "final_test_evaluated": False,
        "limitations": [
            "Persistent verified-example memory changes prompting context, not model weights.",
            "Each round is accepted only when development pass rate improves by the configured minimum gain.",
            "A rejected proposal is not activated; the learning trajectory stops at rejection.",
            "Only independently verified training-split solutions can enter persistent memory.",
            "The final test split is not evaluated by this runner.",
        ],
    }
    save_json(REPORT_PATH, report)

    print("\n===== PERSISTENT LEARNING RESULTS =====")
    print(f"Rounds executed: {report['rounds_executed']}/{max_rounds}")
    print(f"Active memory entries: {initial_active_count} -> {len(active_memory)}")
    if last_round.get("development"):
        print(
            "Active-memory dev pass rate: "
            f"{last_round['development']['active_memory_pass_rate']:.3f}"
        )
        print(
            "Proposed-memory dev pass rate: "
            f"{last_round['development']['proposed_memory_pass_rate']:.3f}"
        )
    if last_round.get("acceptance"):
        print(
            f"Last-round acceptance: {last_round['acceptance']['accepted']} — "
            f"{last_round['acceptance']['reason']}"
        )
    print(f"Stop reason: {stop_reason}")
    print(f"Persistent memory updated: {report['persistent_system_memory_updated']}")
    print("Model training performed: False")
    print("Model updated: False")
    print("Final test evaluated: False")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
