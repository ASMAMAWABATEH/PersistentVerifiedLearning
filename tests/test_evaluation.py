
"""Tests for candidate evaluation and development-set metrics."""

import pytest

from verityloop.benchmark.tasks import get_initial_tasks
from verityloop.evaluation.evaluate import evaluate_candidates
from verityloop.evaluation.splits import split_by_family
from verityloop.recursive_learning.generate import (
    Candidate,
    generate_baseline_candidates,
)


def make_records(tasks):
    return [
        {
            "task_id": task.task_id,
            "family": task.family,
            "prompt": task.prompt,
            "function_name": task.function_name,
            "reference_code": task.reference_code,
            "test_cases": [
                {"inputs": list(case.inputs), "expected": case.expected}
                for case in task.test_cases
            ],
        }
        for task in tasks
    ]


def test_reference_candidates_pass_dev_split():
    tasks = get_initial_tasks()
    records = make_records(tasks)
    splits = split_by_family(records, seed=42)
    candidates = generate_baseline_candidates(records)

    metrics = evaluate_candidates(candidates, splits["dev"], tasks)

    assert metrics.task_count == len(splits["dev"])
    assert metrics.task_pass_rate == 1.0
    assert metrics.test_pass_rate == 1.0


def test_missing_candidate_is_rejected():
    tasks = get_initial_tasks()
    records = make_records(tasks)
    splits = split_by_family(records, seed=42)
    candidates = generate_baseline_candidates(records)
    missing_task_id = splits["dev"][0]["task_id"]
    candidates = [
        candidate
        for candidate in candidates
        if candidate.task_id != missing_task_id
    ]

    with pytest.raises(ValueError, match="No candidate provided"):
        evaluate_candidates(candidates, splits["dev"], tasks)


def test_empty_split_is_rejected():
    tasks = get_initial_tasks()
    candidates = generate_baseline_candidates(make_records(tasks))

    with pytest.raises(ValueError, match="empty split"):
        evaluate_candidates(candidates, [], tasks)


def test_duplicate_candidates_are_rejected():
    tasks = get_initial_tasks()
    records = make_records(tasks)
    candidates = generate_baseline_candidates(records)

    with pytest.raises(ValueError, match="Duplicate candidate"):
        evaluate_candidates(
            [candidates[0], candidates[0]],
            records[:2],
            tasks,
        )
