
"""Automated tests for benchmark task definitions and execution."""

import pytest

from verityloop.benchmark.dataset import (
    build_benchmark,
    validate_reference_solutions,
)
from verityloop.benchmark.runner import evaluate_candidate
from verityloop.benchmark.tasks import get_initial_tasks


def test_initial_benchmark_has_unique_task_ids():
    tasks = get_initial_tasks()
    task_ids = [task.task_id for task in tasks]

    assert len(tasks) == 40
    assert len(task_ids) == len(set(task_ids))


def test_every_task_has_test_cases():
    for task in get_initial_tasks():
        assert task.test_cases
        assert task.prompt
        assert task.reference_code


def test_all_reference_solutions_pass():
    validate_reference_solutions()


def test_correct_candidate_passes():
    task = get_initial_tasks()[0]
    code = "def absolute_value(x):\n    return abs(x)\n"

    result = evaluate_candidate(task, code)

    assert result.passed
    assert result.passed_tests == result.total_tests == 4


def test_incorrect_candidate_fails():
    task = get_initial_tasks()[0]
    code = "def absolute_value(x):\n    return x\n"

    result = evaluate_candidate(task, code)

    assert not result.passed
    assert result.passed_tests < result.total_tests


def test_benchmark_records_have_required_fields():
    records = build_benchmark()
    required = {
        "task_id",
        "family",
        "prompt",
        "function_name",
        "reference_code",
        "test_cases",
    }

    assert records
    for record in records:
        assert required.issubset(record)
