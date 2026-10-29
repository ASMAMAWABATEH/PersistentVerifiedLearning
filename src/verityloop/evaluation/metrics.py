
"""Metrics for benchmark candidate evaluation."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate results from evaluating benchmark tasks."""

    task_count: int
    passed_tasks: int
    passed_tests: int
    total_tests: int
    task_pass_rate: float
    test_pass_rate: float


def calculate_metrics(
    results: list[Any],
) -> EvaluationMetrics:
    """Calculate aggregate task and test pass rates.

    Each result must expose:
    task_id, passed, passed_tests, and total_tests.
    """
    if not results:
        raise ValueError("Cannot calculate metrics for an empty result set.")

    task_ids = [result.task_id for result in results]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task results are not allowed.")

    for result in results:
        if result.total_tests <= 0:
            raise ValueError(
                f"Task {result.task_id} has no executed tests."
            )
        if not 0 <= result.passed_tests <= result.total_tests:
            raise ValueError(
                f"Invalid test counts for task {result.task_id}."
            )
        if result.passed != (
            result.passed_tests == result.total_tests
        ):
            raise ValueError(
                f"Inconsistent pass status for task {result.task_id}."
            )

    task_count = len(results)
    passed_tasks = sum(result.passed for result in results)
    passed_tests = sum(result.passed_tests for result in results)
    total_tests = sum(result.total_tests for result in results)

    return EvaluationMetrics(
        task_count=task_count,
        passed_tasks=passed_tasks,
        passed_tests=passed_tests,
        total_tests=total_tests,
        task_pass_rate=passed_tasks / task_count,
        test_pass_rate=passed_tests / total_tests,
    )
