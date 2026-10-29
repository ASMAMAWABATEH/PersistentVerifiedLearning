
"""Evaluate candidate solutions on a specified benchmark split."""

from typing import Any

from verityloop.benchmark.runner import evaluate_candidate
from verityloop.benchmark.tasks import CodingTask, TestCase
from verityloop.evaluation.metrics import EvaluationMetrics, calculate_metrics
from verityloop.recursive_learning.generate import Candidate


def evaluate_candidates(
    candidates: list[Candidate],
    split_records: list[dict[str, Any]],
    task_catalog: list[CodingTask],
    timeout_seconds: float = 2.0,
) -> EvaluationMetrics:
    """Evaluate one candidate per task in a split and aggregate metrics."""
    if not split_records:
        raise ValueError("Cannot evaluate an empty split.")

    catalog = {task.task_id: task for task in task_catalog}
    if len(catalog) != len(task_catalog):
        raise ValueError("Duplicate task IDs in task catalog.")

    records_by_id = {record["task_id"]: record for record in split_records}
    if len(records_by_id) != len(split_records):
        raise ValueError("Duplicate task IDs in split records.")

    candidates_by_id = {candidate.task_id: candidate for candidate in candidates}
    candidate_ids = [candidate.task_id for candidate in candidates]
    if len(candidates_by_id) != len(candidate_ids):
        raise ValueError("Duplicate candidate task IDs.")

    missing_ids = set(records_by_id) - set(candidates_by_id)
    if missing_ids:
        raise ValueError(
            f"No candidate provided for task(s): {sorted(missing_ids)}"
        )

    results = []
    for task_id, record in records_by_id.items():
        if task_id not in catalog:
            raise ValueError(f"Task {task_id!r} is missing from the task catalog.")

        original_task = catalog[task_id]
        if record["family"] != original_task.family:
            raise ValueError(f"Task family mismatch for {task_id!r}.")

        split_task = CodingTask(
            task_id=task_id,
            family=record["family"],
            prompt=record["prompt"],
            function_name=record["function_name"],
            reference_code=record["reference_code"],
            test_cases=tuple(
                TestCase(
                    inputs=tuple(case["inputs"]),
                    expected=case["expected"],
                )
                for case in record["test_cases"]
            ),
        )

        result = evaluate_candidate(
            split_task,
            candidates_by_id[task_id].code,
            timeout_seconds=timeout_seconds,
        )
        results.append(result)

    return calculate_metrics(results)
