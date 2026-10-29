
"""Build and evaluate the initial VerityLoop coding benchmark."""

import json
from pathlib import Path
from typing import Any

from verityloop.benchmark.runner import evaluate_candidate
from verityloop.benchmark.tasks import CodingTask, get_initial_tasks


def build_benchmark() -> list[dict[str, Any]]:
    """Create benchmark records from the predefined coding tasks."""
    records = []

    for task in get_initial_tasks():
        records.append({
            "task_id": task.task_id,
            "family": task.family,
            "prompt": task.prompt,
            "function_name": task.function_name,
            "reference_code": task.reference_code,
            "test_cases": [
                {"inputs": list(case.inputs), "expected": case.expected}
                for case in task.test_cases
            ],
        })

    return records


def evaluate_reference_solutions() -> list[dict[str, Any]]:
    """Check that every reference implementation passes its own tests."""
    results = []

    for task in get_initial_tasks():
        result = evaluate_candidate(task, task.reference_code)
        results.append({
            "task_id": task.task_id,
            "passed": result.passed,
            "passed_tests": result.passed_tests,
            "total_tests": result.total_tests,
            "pass_rate": result.pass_rate,
        })

    return results


def save_benchmark(path: str | Path) -> None:
    """Write benchmark records to JSON Lines format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in build_benchmark():
            file.write(json.dumps(record) + "\n")


def validate_reference_solutions() -> None:
    """Raise an error if any reference implementation fails."""
    results = evaluate_reference_solutions()
    failed = [result for result in results if not result["passed"]]

    if failed:
        details = ", ".join(item["task_id"] for item in failed)
        raise AssertionError(f"Reference solutions failed: {details}")
