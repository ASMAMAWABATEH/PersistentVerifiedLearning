"""Independent verification of generated candidates."""

from dataclasses import dataclass

from verityloop.benchmark.runner import evaluate_candidate
from verityloop.benchmark.tasks import CodingTask, get_initial_tasks
from verityloop.recursive_learning.generate import Candidate


@dataclass(frozen=True)
class VerificationReport:
    """Benchmark evidence for one candidate."""

    task_id: str
    passed: bool
    passed_tests: int
    total_tests: int
    pass_rate: float
    errors: tuple[str, ...] = ()


def verify_candidate(
    candidate: Candidate,
    task_lookup: dict[str, CodingTask],
    timeout_seconds: float = 2.0,
) -> VerificationReport:
    """Evaluate a candidate against its task's predefined test cases."""
    task = task_lookup.get(candidate.task_id)
    if task is None:
        raise ValueError(f"Unknown task ID: {candidate.task_id}")

    result = evaluate_candidate(
        task,
        candidate.code,
        timeout_seconds=timeout_seconds,
    )
    errors = tuple(
        test.error for test in result.test_results if test.error is not None
    )

    return VerificationReport(
        task_id=result.task_id,
        passed=result.passed,
        passed_tests=result.passed_tests,
        total_tests=result.total_tests,
        pass_rate=result.pass_rate,
        errors=errors,
    )


def verify_candidates(
    candidates: list[Candidate],
    tasks: list[CodingTask] | None = None,
    timeout_seconds: float = 2.0,
) -> list[VerificationReport]:
    """Verify candidates independently and return their reports."""
    if tasks is None:
        tasks = get_initial_tasks()

    task_lookup = {task.task_id: task for task in tasks}
    if len(task_lookup) != len(tasks):
        raise ValueError("Benchmark contains duplicate task IDs.")

    return [
        verify_candidate(candidate, task_lookup, timeout_seconds)
        for candidate in candidates
    ]
