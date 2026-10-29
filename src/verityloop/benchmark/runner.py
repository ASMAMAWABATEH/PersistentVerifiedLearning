
"""Execute candidate solutions against predefined benchmark tests."""

from dataclasses import asdict, dataclass
import json
import subprocess
import sys
from typing import Any

from verityloop.benchmark.tasks import CodingTask


@dataclass(frozen=True)
class TestResult:
    """Outcome of one test case."""

    passed: bool
    expected: Any
    actual: Any = None
    error: str | None = None


@dataclass(frozen=True)
class CandidateResult:
    """Aggregate evaluation results for a candidate implementation."""

    task_id: str
    passed: bool
    passed_tests: int
    total_tests: int
    test_results: tuple[TestResult, ...]

    @property
    def pass_rate(self) -> float:
        """Fraction of test cases passed."""
        return self.passed_tests / self.total_tests if self.total_tests else 0.0


def _run_candidate(
    code: str,
    function_name: str,
    inputs: tuple[Any, ...],
    timeout_seconds: float,
) -> tuple[Any, str | None]:
    """Run a candidate in a separate Python process and return its JSON result."""
    payload = {
        "code": code,
        "function_name": function_name,
        "inputs": inputs,
    }

    child_script = r'''
import json
import sys
import traceback

payload = json.loads(sys.stdin.read())

try:
    namespace = {"__builtins__": __builtins__}
    exec(compile(payload["code"], "<candidate>", "exec"), namespace)
    function = namespace[payload["function_name"]]
    result = function(*payload["inputs"])
    print(json.dumps({"ok": True, "result": result}))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}"
    }))
'''

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", child_script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout_seconds} seconds"

    if completed.returncode != 0:
        message = completed.stderr.strip() or "Candidate process failed"
        return None, message[-1000:]

    try:
        output = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None, "Candidate returned a non-JSON result or printed unexpected output"

    if not output.get("ok"):
        return None, output.get("error", "Candidate execution failed")

    return output.get("result"), None


def evaluate_candidate(
    task: CodingTask,
    candidate_code: str,
    timeout_seconds: float = 2.0,
) -> CandidateResult:
    """Evaluate candidate code against every test case for a task."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    results = []

    for case in task.test_cases:
        actual, error = _run_candidate(
            code=candidate_code,
            function_name=task.function_name,
            inputs=case.inputs,
            timeout_seconds=timeout_seconds,
        )

        passed = error is None and actual == case.expected
        results.append(
            TestResult(
                passed=passed,
                expected=case.expected,
                actual=actual,
                error=error,
            )
        )

    passed_tests = sum(result.passed for result in results)

    return CandidateResult(
        task_id=task.task_id,
        passed=passed_tests == len(results) and bool(results),
        passed_tests=passed_tests,
        total_tests=len(results),
        test_results=tuple(results),
    )


def result_to_dict(result: CandidateResult) -> dict[str, Any]:
    """Convert a result to a JSON-serializable dictionary."""
    return asdict(result)


def save_result(result: CandidateResult, path: str) -> None:
    """Save one evaluation result as JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result_to_dict(result), file, indent=2)
        file.write("\n")
