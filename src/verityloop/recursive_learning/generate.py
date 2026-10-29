
"""Candidate generation for verified recursive learning."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """A proposed solution awaiting independent verification."""

    task_id: str
    code: str
    round_number: int
    source: str = "baseline"


def generate_baseline_candidates(
    tasks: list[dict[str, Any]],
) -> list[Candidate]:
    """Create initial candidates from the benchmark reference solutions.

    This baseline is for pipeline validation only. It is not a learned
    model and must not be presented as evidence of self-improvement.
    """
    candidates = []

    for task in tasks:
        candidates.append(
            Candidate(
                task_id=task["task_id"],
                code=task["reference_code"],
                round_number=0,
                source="reference_baseline",
            )
        )

    return candidates
