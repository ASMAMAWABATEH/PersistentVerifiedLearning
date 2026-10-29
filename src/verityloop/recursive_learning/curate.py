
"""Evidence-based candidate curation."""

from dataclasses import dataclass

from verityloop.recursive_learning.generate import Candidate
from verityloop.recursive_learning.verify import VerificationReport


@dataclass(frozen=True)
class CurationResult:
    """Accepted candidates and explicit rejection records."""

    accepted: tuple[Candidate, ...]
    rejected: tuple[tuple[Candidate, str], ...]


def curate_candidates(
    candidates: list[Candidate],
    reports: list[VerificationReport],
) -> CurationResult:
    """Accept only candidates that pass every test."""
    if len(candidates) != len(reports):
        raise ValueError(
            "Each candidate must have exactly one verification report."
        )

    accepted = []
    rejected = []
    seen_task_ids = set()

    for candidate, report in zip(candidates, reports):
        if candidate.task_id != report.task_id:
            raise ValueError(
                f"Candidate/report mismatch: {candidate.task_id} "
                f"!= {report.task_id}"
            )

        if candidate.task_id in seen_task_ids:
            raise ValueError(
                f"Duplicate candidate for task: {candidate.task_id}"
            )
        seen_task_ids.add(candidate.task_id)

        if report.total_tests <= 0:
            rejected.append((candidate, "No tests were executed."))
        elif report.passed and report.passed_tests == report.total_tests:
            accepted.append(candidate)
        else:
            reason = (
                f"Passed {report.passed_tests}/{report.total_tests} tests."
            )
            rejected.append((candidate, reason))

    return CurationResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )
