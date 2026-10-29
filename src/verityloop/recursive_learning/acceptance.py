
"""Acceptance and rollback policy for recursive learning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceDecision:
    """Decision and evidence for a proposed model update."""

    accepted: bool
    reason: str
    baseline_dev_score: float
    candidate_dev_score: float
    improvement: float


def decide_acceptance(
    baseline_dev_score: float,
    candidate_dev_score: float,
    minimum_gain: float = 0.02,
    require_no_regression: bool = True,
) -> AcceptanceDecision:
    """Accept an update only when its development score meets policy."""
    values = {
        "baseline_dev_score": baseline_dev_score,
        "candidate_dev_score": candidate_dev_score,
        "minimum_gain": minimum_gain,
    }

    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    improvement = candidate_dev_score - baseline_dev_score

    if require_no_regression and improvement < 0:
        return AcceptanceDecision(
            accepted=False,
            reason="Rejected: development score regressed.",
            baseline_dev_score=baseline_dev_score,
            candidate_dev_score=candidate_dev_score,
            improvement=improvement,
        )

    if improvement < minimum_gain:
        return AcceptanceDecision(
            accepted=False,
            reason=(
                f"Rejected: gain {improvement:.3f} is below "
                f"the required {minimum_gain:.3f}."
            ),
            baseline_dev_score=baseline_dev_score,
            candidate_dev_score=candidate_dev_score,
            improvement=improvement,
        )

    return AcceptanceDecision(
        accepted=True,
        reason="Accepted: development improvement meets the policy.",
        baseline_dev_score=baseline_dev_score,
        candidate_dev_score=candidate_dev_score,
        improvement=improvement,
    )
