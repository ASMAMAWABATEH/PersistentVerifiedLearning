
"""Prepare a verified candidate set for a recursive-learning round."""

from dataclasses import dataclass

from verityloop.recursive_learning.generate import Candidate
from verityloop.recursive_learning.curate import CurationResult


@dataclass(frozen=True)
class RoundCandidateSet:
    """Auditable output of one candidate-selection round."""

    round_number: int
    candidates: tuple[Candidate, ...]
    accepted_count: int
    rejected_count: int


def prepare_candidate_set(
    round_number: int,
    curation: CurationResult,
) -> RoundCandidateSet:
    """Prepare curated candidates for downstream evaluation."""
    if round_number < 0:
        raise ValueError("round_number must be non-negative.")

    return RoundCandidateSet(
        round_number=round_number,
        candidates=curation.accepted,
        accepted_count=len(curation.accepted),
        rejected_count=len(curation.rejected),
    )
