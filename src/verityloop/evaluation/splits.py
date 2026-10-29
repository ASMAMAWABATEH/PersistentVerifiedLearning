
"""Deterministic data splits with task-family leakage prevention."""

import random
from collections import defaultdict
from typing import Any


def split_by_family(
    records: list[dict[str, Any]],
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split records by family so one family cannot appear in multiple splits."""
    fractions = (train_fraction, dev_fraction, test_fraction)

    if any(not isinstance(x, (int, float)) or x < 0 for x in fractions):
        raise ValueError("Split fractions must be non-negative numbers.")

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")

    if any(not isinstance(record.get("family"), str) or not record["family"] for record in records):
        raise ValueError("Every record must have a non-empty family.")

    families = defaultdict(list)
    for record in records:
        families[record["family"]].append(record)

    names = list(families)
    if len(names) < 3:
        raise ValueError("At least three distinct task families are required.")

    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("All three split fractions must be positive.")

    rng = random.Random(seed)
    rng.shuffle(names)

    # Allocate whole families using the largest-remainder method.
    exact_counts = [len(names) * fraction for fraction in fractions]
    counts = [int(count) for count in exact_counts]
    remaining = len(names) - sum(counts)
    order = sorted(
        range(3),
        key=lambda i: (exact_counts[i] - counts[i], -i),
        reverse=True,
    )

    for index in order[:remaining]:
        counts[index] += 1

    # Ensure all splits are represented, even for small development benchmarks.
    for empty_index in [i for i, count in enumerate(counts) if count == 0]:
        donor = max(range(3), key=lambda i: counts[i])
        if counts[donor] <= 1:
            raise ValueError("Not enough families to populate every split.")
        counts[donor] -= 1
        counts[empty_index] += 1

    train_end = counts[0]
    dev_end = train_end + counts[1]

    family_assignment = {}
    for family in names[:train_end]:
        family_assignment[family] = "train"
    for family in names[train_end:dev_end]:
        family_assignment[family] = "dev"
    for family in names[dev_end:]:
        family_assignment[family] = "test"

    splits = {"train": [], "dev": [], "test": []}
    for record in records:
        splits[family_assignment[record["family"]]].append(record)

    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Check required splits and enforce family-level separation."""
    required = {"train", "dev", "test"}
    if set(splits) != required:
        raise ValueError(f"Splits must contain exactly: {sorted(required)}")

    if any(not splits[name] for name in required):
        raise ValueError("Every split must contain at least one record.")

    family_sets = {
        name: {record["family"] for record in records}
        for name, records in splits.items()
    }

    if family_sets["train"] & family_sets["dev"]:
        raise ValueError("Task-family leakage between train and dev.")
    if family_sets["train"] & family_sets["test"]:
        raise ValueError("Task-family leakage between train and test.")
    if family_sets["dev"] & family_sets["test"]:
        raise ValueError("Task-family leakage between dev and test.")

    all_records = [record for records in splits.values() for record in records]
    if len(all_records) != sum(len(records) for records in splits.values()):
        raise ValueError("Split integrity check failed.")
