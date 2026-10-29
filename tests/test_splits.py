
"""Automated tests for deterministic, family-level data splitting."""

import pytest

from verityloop.benchmark.dataset import build_benchmark
from verityloop.evaluation.splits import split_by_family, validate_splits


def family_sets(splits):
    return {
        name: {record["family"] for record in records}
        for name, records in splits.items()
    }


def test_all_records_appear_exactly_once():
    records = build_benchmark()
    splits = split_by_family(records, seed=42)

    original_ids = sorted(record["task_id"] for record in records)
    split_ids = sorted(
        record["task_id"]
        for items in splits.values()
        for record in items
    )

    assert split_ids == original_ids
    assert len(split_ids) == len(set(split_ids))


def test_families_do_not_overlap():
    splits = split_by_family(build_benchmark(), seed=42)
    families = family_sets(splits)

    assert families["train"].isdisjoint(families["dev"])
    assert families["train"].isdisjoint(families["test"])
    assert families["dev"].isdisjoint(families["test"])


def test_every_split_is_nonempty():
    splits = split_by_family(build_benchmark(), seed=42)

    assert set(splits) == {"train", "dev", "test"}
    assert all(splits[name] for name in splits)


def test_same_seed_produces_same_assignment():
    records = build_benchmark()

    first = split_by_family(records, seed=42)
    second = split_by_family(records, seed=42)

    first_ids = {
        name: [record["task_id"] for record in items]
        for name, items in first.items()
    }
    second_ids = {
        name: [record["task_id"] for record in items]
        for name, items in second.items()
    }

    assert first_ids == second_ids


def test_invalid_fractions_are_rejected():
    records = build_benchmark()

    with pytest.raises(ValueError, match="sum to 1"):
        split_by_family(
            records,
            train_fraction=0.7,
            dev_fraction=0.2,
            test_fraction=0.2,
        )


def test_too_few_families_are_rejected():
    records = build_benchmark()[:2]

    with pytest.raises(ValueError, match="At least three"):
        split_by_family(records)


def test_validator_rejects_family_leakage():
    records = build_benchmark()
    overlapping = {
        "train": [records[0]],
        "dev": [records[1]],
        "test": [records[2]],
    }

    # Inject the training family into the development split.
    overlapping["dev"].append({
        **records[0],
        "task_id": "leaked_task",
    })

    with pytest.raises(ValueError, match="leakage"):
        validate_splits(overlapping)
