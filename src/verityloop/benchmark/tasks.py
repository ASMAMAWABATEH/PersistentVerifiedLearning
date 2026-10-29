
"""Data structures for independently testable coding benchmark tasks."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TestCase:
    """One input/output example used to check a candidate solution."""

    inputs: tuple[Any, ...]
    expected: Any


@dataclass(frozen=True)
class CodingTask:
    """A coding problem with a reference implementation and test cases."""

    task_id: str
    family: str
    prompt: str
    function_name: str
    reference_code: str
    test_cases: tuple[TestCase, ...]


def get_initial_tasks() -> list[CodingTask]:
    """Return the deterministic benchmark tasks."""
    tasks = [
        CodingTask(
            task_id="absolute_value_v1",
            family="absolute_value",
            prompt="Implement absolute_value(x), returning the non-negative magnitude of x.",
            function_name="absolute_value",
            reference_code="def absolute_value(x):\n    return abs(x)\n",
            test_cases=(
                TestCase((-5,), 5),
                TestCase((0,), 0),
                TestCase((7,), 7),
                TestCase((-2.5,), 2.5),
            ),
        ),
        CodingTask(
            task_id="maximum_v1",
            family="maximum",
            prompt="Implement maximum(a, b), returning the larger of two numbers.",
            function_name="maximum",
            reference_code="def maximum(a, b):\n    return a if a >= b else b\n",
            test_cases=(
                TestCase((3, 5), 5),
                TestCase((9, 2), 9),
                TestCase((-4, -1), -1),
                TestCase((6, 6), 6),
            ),
        ),
        CodingTask(
            task_id="is_even_v1",
            family="is_even",
            prompt="Implement is_even(n), returning True exactly when integer n is even.",
            function_name="is_even",
            reference_code="def is_even(n):\n    return n % 2 == 0\n",
            test_cases=(
                TestCase((2,), True),
                TestCase((3,), False),
                TestCase((0,), True),
                TestCase((-4,), True),
            ),
        ),
        CodingTask(
            task_id="count_vowels_v1",
            family="count_vowels",
            prompt="Implement count_vowels(text), counting a, e, i, o, u regardless of case.",
            function_name="count_vowels",
            reference_code=(
                "def count_vowels(text):\n"
                "    return sum(ch.lower() in 'aeiou' for ch in text)\n"
            ),
            test_cases=(
                TestCase(("hello",), 2),
                TestCase(("AEIOU",), 5),
                TestCase(("",), 0),
                TestCase(("rhythm",), 0),
            ),
        ),
        CodingTask(
            task_id="clamp_v1",
            family="clamp",
            prompt="Implement clamp(x, low, high), restricting x to the inclusive interval.",
            function_name="clamp",
            reference_code=(
                "def clamp(x, low, high):\n"
                "    return min(max(x, low), high)\n"
            ),
            test_cases=(
                TestCase((5, 0, 10), 5),
                TestCase((-3, 0, 10), 0),
                TestCase((15, 0, 10), 10),
                TestCase((0, 0, 10), 0),
            ),
        ),
        CodingTask(
            task_id="second_largest_v1",
            family="second_largest",
            prompt=(
                "Implement second_largest(values), returning the second-largest "
                "distinct number. Assume at least two distinct values."
            ),
            function_name="second_largest",
            reference_code=(
                "def second_largest(values):\n"
                "    return sorted(set(values))[-2]\n"
            ),
            test_cases=(
                TestCase(([1, 3, 2],), 2),
                TestCase(([5, 5, 4],), 4),
                TestCase(([-1, -5, -3],), -3),
                TestCase(([1, 2],), 1),
            ),
        ),
        CodingTask(
            task_id="reverse_text_v1",
            family="reverse_text",
            prompt="Implement reverse_text(text), returning the characters in reverse order.",
            function_name="reverse_text",
            reference_code="def reverse_text(text):\n    return text[::-1]\n",
            test_cases=(
                TestCase(("hello",), "olleh"),
                TestCase(("",), ""),
                TestCase(("a",), "a"),
                TestCase(("ab cd",), "dc ba"),
            ),
        ),
        CodingTask(
            task_id="is_palindrome_v1",
            family="is_palindrome",
            prompt=(
                "Implement is_palindrome(text), returning True if text reads "
                "the same forward and backward, ignoring case."
            ),
            function_name="is_palindrome",
            reference_code=(
                "def is_palindrome(text):\n"
                "    text = text.lower()\n"
                "    return text == text[::-1]\n"
            ),
            test_cases=(
                TestCase(("level",), True),
                TestCase(("Racecar",), True),
                TestCase(("hello",), False),
                TestCase(("",), True),
            ),
        ),
    ]
    from verityloop.benchmark.additional_tasks import get_additional_tasks
    return tasks + get_additional_tasks()
