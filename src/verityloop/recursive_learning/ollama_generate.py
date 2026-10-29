"""Generate coding candidates using a locally running Ollama model."""

import json
import os
import re
from urllib.request import Request, urlopen

from verityloop.benchmark.tasks import CodingTask
from verityloop.recursive_learning.generate import Candidate


def generate_ollama_candidates(
    tasks: list[CodingTask],
    model: str = "qwen2.5:3b",
    timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT", "180")),
    example_tasks: list[CodingTask] | None = None,
    example_solutions: list[tuple[CodingTask, str]] | None = None,
) -> list[Candidate]:
    """Generate candidates, optionally using verified training examples."""
    examples = example_tasks or []
    verified_solutions = example_solutions or []
    candidates = []

    for task in tasks:
        prompt_parts = [
            "Solve this Python coding benchmark task.",
            f"Task: {task.prompt}",
            f"Required function name: {task.function_name}",
        ]

        if examples or verified_solutions:
            prompt_parts.append(
                "\nVerified examples from the training split follow. "
                "Use them to learn the expected solution style and reasoning."
            )
            for example in examples:
                prompt_parts.extend([
                    f"\nExample task: {example.prompt}",
                    "Verified reference solution:",
                    example.reference_code,
                ])
            for example, solution in verified_solutions:
                prompt_parts.extend([
                    f"\nExample task: {example.prompt}",
                    "Independently verified model-generated solution:",
                    solution,
                ])

        prompt_parts.append(
            "\nNow solve the target task above. Return only a complete "
            "Python function definition, without Markdown fences, "
            "explanations, or test code."
        )
        prompt = "\n".join(prompt_parts)

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")

        request = Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))

        code = result.get("response", "").strip()
        match = re.search(
            r"```(?:python)?\s*(.*?)```",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            code = match.group(1).strip()

        if not code:
            raise RuntimeError(f"Ollama returned empty code for {task.task_id}")

        condition = "recursive" if verified_solutions else ("few_shot" if examples else "zero_shot")
        candidates.append(Candidate(
            task_id=task.task_id,
            code=code,
            round_number=0,
            source=f"ollama:{model}:{condition}",
        ))
        print(f"Generated ({condition}): {task.task_id}")

    return candidates
