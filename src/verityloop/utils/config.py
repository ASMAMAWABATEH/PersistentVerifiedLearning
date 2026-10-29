
"""Load and validate VerityLoop experiment configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate its required sections."""
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    required_sections = {
        "project",
        "benchmark",
        "recursive_learning",
        "distillation",
        "evaluation",
        "outputs",
    }
    missing = required_sections - config.keys()

    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    project = config["project"]
    if not isinstance(project, dict) or "seed" not in project:
        raise ValueError("The project section must define a seed.")

    fractions = config["benchmark"]
    train = fractions["train_fraction"]
    dev = fractions["dev_fraction"]
    test = fractions["test_fraction"]

    if any(not isinstance(x, (int, float)) or x < 0 for x in (train, dev, test)):
        raise ValueError("Split fractions must be non-negative numbers.")

    if abs(train + dev + test - 1.0) > 1e-9:
        raise ValueError("Train, development, and test fractions must sum to 1.")

    distillation = config["distillation"]
    if distillation["temperature"] <= 0:
        raise ValueError("Distillation temperature must be positive.")

    if not 0.0 <= distillation["alpha"] <= 1.0:
        raise ValueError("Distillation alpha must be between 0 and 1.")

    return config
