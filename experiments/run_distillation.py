"""Lightweight synthetic-data NumPy knowledge-distillation experiment.

Compares a supervised student with a student trained using teacher soft targets.
It is a mechanism benchmark, not distillation of verified coding examples or
evidence of improved performance on the coding-task benchmark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from verityloop.distillation.losses import softmax
from verityloop.distillation.models import MLP
from verityloop.distillation.train import train_model

SEED = 42
N_CLASSES = 3
INPUT_DIM = 2
TRAIN_PER_CLASS = 500
DEV_PER_CLASS = 150
TEST_PER_CLASS = 250
EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 0.08
TEMPERATURE = 3.0
ALPHA = 0.5


def make_data(rng: np.random.Generator, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible, nonlinear three-class classification dataset."""
    centers = np.array([[-1.4, -0.8], [1.4, -0.8], [0.0, 1.4]])
    xs, ys = [], []
    for label, center in enumerate(centers):
        x = rng.normal(center, 0.72, size=(per_class, INPUT_DIM))
        xs.append(x)
        ys.append(np.full(per_class, label, dtype=np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = rng.permutation(len(y))
    return x[order].astype(np.float64), y[order]


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    start = time.perf_counter()
    probs = model.predict_proba(x)
    elapsed = time.perf_counter() - start
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(np.mean(pred == y)),
        "cross_entropy": float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)))),
        "inference_ms_per_sample": float(elapsed * 1000 / len(y)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    x_train, y_train = make_data(rng, TRAIN_PER_CLASS)
    x_dev, y_dev = make_data(rng, DEV_PER_CLASS)
    x_test, y_test = make_data(rng, TEST_PER_CLASS)

    # Standardize from training data only.
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std
    x_test = (x_test - mean) / std

    teacher = MLP(hidden=64, seed=SEED + 1)
    train_model(
        teacher, x_train, y_train, EPOCHS, SEED + 2,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
    )
    teacher_dev = evaluate(teacher, x_dev, y_dev)

    # Teacher soft targets are generated for training examples only.
    teacher_targets = softmax(teacher.forward(x_train)[1], TEMPERATURE)

    baseline = MLP(hidden=16, seed=SEED + 3)
    train_model(
        baseline, x_train, y_train, EPOCHS, SEED + 4,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
    )

    distilled = MLP(hidden=16, seed=SEED + 3)
    train_model(
        distilled, x_train, y_train, EPOCHS, SEED + 4,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        teacher_probs=teacher_targets,
        temperature=TEMPERATURE,
        alpha=ALPHA,
    )

    # Development metrics are reported separately from final test metrics.
    dev_results = {
        "teacher": evaluate(teacher, x_dev, y_dev),
        "student_supervised": evaluate(baseline, x_dev, y_dev),
        "student_distilled": evaluate(distilled, x_dev, y_dev),
    }
    test_results = {
        "teacher": evaluate(teacher, x_test, y_test),
        "student_supervised": evaluate(baseline, x_test, y_test),
        "student_distilled": evaluate(distilled, x_test, y_test),
    }

    results = {
        "experiment": "numpy_knowledge_distillation",
        "seed": SEED,
        "dataset": {
            "type": "synthetic_three_class",
            "train_examples": len(y_train),
            "development_examples": len(y_dev),
            "test_examples": len(y_test),
            "input_features": INPUT_DIM,
            "classes": N_CLASSES,
        },
        "method": {
            "teacher_hidden_units": 64,
            "student_hidden_units": 16,
            "epochs": EPOCHS,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "optimizer": "mini_batch_SGD",
            "framework": "NumPy",
        },
        "teacher_development": teacher_dev,
        "development_metrics": dev_results,
        "final_test_metrics": test_results,
        "model_efficiency": {
            "teacher_parameters": teacher.parameter_count(),
            "student_parameters": baseline.parameter_count(),
            "student_parameter_reduction_fraction": float(
                1 - baseline.parameter_count() / teacher.parameter_count()
            ),
            "teacher_parameter_bytes": teacher.size_bytes(),
            "student_parameter_bytes": baseline.size_bytes(),
        },
        "interpretation": (
            "Synthetic-data proof of concept only; does not establish "
            "improvement on VerityLoop coding tasks or real-world deployment."
        ),
    }

    output = Path("results/logs/distillation_experiment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
