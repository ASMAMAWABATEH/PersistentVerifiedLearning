"""Tests for the NumPy knowledge-distillation experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.run_distillation import MLP
from verityloop.distillation.losses import (
    cross_entropy,
    distillation_kl,
    knowledge_distillation_loss,
    one_hot,
    softmax,
)
from verityloop.distillation.train import train_model


def test_softmax_probabilities_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [-2.0, 0.0, 4.0]])
    probabilities = softmax(logits)

    assert probabilities.shape == logits.shape
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_softmax_is_stable_for_large_logits():
    probabilities = softmax(np.array([[10000.0, 10001.0, 9999.0]]))

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_one_hot_encodes_class_labels():
    encoded = one_hot(np.array([2, 0, 1]))

    assert encoded.shape == (3, 3)
    np.testing.assert_array_equal(
        encoded,
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    )


def test_model_initialization_is_reproducible():
    first = MLP(hidden=16, seed=123)
    second = MLP(hidden=16, seed=123)

    for first_param, second_param in zip(
        (first.w1, first.b1, first.w2, first.b2),
        (second.w1, second.b1, second.w2, second.b2),
        strict=True,
    ):
        np.testing.assert_array_equal(first_param, second_param)


def test_student_parameter_count_and_size():
    student = MLP(hidden=16, seed=123)

    assert student.parameter_count() == 99
    assert student.size_bytes() == 99 * np.dtype(np.float64).itemsize


def test_knowledge_distillation_loss_matches_weighted_objective():
    teacher_logits = np.array([[1.0, -0.5, 0.2], [-0.3, 0.7, 0.1]])
    student_logits = np.array([[0.2, 0.5, -0.1], [0.4, -0.2, 0.8]])
    targets = np.array([1, 2])
    temperature = 2.5
    alpha = 0.7
    teacher_probs = softmax(teacher_logits, temperature)

    loss = knowledge_distillation_loss(
        student_logits, teacher_probs, targets, temperature, alpha
    )
    expected = alpha * distillation_kl(
        teacher_probs, student_logits, temperature
    ) + (1.0 - alpha) * cross_entropy(softmax(student_logits), targets)

    np.testing.assert_allclose(loss, expected)


def test_distillation_training_updates_student_weights():
    x = np.array([[-1.0, 0.5], [0.3, 1.2], [1.0, -0.7]])
    y = np.array([0, 1, 2])
    teacher = MLP(hidden=8, seed=1)
    student = MLP(hidden=4, seed=2)
    teacher_probs = softmax(teacher.forward(x)[1], temperature=2.0)
    before = [parameter.copy() for parameter in (
        student.w1, student.b1, student.w2, student.b2
    )]

    train_model(
        student,
        x,
        y,
        epochs=2,
        seed=3,
        batch_size=3,
        teacher_probs=teacher_probs,
        temperature=2.0,
        alpha=0.6,
    )

    after = (student.w1, student.b1, student.w2, student.b2)
    assert any(not np.array_equal(old, new) for old, new in zip(before, after, strict=True))
