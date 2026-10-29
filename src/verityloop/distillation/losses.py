"""Numerically stable loss and probability functions for knowledge distillation."""

import numpy as np

N_CLASSES = 3


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert logits to probabilities with numerical stabilization."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(shifted)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Encode integer class labels as one-hot rows."""
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("class labels are outside the valid range")
    return np.eye(n_classes)[y]


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Compute mean cross-entropy from probabilities and integer labels."""
    selected = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.maximum(selected, 1e-12))))


def distillation_kl(
    teacher_probs: np.ndarray,
    student_logits: np.ndarray,
    temperature: float,
) -> float:
    """Compute temperature-scaled KL divergence from teacher to student."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_probs = softmax(student_logits, temperature)
    teacher_log = np.log(np.maximum(teacher_probs, 1e-12))
    student_log = np.log(np.maximum(student_probs, 1e-12))
    per_sample = np.sum(teacher_probs * (teacher_log - student_log), axis=1)
    return float(temperature**2 * np.mean(per_sample))


def knowledge_distillation_loss(
    student_logits: np.ndarray,
    teacher_probs: np.ndarray,
    targets: np.ndarray,
    temperature: float,
    alpha: float,
) -> float:
    """Compute alpha*T^2*KL(teacher || student) + (1-alpha)*CE(student, targets)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if teacher_probs.shape != student_logits.shape:
        raise ValueError("teacher_probs and student_logits must have the same shape")
    if len(targets) != len(student_logits):
        raise ValueError("targets and student_logits must have the same length")

    kl_loss = distillation_kl(teacher_probs, student_logits, temperature)
    hard_loss = cross_entropy(softmax(student_logits), targets)
    return float(alpha * kl_loss + (1.0 - alpha) * hard_loss)
