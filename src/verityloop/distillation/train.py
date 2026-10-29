"""Training routines for supervised learning and knowledge distillation."""

import numpy as np

from verityloop.distillation.losses import one_hot, softmax


def train_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    seed: int,
    batch_size: int = 64,
    learning_rate: float = 0.08,
    teacher_probs: np.ndarray | None = None,
    temperature: float = 3.0,
    alpha: float = 0.5,
) -> None:
    """Train an MLP using SGD, optionally distilling teacher probabilities."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs < 0:
        raise ValueError("epochs cannot be negative")
    if teacher_probs is not None and temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if teacher_probs is not None and teacher_probs.shape != (len(x), model.b2.size):
        raise ValueError("teacher_probs must match the sample and class dimensions")

    rng = np.random.default_rng(seed)
    targets = one_hot(y, n_classes=model.b2.size)
    n_samples = len(x)

    for _ in range(epochs):
        indices = rng.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            batch = indices[start : start + batch_size]
            xb, yb = x[batch], targets[batch]
            hidden, logits = model.forward(xb)
            probs = softmax(logits)

            if teacher_probs is None:
                grad_logits = probs - yb
            else:
                teacher = teacher_probs[batch]
                student_t = softmax(logits, temperature)
                grad_logits = (
                    (1.0 - alpha) * (probs - yb)
                    + alpha
                    * temperature
                    * (student_t - teacher)
                )

            grad_logits /= len(batch)
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = (grad_logits @ model.w2.T) * (1.0 - hidden**2)
            grad_w1 = xb.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            model.w1 -= learning_rate * grad_w1
            model.b1 -= learning_rate * grad_b1
            model.w2 -= learning_rate * grad_w2
            model.b2 -= learning_rate * grad_b2
