"""NumPy model architectures for knowledge distillation."""

import numpy as np

from verityloop.distillation.losses import softmax


class MLP:
    """A one-hidden-layer MLP with tanh activation and softmax outputs."""

    def __init__(
        self,
        hidden: int,
        seed: int,
        input_dim: int = 2,
        n_classes: int = 3,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(
            0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden)
        )
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(
            0.0, np.sqrt(1.0 / hidden), size=(hidden, n_classes)
        )
        self.b2 = np.zeros(n_classes)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return hidden activations and output logits."""
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return hidden, logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        _, logits = self.forward(x)
        return softmax(logits)

    def parameter_count(self) -> int:
        """Count trainable scalar parameters."""
        return sum(
            parameter.size
            for parameter in (self.w1, self.b1, self.w2, self.b2)
        )

    def size_bytes(self) -> int:
        """Return the memory occupied by model parameters."""
        return sum(
            parameter.nbytes
            for parameter in (self.w1, self.b1, self.w2, self.b2)
        )
