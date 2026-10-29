"""Knowledge distillation models, losses, and training utilities."""

from verityloop.distillation.losses import (
    cross_entropy,
    distillation_kl,
    knowledge_distillation_loss,
    one_hot,
    softmax,
)
from verityloop.distillation.models import MLP
from verityloop.distillation.train import train_model

__all__ = [
    "MLP",
    "cross_entropy",
    "distillation_kl",
    "knowledge_distillation_loss",
    "one_hot",
    "softmax",
    "train_model",
]
