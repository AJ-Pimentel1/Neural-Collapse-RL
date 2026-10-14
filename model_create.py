"""Fixed-architecture MLP used in all experiments.

Architecture, depth, and width are held fixed across every run; only the
optimization hyperparameters are controlled by the agents.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    """Architecture parameters (fixed for the whole project).

    Fields to define: input_dim, hidden_width, depth (number of hidden layers,
    2-3), num_classes, activation (e.g. ReLU), use_bias for the classifier.
    """


class NCMLP(nn.Module):
    """MLP split into a feature extractor and a linear classifier head.

    The split exposes the penultimate-layer features h(x) and the classifier
    weights W needed for the NC1-NC4 metrics.
    """

    def __init__(self, config: MLPConfig):
        """Build the feature extractor (depth x [Linear, activation]) and the final Linear classifier."""

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate-layer activations h(x), shape (batch, hidden_width)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits W h(x) + b, shape (batch, num_classes)."""

    @property
    def classifier_weight(self) -> torch.Tensor:
        """Return the classifier weight matrix W, shape (num_classes, hidden_width)."""

    @property
    def classifier_bias(self) -> torch.Tensor | None:
        """Return the classifier bias b, or None if the classifier has no bias."""


def init_weights(module: nn.Module) -> None:
    """Apply the standard initialization (e.g. Kaiming for ReLU layers, zero biases)."""


def build_model(config: MLPConfig, seed: int, device: str | torch.device = "cpu") -> NCMLP:
    """Seed torch, construct an NCMLP, apply init_weights, and move it to device.

    Each RL episode calls this to start from fresh random weights.
    """


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
