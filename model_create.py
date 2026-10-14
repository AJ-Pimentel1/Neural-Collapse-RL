"""Fixed-architecture MLP used in all experiments.

Architecture, depth, and width are held fixed across every run; only the
optimization hyperparameters are controlled by the agents.
"""

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class MLPConfig:
    """Architecture parameters (fixed for the whole project).

    Fields to define: input_dim, hidden_width, depth (number of hidden layers,
    2-3), num_classes, activation (e.g. "relu"), use_bias for the classifier,
    kernel_initializer (e.g. "he_normal").
    """


class NCMLP(tf.keras.Model):
    """MLP split into a feature extractor and a linear classifier head.

    The split exposes the penultimate-layer features h(x) and the classifier
    weights W needed for the NC1-NC4 metrics.
    """

    def __init__(self, config: MLPConfig):
        """Build the feature extractor (depth x Dense with activation) and the final linear Dense classifier (no activation)."""

    def extract_features(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Return the penultimate-layer activations h(x), shape (batch, hidden_width)."""

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Return the logits W h(x) + b, shape (batch, num_classes)."""

    @property
    def classifier_weight(self) -> tf.Tensor:
        """Return the classifier weight matrix W, shape (num_classes, hidden_width).

        Keras stores the Dense kernel as (hidden_width, num_classes), so this
        returns its transpose.
        """

    @property
    def classifier_bias(self) -> tf.Tensor | None:
        """Return the classifier bias b, or None if the classifier has no bias."""


def build_model(config: MLPConfig, seed: int) -> NCMLP:
    """Seed TF/NumPy/Python (tf.keras.utils.set_random_seed), construct an NCMLP,
    and build its weights by calling it on a dummy batch of shape (1, input_dim).

    Each RL episode calls this to start from fresh random weights.
    """


def count_parameters(model: tf.keras.Model) -> int:
    """Return the number of trainable parameters."""
