"""Fixed-architecture MLP used in all experiments.

Architecture, depth, and width are held fixed across every run; only the
optimization hyperparameters are controlled by the agents.
"""

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class MLPConfig:
    """Architecture parameters (fixed for the whole project).

    3 x 256 gives ~140k parameters, far more than N = 5,000, so interpolation
    is possible, and a penultimate width far above C - 1 = 9 so an ETF fits.
    """
    input_dim: int = 20
    hidden_width: int = 256
    depth: int = 3
    num_classes: int = 10
    activation: str = "relu"
    use_bias: bool = True
    kernel_initializer: str = "he_normal"


class NCMLP(tf.keras.Model):
    """MLP split into a feature extractor and a linear classifier head.

    The split exposes the penultimate-layer features h(x) and the classifier
    weights W needed for the NC1-NC4 metrics.
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.mlp_config = config
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(config.hidden_width, activation=config.activation,
                                  kernel_initializer=config.kernel_initializer)
            for _ in range(config.depth)
        ])
        self.classifier = tf.keras.layers.Dense(config.num_classes, use_bias=config.use_bias,
                                                kernel_initializer=config.kernel_initializer)

    def extract_features(self, x, training=False):
        """Return the penultimate-layer activations h(x), shape (batch, hidden_width)."""
        return self.feature_extractor(x, training=training)

    def call(self, x, training=False):
        """Return the logits W h(x) + b, shape (batch, num_classes)."""
        return self.classifier(self.extract_features(x, training=training))

    @property
    def classifier_weight(self):
        """Classifier weight matrix W with rows w_c, shape (num_classes, hidden_width).

        Keras stores the Dense kernel as (hidden_width, num_classes), so this
        returns its transpose.
        """
        return tf.transpose(self.classifier.kernel)

    @property
    def classifier_bias(self):
        """Classifier bias b, or None if the classifier has no bias."""
        return self.classifier.bias if self.mlp_config.use_bias else None


def build_model(config: MLPConfig, seed: int) -> NCMLP:
    """Seed Python/NumPy/TF, construct an NCMLP, and build its weights on a dummy batch.

    Each run (and each RL episode) calls this to start from fresh random weights.
    """
    tf.keras.utils.set_random_seed(seed)
    model = NCMLP(config)
    model(tf.zeros((1, config.input_dim)))
    return model


def count_parameters(model: tf.keras.Model) -> int:
    """Return the number of trainable parameters."""
    return int(sum(tf.size(v) for v in model.trainable_variables))
