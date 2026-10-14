"""Neural collapse metrics (NC1-NC4), accuracy, and metric standardization.

All quantities are computed on the training set (there is no held-out split),
every K epochs, from a full pass over the penultimate-layer features.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class ClassStatistics:
    """Feature statistics needed by the NC metrics.

    Fields to define: global_mean (d,), class_means (C, d), sigma_w (d, d)
    within-class covariance, sigma_b (d, d) between-class covariance.
    """


@torch.no_grad()
def collect_features(model, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model over the full loader in eval mode.

    Return (features, labels, logits): penultimate-layer activations,
    integer labels, and classifier outputs for every training sample.
    """


def compute_class_statistics(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> ClassStatistics:
    """Compute the global mean, class means, Sigma_W, and Sigma_B from the features."""


def compute_nc1(stats: ClassStatistics, num_classes: int) -> float:
    """Variability collapse: Tr(Sigma_W Sigma_B^dagger) / C, using the pseudo-inverse of Sigma_B."""


def compute_nc2(stats: ClassStatistics) -> float:
    """Convergence to simplex ETF.

    Center the class means by the global mean, normalize them, form their Gram
    matrix, and return its (Frobenius) deviation from the canonical simplex-ETF
    Gram matrix (C/(C-1)) * (I - 11^T / C).
    """


def compute_nc3(stats: ClassStatistics, classifier_weight: torch.Tensor) -> float:
    """Convergence to self-duality.

    Return the misalignment between the normalized classifier weights W and the
    normalized centered class means, e.g. || W/||W||_F - M/||M||_F ||_F.
    """


def compute_nc4(features: torch.Tensor, logits: torch.Tensor, stats: ClassStatistics) -> float:
    """Simplification to nearest class center.

    Return the fraction of training samples where the classifier's argmax
    prediction disagrees with the nearest (Euclidean) class-mean prediction.
    """


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return training accuracy in [0, 1]."""


def compute_norms(features: torch.Tensor, classifier_weight: torch.Tensor) -> tuple[float, float]:
    """Return (mean feature norm ||f||, classifier weight norm ||W||) for the state vector."""


def evaluate_checkpoint(model, loader: DataLoader, num_classes: int, device) -> dict:
    """Compute every quantity for one checkpoint.

    Return a dict with keys: loss, acc, nc1, nc2, nc3, nc4, feature_norm,
    weight_norm. This is what the environment logs and turns into a state.
    """


class NCStandardizer:
    """Z-scores the raw NC metrics so no single metric dominates D_NC.

    Statistics (mu_i, s_i) are fit once on a reference pool of runs (SGD
    baseline plus randomly sampled hyperparameters) and then frozen, so the
    reward stays stationary during agent training.
    """

    METRICS = ("nc1", "nc2", "nc3", "nc4")

    def __init__(self):
        """Initialize empty mean/std containers and a fitted flag."""

    def fit(self, records: list[dict]) -> "NCStandardizer":
        """Compute mu_i and s_i for each NC metric over all checkpoints in records."""

    def transform(self, metrics: dict) -> dict:
        """Return standardized metrics (NC_i - mu_i) / s_i for each NC metric."""

    def composite_distance(self, metrics: dict) -> float:
        """Return D_NC: the equal-weight average of the four standardized metrics."""

    def save(self, path: str) -> None:
        """Write the frozen statistics to disk (e.g. JSON)."""

    @classmethod
    def load(cls, path: str) -> "NCStandardizer":
        """Load frozen statistics written by save."""
