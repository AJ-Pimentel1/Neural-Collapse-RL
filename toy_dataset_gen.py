"""Class-balanced datasets for the neural collapse RL experiments.

Every dataset produced here is balanced (equal samples per class) because the
simplex-ETF structure of NC is contingent on balanced classes. There is no
held-out split: all metrics are computed on the training set.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class GaussianMixtureConfig:
    """Parameters for the synthetic Gaussian-mixture dataset.

    Fields to define: num_classes (10), input_dim, samples_per_class,
    cluster_std, mean_radius (spread of the class means), seed.
    """


def generate_class_means(config: GaussianMixtureConfig) -> np.ndarray:
    """Return a (num_classes, input_dim) array of class centers.

    Centers should be well separated (e.g. random directions scaled to
    mean_radius) and drawn deterministically from config.seed.
    """


def sample_gaussian_mixture(config: GaussianMixtureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sample a balanced Gaussian mixture.

    Draw exactly samples_per_class points around each class mean with
    isotropic noise of scale cluster_std. Return (X, y) with X of shape
    (num_classes * samples_per_class, input_dim) and integer labels y.
    """


class GaussianMixtureDataset(Dataset):
    """Torch Dataset wrapping a sampled Gaussian mixture."""

    def __init__(self, config: GaussianMixtureConfig):
        """Sample the data via sample_gaussian_mixture and store it as tensors."""

    def __len__(self) -> int:
        """Return the total number of samples."""

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (x, y) pair at index idx."""


def make_balanced_mnist_subset(samples_per_class: int, root: str, seed: int) -> Dataset:
    """Load MNIST train split and return a class-balanced subset.

    Select exactly samples_per_class images per digit (seeded), flatten each
    image to a vector so it can be fed to the MLP, and normalize.
    """


def check_class_balance(labels: np.ndarray | torch.Tensor) -> None:
    """Assert that every class has the same number of samples."""


def build_dataloader(dataset: Dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    """Build a uniform-sampling DataLoader with the given batch size.

    Batch size is an agent action, so this is called again whenever the agent
    changes it. Use a seeded generator so shuffling is reproducible.
    """


def build_eval_loader(dataset: Dataset, batch_size: int = 1024) -> DataLoader:
    """Build a non-shuffled DataLoader over the full training set for NC metric passes."""
