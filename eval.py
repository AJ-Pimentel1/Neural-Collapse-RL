"""Neural collapse metrics (NC1-NC4), accuracy, norms, and metric standardization.

All quantities are computed on the training set from a single full pass over
the penultimate-layer features. Metrics are computed in float64 so the
pseudo-inverse in NC1 is stable once Sigma_B becomes (near) rank C - 1.
"""

import math
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class ClassStatistics:
    """Feature statistics needed by the NC metrics."""
    global_mean: tf.Tensor  # (d,)
    class_means: tf.Tensor  # (C, d), uncentered
    M: tf.Tensor            # (d, C), class means centered by the global mean, columns mu_c
    sigma_w: tf.Tensor      # (d, d) within-class covariance
    sigma_b: tf.Tensor      # (d, d) between-class covariance


def collect_features(model, X):
    """Return (features H, logits) for every sample in X with training=False."""
    H = model.extract_features(X, training=False)
    return H, model.classifier(H)


def compute_class_statistics(H, y, num_classes) -> ClassStatistics:
    """Compute the global mean, class means, centered mean matrix M, Sigma_W, and Sigma_B."""
    H = tf.cast(H, tf.float64)
    global_mean = tf.reduce_mean(H, axis=0)
    class_means = tf.stack([tf.reduce_mean(tf.boolean_mask(H, y == c), axis=0)
                            for c in range(num_classes)])
    within = H - tf.gather(class_means, y)
    sigma_w = tf.matmul(within, within, transpose_a=True) / tf.cast(tf.shape(H)[0], tf.float64)
    M = tf.transpose(class_means - global_mean)
    sigma_b = tf.matmul(M, M, transpose_b=True) / num_classes
    return ClassStatistics(global_mean, class_means, M, sigma_w, sigma_b)


def compute_nc1(stats: ClassStatistics, num_classes) -> float:
    """Variability collapse: Tr(Sigma_W Sigma_B^dagger) / C."""
    return float(tf.linalg.trace(stats.sigma_w @ tf.linalg.pinv(stats.sigma_b))) / num_classes


def compute_nc2(stats: ClassStatistics, num_classes) -> float:
    """Convergence to simplex ETF.

    Frobenius distance between the normalized class-mean Gram matrix
    M^T M / ||M^T M||_F and the normalized simplex-ETF Gram matrix
    (C/(C-1)) (I - 11^T / C).
    """
    G = tf.matmul(stats.M, stats.M, transpose_a=True)
    C = num_classes
    etf = (C / (C - 1)) * (tf.eye(C, dtype=tf.float64) - tf.ones((C, C), dtype=tf.float64) / C)
    return float(tf.norm(G / tf.norm(G) - etf / tf.norm(etf)))


def compute_nc3(stats: ClassStatistics, classifier_weight) -> float:
    """Convergence to self-duality: || W^T/||W||_F - M/||M||_F ||_F, with W of shape (C, d)."""
    Wt = tf.transpose(tf.cast(classifier_weight, tf.float64))
    return float(tf.norm(Wt / tf.norm(Wt) - stats.M / tf.norm(stats.M)))


def compute_nc4(H, logits, stats: ClassStatistics) -> float:
    """Fraction of samples where the network's argmax disagrees with the nearest class mean."""
    H = tf.cast(H, tf.float64)
    sq_dists = tf.reduce_sum(tf.square(H[:, None, :] - stats.class_means[None, :, :]), axis=-1)
    ncc_preds = tf.argmin(sq_dists, axis=1)
    model_preds = tf.argmax(logits, axis=1)
    return float(tf.reduce_mean(tf.cast(model_preds != ncc_preds, tf.float64)))


def compute_loss_and_accuracy(logits, y) -> tuple[float, float]:
    """Return (mean cross-entropy loss, accuracy in [0, 1])."""
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
    acc = tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32) == y, tf.float32))
    return float(loss), float(acc)


def compute_norms(H, classifier_weight) -> tuple[float, float]:
    """Return (mean feature norm ||h||, classifier Frobenius norm ||W||_F)."""
    return float(tf.reduce_mean(tf.norm(H, axis=1))), float(tf.norm(classifier_weight))


def evaluate_checkpoint(model, X, y, num_classes) -> dict:
    """Compute every training-set quantity for one checkpoint.

    Returns a dict of floats: loss, acc, nc1, nc2, nc3, nc4, feature_norm,
    weight_norm. If the loss is not finite (a diverged run) the NC metrics and
    norms are NaN instead of being computed.
    """
    H, logits = collect_features(model, X)
    loss, acc = compute_loss_and_accuracy(logits, y)
    out = {"loss": loss, "acc": acc}
    if not math.isfinite(loss):
        return out | {k: float("nan") for k in ("nc1", "nc2", "nc3", "nc4", "feature_norm", "weight_norm")}
    W = model.classifier_weight
    stats = compute_class_statistics(H, y, num_classes)
    feature_norm, weight_norm = compute_norms(H, W)
    return out | {
        "nc1": compute_nc1(stats, num_classes),
        "nc2": compute_nc2(stats, num_classes),
        "nc3": compute_nc3(stats, W),
        "nc4": compute_nc4(H, logits, stats),
        "feature_norm": feature_norm,
        "weight_norm": weight_norm,
    }


class NCStandardizer:
    """Standardizes the raw NC metrics so no single metric dominates D.

    NC~_i = clip((NC_i - mu_i) / s_i, -3, 3) and D = mean_i NC~_i (lower = more
    collapsed). (mu_i, s_i) are fit once on a reference pool of runs (SGD
    baseline plus runs with uniformly random actions) and then frozen, so the
    reward stays stationary during agent training.
    """

    METRICS = ("nc1", "nc2", "nc3", "nc4")

    def __init__(self):
        """Initialize empty mean/std containers and a fitted flag."""

    def fit(self, records: list[dict]) -> "NCStandardizer":
        """Compute mu_i and s_i for each NC metric over all checkpoints in records."""

    def transform(self, metrics: dict) -> dict:
        """Return clip((NC_i - mu_i) / s_i, -3, 3) for each NC metric."""

    def composite_distance(self, metrics: dict) -> float:
        """Return D: the equal-weight average of the four standardized metrics."""

    def save(self, path: str) -> None:
        """Write the frozen statistics to disk (e.g. JSON)."""

    @classmethod
    def load(cls, path: str) -> "NCStandardizer":
        """Load frozen statistics written by save."""
