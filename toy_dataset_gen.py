"""Class-balanced, overlapping Gaussian mixture with equidistant class centers.

The dataset is balanced (equal samples per class) because the simplex-ETF
structure of NC is contingent on balanced classes. There is no held-out split:
the question is about NC emerging on the training data, so every metric is
computed on the training set.
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs


NUM_CLASSES = 10
N_PER_CLASS = 500  # samples per class (N = 5,000)
N_FEATURES = 20

# Class c is centered at CENTER_RADIUS * e_c, so every pair of centers is the
# same distance apart (sqrt(2) * CENTER_RADIUS) and every pair of classes
# overlaps equally. Randomly placed centers (make_blobs' center_box) were
# unevenly spaced, so the class means kept a lopsided geometry that works
# against a simplex ETF (NC2) and against variability collapse (NC1).
# Calibrate CENTER_RADIUS so that (1) a linear classifier cannot fit the data
# (logistic regression ~82% training accuracy at 2.5) and (2) the SGD baseline
# reaches 100% training accuracy and keeps collapsing afterwards.
# Run this file to check criterion (1).
CENTER_RADIUS = 2.5
CENTERS = CENTER_RADIUS * np.eye(NUM_CLASSES, N_FEATURES)

# An int n_samples is split evenly across centers, so classes are balanced.
X, y = make_blobs(
    n_samples = NUM_CLASSES * N_PER_CLASS,
    n_features = N_FEATURES,
    centers = CENTERS,
    cluster_std = 1.0,
    random_state = 42
)
X, y = X.astype("float32"), y.astype("int32")


def build_dataset(batch_size, shuffle=True, seed=42):
    """Batched tf.data pipeline over (X, y).

    Training code rebuilds this every epoch with a seed derived from the run
    seed and the epoch, so minibatch order is reproducible and differs across
    epochs. Use shuffle=False for a fixed-order pass.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    # Calibration check for criterion (1): a linear classifier must not fit the data.
    from sklearn.linear_model import LogisticRegression

    print("data:", X.shape, "per class:", np.bincount(y))
    logreg = LogisticRegression(max_iter=2000).fit(X, y)
    print(f"logistic regression train acc: {logreg.score(X, y):.3f} (target ~0.82)")
