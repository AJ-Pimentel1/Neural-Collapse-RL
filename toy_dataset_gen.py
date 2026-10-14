"""Class-balanced datasets for the neural collapse RL experiments.

Every dataset produced here is balanced (equal samples per class) because the
simplex-ETF structure of NC is contingent on balanced classes. There is no
held-out split: all metrics are computed on the training set.
"""

import tensorflow as tf
from sklearn.datasets import make_blobs


# An int n_samples is split evenly across centers, so classes are balanced.
X, y = make_blobs(
    n_samples = 20000,
    n_features = 20,
    centers = 10,
    cluster_std = 1.0,
    random_state = 42
)
X, y = X.astype("float32"), y.astype("int32")


def build_dataset(batch_size, shuffle=True, seed=42):
    """Batched tf.data pipeline over (X, y). Rebuilt whenever the agent changes batch size;
    use shuffle=False for the full-pass NC metric evaluation."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    # Quick sanity check: iterate over the dataset and print shapes.
    for x, y in build_dataset(batch_size=32):
        print(x.shape, y.shape)
    
    
