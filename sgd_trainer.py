"""SGD-with-momentum training core shared by the baseline and the agent environment.

Fixed across all runs: architecture, optimizer family (SGD with momentum),
cross-entropy loss, balanced data with uniform sampling. The hyperparameters
in HyperparamAction are the only things that vary.
"""

import csv
import math
import os
from dataclasses import asdict, dataclass

import tensorflow as tf
from tqdm import tqdm

from eval import evaluate_checkpoint
from model_create import MLPConfig, build_model
from toy_dataset_gen import NUM_CLASSES, X, build_dataset, y


@dataclass
class HyperparamAction:
    """Hyperparameters applied for one epoch (or one K-epoch decision interval)."""
    lr: float
    schedule: str = "constant"  # within-interval shape: "constant", "cosine", or "linear"
    momentum: float = 0.9
    batch_size: int = 128
    weight_decay: float = 5e-4


def lr_at_step(base_lr, schedule, step, total_steps):
    """Learning rate at a step within an interval: constant, cosine decay, or linear decay from base_lr."""
    frac = step / total_steps
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        return base_lr * 0.5 * (1 + math.cos(math.pi * frac))
    if schedule == "linear":
        return base_lr * (1 - frac)
    raise ValueError(f"unknown schedule shape {schedule!r}")


class SGDTrainer:
    """SGD with heavy-ball momentum, matching torch.optim.SGD (dampening 0, coupled weight decay):

        g <- grad + wd * theta;   buf <- rho * buf + g;   theta <- theta - lr * buf

    Momentum and weight decay live in tf.Variables and the learning rate is a
    step argument, so all three can change between steps without retracing the
    compiled step, and the momentum buffers persist across changes.
    """

    def __init__(self, model):
        self.model = model
        self.momentum = tf.Variable(0.0, trainable=False)
        self.weight_decay = tf.Variable(0.0, trainable=False)
        self.buffers = [tf.Variable(tf.zeros(v.shape, v.dtype), trainable=False)
                        for v in model.trainable_variables]
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def apply_action(self, action: HyperparamAction):
        self.momentum.assign(action.momentum)
        self.weight_decay.assign(action.weight_decay)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([], tf.float32)])
    def train_step(self, x, y, lr):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(y, self.model(x, training=True))
        grads = tape.gradient(loss, self.model.trainable_variables)
        for v, g, buf in zip(self.model.trainable_variables, grads, self.buffers):
            buf.assign(self.momentum * buf + g + self.weight_decay * v)
            v.assign_sub(lr * buf)
        return loss

    def train_epoch(self, action: HyperparamAction, shuffle_seed, interval_epoch=0, interval_epochs=1):
        """Run one epoch under action.

        The schedule shape spans interval_epochs epochs; interval_epoch is this
        epoch's position in that interval. The baseline uses one-epoch
        constant intervals.
        """
        self.apply_action(action)
        steps_per_epoch = math.ceil(len(X) / action.batch_size)
        total_steps = interval_epochs * steps_per_epoch
        step = interval_epoch * steps_per_epoch
        for xb, yb in build_dataset(action.batch_size, shuffle=True, seed=shuffle_seed):
            lr = lr_at_step(action.lr, action.schedule, step, total_steps)
            self.train_step(xb, yb, tf.constant(lr, tf.float32))
            step += 1


CSV_FIELDS = ["seed", "epoch", "progress",
              # hyperparameters used during this epoch, i.e. a_{t-1} for the state at this checkpoint
              "lr", "schedule", "momentum", "batch_size", "weight_decay",
              # training-set state (raw NC metrics; standardized ones are computed later)
              "loss", "acc", "nc1", "nc2", "nc3", "nc4", "feature_norm", "weight_norm"]


def checkpoint_row(model, seed, epoch, epochs, action: HyperparamAction | None):
    """Evaluate the model and return one CSV row for this epoch."""
    metrics = evaluate_checkpoint(model, X, y, NUM_CLASSES)
    hparams = asdict(action) if action is not None else {k: "" for k in CSV_FIELDS[3:8]}
    return {"seed": seed, "epoch": epoch, "progress": epoch / epochs, **hparams, **metrics}


def run_fixed_schedule(schedule_fn, seed, epochs, csv_path, desc="", config: MLPConfig = MLPConfig()):
    """Train a fresh model with a fixed, non-adaptive per-epoch schedule and log every epoch to CSV.

    schedule_fn(epoch) returns the HyperparamAction for epoch 1..epochs. Row 0
    is the initialization (no action yet). Stops early if the loss diverges.
    Used for the LR sweep, the baseline, the reference pool, and replaying
    extracted policies. Returns the list of rows.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    model = build_model(config, seed)
    trainer = SGDTrainer(model)
    rows = [checkpoint_row(model, seed, 0, epochs, None)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(rows[0])
        bar = tqdm(range(1, epochs + 1), desc=desc, leave=True)
        for epoch in bar:
            action = schedule_fn(epoch)
            trainer.train_epoch(action, shuffle_seed=seed * 100_003 + epoch)
            row = checkpoint_row(model, seed, epoch, epochs, action)
            rows.append(row)
            writer.writerow(row)
            f.flush()
            bar.set_postfix(lr=f"{action.lr:.2g}", loss=f"{row['loss']:.3f}", acc=f"{row['acc']:.3f}",
                            nc1=f"{row['nc1']:.3g}", nc4=f"{row['nc4']:.3f}")
            if not math.isfinite(row["loss"]):
                bar.write(f"{desc}: loss diverged at epoch {epoch}, stopping run")
                break
    return rows


def read_csv(path):
    """Read a run CSV into a list of dicts with numeric fields converted to float."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            if k != "schedule":
                row[k] = float(v) if v != "" else float("nan")
    return rows
    