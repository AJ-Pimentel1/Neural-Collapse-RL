"""SGD baseline: initial-LR sweep, multi-seed baseline runs, and seed-averaged curves.

Recipe (standard in the NC literature): momentum 0.9, batch size 128, weight
decay 5e-4, constant LR divided by 10 at T/3 and 2T/3. The initial LR is
chosen by a sweep on training accuracy, never on the NC metrics, so the
yardstick is not biased toward or against collapse.

Usage:
    python train_baseline.py sweep              # LR sweep -> results/lr_sweep/
    python train_baseline.py baseline           # S seeds at the selected LR -> results/baseline/
    python train_baseline.py aggregate          # recompute baseline_mean.csv from finished seeds

Finished runs (a CSV with every epoch) are skipped, so an interrupted command
can simply be rerun.
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import tensorflow as tf

from sgd_trainer import CSV_FIELDS, HyperparamAction, read_csv, run_fixed_schedule

RESULTS_DIR = "results"
SWEEP_DIR = os.path.join(RESULTS_DIR, "lr_sweep")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")

EPOCHS = 200
SWEEP_LRS = [0.003, 0.01, 0.03, 0.1, 0.3]
SWEEP_SEEDS = [100, 101, 102]         # disjoint from the baseline seeds
BASELINE_SEEDS = list(range(10))      # S = 10


def step_decay_schedule(base_lr, epochs):
    """Baseline recipe: constant LR, divided by 10 after T/3 and again after 2T/3 epochs."""
    def schedule(epoch):
        drops = (epoch > epochs // 3) + (epoch > 2 * epochs // 3)
        return HyperparamAction(lr=base_lr * 0.1 ** drops, schedule="constant",
                                momentum=0.9, batch_size=128, weight_decay=5e-4)
    return schedule


def is_complete(csv_path, epochs):
    """True if csv_path holds a finished run: rows for epochs 0..T, or a run that diverged."""
    if not os.path.exists(csv_path):
        return False
    rows = read_csv(csv_path)
    return len(rows) == epochs + 1 or not math.isfinite(rows[-1]["loss"])


def run_seeds(base_lr, seeds, epochs, out_dir, prefix):
    """Run the baseline recipe at base_lr for each seed, skipping finished runs."""
    for i, seed in enumerate(seeds, 1):
        path = os.path.join(out_dir, f"{prefix}seed{seed}.csv")
        if is_complete(path, epochs):
            print(f"skip {path} (complete)")
            continue
        run_fixed_schedule(step_decay_schedule(base_lr, epochs), seed, epochs, path,
                           desc=f"lr={base_lr:g} seed={seed} [{i}/{len(seeds)}]")


def epochs_to_fit(rows):
    """First epoch with 100% training accuracy, or inf if never reached."""
    return next((r["epoch"] for r in rows if r["acc"] >= 1.0), math.inf)


def sweep(args):
    summary = []
    for lr in args.lrs:
        run_seeds(lr, args.seeds, args.epochs, SWEEP_DIR, prefix=f"lr{lr:g}_")
        runs = [read_csv(os.path.join(SWEEP_DIR, f"lr{lr:g}_seed{s}.csv")) for s in args.seeds]
        # A diverged run stops early and counts as 0 final accuracy.
        final_accs = [r[-1]["acc"] if len(r) == args.epochs + 1 else 0.0 for r in runs]
        summary.append({"lr": lr,
                        "mean_final_acc": float(np.mean(final_accs)),
                        "mean_epochs_to_fit": float(np.mean([epochs_to_fit(r) for r in runs])),
                        "n_fit": sum(epochs_to_fit(r) < math.inf for r in runs)})

    # Highest mean final training accuracy; ties broken by fewest epochs to 100% train accuracy.
    best = max(summary, key=lambda s: (round(s["mean_final_acc"], 4), -s["mean_epochs_to_fit"]))
    with open(os.path.join(SWEEP_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(SWEEP_DIR, "selected_lr.json"), "w") as f:
        json.dump({"lr": best["lr"], "criterion": "max mean final train acc, then min epochs to fit",
                   "epochs": args.epochs, "seeds": args.seeds}, f, indent=2)
    for s in summary:
        print(s)
    print(f"selected lr = {best['lr']:g} -> {os.path.join(SWEEP_DIR, 'selected_lr.json')}")


def aggregate(seeds, epochs):
    """Write baseline_mean.csv: per-epoch mean and std across seeds of every numeric column.

    acc_mean is the frozen Acc^SGD_t curve. D^SGD_t is computed from the raw NC
    columns once the standardizer is fit on the reference pool.
    """
    runs = [read_csv(os.path.join(BASELINE_DIR, f"seed{s}.csv")) for s in seeds]
    runs = [r for r in runs if len(r) == epochs + 1]
    numeric = [k for k in CSV_FIELDS if k not in ("seed", "epoch", "progress", "schedule")]
    fields = ["epoch", "progress", "n_seeds"] + [f"{k}_{stat}" for k in numeric for stat in ("mean", "std")]
    path = os.path.join(BASELINE_DIR, "baseline_mean.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in range(epochs + 1):
            row = {"epoch": e, "progress": e / epochs, "n_seeds": len(runs)}
            for k in numeric:
                vals = np.array([r[e][k] for r in runs], dtype=float)
                row[f"{k}_mean"] = np.nanmean(vals) if not np.all(np.isnan(vals)) else ""
                row[f"{k}_std"] = np.nanstd(vals) if not np.all(np.isnan(vals)) else ""
            writer.writerow(row)
    print(f"wrote {path} from {len(runs)}/{len(seeds)} complete seeds")


def baseline(args):
    lr = args.lr
    if lr is None:
        with open(os.path.join(SWEEP_DIR, "selected_lr.json")) as f:
            lr = json.load(f)["lr"]
    print(f"baseline lr = {lr:g}, seeds = {args.seeds}")
    run_seeds(lr, args.seeds, args.epochs, BASELINE_DIR, prefix="")
    aggregate(args.seeds, args.epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sweep", help="initial-LR sweep on training accuracy")
    p.add_argument("--lrs", type=float, nargs="+", default=SWEEP_LRS)
    p.add_argument("--seeds", type=int, nargs="+", default=SWEEP_SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p = sub.add_parser("baseline", help="multi-seed baseline at the selected LR")
    p.add_argument("--lr", type=float, default=None, help="default: results/lr_sweep/selected_lr.json")
    p.add_argument("--seeds", type=int, nargs="+", default=BASELINE_SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p = sub.add_parser("aggregate", help="recompute baseline_mean.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=BASELINE_SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    tf.config.experimental.enable_op_determinism()
    {"sweep": sweep, "baseline": baseline, "aggregate": lambda a: aggregate(a.seeds, a.epochs)}[args.command](args)


if __name__ == "__main__":
    main()
