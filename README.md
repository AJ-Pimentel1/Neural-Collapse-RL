# Neural-Collapse-RL

Adversarial reinforcement learning agents as a causal probe of **neural collapse** (NC).

Course project for Sequential Decision Making, Yale University (Adrian Pimentel).

## Overview

Late in training, deep classifiers tend to organize their penultimate-layer features into a highly symmetric configuration (Papyan, Han & Donoho, 2020):

| Metric | Property | What we measure |
|---|---|---|
| NC1 | Variability collapse | $\mathrm{Tr}(\Sigma_W \Sigma_B^{\dagger})/C$ |
| NC2 | Convergence to a simplex ETF | Deviation of the centered class-mean Gram matrix from the simplex-ETF Gram matrix |
| NC3 | Self-duality | Misalignment between classifier weights and centered class means |
| NC4 | Nearest class-center decisions | Disagreement rate between the classifier and the nearest-class-mean rule |

NC is usually observed as an *outcome* of standard training. This project asks **which aspects of optimization cause it**, by treating training itself as an environment and letting RL agents control the optimizer's hyperparameters.

Two agents each control their own copy of the same network:

- **NC-Expression agent:** rewarded for driving the network toward collapse.
- **NC-Suppression agent:** rewarded for keeping the network away from collapse.

Comparing what the two discover, and replaying their learned schedules with plain SGD (no RL in the loop), tests whether a specific optimization schedule causes or prevents NC, and whether NC is necessary for fitting the data at all.

## Method

One RL episode is one full training run from random initialization. The agent acts every $K$ epochs.

**Action** (applied for the next $K$ epochs): learning rate, learning-rate schedule shape within the interval (constant / cosine / linear decay), SGD momentum, batch size, and weight decay.

**Held fixed:** architecture, depth and width (small MLP), optimizer family (SGD with momentum), cross-entropy loss, and a class-balanced dataset with uniform sampling. Class balance is fixed because the simplex-ETF structure of NC is known to depend on it (Dang et al., ICML 2024).

**State:** training accuracy, the four standardized NC metrics, mean feature norm, classifier-weight norm, the previous action, and progress through training $t/H$.

**Standardized collapse distance:** each NC metric is z-scored using statistics from a reference pool of runs (the SGD baseline plus runs with random hyperparameters), so no metric dominates:

$$\widetilde{NC}_i = \mathrm{clip}\left(\frac{NC_i - \mu_i}{s_i}, -3, 3\right), \qquad D_t = \frac{1}{4}\sum_{i=1}^{4}\widetilde{NC}_i$$

**SGD baseline:** momentum 0.9, batch size 128, weight decay $5\times10^{-4}$, and a constant learning rate divided by 10 at $T/3$ and $2T/3$. The initial learning rate is picked by a sweep on training accuracy (not NC). The baseline is run on 10 seeds, and its seed-averaged curves $\overline{\text{Acc}}^{SGD}_t$ and $\overline{D}^{SGD}_t$ are frozen before agent training.

**Reward**, measured against the seed-averaged baseline at the same epoch:

$$r_t = \tfrac12\left(\text{Acc}_{t+1} - \overline{\text{Acc}}^{SGD}_{t+1}\right) + \tfrac12\, s\left(\overline{D}^{SGD}_{t+1} - D_{t+1}\right)$$

with $s = +1$ for the Expression agent and $s = -1$ for the Suppression agent. A policy that copies SGD earns zero return in expectation.

**Termination:** after a warm-up, an episode ends with a penalty if training accuracy falls more than 20 percentage points below the baseline at the same epoch.

All metrics are computed on the training set; there is no held-out split, since the question is about NC emerging on the training data.

## Repository layout

| File | Contents | Status |
|---|---|---|
| `toy_dataset_gen.py` | Balanced, heavily overlapping 10-class Gaussian mixture (500 per class) and a `tf.data` pipeline builder | Implemented |
| `model_create.py` | Fixed 3 x 256 ReLU MLP with separate feature extractor and linear classifier | Implemented |
| `eval.py` | NC1-NC4, loss/accuracy, norms (implemented); `NCStandardizer` | Partial |
| `sgd_trainer.py` | SGD-with-momentum trainer (PyTorch-style momentum and weight decay), per-epoch CSV logging | Implemented |
| `train_baseline.py` | Initial-LR sweep, 10-seed SGD baseline, seed-averaged curves | Implemented |
| `train_agents.py` | Reference pool, `NCTrainingEnv`, PPO agents, policy extraction, search comparison | Skeleton |
| `visualization.py` | Plots of metric trajectories and feature projections | Empty |

## Setup

```bash
conda env create -f environment.yml
conda activate ncollapse
```

Check the dataset calibration (logistic regression should reach at most ~50% training accuracy):

```bash
python toy_dataset_gen.py
```

Run the SGD baseline (200 epochs per run; finished runs are skipped if a command is rerun):

```bash
python train_baseline.py sweep       # LR sweep -> results/lr_sweep/ (summary.csv, selected_lr.json)
python train_baseline.py baseline    # 10 seeds at the selected LR -> results/baseline/
```

Each run writes a CSV with one row per epoch: the hyperparameters used that epoch, training loss and accuracy, raw NC1-NC4, and feature and weight norms. `results/baseline/baseline_mean.csv` holds the per-epoch mean and std across seeds.

## Planned experiments

1. **Baseline and reference pool:** standard SGD to confirm NC emerges on the toy data; random-hyperparameter runs to fit the metric standardizer.
2. **Competing agents:** train the Expression and Suppression agents with PPO and compare final accuracy and NC metrics against the baseline.
3. **Policy extraction:** replay each agent's learned schedule as fixed SGD over several seeds, with per-hyperparameter ablations.
4. **Follow-up:** repeat with a different optimizer family (e.g. Adam/AdamW).

## References

- V. Papyan, X.Y. Han, D.L. Donoho. *Prevalence of neural collapse during the terminal phase of deep learning training.* PNAS, 2020.
- H. Dang, T. Tran Huu, T.M. Nguyen, N. Ho. *Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Features Model.* ICML 2024.
