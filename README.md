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

**State:** loss, accuracy, the four standardized NC metrics, feature and classifier-weight norms, and the current hyperparameters.

**Standardized collapse distance:** each NC metric is z-scored using statistics from a reference pool of runs (the SGD baseline plus runs with random hyperparameters), so no metric dominates:

$$\widetilde{NC}_i = \frac{NC_i - \mu_i}{s_i}, \qquad D_{NC} = \frac{1}{4}\sum_{i=1}^{4}\widetilde{NC}_i$$

**Reward**, with a tunable weight $\beta \in [0,1]$ on accuracy:

$$R_t = \beta\,\Delta\text{Acc}_t + (1-\beta)\,(\mp\Delta D_{NC})$$

($-$ for the Expression agent, $+$ for the Suppression agent).

**Termination:** after a warm-up (e.g. 40 epochs), an episode ends with a penalty if training accuracy falls more than a threshold (e.g. 20 percentage points) below the SGD baseline at the same epoch.

All metrics are computed on the training set; there is no held-out split, since the question is about NC emerging on the training data.

## Repository layout

| File | Contents | Status |
|---|---|---|
| `toy_dataset_gen.py` | Balanced 10-class Gaussian-blob dataset (`make_blobs`) and a `tf.data` pipeline builder | Implemented |
| `model_create.py` | Fixed MLP with separate feature extractor and linear classifier | Skeleton |
| `eval.py` | NC1-NC4, loss/accuracy, norms, and the `NCStandardizer` | Skeleton |
| `train.py` | SGD training loop, `NCTrainingEnv`, PPO agent, policy extraction and ablations | Skeleton |
| `visualization.py` | Plots of metric trajectories and feature projections | Empty |

## Setup

```bash
conda env create -f environment.yml
conda activate ncollapse
```

Sanity-check the dataset pipeline:

```bash
python toy_dataset_gen.py
```

## Planned experiments

1. **Baseline and reference pool:** standard SGD to confirm NC emerges on the toy data; random-hyperparameter runs to fit the metric standardizer.
2. **Competing agents:** train the Expression and Suppression agents with PPO and compare final accuracy and NC metrics against the baseline.
3. **Policy extraction:** replay each agent's learned schedule as fixed SGD over several seeds, with per-hyperparameter ablations.
4. **Follow-up:** repeat with a different optimizer family (e.g. Adam/AdamW).

## References

- V. Papyan, X.Y. Han, D.L. Donoho. *Prevalence of neural collapse during the terminal phase of deep learning training.* PNAS, 2020.
- H. Dang, T. Tran Huu, T.M. Nguyen, N. Ho. *Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Features Model.* ICML 2024.
