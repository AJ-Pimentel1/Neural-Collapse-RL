"""Training loops, the training-as-MDP environment, and RL agent training.

Fixed across all runs: architecture, optimizer family (SGD with momentum),
cross-entropy loss, balanced data with uniform sampling, total epoch budget.
Controlled by the agents: learning rate + schedule shape, momentum, batch
size, weight decay.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration and actions
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Experiment-wide settings.

    Fields to define: total_epochs, K (epochs per decision interval),
    warmup_epochs E_w (e.g. 40), divergence_delta (e.g. 0.20),
    termination_penalty, beta (accuracy vs. NC reward weight), baseline
    hyperparameters, action ranges (lr, momentum, weight decay, batch sizes),
    device, seed, output paths.
    """


class ScheduleShape(Enum):
    """Within-interval learning-rate schedule shapes: CONSTANT, COSINE, LINEAR."""


@dataclass
class HyperparamAction:
    """One agent action, applied for the next K epochs.

    Fields to define: lr (eta_t), schedule (sigma_t), momentum (rho_t),
    batch_size (B_t), weight_decay (lambda_t).
    """


def decode_action(raw_action, config: TrainConfig) -> HyperparamAction:
    """Map the policy's raw output to a HyperparamAction.

    Continuous parts (lr, momentum, weight decay) are squashed/clipped into
    their ranges; discrete parts (schedule shape, batch size) are indexed from
    their option lists.
    """


def sample_random_action(config: TrainConfig, rng: np.random.Generator) -> HyperparamAction:
    """Sample a uniformly random action from the action space (used for the reference pool)."""


# ---------------------------------------------------------------------------
# Plain SGD training
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, action: HyperparamAction) -> torch.optim.SGD:
    """Create an SGD-with-momentum optimizer from the action's lr, momentum, and weight decay."""


def apply_action_to_optimizer(optimizer: torch.optim.SGD, action: HyperparamAction) -> None:
    """Update lr, momentum, and weight decay in place in each param group.

    Done in place (rather than rebuilding the optimizer) so momentum buffers
    persist across decision intervals.
    """


def build_interval_scheduler(optimizer: torch.optim.SGD, action: HyperparamAction, steps_in_interval: int):
    """Return a per-step LR scheduler implementing the action's schedule shape over one K-epoch interval."""


def train_one_epoch(model: nn.Module, loader, optimizer, scheduler, device) -> float:
    """Run one epoch of cross-entropy SGD, stepping the scheduler each batch; return mean training loss."""


def train_interval(model, dataset, optimizer, action: HyperparamAction, config: TrainConfig, rng) -> float:
    """Train for K epochs under one action.

    Rebuild the DataLoader for the action's batch size, apply the action to the
    optimizer, build the interval scheduler, and call train_one_epoch K times.
    Return the last epoch's loss.
    """


def run_fixed_schedule(schedule: list[HyperparamAction], config: TrainConfig, seed: int) -> list[dict]:
    """Train a fresh model with a fixed, non-adaptive sequence of actions (one per interval).

    Evaluate NC metrics and accuracy every K epochs and return the list of
    checkpoint dicts. Used for the baseline, the reference pool, and policy
    extraction replay.
    """


def run_baseline(config: TrainConfig, seed: int) -> list[dict]:
    """Run standard SGD with the fixed baseline hyperparameters for the full budget.

    Returns the checkpoint trajectory; its training-accuracy curve is the
    reference for the divergence-based termination rule.
    """


def collect_reference_pool(config: TrainConfig, n_runs: int, seed: int) -> list[dict]:
    """Collect checkpoints from the baseline plus n_runs runs with random action schedules.

    The pooled records are used to fit the NCStandardizer.
    """


# ---------------------------------------------------------------------------
# Training as an MDP
# ---------------------------------------------------------------------------

class AgentMode(Enum):
    """Which objective the agent optimizes: EXPRESSION (more NC) or SUPPRESSION (less NC)."""


class NCTrainingEnv:
    """Gym-style environment where one episode is one full training run.

    Each step applies one action for K epochs, then evaluates the network and
    returns (state, reward, done, info).
    """

    def __init__(self, config: TrainConfig, mode: AgentMode, dataset, standardizer, baseline_acc_curve: np.ndarray):
        """Store the config, objective mode, dataset, frozen NCStandardizer, and baseline accuracy-per-epoch curve."""

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new episode from freshly initialized weights.

        Build the model and optimizer, evaluate the initial checkpoint, reset
        the epoch counter and previous metrics, and return the initial state.
        """

    def step(self, raw_action) -> tuple[np.ndarray, float, bool, dict]:
        """Decode the action, train for K epochs, evaluate, compute reward, and check termination.

        info should include the raw metrics, D_NC, and the decoded action for
        logging and policy extraction.
        """

    def _build_state(self, metrics: dict, prev_action: HyperparamAction) -> np.ndarray:
        """Assemble s_t = [loss, acc, standardized NC1-NC4, ||f||, ||W||, previous eta, rho, B, lambda]."""

    def _compute_reward(self, metrics: dict, prev_metrics: dict) -> float:
        """Return beta * dAcc + (1 - beta) * (-/+ dD_NC).

        The sign on dD_NC is negative for EXPRESSION (reward decreasing D_NC)
        and positive for SUPPRESSION (reward increasing D_NC).
        """

    def _check_divergence(self, metrics: dict) -> bool:
        """Return True if, after warmup_epochs, training accuracy is more than
        divergence_delta below the baseline's accuracy at the same epoch."""

    def _is_episode_over(self) -> bool:
        """Return True once the total epoch budget has been used."""


# ---------------------------------------------------------------------------
# RL agent
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Policy and value network for a mixed action space.

    Shared trunk over the state; Gaussian heads for the continuous actions
    (lr, momentum, weight decay), categorical heads for the discrete actions
    (schedule shape, batch size), and a scalar value head.
    """

    def __init__(self, state_dim: int, config: TrainConfig):
        """Build the trunk, action heads, and value head."""

    def forward(self, state: torch.Tensor):
        """Return the action distributions and the state value."""

    def act(self, state: torch.Tensor, deterministic: bool = False):
        """Sample (or take the mode of) an action; return action, log-prob, value."""

    def evaluate_actions(self, states: torch.Tensor, actions):
        """Return log-probs, entropies, and values for a batch of stored actions (for the PPO update)."""


class RolloutBuffer:
    """Stores transitions from one or more episodes for a PPO update."""

    def __init__(self):
        """Initialize empty lists for states, actions, log-probs, rewards, values, dones."""

    def add(self, state, action, log_prob, reward, value, done) -> None:
        """Append one transition."""

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute discounted returns and GAE advantages over the stored transitions."""

    def clear(self) -> None:
        """Empty the buffer after an update."""


class PPOAgent:
    """PPO learner that wraps an ActorCritic."""

    def __init__(self, state_dim: int, config: TrainConfig):
        """Build the ActorCritic, optimizer, and PPO hyperparameters (clip, epochs, minibatch size, entropy coef)."""

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Return (raw_action, log_prob, value) for an environment state."""

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run clipped-objective PPO updates on the buffer; return loss statistics for logging."""

    def save(self, path: str) -> None:
        """Save the policy weights."""

    def load(self, path: str) -> None:
        """Load policy weights."""


def train_agent(env: NCTrainingEnv, agent: PPOAgent, n_episodes: int, config: TrainConfig) -> list[dict]:
    """Main RL loop: roll out episodes, fill the buffer, run PPO updates, and log per-episode summaries."""


# ---------------------------------------------------------------------------
# Policy extraction
# ---------------------------------------------------------------------------

def extract_schedule(env: NCTrainingEnv, agent: PPOAgent, seed: int) -> list[HyperparamAction]:
    """Roll out the trained policy deterministically for one episode and record the action at each interval."""


def ablate_schedule(schedule: list[HyperparamAction], keep: set[str], config: TrainConfig) -> list[HyperparamAction]:
    """Return a copy of schedule where only the hyperparameters in keep follow the
    extracted schedule and all others are reset to baseline values."""


def replay_and_compare(schedule: list[HyperparamAction], config: TrainConfig, seeds: list[int]) -> dict:
    """Replay a fixed schedule with plain SGD (no RL) over several seeds and
    summarize final accuracy and NC metrics versus the baseline."""


def main():
    """Parse arguments and run: baseline -> reference pool / standardizer fit ->
    train expression and suppression agents -> policy extraction and ablations."""


if __name__ == "__main__":
    main()
