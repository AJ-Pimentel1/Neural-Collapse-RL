"""Training-as-MDP environment, PPO agents, policy extraction, and the search comparison.

Fixed across all runs: architecture, optimizer family (SGD with momentum),
cross-entropy loss, balanced data with uniform sampling, total epoch budget.
Controlled by the agents: learning rate + schedule shape, momentum, batch
size, weight decay. SGD training itself lives in sgd_trainer.py; the frozen
baseline curves come from train_baseline.py.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import tensorflow as tf

from sgd_trainer import HyperparamAction


# ---------------------------------------------------------------------------
# Configuration and actions
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Experiment-wide settings.

    Fields to define: total_epochs T, K (epochs per decision interval; H = T/K),
    warmup_epochs E_w, divergence_delta (0.20), termination_penalty, action
    ranges (lr [1e-3, 1] log scale, momentum [0, 0.99], weight decay
    [1e-6, 1e-2] log scale, batch sizes {32, ..., 512}), seed, output paths.
    """


def decode_action(raw_action, config: TrainConfig) -> HyperparamAction:
    """Map the policy's raw output to a HyperparamAction.

    Continuous parts are squashed into their ranges (log scale for lr and
    weight decay); discrete parts (schedule shape: "constant", "cosine",
    "linear"; batch size) are indexed from their option lists.
    """


def sample_random_action(config: TrainConfig, rng: np.random.Generator) -> HyperparamAction:
    """Sample a uniformly random action from the action space (used for the reference pool)."""


# ---------------------------------------------------------------------------
# Reference pool
# ---------------------------------------------------------------------------

def collect_reference_pool(config: TrainConfig, n_runs: int, seed: int) -> list[dict]:
    """Run n_runs episodes with uniformly random actions (sample_random_action per
    interval) via sgd_trainer.run_fixed_schedule, and pool their checkpoints with
    the baseline seeds' checkpoints (results/baseline/seed*.csv).

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

    def __init__(self, config: TrainConfig, mode: AgentMode, standardizer, baseline_mean_csv: str):
        """Store the config, objective mode, frozen NCStandardizer, and the frozen
        seed-averaged baseline curves Acc^SGD_t and D^SGD_t (from baseline_mean.csv)."""

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new episode from freshly initialized weights.

        Build the model and sgd_trainer.SGDTrainer, evaluate the initial
        checkpoint, reset the epoch counter, and return the initial state.
        """

    def step(self, raw_action) -> tuple[np.ndarray, float, bool, dict]:
        """Decode the action, train K epochs with SGDTrainer.train_epoch (passing the
        position within the interval so the schedule shape spans all K epochs),
        evaluate, compute reward, and check termination.

        info should include the raw metrics, D_NC, and the decoded action for
        logging and policy extraction.
        """

    def _build_state(self, metrics: dict, prev_action: HyperparamAction, t: int) -> np.ndarray:
        """Assemble s_t = (Acc_t, standardized NC1-NC4, ||h_t||, ||W_t||, a_{t-1}, t/H)."""

    def _compute_reward(self, metrics: dict, epoch: int) -> float:
        """Return 1/2 (Acc - Acc^SGD) + 1/2 s (D^SGD - D) at this epoch.

        s = +1 for EXPRESSION, -1 for SUPPRESSION. A policy that copies SGD
        earns zero return in expectation. Returns are undiscounted.
        """

    def _check_divergence(self, metrics: dict, epoch: int) -> bool:
        """Return True if, after warmup_epochs, training accuracy is more than
        divergence_delta (20 points) below Acc^SGD at the same epoch."""

    def _is_episode_over(self) -> bool:
        """Return True once the total epoch budget has been used."""


# ---------------------------------------------------------------------------
# RL agent
# ---------------------------------------------------------------------------

class ActorCritic(tf.keras.Model):
    """Policy and value network for a mixed action space.

    Shared trunk over the state; Gaussian heads (mean + learned log-std) for
    the continuous actions (lr, momentum, weight decay), categorical logit
    heads for the discrete actions (schedule shape, batch size), and a scalar
    value head.
    """

    def __init__(self, state_dim: int, config: TrainConfig):
        """Build the trunk, action heads, and value head as Dense layers."""

    def call(self, state: tf.Tensor):
        """Return the action-distribution parameters (Gaussian means/log-stds,
        categorical logits) and the state value."""

    def act(self, state: tf.Tensor, deterministic: bool = False):
        """Sample (or take the mode of) an action; return action, log-prob, value."""

    def evaluate_actions(self, states: tf.Tensor, actions):
        """Return log-probs, entropies, and values for a batch of stored actions (for the PPO update).

        Log-probs and entropies can be written by hand for Gaussian and
        categorical heads, or via tensorflow_probability.distributions.
        """


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
        """Build the ActorCritic, a tf.keras.optimizers.Adam, and PPO hyperparameters
        (clip, epochs, minibatch size, entropy coef, value coef)."""

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Return (raw_action, log_prob, value) for an environment state."""

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run clipped-objective PPO updates on the buffer with tf.GradientTape;
        return loss statistics for logging."""

    def save(self, path: str) -> None:
        """Save the policy weights (model.save_weights)."""

    def load(self, path: str) -> None:
        """Load policy weights (model.load_weights)."""


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


# ---------------------------------------------------------------------------
# Learned schedules versus hyperparameter search
# ---------------------------------------------------------------------------

def run_constant_search(n_runs: int, mode: AgentMode, config: TrainConfig, seeds: list[int]) -> dict:
    """Random search over constant settings held for the whole run, given the same
    number of training runs the RL agent consumed. Select the setting with the
    highest mean return across seeds and record best-so-far vs. runs consumed."""


def main():
    """Parse arguments and run: reference pool / standardizer fit -> train expression
    and suppression agents -> policy extraction, ablations, and the search comparison."""


if __name__ == "__main__":
    main()
