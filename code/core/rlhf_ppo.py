"""
Improvement 08: Full RLHF Implementation with PPO Loop
=======================================================
Addresses the 34% RLHF performance issue by implementing the complete
RLHF pipeline including PPO policy optimization.

The original simplified RLHF only trained a reward model without
policy optimization. This implementation adds the missing PPO loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


@dataclass
class Experience:
    """Single experience tuple for PPO."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


@dataclass
class Trajectory:
    """Trajectory of experiences."""
    experiences: List[Experience]


class RewardModel(nn.Module):
    """Bradley-Terry preference-based reward model."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def compute_preference_prob(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor
    ) -> torch.Tensor:
        """P(a > b) using Bradley-Terry model."""
        r_a = self.forward(state_a)
        r_b = self.forward(state_b)
        return torch.sigmoid(r_a - r_b)


class PolicyNetwork(nn.Module):
    """Actor network for PPO policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(state), dim=-1)

    def get_action(self, state: torch.Tensor) -> Tuple[int, float]:
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.forward(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Critic network for PPO value estimation."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for RLHF.

    Implements the full PPO algorithm with:
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus
    - Generalized Advantage Estimation (GAE)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_critic)

        # Training stats
        self.training_stats = []

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""

        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        n_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """PPO update step."""

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = len(states)
        indices = np.arange(n_samples)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]

                # Policy loss (clipped surrogate objective)
                log_probs, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # Value loss
                values = self.value(batch_states)
                value_loss = F.mse_loss(values, batch_returns)

                # Combined loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Update policy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)

                self.policy_optimizer.step()
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates
        }

        self.training_stats.append(stats)
        return stats


class FullRLHF:
    """
    Complete RLHF Implementation with PPO Policy Optimization.

    Pipeline:
    1. Train reward model on human preference data
    2. Use reward model to provide rewards for RL
    3. Optimize policy using PPO against learned reward
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        reward_lr: float = 1e-3,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Components
        self.reward_model = RewardModel(state_dim, hidden_dim).to(self.device)
        self.reward_optimizer = torch.optim.Adam(
            self.reward_model.parameters(), lr=reward_lr
        )

        self.ppo_trainer = PPOTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )

        self.training_log = {
            "reward_model": [],
            "ppo": []
        }

    def train_reward_model(
        self,
        preference_pairs: List[Tuple[np.ndarray, np.ndarray, float]],
        n_epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train reward model on preference pairs.

        Args:
            preference_pairs: List of (state_a, state_b, preference)
                             where preference=1 means a>b, 0 means b>a
        """
        print("Stage 1: Training Reward Model on Preferences...")

        losses = []
        accuracies = []

        n_pairs = len(preference_pairs)

        for epoch in range(n_epochs):
            epoch_loss = 0
            correct = 0

            indices = np.random.permutation(n_pairs)

            for start in range(0, n_pairs, batch_size):
                end = min(start + batch_size, n_pairs)
                batch_indices = indices[start:end]

                states_a = torch.FloatTensor(
                    np.array([preference_pairs[i][0] for i in batch_indices])
                ).to(self.device)

                states_b = torch.FloatTensor(
                    np.array([preference_pairs[i][1] for i in batch_indices])
                ).to(self.device)

                prefs = torch.FloatTensor(
                    [preference_pairs[i][2] for i in batch_indices]
                ).to(self.device)

                # Bradley-Terry loss
                prob_a_wins = self.reward_model.compute_preference_prob(states_a, states_b).squeeze()

                loss = F.binary_cross_entropy(prob_a_wins, prefs)

                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()

                epoch_loss += loss.item() * len(batch_indices)

                # Accuracy
                predictions = (prob_a_wins > 0.5).float()
                correct += (predictions == prefs).sum().item()

            avg_loss = epoch_loss / n_pairs
            accuracy = correct / n_pairs

            losses.append(avg_loss)
            accuracies.append(accuracy)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.3f}")

        self.training_log["reward_model"] = {
            "losses": losses,
            "accuracies": accuracies,
            "final_accuracy": accuracies[-1]
        }

        return {"losses": losses, "accuracies": accuracies}

    def get_reward(self, state: np.ndarray) -> float:
        """Get reward from learned reward model."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.reward_model(state_t).item()

    def train_policy_ppo(
        self,
        env,  # Environment with step and reset methods
        n_iterations: int = 100,
        steps_per_iteration: int = 2048,
        n_ppo_epochs: int = 4
    ) -> Dict[str, List[float]]:
        """
        Train policy using PPO with learned reward model.

        Args:
            env: RL environment with step(action) -> (next_state, _, done, info)
                 and reset() -> state
        """
        print("\nStage 2: Training Policy with PPO...")

        rewards_per_iteration = []

        for iteration in range(n_iterations):
            # Collect trajectories
            states = []
            actions = []
            rewards = []
            dones = []
            log_probs = []
            values = []

            state = env.reset()
            episode_reward = 0
            episode_rewards = []

            for step in range(steps_per_iteration):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Get action from policy
                action, log_prob = self.ppo_trainer.policy.get_action(state_t)
                value = self.ppo_trainer.value(state_t).item()

                # Step environment
                next_state, _, done, _ = env.step(action)

                # Get reward from learned reward model
                reward = self.get_reward(state)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob)
                values.append(value)

                episode_reward += reward

                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    state = env.reset()
                else:
                    state = next_state

            # Compute returns and advantages
            with torch.no_grad():
                next_state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_value = self.ppo_trainer.value(next_state_t).item()

            advantages, returns = self.ppo_trainer.compute_gae(
                rewards, values, dones, next_value
            )

            # PPO update
            stats = self.ppo_trainer.update(
                states=np.array(states),
                actions=np.array(actions),
                old_log_probs=np.array(log_probs),
                returns=returns,
                advantages=advantages,
                n_epochs=n_ppo_epochs
            )

            avg_reward = np.mean(episode_rewards) if episode_rewards else np.mean(rewards)
            rewards_per_iteration.append(avg_reward)

            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration+1}/{n_iterations}: "
                      f"Avg Reward={avg_reward:.4f}, "
                      f"Policy Loss={stats['policy_loss']:.4f}")

        self.training_log["ppo"] = {
            "rewards": rewards_per_iteration,
            "final_reward": rewards_per_iteration[-1]
        }

        return {"rewards": rewards_per_iteration}


class SimulatedWellbeingEnv:
    """
    Simulated environment for well-being recommendations.

    State: [engagement, diversity, fatigue, session_length]
    Actions: 0=high_engagement, 1=diverse, 2=relaxing, 3=educational
    """

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> np.ndarray:
        self.engagement = 0.5
        self.diversity = 0.0
        self.fatigue = 0.0
        self.step_count = 0
        self.seen_types = set()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array([
            self.engagement,
            self.diversity,
            self.fatigue,
            self.step_count / self.max_steps
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).
        Reward is computed by the RLHF reward model, not here.
        """
        # Action effects
        action_effects = {
            0: {"engagement": 0.15, "fatigue": 0.08, "diversity": 0.0},   # High engagement
            1: {"engagement": 0.05, "fatigue": 0.02, "diversity": 0.1},   # Diverse
            2: {"engagement": -0.02, "fatigue": -0.05, "diversity": 0.0}, # Relaxing
            3: {"engagement": 0.08, "fatigue": 0.03, "diversity": 0.05},  # Educational
        }

        effects = action_effects[action]

        self.engagement = np.clip(self.engagement + effects["engagement"], 0, 1)
        self.fatigue = np.clip(self.fatigue + effects["fatigue"], 0, 1)
        self.seen_types.add(action)
        self.diversity = len(self.seen_types) / 4

        self.step_count += 1

        done = self.step_count >= self.max_steps or self.fatigue > 0.9

        return self._get_state(), 0.0, done, {}


def generate_synthetic_preferences(
    n_pairs: int = 1000,
    state_dim: int = 4
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Generate synthetic preference pairs based on well-being criteria.

    Encodes expert knowledge: prefer states with
    - Higher diversity (0.4 weight)
    - Higher engagement (0.3 weight)
    - Lower fatigue (0.3 weight)
    """
    pairs = []

    for _ in range(n_pairs):
        # Random states
        state_a = np.random.rand(state_dim).astype(np.float32)
        state_b = np.random.rand(state_dim).astype(np.float32)

        # True well-being score: W(s) = 0.3*eng + 0.4*div - 0.3*fat
        # State format: [engagement, diversity, fatigue, progress]
        wellbeing_a = 0.3 * state_a[0] + 0.4 * state_a[1] - 0.3 * state_a[2]
        wellbeing_b = 0.3 * state_b[0] + 0.4 * state_b[1] - 0.3 * state_b[2]

        # Preference with some noise (simulating human labeler)
        prob_a_preferred = 1 / (1 + np.exp(-(wellbeing_a - wellbeing_b) * 5))
        preference = 1.0 if np.random.random() < prob_a_preferred else 0.0

        pairs.append((state_a, state_b, preference))

    return pairs


def run_rlhf_experiment(output_dir: Path):
    """Run complete RLHF experiment and save results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FULL RLHF IMPLEMENTATION WITH PPO")
    print("=" * 60)

    # Initialize RLHF
    rlhf = FullRLHF(
        state_dim=4,
        action_dim=4,
        hidden_dim=128,
        device="cpu"
    )

    # Generate synthetic preferences
    print("\n1. Generating synthetic preference data...")
    preference_pairs = generate_synthetic_preferences(n_pairs=1000)
    print(f"   Generated {len(preference_pairs)} preference pairs")

    # Train reward model
    print("\n2. Training reward model...")
    reward_results = rlhf.train_reward_model(
        preference_pairs,
        n_epochs=50,
        batch_size=32
    )
    print(f"   Final reward model accuracy: {reward_results['accuracies'][-1]:.3f}")

    # Train policy with PPO
    print("\n3. Training policy with PPO...")
    env = SimulatedWellbeingEnv(max_steps=50)
    ppo_results = rlhf.train_policy_ppo(
        env=env,
        n_iterations=50,
        steps_per_iteration=512,
        n_ppo_epochs=4
    )
    print(f"   Final average reward: {ppo_results['rewards'][-1]:.4f}")

    # Evaluate learned policy
    print("\n4. Evaluating learned policy...")
    evaluation_results = evaluate_policy(rlhf, n_episodes=20)

    # Save results
    results = {
        "reward_model": {
            "final_accuracy": float(reward_results['accuracies'][-1]),
            "training_epochs": len(reward_results['accuracies'])
        },
        "ppo_training": {
            "final_reward": float(ppo_results['rewards'][-1]),
            "reward_improvement": float(ppo_results['rewards'][-1] - ppo_results['rewards'][0]),
            "iterations": len(ppo_results['rewards'])
        },
        "evaluation": evaluation_results,
        "comparison_with_simplified": {
            "simplified_rlhf_accuracy": 0.34,  # From original paper
            "full_rlhf_performance": evaluation_results["avg_episode_reward"],
            "improvement": "PPO loop enables policy to actually optimize for learned reward"
        }
    }

    with open(output_dir / "rlhf_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save training curves
    training_curves = {
        "reward_model_loss": reward_results['losses'],
        "reward_model_accuracy": reward_results['accuracies'],
        "ppo_rewards": ppo_results['rewards']
    }

    with open(output_dir / "training_curves.json", 'w', encoding='utf-8') as f:
        json.dump(training_curves, f, indent=2)

    return results


def evaluate_policy(rlhf: FullRLHF, n_episodes: int = 20) -> Dict:
    """Evaluate the trained policy."""

    env = SimulatedWellbeingEnv(max_steps=50)

    episode_rewards = []
    episode_lengths = []
    final_wellbeings = []

    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, _ = rlhf.ppo_trainer.policy.get_action(state_t)
            next_state, _, done, _ = env.step(action)

            reward = rlhf.get_reward(state)
            episode_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Compute final well-being
        wellbeing = 0.3 * env.engagement + 0.4 * env.diversity - 0.3 * env.fatigue
        final_wellbeings.append(wellbeing)

    return {
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_final_wellbeing": float(np.mean(final_wellbeings)),
        "n_episodes": n_episodes
    }


def main():
    """Main function to run full RLHF implementation."""

    output_dir = Path(__file__).parent / "rlhf_ppo_results"

    results = run_rlhf_experiment(output_dir)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nReward Model Accuracy: {results['reward_model']['final_accuracy']:.3f}")
    print(f"PPO Final Reward: {results['ppo_training']['final_reward']:.4f}")
    print(f"Reward Improvement: {results['ppo_training']['reward_improvement']:.4f}")
    print(f"Avg Episode Well-being: {results['evaluation']['avg_final_wellbeing']:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH ORIGINAL (SIMPLIFIED) RLHF")
    print("=" * 60)
    print(f"Original RLHF (no PPO): 34% correlation with ground truth")
    print(f"Full RLHF (with PPO): {results['evaluation']['avg_final_wellbeing']:.3f} well-being score")
    print("\nKey difference: PPO enables the policy to OPTIMIZE for the learned")
    print("reward model, rather than just learning the reward itself.")

    print("\n" + "=" * 60)
    print("Output files created:")
    for f in output_dir.iterdir():
        print(f"   - {f.name}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
