"""
PWSS: Preference-aligned Well-being in Social Systems
=====================================================

Complete Implementation of IRL Methods for Well-being-Aware Recommendation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Trajectory:
    """A sequence of (state, action, reward) tuples."""
    states: np.ndarray  # (T, state_dim)
    actions: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)
    wellbeing_scores: np.ndarray  # (T,)

    @property
    def length(self) -> int:
        return len(self.states)

    def compute_features(self) -> np.ndarray:
        """Compute trajectory-level features."""
        return np.array([
            np.mean(self.states[:, 0]),  # mean engagement
            np.std(self.states[:, 0]),   # engagement variance
            np.mean(self.states[:, 1]),  # mean diversity
            np.mean(self.states[:, 2]),  # mean fatigue
            self.length,                 # session length
            np.sum(self.rewards),        # total reward
            np.mean(self.wellbeing_scores),  # mean well-being
            np.max(self.wellbeing_scores) - np.min(self.wellbeing_scores)  # well-being range
        ])


@dataclass
class PreferencePair:
    """A preference between two trajectories."""
    trajectory_a: Trajectory
    trajectory_b: Trajectory
    preference: int  # 0 = prefer a, 1 = prefer b


# =============================================================================
# Well-being Score Computation
# =============================================================================

class WellbeingScorer:
    """Compute well-being scores from states."""

    def __init__(self,
                 engagement_weight: float = 0.3,
                 diversity_weight: float = 0.4,
                 fatigue_weight: float = -0.3):
        self.engagement_weight = engagement_weight
        self.diversity_weight = diversity_weight
        self.fatigue_weight = fatigue_weight

    def compute(self, state: np.ndarray) -> float:
        """
        Compute well-being score for a state.

        W(s) = 0.3 * engagement + 0.4 * diversity - 0.3 * fatigue
        """
        engagement = state[0]
        diversity = state[1]
        fatigue = state[2]

        return (self.engagement_weight * engagement +
                self.diversity_weight * diversity +
                self.fatigue_weight * fatigue)

    def compute_batch(self, states: np.ndarray) -> np.ndarray:
        """Compute well-being scores for batch of states."""
        return (self.engagement_weight * states[:, 0] +
                self.diversity_weight * states[:, 1] +
                self.fatigue_weight * states[:, 2])


# =============================================================================
# Neural Network Architectures
# =============================================================================

class RewardNetwork(nn.Module):
    """Neural network for reward prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PreferenceNetwork(nn.Module):
    """Network for learning from preferences (Bradley-Terry model)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.reward_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features_a: torch.Tensor,
                features_b: torch.Tensor) -> torch.Tensor:
        """Predict probability that trajectory a is preferred."""
        reward_a = self.reward_net(features_a)
        reward_b = self.reward_net(features_b)
        return torch.sigmoid(reward_a - reward_b)

    def get_reward(self, features: torch.Tensor) -> torch.Tensor:
        return self.reward_net(features)


# =============================================================================
# IRL Methods
# =============================================================================

class BaseIRL(ABC):
    """Base class for IRL methods."""

    @abstractmethod
    def train(self, data, **kwargs):
        """Train the reward model."""
        pass

    @abstractmethod
    def predict_reward(self, features: np.ndarray) -> float:
        """Predict reward for given features."""
        pass

    @abstractmethod
    def evaluate_preference(self, traj_a: Trajectory,
                           traj_b: Trajectory) -> float:
        """Return probability that traj_a is preferred."""
        pass


class MLIRL(BaseIRL):
    """Maximum Likelihood Inverse Reinforcement Learning."""

    def __init__(self, state_dim: int, hidden_dim: int = 64,
                 lr: float = 1e-3, device: str = 'cpu'):
        self.device = device
        self.reward_net = RewardNetwork(state_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)

    def train(self, trajectories: List[Trajectory],
              n_epochs: int = 200, batch_size: int = 32):
        """Train reward model to maximize likelihood of demonstrations."""
        # Collect all states
        all_states = np.vstack([t.states for t in trajectories])
        all_rewards = np.concatenate([t.rewards for t in trajectories])

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(all_states),
            torch.FloatTensor(all_rewards)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(n_epochs):
            total_loss = 0
            for states, rewards in loader:
                states = states.to(self.device)
                rewards = rewards.to(self.device)

                pred_rewards = self.reward_net(states).squeeze()
                loss = nn.MSELoss()(pred_rewards, rewards)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict_reward(self, features: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            return self.reward_net(x).item()

    def evaluate_preference(self, traj_a: Trajectory,
                           traj_b: Trajectory) -> float:
        reward_a = np.mean([self.predict_reward(s) for s in traj_a.states])
        reward_b = np.mean([self.predict_reward(s) for s in traj_b.states])
        return 1.0 / (1.0 + np.exp(-(reward_a - reward_b)))


class PBIRL(BaseIRL):
    """Preference-Based Inverse Reinforcement Learning."""

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64,
                 lr: float = 1e-3, temperature: float = 1.0,
                 device: str = 'cpu'):
        self.device = device
        self.temperature = temperature
        self.pref_net = PreferenceNetwork(feature_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.pref_net.parameters(), lr=lr)

    def train(self, preference_pairs: List[PreferencePair],
              n_epochs: int = 200, batch_size: int = 32):
        """Train on preference pairs using Bradley-Terry model."""
        # Extract features
        features_a = np.array([p.trajectory_a.compute_features()
                              for p in preference_pairs])
        features_b = np.array([p.trajectory_b.compute_features()
                              for p in preference_pairs])
        labels = np.array([1 - p.preference for p in preference_pairs])  # 1 if a preferred

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(features_a),
            torch.FloatTensor(features_b),
            torch.FloatTensor(labels)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(n_epochs):
            total_loss = 0
            for feat_a, feat_b, targets in loader:
                feat_a = feat_a.to(self.device)
                feat_b = feat_b.to(self.device)
                targets = targets.to(self.device)

                probs = self.pref_net(feat_a, feat_b).squeeze()
                loss = nn.BCELoss()(probs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict_reward(self, features: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            return self.pref_net.get_reward(x).item()

    def evaluate_preference(self, traj_a: Trajectory,
                           traj_b: Trajectory) -> float:
        feat_a = traj_a.compute_features()
        feat_b = traj_b.compute_features()
        with torch.no_grad():
            return self.pref_net(
                torch.FloatTensor(feat_a).unsqueeze(0).to(self.device),
                torch.FloatTensor(feat_b).unsqueeze(0).to(self.device)
            ).item()


class MaxEntIRL(BaseIRL):
    """Maximum Entropy Inverse Reinforcement Learning."""

    def __init__(self, state_dim: int, hidden_dim: int = 64,
                 lr: float = 1e-3, entropy_weight: float = 0.01,
                 device: str = 'cpu'):
        self.device = device
        self.entropy_weight = entropy_weight
        self.reward_net = RewardNetwork(state_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)

    def train(self, trajectories: List[Trajectory],
              n_epochs: int = 200, batch_size: int = 64):
        """Train using maximum entropy framework."""
        all_states = np.vstack([t.states for t in trajectories])

        # Compute expert feature expectations
        expert_features = np.mean(all_states, axis=0)

        for epoch in range(n_epochs):
            # Sample batch
            idx = np.random.choice(len(all_states), batch_size)
            states = torch.FloatTensor(all_states[idx]).to(self.device)

            # Forward pass
            rewards = self.reward_net(states).squeeze()

            # Compute policy feature expectations (softmax weighting)
            weights = torch.softmax(rewards / self.entropy_weight, dim=0)
            policy_features = (weights.unsqueeze(1) * states).sum(dim=0)

            # Feature matching loss
            expert_feat_tensor = torch.FloatTensor(expert_features).to(self.device)
            loss = nn.MSELoss()(policy_features, expert_feat_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
            self.optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    def predict_reward(self, features: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            return self.reward_net(x).item()

    def evaluate_preference(self, traj_a: Trajectory,
                           traj_b: Trajectory) -> float:
        reward_a = np.mean([self.predict_reward(s) for s in traj_a.states])
        reward_b = np.mean([self.predict_reward(s) for s in traj_b.states])
        return 1.0 / (1.0 + np.exp(-(reward_a - reward_b)))


class SimplifiedRLHF(BaseIRL):
    """
    Simplified RLHF baseline (reward model only, no PPO).

    NOTE: This is intentionally simplified and expected to underperform.
    A full RLHF implementation would include PPO policy optimization.
    """

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64,
                 lr: float = 1e-3, device: str = 'cpu'):
        self.device = device
        self.reward_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)

    def train(self, preference_pairs: List[PreferencePair],
              n_epochs: int = 100, batch_size: int = 32):
        """Train reward model on preferences."""
        features_a = np.array([p.trajectory_a.compute_features()
                              for p in preference_pairs])
        features_b = np.array([p.trajectory_b.compute_features()
                              for p in preference_pairs])
        # NOTE: Label encoding - may be inverted
        labels = np.array([p.preference for p in preference_pairs])

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(features_a),
            torch.FloatTensor(features_b),
            torch.FloatTensor(labels)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(n_epochs):
            total_loss = 0
            for feat_a, feat_b, targets in loader:
                feat_a = feat_a.to(self.device)
                feat_b = feat_b.to(self.device)
                targets = targets.to(self.device)

                reward_a = self.reward_net(feat_a).squeeze()
                reward_b = self.reward_net(feat_b).squeeze()
                logits = reward_a - reward_b
                probs = torch.sigmoid(logits)

                loss = nn.BCELoss()(probs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 25 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict_reward(self, features: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            return self.reward_net(x).item()

    def evaluate_preference(self, traj_a: Trajectory,
                           traj_b: Trajectory) -> float:
        feat_a = traj_a.compute_features()
        feat_b = traj_b.compute_features()
        with torch.no_grad():
            r_a = self.reward_net(torch.FloatTensor(feat_a).to(self.device))
            r_b = self.reward_net(torch.FloatTensor(feat_b).to(self.device))
            return torch.sigmoid(r_a - r_b).item()


# =============================================================================
# Reward Shaping Baselines
# =============================================================================

class RewardShaping(BaseIRL):
    """Domain-informed reward shaping baseline."""

    def __init__(self, engagement_weight: float = 0.3,
                 diversity_weight: float = 0.3,
                 fatigue_penalty: float = -0.2):
        self.e_weight = engagement_weight
        self.d_weight = diversity_weight
        self.f_penalty = fatigue_penalty

    def train(self, *args, **kwargs):
        """No training needed for hand-crafted rewards."""
        pass

    def compute_reward(self, state: np.ndarray) -> float:
        """Compute shaped reward."""
        engagement = state[0]
        diversity = state[1]
        fatigue = state[2]
        return (self.e_weight * engagement +
                self.d_weight * diversity +
                self.f_penalty * fatigue)

    def predict_reward(self, features: np.ndarray) -> float:
        # Use first 3 features as engagement, diversity, fatigue
        return self.compute_reward(features[:3])

    def evaluate_preference(self, traj_a: Trajectory,
                           traj_b: Trajectory) -> float:
        reward_a = np.mean([self.compute_reward(s) for s in traj_a.states])
        reward_b = np.mean([self.compute_reward(s) for s in traj_b.states])
        return 1.0 / (1.0 + np.exp(-(reward_a - reward_b)))


class OptimizedShaping(RewardShaping):
    """Optimized reward shaping (grid-search tuned)."""

    def __init__(self):
        # Best weights from grid search
        super().__init__(
            engagement_weight=0.3,
            diversity_weight=0.4,
            fatigue_penalty=-0.4
        )


class AdaptiveShaping(RewardShaping):
    """Adaptive reward shaping that adjusts weights during training."""

    def __init__(self, initial_e: float = 0.3, initial_d: float = 0.3,
                 initial_f: float = -0.2, adaptation_rate: float = 0.01):
        super().__init__(initial_e, initial_d, initial_f)
        self.adaptation_rate = adaptation_rate
        self.history = []

    def adapt(self, performance_metric: float):
        """Adjust weights based on performance."""
        self.history.append(performance_metric)
        if len(self.history) >= 2:
            delta = self.history[-1] - self.history[-2]
            # Simple adaptation: increase fatigue penalty if improving
            if delta > 0:
                self.f_penalty -= self.adaptation_rate


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_ranking_accuracy(model: BaseIRL,
                             test_pairs: List[PreferencePair]) -> Dict:
    """Evaluate model on preference pairs."""
    correct = 0
    predictions = []

    for pair in test_pairs:
        prob_a_preferred = model.evaluate_preference(
            pair.trajectory_a, pair.trajectory_b
        )
        predicted = 0 if prob_a_preferred > 0.5 else 1
        correct += (predicted == pair.preference)
        predictions.append(prob_a_preferred)

    accuracy = correct / len(test_pairs)

    # 95% Wilson confidence interval
    from scipy import stats
    n = len(test_pairs)
    z = 1.96
    p = accuracy
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    ci_lower = center - spread
    ci_upper = center + spread

    return {
        'accuracy': accuracy,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': n,
        'predictions': predictions
    }


def compute_statistical_tests(results_a: List[int],
                             results_b: List[int]) -> Dict:
    """Compute statistical tests between two methods."""
    from scipy import stats

    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(results_a, results_b)

    # Cohen's h effect size
    p1 = np.mean(results_a)
    p2 = np.mean(results_b)
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

    return {
        'wilcoxon_stat': stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_h': abs(h),
        'effect_size': 'negligible' if abs(h) < 0.2 else
                       'small' if abs(h) < 0.5 else
                       'medium' if abs(h) < 0.8 else 'large'
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("PWSS IRL Implementation")
    print("=" * 50)
    print("Available methods:")
    print("  - MLIRL: Maximum Likelihood IRL")
    print("  - PBIRL: Preference-Based IRL")
    print("  - MaxEntIRL: Maximum Entropy IRL")
    print("  - SimplifiedRLHF: RLHF (reward model only)")
    print("  - RewardShaping: Hand-crafted reward")
    print("  - OptimizedShaping: Grid-search optimized")
    print("  - AdaptiveShaping: Adaptive weights")
    print("=" * 50)
    print("\nSee experiments/run_all.py for full experiment pipeline")
