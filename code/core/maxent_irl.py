"""
Maximum Entropy Inverse Reinforcement Learning
Based on Ziebart et al. (2008)

Learns reward function from expert demonstrations.
"""

import numpy as np
from scipy.special import logsumexp
from typing import List, Dict, Tuple
import sys
sys.path.append('..')
from environment import SocialMediaEnv, demonstrations_to_arrays


class MaxEntIRL:
    """
    MaxEnt IRL with linear reward function.
    R(s, a) = theta^T * phi(s, a)
    """

    def __init__(self, state_dim=3, action_dim=5, n_features=12, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_features = n_features
        self.lr = lr

        # Initialize reward parameters
        self.theta = np.random.randn(n_features) * 0.1

        # For tracking
        self.loss_history = []

    def compute_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute feature vector phi(s, a).
        Interpretable features related to well-being.
        """
        mood, engagement, fatigue = state

        features = np.zeros(self.n_features)

        # State features
        features[0] = mood                    # current mood
        features[1] = engagement              # current engagement
        features[2] = fatigue                 # current fatigue
        features[3] = mood * (1 - fatigue)    # mood adjusted for fatigue
        features[4] = mood ** 2               # nonlinear mood effect

        # Action-specific features
        features[5] = float(action == 0)      # positive content
        features[6] = float(action == 2)      # polarizing content
        features[7] = float(action == 4)      # taking break

        # Interaction features
        features[8] = float(action == 0) * mood           # positive when happy
        features[9] = float(action == 4) * fatigue        # break when tired
        features[10] = float(action == 2) * (1 - fatigue) # polarizing when not tired
        features[11] = float(action == 3) * engagement    # interest when engaged

        return features

    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """Compute reward for state-action pair"""
        features = self.compute_features(state, action)
        return np.dot(self.theta, features)

    def compute_expert_feature_expectations(self, states: np.ndarray,
                                           actions: np.ndarray) -> np.ndarray:
        """
        Compute empirical feature expectations from expert data.
        E_expert[phi(s,a)]
        """
        feat_sum = np.zeros(self.n_features)
        n_samples = len(states)

        for i in range(n_samples):
            feat = self.compute_features(states[i], actions[i])
            feat_sum += feat

        return feat_sum / n_samples

    def compute_policy_feature_expectations(self, env: SocialMediaEnv,
                                           n_samples=1000) -> np.ndarray:
        """
        Estimate feature expectations under current reward.
        Uses soft value iteration to get policy.
        """
        feat_sum = np.zeros(self.n_features)

        for _ in range(n_samples):
            state = env.reset()
            done = False
            step_count = 0

            while not done and step_count < 50:
                # Soft policy based on current reward
                action = self._sample_action(state)
                feat = self.compute_features(state, action)
                feat_sum += feat

                state, _, done, _ = env.step(action)
                step_count += 1

        return feat_sum / n_samples

    def _sample_action(self, state: np.ndarray, temperature=1.0) -> int:
        """Sample action using softmax over Q values"""
        q_values = np.array([
            self.compute_reward(state, a) for a in range(self.action_dim)
        ])

        # Softmax
        q_shifted = q_values - np.max(q_values)
        probs = np.exp(q_shifted / temperature)
        probs = probs / (np.sum(probs) + 1e-8)

        return np.random.choice(self.action_dim, p=probs)

    def train(self, expert_states: np.ndarray, expert_actions: np.ndarray,
              env: SocialMediaEnv, n_iterations=500, verbose=True):
        """
        Train IRL using gradient descent on feature matching objective.
        """
        expert_feat_exp = self.compute_expert_feature_expectations(
            expert_states, expert_actions
        )

        if verbose:
            print(f"Expert feature expectations computed")
            print(f"Starting {n_iterations} training iterations...")

        for it in range(n_iterations):
            # Compute policy feature expectations
            policy_feat_exp = self.compute_policy_feature_expectations(
                env, n_samples=200
            )

            # Gradient: expert features - policy features
            gradient = expert_feat_exp - policy_feat_exp

            # Update theta
            self.theta += self.lr * gradient

            # Compute loss (L2 distance between feature expectations)
            loss = np.linalg.norm(gradient)
            self.loss_history.append(loss)

            if verbose and (it + 1) % 50 == 0:
                print(f"Iteration {it+1}/{n_iterations}, Loss: {loss:.4f}")

            # Early stopping
            if loss < 0.01:
                print(f"Converged at iteration {it+1}")
                break

        return self.theta

    def get_learned_reward_weights(self) -> Dict[str, float]:
        """Return interpretable reward weights"""
        feature_names = [
            'mood', 'engagement', 'fatigue', 'adjusted_mood', 'mood_squared',
            'positive_action', 'polarizing_action', 'break_action',
            'positive_when_happy', 'break_when_tired', 'polarizing_fresh', 'interest_engaged'
        ]

        return {name: float(self.theta[i]) for i, name in enumerate(feature_names)}


class SimplifiedMaxEntIRL:
    """
    Simplified version for faster training.
    Uses direct feature matching without full policy rollouts.
    """

    def __init__(self, n_features=6, lr=0.05):
        self.n_features = n_features
        self.lr = lr
        self.theta = np.random.randn(n_features) * 0.1
        self.loss_history = []

    def compute_features(self, state, action):
        """Simplified feature set"""
        mood, engagement, fatigue = state

        features = np.array([
            mood,
            -fatigue,
            engagement,
            float(action == 0) * 0.5,  # positive
            float(action == 4) * fatigue,  # break when tired
            float(action == 2) * (-0.3)  # polarizing penalty
        ])
        return features

    def compute_reward(self, state, action):
        features = self.compute_features(state, action)
        return np.dot(self.theta, features)

    def train(self, states, actions, n_iterations=200, verbose=True):
        """
        Simple gradient-based training on demonstration data.
        """
        n_samples = len(states)

        for it in range(n_iterations):
            total_loss = 0

            for i in range(n_samples):
                state = states[i]
                expert_action = actions[i]

                # Compute Q values for all actions
                q_values = [self.compute_reward(state, a) for a in range(5)]
                q_values = np.array(q_values)

                # Softmax probabilities
                probs = np.exp(q_values - np.max(q_values))
                probs = probs / (np.sum(probs) + 1e-8)

                # Gradient: push up expert action, push down others
                for a in range(5):
                    feat = self.compute_features(state, a)
                    if a == expert_action:
                        grad = (1 - probs[a]) * feat
                    else:
                        grad = -probs[a] * feat
                    self.theta += self.lr * grad / n_samples

                # Cross entropy loss
                loss = -np.log(probs[expert_action] + 1e-8)
                total_loss += loss

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

            if verbose and (it + 1) % 50 == 0:
                print(f"Iteration {it+1}, Avg Loss: {avg_loss:.4f}")

        return self.theta


if __name__ == "__main__":
    print("Testing MaxEnt IRL...")

    # Create environment and generate demonstrations
    env = SocialMediaEnv(seed=42)

    print("Generating expert demonstrations...")
    from environment import generate_expert_demonstrations
    demos = generate_expert_demonstrations(env, n_episodes=100)
    states, actions, _ = demonstrations_to_arrays(demos)

    print(f"Training data: {len(states)} state-action pairs")

    # Train IRL
    print("\nTraining Simplified MaxEnt IRL...")
    irl = SimplifiedMaxEntIRL()
    theta = irl.train(states, actions, n_iterations=100)

    print("\nLearned reward weights:")
    feature_names = ['mood', 'neg_fatigue', 'engagement', 'positive_bonus', 'break_tired', 'polarizing_penalty']
    for name, w in zip(feature_names, theta):
        print(f"  {name}: {w:.3f}")

    print("\nTraining complete!")
