"""
RLHF (Reinforcement Learning from Human Feedback) Baseline
Trains a reward model from preferences, then uses PPO with learned reward
"""

import numpy as np
from typing import Tuple, List, Dict
import random


class RewardModelNetwork:
    """
    Neural network reward model trained from human preferences.
    Simple 2-layer network: input -> hidden -> output scalar reward
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, learning_rate: float = 0.001):
        """
        Args:
            state_dim: Input state dimension
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for updates
        """
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Initialize weights randomly
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)

    def relu_grad(self, x):
        """ReLU gradient"""
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_grad(self, x):
        """Sigmoid gradient"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, state: np.ndarray) -> Tuple[float, Dict]:
        """
        Forward pass through network.
        state: shape (state_dim,) or (batch_size, state_dim)
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Hidden layer
        self.z1 = np.dot(state, self.W1) + self.b1
        self.h1 = self.relu(self.z1)

        # Output layer
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        reward = self.z2.flatten()

        cache = {'state': state, 'z1': self.z1, 'h1': self.h1, 'z2': self.z2}

        return reward[0] if len(reward) == 1 else reward, cache

    def backward(self, cache: Dict, reward_gradient: np.ndarray) -> Dict:
        """
        Backward pass to compute weight gradients.
        """
        state = cache['state']
        h1 = cache['h1']
        z1 = cache['z1']

        # Output layer gradient
        dL_dz2 = reward_gradient.reshape(-1, 1)
        dL_dW2 = np.dot(h1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        dL_dh1 = np.dot(dL_dz2, self.W2.T)

        # Hidden layer gradient (ReLU)
        dL_dz1 = dL_dh1 * self.relu_grad(z1)
        dL_dW1 = np.dot(state.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        return {
            'dW1': dL_dW1,
            'db1': dL_db1,
            'dW2': dL_dW2,
            'db2': dL_db2
        }

    def update_weights(self, gradients: Dict):
        """Update weights using computed gradients"""
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']

    def predict(self, state: np.ndarray) -> float:
        """Predict reward for a state"""
        reward, _ = self.forward(state)
        return reward


class RLHFTrainer:
    """
    RLHF: Train reward model from preferences, then train policy with learned rewards.
    """

    def __init__(self, state_dim: int = 3, hidden_dim: int = 64):
        """Initialize reward model"""
        self.state_dim = state_dim
        self.reward_model = RewardModelNetwork(state_dim, hidden_dim)

    def compute_trajectory_reward(self, trajectory: Dict) -> float:
        """Sum of rewards across trajectory"""
        states = trajectory['states']
        gamma = 0.99
        total_reward = 0.0

        for i, state in enumerate(states):
            r = self.reward_model.predict(state)
            total_reward += (gamma ** i) * r

        return total_reward

    def train_reward_model(
        self,
        preference_data: List[Dict],
        n_epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Train reward model using Bradley-Terry preference model.

        Minimizes: -log P(preference | rewards)
        """
        history = {'epoch': [], 'loss': [], 'accuracy': []}

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            n_batches = 0

            # Shuffle and batch
            indices = np.random.permutation(len(preference_data))

            for start_idx in range(0, len(preference_data), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_size_actual = len(batch_indices)

                for idx in batch_indices:
                    pref_pair = preference_data[idx]
                    traj_a = pref_pair['trajectory_a']
                    traj_b = pref_pair['trajectory_b']
                    preference = pref_pair['preference']
                    confidence = pref_pair.get('confidence', 1.0)

                    # Forward pass
                    V_a = self.compute_trajectory_reward(traj_a)
                    V_b = self.compute_trajectory_reward(traj_b)

                    # Compute loss using preference model
                    advantage = V_a - V_b
                    prob_prefer_a = 1.0 / (1.0 + np.exp(-np.clip(advantage, -500, 500)))

                    if preference == 1:
                        target = confidence * prob_prefer_a + (1 - confidence) * 0.5
                    else:
                        target = confidence * (1 - prob_prefer_a) + (1 - confidence) * 0.5

                    loss = -np.log(np.clip(target, 1e-10, 1.0))
                    epoch_loss += loss

                    # Backward pass (simplified)
                    if preference == 1:
                        error = (1 - prob_prefer_a) * confidence
                    else:
                        error = -prob_prefer_a * confidence

                    # Update reward model (simplified gradient-based update)
                    for state in traj_a['states']:
                        _, cache = self.reward_model.forward(state)
                        gradients = self.reward_model.backward(cache, np.array([error]))
                        self.reward_model.update_weights(gradients)

                    for state in traj_b['states']:
                        _, cache = self.reward_model.forward(state)
                        gradients = self.reward_model.backward(cache, np.array([-error]))
                        self.reward_model.update_weights(gradients)

                    # Check accuracy
                    predicted = 1 if V_a > V_b else 0
                    if predicted == preference:
                        correct_predictions += 1

                epoch_loss /= batch_size_actual
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            accuracy = correct_predictions / len(preference_data)

            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")

        return history

    def get_learned_reward_function(self):
        """Return the trained reward model"""
        return self.reward_model


class RLHFPolicy:
    """
    Policy trained with RLHF rewards using simple gradient-based optimization.
    """

    def __init__(self, state_dim: int = 3, action_dim: int = 5, learning_rate: float = 0.01):
        r"""
        Initialize policy.
        Uses linear policy: pi(a|s) \propto exp(theta · phi(s,a))
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Policy parameters (linear): state_dim + action_dim (one-hot)
        self.policy_params = np.random.normal(0, 0.1, state_dim + action_dim)

    def featurize(self, state: np.ndarray, action: int) -> np.ndarray:
        """Feature vector: [state features, action one-hot]"""
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1
        return np.concatenate([state, action_onehot])

    def action_logits(self, state: np.ndarray) -> np.ndarray:
        """Compute logits for each action"""
        logits = np.zeros(self.action_dim)
        for a in range(self.action_dim):
            features = self.featurize(state, a)
            logits[a] = np.dot(self.policy_params, features)
        return logits

    def action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities using softmax"""
        logits = self.action_logits(state)
        logits = np.clip(logits, -500, 500)  # Prevent overflow
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)

    def sample_action(self, state: np.ndarray) -> int:
        """Sample action from policy"""
        probs = self.action_probabilities(state)
        return np.random.choice(self.action_dim, p=probs)

    def update_with_advantage(self, state: np.ndarray, action: int, advantage: float):
        """
        Update policy parameters with advantage (REINFORCE-style update).
        gradient of log pi(a|s) w.r.t. params
        """
        features = self.featurize(state, action)
        self.policy_params += self.learning_rate * advantage * features

    def train_from_trajectories(
        self,
        trajectories: List[Dict],
        reward_model,
        n_epochs: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train policy using rewards from the learned reward model.
        """
        history = {'epoch': [], 'avg_return': []}

        for epoch in range(n_epochs):
            epoch_return = 0.0
            n_trajectories = 0

            for trajectory in trajectories:
                states = trajectory['states']
                actions = trajectory['actions']

                # Compute returns with learned reward
                returns = np.zeros(len(states))
                gamma = 0.99
                cumulative_return = 0.0

                for i in range(len(states) - 1, -1, -1):
                    reward = reward_model.predict(states[i])
                    cumulative_return = reward + gamma * cumulative_return
                    returns[i] = cumulative_return

                # Baseline (mean return)
                baseline = np.mean(returns)

                # Update policy
                for state, action, return_val in zip(states, actions, returns):
                    advantage = return_val - baseline
                    self.update_with_advantage(state, action, advantage)

                epoch_return += np.sum(returns)
                n_trajectories += 1

            avg_return = epoch_return / max(1, n_trajectories)
            history['epoch'].append(epoch)
            history['avg_return'].append(avg_return)

            if verbose and (epoch + 1) % max(1, n_epochs // 3) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: Avg Return={avg_return:.4f}")

        return history


if __name__ == "__main__":
    print("Testing RLHF Implementation...")

    # Create dummy preference data for reward model training
    dummy_prefs = []
    for i in range(50):
        traj_a = {
            'states': [np.random.uniform(0, 1, 3) for _ in range(10)],
            'actions': [np.random.randint(0, 5) for _ in range(10)]
        }
        traj_b = {
            'states': [np.random.uniform(0, 1, 3) for _ in range(10)],
            'actions': [np.random.randint(0, 5) for _ in range(10)]
        }
        dummy_prefs.append({
            'trajectory_a': traj_a,
            'trajectory_b': traj_b,
            'preference': np.random.randint(0, 2),
            'confidence': np.random.uniform(0.5, 1.0)
        })

    print("Step 1: Training reward model from preferences...")
    rlhf = RLHFTrainer(state_dim=3)
    reward_history = rlhf.train_reward_model(dummy_prefs, n_epochs=30, verbose=True)

    print("\nStep 2: Testing learned reward model...")
    test_state = np.array([0.5, 0.5, 0.3])
    predicted_reward = rlhf.reward_model.predict(test_state)
    print(f"Predicted reward for test state: {predicted_reward:.4f}")

    print("\nStep 3: Training policy with learned rewards...")
    dummy_trajectories = [
        {
            'states': [np.random.uniform(0, 1, 3) for _ in range(15)],
            'actions': [np.random.randint(0, 5) for _ in range(15)]
        }
        for _ in range(20)
    ]

    policy = RLHFPolicy(state_dim=3, action_dim=5)
    policy_history = policy.train_from_trajectories(
        dummy_trajectories,
        rlhf.reward_model,
        n_epochs=20,
        verbose=True
    )

    print("\nRLHF training complete!")
    print(f"Reward model final accuracy: {reward_history['accuracy'][-1]:.3f}")
    print(f"Policy final avg return: {policy_history['avg_return'][-1]:.4f}")
