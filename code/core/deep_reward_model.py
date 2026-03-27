"""
Deep Neural Network Reward Model (DNN-RM)
Learns complex non-linear reward functions from expert demonstrations or preferences
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.special import xlogy


class DenseLayer:
    """Fully connected neural network layer with batch normalization"""

    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Args:
            input_size: Input dimension
            output_size: Output dimension
            activation: 'relu', 'tanh', 'linear', or 'sigmoid'
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights using Xavier initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros((1, output_size))

        # For batch normalization
        self.gamma = np.ones((1, output_size))
        self.beta = np.zeros((1, output_size))

    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass.
        Returns output and cache for backward pass.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Linear transformation
        Z = np.dot(X, self.W) + self.b

        # Store for backward
        cache = {'X': X, 'Z': Z}

        # Activation
        if self.activation == 'relu':
            A = np.maximum(0, Z)
            cache['A'] = A
        elif self.activation == 'tanh':
            A = np.tanh(Z)
            cache['A'] = A
        elif self.activation == 'sigmoid':
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
            cache['A'] = A
        else:  # linear
            A = Z
            cache['A'] = A

        return A, cache

    def backward(self, dA: np.ndarray, cache: Dict, learning_rate: float = 0.001):
        """
        Backward pass and update weights.
        """
        X = cache['X']
        Z = cache['Z']
        A = cache['A']

        # Activation gradient
        if self.activation == 'relu':
            dZ = dA * (Z > 0).astype(float)
        elif self.activation == 'tanh':
            dZ = dA * (1 - A ** 2)
        elif self.activation == 'sigmoid':
            dZ = dA * A * (1 - A)
        else:
            dZ = dA

        # Weight and bias gradients
        dW = np.dot(X.T, dZ) / X.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]

        # Gradient w.r.t. input (for backprop)
        dX = np.dot(dZ, self.W.T)

        # Update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX


class DeepRewardModel:
    """
    Deep neural network for learning reward functions.
    Architecture: input -> hidden1 -> hidden2 -> output (scalar reward)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
        learning_rate: float = 0.001,
        l2_penalty: float = 0.0001
    ):
        """
        Args:
            state_dim: Input state dimension
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimization
            l2_penalty: L2 regularization coefficient
        """
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Build network layers
        self.layers = []
        layer_dims = [state_dim] + hidden_dims + [1]

        for i in range(len(layer_dims) - 1):
            # Last layer is linear, others are relu
            activation = 'linear' if i == len(layer_dims) - 2 else 'relu'
            self.layers.append(DenseLayer(layer_dims[i], layer_dims[i + 1], activation))

    def forward(self, state: np.ndarray) -> Tuple[float, List[Dict]]:
        """
        Forward pass through network.
        Returns scalar reward and caches for backward pass.
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)

        A = state
        caches = []

        for layer in self.layers:
            A, cache = layer.forward(A, training=True)
            caches.append(cache)

        reward = A.flatten()[0] if A.shape[0] == 1 else A.flatten()
        return reward, caches

    def predict(self, state: np.ndarray) -> float:
        """Predict reward for a state (inference mode)"""
        reward, _ = self.forward(state)
        return reward if isinstance(reward, float) or reward.size == 1 else reward[0]

    def backward_pass(self, loss_gradient: float, caches: List[Dict]):
        """
        Backward pass through entire network.
        """
        dA = np.array([[loss_gradient]])  # Start with output gradient

        for i in range(len(self.layers) - 1, -1, -1):
            dA = self.layers[i].backward(dA, caches[i], self.learning_rate)

    def compute_trajectory_value(self, trajectory: Dict) -> float:
        """Compute discounted sum of rewards for a trajectory"""
        states = trajectory['states']
        gamma = 0.99
        value = 0.0

        for i, state in enumerate(states):
            r = self.predict(state)
            value += (gamma ** i) * r

        return value

    def train_from_demonstrations(
        self,
        demonstrations: List[Dict],
        n_epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Train reward model from expert demonstrations using maximum margin approach.
        Assumes expert demonstrations have higher returns than random demonstrations.
        """
        history = {'epoch': [], 'loss': [], 'margin_accuracy': []}

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            margin_correct = 0

            # Shuffle demonstrations
            indices = np.random.permutation(len(demonstrations))

            for start_idx in range(0, len(demonstrations), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                for idx in batch_indices:
                    expert_traj = demonstrations[idx]

                    # Generate random trajectory as negative example
                    random_traj = {
                        'states': [np.random.uniform(0, 1, self.state_dim) for _ in range(len(expert_traj['states']))],
                        'actions': expert_traj['actions'].copy()
                    }

                    # Compute trajectory values
                    V_expert = self.compute_trajectory_value(expert_traj)
                    V_random = self.compute_trajectory_value(random_traj)

                    # Maximum margin loss
                    margin = 1.0  # Desired margin
                    loss = max(0, margin - (V_expert - V_random))

                    # Add L2 regularization
                    l2_loss = 0.0
                    for layer in self.layers:
                        l2_loss += np.sum(layer.W ** 2) * self.l2_penalty

                    total_loss = loss + l2_loss

                    # Backward pass (simplified: gradient w.r.t. difference)
                    if loss > 0:
                        # Gradient points to increase expert value, decrease random
                        alpha = 0.01
                        self.learning_rate = alpha

                        # Manual gradient update
                        grad_sign = 1.0 if V_expert < V_random else -1.0
                        loss_gradient = grad_sign / len(expert_traj['states'])

                        for state in expert_traj['states']:
                            _, caches = self.forward(np.array(state))
                            self.backward_pass(loss_gradient, caches)

                    epoch_loss += total_loss

                    # Check margin
                    if V_expert > V_random:
                        margin_correct += 1

            avg_loss = epoch_loss / max(1, len(demonstrations))
            margin_accuracy = margin_correct / len(demonstrations)

            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            history['margin_accuracy'].append(margin_accuracy)

            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: Loss={avg_loss:.4f}, Margin Accuracy={margin_accuracy:.3f}")

        return history

    def train_from_preferences(
        self,
        preference_data: List[Dict],
        n_epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Train reward model from pairwise preferences.
        Uses Bradley-Terry model for preference likelihood.
        """
        history = {'epoch': [], 'loss': [], 'accuracy': []}

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            correct_predictions = 0

            indices = np.random.permutation(len(preference_data))

            for start_idx in range(0, len(preference_data), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                for idx in batch_indices:
                    pref_pair = preference_data[idx]
                    traj_a = pref_pair['trajectory_a']
                    traj_b = pref_pair['trajectory_b']
                    preference = pref_pair['preference']
                    confidence = pref_pair.get('confidence', 1.0)

                    # Forward pass
                    V_a = self.compute_trajectory_value(traj_a)
                    V_b = self.compute_trajectory_value(traj_b)

                    # Bradley-Terry preference likelihood
                    advantage = V_a - V_b
                    prob_prefer_a = 1.0 / (1.0 + np.exp(-np.clip(advantage, -500, 500)))

                    if preference == 1:
                        target_prob = confidence * prob_prefer_a + (1 - confidence) * 0.5
                    else:
                        target_prob = confidence * (1 - prob_prefer_a) + (1 - confidence) * 0.5

                    # Cross-entropy loss
                    loss = -np.log(np.clip(target_prob, 1e-10, 1.0))
                    epoch_loss += loss

                    # Backward pass
                    if preference == 1:
                        error = (1 - prob_prefer_a) * confidence
                    else:
                        error = -prob_prefer_a * confidence

                    # Update for both trajectories
                    for state in traj_a['states']:
                        _, caches = self.forward(np.array(state))
                        self.backward_pass(error * 0.01, caches)

                    for state in traj_b['states']:
                        _, caches = self.forward(np.array(state))
                        self.backward_pass(-error * 0.01, caches)

                    # Check accuracy
                    predicted = 1 if V_a > V_b else 0
                    if predicted == preference:
                        correct_predictions += 1

            avg_loss = epoch_loss / len(preference_data)
            accuracy = correct_predictions / len(preference_data)

            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")

        return history

    def get_feature_importance(self, state_samples: List[np.ndarray]) -> Dict:
        """
        Estimate feature importance by computing gradient variance.
        """
        importance = np.zeros(self.state_dim)

        for state in state_samples:
            # Compute gradient of reward w.r.t. state
            eps = 1e-5
            grads = np.zeros(self.state_dim)

            for i in range(self.state_dim):
                state_plus = state.copy()
                state_minus = state.copy()
                state_plus[i] += eps
                state_minus[i] -= eps

                r_plus = self.predict(state_plus)
                r_minus = self.predict(state_minus)
                grads[i] = (r_plus - r_minus) / (2 * eps)

            importance += grads ** 2

        importance /= len(state_samples)

        return {
            'feature_importance': importance,
            'feature_names': ['mood', 'engagement', 'fatigue']  # For 3D state
        }


if __name__ == "__main__":
    print("Testing Deep Neural Network Reward Model...\n")

    # Create dummy demonstrations
    dummy_demos = []
    for i in range(50):
        demo = {
            'states': [np.random.uniform(0, 1, 3) for _ in range(15)],
            'actions': [np.random.randint(0, 5) for _ in range(15)]
        }
        dummy_demos.append(demo)

    print("Step 1: Training Deep Reward Model from Demonstrations...")
    deep_rm = DeepRewardModel(state_dim=3, hidden_dims=[128, 64, 32])
    demo_history = deep_rm.train_from_demonstrations(dummy_demos, n_epochs=30, verbose=True)

    print("\nStep 2: Testing prediction on sample state...")
    test_state = np.array([0.6, 0.4, 0.3])
    predicted_reward = deep_rm.predict(test_state)
    print(f"Predicted reward for state {test_state}: {predicted_reward:.4f}")

    # Test with preferences
    print("\n" + "="*60)
    print("Step 3: Training Deep Reward Model from Preferences...")
    print("="*60)

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

    deep_rm_pref = DeepRewardModel(state_dim=3, hidden_dims=[64, 32])
    pref_history = deep_rm_pref.train_from_preferences(dummy_prefs, n_epochs=30, verbose=True)

    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Demonstrations - Final Loss: {demo_history['loss'][-1]:.4f}")
    print(f"Preferences - Final Loss: {pref_history['loss'][-1]:.4f}")
    print(f"Preferences - Final Accuracy: {pref_history['accuracy'][-1]:.3f}")

    print("\nStep 4: Feature Importance Analysis")
    print("="*60)
    sample_states = [np.random.uniform(0, 1, 3) for _ in range(100)]
    importance = deep_rm_pref.get_feature_importance(sample_states)
    print("Feature Importance (Gradient Variance):")
    for name, imp in zip(importance['feature_names'], importance['feature_importance']):
        print(f"  {name:15s}: {imp:.6f}")
