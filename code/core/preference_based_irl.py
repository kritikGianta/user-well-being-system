"""
Preference-Based Inverse Reinforcement Learning (PB-IRL)
Learns reward function from pairwise trajectory preferences instead of expert demonstrations
"""

import numpy as np
from typing import Tuple, List, Dict
import scipy.optimize as optimize
from scipy.special import xlogy


class PreferenceBasedIRL:
    """
    Learn reward functions from pairwise trajectory preferences.

    Theoretical basis:
    - Bradley-Terry model for preference likelihood
    - Maximum likelihood estimation of reward parameters
    - Preference elicitation with confidence weighting
    """

    def __init__(self, state_dim: int, feature_dim: int = None, learning_rate: float = 0.01):
        """
        Args:
            state_dim: Dimension of state space
            feature_dim: Dimension of feature space (if None, equals state_dim)
            learning_rate: Learning rate for gradient updates
        """
        self.state_dim = state_dim
        self.feature_dim = feature_dim or state_dim
        self.learning_rate = learning_rate

        # Initialize reward weights
        self.reward_weights = np.random.normal(0, 0.1, self.feature_dim)

    def featurize(self, state: np.ndarray, action: int = None) -> np.ndarray:
        """
        Convert state (and optionally action) to feature vector.
        Default: identity mapping for states, add action one-hot.
        """
        if action is not None:
            action_onehot = np.zeros(5)
            action_onehot[action] = 1
            return np.concatenate([state, action_onehot])
        return state

    def compute_trajectory_reward(self, trajectory: Dict) -> float:
        """
        Compute total reward for a trajectory using current weight vector.

        Args:
            trajectory: Dict with 'states' and 'actions' keys

        Returns:
            Cumulative discounted reward (gamma=0.99)
        """
        states = trajectory['states']
        actions = trajectory['actions']
        gamma = 0.99
        cumulative_reward = 0.0

        for i, (state, action) in enumerate(zip(states, actions)):
            features = self.featurize(state, action)
            reward = np.dot(self.reward_weights, features)
            cumulative_reward += (gamma ** i) * reward

        return cumulative_reward

    def compute_trajectory_feature_sum(self, trajectory: Dict) -> np.ndarray:
        """
        Compute discounted sum of features over a trajectory.
        Used for gradient computation.
        """
        states = trajectory['states']
        actions = trajectory['actions']
        gamma = 0.99
        feature_sum = np.zeros(self.feature_dim)

        for i, (state, action) in enumerate(zip(states, actions)):
            features = self.featurize(state, action)
            feature_sum += (gamma ** i) * features

        return feature_sum

    def preference_likelihood(
        self,
        traj_a: Dict,
        traj_b: Dict,
        preference: int,
        confidence: float = 1.0
    ) -> float:
        """
        Bradley-Terry model: P(prefer a over b) = 1 / (1 + exp(-(V_a - V_b)))

        Args:
            traj_a: Trajectory A
            traj_b: Trajectory B
            preference: 1 if prefer A, 0 if prefer B
            confidence: Confidence in the preference (0-1)

        Returns:
            Log-likelihood of observed preference
        """
        V_a = self.compute_trajectory_reward(traj_a)
        V_b = self.compute_trajectory_reward(traj_b)

        # Sigmoid preference probability
        advantage = V_a - V_b
        prob_prefer_a = 1.0 / (1.0 + np.exp(-advantage))

        if preference == 1:  # prefer A
            # Interpolate between observed and uniform depending on confidence
            prob = confidence * prob_prefer_a + (1 - confidence) * 0.5
            return np.log(np.clip(prob, 1e-10, 1.0))
        else:  # prefer B
            prob = confidence * (1 - prob_prefer_a) + (1 - confidence) * 0.5
            return np.log(np.clip(prob, 1e-10, 1.0))

    def compute_gradient(
        self,
        traj_a: Dict,
        traj_b: Dict,
        preference: int,
        confidence: float = 1.0
    ) -> np.ndarray:
        """
        Compute gradient of log-likelihood w.r.t. weights.
        Gradient points in direction of increasing likelihood.
        """
        V_a = self.compute_trajectory_reward(traj_a)
        V_b = self.compute_trajectory_reward(traj_b)
        advantage = V_a - V_b

        # Sigmoid derivative
        prob_prefer_a = 1.0 / (1.0 + np.exp(-advantage))

        # Feature sum differences
        feat_sum_a = self.compute_trajectory_feature_sum(traj_a)
        feat_sum_b = self.compute_trajectory_feature_sum(traj_b)
        feat_diff = feat_sum_a - feat_sum_b

        if preference == 1:  # prefer A
            # Gradient is proportional to (prob_prefer_b) * feat_diff
            error_term = (1 - prob_prefer_a) * confidence
        else:  # prefer B
            # Gradient is proportional to -(prob_prefer_a) * feat_diff
            error_term = -prob_prefer_a * confidence

        gradient = error_term * feat_diff
        return gradient

    def train(
        self,
        preference_data: List[Dict],
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Train reward function from preference comparisons.

        Args:
            preference_data: List of dicts with 'trajectory_a', 'trajectory_b',
                           'preference', 'confidence'
            n_epochs: Number of training epochs
            batch_size: Batch size for gradient updates
            verbose: Print progress

        Returns:
            Training history
        """
        history = {'epoch': [], 'loss': [], 'accuracy': []}

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            correct_predictions = 0

            # Shuffle data
            indices = np.random.permutation(len(preference_data))

            for start_idx in range(0, len(preference_data), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_loss = 0.0

                for idx in batch_indices:
                    pref_pair = preference_data[idx]
                    traj_a = pref_pair['trajectory_a']
                    traj_b = pref_pair['trajectory_b']
                    preference = pref_pair['preference']
                    confidence = pref_pair.get('confidence', 1.0)

                    # Compute loss (negative log-likelihood)
                    ll = self.preference_likelihood(traj_a, traj_b, preference, confidence)
                    batch_loss -= ll

                    # Compute gradient and update weights
                    gradient = self.compute_gradient(traj_a, traj_b, preference, confidence)
                    self.reward_weights += self.learning_rate * gradient

                    # Check accuracy
                    V_a = self.compute_trajectory_reward(traj_a)
                    V_b = self.compute_trajectory_reward(traj_b)
                    predicted = 1 if V_a > V_b else 0
                    if predicted == preference:
                        correct_predictions += 1

                epoch_loss += batch_loss / len(batch_indices)

            avg_loss = epoch_loss / max(1, len(preference_data) // batch_size)
            accuracy = correct_predictions / len(preference_data)

            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")

        return history

    def get_reward_weights(self) -> np.ndarray:
        """Return current reward weights"""
        return self.reward_weights.copy()

    def normalize_weights(self):
        """Normalize weights to unit norm"""
        norm = np.linalg.norm(self.reward_weights)
        if norm > 0:
            self.reward_weights /= norm


class MaximumLikelihoodIRL:
    """
    Maximum likelihood approach to preference-based IRL.
    Uses numerical optimization to find best reward weights.
    """

    def __init__(self, state_dim: int, feature_dim: int = None):
        self.state_dim = state_dim
        self.feature_dim = feature_dim or state_dim
        self.reward_weights = np.random.normal(0, 0.1, self.feature_dim)

    def featurize(self, state: np.ndarray, action: int = None) -> np.ndarray:
        """Feature extraction"""
        if action is not None:
            action_onehot = np.zeros(5)
            action_onehot[action] = 1
            return np.concatenate([state, action_onehot])
        return state

    def compute_trajectory_reward(self, trajectory: Dict, weights: np.ndarray = None) -> float:
        """Compute trajectory reward"""
        if weights is None:
            weights = self.reward_weights
        states = trajectory['states']
        actions = trajectory['actions']
        gamma = 0.99
        reward = 0.0

        for i, (state, action) in enumerate(zip(states, actions)):
            features = self.featurize(state, action)
            reward += (gamma ** i) * np.dot(weights, features)

        return reward

    def negative_log_likelihood(self, weights: np.ndarray, preference_data: List[Dict]) -> float:
        """
        Objective function: negative log-likelihood of observed preferences.
        Used for scipy.optimize.minimize
        """
        nll = 0.0

        for pref_pair in preference_data:
            traj_a = pref_pair['trajectory_a']
            traj_b = pref_pair['trajectory_b']
            preference = pref_pair['preference']
            confidence = pref_pair.get('confidence', 1.0)

            V_a = self.compute_trajectory_reward(traj_a, weights)
            V_b = self.compute_trajectory_reward(traj_b, weights)
            advantage = V_a - V_b

            # Bradley-Terry model
            prob_prefer_a = 1.0 / (1.0 + np.exp(-advantage))

            if preference == 1:
                prob = confidence * prob_prefer_a + (1 - confidence) * 0.5
            else:
                prob = confidence * (1 - prob_prefer_a) + (1 - confidence) * 0.5

            nll -= np.log(np.clip(prob, 1e-10, 1.0))

        return nll

    def train(self, preference_data: List[Dict], verbose: bool = True) -> Dict:
        """
        Train using numerical optimization.
        """
        result = optimize.minimize(
            self.negative_log_likelihood,
            self.reward_weights,
            args=(preference_data,),
            method='L-BFGS-B',
            options={'disp': verbose, 'maxiter': 1000}
        )

        self.reward_weights = result.x

        return {
            'success': result.success,
            'final_loss': result.fun,
            'n_iterations': result.nit,
            'weights': self.reward_weights.copy()
        }

    def get_reward_weights(self) -> np.ndarray:
        """Return trained weights"""
        return self.reward_weights.copy()


def evaluate_preference_accuracy(
    irl_model,
    preference_data: List[Dict]
) -> Dict:
    """
    Evaluate how well the learned reward function explains preferences.
    """
    correct = 0
    total = 0
    confidence_scores = []

    for pref_pair in preference_data:
        traj_a = pref_pair['trajectory_a']
        traj_b = pref_pair['trajectory_b']
        actual = pref_pair['preference']
        confidence = pref_pair.get('confidence', 1.0)

        V_a = irl_model.compute_trajectory_reward(traj_a)
        V_b = irl_model.compute_trajectory_reward(traj_b)

        predicted = 1 if V_a > V_b else 0

        if predicted == actual:
            correct += 1
        total += 1

        # Compute prediction confidence (how much V_a differs from V_b)
        confidence_scores.append(abs(V_a - V_b))

    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
        'min_confidence': np.min(confidence_scores) if confidence_scores else 0.0,
        'max_confidence': np.max(confidence_scores) if confidence_scores else 0.0
    }


if __name__ == "__main__":
    print("Testing Preference-Based IRL...")

    # Create dummy preference data
    dummy_prefs = []
    for i in range(100):
        # Trajectory should have 'states' and 'actions'
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

    print(f"\nTesting PreferenceBasedIRL with {len(dummy_prefs)} preferences...")
    pb_irl = PreferenceBasedIRL(state_dim=3)
    hist = pb_irl.train(dummy_prefs, n_epochs=50, verbose=True)

    print("\nTraining history (final 5 epochs):")
    for i in [-5, -4, -3, -2, -1]:
        print(f"  Epoch {hist['epoch'][i]}: Loss={hist['loss'][i]:.4f}, Acc={hist['accuracy'][i]:.3f}")

    print(f"\nLearned reward weights: {pb_irl.get_reward_weights()}")

    print("\n" + "="*50)
    print("Testing MaximumLikelihoodIRL...")
    ml_irl = MaximumLikelihoodIRL(state_dim=3)
    result = ml_irl.train(dummy_prefs, verbose=False)

    print(f"Optimization result: {result['success']}")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Iterations: {result['n_iterations']}")
    print(f"Learned weights: {result['weights']}")

    eval_acc = evaluate_preference_accuracy(ml_irl, dummy_prefs)
    print(f"\nPreference prediction accuracy: {eval_acc['accuracy']:.3f}")
