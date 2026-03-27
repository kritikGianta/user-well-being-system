"""
Counterfactual Evaluation Methods
=================================

Implements IPS, SNIPS, DR estimators for offline policy evaluation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result from a counterfactual estimator."""
    estimate: float
    variance: float
    std_error: float
    n_samples: int
    estimator_name: str
    additional_info: Dict = None


class CounterfactualEvaluator:
    """
    Counterfactual evaluation for offline recommendation.

    Implements:
    - IPS (Inverse Propensity Scoring)
    - SNIPS (Self-Normalized IPS)
    - CIPS (Clipped IPS)
    - DR (Doubly Robust)
    """

    def __init__(self, clip_threshold: float = 100.0):
        self.clip_threshold = clip_threshold

    def compute_ips(self,
                    rewards: np.ndarray,
                    propensities_logged: np.ndarray,
                    propensities_target: np.ndarray) -> EvaluationResult:
        """
        Inverse Propensity Scoring estimator.

        V_IPS = (1/n) * sum_i [ r_i * p_target(a_i|s_i) / p_logged(a_i|s_i) ]

        Args:
            rewards: Observed rewards for logged actions
            propensities_logged: P(a|s) under logging policy
            propensities_target: P(a|s) under target policy

        Returns:
            EvaluationResult with estimate and statistics
        """
        n = len(rewards)

        # Importance weights
        weights = propensities_target / np.maximum(propensities_logged, 1e-8)

        # IPS estimate
        ips_estimate = np.mean(rewards * weights)

        # Variance estimate
        ips_variance = np.var(rewards * weights) / n

        return EvaluationResult(
            estimate=float(ips_estimate),
            variance=float(ips_variance),
            std_error=float(np.sqrt(ips_variance)),
            n_samples=n,
            estimator_name="IPS",
            additional_info={
                "mean_weight": float(np.mean(weights)),
                "max_weight": float(np.max(weights)),
                "effective_sample_size": float(n / (1 + np.var(weights)))
            }
        )

    def compute_snips(self,
                      rewards: np.ndarray,
                      propensities_logged: np.ndarray,
                      propensities_target: np.ndarray) -> EvaluationResult:
        """
        Self-Normalized Inverse Propensity Scoring.

        V_SNIPS = sum_i [ r_i * w_i ] / sum_i [ w_i ]

        More stable than IPS when weights have high variance.
        """
        n = len(rewards)

        # Importance weights
        weights = propensities_target / np.maximum(propensities_logged, 1e-8)

        # SNIPS estimate
        sum_weighted_rewards = np.sum(rewards * weights)
        sum_weights = np.sum(weights)
        snips_estimate = sum_weighted_rewards / max(sum_weights, 1e-8)

        # Bootstrap variance estimate
        n_bootstrap = 100
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            boot_estimate = (np.sum(rewards[idx] * weights[idx]) /
                           max(np.sum(weights[idx]), 1e-8))
            bootstrap_estimates.append(boot_estimate)

        snips_variance = np.var(bootstrap_estimates)

        return EvaluationResult(
            estimate=float(snips_estimate),
            variance=float(snips_variance),
            std_error=float(np.sqrt(snips_variance)),
            n_samples=n,
            estimator_name="SNIPS",
            additional_info={
                "effective_sample_size": float(sum_weights ** 2 / np.sum(weights ** 2)),
                "sum_weights": float(sum_weights)
            }
        )

    def compute_cips(self,
                     rewards: np.ndarray,
                     propensities_logged: np.ndarray,
                     propensities_target: np.ndarray) -> EvaluationResult:
        """
        Clipped Inverse Propensity Scoring.

        Clips importance weights to reduce variance at cost of some bias.
        """
        n = len(rewards)

        # Importance weights with clipping
        weights = propensities_target / np.maximum(propensities_logged, 1e-8)
        clipped_weights = np.clip(weights, 0, self.clip_threshold)

        # CIPS estimate
        cips_estimate = np.mean(rewards * clipped_weights)

        # Variance
        cips_variance = np.var(rewards * clipped_weights) / n

        # Track clipping statistics
        n_clipped = np.sum(weights > self.clip_threshold)

        return EvaluationResult(
            estimate=float(cips_estimate),
            variance=float(cips_variance),
            std_error=float(np.sqrt(cips_variance)),
            n_samples=n,
            estimator_name="CIPS",
            additional_info={
                "clip_threshold": self.clip_threshold,
                "n_clipped": int(n_clipped),
                "pct_clipped": float(n_clipped / n * 100)
            }
        )

    def compute_doubly_robust(self,
                              rewards: np.ndarray,
                              propensities_logged: np.ndarray,
                              propensities_target: np.ndarray,
                              reward_predictions: np.ndarray) -> EvaluationResult:
        """
        Doubly Robust estimator.

        V_DR = (1/n) * sum_i [ r_hat(s_i) + w_i * (r_i - r_hat(s_i)) ]

        Combines IPS with a reward model for lower variance.
        Consistent if either propensities OR reward model is correct.
        """
        n = len(rewards)

        # Importance weights
        weights = propensities_target / np.maximum(propensities_logged, 1e-8)

        # DR estimate
        residuals = rewards - reward_predictions
        dr_estimate = np.mean(reward_predictions + weights * residuals)

        # Variance estimate
        dr_variance = np.var(reward_predictions + weights * residuals) / n

        return EvaluationResult(
            estimate=float(dr_estimate),
            variance=float(dr_variance),
            std_error=float(np.sqrt(dr_variance)),
            n_samples=n,
            estimator_name="DR",
            additional_info={
                "reward_model_mse": float(np.mean(residuals ** 2)),
                "reward_model_bias": float(np.mean(residuals))
            }
        )

    def evaluate_all(self,
                     rewards: np.ndarray,
                     propensities_logged: np.ndarray,
                     propensities_target: np.ndarray,
                     reward_predictions: Optional[np.ndarray] = None) -> Dict[str, EvaluationResult]:
        """
        Run all estimators and return results.
        """
        results = {
            "IPS": self.compute_ips(rewards, propensities_logged, propensities_target),
            "SNIPS": self.compute_snips(rewards, propensities_logged, propensities_target),
            "CIPS": self.compute_cips(rewards, propensities_logged, propensities_target),
        }

        if reward_predictions is not None:
            results["DR"] = self.compute_doubly_robust(
                rewards, propensities_logged, propensities_target, reward_predictions
            )

        return results


def estimate_propensities_uniform(n_samples: int, n_items: int) -> np.ndarray:
    """Assume uniform random logging policy."""
    return np.ones(n_samples) / n_items


def estimate_propensities_from_model(reward_model, states: np.ndarray,
                                     logged_actions: np.ndarray,
                                     n_items: int,
                                     temperature: float = 1.0) -> np.ndarray:
    """
    Estimate target policy propensities from learned reward model.

    Uses softmax policy: p(a|s) = exp(R(s,a)/T) / sum_a' exp(R(s,a')/T)
    """
    propensities = []

    for i in range(len(states)):
        state = states[i]
        action = logged_actions[i]

        # Compute rewards for all actions (simplified)
        # In practice, would need state-action features
        rewards = np.array([reward_model.predict_reward(state)
                          for _ in range(n_items)])

        # Softmax
        exp_rewards = np.exp((rewards - np.max(rewards)) / temperature)
        probs = exp_rewards / np.sum(exp_rewards)

        propensities.append(probs[action])

    return np.array(propensities)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Counterfactual Evaluation Module")
    print("=" * 50)

    # Demo with synthetic data
    np.random.seed(42)
    n = 1000
    n_items = 100

    # Synthetic logged data
    rewards = np.random.randn(n) + 1
    prop_logged = np.ones(n) / n_items
    prop_target = np.random.uniform(0.5, 1.5, n) / n_items
    reward_predictions = rewards + np.random.randn(n) * 0.5

    evaluator = CounterfactualEvaluator()
    results = evaluator.evaluate_all(
        rewards, prop_logged, prop_target, reward_predictions
    )

    print("\nDemo Results:")
    for name, result in results.items():
        print(f"  {name}: {result.estimate:.4f} +/- {result.std_error:.4f}")
