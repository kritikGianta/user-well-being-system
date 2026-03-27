"""
Reward Shaping Baselines for Comparison
Different approaches to shape and design reward functions
"""

import numpy as np
from typing import Callable, Dict, Tuple


class RewardShapingBaseline:
    """
    Base class for reward shaping approaches.
    Compares different manual reward designs against learned rewards.
    """

    def __init__(self, state_dim: int = 3, action_dim: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """Compute shaped reward for state-action pair"""
        raise NotImplementedError


class SimpleRewardShaping(RewardShapingBaseline):
    """
    Simple hand-crafted rewards:
    - Reward mood, penalize fatigue
    - Encourage engagement
    - Basic action preferences
    """

    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """
        Manual reward design:
        r(s,a) = w_mood * mood - w_fat * fatigue + w_eng * engagement + action_bonus
        """
        mood, engagement, fatigue = state

        # Base reward
        reward = 0.5 * mood - 0.3 * fatigue + 0.2 * engagement

        # Action bonuses (heuristic)
        action_bonuses = {
            0: 0.05,    # positive content - small bonus
            1: 0.0,     # neutral - no bonus
            2: -0.05,   # polarizing - penalty
            3: 0.10,    # interest - bonus
            4: 0.15 if fatigue > 0.6 else 0.0  # break - bonus if tired
        }

        reward += action_bonuses.get(action, 0.0)

        return reward


class AdaptiveRewardShaping(RewardShapingBaseline):
    """
    Reward shaping that adapts based on user state.
    Different reward structures for different situations.
    """

    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """
        Adaptive rewards:
        - If low mood: prioritize mood boosting actions
        - If high fatigue: prioritize rest
        - If low engagement: prioritize engaging content
        """
        mood, engagement, fatigue = state

        # Adaptive weights based on state
        if fatigue > 0.75:
            # When very tired: prioritize rest
            if action == 4:  # break
                return 0.3
            else:
                return -0.1

        if mood < 0.35:
            # When mood is low: prioritize positive content and breaks
            if action == 0:  # positive
                return 0.25
            elif action == 4:  # break
                return 0.15
            else:
                return -0.05

        if engagement < 0.25:
            # When engagement is low: prioritize interesting content
            if action == 3:  # interest
                return 0.25
            elif action == 0:  # positive
                return 0.10
            else:
                return 0.0

        # Default healthy state
        mood_weight = 0.4
        eng_weight = 0.3
        fat_weight = -0.3

        reward = mood_weight * mood + eng_weight * engagement + fat_weight * fatigue

        # Slight preference for engagement-focused content (action 3)
        if action == 3:
            reward += 0.05

        return reward


class WellBeingRewardShaping(RewardShapingBaseline):
    """
    Comprehensive well-being focused reward shaping.
    Incorporates psychological principles:
    - Diminishing returns on engagement/stimulation
    - Importance of mood stability
    - Fatigue recovery value
    """

    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """
        Well-being focused reward:
        Combines multiple psychological factors
        """
        mood, engagement, fatigue = state

        # Non-linear mood component (diminishing returns)
        if mood > 0.7:
            mood_value = 0.7 + 0.1 * (mood - 0.7)  # slower growth when high
        else:
            mood_value = mood * 1.0

        # Non-linear engagement component
        # High engagement is good but not unlimited
        if engagement > 0.8:
            eng_value = 0.8 + 0.1 * (engagement - 0.8)  # diminishing
        else:
            eng_value = engagement * 0.8

        # Fatigue penalty increases non-linearly
        if fatigue > 0.7:
            fat_penalty = 0.3 + 0.2 * (fatigue - 0.7) ** 2  # exponential penalty
        else:
            fat_penalty = 0.2 * fatigue

        base_reward = 0.45 * mood_value + 0.25 * eng_value - fat_penalty

        # Action-specific bonuses
        action_effects = {
            0: 0.08,    # positive - boosts mood
            1: 0.02,    # neutral - safe
            2: -0.10,   # polarizing - harmful
            3: 0.12,    # interest - healthy engagement
            4: 0.20 if fatigue > 0.6 else 0.05  # break - value depends on fatigue
        }

        return base_reward + action_effects.get(action, 0.0)


class SafetyConstrainedRewardShaping(RewardShapingBaseline):
    """
    Reward shaping with explicit safety constraints.
    Prevents harmful behaviors and excessive usage.
    """

    def __init__(self, state_dim: int = 3, action_dim: int = 5, max_session_time: int = 100):
        super().__init__(state_dim, action_dim)
        self.max_session_time = max_session_time
        self.session_steps = 0

    def reset_session(self):
        """Reset session step counter"""
        self.session_steps = 0

    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """
        Safety-constrained reward:
        - Penalizes excessive polarizing content
        - Encourages breaks before fatigue becomes critical
        - Prevents unhealthy behavior patterns
        """
        mood, engagement, fatigue = state
        self.session_steps += 1

        # Base well-being reward
        reward = 0.5 * mood - 0.3 * fatigue + 0.2 * engagement

        # Safety penalties
        # Penalty for high fatigue + no break
        if fatigue > 0.7:
            if action != 4:  # not taking break
                reward -= 0.15  # strong penalty
            else:
                reward += 0.25  # reward for breaking

        # Penalty for polarizing content
        if action == 2:  # polarizing
            reward -= 0.15  # always penalize polarizing

        # Session time penalty (encourage breaks/logout)
        session_fraction = min(1.0, self.session_steps / self.max_session_time)
        if session_fraction > 0.7:
            if action == 4:  # break/logout
                reward += 0.1
            else:
                reward -= 0.05

        return reward


class OptimizedIRLRewardShaping(RewardShapingBaseline):
    """
    Reward function optimized for IRL training success.
    Designed to have learnable structure and clear dependencies.
    """

    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """
        Structured reward for IRL:
        Separates state-based and action-based components
        """
        mood, engagement, fatigue = state

        # State-based component (what we want the model to learn)
        state_reward = (
            0.6 * mood +           # mood is important
            0.2 * engagement -     # engagement is secondary
            0.3 * fatigue          # fatigue should be minimized
        )

        # Action-based component (structural preferences)
        action_rewards = {
            0: 0.10,   # positive content: +10% bonus
            1: 0.00,   # neutral: baseline
            2: -0.15,  # polarizing: -15% penalty
            3: 0.15,   # interest: +15% bonus
            4: -0.05   # break: small penalty except when needed
        }

        action_bonus = action_rewards.get(action, 0.0)

        # Context-dependent bonus for break
        if action == 4 and fatigue > 0.65:
            action_bonus = 0.15

        return state_reward + action_bonus

    def get_feature_weights(self) -> Dict:
        """Return interpretable feature weights for analysis"""
        return {
            'mood_weight': 0.6,
            'engagement_weight': 0.2,
            'fatigue_weight': -0.3,
            'action_preference': {
                'positive': 0.10,
                'neutral': 0.00,
                'polarizing': -0.15,
                'interest': 0.15,
                'break': -0.05  # (increased if fatigued)
            }
        }


def compare_reward_shapers(state: np.ndarray, action: int) -> Dict:
    """
    Compare all reward shaping methods on a given state-action pair.
    """
    shapers = {
        'simple': SimpleRewardShaping(),
        'adaptive': AdaptiveRewardShaping(),
        'wellbeing': WellBeingRewardShaping(),
        'optimized_irl': OptimizedIRLRewardShaping(),
    }

    rewards = {}
    for name, shaper in shapers.items():
        rewards[name] = shaper.compute_reward(state, action)

    return rewards


def generate_shaped_trajectories(
    trajectories: list,
    reward_shaper: RewardShapingBaseline
) -> list:
    """
    Add shaped rewards to trajectories.
    """
    shaped_trajectories = []

    for traj in trajectories:
        shaped_traj = {
            'states': traj['states'].copy() if hasattr(traj['states'], 'copy') else traj['states'][:],
            'actions': traj['actions'].copy() if hasattr(traj['actions'], 'copy') else traj['actions'][:],
            'shaped_rewards': []
        }

        for state, action in zip(traj['states'], traj['actions']):
            reward = reward_shaper.compute_reward(state, action)
            shaped_traj['shaped_rewards'].append(reward)

        shaped_trajectories.append(shaped_traj)

    return shaped_trajectories


if __name__ == "__main__":
    print("Testing Reward Shaping Baselines...\n")

    # Test different reward shapers on sample states
    test_states = [
        (np.array([0.8, 0.6, 0.2]), "Happy, engaged, not tired"),
        (np.array([0.3, 0.4, 0.8]), "Sad, disengaged, very tired"),
        (np.array([0.5, 0.3, 0.3]), "Neutral, low engagement, rested"),
    ]

    actions = ["Positive", "Neutral", "Polarizing", "Interest", "Break"]

    for state, description in test_states:
        print(f"State: {description}")
        print(f"Values: mood={state[0]:.2f}, engagement={state[1]:.2f}, fatigue={state[2]:.2f}\n")

        for action_idx in range(5):
            rewards = compare_reward_shapers(state, action_idx)
            print(f"  Action: {actions[action_idx]}")
            for shaper_name, reward in rewards.items():
                print(f"    {shaper_name:20s}: {reward:+.4f}")
            print()

    print("\n" + "="*60)
    print("Feature Weights (OptimizedIRLRewardShaping):")
    print("="*60)

    irl_shaper = OptimizedIRLRewardShaping()
    weights = irl_shaper.get_feature_weights()

    print("\nState Feature Weights:")
    print(f"  Mood:       {weights['mood_weight']:+.2f}")
    print(f"  Engagement: {weights['engagement_weight']:+.2f}")
    print(f"  Fatigue:    {weights['fatigue_weight']:+.2f}")

    print("\nAction Preferences:")
    for action_name, weight in weights['action_preference'].items():
        print(f"  {action_name:15s}: {weight:+.2f}")

    print("\n" + "="*60)
    print("Reward Shaper Characteristics:")
    print("="*60)

    characteristics = {
        'Simple': "Hand-crafted, interpretable, fixed structure",
        'Adaptive': "Context-aware, adjusts based on state, good coverage",
        'WellBeing': "Psychological principles, diminishing returns, comprehensive",
        'OptimizedIRL': "Designed for IRL learnability, clear structure, balanced",
        'SafetyConstrained': "Emphasizes safety, prevents harmful behaviors"
    }

    for name, desc in characteristics.items():
        print(f"{name:20s}: {desc}")
