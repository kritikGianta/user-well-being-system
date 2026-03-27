"""
Synthetic Social Media Environment for IRL Experiments
Simulates user interactions and mood dynamics
"""

import numpy as np
from typing import Tuple, Dict, List
import random

class SocialMediaEnv:
    """
    Simulated social media environment with mood dynamics.

    State: (mood, engagement_interest, fatigue) all in [0,1]
    Actions: 0=positive, 1=neutral, 2=polarizing, 3=interest, 4=break
    """

    def __init__(self, user_params=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # User-specific parameters (sampled if not provided)
        if user_params is None:
            user_params = self._sample_user_params()

        self.user_params = user_params
        self.state = None
        self.steps = 0
        self.max_steps = 100

        # action effects on state (base effects, modified by user params)
        self.action_effects = {
            0: {'mood': 0.08, 'engagement': 0.05, 'fatigue': 0.02},   # positive content
            1: {'mood': 0.0, 'engagement': 0.02, 'fatigue': 0.03},    # neutral
            2: {'mood': -0.05, 'engagement': 0.12, 'fatigue': 0.08},  # polarizing
            3: {'mood': 0.03, 'engagement': 0.08, 'fatigue': 0.04},   # interest-based
            4: {'mood': 0.02, 'engagement': -0.15, 'fatigue': -0.12}  # break/rest
        }

    def _sample_user_params(self) -> Dict:
        """Sample random user characteristics"""
        return {
            'mood_sensitivity': np.random.uniform(0.5, 1.5),
            'fatigue_rate': np.random.uniform(0.8, 1.2),
            'recovery_rate': np.random.uniform(0.8, 1.2),
            'engagement_decay': np.random.uniform(0.02, 0.06),
            'baseline_mood': np.random.uniform(0.4, 0.7),
            'variance_level': np.random.uniform(0.05, 0.15)
        }

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        baseline = self.user_params['baseline_mood']
        self.state = np.array([
            baseline + np.random.normal(0, 0.05),  # mood
            0.5 + np.random.normal(0, 0.1),        # engagement interest
            0.1 + np.random.uniform(0, 0.1)        # fatigue
        ])
        self.state = np.clip(self.state, 0.01, 0.99)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info)
        """
        if self.state is None:
            raise ValueError("Call reset() first")

        effects = self.action_effects[action]
        params = self.user_params

        # Apply action effects with user-specific modifiers
        noise = np.random.normal(0, params['variance_level'], 3)

        delta_mood = effects['mood'] * params['mood_sensitivity'] + noise[0]
        delta_eng = effects['engagement'] + noise[1]
        delta_fat = effects['fatigue'] * params['fatigue_rate'] + noise[2]

        # Natural dynamics
        natural_mood_drift = (params['baseline_mood'] - self.state[0]) * 0.03
        natural_eng_decay = -self.state[1] * params['engagement_decay']

        # Update state
        new_mood = self.state[0] + delta_mood + natural_mood_drift
        new_eng = self.state[1] + delta_eng + natural_eng_decay
        new_fat = self.state[2] + delta_fat

        # fatigue affects mood negatively
        if new_fat > 0.7:
            new_mood -= 0.02 * (new_fat - 0.7)

        self.state = np.array([new_mood, new_eng, new_fat])
        self.state = np.clip(self.state, 0.01, 0.99)

        self.steps += 1
        done = self.steps >= self.max_steps

        # True reward (what we try to recover via IRL)
        reward = self._compute_true_reward(action)

        info = {
            'mood': self.state[0],
            'engagement': self.state[1],
            'fatigue': self.state[2]
        }

        return self.state.copy(), reward, done, info

    def _compute_true_reward(self, action: int) -> float:
        """
        Ground truth reward - weighted well-being objective.
        This is what we try to recover via IRL.
        """
        mood, engagement, fatigue = self.state

        # True reward weights (hidden from IRL)
        alpha = 0.6  # mood importance
        beta = 0.25  # fatigue penalty
        gamma = 0.15 # engagement value

        reward = alpha * mood - beta * fatigue + gamma * engagement

        # Small bonus for break when fatigue high
        if action == 4 and fatigue > 0.6:
            reward += 0.05

        return reward

    def get_action_space_size(self) -> int:
        return 5

    def get_state_dim(self) -> int:
        return 3


def generate_expert_demonstrations(env, n_episodes=200, policy_type='optimal'):
    """
    Generate expert demonstrations using either optimal or near-optimal policy.
    These serve as training data for IRL.
    """
    demonstrations = []

    for ep in range(n_episodes):
        trajectory = {'states': [], 'actions': [], 'rewards': []}
        state = env.reset()
        done = False

        while not done:
            trajectory['states'].append(state.copy())

            if policy_type == 'optimal':
                # Near-optimal policy based on true reward structure
                action = _expert_policy(state, env)
            else:
                action = np.random.randint(5)

            trajectory['actions'].append(action)

            next_state, reward, done, info = env.step(action)
            trajectory['rewards'].append(reward)
            state = next_state

        demonstrations.append(trajectory)

        if (ep + 1) % 50 == 0:
            print(f"Generated {ep + 1}/{n_episodes} demonstrations")

    return demonstrations


def _expert_policy(state, env):
    """
    Expert policy that approximates optimal behavior.
    Uses heuristics based on true reward structure.
    """
    mood, engagement, fatigue = state

    # If very fatigued, take break
    if fatigue > 0.7:
        return 4  # break

    # If mood is low, prefer positive content
    if mood < 0.4:
        return 0 if np.random.random() > 0.2 else 3  # mostly positive, sometimes interest

    # If engagement is low, boost with interest-based
    if engagement < 0.3:
        return 3  # interest-based

    # Normal behavior - mix of positive and interest
    probs = [0.35, 0.15, 0.05, 0.40, 0.05]  # favor positive and interest

    # Small probability of polarizing (real users sometimes engage with it)
    return np.random.choice(5, p=probs)


def demonstrations_to_arrays(demonstrations):
    """Convert list of trajectories to numpy arrays for training"""
    all_states = []
    all_actions = []
    all_next_states = []

    for traj in demonstrations:
        states = traj['states']
        actions = traj['actions']

        for i in range(len(states) - 1):
            all_states.append(states[i])
            all_actions.append(actions[i])
            all_next_states.append(states[i + 1])

    return (
        np.array(all_states),
        np.array(all_actions),
        np.array(all_next_states)
    )


if __name__ == "__main__":
    # Quick test
    print("Testing synthetic environment...")
    env = SocialMediaEnv(seed=42)

    state = env.reset()
    print(f"Initial state: mood={state[0]:.3f}, eng={state[1]:.3f}, fat={state[2]:.3f}")

    total_reward = 0
    for _ in range(20):
        action = np.random.randint(5)
        state, reward, done, info = env.step(action)
        total_reward += reward

    print(f"After 20 steps: mood={state[0]:.3f}, eng={state[1]:.3f}, fat={state[2]:.3f}")
    print(f"Total reward: {total_reward:.3f}")

    print("\nGenerating sample demonstrations...")
    demos = generate_expert_demonstrations(env, n_episodes=10)
    print(f"Generated {len(demos)} demonstrations")
    print(f"Avg trajectory length: {np.mean([len(d['states']) for d in demos]):.1f}")
