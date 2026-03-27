"""
Proximal Policy Optimization (PPO) for learned reward.
Phase 2 of the IRL -> RL pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict
import sys
sys.path.append('..')


class PolicyNetwork(nn.Module):
    """Simple MLP policy network"""

    def __init__(self, state_dim=3, action_dim=5, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()

        return action.item(), probs.squeeze()


class ValueNetwork(nn.Module):
    """Value function estimator"""

    def __init__(self, state_dim=3, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOTrainer:
    """
    PPO algorithm implementation.
    Trains policy to maximize learned reward function.
    """

    def __init__(self, state_dim=3, action_dim=5, hidden_size=64,
                 lr=3e-4, gamma=0.99, clip_ratio=0.2, epochs=10):

        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs

        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_size)
        self.value = ValueNetwork(state_dim, hidden_size)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        # Tracking
        self.policy_losses = []
        self.value_losses = []
        self.episode_rewards = []

    def compute_returns(self, rewards: List[float], dones: List[bool]) -> np.ndarray:
        """Compute discounted returns"""
        returns = np.zeros(len(rewards))
        running_return = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        return returns

    def collect_rollout(self, env, reward_fn, n_steps=2048):
        """
        Collect experience using current policy.
        reward_fn: function that takes (state, action) and returns reward
        """
        states, actions, rewards, dones, old_probs = [], [], [], [], []

        state = env.reset()

        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state)

            with torch.no_grad():
                logits = self.policy(state_tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1).squeeze()
                dist = Categorical(probs)
                action = dist.sample().item()

            # Use learned reward instead of environment reward
            reward = reward_fn(state, action)

            next_state, _, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_probs.append(probs[action].item())

            if done:
                state = env.reset()
            else:
                state = next_state

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'old_probs': np.array(old_probs)
        }

    def update(self, rollout: Dict):
        """PPO update step"""
        states = torch.FloatTensor(rollout['states'])
        actions = torch.LongTensor(rollout['actions'])
        old_probs = torch.FloatTensor(rollout['old_probs'])

        # Compute returns and advantages
        returns = self.compute_returns(rollout['rewards'], rollout['dones'])
        returns = torch.FloatTensor(returns)

        with torch.no_grad():
            values = self.value(states)
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        batch_size = min(256, len(states))
        indices = np.arange(len(states))

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_probs = old_probs[batch_idx]

                # Policy loss
                logits = self.policy(batch_states)
                probs = torch.softmax(logits, dim=-1)
                new_probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze()

                ratio = new_probs / (batch_old_probs + 1e-8)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()

                # Value loss
                values = self.value(batch_states)
                value_loss = nn.MSELoss()(values, batch_returns)

                # Update
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())

    def train(self, env, reward_fn, n_iterations=50, n_steps_per_iter=1024, verbose=True):
        """Main training loop"""
        if verbose:
            print("Starting PPO training...")

        for it in range(n_iterations):
            # Collect rollout
            rollout = self.collect_rollout(env, reward_fn, n_steps=n_steps_per_iter)

            mean_reward = np.mean(rollout['rewards'])
            self.episode_rewards.append(mean_reward)

            # Update policy
            self.update(rollout)

            if verbose and (it + 1) % 10 == 0:
                print(f"Iteration {it+1}/{n_iterations}, Avg Reward: {mean_reward:.4f}")

        return self.policy

    def evaluate(self, env, reward_fn=None, n_episodes=20, use_true_reward=False):
        """
        Evaluate trained policy.
        If use_true_reward, uses environment's true reward.
        """
        episode_returns = []
        mood_outcomes = []
        fatigue_outcomes = []

        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            moods = []
            fatigues = []

            while not done:
                action, _ = self.policy.get_action(state, deterministic=True)

                next_state, true_reward, done, info = env.step(action)

                if use_true_reward:
                    total_reward += true_reward
                elif reward_fn is not None:
                    total_reward += reward_fn(state, action)

                moods.append(info['mood'])
                fatigues.append(info['fatigue'])
                state = next_state

            episode_returns.append(total_reward)
            mood_outcomes.append(np.mean(moods))
            fatigue_outcomes.append(np.mean(fatigues))

        return {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_mood': np.mean(mood_outcomes),
            'std_mood': np.std(mood_outcomes),
            'mean_fatigue': np.mean(fatigue_outcomes),
            'std_fatigue': np.std(fatigue_outcomes)
        }


class BaselinePolicy:
    """Baseline policies for comparison"""

    @staticmethod
    def engagement_only(state):
        """Policy that maximizes engagement (proxy metric)"""
        mood, engagement, fatigue = state

        # Prefer polarizing and interest content for engagement
        if engagement < 0.3:
            return 3  # interest
        elif np.random.random() < 0.4:
            return 2  # polarizing (high engagement)
        else:
            return 3  # interest

    @staticmethod
    def random_policy(state):
        """Random baseline"""
        return np.random.randint(5)

    @staticmethod
    def behavioral_cloning(state, demo_states, demo_actions):
        """Simple BC: nearest neighbor in state space"""
        distances = np.linalg.norm(demo_states - state, axis=1)
        nearest_idx = np.argmin(distances)
        return demo_actions[nearest_idx]


if __name__ == "__main__":
    from environment import SocialMediaEnv, generate_expert_demonstrations
    from maxent_irl import SimplifiedMaxEntIRL, demonstrations_to_arrays

    print("Full Pipeline Test")
    print("=" * 50)

    # Create environment
    env = SocialMediaEnv(seed=42)

    # Phase 1: Generate demonstrations and train IRL
    print("\n[Phase 1] Generating demonstrations...")
    demos = generate_expert_demonstrations(env, n_episodes=100)
    states, actions, _ = demonstrations_to_arrays(demos)

    print(f"Training IRL on {len(states)} samples...")
    irl = SimplifiedMaxEntIRL()
    irl.train(states, actions, n_iterations=100, verbose=False)

    print("Learned reward weights:")
    feature_names = ['mood', 'neg_fatigue', 'engagement', 'positive', 'break_tired', 'polar_penalty']
    for name, w in zip(feature_names, irl.theta):
        print(f"  {name}: {w:.3f}")

    # Phase 2: Train policy with PPO
    print("\n[Phase 2] Training policy with PPO...")
    ppo = PPOTrainer()
    policy = ppo.train(env, irl.compute_reward, n_iterations=30, verbose=True)

    # Evaluate
    print("\n[Evaluation]")
    results = ppo.evaluate(env, use_true_reward=True, n_episodes=50)
    print(f"Our Method (IRL+PPO):")
    print(f"  True Reward: {results['mean_return']:.3f} +/- {results['std_return']:.3f}")
    print(f"  Avg Mood: {results['mean_mood']:.3f} +/- {results['std_mood']:.3f}")
    print(f"  Avg Fatigue: {results['mean_fatigue']:.3f} +/- {results['std_fatigue']:.3f}")
