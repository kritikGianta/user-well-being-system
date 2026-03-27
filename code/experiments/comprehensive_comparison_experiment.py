"""
Comprehensive Comparison Experiment
Addresses all reviewer feedback items for the research paper.

This experiment:
1. Uses noisy environment with realistic effect sizes (30-50% reduced)
2. Generates semi-real dataset proxy with Reddit/Twitter characteristics
3. Trains preference-based IRL (instead of expert demonstrations)
4. Compares with RLHF baseline
5. Implements reward shaping baselines
6. Uses deep neural network reward model
7. Adds multi-session user modeling
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

# Import all modules
from noisy_environment import NoisySocialMediaEnv, generate_preference_based_data
from semi_real_dataset import SemiRealDatasetProxy
from preference_based_irl import PreferenceBasedIRL, MaximumLikelihoodIRL, evaluate_preference_accuracy
from rlhf_baseline import RLHFTrainer, RewardModelNetwork
from reward_shaping_baselines import (
    SimpleRewardShaping, AdaptiveRewardShaping,
    WellBeingRewardShaping, OptimizedIRLRewardShaping
)
from deep_reward_model import DeepRewardModel
from multi_session_modeling import MultiSessionUserModel, MultiSessionDataGenerator
from maxent_irl import MaxEntIRL


class ComprehensiveExperiment:
    """Main experiment class integrating all components."""

    def __init__(self, output_dir: str = "results/comprehensive_comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

        # Results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'experiment_type': 'comprehensive_comparison'
            },
            'data_generation': {},
            'models': {},
            'evaluation': {},
            'comparisons': {}
        }

        # Initialize environment
        self.noisy_env = NoisySocialMediaEnv(noise_level='moderate')

    def run_step1_data_generation(self):
        """Step 1: Generate all required datasets."""
        print("\n" + "="*60)
        print("STEP 1: DATA GENERATION")
        print("="*60)

        # 1.1: Generate semi-real dataset
        print("\n[1.1] Generating Semi-Real Dataset Proxy...")
        dataset_proxy = SemiRealDatasetProxy(n_users=500)
        sessions_df = dataset_proxy.generate_sessions()

        # Save dataset
        sessions_df.to_csv(os.path.join(self.output_dir, "semi_real_sessions.csv"), index=False)

        self.results['data_generation']['semi_real'] = {
            'n_users': 500,
            'n_sessions': len(sessions_df),
            'user_types': sessions_df['user_type'].value_counts().to_dict()
        }
        print(f"[OK] Generated {len(sessions_df)} sessions from 500 users")

        # 1.2: Generate preference-based data
        print("\n[1.2] Generating Preference-Based Data...")
        self.preference_data = generate_preference_based_data(
            env=self.noisy_env,
            n_episodes=200,
            n_comparisons_per_episode=5
        )

        self.results['data_generation']['preferences'] = {
            'n_pairs': len(self.preference_data),
            'avg_confidence': float(np.mean([p['confidence'] for p in self.preference_data]))
        }
        print(f"[OK] Generated {len(self.preference_data)} preference pairs")

        # 1.3: Generate demonstration data (for baselines)
        print("\n[1.3] Generating Expert Demonstrations...")
        self.demonstrations = self._generate_demonstrations(n_demos=200)

        self.results['data_generation']['demonstrations'] = {
            'n_demos': len(self.demonstrations),
            'avg_length': np.mean([len(d['states']) for d in self.demonstrations])
        }
        print(f"[OK] Generated {len(self.demonstrations)} demonstrations")

        # 1.4: Generate multi-session data
        print("\n[1.4] Generating Multi-Session Data...")
        multi_session_gen = MultiSessionDataGenerator(n_users=100)
        self.multi_session_data = multi_session_gen.generate_longitudinal_data(
            sessions_per_user=10,
            days_span=30
        )

        self.results['data_generation']['multi_session'] = {
            'n_users': 100,
            'total_sessions': sum(len(u['sessions']) for u in self.multi_session_data),
            'avg_sessions_per_user': np.mean([len(u['sessions']) for u in self.multi_session_data])
        }
        print(f"[OK] Generated multi-session data for 100 users")

    def _generate_demonstrations(self, n_demos: int = 200) -> List[Dict]:
        """Generate expert demonstrations using well-being maximizing policy."""
        demonstrations = []

        for _ in range(n_demos):
            states = []
            actions = []
            rewards = []

            state = self.noisy_env.reset()
            done = False
            step = 0

            while not done and step < 30:
                # Expert policy: maximize well-being
                mood, engagement, fatigue = state

                if fatigue > 0.7:
                    action = 4  # Take break
                elif mood < 0.4:
                    action = 0  # Positive content
                elif engagement < 0.3:
                    action = 3  # Interest-based
                else:
                    action = np.random.choice([0, 1, 3])  # Mix of good actions

                states.append(state.copy())
                actions.append(action)

                state, reward, done, _ = self.noisy_env.step(action)
                rewards.append(reward)
                step += 1

            demonstrations.append({
                'states': states,
                'actions': actions,
                'rewards': rewards
            })

        return demonstrations

    def run_step2_model_training(self):
        """Step 2: Train all models."""
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING")
        print("="*60)

        # 2.1: Preference-Based IRL
        print("\n[2.1] Training Preference-Based IRL...")
        start_time = time.time()
        self.pb_irl = PreferenceBasedIRL(state_dim=3, n_features=8)
        pb_history = self.pb_irl.train(self.preference_data, n_epochs=100, verbose=False)
        pb_time = time.time() - start_time

        pb_eval = evaluate_preference_accuracy(self.pb_irl, self.preference_data[:200])
        print(f"[OK] PB-IRL trained in {pb_time:.2f}s, accuracy: {pb_eval['accuracy']:.3f}")

        self.results['models']['pb_irl'] = {
            'training_time': pb_time,
            'accuracy': pb_eval['accuracy'],
            'weights': self.pb_irl.reward_weights.tolist()
        }

        # 2.2: Maximum Likelihood IRL
        print("\n[2.2] Training Maximum Likelihood IRL...")
        start_time = time.time()
        self.ml_irl = MaximumLikelihoodIRL(state_dim=3, feature_dim=8)
        ml_history = self.ml_irl.train(self.preference_data, n_epochs=100, verbose=False)
        ml_time = time.time() - start_time

        ml_eval = evaluate_preference_accuracy(self.ml_irl, self.preference_data[:200])
        print(f"[OK] ML-IRL trained in {ml_time:.2f}s, accuracy: {ml_eval['accuracy']:.3f}")

        self.results['models']['ml_irl'] = {
            'training_time': ml_time,
            'accuracy': ml_eval['accuracy'],
            'weights': self.ml_irl.reward_weights.tolist()
        }

        # 2.3: RLHF Baseline
        print("\n[2.3] Training RLHF Baseline...")
        start_time = time.time()
        self.rlhf = RLHFTrainer(state_dim=3, action_dim=5)
        rlhf_history = self.rlhf.train_reward_model(self.preference_data, n_epochs=50)
        rlhf_time = time.time() - start_time

        rlhf_accuracy = rlhf_history['accuracy'][-1] if rlhf_history['accuracy'] else 0.0
        print(f"[OK] RLHF trained in {rlhf_time:.2f}s, accuracy: {rlhf_accuracy:.3f}")

        self.results['models']['rlhf'] = {
            'training_time': rlhf_time,
            'accuracy': rlhf_accuracy,
            'history': {k: [float(v) for v in vals] for k, vals in rlhf_history.items()}
        }

        # 2.4: Deep Reward Model
        print("\n[2.4] Training Deep Reward Model...")
        start_time = time.time()
        self.deep_rm = DeepRewardModel(state_dim=3, action_dim=5, hidden_dims=[128, 64, 32])
        deep_history = self.deep_rm.train_from_preferences(
            self.preference_data,
            n_epochs=100,
            verbose=False
        )
        deep_time = time.time() - start_time

        deep_accuracy = deep_history['accuracy'][-1] if deep_history.get('accuracy') else 0.0
        print(f"[OK] Deep RM trained in {deep_time:.2f}s, accuracy: {deep_accuracy:.3f}")

        self.results['models']['deep_rm'] = {
            'training_time': deep_time,
            'accuracy': deep_accuracy,
            'architecture': [3, 128, 64, 32, 1],
            'history': deep_history
        }

        # 2.5: Traditional MaxEnt IRL (for comparison)
        print("\n[2.5] Training Traditional MaxEnt IRL (baseline)...")
        start_time = time.time()
        self.maxent_irl = MaxEntIRL(state_dim=3, action_dim=5, n_features=12)

        # Convert demonstrations for MaxEnt
        states_list = []
        actions_list = []
        for demo in self.demonstrations[:100]:
            for state, action in zip(demo['states'], demo['actions']):
                states_list.append(state)
                actions_list.append(action)

        maxent_history = self.maxent_irl.train(
            np.array(states_list),
            np.array(actions_list),
            env=self.noisy_env,
            n_iterations=50,
            verbose=False
        )
        maxent_time = time.time() - start_time
        print(f"[OK] MaxEnt IRL trained in {maxent_time:.2f}s")

        self.results['models']['maxent_irl'] = {
            'weights': self.maxent_irl.theta.tolist(),
            'training_time': maxent_time
        }

        # 2.6: Reward Shaping Baselines (no training needed)
        print("\n[2.6] Initializing Reward Shaping Baselines...")
        self.reward_shapers = {
            'simple': SimpleRewardShaping(),
            'adaptive': AdaptiveRewardShaping(),
            'wellbeing': WellBeingRewardShaping(),
            'optimized_irl': OptimizedIRLRewardShaping()
        }
        print(f"[OK] Initialized {len(self.reward_shapers)} reward shaping baselines")

    def run_step3_evaluation(self):
        """Step 3: Comprehensive evaluation of all models."""
        print("\n" + "="*60)
        print("STEP 3: EVALUATION")
        print("="*60)

        # 3.1: Preference prediction accuracy
        print("\n[3.1] Evaluating Preference Prediction...")
        test_prefs = self.preference_data[800:]  # Last 200 for testing

        preference_results = {}

        # PB-IRL
        pb_eval = evaluate_preference_accuracy(self.pb_irl, test_prefs)
        preference_results['pb_irl'] = pb_eval

        # ML-IRL
        ml_eval = evaluate_preference_accuracy(self.ml_irl, test_prefs)
        preference_results['ml_irl'] = ml_eval

        print(f"[OK] PB-IRL accuracy: {pb_eval['accuracy']:.3f}")
        print(f"[OK] ML-IRL accuracy: {ml_eval['accuracy']:.3f}")

        self.results['evaluation']['preference_accuracy'] = preference_results

        # 3.2: Well-being metrics evaluation
        print("\n[3.2] Evaluating Well-Being Metrics...")
        wellbeing_results = self._evaluate_wellbeing_metrics()
        self.results['evaluation']['wellbeing'] = wellbeing_results

        # 3.3: Multi-session evaluation
        print("\n[3.3] Evaluating Multi-Session Performance...")
        multi_session_results = self._evaluate_multi_session()
        self.results['evaluation']['multi_session'] = multi_session_results

    def _evaluate_wellbeing_metrics(self) -> Dict:
        """Evaluate models on well-being metrics."""
        results = {}
        n_episodes = 50

        models = {
            'pb_irl': self.pb_irl,
            'ml_irl': self.ml_irl,
            'simple_shaping': self.reward_shapers['simple'],
            'adaptive_shaping': self.reward_shapers['adaptive'],
            'wellbeing_shaping': self.reward_shapers['wellbeing']
        }

        for name, model in models.items():
            metrics = {'mood': [], 'engagement': [], 'fatigue': [], 'total_reward': []}

            for _ in range(n_episodes):
                state = self.noisy_env.reset()
                total_reward = 0
                episode_moods = []
                episode_engagement = []
                episode_fatigue = []

                for step in range(30):
                    # Get action based on model type
                    if hasattr(model, 'compute_reward'):
                        # IRL model - choose action with highest reward
                        q_values = [model.compute_reward(state, a) if hasattr(model, 'compute_reward') and callable(getattr(model, 'compute_reward')) else 0 for a in range(5)]
                        action = np.argmax(q_values)
                    elif hasattr(model, 'get_shaped_reward'):
                        # Reward shaping - use greedy policy
                        rewards = [model.get_shaped_reward(state, a, state) for a in range(5)]
                        action = np.argmax(rewards)
                    else:
                        action = np.random.randint(0, 5)

                    next_state, reward, done, _ = self.noisy_env.step(action)
                    total_reward += reward

                    episode_moods.append(state[0])
                    episode_engagement.append(state[1])
                    episode_fatigue.append(state[2])

                    state = next_state
                    if done:
                        break

                metrics['mood'].append(np.mean(episode_moods))
                metrics['engagement'].append(np.mean(episode_engagement))
                metrics['fatigue'].append(np.mean(episode_fatigue))
                metrics['total_reward'].append(total_reward)

            results[name] = {
                'avg_mood': float(np.mean(metrics['mood'])),
                'avg_engagement': float(np.mean(metrics['engagement'])),
                'avg_fatigue': float(np.mean(metrics['fatigue'])),
                'avg_reward': float(np.mean(metrics['total_reward'])),
                'std_reward': float(np.std(metrics['total_reward']))
            }
            print(f"  {name}: mood={results[name]['avg_mood']:.3f}, reward={results[name]['avg_reward']:.2f}")

        return results

    def _evaluate_multi_session(self) -> Dict:
        """Evaluate habituation and long-term effects."""
        results = {
            'habituation_detected': False,
            'addiction_risk_users': 0,
            'session_metrics': []
        }

        for user_data in self.multi_session_data[:20]:  # Sample 20 users
            sessions = user_data['sessions']

            # Track engagement over sessions
            engagement_trend = [s['avg_engagement'] for s in sessions]

            # Check for habituation (decreasing engagement)
            if len(engagement_trend) > 3:
                slope = np.polyfit(range(len(engagement_trend)), engagement_trend, 1)[0]
                if slope < -0.01:
                    results['habituation_detected'] = True

            # Check addiction risk
            if user_data.get('addiction_score', 0) > 0.7:
                results['addiction_risk_users'] += 1

        print(f"  Habituation detected: {results['habituation_detected']}")
        print(f"  Addiction risk users: {results['addiction_risk_users']}/20")

        return results

    def run_step4_visualization(self):
        """Step 4: Generate all figures for the paper."""
        print("\n" + "="*60)
        print("STEP 4: VISUALIZATION")
        print("="*60)

        fig_dir = os.path.join(self.output_dir, "figures")

        # Figure 1: Model Comparison Bar Chart
        print("\n[4.1] Generating Model Comparison Figure...")
        self._plot_model_comparison(fig_dir)

        # Figure 2: Preference Accuracy Over Training
        print("\n[4.2] Generating Training Curves...")
        self._plot_training_curves(fig_dir)

        # Figure 3: Well-being Metrics Comparison
        print("\n[4.3] Generating Well-being Comparison...")
        self._plot_wellbeing_comparison(fig_dir)

        # Figure 4: Multi-Session Analysis
        print("\n[4.4] Generating Multi-Session Analysis...")
        self._plot_multi_session_analysis(fig_dir)

        # Figure 5: Reward Weight Visualization
        print("\n[4.5] Generating Reward Weight Visualization...")
        self._plot_reward_weights(fig_dir)

        print(f"\n[OK] All figures saved to {fig_dir}")

    def _plot_model_comparison(self, fig_dir: str):
        """Bar chart comparing all models."""
        models = list(self.results['evaluation']['wellbeing'].keys())
        rewards = [self.results['evaluation']['wellbeing'][m]['avg_reward'] for m in models]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, rewards, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
        plt.xlabel('Model')
        plt.ylabel('Average Episode Reward')
        plt.title('Model Comparison: Average Reward')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'model_comparison.png'), dpi=150)
        plt.close()

    def _plot_training_curves(self, fig_dir: str):
        """Training curves for IRL models."""
        plt.figure(figsize=(10, 6))

        if 'history' in self.results['models'].get('rlhf', {}):
            history = self.results['models']['rlhf']['history']
            if 'accuracy' in history:
                plt.plot(history['accuracy'], label='RLHF', linewidth=2)

        if 'history' in self.results['models'].get('deep_rm', {}):
            history = self.results['models']['deep_rm']['history']
            if 'accuracy' in history:
                plt.plot(history['accuracy'], label='Deep RM', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Preference Prediction Accuracy')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'training_curves.png'), dpi=150)
        plt.close()

    def _plot_wellbeing_comparison(self, fig_dir: str):
        """Radar chart for well-being metrics."""
        metrics = ['avg_mood', 'avg_engagement', 'avg_fatigue', 'avg_reward']
        models = list(self.results['evaluation']['wellbeing'].keys())[:3]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.25

        for i, model in enumerate(models):
            values = [self.results['evaluation']['wellbeing'][model][m] for m in metrics]
            # Normalize for display
            values = [v if m != 'avg_reward' else v/10 for v, m in zip(values, metrics)]
            ax.bar(x + i*width, values, width, label=model)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Value (reward scaled by 0.1)')
        ax.set_title('Well-being Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Mood', 'Engagement', 'Fatigue', 'Reward/10'])
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'wellbeing_comparison.png'), dpi=150)
        plt.close()

    def _plot_multi_session_analysis(self, fig_dir: str):
        """Multi-session engagement trends."""
        plt.figure(figsize=(10, 6))

        # Plot engagement trends for several users
        for i, user_data in enumerate(self.multi_session_data[:5]):
            sessions = user_data['sessions']
            engagement = [s['avg_engagement'] for s in sessions]
            plt.plot(engagement, label=f"User {i+1}", alpha=0.7)

        plt.xlabel('Session Number')
        plt.ylabel('Average Engagement')
        plt.title('Multi-Session Engagement Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'multi_session_trends.png'), dpi=150)
        plt.close()

    def _plot_reward_weights(self, fig_dir: str):
        """Visualize learned reward weights."""
        plt.figure(figsize=(12, 6))

        # Get weights from PB-IRL
        weights = self.results['models']['pb_irl']['weights']
        feature_names = ['mood', 'engagement', 'fatigue', 'mood_sq',
                        'pos_action', 'polar_action', 'break_action', 'interact'][:len(weights)]

        colors = ['green' if w > 0 else 'red' for w in weights]
        plt.barh(feature_names, weights, color=colors)
        plt.xlabel('Weight Value')
        plt.title('Learned Reward Weights (PB-IRL)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'reward_weights.png'), dpi=150)
        plt.close()

    def run_step5_save_results(self):
        """Step 5: Save all results."""
        print("\n" + "="*60)
        print("STEP 5: SAVING RESULTS")
        print("="*60)

        # Save JSON results
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[OK] Results saved to {results_path}")

        # Generate summary report
        self._generate_summary_report()

    def _generate_summary_report(self):
        """Generate human-readable summary."""
        report_path = os.path.join(self.output_dir, "SUMMARY_REPORT.txt")

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("COMPREHENSIVE COMPARISON EXPERIMENT - SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")

            f.write(f"Generated: {self.results['metadata']['timestamp']}\n\n")

            f.write("DATA GENERATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Semi-real sessions: {self.results['data_generation']['semi_real']['n_sessions']}\n")
            f.write(f"Preference pairs: {self.results['data_generation']['preferences']['n_pairs']}\n")
            f.write(f"Expert demonstrations: {self.results['data_generation']['demonstrations']['n_demos']}\n\n")

            f.write("MODEL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            for model, data in self.results['models'].items():
                if 'accuracy' in data:
                    f.write(f"{model}: accuracy={data['accuracy']:.3f}, time={data['training_time']:.1f}s\n")
                elif 'training_time' in data:
                    f.write(f"{model}: time={data['training_time']:.1f}s\n")

            f.write("\nWELL-BEING EVALUATION\n")
            f.write("-"*40 + "\n")
            for model, metrics in self.results['evaluation']['wellbeing'].items():
                f.write(f"{model}: mood={metrics['avg_mood']:.3f}, reward={metrics['avg_reward']:.2f}\n")

            f.write("\nMULTI-SESSION ANALYSIS\n")
            f.write("-"*40 + "\n")
            ms = self.results['evaluation']['multi_session']
            f.write(f"Habituation detected: {ms['habituation_detected']}\n")
            f.write(f"Addiction risk users: {ms['addiction_risk_users']}\n")

            f.write("\n" + "="*60 + "\n")
            f.write("EXPERIMENT COMPLETED SUCCESSFULLY\n")
            f.write("="*60 + "\n")

        print(f"[OK] Summary report saved to {report_path}")

    def run_all(self):
        """Run the complete experiment."""
        print("\n" + "="*60)
        print("COMPREHENSIVE COMPARISON EXPERIMENT")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.run_step1_data_generation()
            self.run_step2_model_training()
            self.run_step3_evaluation()
            self.run_step4_visualization()
            self.run_step5_save_results()

            print("\n" + "="*60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Results saved to: {self.output_dir}")
            print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    experiment = ComprehensiveExperiment()
    experiment.run_all()
