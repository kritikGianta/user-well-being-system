# Well-being Aware Social Media Recommendation Using IRL

## Final Research Package - Complete Documentation

**Title**: Aligning Social Media Recommendation Systems with Human Well-being: A Comprehensive Evaluation of Inverse Reinforcement Learning Approaches

**Version**: Final (v2.0)
**Last Updated**: March 2026
**Status**: Ready for Submission

---

## Abstract

This research provides a comprehensive empirical evaluation of Inverse Reinforcement Learning (IRL) methods for well-being-aware social media recommendation. Through experiments on KuaiRec 2.0 (69,047 trajectories, 10,000 preference pairs from 500 users), we systematically compare 8 methods including ML-IRL, PB-IRL, MaxEnt-IRL, complete RLHF with PPO, and reward shaping baselines.

**Key Findings**:
- Domain-informed reward shaping achieves 70.2% ranking accuracy
- Complete RLHF with PPO achieves 73.1% accuracy (vs 34% for simplified RLHF)
- Well-being policies maintain 0.384 final well-being vs 0.035 for engagement-greedy
- Policy rollout reveals 2.3x longer sessions with well-being optimization

---

## Directory Structure

```
final_files/
|
+-- README.md                    # This file
+-- LICENSE                      # MIT License
|
+-- paper/                       # Research Papers
|   +-- FINAL_RESEARCH_PAPER.docx       # Main paper (ALL improvements)
|   +-- COMBINED_RESEARCH_PAPER.docx    # Combined paper (v1)
|   +-- FINAL_REPORT_UPDATED.docx       # Technical report
|   +-- 03_RESEARCH_PAPER.docx          # Conference paper
|   +-- AAAI_PAPER.docx                 # AAAI format
|
+-- code/                        # Source Code
|   +-- core/                    # Core implementations
|   |   +-- irl_methods.py              # All IRL methods (ML-IRL, PB-IRL, MaxEnt, RLHF)
|   |   +-- rlhf_ppo.py                 # Complete RLHF with PPO
|   |   +-- policy_rollout.py           # Policy simulation & analysis
|   |   +-- qualitative_examples.py     # Trajectory comparison generator
|   |   +-- environment.py              # MDP environment
|   |   +-- maxent_irl.py               # MaxEnt IRL implementation
|   |   +-- preference_based_irl.py     # Preference-based IRL
|   |   +-- ppo.py                      # PPO algorithm
|   |   +-- deep_reward_model.py        # Neural reward networks
|   |
|   +-- baselines/               # Baseline methods
|   |   +-- reward_shaping_baselines.py # Simple/Adaptive/Optimized shaping
|   |   +-- rlhf_baseline.py            # Simplified RLHF baseline
|   |
|   +-- evaluation/              # Evaluation code
|   |   +-- counterfactual.py           # IPS, SNIPS, CIPS, DR estimators
|   |
|   +-- experiments/             # Experiment scripts
|       +-- comprehensive_comparison_experiment.py
|       +-- run_all.py
|
+-- data/                        # Data Files
|   +-- raw/                     # Raw experimental data
|   |   +-- demonstrations.json
|   |   +-- preference_data.json
|   |   +-- state_action_pairs.json
|   |
|   +-- results/                 # Experiment results
|   |   +-- rlhf_results.json           # RLHF with PPO results
|   |   +-- training_curves.json        # Training dynamics
|   |   +-- comparison_summary.json     # Method comparisons
|   |   +-- trajectory_comparisons.json # Qualitative examples
|   |   +-- policy_comparison_table.json
|   |   +-- counterfactual_evaluation_results.json
|   |   +-- (+ various experiment results)
|   |
|   +-- processed/               # Processed data
|       +-- recsys_citations.json
|       +-- contribution_reframing.json
|       +-- technical_investigation.json
|
+-- figures/                     # All Figures and Plots
|   +-- policy_rollout/          # Policy analysis figures
|   |   +-- fatigue_over_time.png
|   |   +-- engagement_burnout_tradeoff.png
|   |   +-- wellbeing_trajectory.png
|   |
|   +-- training_curves/         # Training visualizations
|   |
|   +-- comparisons/             # Method comparison plots
|   |   +-- 01_maxent_loss_curve.png
|   |   +-- 02_rlhf_loss_dynamics.png
|   |   +-- 03_rlhf_accuracy_dynamics.png
|   |   +-- 04_reward_shaping_comparison.png
|   |   +-- 05_method_accuracy_comparison.png
|   |   +-- 06_training_time_comparison.png
|   |   +-- 07_user_type_distribution.png
|   |   +-- 08_engagement_wellbeing_tradeoff.png
|   |   +-- (+ more comparison figures)
|   |
|   +-- ablations/               # Ablation study figures
|       +-- ablation_analysis.png
|       +-- cold_start_analysis.png
|       +-- diversity_analysis.png
|       +-- statistical_analysis.png
|       +-- threshold_analysis.png
|
+-- tables/                      # Table Data
|   +-- policy_comparison_table.json
|   +-- PAPER_TABLES_TEMPLATE.md
|
+-- configs/                     # Configuration Files
|   +-- hyperparameters.yaml
|   +-- hyperparameters.json
|   +-- requirements.txt
|
+-- documentation/               # Documentation
|   +-- IMPROVEMENTS_README.md
|   +-- PAPER_UPDATE_SUMMARY.txt
|   +-- IMPROVEMENTS_MASTER_SUMMARY.txt
|   +-- trajectory_comparisons.md
|   +-- MATHEMATICAL_FORMULATIONS.md
|   +-- QUICKSTART.md
|
+-- supplementary/               # Supplementary Materials
    +-- trajectory_figure.tex          # LaTeX figure code
    +-- additional_recsys_citations.bib
    +-- citations_to_add.bib
```

---

## Quick Start

### Installation

```bash
cd final_files
pip install -r configs/requirements.txt
```

### Requirements

```
numpy>=1.21.0
torch>=1.10.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
python-docx>=0.8.11
scipy>=1.7.0
```

### Running Core Experiments

```python
# 1. Run IRL Methods Comparison
from code.core.irl_methods import MLIRL, PBIRL, SimplifiedRLHF, OptimizedShaping

# 2. Run Policy Rollout Analysis
python code/core/policy_rollout.py

# 3. Generate Qualitative Examples
python code/core/qualitative_examples.py

# 4. Run Full RLHF with PPO
python code/core/rlhf_ppo.py

# 5. Run Counterfactual Evaluation
from code.evaluation.counterfactual import CounterfactualEvaluator
```

---

## Main Results Summary

### Method Comparison (Ranking Accuracy)

| Method | Accuracy | 95% CI | Group |
|--------|----------|--------|-------|
| Optimized Shaping | 70.2% | [68.1, 72.2] | A |
| PB-IRL | 68.85% | [66.7, 70.9] | A |
| Full RLHF (PPO) | 73.1%* | - | A |
| Adaptive Shaping | 65.3% | [63.1, 67.4] | A |
| Deep Reward Model | 61.5% | [59.3, 63.7] | B |
| ML-IRL | 56.2% | [54.0, 58.5] | B |
| MaxEnt-IRL | 52.3% | [50.0, 54.5] | B |
| Simplified RLHF | 34.0% | [31.8, 36.2] | C |

*Full RLHF accuracy is reward model accuracy; policy achieves 0.598 well-being score.

### Policy Rollout Results (50-step sessions, 100 users)

| Policy | Avg Session | Final Fatigue | Final W(s) | Peak Burnout |
|--------|-------------|---------------|------------|--------------|
| Engagement Greedy | 16.9 | 0.873 | 0.035 | 0.847 |
| Diversity | 28.3 | 0.512 | 0.218 | 0.412 |
| **Well-being (Ours)** | **38.1** | **0.234** | **0.384** | **0.152** |
| IRL Learned | 41.2 | 0.198 | 0.421 | 0.123 |

### Qualitative Trajectory Analysis (5 Case Studies)

| Metric | Engagement-Greedy | Well-being Policy | Improvement |
|--------|-------------------|-------------------|-------------|
| Win Rate | 0/5 | **5/5** | +100% |
| Avg Well-being | 0.224 | **0.384** | +0.160 |
| Avg Watch Time | 46.5 min | **86.4 min** | +39.9 min |
| Session Outcome | Moderate fatigue | Satisfied | Better |

---

## Key Contributions

1. **Comprehensive IRL Evaluation**: First large-scale comparison of 8 IRL methods on real social media data (KuaiRec 2.0)

2. **Policy Rollout Analysis**: Novel visualization of engagement-burnout trade-offs over extended sessions

3. **Complete RLHF Pipeline**: Full implementation with PPO (vs simplified reward-model-only variants)

4. **Qualitative Evidence**: Concrete trajectory comparisons demonstrating session improvements

5. **Counterfactual Evaluation**: IPS, SNIPS, CIPS, and Doubly Robust estimators for robust offline assessment

6. **Reproducibility**: Complete code release with hyperparameters, configs, and trained models

---

## Well-being Score and Evaluation

**Training Reward Signal (R_shaping)**:
```
R_shaping(s) = 0.3 * engagement + 0.4 * diversity - 0.3 * fatigue
```

**Evaluation Metric (Wellbeing Proxy)**:
```
Wellbeing(s) = 0.4 * mood_proxy + 0.3 * engagement_quality - 0.3 * fatigue
```

Where mood_proxy = 0.4·engagement + 0.3·dwell_time - 0.3·skip_rate

> Note: The training reward and evaluation metric are intentionally different to avoid circular evaluation.

Feature definitions:
- **Engagement**: Normalized interaction intensity (0-1)
- **Diversity**: Content category entropy (0-1)
- **Fatigue**: Accumulated session burden (0-1)
- **Mood Proxy**: Behavioral composite from watch patterns (0-1)

---

## Citation

```bibtex
@inproceedings{wellbeing_irl_2024,
  title={Aligning Social Media Recommendation Systems with Human Well-being:
         A Comprehensive Evaluation of Inverse Reinforcement Learning Approaches},
  author={Anonymous},
  booktitle={Proceedings of [Conference]},
  year={2024}
}
```

---

## File Descriptions

### Papers

| File | Description |
|------|-------------|
| `FINAL_RESEARCH_PAPER.docx` | **Main paper with ALL improvements** (recommended) |
| `COMBINED_RESEARCH_PAPER.docx` | Combined paper version 1 |
| `FINAL_REPORT_UPDATED.docx` | Detailed technical report |
| `AAAI_PAPER.docx` | AAAI conference format |

### Key Code Files

| File | Description |
|------|-------------|
| `irl_methods.py` | All IRL implementations (ML-IRL, PB-IRL, MaxEnt, RLHF, etc.) |
| `rlhf_ppo.py` | **Complete RLHF with PPO** (fixes 34% issue) |
| `policy_rollout.py` | Policy simulation with fatigue/burnout analysis |
| `qualitative_examples.py` | Trajectory comparison generator |
| `counterfactual.py` | IPS, SNIPS, CIPS, DR estimators |

### Key Figures

| File | Description |
|------|-------------|
| `fatigue_over_time.png` | Fatigue accumulation curves (4 policies) |
| `engagement_burnout_tradeoff.png` | Dual-axis engagement vs burnout |
| `wellbeing_trajectory.png` | Well-being score over time |
| `method_accuracy_comparison.png` | Main results comparison |

---

## Hyperparameters

### IRL Methods
```yaml
learning_rate: 0.001
batch_size: 32
hidden_dims: [64, 64]
training_epochs: 200
optimizer: Adam
early_stopping: true
```

### RLHF with PPO
```yaml
reward_model:
  lr: 0.001
  hidden_dim: 128
  epochs: 50

ppo:
  lr_actor: 0.0003
  lr_critic: 0.001
  clip_epsilon: 0.2
  gae_lambda: 0.95
  entropy_coef: 0.01
  iterations: 50
  steps_per_iteration: 512
```

### Reward Shaping
```yaml
simple: {E: 0.2, D: 0.3, F: -0.5}
adaptive: {E: f(fatigue), D: 0.3, F: -0.5}
optimized: {E: 0.3, D: 0.4, F: -0.4}
```

---

## Reproducibility

All experiments use fixed random seeds (42) for reproducibility.

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
```

Computational requirements:
- **CPU**: Intel Core i7 or equivalent
- **RAM**: 16GB
- **GPU**: Not required (all CPU-trainable)
- **Time**: ~2 hours for full experimental suite

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions, issues, or collaboration inquiries, please contact the authors.

---

**Research Package Complete** - All code, data, figures, and documentation included.
