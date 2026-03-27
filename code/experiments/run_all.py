"""
Master Script: Run All Training Steps
"""

import subprocess
import sys
import os
import time

print("="*70)
print("MASTER SCRIPT: Running All Experiments")
print("="*70)

scripts = [
    ('train_01_generate_data.py', 'Generating data'),
    ('train_02_preference_irl.py', 'Training Preference IRL'),
    ('train_03_maxent_irl.py', 'Training MaxEnt IRL'),
    ('train_04_rlhf.py', 'Training RLHF'),
    ('train_05_deep_reward.py', 'Training Deep Reward Model'),
    ('train_06_reward_shaping.py', 'Evaluating Reward Shaping'),
    ('aggregate_results.py', 'Aggregating Results'),
    ('generate_plots.py', 'Generating Plots')
]

total_start = time.time()
results = []

for i, (script, description) in enumerate(scripts, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/{len(scripts)}] {description}")
    print(f"Running: {script}")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True,
            timeout=600  # 10 minute timeout per script
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            results.append((script, 'SUCCESS', elapsed))
            print(f"\n[SUCCESS] {script} completed in {elapsed:.1f}s")
        else:
            results.append((script, 'FAILED', elapsed))
            print(f"\n[FAILED] {script} failed (exit code {result.returncode})")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        results.append((script, 'TIMEOUT', elapsed))
        print(f"\n[TIMEOUT] {script} timed out after {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        results.append((script, f'ERROR: {str(e)}', elapsed))
        print(f"\n[ERROR] {script}: {str(e)}")

total_time = time.time() - total_start

# Print summary
print("\n" + "="*70)
print("EXECUTION SUMMARY")
print("="*70)
print(f"{'Script':<35} {'Status':<15} {'Time':<10}")
print("-"*70)

for script, status, elapsed in results:
    print(f"{script:<35} {status:<15} {elapsed:.1f}s")

print("-"*70)
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

# Check if all succeeded
successes = sum(1 for _, status, _ in results if status == 'SUCCESS')
print(f"\nCompleted: {successes}/{len(scripts)} scripts successful")

if successes == len(scripts):
    print("\n[ALL COMPLETE] All experiments finished successfully!")
    print("\nOutput files:")
    print("  - results/modular/AGGREGATED_RESULTS.json")
    print("  - results/modular/SUMMARY_REPORT.txt")
    print("  - results/modular/figures/*.png")
else:
    print("\n[WARNING] Some scripts failed. Check output above for details.")
