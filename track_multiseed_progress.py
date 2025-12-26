#!/usr/bin/env python3
"""Track progress of multi-seed baseline training."""

import re
import os
from pathlib import Path
from glob import glob

def parse_log(log_path):
    """Extract latest iteration and best val loss from log file."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Find all step lines with validation loss
    step_pattern = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")
    matches = list(step_pattern.finditer(content))

    if not matches:
        return None

    # Extract all validation losses
    val_losses = []
    train_losses = []
    steps = []

    for match in matches:
        step = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        steps.append(step)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Find best validation loss
    best_idx = val_losses.index(min(val_losses))

    return {
        'current_step': steps[-1],
        'best_val': val_losses[best_idx],
        'best_train': train_losses[best_idx],
        'best_step': steps[best_idx],
    }

def main():
    print("Tracking multi-seed baseline progress...\n")

    # Find all baseline seed log files
    log_pattern = "logs/*_baseline_seed*.out"
    log_files = sorted(glob(log_pattern))

    if not log_files:
        print("No log files found matching:", log_pattern)
        return

    results = []

    for log_path in log_files:
        # Extract seed number from filename
        match = re.search(r'seed(\d+)\.out', log_path)
        if not match:
            continue

        seed_num = int(match.group(1))
        data = parse_log(log_path)

        if data:
            results.append({
                'seed': seed_num,
                **data
            })

    if not results:
        print("No training data found in log files yet.")
        return

    # Sort by seed number
    results.sort(key=lambda x: x['seed'])

    # Print table
    print(f"{'Seed':<6} {'Best Val':<10} {'Best Train':<12} {'Gen Gap':<10} {'Step @ Best':<12} {'Current':<10} {'Progress':<10}")
    print("-" * 80)

    for r in results:
        gen_gap = r['best_val'] - r['best_train']
        progress = f"{r['current_step']}/15000"
        pct = int(100 * r['current_step'] / 15000)

        print(f"{r['seed']:<6} {r['best_val']:<10.4f} {r['best_train']:<12.4f} "
              f"{gen_gap:<10.4f} {r['best_step']:<12} {progress:<10} {pct:>3}%")

    # Summary statistics
    if len(results) > 0:
        val_losses = [r['best_val'] for r in results]
        import numpy as np

        print("\n" + "="*80)
        print("SUMMARY STATISTICS:")
        print(f"  Completed seeds: {len(results)}/10")
        print(f"  Best val loss (mean): {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"  Best val loss (range): [{np.min(val_losses):.4f}, {np.max(val_losses):.4f}]")

if __name__ == "__main__":
    main()
