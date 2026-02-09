#!/usr/bin/env python3
"""Detailed comparison of Tier 1 experiments across full token range"""

import re
import matplotlib.pyplot as plt
import numpy as np

TOKENS_PER_ITER = 1_572_864

def parse_log(filename):
    """Extract (iter, tokens, train_loss, val_loss) tuples"""
    pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
    results = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    iter_num = int(match.group(1))
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))
                    tokens = iter_num * TOKENS_PER_ITER
                    results.append((iter_num, tokens, train_loss, val_loss))
    except FileNotFoundError:
        return []

    return results

def main():
    experiments = [
        ("36L r=384 hybrid", "logs/610968_tier1_36L_r384.out", "blue"),
        ("24L r=512 hybrid", "logs/610969_tier1_24L_r512.out", "red"),
        ("24L r=384 full block", "logs/610970_tier1_24L_fb384.out", "green"),
    ]

    # Parse data
    all_data = {}
    for name, logfile, _ in experiments:
        data = parse_log(logfile)
        all_data[name] = data
        if data:
            print(f"{name:<25}: {len(data)} checkpoints, latest {data[-1][1]/1e9:.2f}B tokens, val {data[-1][3]:.4f}")

    # Detailed table
    print("\n" + "="*100)
    print("TIER 1 DETAILED COMPARISON")
    print("="*100)
    print(f"{'Tokens':<10} {'Iters':<8} {'36L r=384':<15} {'24L r=512':<15} {'24L full':<15} {'Best':<20} {'Gap':<10}")
    print("-"*100)

    # Find all unique iterations
    all_iters = set()
    for data in all_data.values():
        all_iters.update([it for it, _, _, _ in data])

    best_per_checkpoint = {}

    for iter_num in sorted(all_iters):
        tokens = iter_num * TOKENS_PER_ITER
        row_data = {}

        for name, _, _ in experiments:
            data = all_data[name]
            matching = [(t, v) for i, t, _, v in data if i == iter_num]
            if matching:
                row_data[name] = matching[0][1]  # val_loss

        if len(row_data) >= 2:  # At least 2 experiments at this checkpoint
            best_name = min(row_data.items(), key=lambda x: x[1])[0]
            best_val = row_data[best_name]
            worst_val = max(row_data.values())
            gap = ((worst_val - best_val) / best_val) * 100

            best_per_checkpoint[tokens] = best_name

            # Format row
            vals = [f"{row_data.get(name, float('nan')):.4f}" if name in row_data else "-"
                    for name, _, _ in experiments]

            print(f"{tokens/1e9:<10.2f} {iter_num:<8} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15} "
                  f"{best_name.split()[0]:<20} {gap:>9.1f}%")

    print("-"*100)

    # Summary statistics
    print("\nWINNER BY TOKEN RANGE:")
    winner_counts = {}
    for tokens in sorted(best_per_checkpoint.keys()):
        winner = best_per_checkpoint[tokens]
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        if tokens > 0:  # Skip initialization
            print(f"  {tokens/1e9:>6.2f}B tokens: {winner}")

    print(f"\nOVERALL WINS:")
    for name in sorted(winner_counts.keys(), key=lambda x: winner_counts[x], reverse=True):
        print(f"  {name:<25}: {winner_counts[name]} checkpoints")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for name, _, color in experiments:
        data = all_data[name]
        if data:
            tokens = [t/1e9 for _, t, _, _ in data]
            train_losses = [tl for _, _, tl, _ in data]
            val_losses = [vl for _, _, _, vl in data]

            ax1.plot(tokens, val_losses, marker='o', label=name, color=color, linewidth=2, markersize=6)
            ax2.plot(tokens, train_losses, marker='s', label=name, color=color, linewidth=2, markersize=6, alpha=0.7)

    ax1.set_xlabel('Tokens Trained (Billions)', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Tier 1 Experiments: Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Tokens Trained (Billions)', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Tier 1 Experiments: Training Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tier1_detailed_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: tier1_detailed_curves.png")

    # Convergence rate analysis
    print("\nCONVERGENCE RATE (1B→6B tokens):")
    for name, _, _ in experiments:
        data = all_data[name]
        if len(data) >= 2:
            # Find ~1B and ~6B checkpoints
            data_1b = [vl for _, t, _, vl in data if 1.4e9 <= t <= 1.8e9]
            data_6b = [vl for _, t, _, vl in data if 5.8e9 <= t <= 6.5e9]

            if data_1b and data_6b:
                improvement = data_1b[0] - data_6b[0]
                rate = improvement / 5.0  # per billion tokens
                print(f"  {name:<25}: {data_1b[0]:.4f} → {data_6b[0]:.4f} "
                      f"({improvement:+.4f}, {rate:.4f}/B tokens)")

if __name__ == "__main__":
    main()
