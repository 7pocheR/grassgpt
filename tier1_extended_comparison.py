#!/usr/bin/env python3
"""Extended comparison of Tier 1 experiments with full continuation data"""

import re
import matplotlib.pyplot as plt

TOKENS_PER_ITER = 1_572_864

def parse_all_logs(log_patterns):
    """Parse multiple log files and combine, removing duplicates"""
    pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
    all_data = {}

    for log_file in log_patterns:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        iter_num = int(match.group(1))
                        train_loss = float(match.group(2))
                        val_loss = float(match.group(3))
                        tokens = iter_num * TOKENS_PER_ITER
                        # Keep latest entry for each iteration
                        all_data[iter_num] = (iter_num, tokens, train_loss, val_loss)
        except FileNotFoundError:
            pass

    # Sort by iteration
    return sorted(all_data.values(), key=lambda x: x[0])

def main():
    experiments = [
        ("36L r=384 hybrid", [
            "logs/610968_tier1_36L_r384.out",
            "logs/610971_tier1_36L_r384_cont.out",
            "logs/610972_tier1_36L_r384_cont.out",
            "logs/610973_tier1_36L_r384_cont.out",
            "logs/610974_tier1_36L_r384_cont.out",
        ], "blue"),
        ("24L r=512 hybrid", [
            "logs/610969_tier1_24L_r512.out",
            "logs/611026_tier1_24L_r512_cont.out",
            "logs/611027_tier1_24L_r512_cont.out",
            "logs/611028_tier1_24L_r512_cont.out",
            "logs/611029_tier1_24L_r512_cont.out",
        ], "red"),
        ("24L r=384 full block", [
            "logs/610970_tier1_24L_fb384.out",
            "logs/611036_tier1_24L_fb_cont.out",
            "logs/611037_tier1_24L_fb_cont.out",
            "logs/611038_tier1_24L_fb_cont.out",
        ], "green"),
    ]

    # Parse all data
    all_data = {}
    for name, log_files, _ in experiments:
        data = parse_all_logs(log_files)
        all_data[name] = data
        if data:
            print(f"{name:<25}: {len(data)} checkpoints, {data[-1][1]/1e9:.1f}B tokens, val {data[-1][3]:.4f}")

    # Detailed comparison table
    print("\n" + "="*110)
    print("TIER 1 EXTENDED COMPARISON")
    print("="*110)
    print(f"{'Tokens':<10} {'Iters':<8} {'36L r=384':<15} {'24L r=512':<15} {'24L full':<15} {'Best':<25} {'Gap':<10}")
    print("-"*110)

    # Find all unique iterations
    all_iters = set()
    for data in all_data.values():
        all_iters.update([it for it, _, _, _ in data])

    best_per_checkpoint = {}

    for iter_num in sorted(all_iters):
        if iter_num % 1000 != 0:  # Only show 1k intervals
            continue

        tokens = iter_num * TOKENS_PER_ITER
        row_data = {}

        for name, _, _ in experiments:
            data = all_data[name]
            matching = [v for i, t, _, v in data if i == iter_num]
            if matching:
                row_data[name] = matching[-1]  # Take last if duplicates

        if len(row_data) >= 2:  # At least 2 experiments
            best_name = min(row_data.items(), key=lambda x: x[1])[0]
            best_val = row_data[best_name]
            worst_val = max(row_data.values())
            gap = ((worst_val - best_val) / best_val) * 100

            best_per_checkpoint[tokens] = best_name

            # Format row
            vals = [f"{row_data.get(name, float('nan')):.4f}" if name in row_data else "-"
                    for name, _, _ in experiments]

            print(f"{tokens/1e9:<10.2f} {iter_num:<8} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15} "
                  f"{best_name.split()[0]:<25} {gap:>9.1f}%")

    print("-"*110)

    # Winner summary
    print("\nWINNER BY TOKEN RANGE:")
    current_winner = None
    range_start = None

    for tokens in sorted(best_per_checkpoint.keys()):
        if tokens == 0:  # Skip init
            continue
        winner = best_per_checkpoint[tokens]
        if winner != current_winner:
            if current_winner:
                print(f"  {range_start/1e9:>6.1f}B - {prev_tokens/1e9:>6.1f}B: {current_winner}")
            current_winner = winner
            range_start = tokens
        prev_tokens = tokens

    if current_winner:
        print(f"  {range_start/1e9:>6.1f}B - {prev_tokens/1e9:>6.1f}B: {current_winner}")

    # Overall statistics
    winner_counts = {}
    for winner in best_per_checkpoint.values():
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    print(f"\nOVERALL WINS (excluding init):")
    total = sum(winner_counts.values()) - 1  # Exclude init
    for name in sorted(winner_counts.keys(), key=lambda x: winner_counts[x], reverse=True):
        count = winner_counts[name]
        if name in best_per_checkpoint.get(0, ""):
            count -= 1  # Don't count init
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {name:<25}: {count:>2}/{total} checkpoints ({pct:>5.1f}%)")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for name, _, color in experiments:
        data = all_data[name]
        if data:
            tokens = [t/1e9 for _, t, _, _ in data if _ > 0]  # Exclude init
            val_losses = [vl for _, t, _, vl in data if _ > 0]
            train_losses = [tl for _, t, tl, _ in data if _ > 0]

            ax1.plot(tokens, val_losses, marker='o', label=name, color=color,
                    linewidth=2, markersize=4, markevery=2)

            ax2.plot(tokens, train_losses, marker='s', label=name, color=color,
                    linewidth=2, markersize=4, markevery=2, alpha=0.7)

    # Add crossover annotations if needed
    ax1.axhline(y=3.1, color='gray', linestyle='--', alpha=0.5, label='3.1 target')

    ax1.set_xlabel('Tokens Trained (Billions)', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Tier 1 Extended: Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Tokens Trained (Billions)', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Tier 1 Extended: Training Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tier1_extended_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: tier1_extended_curves.png")

    # Final checkpoints
    print("\nFINAL CHECKPOINTS:")
    for name, _, _ in experiments:
        data = all_data[name]
        if data:
            final = data[-1]
            print(f"  {name:<25}: iter {final[0]:>5}, {final[1]/1e9:>5.1f}B tokens, "
                  f"train {final[2]:.4f}, val {final[3]:.4f}")

if __name__ == "__main__":
    main()
