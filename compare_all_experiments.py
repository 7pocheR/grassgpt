#!/usr/bin/env python3
"""Compare all experiments: Baseline + Previous Grassmann + Tier 1"""

import re
import sys

def parse_log(filename, tokens_per_iter):
    """Extract (iter, tokens, train_loss, val_loss) tuples from log file"""
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
                    tokens = iter_num * tokens_per_iter
                    results.append((iter_num, tokens, train_loss, val_loss))
    except FileNotFoundError:
        return []

    return results

def format_tokens(tokens):
    """Format token count in billions"""
    return f"{tokens / 1e9:.2f}B"

def find_closest(data, target_tokens, tolerance=0.5e9):
    """Find closest data point to target token count within tolerance"""
    best = None
    best_diff = float('inf')

    for iter_num, tokens, train_loss, val_loss in data:
        diff = abs(tokens - target_tokens)
        if diff < best_diff and diff <= tolerance:
            best_diff = diff
            best = (iter_num, tokens, train_loss, val_loss)

    return best

def main():
    # Token counts per iteration
    BASELINE_TOKENS = 3_145_728  # batch=64, grad_accum=12, 4 GPUs
    GRASSMANN_TOKENS = 1_572_864  # batch=32, grad_accum=12, 4 GPUs (or 16×24)

    experiments = [
        ("Baseline (24L AdamW)", "logs/603344_gpt2m_base_gate.out", BASELINE_TOKENS),
        ("Grass elementwise r=384 (24L)", "logs/605987_gpt2m_grass_r384.out", GRASSMANN_TOKENS),
        ("Grass hybrid r=384 (24L)", "logs/608059_gpt2m_hybgate_r384.out", GRASSMANN_TOKENS),
        ("Tier1: 36L r=384 hybrid", "logs/610968_tier1_36L_r384.out", GRASSMANN_TOKENS),
        ("Tier1: 24L r=512 hybrid", "logs/610969_tier1_24L_r512.out", GRASSMANN_TOKENS),
        ("Tier1: 24L r=384 full block", "logs/610970_tier1_24L_fb384.out", GRASSMANN_TOKENS),
    ]

    # Parse all logs
    all_data = {}
    print("Loading data...")
    for name, logfile, tokens_per_iter in experiments:
        data = parse_log(logfile, tokens_per_iter)
        all_data[name] = data
        if data:
            print(f"  {name:<35}: {len(data):>3} checkpoints, up to {format_tokens(data[-1][1]):>7}")
        else:
            print(f"  {name:<35}: No data")

    print("\n" + "="*140)
    print("TOKEN-MATCHED COMPARISON: All Experiments")
    print("="*140)
    print("Tokens/iter: Baseline=3.15M (2×), Grassmann=1.57M")
    print("-"*140)

    # Target token milestones (in billions)
    milestones = [0, 1.5, 3, 4.5, 6, 7.5, 9, 12, 15, 20, 25, 30, 40, 50, 60, 70]

    print(f"{'Tokens':<10} {'Baseline':<12} {'G-elem r384':<12} {'G-hyb r384':<12} "
          f"{'T1-36L r384':<12} {'T1-24L r512':<12} {'T1-24L full':<12} {'Best':<20}")
    print("-"*140)

    for milestone_b in milestones:
        milestone = milestone_b * 1e9
        row = [format_tokens(milestone)]

        vals = {}
        for name, _, _ in experiments:
            data = all_data[name]
            closest = find_closest(data, milestone, tolerance=0.8e9)
            if closest:
                _, tokens, _, val_loss = closest
                vals[name] = val_loss
                row.append(f"{val_loss:.4f}")
            else:
                row.append("-")

        # Find best
        if vals:
            best_name = min(vals.items(), key=lambda x: x[1])[0]
            best_val = vals[best_name]
            # Shorten name for display
            short_name = best_name.split()[0] if "Tier1" in best_name else best_name.split()[0]
            row.append(f"{short_name} ({best_val:.4f})")
        else:
            row.append("-")

        # Only print if we have data
        if any(v != "-" for v in row[1:]):
            print(f"{row[0]:<10} {row[1]:<12} {row[2]:<12} {row[3]:<12} "
                  f"{row[4]:<12} {row[5]:<12} {row[6]:<12} {row[7]:<20}")

    print("-"*140)

    # Summary
    print("\nSUMMARY:")
    print(f"{'Experiment':<35} {'Latest Iter':<12} {'Tokens':<10} {'Train Loss':<12} {'Val Loss':<12}")
    print("-"*80)
    for name in [exp[0] for exp in experiments]:
        data = all_data[name]
        if data:
            last_iter, last_tokens, last_train, last_val = data[-1]
            print(f"{name:<35} {last_iter:>11} {format_tokens(last_tokens):>9} {last_train:>11.4f} {last_val:>11.4f}")
        else:
            print(f"{name:<35} {'No data':>11}")

    # Key insights
    print("\n" + "="*140)
    print("KEY INSIGHTS:")
    print("="*140)

    # Compare at 6B tokens (where we have good data)
    milestone_6b = 6e9
    comparison_6b = {}
    for name, _, _ in experiments:
        data = all_data[name]
        closest = find_closest(data, milestone_6b, tolerance=0.8e9)
        if closest:
            comparison_6b[name] = closest[3]  # val_loss

    if comparison_6b:
        best_6b = min(comparison_6b.items(), key=lambda x: x[1])
        baseline_6b = comparison_6b.get("Baseline (24L AdamW)", None)

        print(f"\nAt ~6B tokens:")
        for name in sorted(comparison_6b.keys(), key=lambda x: comparison_6b[x]):
            val = comparison_6b[name]
            if baseline_6b:
                improvement = ((baseline_6b - val) / baseline_6b) * 100
                print(f"  {name:<35}: {val:.4f} ({improvement:+.1f}% vs baseline)")
            else:
                print(f"  {name:<35}: {val:.4f}")

        print(f"\n  Best: {best_6b[0]} ({best_6b[1]:.4f})")

if __name__ == "__main__":
    main()
