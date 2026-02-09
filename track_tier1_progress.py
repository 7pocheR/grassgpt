#!/usr/bin/env python3
"""Track and compare Tier 1 experiment progress"""

import re
import sys

# Tokens per iteration for all Tier 1 experiments
TOKENS_PER_ITER = 1_572_864

def parse_log(filename):
    """Extract (iter, train_loss, val_loss) tuples from log file"""
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

def format_tokens(tokens):
    """Format token count in billions"""
    return f"{tokens / 1e9:.2f}B"

def main():
    experiments = [
        ("36L r=384 hybrid", "logs/610968_tier1_36L_r384.out"),
        ("24L r=512 hybrid", "logs/610969_tier1_24L_r512.out"),
        ("24L r=384 full block", "logs/610970_tier1_24L_fb384.out"),
    ]

    # Parse all logs
    all_data = {}
    for name, logfile in experiments:
        data = parse_log(logfile)
        all_data[name] = data
        print(f"Loaded {len(data)} checkpoints from {name}")

    print("\n" + "="*100)
    print("TIER 1 PROGRESS COMPARISON")
    print("="*100)
    print(f"{'Tokens':<12} {'36L r=384':<20} {'24L r=512':<20} {'24L full block':<20} {'Best':<15}")
    print("-"*100)

    # Find all unique token counts (approximately)
    all_tokens = set()
    for data in all_data.values():
        all_tokens.update([tokens for _, tokens, _, _ in data])

    # Sort and display
    for tokens in sorted(all_tokens):
        row = [format_tokens(tokens)]

        # Get val loss for each experiment at this token count
        vals = {}
        for name in [exp[0] for exp in experiments]:
            data = all_data[name]
            # Find closest match
            matching = [val for _, t, _, val in data if t == tokens]
            if matching:
                val_loss = matching[0]
                vals[name] = val_loss
                row.append(f"{val_loss:.4f}")
            else:
                row.append("-")

        # Determine best (if we have values)
        if vals:
            best_name = min(vals.items(), key=lambda x: x[1])[0]
            best_val = vals[best_name]
            row.append(f"{best_name.split()[0]} ({best_val:.4f})")
        else:
            row.append("-")

        print(f"{row[0]:<12} {row[1]:<20} {row[2]:<20} {row[3]:<20} {row[4]:<15}")

    print("-"*100)

    # Summary statistics
    print("\nLATEST CHECKPOINTS:")
    for name in [exp[0] for exp in experiments]:
        data = all_data[name]
        if data:
            last_iter, last_tokens, last_train, last_val = data[-1]
            print(f"  {name:<25}: iter {last_iter:>6}, tokens {format_tokens(last_tokens):>7}, "
                  f"train {last_train:.4f}, val {last_val:.4f}")
        else:
            print(f"  {name:<25}: No data yet")

if __name__ == "__main__":
    main()
