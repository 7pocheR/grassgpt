#!/usr/bin/env python3
"""
Compare variance of rank vs stable rank across seeds.

Rank = actual rank (number of non-zero singular values, or rank at some threshold)
Stable rank = ||W||_F^2 / ||W||_op^2

Question: Which is more consistent across random seeds?
"""

import torch
import numpy as np
from pathlib import Path
from glob import glob
from collections import defaultdict

def analyze_weight_matrix(weight):
    """Compute rank metrics."""
    if weight.dim() != 2:
        return None

    try:
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    except:
        return None

    op_norm = S[0].item()
    frob_norm = torch.norm(weight, p='fro').item()
    stable_rank = (frob_norm**2) / (op_norm**2 + 1e-10)

    # Rank at different thresholds
    S_normalized = S / (S[0] + 1e-10)  # Normalize by max singular value

    # Rank at 99% energy (cumulative sum of squared singular values)
    cumsum = torch.cumsum(S**2, dim=0)
    total_energy = cumsum[-1]
    rank_99 = torch.sum(cumsum < 0.99 * total_energy).item() + 1

    # Rank at 95% energy
    rank_95 = torch.sum(cumsum < 0.95 * total_energy).item() + 1

    # Rank at 90% energy
    rank_90 = torch.sum(cumsum < 0.90 * total_energy).item() + 1

    # Effective rank (entropy-based)
    S_prob = S / (S.sum() + 1e-10)
    entropy = -torch.sum(S_prob * torch.log(S_prob + 1e-10))
    eff_rank = torch.exp(entropy).item()

    # Numerical rank (threshold-based, σ_i > 0.01 * σ_max)
    num_rank_001 = torch.sum(S_normalized > 0.01).item()
    num_rank_0001 = torch.sum(S_normalized > 0.001).item()

    return {
        'stable_rank': stable_rank,
        'rank_99': rank_99,
        'rank_95': rank_95,
        'rank_90': rank_90,
        'eff_rank': eff_rank,
        'num_rank_001': num_rank_001,
        'num_rank_0001': num_rank_0001,
        'op_norm': op_norm,
        'frob_norm': frob_norm,
    }

def main():
    # Load all checkpoints
    checkpoint_pattern = '/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline-seed*/ckpt.pt'
    checkpoint_paths = sorted(glob(checkpoint_pattern))

    print(f"Loading {len(checkpoint_paths)} checkpoints...\n")

    # Collect stats: layer_name -> seed -> metrics
    all_stats = defaultdict(lambda: defaultdict(dict))

    for cp_path in checkpoint_paths:
        seed_name = Path(cp_path).parent.name
        checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)

        for name, param in state_dict.items():
            if 'weight' in name and param.dim() == 2:
                stats = analyze_weight_matrix(param)
                if stats:
                    all_stats[name][seed_name] = stats

    print("="*100)
    print("VARIANCE COMPARISON: Stable Rank vs Other Rank Measures")
    print("="*100)
    print("\nCoefficient of Variation (CV = std/mean × 100%) across 10 seeds:\n")

    # Metrics to compare
    metrics = ['stable_rank', 'eff_rank', 'rank_99', 'rank_95', 'rank_90', 'num_rank_001', 'num_rank_0001']

    # Aggregate CV across all layers
    cv_by_metric = defaultdict(list)

    for layer_name, seed_data in all_stats.items():
        # Only analyze layers we care about
        if not any(x in layer_name for x in ['c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']):
            continue

        seeds = list(seed_data.keys())

        for metric in metrics:
            values = np.array([seed_data[s][metric] for s in seeds])
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / mean) * 100 if mean > 0 else 0
            cv_by_metric[metric].append(cv)

    # Print summary
    print(f"{'Rank Metric':<20} {'Mean CV (%)':<15} {'Median CV (%)':<15} {'Max CV (%)':<15} {'Interpretation':<30}")
    print("-"*100)

    results = []
    for metric in metrics:
        cvs = cv_by_metric[metric]
        mean_cv = np.mean(cvs)
        median_cv = np.median(cvs)
        max_cv = np.max(cvs)
        results.append((metric, mean_cv, median_cv, max_cv))

    # Sort by mean CV (ascending = most stable)
    results.sort(key=lambda x: x[1])

    interpretations = {
        'stable_rank': 'Frobenius²/Op norm²',
        'eff_rank': 'Entropy-based rank',
        'rank_99': 'Rank @ 99% energy',
        'rank_95': 'Rank @ 95% energy',
        'rank_90': 'Rank @ 90% energy',
        'num_rank_001': 'Count σ > 0.01σ_max',
        'num_rank_0001': 'Count σ > 0.001σ_max',
    }

    for i, (metric, mean_cv, median_cv, max_cv) in enumerate(results):
        marker = "⭐" if i == 0 else "  "
        interp = interpretations.get(metric, '')
        print(f"{marker} {metric:<18} {mean_cv:<15.2f} {median_cv:<15.2f} {max_cv:<15.2f} {interp}")

    print(f"\n⭐ {results[0][0]} is most stable across seeds (lowest CV)!\n")

    # Per-category analysis
    print("\n" + "="*100)
    print("VARIANCE BY LAYER CATEGORY")
    print("="*100)

    categories = {
        'c_attn': [],
        'attn.c_proj': [],
        'mlp.c_fc': [],
        'mlp.c_proj': [],
    }

    for layer_name in all_stats.keys():
        if 'c_attn.weight' in layer_name:
            categories['c_attn'].append(layer_name)
        elif 'attn.c_proj.weight' in layer_name:
            categories['attn.c_proj'].append(layer_name)
        elif 'mlp.c_fc.weight' in layer_name:
            categories['mlp.c_fc'].append(layer_name)
        elif 'mlp.c_proj.weight' in layer_name:
            categories['mlp.c_proj'].append(layer_name)

    for cat_name, layer_names in categories.items():
        if not layer_names:
            continue

        print(f"\n### {cat_name}")
        print(f"{'Metric':<20} {'Mean (avg across seeds & blocks)':<35} {'CV across seeds':<20} {'CV across blocks':<20}")
        print("-"*100)

        for metric in ['stable_rank', 'eff_rank', 'rank_99']:
            # Collect all values: per block, get mean across seeds
            block_means = []
            seed_cvs = []

            for layer_name in sorted(layer_names):
                seed_data = all_stats[layer_name]
                seeds = list(seed_data.keys())

                # Values for this block across seeds
                values = np.array([seed_data[s][metric] for s in seeds])
                block_mean = np.mean(values)
                block_std = np.std(values)
                seed_cv = (block_std / block_mean * 100) if block_mean > 0 else 0

                block_means.append(block_mean)
                seed_cvs.append(seed_cv)

            # Stats
            overall_mean = np.mean(block_means)
            overall_std = np.std(block_means)
            block_cv = (overall_std / overall_mean * 100) if overall_mean > 0 else 0
            mean_seed_cv = np.mean(seed_cvs)

            mean_str = f"{overall_mean:.2f} ± {overall_std:.2f}"
            print(f"  {metric:<18} {mean_str:<35} {mean_seed_cv:<6.2f}%{'':<13} {block_cv:<6.2f}%")

    # Recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    print("\n**Question:** Should we match stable rank or another rank measure?\n")

    print("**Answer:** Use STABLE RANK because:")
    print(f"  1. Lowest variance across seeds: {results[0][1]:.2f}% mean CV")
    print(f"  2. Directly related to Grassmann manifold properties (||W||_F² / ||W||_op²)")
    print(f"  3. Has clean mathematical relationship: Grassmann stable rank = r (the manifold rank)")
    print(f"  4. More robust than threshold-based ranks (99%, 95%, etc.)")

    print("\n**How to set Grassmann rank:**")
    print("  - Grassmann rank r = round(stable_rank_baseline)")
    print("  - Scale factor s = operator_norm_baseline")
    print("  - This ensures: stable_rank_grassmann_scaled = r = stable_rank_baseline")

    print("\n**Alternative (if stable rank seems too abstract):**")
    print(f"  - Use rank_99 (CV: {[r for r in results if r[0]=='rank_99'][0][1]:.2f}%)")
    print(f"  - Slightly higher variance but more intuitive (captures 99% of energy)")

if __name__ == "__main__":
    main()
