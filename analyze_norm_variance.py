#!/usr/bin/env python3
"""
Analyze variance patterns in weight norms:
1. Which norm type has smallest variance across seeds
2. Variance across blocks within same layer category
"""

import torch
import numpy as np
from pathlib import Path
from glob import glob
from collections import defaultdict
import sys

def analyze_weight_matrix(weight):
    """Compute all norms for a weight matrix."""
    if weight.dim() != 2:
        return None

    try:
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    except:
        return None

    op_norm = S[0].item()
    frob_norm = torch.norm(weight, p='fro').item()
    nuclear_norm = torch.sum(S).item()
    l1_norm = torch.max(torch.sum(torch.abs(weight), dim=0)).item()
    linf_norm = torch.max(torch.sum(torch.abs(weight), dim=1)).item()

    # Stable rank
    stable_rank = (frob_norm**2) / (op_norm**2 + 1e-10)

    return {
        'op_norm': op_norm,
        'frob_norm': frob_norm,
        'nuclear_norm': nuclear_norm,
        'l1_norm': l1_norm,
        'linf_norm': linf_norm,
        'stable_rank': stable_rank,
    }

def main():
    # Load all checkpoints
    checkpoint_pattern = '/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline-seed*/ckpt.pt'
    checkpoint_paths = sorted(glob(checkpoint_pattern))

    if not checkpoint_paths:
        print(f"No checkpoints found")
        return

    print(f"Loading {len(checkpoint_paths)} checkpoints...\n")

    # Collect all norms: layer_name -> seed -> {norm_type -> value}
    all_data = defaultdict(lambda: defaultdict(dict))

    for cp_path in checkpoint_paths:
        seed_name = Path(cp_path).parent.name
        checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)

        for name, param in state_dict.items():
            if 'weight' in name and param.dim() == 2:
                norms = analyze_weight_matrix(param)
                if norms:
                    all_data[name][seed_name] = norms

    # Analysis 1: Which norm has smallest coefficient of variation (CV) across seeds?
    print("="*80)
    print("ANALYSIS 1: Coefficient of Variation (CV = std/mean) for Each Norm Type")
    print("="*80)
    print("(Lower CV = more stable across random seeds)\n")

    norm_types = ['op_norm', 'frob_norm', 'nuclear_norm', 'l1_norm', 'linf_norm', 'stable_rank']

    # Aggregate CV across all layers
    cv_by_norm_type = defaultdict(list)

    for layer_name, seed_data in all_data.items():
        seeds = list(seed_data.keys())

        for norm_type in norm_types:
            values = np.array([seed_data[s][norm_type] for s in seeds])
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / mean) * 100 if mean > 0 else 0
            cv_by_norm_type[norm_type].append(cv)

    # Print summary
    print(f"{'Norm Type':<20} {'Mean CV (%)':<15} {'Median CV (%)':<15} {'Max CV (%)':<15}")
    print("-"*70)

    results = []
    for norm_type in norm_types:
        cvs = cv_by_norm_type[norm_type]
        mean_cv = np.mean(cvs)
        median_cv = np.median(cvs)
        max_cv = np.max(cvs)
        results.append((norm_type, mean_cv, median_cv, max_cv))

    # Sort by mean CV (ascending = most stable)
    results.sort(key=lambda x: x[1])

    for i, (norm_type, mean_cv, median_cv, max_cv) in enumerate(results):
        marker = "⭐" if i == 0 else "  "
        print(f"{marker} {norm_type:<17} {mean_cv:<15.2f} {median_cv:<15.2f} {max_cv:<15.2f}")

    print(f"\n⭐ {results[0][0]} has the lowest variance across seeds!\n")

    # Analysis 2: Variance across blocks within same layer category
    print("\n" + "="*80)
    print("ANALYSIS 2: Variance Across Blocks Within Same Layer Category")
    print("="*80)
    print("(Higher CV = more block-to-block variation)\n")

    # Group layers by category and block number
    categories = {
        'c_attn': [],
        'attn.c_proj': [],
        'mlp.c_fc': [],
        'mlp.c_proj': [],
    }

    for layer_name in all_data.keys():
        if 'c_attn.weight' in layer_name:
            categories['c_attn'].append(layer_name)
        elif 'attn.c_proj.weight' in layer_name:
            categories['attn.c_proj'].append(layer_name)
        elif 'mlp.c_fc.weight' in layer_name:
            categories['mlp.c_fc'].append(layer_name)
        elif 'mlp.c_proj.weight' in layer_name:
            categories['mlp.c_proj'].append(layer_name)

    # For each category, compute mean value per block (averaged across seeds)
    # Then compute CV across blocks
    for category, layer_names in categories.items():
        if not layer_names:
            continue

        print(f"\n### {category}")
        print(f"Number of blocks: {len(layer_names)}\n")

        # For each norm type, get values per block (averaged across seeds)
        for norm_type in norm_types:
            block_means = []

            for layer_name in sorted(layer_names):
                seed_data = all_data[layer_name]
                seeds = list(seed_data.keys())

                # Average across seeds for this block
                values = np.array([seed_data[s][norm_type] for s in seeds])
                block_mean = np.mean(values)
                block_means.append(block_mean)

            # Compute CV across blocks
            block_means = np.array(block_means)
            mean_across_blocks = np.mean(block_means)
            std_across_blocks = np.std(block_means)
            cv_across_blocks = (std_across_blocks / mean_across_blocks) * 100

            print(f"  {norm_type:<20} mean={mean_across_blocks:8.2f}  std={std_across_blocks:7.2f}  CV={cv_across_blocks:6.2f}%")

    # Detailed per-layer breakdown for operator norm
    print("\n" + "="*80)
    print("ANALYSIS 3: Per-Block Operator Norm Details")
    print("="*80)

    for category, layer_names in categories.items():
        if not layer_names:
            continue

        print(f"\n### {category}")
        print(f"{'Block':<8} {'Op Norm (mean±std)':<25} {'Range':<20} {'CV (%)':<10}")
        print("-"*70)

        block_values = []
        for layer_name in sorted(layer_names):
            # Extract block number
            import re
            match = re.search(r'\.h\.(\d+)\.', layer_name)
            block_num = int(match.group(1)) if match else -1

            seed_data = all_data[layer_name]
            seeds = list(seed_data.keys())

            values = np.array([seed_data[s]['op_norm'] for s in seeds])
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            cv = (std_val / mean_val) * 100

            block_values.append(mean_val)

            print(f"Block {block_num}  {mean_val:.4f}±{std_val:.4f}          [{min_val:.4f}, {max_val:.4f}]     {cv:.2f}%")

        # Summary for category
        block_values = np.array(block_values)
        overall_mean = np.mean(block_values)
        overall_std = np.std(block_values)
        cv_blocks = (overall_std / overall_mean) * 100

        print(f"\nCategory summary:")
        print(f"  Mean across blocks: {overall_mean:.4f}")
        print(f"  Std across blocks:  {overall_std:.4f}")
        print(f"  CV across blocks:   {cv_blocks:.2f}%")
        print(f"  Range: [{np.min(block_values):.4f}, {np.max(block_values):.4f}]")

if __name__ == "__main__":
    main()
