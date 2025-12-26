#!/usr/bin/env python3
"""
Compute per-layer Grassmann rank and scaling to match both:
1. Operator norm (spectral norm)
2. Stable rank

For each layer, we need:
- rank = round(stable_rank_target)
- scale = operator_norm_target
"""

import torch
import numpy as np
from pathlib import Path
from glob import glob
from collections import defaultdict

def analyze_weight_matrix(weight):
    """Compute all necessary statistics."""
    if weight.dim() != 2:
        return None

    try:
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    except:
        return None

    op_norm = S[0].item()
    frob_norm = torch.norm(weight, p='fro').item()
    stable_rank = (frob_norm**2) / (op_norm**2 + 1e-10)

    return {
        'op_norm': op_norm,
        'stable_rank': stable_rank,
        'shape': list(weight.shape),
    }

def main():
    # Load all checkpoints
    checkpoint_pattern = '/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline-seed*/ckpt.pt'
    checkpoint_paths = sorted(glob(checkpoint_pattern))

    print(f"Loading {len(checkpoint_paths)} checkpoints...\n")

    # Collect stats: layer_name -> list of measurements across seeds
    all_stats = defaultdict(list)

    for cp_path in checkpoint_paths:
        checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)

        for name, param in state_dict.items():
            if 'weight' in name and param.dim() == 2:
                stats = analyze_weight_matrix(param)
                if stats:
                    all_stats[name].append(stats)

    # Aggregate statistics across seeds
    print("="*100)
    print("PER-LAYER GRASSMANN CONFIGURATION")
    print("="*100)
    print("\nTo match both operator norm AND stable rank from baseline:\n")
    print(f"{'Layer':<50} {'Shape':<15} {'Op Norm':<20} {'Stable Rank':<20} {'Grassmann Rank':<15} {'Scale Factor':<15}")
    print("-"*150)

    # Focus on layers we care about
    layer_order = []
    for name in sorted(all_stats.keys()):
        if any(x in name for x in ['c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']):
            layer_order.append(name)

    # Generate configuration
    configs = []

    for name in layer_order:
        stats_list = all_stats[name]

        # Compute mean and std across seeds
        op_norms = [s['op_norm'] for s in stats_list]
        stable_ranks = [s['stable_rank'] for s in stats_list]

        op_norm_mean = np.mean(op_norms)
        op_norm_std = np.std(op_norms)
        sr_mean = np.mean(stable_ranks)
        sr_std = np.std(stable_ranks)

        shape = stats_list[0]['shape']

        # Grassmann configuration
        # rank = stable_rank (rounded to int)
        # scale = operator_norm
        grass_rank = int(round(sr_mean))
        scale_factor = op_norm_mean

        # Validate rank doesn't exceed matrix dimensions
        max_rank = min(shape)
        if grass_rank > max_rank:
            grass_rank = max_rank
            print(f"! Warning: {name} stable rank {sr_mean:.1f} exceeds max rank {max_rank}, capping at {max_rank}")

        configs.append({
            'name': name,
            'shape': shape,
            'op_norm_mean': op_norm_mean,
            'op_norm_std': op_norm_std,
            'sr_mean': sr_mean,
            'sr_std': sr_std,
            'grass_rank': grass_rank,
            'scale_factor': scale_factor,
        })

        # Print row
        shape_str = f"{shape[0]}×{shape[1]}"
        op_norm_str = f"{op_norm_mean:.4f} ± {op_norm_std:.4f}"
        sr_str = f"{sr_mean:.2f} ± {sr_std:.2f}"

        print(f"{name:<50} {shape_str:<15} {op_norm_str:<20} {sr_str:<20} {grass_rank:<15} {scale_factor:.4f}")

    # Group by category and show summary
    print("\n" + "="*100)
    print("SUMMARY BY LAYER CATEGORY")
    print("="*100)

    categories = {
        'c_attn': [],
        'attn.c_proj': [],
        'mlp.c_fc': [],
        'mlp.c_proj': [],
    }

    for cfg in configs:
        name = cfg['name']
        if 'c_attn.weight' in name:
            categories['c_attn'].append(cfg)
        elif 'attn.c_proj.weight' in name:
            categories['attn.c_proj'].append(cfg)
        elif 'mlp.c_fc.weight' in name:
            categories['mlp.c_fc'].append(cfg)
        elif 'mlp.c_proj.weight' in name:
            categories['mlp.c_proj'].append(cfg)

    for cat_name, cfgs in categories.items():
        if not cfgs:
            continue

        print(f"\n### {cat_name}")

        ranks = [c['grass_rank'] for c in cfgs]
        scales = [c['scale_factor'] for c in cfgs]

        print(f"  Ranks across blocks: {ranks}")
        print(f"  Rank range: [{min(ranks)}, {max(ranks)}]")
        print(f"  Rank CV: {(np.std(ranks)/np.mean(ranks)*100):.2f}%")
        print(f"  ")
        print(f"  Scales across blocks: {[f'{s:.4f}' for s in scales]}")
        print(f"  Scale mean: {np.mean(scales):.4f} ± {np.std(scales):.4f}")
        print(f"  Scale CV: {(np.std(scales)/np.mean(scales)*100):.2f}%")

    # Implementation recommendations
    print("\n" + "="*100)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("="*100)

    print("\n## Option 1: Per-Block Configuration (Exact Match)")
    print("```python")
    print("# In model.py - setup per-block rank and scaling")
    print("def setup_grassmann_exact_match(self):")

    for cat_name, cfgs in categories.items():
        if not cfgs:
            continue
        print(f"\n    # {cat_name}")
        for cfg in cfgs:
            block_num = None
            import re
            match = re.search(r'\.h\.(\d+)\.', cfg['name'])
            if match:
                block_num = int(match.group(1))
                print(f"    self.transformer.h[{block_num}].{cat_name.replace('.', '_')}_rank = {cfg['grass_rank']}")
                print(f"    self.transformer.h[{block_num}].{cat_name.replace('.', '_')}_scale = {cfg['scale_factor']:.4f}")

    print("```")

    print("\n## Option 2: Category-Average (Simpler, May Be Good Enough)")
    print("```python")
    print("# Use average rank and scale per category")
    for cat_name, cfgs in categories.items():
        if not cfgs:
            continue
        ranks = [c['grass_rank'] for c in cfgs]
        scales = [c['scale_factor'] for c in cfgs]
        avg_rank = int(round(np.mean(ranks)))
        avg_scale = np.mean(scales)

        print(f"{cat_name}_rank = {avg_rank}  # Stable rank target")
        print(f"{cat_name}_scale = {avg_scale:.4f}  # Operator norm target")
    print("```")

    print("\n## Variance Analysis")
    print("\nWhich categories need per-block vs can use average?")
    for cat_name, cfgs in categories.items():
        if not cfgs:
            continue
        ranks = [c['grass_rank'] for c in cfgs]
        scales = [c['scale_factor'] for c in cfgs]

        rank_cv = (np.std(ranks)/np.mean(ranks)*100) if np.mean(ranks) > 0 else 0
        scale_cv = (np.std(scales)/np.mean(scales)*100) if np.mean(scales) > 0 else 0

        if rank_cv < 20 and scale_cv < 20:
            verdict = "✓ Can use category average"
        else:
            verdict = "⚠ Consider per-block configuration"

        print(f"{cat_name:15s} - Rank CV: {rank_cv:6.2f}%  Scale CV: {scale_cv:6.2f}%  → {verdict}")

if __name__ == "__main__":
    main()
