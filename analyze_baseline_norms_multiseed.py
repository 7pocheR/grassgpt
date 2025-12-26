#!/usr/bin/env python3
"""
Analyze weight matrix operator norms from multiple baseline checkpoints.

Computes statistics across multiple random seeds to get robust estimates
of target operator norms for Grassmann scaling.
"""

import torch
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict

def analyze_weight_matrix(name, weight):
    """Compute statistics for a weight matrix."""
    if weight.dim() != 2:
        return None

    # Compute SVD
    try:
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    except:
        print(f"Warning: SVD failed for {name}")
        return None

    # Operator norm (largest singular value / spectral norm / 2-norm)
    op_norm = S[0].item()

    # Frobenius norm (sqrt of sum of squared elements)
    frob_norm = torch.norm(weight, p='fro').item()

    # Nuclear norm (sum of singular values / trace norm)
    nuclear_norm = torch.sum(S).item()

    # L1 norm (max column sum)
    l1_norm = torch.max(torch.sum(torch.abs(weight), dim=0)).item()

    # L-infinity norm (max row sum)
    linf_norm = torch.max(torch.sum(torch.abs(weight), dim=1)).item()

    # Effective rank (entropy of normalized singular values)
    S_normalized = S / (S.sum() + 1e-10)
    entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-10))
    eff_rank = torch.exp(entropy).item()

    # Condition number
    cond_number = S[0].item() / (S[-1].item() + 1e-10)

    # Rank at 99% energy
    cumsum = torch.cumsum(S**2, dim=0)
    total_energy = cumsum[-1]
    rank_99 = torch.sum(cumsum < 0.99 * total_energy).item() + 1

    # Stable rank (||W||_F^2 / ||W||_op^2)
    stable_rank = (frob_norm**2) / (op_norm**2 + 1e-10)

    return {
        'shape': list(weight.shape),
        'op_norm': op_norm,
        'frob_norm': frob_norm,
        'nuclear_norm': nuclear_norm,
        'l1_norm': l1_norm,
        'linf_norm': linf_norm,
        'eff_rank': eff_rank,
        'stable_rank': stable_rank,
        'actual_rank': min(weight.shape),
        'rank_99': rank_99,
        'cond_number': cond_number,
        'singular_values': S.cpu().numpy()[:10],  # Top 10
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze baseline weight norms across multiple seeds')
    parser.add_argument('--checkpoint-pattern', type=str,
                       default='/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline-seed*/ckpt.pt',
                       help='Glob pattern for checkpoint files')
    parser.add_argument('--output', type=str, default='baseline_weight_analysis_multiseed.md',
                       help='Output file')
    args = parser.parse_args()

    # Find all checkpoint files
    from glob import glob
    checkpoint_paths = sorted(glob(args.checkpoint_pattern))

    if not checkpoint_paths:
        print(f"Error: No checkpoints found matching: {args.checkpoint_pattern}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoints:")
    for cp in checkpoint_paths:
        print(f"  {cp}")

    # Collect statistics across all seeds
    all_stats = defaultdict(list)  # name -> [stats_seed0, stats_seed1, ...]
    checkpoint_info = []

    for cp_path in checkpoint_paths:
        print(f"\nAnalyzing: {cp_path}")
        checkpoint = torch.load(cp_path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        checkpoint_info.append({
            'path': cp_path,
            'iter_num': checkpoint.get('iter_num', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
        })

        # Analyze each weight matrix
        for name, param in state_dict.items():
            if 'weight' in name and param.dim() == 2:
                result = analyze_weight_matrix(name, param)
                if result:
                    all_stats[name].append(result)

    # Compute mean and std across seeds
    aggregated_stats = {}
    for name, stats_list in all_stats.items():
        if not stats_list:
            continue

        # Extract arrays for each metric
        op_norms = np.array([s['op_norm'] for s in stats_list])
        frob_norms = np.array([s['frob_norm'] for s in stats_list])
        nuclear_norms = np.array([s['nuclear_norm'] for s in stats_list])
        l1_norms = np.array([s['l1_norm'] for s in stats_list])
        linf_norms = np.array([s['linf_norm'] for s in stats_list])
        eff_ranks = np.array([s['eff_rank'] for s in stats_list])
        stable_ranks = np.array([s['stable_rank'] for s in stats_list])
        cond_numbers = np.array([s['cond_number'] for s in stats_list])
        rank_99s = np.array([s['rank_99'] for s in stats_list])

        aggregated_stats[name] = {
            'shape': stats_list[0]['shape'],
            'op_norm_mean': np.mean(op_norms),
            'op_norm_std': np.std(op_norms),
            'op_norm_min': np.min(op_norms),
            'op_norm_max': np.max(op_norms),
            'frob_norm_mean': np.mean(frob_norms),
            'frob_norm_std': np.std(frob_norms),
            'nuclear_norm_mean': np.mean(nuclear_norms),
            'nuclear_norm_std': np.std(nuclear_norms),
            'l1_norm_mean': np.mean(l1_norms),
            'l1_norm_std': np.std(l1_norms),
            'linf_norm_mean': np.mean(linf_norms),
            'linf_norm_std': np.std(linf_norms),
            'eff_rank_mean': np.mean(eff_ranks),
            'eff_rank_std': np.std(eff_ranks),
            'stable_rank_mean': np.mean(stable_ranks),
            'stable_rank_std': np.std(stable_ranks),
            'cond_number_mean': np.mean(cond_numbers),
            'rank_99_mean': np.mean(rank_99s),
        }

    # Generate markdown report
    lines = []
    lines.append(f"# Baseline Weight Matrix Analysis (Multi-Seed)")
    lines.append("")
    lines.append(f"**Number of seeds:** {len(checkpoint_paths)}")
    lines.append(f"**Checkpoint pattern:** `{args.checkpoint_pattern}`")
    lines.append("")

    # Checkpoint details
    lines.append("## Checkpoint Details")
    lines.append("")
    lines.append("| Seed | Iteration | Best Val Loss |")
    lines.append("|------|-----------|---------------|")
    for i, info in enumerate(checkpoint_info):
        seed_name = Path(info['path']).parent.name
        lines.append(f"| {seed_name} | {info['iter_num']} | {info['best_val_loss']:.4f} |")
    lines.append("")

    # Per-layer statistics table
    lines.append("## Per-Layer Weight Matrix Statistics (Averaged Across Seeds)")
    lines.append("")
    lines.append("| Layer | Shape | Op Norm | Frob Norm | Nuclear Norm | L1 Norm | L∞ Norm | Stable Rank | Eff Rank | Rank@99% | Condition |")
    lines.append("|-------|-------|---------|-----------|--------------|---------|---------|-------------|----------|----------|-----------|")

    # Group by layer type for later summary
    layer_groups = {
        'c_attn': [],
        'attn.c_proj': [],
        'mlp.c_fc': [],
        'mlp.c_proj': [],
        'other': []
    }

    for name, stats in sorted(aggregated_stats.items()):
        # Categorize
        if 'c_attn.weight' in name:
            layer_groups['c_attn'].append((name, stats))
        elif 'attn.c_proj.weight' in name:
            layer_groups['attn.c_proj'].append((name, stats))
        elif 'mlp.c_fc.weight' in name:
            layer_groups['mlp.c_fc'].append((name, stats))
        elif 'mlp.c_proj.weight' in name:
            layer_groups['mlp.c_proj'].append((name, stats))
        else:
            layer_groups['other'].append((name, stats))

        # Add to table - show each layer individually with all norms
        shape_str = f"{stats['shape'][0]}×{stats['shape'][1]}"
        op_norm_str = f"{stats['op_norm_mean']:.4f}±{stats['op_norm_std']:.4f}"
        frob_norm_str = f"{stats['frob_norm_mean']:.2f}±{stats['frob_norm_std']:.2f}"
        nuclear_norm_str = f"{stats['nuclear_norm_mean']:.1f}±{stats['nuclear_norm_std']:.2f}"
        l1_norm_str = f"{stats['l1_norm_mean']:.2f}±{stats['l1_norm_std']:.3f}"
        linf_norm_str = f"{stats['linf_norm_mean']:.2f}±{stats['linf_norm_std']:.3f}"
        stable_rank_str = f"{stats['stable_rank_mean']:.1f}±{stats['stable_rank_std']:.2f}"

        lines.append(
            f"| {name} | {shape_str} | {op_norm_str} | {frob_norm_str} | "
            f"{nuclear_norm_str} | {l1_norm_str} | {linf_norm_str} | "
            f"{stable_rank_str} | {stats['eff_rank_mean']:.1f} | "
            f"{stats['rank_99_mean']:.1f} | {stats['cond_number_mean']:.1f} |"
        )

    lines.append("")

    # Summary by layer type
    lines.append("## Summary by Layer Type")
    lines.append("")

    for layer_type, items in layer_groups.items():
        if not items:
            continue

        lines.append(f"### {layer_type}")
        lines.append("")

        op_norms_mean = [stats['op_norm_mean'] for _, stats in items]
        op_norms_std = [stats['op_norm_std'] for _, stats in items]

        lines.append(f"- **Count:** {len(items)}")
        lines.append(f"- **Operator norm (mean):** {np.mean(op_norms_mean):.4f} ± {np.mean(op_norms_std):.4f}")
        lines.append(f"- **Operator norm (range):** [{np.min(op_norms_mean):.4f}, {np.max(op_norms_mean):.4f}]")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations for Grassmann Scaling")
    lines.append("")
    lines.append("Since Grassmann manifold fixes operator norm = 1, we need to scale outputs to match baseline:")
    lines.append("")

    recommendations = {}
    for layer_type, items in [('c_attn', layer_groups['c_attn']),
                               ('mlp.c_fc', layer_groups['mlp.c_fc']),
                               ('attn.c_proj', layer_groups['attn.c_proj'])]:
        if items:
            avg_op_norm = np.mean([stats['op_norm_mean'] for _, stats in items])
            std_op_norm = np.mean([stats['op_norm_std'] for _, stats in items])
            recommendations[layer_type] = (avg_op_norm, std_op_norm)
            lines.append(f"- **{layer_type}**: Scale by **{avg_op_norm:.4f} ± {std_op_norm:.4f}** (mean operator norm across seeds)")

    lines.append("")
    lines.append("### Implementation")
    lines.append("")
    lines.append("```python")
    lines.append("# In model.py, multiply Grassmann weight outputs by these factors:")
    lines.append("")

    if 'c_attn' in recommendations:
        avg, std = recommendations['c_attn']
        lines.append(f"# c_attn: scale = {avg:.4f}")
        lines.append(f"self.c_attn_scale = {avg:.4f}  # Robust across {len(checkpoint_paths)} seeds (±{std:.4f})")
        lines.append("")

    if 'mlp.c_fc' in recommendations:
        avg, std = recommendations['mlp.c_fc']
        lines.append(f"# mlp.c_fc: scale = {avg:.4f}")
        lines.append(f"self.mlp_c_fc_scale = {avg:.4f}  # Robust across {len(checkpoint_paths)} seeds (±{std:.4f})")
        lines.append("")

    if 'attn.c_proj' in recommendations:
        avg, std = recommendations['attn.c_proj']
        lines.append(f"# attn.c_proj: scale = {avg:.4f}")
        lines.append(f"self.attn_c_proj_scale = {avg:.4f}  # Robust across {len(checkpoint_paths)} seeds (±{std:.4f})")

    lines.append("```")
    lines.append("")

    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Analysis complete!")
    print(f"  Report: {output_path}")

    # Print key findings to console
    print("\n=== Key Findings (Averaged Across Seeds) ===")
    for layer_type in ['c_attn', 'mlp.c_fc', 'attn.c_proj']:
        if layer_type in recommendations:
            avg, std = recommendations[layer_type]
            print(f"{layer_type:15s}: operator norm = {avg:.4f} ± {std:.4f} → scale Grassmann by this factor")

if __name__ == "__main__":
    main()
