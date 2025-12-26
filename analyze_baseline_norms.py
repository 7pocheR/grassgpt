#!/usr/bin/env python3
"""
Analyze weight matrix operator norms from baseline checkpoint.

This helps determine appropriate scaling factors for Grassmann weights,
since Grassmann manifold fixes operator norm = 1.
"""

import torch
import argparse
from pathlib import Path
import numpy as np

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

    # Operator norm (largest singular value)
    op_norm = S[0].item()

    # Frobenius norm
    frob_norm = torch.norm(weight).item()

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

    return {
        'shape': list(weight.shape),
        'op_norm': op_norm,
        'frob_norm': frob_norm,
        'eff_rank': eff_rank,
        'actual_rank': min(weight.shape),
        'rank_99': rank_99,
        'cond_number': cond_number,
        'singular_values': S.cpu().numpy()[:10],  # Top 10
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze baseline weight norms')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to baseline checkpoint')
    parser.add_argument('--output', type=str, default='baseline_weight_analysis.md', help='Output file')
    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Analyze each weight matrix
    results = {}
    for name, param in state_dict.items():
        if 'weight' in name and param.dim() == 2:
            result = analyze_weight_matrix(name, param)
            if result:
                results[name] = result

    # Generate markdown report
    lines = []
    lines.append("# Baseline Weight Matrix Analysis")
    lines.append("")
    lines.append(f"**Checkpoint:** {checkpoint_path}")
    lines.append(f"**Iteration:** {checkpoint.get('iter_num', 'unknown')}")
    lines.append(f"**Best val loss:** {checkpoint.get('best_val_loss', 'unknown')}")
    lines.append("")

    # Summary table
    lines.append("## Weight Matrix Statistics")
    lines.append("")
    lines.append("| Layer | Shape | Op Norm | Frob Norm | Eff Rank | Rank@99% | Condition |")
    lines.append("|-------|-------|---------|-----------|----------|----------|-----------|")

    # Group by layer type
    layer_groups = {
        'c_attn': [],
        'attn.c_proj': [],
        'mlp.c_fc': [],
        'mlp.c_proj': [],
        'other': []
    }

    for name, stats in sorted(results.items()):
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

        # Add to table
        shape_str = f"{stats['shape'][0]}×{stats['shape'][1]}"
        lines.append(
            f"| {name} | {shape_str} | {stats['op_norm']:.4f} | "
            f"{stats['frob_norm']:.2f} | {stats['eff_rank']:.1f} | "
            f"{stats['rank_99']} | {stats['cond_number']:.1f} |"
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

        op_norms = [stats['op_norm'] for _, stats in items]
        frob_norms = [stats['frob_norm'] for _, stats in items]
        eff_ranks = [stats['eff_rank'] for _, stats in items]

        lines.append(f"- **Count:** {len(items)}")
        lines.append(f"- **Operator norm:** mean={np.mean(op_norms):.4f}, std={np.std(op_norms):.4f}, range=[{np.min(op_norms):.4f}, {np.max(op_norms):.4f}]")
        lines.append(f"- **Frobenius norm:** mean={np.mean(frob_norms):.2f}, std={np.std(frob_norms):.2f}")
        lines.append(f"- **Effective rank:** mean={np.mean(eff_ranks):.1f}, std={np.std(eff_ranks):.1f}")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations for Grassmann Scaling")
    lines.append("")
    lines.append("Since Grassmann manifold fixes operator norm = 1, we need to scale outputs to match baseline:")
    lines.append("")

    for layer_type, items in [('c_attn', layer_groups['c_attn']),
                               ('mlp.c_fc', layer_groups['mlp.c_fc']),
                               ('attn.c_proj', layer_groups['attn.c_proj'])]:
        if items:
            avg_op_norm = np.mean([stats['op_norm'] for _, stats in items])
            lines.append(f"- **{layer_type}**: Scale by {avg_op_norm:.4f} (mean operator norm)")

    lines.append("")
    lines.append("### Implementation")
    lines.append("")
    lines.append("```python")
    lines.append("# In model.py, after Grassmann update:")

    for layer_type, items in [('c_attn', layer_groups['c_attn']),
                               ('mlp.c_fc', layer_groups['mlp.c_fc'])]:
        if items:
            avg_op_norm = np.mean([stats['op_norm'] for _, stats in items])
            lines.append(f"# {layer_type}: multiply output by {avg_op_norm:.4f}")
            lines.append(f"block.{layer_type.replace('.', '_')}_scale = {avg_op_norm:.4f}")

    lines.append("```")
    lines.append("")

    # Singular value distributions
    lines.append("## Singular Value Analysis")
    lines.append("")
    lines.append("Top 10 singular values for key layers:")
    lines.append("")

    for layer_type in ['c_attn', 'mlp.c_fc', 'attn.c_proj']:
        items = layer_groups[layer_type]
        if items and len(items) > 0:
            name, stats = items[0]  # Show first block as example
            sv = stats['singular_values']
            sv_str = ", ".join([f"{s:.3f}" for s in sv[:10]])
            lines.append(f"**{name}:** [{sv_str}, ...]")
            lines.append("")

    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Analysis complete!")
    print(f"  Report: {output_path}")

    # Print key findings to console
    print("\n=== Key Findings ===")
    for layer_type, items in [('c_attn', layer_groups['c_attn']),
                               ('mlp.c_fc', layer_groups['mlp.c_fc']),
                               ('attn.c_proj', layer_groups['attn.c_proj'])]:
        if items:
            avg_op_norm = np.mean([stats['op_norm'] for _, stats in items])
            print(f"{layer_type:15s}: operator norm = {avg_op_norm:.4f} → scale Grassmann by this factor")

if __name__ == "__main__":
    main()
