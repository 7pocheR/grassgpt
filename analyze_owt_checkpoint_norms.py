#!/usr/bin/env python3
"""
Analyze operator norms and stable ranks from OpenWebText baseline checkpoints.
Average across multiple checkpoints for robustness.
"""

import torch
import numpy as np
from pathlib import Path
import argparse

def analyze_weight_matrix(W):
    """Compute operator norm, ranks at multiple variance thresholds, and effective rank."""
    # Compute SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Operator norm (largest singular value)
    op_norm = S[0].item()

    # Compute ranks at multiple variance thresholds
    cumsum = torch.cumsum(S**2, dim=0)
    total_energy = cumsum[-1]

    rank_50 = torch.sum(cumsum < 0.50 * total_energy).item() + 1
    rank_85 = torch.sum(cumsum < 0.85 * total_energy).item() + 1
    rank_95 = torch.sum(cumsum < 0.95 * total_energy).item() + 1
    rank_99 = torch.sum(cumsum < 0.99 * total_energy).item() + 1

    # Effective rank (entropy-based)
    S_normalized = S / (S.sum() + 1e-10)
    entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-10))
    eff_rank = torch.exp(entropy).item()

    return {
        'op_norm': op_norm,
        'rank_50': rank_50,
        'rank_85': rank_85,
        'rank_95': rank_95,
        'rank_99': rank_99,
        'eff_rank': eff_rank
    }

def main():
    # Load checkpoint
    ckpt_path = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-baseline/ckpt.pt'
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    state_dict = checkpoint['model']
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Checkpoint info:")
    print(f"  Iteration: {iter_num}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print()
    
    # Analyze per layer type
    results = {
        'c_attn': [],
        'attn.c_proj': [],
        'mlp.c_fc': [],
        'mlp.c_proj': []
    }
    
    n_layer = 6
    
    for layer_idx in range(n_layer):
        # c_attn
        key = f'_orig_mod.transformer.h.{layer_idx}.attn.c_attn.weight'
        if key in state_dict:
            W = state_dict[key].float()
            stats = analyze_weight_matrix(W)
            stats['layer'] = layer_idx
            results['c_attn'].append(stats)
        
        # attn.c_proj
        key = f'_orig_mod.transformer.h.{layer_idx}.attn.c_proj.weight'
        if key in state_dict:
            W = state_dict[key].float()
            stats = analyze_weight_matrix(W)
            stats['layer'] = layer_idx
            results['attn.c_proj'].append(stats)
        
        # mlp.c_fc
        key = f'_orig_mod.transformer.h.{layer_idx}.mlp.c_fc.weight'
        if key in state_dict:
            W = state_dict[key].float()
            stats = analyze_weight_matrix(W)
            stats['layer'] = layer_idx
            results['mlp.c_fc'].append(stats)
        
        # mlp.c_proj
        key = f'_orig_mod.transformer.h.{layer_idx}.mlp.c_proj.weight'
        if key in state_dict:
            W = state_dict[key].float()
            stats = analyze_weight_matrix(W)
            stats['layer'] = layer_idx
            results['mlp.c_proj'].append(stats)
    
    # Print per-layer results
    print("="*80)
    print("PER-LAYER ANALYSIS")
    print("="*80)
    print()
    
    for layer_type in ['c_attn', 'mlp.c_fc', 'attn.c_proj', 'mlp.c_proj']:
        if not results[layer_type]:
            continue

        print(f"### {layer_type}")
        print()
        print("| Layer | Op Norm | Rank@50% | Rank@85% | Rank@95% | Rank@99% | Eff Rank |")
        print("|-------|---------|----------|----------|----------|----------|----------|")

        for stats in results[layer_type]:
            print(f"| {stats['layer']} | {stats['op_norm']:.4f} | {stats['rank_50']} | {stats['rank_85']} | {stats['rank_95']} | {stats['rank_99']} | {stats['eff_rank']:.1f} |")

        print()
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS (Averaged across layers)")
    print("="*80)
    print()
    
    for layer_type in ['c_attn', 'mlp.c_fc', 'attn.c_proj', 'mlp.c_proj']:
        if not results[layer_type]:
            continue

        op_norms = [s['op_norm'] for s in results[layer_type]]
        rank_50s = [s['rank_50'] for s in results[layer_type]]
        rank_85s = [s['rank_85'] for s in results[layer_type]]
        rank_95s = [s['rank_95'] for s in results[layer_type]]
        rank_99s = [s['rank_99'] for s in results[layer_type]]
        eff_ranks = [s['eff_rank'] for s in results[layer_type]]

        print(f"{layer_type}:")
        print(f"  Operator norm:  mean={np.mean(op_norms):.4f}, std={np.std(op_norms):.4f}, min={np.min(op_norms):.4f}, max={np.max(op_norms):.4f}")
        print(f"  Rank@50%:       mean={np.mean(rank_50s):.1f}, std={np.std(rank_50s):.1f}")
        print(f"  Rank@85%:       mean={np.mean(rank_85s):.1f}, std={np.std(rank_85s):.1f}")
        print(f"  Rank@95%:       mean={np.mean(rank_95s):.1f}, std={np.std(rank_95s):.1f}")
        print(f"  Rank@99%:       mean={np.mean(rank_99s):.1f}, std={np.std(rank_99s):.1f}")
        print(f"  Effective rank: mean={np.mean(eff_ranks):.1f}, std={np.std(eff_ranks):.1f}")
        print()
    
    # Print recommended configs
    print("="*80)
    print("RECOMMENDED GRASSMANN CONFIGS")
    print("="*80)
    print()
    
    # Use c_attn and mlp.c_fc (what we optimize with Grassmann in phase2.5)
    c_attn_norms = [s['op_norm'] for s in results['c_attn']]
    mlp_fc_norms = [s['op_norm'] for s in results['mlp.c_fc']]

    c_attn_r50 = [s['rank_50'] for s in results['c_attn']]
    c_attn_r85 = [s['rank_85'] for s in results['c_attn']]
    c_attn_r95 = [s['rank_95'] for s in results['c_attn']]
    c_attn_r99 = [s['rank_99'] for s in results['c_attn']]

    mlp_fc_r50 = [s['rank_50'] for s in results['mlp.c_fc']]
    mlp_fc_r85 = [s['rank_85'] for s in results['mlp.c_fc']]
    mlp_fc_r95 = [s['rank_95'] for s in results['mlp.c_fc']]
    mlp_fc_r99 = [s['rank_99'] for s in results['mlp.c_fc']]

    print("# For uniform scaling across all layers:")
    print(f"grass_scale_c_attn = {np.mean(c_attn_norms):.4f}  # (was 4.6293 from Shakespeare)")
    print(f"grass_scale_mlp_fc = {np.mean(mlp_fc_norms):.4f}  # (was 10.1372 from Shakespeare)")
    print()

    print("# Rank recommendations at different variance thresholds:")
    print(f"# c_attn ranks: @50%={int(np.mean(c_attn_r50))}, @85%={int(np.mean(c_attn_r85))}, @95%={int(np.mean(c_attn_r95))}, @99%={int(np.mean(c_attn_r99))}")
    print(f"# mlp.c_fc ranks: @50%={int(np.mean(mlp_fc_r50))}, @85%={int(np.mean(mlp_fc_r85))}, @95%={int(np.mean(mlp_fc_r95))}, @99%={int(np.mean(mlp_fc_r99))}")
    print()

    print("# Conservative (50% variance):")
    print(f"grass_rank_c_attn = {int(np.mean(c_attn_r50))}")
    print(f"grass_rank_mlp_fc = {int(np.mean(mlp_fc_r50))}")
    print()

    print("# Moderate (85% variance):")
    print(f"grass_rank_c_attn = {int(np.mean(c_attn_r85))}")
    print(f"grass_rank_mlp_fc = {int(np.mean(mlp_fc_r85))}")
    print()

    print("# Aggressive (95% variance):")
    print(f"grass_rank_c_attn = {int(np.mean(c_attn_r95))}")
    print(f"grass_rank_mlp_fc = {int(np.mean(mlp_fc_r95))}")
    print()

    print("# Very Aggressive (99% variance - original stable_rank_x2 x2):")
    print(f"grass_rank_c_attn = {int(np.mean(c_attn_r99) * 2)}")
    print(f"grass_rank_mlp_fc = {int(np.mean(mlp_fc_r99) * 2)}")
    print()

    print("# For per-block strategy (99% variance):")
    print(f"grass_rank_c_attn = {[s['rank_99'] for s in results['c_attn']]}")
    print(f"grass_scale_c_attn = {[round(s['op_norm'], 4) for s in results['c_attn']]}")
    print(f"grass_rank_mlp_fc = {[s['rank_99'] for s in results['mlp.c_fc']]}")
    print(f"grass_scale_mlp_fc = {[round(s['op_norm'], 4) for s in results['mlp.c_fc']]}")
    print()
    
    # Comparison with Shakespeare
    print("="*80)
    print("COMPARISON: Shakespeare vs OpenWebText")
    print("="*80)
    print()
    
    shakespeare_c_attn = 4.6293
    shakespeare_mlp_fc = 10.1372
    
    owt_c_attn = np.mean(c_attn_norms)
    owt_mlp_fc = np.mean(mlp_fc_norms)
    
    print(f"c_attn operator norm:")
    print(f"  Shakespeare: {shakespeare_c_attn:.4f}")
    print(f"  OpenWebText: {owt_c_attn:.4f}")
    print(f"  Difference:  {owt_c_attn - shakespeare_c_attn:+.4f} ({((owt_c_attn/shakespeare_c_attn - 1) * 100):+.1f}%)")
    print()
    
    print(f"mlp.c_fc operator norm:")
    print(f"  Shakespeare: {shakespeare_mlp_fc:.4f}")
    print(f"  OpenWebText: {owt_mlp_fc:.4f}")
    print(f"  Difference:  {owt_mlp_fc - shakespeare_mlp_fc:+.4f} ({((owt_mlp_fc/shakespeare_mlp_fc - 1) * 100):+.1f}%)")
    print()
    
    print("Impact of using Shakespeare norms on OpenWebText:")
    if owt_c_attn > shakespeare_c_attn:
        print(f"  c_attn:   UNDER-scaled by {((shakespeare_c_attn/owt_c_attn - 1) * 100):.1f}%")
    else:
        print(f"  c_attn:   OVER-scaled by {((shakespeare_c_attn/owt_c_attn - 1) * 100):.1f}%")
    
    if owt_mlp_fc > shakespeare_mlp_fc:
        print(f"  mlp.c_fc: UNDER-scaled by {((shakespeare_mlp_fc/owt_mlp_fc - 1) * 100):.1f}%")
    else:
        print(f"  mlp.c_fc: OVER-scaled by {((shakespeare_mlp_fc/owt_mlp_fc - 1) * 100):.1f}%")

if __name__ == "__main__":
    main()
