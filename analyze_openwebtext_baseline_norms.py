"""
Analyze OpenWebText baseline checkpoint to extract:
- Operator norms (max singular value)
- Stable ranks (number of singular values to capture 99% variance)
- Compare with Shakespeare baseline values
"""

import torch
import numpy as np
import os

# Load the checkpoint
ckpt_path = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-baseline/ckpt.pt'
print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

model_state = checkpoint['model']
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']

print(f"\nCheckpoint info:")
print(f"  Iteration: {iter_num}")
print(f"  Best val loss: {best_val_loss:.4f}")

# Analyze c_attn and mlp.c_fc weights
print(f"\n{'='*60}")
print("Operator Norm and Stable Rank Analysis")
print(f"{'='*60}\n")

# Storage for results
c_attn_results = []
mlp_fc_results = []

n_layer = 6  # From config

for layer_idx in range(n_layer):
    # c_attn weight (combined QKV, shape 1152, 384)
    c_attn_key = f'transformer.h.{layer_idx}.attn.c_attn.weight'
    if c_attn_key in model_state:
        W = model_state[c_attn_key].float().numpy()
        
        # Compute SVD
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        
        # Operator norm (largest singular value)
        op_norm = S[0]
        
        # Stable rank (99% variance)
        cumsum = np.cumsum(S**2)
        total = cumsum[-1]
        stable_rank = np.searchsorted(cumsum, 0.99 * total) + 1
        
        c_attn_results.append({
            'layer': layer_idx,
            'op_norm': op_norm,
            'stable_rank': stable_rank,
            'shape': W.shape
        })
        
        print(f"Layer {layer_idx} c_attn:")
        print(f"  Shape: {W.shape}")
        print(f"  Operator norm: {op_norm:.4f}")
        print(f"  Stable rank (99%): {stable_rank}")
    
    # mlp.c_fc weight (shape 1536, 384)
    mlp_fc_key = f'transformer.h.{layer_idx}.mlp.c_fc.weight'
    if mlp_fc_key in model_state:
        W = model_state[mlp_fc_key].float().numpy()
        
        # Compute SVD
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        
        # Operator norm
        op_norm = S[0]
        
        # Stable rank (99%)
        cumsum = np.cumsum(S**2)
        total = cumsum[-1]
        stable_rank = np.searchsorted(cumsum, 0.99 * total) + 1
        
        mlp_fc_results.append({
            'layer': layer_idx,
            'op_norm': op_norm,
            'stable_rank': stable_rank,
            'shape': W.shape
        })
        
        print(f"Layer {layer_idx} mlp.c_fc:")
        print(f"  Shape: {W.shape}")
        print(f"  Operator norm: {op_norm:.4f}")
        print(f"  Stable rank (99%): {stable_rank}")
    
    print()

# Compute averages
print(f"{'='*60}")
print("Summary Statistics")
print(f"{'='*60}\n")

if c_attn_results:
    avg_op_norm_c_attn = np.mean([r['op_norm'] for r in c_attn_results])
    avg_stable_rank_c_attn = np.mean([r['stable_rank'] for r in c_attn_results])
    print(f"c_attn (averaged across {len(c_attn_results)} layers):")
    print(f"  Average operator norm: {avg_op_norm_c_attn:.4f}")
    print(f"  Average stable rank: {avg_stable_rank_c_attn:.1f}")
    print(f"  Stable rank × 2: {avg_stable_rank_c_attn * 2:.1f}")
    print()

if mlp_fc_results:
    avg_op_norm_mlp_fc = np.mean([r['op_norm'] for r in mlp_fc_results])
    avg_stable_rank_mlp_fc = np.mean([r['stable_rank'] for r in mlp_fc_results])
    print(f"mlp.c_fc (averaged across {len(mlp_fc_results)} layers):")
    print(f"  Average operator norm: {avg_op_norm_mlp_fc:.4f}")
    print(f"  Average stable rank: {avg_stable_rank_mlp_fc:.1f}")
    print(f"  Stable rank × 2: {avg_stable_rank_mlp_fc * 2:.1f}")
    print()

# Comparison with Shakespeare
print(f"{'='*60}")
print("Comparison with Shakespeare Baseline")
print(f"{'='*60}\n")

shakespeare_c_attn_norm = 4.6293
shakespeare_mlp_fc_norm = 10.1372
shakespeare_c_attn_stable_rank = 24
shakespeare_mlp_fc_stable_rank = 10

if c_attn_results:
    print(f"c_attn operator norm:")
    print(f"  OpenWebText: {avg_op_norm_c_attn:.4f}")
    print(f"  Shakespeare: {shakespeare_c_attn_norm:.4f}")
    print(f"  Difference: {avg_op_norm_c_attn - shakespeare_c_attn_norm:+.4f} ({((avg_op_norm_c_attn/shakespeare_c_attn_norm - 1) * 100):+.1f}%)")
    print()
    
    print(f"c_attn stable rank:")
    print(f"  OpenWebText: {avg_stable_rank_c_attn:.1f}")
    print(f"  Shakespeare: {shakespeare_c_attn_stable_rank}")
    print(f"  Difference: {avg_stable_rank_c_attn - shakespeare_c_attn_stable_rank:+.1f}")
    print()

if mlp_fc_results:
    print(f"mlp.c_fc operator norm:")
    print(f"  OpenWebText: {avg_op_norm_mlp_fc:.4f}")
    print(f"  Shakespeare: {shakespeare_mlp_fc_norm:.4f}")
    print(f"  Difference: {avg_op_norm_mlp_fc - shakespeare_mlp_fc_norm:+.4f} ({((avg_op_norm_mlp_fc/shakespeare_mlp_fc_norm - 1) * 100):+.1f}%)")
    print()
    
    print(f"mlp.c_fc stable rank:")
    print(f"  OpenWebText: {avg_stable_rank_mlp_fc:.1f}")
    print(f"  Shakespeare: {shakespeare_mlp_fc_stable_rank}")
    print(f"  Difference: {avg_stable_rank_mlp_fc - shakespeare_mlp_fc_stable_rank:+.1f}")
    print()

# Print recommended config values
print(f"{'='*60}")
print("Recommended Config Values for OpenWebText")
print(f"{'='*60}\n")

if c_attn_results and mlp_fc_results:
    print("# For stable_rank_x2 strategy:")
    print(f"grass_rank_c_attn = {int(avg_stable_rank_c_attn * 2)}")
    print(f"grass_rank_mlp_fc = {int(avg_stable_rank_mlp_fc * 2)}")
    print(f"grass_scale_c_attn = {avg_op_norm_c_attn:.4f}")
    print(f"grass_scale_mlp_fc = {avg_op_norm_mlp_fc:.4f}")
    print()
    
    print("# For per-block strategy:")
    print(f"grass_rank_c_attn = {[r['stable_rank'] for r in c_attn_results]}")
    print(f"grass_scale_c_attn = {[round(r['op_norm'], 4) for r in c_attn_results]}")
    print(f"grass_rank_mlp_fc = {[r['stable_rank'] for r in mlp_fc_results]}")
    print(f"grass_scale_mlp_fc = {[round(r['op_norm'], 4) for r in mlp_fc_results]}")
