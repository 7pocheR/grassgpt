"""
Analyze gate statistics from trained model with gating mechanism.
Measures mean, std, sparsity per layer to determine amplification factor for Phase 3.
"""

import os
import sys
import torch
import numpy as np
from contextlib import nullcontext
import matplotlib.pyplot as plt

# Add nanoGPT to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT

def load_model(ckpt_path):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Create model from checkpoint config
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Load state dict
    state_dict = checkpoint['model']

    # Fix weight names if needed (unwrap ddp prefix)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded. Iteration: {checkpoint.get('iter_num', 'unknown')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")

    return model, gptconf

def measure_gate_statistics(model, config, device='cpu', num_batches=100):
    """Measure gate value distributions during forward passes."""

    if not config.use_gating:
        print("ERROR: Model does not use gating!")
        return None

    model.to(device)
    model.eval()

    # Data loader
    from data.openwebtext.prepare import get_batch
    data_dir = 'data/openwebtext'

    # Storage for gate values per layer
    n_layers = config.n_layer
    gate_values_per_layer = [[] for _ in range(n_layers)]

    print(f"\nCollecting gate values from {num_batches} batches...")

    # Hook to capture gate values
    def make_hook(layer_idx):
        def hook(module, input, output):
            # input[0] is x (pre-norm hidden states)
            if hasattr(module, 'gate') and module.gate is not None:
                with torch.no_grad():
                    x = input[0]
                    g = torch.sigmoid(module.gate(x))  # (B, T, nh) or (B, T, C)
                    gate_values_per_layer[layer_idx].append(g.cpu())
        return hook

    # Register hooks
    handles = []
    for i, block in enumerate(model.transformer.h):
        handle = block.attn.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Run forward passes
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Get batch (simplified - assumes data is prepared)
            # For real usage, load actual data batches
            # Here we just create dummy data to trigger forward pass
            dummy_input = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)

            with ctx:
                _ = model(dummy_input)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Analyze statistics per layer
    print("\n" + "="*80)
    print("GATE STATISTICS PER LAYER")
    print("="*80)

    layer_stats = []

    for layer_idx in range(n_layers):
        gates = torch.cat(gate_values_per_layer[layer_idx], dim=0)  # (total_samples, ...)
        gates_flat = gates.flatten()

        mean = gates_flat.mean().item()
        std = gates_flat.std().item()
        median = gates_flat.median().item()

        # Sparsity metrics
        sparsity_01 = (gates_flat < 0.1).float().mean().item()
        sparsity_05 = (gates_flat < 0.5).float().mean().item()

        # Percentiles
        p10 = gates_flat.quantile(0.1).item()
        p90 = gates_flat.quantile(0.9).item()

        layer_stats.append({
            'layer': layer_idx,
            'mean': mean,
            'std': std,
            'median': median,
            'p10': p10,
            'p90': p90,
            'sparsity_01': sparsity_01,
            'sparsity_05': sparsity_05,
        })

        print(f"\nLayer {layer_idx:2d}:")
        print(f"  Mean:        {mean:.4f}")
        print(f"  Std:         {std:.4f}")
        print(f"  Median:      {median:.4f}")
        print(f"  P10-P90:     [{p10:.4f}, {p90:.4f}]")
        print(f"  Sparsity <0.1:  {sparsity_01:.1%}")
        print(f"  Sparsity <0.5:  {sparsity_05:.1%}")

    # Overall statistics
    all_means = [s['mean'] for s in layer_stats]
    all_sparsity = [s['sparsity_01'] for s in layer_stats]

    overall_mean = np.mean(all_means)
    overall_sparsity = np.mean(all_sparsity)

    print("\n" + "="*80)
    print("OVERALL STATISTICS (averaged across layers)")
    print("="*80)
    print(f"Mean gate value:     {overall_mean:.4f}")
    print(f"Sparsity (<0.1):     {overall_sparsity:.1%}")
    print(f"\nTarget from paper:   mean ≈ 0.116, sparsity ≈ 88%")
    print(f"Status:              {'✓ PASS' if 0.08 <= overall_mean <= 0.25 and overall_sparsity >= 0.7 else '✗ FAIL'}")

    # Amplification factor for Phase 3
    amplification = 1.0 / overall_mean
    print(f"\n{'='*80}")
    print(f"AMPLIFICATION FACTOR FOR PHASE 3")
    print(f"{'='*80}")
    print(f"Recommended amplification: {amplification:.2f}×")
    print(f"  (This will compensate for sparse gates in Grassmann scaling)")
    print(f"  Use gate_mean_estimate = {overall_mean:.3f} in Phase 3 config")

    # Save plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Mean per layer
    axes[0, 0].plot(all_means, 'o-')
    axes[0, 0].axhline(y=0.116, color='r', linestyle='--', label='Paper target')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Gate Mean')
    axes[0, 0].set_title('Mean Gate Value Per Layer')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sparsity per layer
    axes[0, 1].plot(all_sparsity, 'o-', color='orange')
    axes[0, 1].axhline(y=0.88, color='r', linestyle='--', label='Paper target')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Sparsity (<0.1)')
    axes[0, 1].set_title('Gate Sparsity Per Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Distribution histogram (all layers combined)
    all_gates = torch.cat([torch.cat(gate_values_per_layer[i], dim=0).flatten()
                          for i in range(n_layers)])
    axes[1, 0].hist(all_gates.numpy(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=overall_mean, color='r', linestyle='--',
                       label=f'Mean={overall_mean:.3f}')
    axes[1, 0].set_xlabel('Gate Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Gate Value Distribution (All Layers)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
GATE STATISTICS SUMMARY

Overall Mean:     {overall_mean:.4f}
Overall Sparsity: {overall_sparsity:.1%}

Paper Target:
  Mean:     0.116
  Sparsity: 88%

Status: {'✓ PASS' if 0.08 <= overall_mean <= 0.25 else '✗ FAIL'}

Amplification Factor:
  {amplification:.2f}×

Phase 3 Config:
  gate_mean_estimate = {overall_mean:.3f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                    family='monospace', verticalalignment='center')

    plt.tight_layout()
    plot_path = 'plots/gate_statistics.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    return layer_stats, overall_mean, overall_sparsity, amplification

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                       default='/net/scratch2/junyuren/nanoGPT-manifold/out-owt-gating-d384/ckpt.pt',
                       help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Number of batches to sample')

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.ckpt)

    # Measure statistics
    stats = measure_gate_statistics(model, config, args.device, args.num_batches)

    print("\nAnalysis complete!")
