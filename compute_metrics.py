#!/usr/bin/env python3
"""
Compute comprehensive metrics for trained checkpoints.

Metrics computed:
1. Perplexity: exp(val_loss)
2. Bits-per-character: val_loss / log(2)
3. Sample quality (text generation)
4. Generalization gap: val_loss - train_loss
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Import from nanoGPT
from model import GPTConfig, GPT
from train import estimate_loss

def compute_perplexity_bpc(val_loss):
    """Compute perplexity and bits-per-character from validation loss."""
    perplexity = np.exp(val_loss)
    bpc = val_loss / np.log(2)
    return perplexity, bpc

def load_checkpoint_and_metrics(checkpoint_path):
    """Load checkpoint and extract metrics."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract losses
    val_loss = checkpoint.get('val_loss', None)
    train_loss = checkpoint.get('train_loss', None)
    best_val_loss = checkpoint.get('best_val_loss', None)
    iter_num = checkpoint.get('iter_num', 0)

    if val_loss is None:
        print(f"No val_loss found in {checkpoint_path}")
        return None

    # Compute metrics
    perplexity, bpc = compute_perplexity_bpc(val_loss)
    gen_gap = val_loss - train_loss if train_loss is not None else None

    return {
        'checkpoint': checkpoint_path,
        'iter_num': iter_num,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'best_val_loss': best_val_loss,
        'perplexity': perplexity,
        'bpc': bpc,
        'gen_gap': gen_gap
    }

def generate_samples(checkpoint_path, num_samples=5, max_new_tokens=200, temperature=0.8, top_k=200):
    """Generate text samples from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return []

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Initialize model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Handle potential prefix
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load tokenizer (assume shakespeare_char for now)
    data_dir = 'data/shakespeare_char'
    meta_path = os.path.join(data_dir, 'meta.pkl')

    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("No meta.pkl found, cannot generate samples")
        return []

    # Generate samples
    samples = []
    start_ids = encode('\n')  # Start with newline
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        for i in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            sample_text = decode(y[0].tolist())
            samples.append(sample_text)
            print(f"Sample {i+1}/{num_samples}:")
            print(sample_text)
            print("-" * 80)

    return samples

def main():
    parser = argparse.ArgumentParser(description='Compute metrics for trained checkpoints')
    parser.add_argument('--checkpoints', type=str, nargs='+',
                        help='List of checkpoint directories or paths')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt.pt',
                        help='Checkpoint filename (default: ckpt.pt)')
    parser.add_argument('--generate_samples', action='store_true',
                        help='Generate text samples')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate per checkpoint')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Max tokens to generate per sample')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k sampling')

    args = parser.parse_args()

    # Collect results
    results = []

    for checkpoint_arg in args.checkpoints:
        # Handle both directory and file paths
        if os.path.isdir(checkpoint_arg):
            checkpoint_path = os.path.join(checkpoint_arg, args.checkpoint_name)
        else:
            checkpoint_path = checkpoint_arg

        # Extract experiment name
        exp_name = Path(checkpoint_path).parent.name

        print(f"\n{'='*80}")
        print(f"Processing: {exp_name}")
        print(f"{'='*80}")

        # Load and compute metrics
        metrics = load_checkpoint_and_metrics(checkpoint_path)
        if metrics is None:
            continue

        metrics['experiment'] = exp_name
        results.append(metrics)

        # Print metrics
        print(f"Iteration: {metrics['iter_num']}")
        print(f"Val Loss: {metrics['val_loss']:.4f}")
        print(f"Train Loss: {metrics['train_loss']:.4f}" if metrics['train_loss'] is not None else "Train Loss: N/A")
        print(f"Best Val Loss: {metrics['best_val_loss']:.4f}" if metrics['best_val_loss'] is not None else "Best Val Loss: N/A")
        print(f"Perplexity: {metrics['perplexity']:.2f}")
        print(f"Bits-per-char: {metrics['bpc']:.4f}")
        print(f"Generalization Gap: {metrics['gen_gap']:.4f}" if metrics['gen_gap'] is not None else "Gen Gap: N/A")

        # Generate samples if requested
        if args.generate_samples:
            print(f"\nGenerating {args.num_samples} samples...")
            samples = generate_samples(
                checkpoint_path,
                num_samples=args.num_samples,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )

    # Print summary table
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Experiment':<30} {'Val Loss':>10} {'Perplexity':>12} {'BPC':>8} {'Gen Gap':>10}")
        print("-" * 80)

        # Sort by val_loss
        results.sort(key=lambda x: x['val_loss'])

        for r in results:
            gen_gap_str = f"{r['gen_gap']:.4f}" if r['gen_gap'] is not None else "N/A"
            print(f"{r['experiment']:<30} {r['val_loss']:>10.4f} {r['perplexity']:>12.2f} {r['bpc']:>8.4f} {gen_gap_str:>10}")

        # Print best
        best = results[0]
        print(f"\nBest: {best['experiment']} (val_loss={best['val_loss']:.4f}, perplexity={best['perplexity']:.2f})")

if __name__ == '__main__':
    main()
