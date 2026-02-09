#!/usr/bin/env python3
"""Track progress of component isolation experiments"""
import re
import os
from pathlib import Path

def parse_steps(log_file):
    """Parse training steps from log file"""
    if not os.path.exists(log_file):
        return []
    
    steps = []
    with open(log_file) as f:
        for line in f:
            match = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                steps.append((step, train_loss, val_loss))
    return steps

def tokens_from_step(step, batch_size=32, grad_accum=12, seq_len=1024, n_gpus=4):
    """Calculate tokens from step number"""
    tokens_per_iter = batch_size * grad_accum * seq_len * n_gpus
    return step * tokens_per_iter / 1e9  # Return in billions

# Job configurations
jobs = {
    '615947_comp_qkv_only': {
        'name': 'QKV-only (hybrid)',
        'batch': 32, 'grad_accum': 12,
        'description': 'c_attn: 3-block r=384 + 48 gates | c_fc: AdamW'
    },
    '615948_comp_mlp_only': {
        'name': 'MLP-only',
        'batch': 32, 'grad_accum': 12,
        'description': 'c_attn: AdamW | c_fc: 4-block r=384 + 4 gates'
    },
    '613768_comp_baseline': {
        'name': 'Baseline',
        'batch': 32, 'grad_accum': 12,
        'description': 'c_attn: AdamW | c_fc: AdamW'
    },
    '608059_gpt2m_hybgate_r384': {
        'name': '24L Hybrid (reference)',
        'batch': 32, 'grad_accum': 12,
        'description': 'c_attn: 3-block r=384 + 48 gates | c_fc: 4-block r=384 + 4 gates'
    }
}

print("=" * 100)
print("COMPONENT ISOLATION EXPERIMENT PROGRESS")
print("=" * 100)
print()

for job_id, config in jobs.items():
    log_file = f'logs/{job_id}.out'
    steps = parse_steps(log_file)
    
    print(f"{config['name']}")
    print(f"  {config['description']}")
    
    if not steps:
        print(f"  Status: Waiting to start (log file not found)")
        print()
        continue
    
    latest_step, latest_train, latest_val = steps[-1]
    tokens_b = tokens_from_step(latest_step, config['batch'], config['grad_accum'])
    
    print(f"  Latest: step {latest_step} | {tokens_b:.2f}B tokens | val loss {latest_val:.4f}")
    
    # Show progress at key checkpoints
    checkpoints = {1000: '1.57B', 3000: '4.72B', 5000: '7.86B', 10000: '15.73B'}
    for step, tokens_label in checkpoints.items():
        for s, t, v in steps:
            if s == step:
                print(f"    @ {tokens_label} tokens: val {v:.4f}")
                break
    print()

print("=" * 100)
print("FAIR COMPARISON (All use same architecture except which layers use Grassmann)")
print("=" * 100)
