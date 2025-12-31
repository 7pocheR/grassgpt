#!/bin/bash
#SBATCH --job-name=gpt2m_grass_gate
#SBATCH --output=logs/%j_gpt2m_grass_gate.out
#SBATCH --error=logs/%j_gpt2m_grass_gate.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs for optimal performance
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# Phase 3: Grassmann + Gating at d=1024
# Note: 12-hour job (partition limit), will need resume for full 100k iters

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Navigate to working directory
cd /home/junyuren/nanoGPT

# Ensure output directory exists
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-gate-x10

echo "=== Configuration ==="
echo "Architecture: GPT-2 Medium (d=1024, 24 layers, 16 heads)"
echo "Grassmann: c_attn + mlp.c_fc (non-skip-facing, 58% coverage)"
echo "Scaling: x=10 uniform"
echo "Gating: Head-wise (G1 position)"
echo "Batch size: 64 per GPU × 12 grad_accum × 4 GPUs = 3072 (3.1M tokens)"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_gated.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-gate-x10/ckpt.pt"
