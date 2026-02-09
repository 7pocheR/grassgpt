#!/bin/bash
#SBATCH --job-name=tier1_24L_fb384
#SBATCH --output=logs/%j_tier1_24L_fb384.out
#SBATCH --error=logs/%j_tier1_24L_fb384.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# TIER 1 EXPERIMENT 3: 24L, r=384, full block decomposition
# Question: Does fine-grained decomposition help 24L like it helps 36L?
# Hypothesis: If outperforms hybrid, block granularity matters

echo "=== TIER 1 EXPERIMENT 3: 24L, r=384, Full Block Decomposition ==="
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
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-fullblock-r384

echo "=== Configuration ==="
echo "Architecture: 24 layers, d=1024, 16 heads (~530M params)"
echo "Grassmann rank: r=384 (37.5% of width)"
echo "Full Block Decomposition:"
echo "  - c_attn: 16×16 grid = 256 blocks per projection (Q/K/V)"
echo "  - Total c_attn blocks: 768 per layer"
echo "  - Block size: 64×64, rank=32 per block"
echo "  - mlp.c_fc: 4 blocks, rank=384"
echo "Training: 100k iters"
echo ""
echo "KEY COMPARISON:"
echo "  24L Hybrid r=384: Val ~3.1 @ 40.9B tokens"
echo "  36L Full r=256: Val ~3.17 @ 20.4B tokens (better than 36L hybrid)"
echo "  This 24L Full r=384: Should show if fine-grained decomp helps 24L"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_fullblock_r384.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-fullblock-r384/ckpt.pt"
