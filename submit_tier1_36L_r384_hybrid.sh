#!/bin/bash
#SBATCH --job-name=tier1_36L_r384
#SBATCH --output=logs/%j_tier1_36L_r384.out
#SBATCH --error=logs/%j_tier1_36L_r384.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# TIER 1 EXPERIMENT 1: 36L, r=384, hybrid gating
# Question: Is 36L slow because of depth or low rank?
# Hypothesis: If faster convergence than r=256, rank was the bottleneck

echo "=== TIER 1 EXPERIMENT 1: 36L, r=384, Hybrid Gating ==="
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
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r384

echo "=== Configuration ==="
echo "Architecture: 36 layers, d=1024, 16 heads (~660M params)"
echo "Grassmann rank: r=384 (37.5% of width) ← INCREASED from r=256"
echo "Gating: Hybrid (48 head-level + 4 block-level = 52 gates/layer)"
echo "Training: 168k iters (100 tokens/param)"
echo ""
echo "KEY COMPARISON:"
echo "  Previous 36L r=256: Slow convergence, val ~3.26 @ 14.2B tokens"
echo "  This 36L r=384: Should converge faster if rank was bottleneck"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_large_grassmann_hybridgate_r384.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r384/ckpt.pt"
