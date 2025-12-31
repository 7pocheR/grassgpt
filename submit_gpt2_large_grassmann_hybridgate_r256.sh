#!/bin/bash
#SBATCH --job-name=gpt2large_hybgate_r256
#SBATCH --output=logs/%j_gpt2large_hybgate_r256.out
#SBATCH --error=logs/%j_gpt2large_hybgate_r256.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs for optimal performance
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# Optimal "Wide-Deep" Architecture: 36 layers, r=256, hybrid gating
# Key improvements: Deeper network, stronger constraint, proper training (100 tokens/param)

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
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r256

echo "=== Optimal Wide-Deep Configuration ==="
echo "Architecture: 36 layers, d=1024, 16 heads (~660M params)"
echo "Grassmann: c_attn (3 blocks) + mlp.c_fc (4 blocks)"
echo ""
echo "Key Improvements over 24-layer:"
echo "  - Depth: 36 layers (50% more, compensates for lower rank)"
echo "  - Rank: r=256 (25% of width, STRONGER constraint vs 37.5%)"
echo "  - Training: 168k iters (100 tokens/param vs 3.8 before!)"
echo "  - LR schedule: Fine-tuning region 150k-168k (18k iters)"
echo ""
echo "Hybrid Gating:"
echo "  - c_attn: 48 head-level gates (16 heads × 3 QKV)"
echo "  - mlp.c_fc: 4 block-level gates"
echo "  - Total: 52 gates/layer, 1.28M gate params"
echo ""
echo "Learning Rates:"
echo "  - Grassmann: 3e-3 (conservative for depth)"
echo "  - Gates: 1.5e-3 (2.5× base)"
echo "  - Base: 6e-4, Embeddings: 3e-4"
echo ""
echo "Batch size: 32 per GPU × 12 grad_accum × 4 GPUs = 1536 (1.6M tokens/iter)"
echo "Total training: 66B tokens (100 tokens/param, ~12.6 days @ 6.5 sec/iter)"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_large_grassmann_hybridgate_r256.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r256/ckpt.pt"
