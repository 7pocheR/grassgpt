#!/bin/bash
#SBATCH --job-name=gpt2m_fullblock_r384
#SBATCH --output=logs/%j_gpt2m_fullblock_r384.out
#SBATCH --error=logs/%j_gpt2m_fullblock_r384.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs for optimal performance
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# Grassmann + Full Block Decomposition at d=1024, r=384
# Full 16×16 grid decomposition for c_attn (768 blocks total for Q/K/V)
# Key innovation: True head independence via 64×64 block grid

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
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-fullblock-r384

echo "=== Configuration ==="
echo "Architecture: GPT-2 Medium (d=1024, 24 layers, 16 heads)"
echo "Grassmann: c_attn (full block decomp) + mlp.c_fc (4 blocks)"
echo ""
echo "Full Block Decomposition:"
echo "  - c_attn: 16×16 grid = 256 blocks per projection (Q/K/V)"
echo "  - Total c_attn blocks: 256 × 3 = 768 blocks per layer"
echo "  - Block size: 64×64, rank=32 (50% of block size)"
echo "  - Gating: per_head (16 gates per projection, 48 total)"
echo ""
echo "MLP (standard block decomp):"
echo "  - mlp.c_fc: 4 blocks, rank=384 (37.5% of width)"
echo "  - Block-level gating: 4 gates"
echo ""
echo "Total per layer:"
echo "  - Grassmann blocks: 768 (c_attn) + 4 (mlp.c_fc) = 772 blocks"
echo "  - Gates: 48 (c_attn) + 4 (mlp.c_fc) = 52 gates"
echo ""
echo "Grassmann LR: 2e-3, Gate LR: 1.2e-3, Embed LR: 3e-4"
echo "Scaling: x=10 uniform → gating → input-dependent range"
echo "Batch size: 32 per GPU × 12 grad_accum × 4 GPUs = 1536 (1.6M tokens)"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_fullblock_r384.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-fullblock-r384/ckpt.pt"
