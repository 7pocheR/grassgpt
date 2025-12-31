#!/bin/bash
#SBATCH --job-name=gpt2m_hybgate_r384
#SBATCH --output=logs/%j_gpt2m_hybgate_r384.out
#SBATCH --error=logs/%j_gpt2m_hybgate_r384.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs for optimal performance
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# Grassmann + Hybrid Gating at d=1024, r=384 (37.5% rank ratio)
# Key innovation: Head-level gating (c_attn) + Block-level gating (mlp.c_fc)
# Parameter efficiency: 1.28M gate params (137× reduction from 176M elementwise)

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
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r384

echo "=== Configuration ==="
echo "Architecture: GPT-2 Medium (d=1024, 24 layers, 16 heads)"
echo "Grassmann: c_attn + mlp.c_fc (non-skip-facing, 58% coverage)"
echo "Rank: r=384 (37.5% of width, 3×128 GPU-aligned)"
echo "Hybrid gating architecture:"
echo "  - c_attn: 48 head-level gates (16 heads × 3 QKV)"
echo "  - mlp.c_fc: 4 block-level gates (4 MLP blocks)"
echo "Total gate params: 1.28M (52 gates/layer, 137× reduction)"
echo "Grassmann LR: 2e-3, Gate LR: 1.2e-3, Embed LR: 3e-4"
echo "Scaling: x=10 uniform → sigmoid gating → input-dependent range"
echo "Batch size: 32 per GPU × 12 grad_accum × 4 GPUs = 1536 (1.6M tokens)"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_hybridgate_r384.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r384/ckpt.pt"
