#!/bin/bash
#SBATCH --job-name=gpt2m_grass_r384
#SBATCH --output=logs/%j_gpt2m_grass_r384.out
#SBATCH --error=logs/%j_gpt2m_grass_r384.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs for optimal performance
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# Grassmann + Per-Layer Gating at d=1024, r=384 (37.5% rank ratio)
# Key innovation: Input-dependent gating after EACH Grassmann layer
# Grassmann provides direction (||W||=1), gating provides adaptive magnitude

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
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-perlayergate-r384

echo "=== Configuration ==="
echo "Architecture: GPT-2 Medium (d=1024, 24 layers, 16 heads)"
echo "Grassmann: c_attn + mlp.c_fc (non-skip-facing, 58% coverage)"
echo "Rank: r=384 (37.5% of width, 3×128 GPU-aligned)"
echo "Per-layer gating: c_attn_gate (3n params) + c_fc_gate (4n params)"
echo "Total gate params: 176M (50% increase)"
echo "Grassmann LR: 2e-3"
echo "Scaling: x=10 uniform → sigmoid gating → [0, 10] range"
echo "Batch size: 32 per GPU × 12 grad_accum × 4 GPUs = 1536 (1.6M tokens)"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_per_layer_gate_r384.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-perlayergate-r384/ckpt.pt"
