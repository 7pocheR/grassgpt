#!/bin/bash
#SBATCH --job-name=gpt2m_hybgate_r384_cont
#SBATCH --output=logs/%j_gpt2m_hybgate_r384_cont.out
#SBATCH --error=logs/%j_gpt2m_hybgate_r384_cont.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs for optimal performance
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# CONTINUATION JOB for Job 608059
# Resumes training from checkpoint after 12-hour time limit

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Continuing from Job 608059"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Navigate to working directory
cd /home/junyuren/nanoGPT

echo "=== Configuration ==="
echo "Resuming from checkpoint: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r384/ckpt.pt"
echo "Architecture: GPT-2 Medium (d=1024, 24 layers, 16 heads)"
echo "Grassmann: c_attn + mlp.c_fc (non-skip-facing, 58% coverage)"
echo "Rank: r=384 (37.5% of width)"
echo "Hybrid gating: 48 head + 4 block = 52 gates/layer"
echo ""

# Run multi-GPU training with PyTorch DDP - RESUME from checkpoint
echo "=== Resuming Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_hybridgate_r384.py \
    --init_from=resume

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r384/ckpt.pt"
