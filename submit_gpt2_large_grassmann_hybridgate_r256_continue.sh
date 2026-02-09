#!/bin/bash
#SBATCH --job-name=gpt2large_hybgate_r256_cont
#SBATCH --output=logs/%j_gpt2large_hybgate_r256_cont.out
#SBATCH --error=logs/%j_gpt2large_hybgate_r256_cont.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs for optimal performance
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# CONTINUATION JOB for Job 608061
# Resumes training from checkpoint after 12-hour time limit
# Optimal "Wide-Deep" Architecture: 36 layers, r=256, hybrid gating

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Continuing from Job 608061"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Navigate to working directory
cd /home/junyuren/nanoGPT

echo "=== Configuration ==="
echo "Resuming from checkpoint: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r256/ckpt.pt"
echo "Architecture: 36 layers, d=1024, 16 heads (~660M params)"
echo "Grassmann: c_attn (3 blocks) + mlp.c_fc (4 blocks)"
echo "Rank: r=256 (25% of width, STRONGER constraint)"
echo "Training target: 168k iters (100 tokens/param)"
echo "Hybrid gating: 48 head + 4 block = 52 gates/layer"
echo ""

# Run multi-GPU training with PyTorch DDP - RESUME from checkpoint
echo "=== Resuming Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_large_grassmann_hybridgate_r256.py \
    --init_from=resume

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r256/ckpt.pt"
