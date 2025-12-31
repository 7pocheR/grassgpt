#!/bin/bash
#SBATCH --job-name=gating_d384
#SBATCH --output=logs/%j_gating_d384.out
#SBATCH --error=logs/%j_gating_d384.err
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Phase 1: Validate gating mechanism at d=384
# Expected duration: ~3 hours for 10k iters on H100

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training
cd /home/junyuren/nanoGPT
python train.py config/train_openwebtext_gating_d384.py

echo "Training complete. Checkpoint saved to /net/scratch2/junyuren/nanoGPT-manifold/out-owt-gating-d384/ckpt.pt"
echo "Next step: Run analyze_gate_stats.py to measure gate statistics"
