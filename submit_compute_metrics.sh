#!/bin/bash
#SBATCH --job-name=compute_metrics
#SBATCH --output=logs/%j_compute_metrics.out
#SBATCH --error=logs/%j_compute_metrics.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
cd /home/junyuren/nanoGPT

# Compute metrics for top configurations
python compute_metrics.py \
  --checkpoints \
    /net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-stable-rank-x2 \
    /net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-lr4x \
    /net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline \
    /net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-stable-rank \
    /net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-lr6x \
  --generate_samples \
  --num_samples=5 \
  --max_tokens=200 \
  --temperature=0.8
