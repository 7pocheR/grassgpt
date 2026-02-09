#!/bin/bash
#SBATCH --job-name=gated_b32
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_baseline_gated_batch32.out
#SBATCH --error=logs/%j_baseline_gated_batch32.err

# Gated AdamW baseline with batch=32 (fair comparison with Grassmann)

source /home/junyuren/.conda/envs/manifold_muon/bin/activate

torchrun --standalone --nproc_per_node=4 train.py \
    config/train_gpt2_medium_baseline_gated_batch32.py
