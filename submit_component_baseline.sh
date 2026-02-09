#!/bin/bash
#SBATCH --job-name=comp_base
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_comp_baseline.out
#SBATCH --error=logs/%j_comp_baseline.err

# Component Isolation Baseline: Pure AdamW (no Grassmann)

# Activate environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training (4 GPUs DDP)
torchrun --standalone --nproc_per_node=4 train.py \
    config/train_gpt2_medium_baseline_component_test.py
