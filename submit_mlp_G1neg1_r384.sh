#!/bin/bash
#SBATCH --job-name=G1neg1_r384
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_mlp_G1neg1_r384.out
#SBATCH --error=logs/%j_mlp_G1neg1_r384.err

# MLP-only G_{1,-1,384} full-rank Grassmann

# Activate environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training (4 GPUs DDP)
torchrun --standalone --nproc_per_node=4 train.py \
    config/train_gpt2_medium_mlp_G1neg1_r384.py
