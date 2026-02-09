#!/bin/bash
#SBATCH --job-name=sm_qkv256
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --exclude=l001
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_small_qkv_r256.out
#SBATCH --error=logs/%j_small_qkv_r256.err

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_small_qkv_grass_r256.py
