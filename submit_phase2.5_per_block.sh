#!/bin/bash
#SBATCH --job-name=phase2.5_per_block
#SBATCH --output=logs/%j_phase2.5_per_block.out
#SBATCH --error=logs/%j_phase2.5_per_block.err
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training
cd /home/junyuren/nanoGPT
python -u train.py config/phase2.5_per_block.py
