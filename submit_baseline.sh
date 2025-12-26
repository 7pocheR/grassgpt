#!/bin/bash
#SBATCH --job-name=baseline_adamw
#SBATCH --output=logs/%j_baseline.out
#SBATCH --error=logs/%j_baseline.err
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training
cd /home/junyuren/nanoGPT
python -u train.py config/train_baby_baseline.py
