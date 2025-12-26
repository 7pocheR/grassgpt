#!/bin/bash
#SBATCH --job-name=phase0.5_qkv_adamw
#SBATCH --output=logs/%j_phase0.5.out
#SBATCH --error=logs/%j_phase0.5.err
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training
cd /home/junyuren/nanoGPT
python -u train.py config/phase0.5_qkv_adamw.py
