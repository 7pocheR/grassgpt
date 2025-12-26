#!/bin/bash
#SBATCH --job-name=phase3.2_qkv_mlp_noskip
#SBATCH --output=logs/%j_phase3.2.out
#SBATCH --error=logs/%j_phase3.2.err
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training
cd /home/junyuren/nanoGPT
python -u train.py config/phase3.2_qkv_mlp_noskip.py
