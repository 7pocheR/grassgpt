#!/bin/bash
#SBATCH --job-name=phase2.5_lr4x
#SBATCH --output=logs/%j_phase2.5_lr4x.out
#SBATCH --error=logs/%j_phase2.5_lr4x.err
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
cd /home/junyuren/nanoGPT
python -u train.py config/phase2.5_lr4x.py
