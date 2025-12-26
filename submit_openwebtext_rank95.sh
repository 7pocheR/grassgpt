#!/bin/bash
#SBATCH --job-name=owt_rank95
#SBATCH --output=logs/%j_owt_rank95.out
#SBATCH --error=logs/%j_owt_rank95.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source /home/junyuren/.conda/envs/manifold_muon/bin/activate

cd /home/junyuren/nanoGPT
python -u train.py config/train_openwebtext_rank95.py
