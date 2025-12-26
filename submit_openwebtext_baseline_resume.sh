#!/bin/bash
#SBATCH --job-name=owt_base_res
#SBATCH --output=logs/%j_owt_baseline_resume.out
#SBATCH --error=logs/%j_owt_baseline_resume.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source /home/junyuren/.conda/envs/manifold_muon/bin/activate

cd /home/junyuren/nanoGPT
python -u train.py config/train_openwebtext_baseline_resume.py
