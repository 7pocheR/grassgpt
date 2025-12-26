#!/bin/bash
#SBATCH --job-name=analyze_norms
#SBATCH --output=logs/%j_analyze_owt_norms.out
#SBATCH --error=logs/%j_analyze_owt_norms.err
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /home/junyuren/.conda/envs/manifold_muon/bin/activate

cd /home/junyuren/nanoGPT
python -u analyze_owt_checkpoint_norms.py
