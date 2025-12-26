#!/bin/bash
#SBATCH --job-name=analyze_variance
#SBATCH --output=logs/%j_analyze_variance.out
#SBATCH --error=logs/%j_analyze_variance.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run analysis
cd /home/junyuren/nanoGPT
python3 -u analyze_norm_variance.py
