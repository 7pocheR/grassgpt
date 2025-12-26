#!/bin/bash
#SBATCH --job-name=compute_config
#SBATCH --output=logs/%j_compute_config.out
#SBATCH --error=logs/%j_compute_config.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run analysis
cd /home/junyuren/nanoGPT
python3 -u compute_per_layer_grassmann_config.py
