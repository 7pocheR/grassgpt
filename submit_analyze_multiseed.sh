#!/bin/bash
#SBATCH --job-name=analyze_multiseed
#SBATCH --output=logs/%j_analyze_multiseed.out
#SBATCH --error=logs/%j_analyze_multiseed.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run analysis
cd /home/junyuren/nanoGPT
python3 -u analyze_baseline_norms_multiseed.py \
  --checkpoint-pattern '/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline-seed*/ckpt.pt' \
  --output baseline_weight_analysis_multiseed.md
