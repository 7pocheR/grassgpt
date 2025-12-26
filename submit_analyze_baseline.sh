#!/bin/bash
#SBATCH --job-name=analyze_baseline
#SBATCH --output=logs/%j_analyze_baseline.out
#SBATCH --error=logs/%j_analyze_baseline.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run analysis
cd /home/junyuren/nanoGPT
python3 -u analyze_baseline_norms.py \
  --checkpoint /net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline/ckpt.pt \
  --output baseline_weight_analysis.md
