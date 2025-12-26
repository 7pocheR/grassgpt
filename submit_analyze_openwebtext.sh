#!/bin/bash
#SBATCH --job-name=analyze_owt
#SBATCH --output=logs/%j_analyze_owt.out
#SBATCH --error=logs/%j_analyze_owt.err
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /home/junyuren/.conda/envs/manifold_muon/bin/activate

cd /home/junyuren/nanoGPT
python -u analyze_openwebtext_baseline_norms.py
