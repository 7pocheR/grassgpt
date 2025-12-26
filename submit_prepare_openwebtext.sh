#!/bin/bash
#SBATCH --job-name=prepare_openwebtext
#SBATCH --output=logs/%j_prepare_openwebtext.out
#SBATCH --error=logs/%j_prepare_openwebtext.err
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# No GPU needed for tokenization (CPU + I/O bound)

source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# CRITICAL: Redirect HuggingFace cache to scratch2 to avoid filling home directory
export HF_HOME=/net/scratch2/junyuren/huggingface_cache
export HF_DATASETS_CACHE=/net/scratch2/junyuren/huggingface_cache/datasets
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE

cd /home/junyuren/nanoGPT/data/openwebtext
python -u prepare_v3.py
