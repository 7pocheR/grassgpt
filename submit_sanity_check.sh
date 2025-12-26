#!/bin/bash
#SBATCH --job-name=gpt_sanity
#SBATCH --output=logs/sanity_%j.out
#SBATCH --error=logs/sanity_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run sanity check
cd /home/junyuren/nanoGPT
python -u train.py config/test_grassmann_sanity.py

echo "Sanity check complete!"
