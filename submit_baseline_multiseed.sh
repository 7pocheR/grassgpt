#!/bin/bash
#SBATCH --job-name=baseline_multiseed
#SBATCH --output=logs/%j_baseline_seed%a.out
#SBATCH --error=logs/%j_baseline_seed%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training with different seed for each array task
cd /home/junyuren/nanoGPT

# Set seed based on array task ID
SEED=$((42 + SLURM_ARRAY_TASK_ID))

# Create seed-specific config
cat > /tmp/baseline_seed${SEED}.py << EOF
# Import base config
exec(open('config/train_baby_baseline_multiseed.py').read())

# Override with seed-specific values
seed_offset = ${SEED}
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline-seed${SEED}'
EOF

# Run training
python -u train.py /tmp/baseline_seed${SEED}.py
