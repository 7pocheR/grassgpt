#!/bin/bash
#SBATCH --job-name=verify_fixes
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

mkdir -p logs

source /home/junyuren/.conda/envs/manifold_muon/bin/activate

cd /home/junyuren/nanoGPT

echo "=== Test 1: Gradient zeroing fix (20 iterations, bfloat16) ==="
python -u train.py config/test_grassmann_sanity.py --max_iters=20

echo ""
echo "=== Test 2: Mixed precision fix (20 iterations, float16) ==="
python -u train.py config/test_float16.py

echo ""
echo "=== Verification complete ==="
