#!/bin/bash
#SBATCH --job-name=tier1_24L_r512
#SBATCH --output=logs/%j_tier1_24L_r512.out
#SBATCH --error=logs/%j_tier1_24L_r512.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# TIER 1 EXPERIMENT 2: 24L, r=512, hybrid gating
# Question: Can 24L break the 3.1 ceiling with higher rank?
# Hypothesis: If ceiling breaks, rank was limiting expressivity

echo "=== TIER 1 EXPERIMENT 2: 24L, r=512, Hybrid Gating ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Navigate to working directory
cd /home/junyuren/nanoGPT

# Ensure output directory exists
mkdir -p /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r512

echo "=== Configuration ==="
echo "Architecture: 24 layers, d=1024, 16 heads (~530M params)"
echo "Grassmann rank: r=512 (50% of width) ← INCREASED from r=384"
echo "Gating: Hybrid (48 head-level + 4 block-level = 52 gates/layer)"
echo "Training: 100k iters"
echo ""
echo "KEY COMPARISON:"
echo "  Previous 24L r=384: Plateau at val ~3.1 (expressivity ceiling?)"
echo "  This 24L r=512: Should break ceiling if rank was the bottleneck"
echo ""

# Run multi-GPU training with PyTorch DDP
echo "=== Starting Training ==="
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_hybridgate_r512.py

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoint saved to: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r512/ckpt.pt"
