#!/bin/bash
#SBATCH --job-name=tier1_24L_r512_cont
#SBATCH --output=logs/%j_tier1_24L_r512_cont.out
#SBATCH --error=logs/%j_tier1_24L_r512_cont.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# CONTINUATION for TIER 1 EXPERIMENT 2: 24L, r=512, hybrid

echo "=== TIER 1 EXPERIMENT 2 (CONTINUATION): 24L, r=512, Hybrid ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
cd /home/junyuren/nanoGPT

echo "Resuming from checkpoint: /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r512/ckpt.pt"
echo ""

torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_hybridgate_r512.py \
    --init_from=resume

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
