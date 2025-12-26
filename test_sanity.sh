#!/bin/bash
# Quick sanity check: 10 iterations on CPU to verify no crashes
# Run this BEFORE submitting SLURM jobs!

echo "=== Sanity Check: 10 iterations per phase ==="
echo ""

cd /home/junyuren/nanoGPT

# Prepare data if needed
if [ ! -f data/shakespeare_char/train.bin ]; then
    echo "Preparing Shakespeare data..."
    python data/shakespeare_char/prepare.py
fi

echo ""
echo "--- Phase 1: QKV ---"
python train.py config/phase1_qkv.py \
    --max_iters=10 \
    --log_interval=1 \
    --device=cpu \
    --compile=False \
    --eval_interval=999999

echo ""
echo "--- Phase 2: MLP.c_fc ---"
python train.py config/phase2_mlp_fc.py \
    --max_iters=10 \
    --log_interval=1 \
    --device=cpu \
    --compile=False \
    --eval_interval=999999

echo ""
echo "--- Phase 3: QKV + MLP.c_fc ---"
python train.py config/phase3_qkv_mlp.py \
    --max_iters=10 \
    --log_interval=1 \
    --device=cpu \
    --compile=False \
    --eval_interval=999999

echo ""
echo "--- Phase 4: Full MLP ---"
python train.py config/phase4_full_mlp.py \
    --max_iters=10 \
    --log_interval=1 \
    --device=cpu \
    --compile=False \
    --eval_interval=999999

echo ""
echo "--- Phase 5: No Skip ---"
python train.py config/phase5_no_skip.py \
    --max_iters=10 \
    --log_interval=1 \
    --device=cpu \
    --compile=False \
    --eval_interval=999999

echo ""
echo "=== Sanity Check Complete ==="
echo "If all phases ran without crashes, ready for SLURM submission!"
