# Phase 1: Test gating mechanism at d=384 on OpenWebText
# Goal: Measure gate statistics (mean ≈ 0.116 expected) before Phase 3 amplification

# Baby GPT architecture (same as previous d=384 experiments)
n_layer = 6
n_embd = 384
n_head = 6
block_size = 1024

# Gating configuration (Qwen NeurIPS 2025 paper)
use_gating = True
gate_type = 'headwise'  # 16K params (384 × 6 × 6 layers) vs 1M elementwise

# Training hyperparameters
learning_rate = 1e-3  # Standard for Baby GPT
max_iters = 10000  # Quick validation run
batch_size = 64
gradient_accumulation_steps = 1

# Regularization
dropout = 0.1
weight_decay = 1e-1

# Evaluation
eval_interval = 500  # Log gate stats every 500 iters
eval_iters = 200
log_interval = 10

# Dataset
dataset = 'openwebtext'
always_save_checkpoint = True

# Output directory - CRITICAL: Use scratch, NOT home!
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-owt-gating-d384'

# Compile for speed
compile = True

# Expected results:
# - Gate mean: 0.10-0.20 (paper reports 0.116 for 15B model)
# - Sparsity: >70% of values <0.1
# - Small PPL improvement over baseline (~0.05-0.1)
