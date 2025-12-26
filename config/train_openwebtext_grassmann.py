# OpenWebText Phase 2.5 - Stable Rank × 2
# Same architecture as Shakespeare: 6L x 384d (~10M params)
# Same Grassmann config as Shakespeare winner (phase2.5_stable_rank_x2)
# Only difference: dataset (9B tokens vs 1.1M)

# I/O
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-grassmann'
eval_interval = 1000
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'openwebtext'
wandb_run_name = 'grassmann-6L-384d-stable-rank-x2'

# Data
dataset = 'openwebtext'
data_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
gradient_accumulation_steps = 40  # Simulate batch size of ~2.6M tokens
batch_size = 64
block_size = 1024  # OpenWebText uses 1024 context (vs 256 for Shakespeare)

# Model - SAME as Shakespeare: 6L x 384d (~10M params)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Same as Shakespeare baseline
bias = False

# AdamW optimizer - Same as Shakespeare
learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000

# Grassmann config - EXACT SAME as Shakespeare phase2.5_stable_rank_x2
use_grassmann = True
grassmann_phase = 'phase2.5'

# Rank matching strategy: stable_rank × 2 (from Shakespeare baseline analysis)
grass_rank_c_attn = 48      # 24 × 2
grass_rank_mlp_fc = 20      # 10 × 2
grass_scale_c_attn = 4.6293    # Operator norm from Shakespeare baseline
grass_scale_mlp_fc = 10.1372   # Operator norm from Shakespeare baseline

# Legacy unified rank
grass_rank = 48
grass_lr = 8e-3  # 8× boost (same as Shakespeare)
grass_a = 1.0    # No-skip parametrization (same as Shakespeare)
grass_b = 0.0
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
