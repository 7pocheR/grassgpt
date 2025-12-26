# Phase 0 Corrected: Only attn.c_proj (8% coverage)
# CORRECTED operator norms from OpenWebText baseline analysis
# Moderate rank (85% variance) instead of defaults

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-phase0-corrected'
eval_interval = 1000
eval_iters = 200
log_interval = 100
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'openwebtext-phase0-corrected'

# Dataset - OpenWebText
dataset = 'openwebtext'
data_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
gradient_accumulation_steps = 40  # Simulate batch size of ~2.6M tokens
batch_size = 64
block_size = 1024  # OpenWebText uses 1024 context

# Baby GPT architecture - SAME as Shakespeare for fair comparison
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Same as baseline
bias = False

# AdamW for non-Grassmann params
learning_rate = 1e-3  # AdamW base LR
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4
weight_decay = 1e-1  # Match baseline
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000  # Longer warmup for larger dataset

# Compile settings
compile = True

# Grassmann Phase 0: Only attn.c_proj
use_grassmann = True
grassmann_phase = 'phase0'

# CORRECTED operator norm for attn.c_proj (from OpenWebText baseline analysis)
# NOTE: Phase 0 uses attn.c_proj, NOT c_attn!
grass_scale_c_proj = 2.66  # OpenWebText measured (not Shakespeare)

# Moderate rank (85% variance for attn.c_proj)
# OpenWebText attn.c_proj: rank@50%=37, rank@85%=118, rank@95%=179, rank@99%=246
grass_rank_c_proj = 118  # 85% variance (was using defaults before)

# Conservative LR (skip-compatible parametrization)
grass_lr = 1e-3  # Same as AdamW (attn.c_proj faces skip connection)
grass_a = 0.0    # Skip-compatible
grass_b = -1.0   # Skip-compatible

# Dual ascent parameters
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# Dropout for Grassmann components
grassmann_dropout = None  # Use automatic dropout/2 = 0.1

# System
dtype = 'bfloat16'
