# Phase 2.5 Variant 2: Match Effective Rank + Operator Norm
# c_attn: rank=96 (current/4), scale=4.6293
# mlp.c_fc: rank=48 (current/4), scale=10.1372
# Note: Intermediate regularization level

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-eff-rank'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'phase2.5-eff-rank'

# Dataset
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Baby GPT architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# AdamW for non-Grassmann params
learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 100

# Grassmann config - MATCH CURRENT/4 (intermediate)
use_grassmann = True
grassmann_phase = 'phase2.5'

# Rank matching strategy: current/4 = 192/4 = 48
grass_rank_c_attn = 96      # 192/2 for c_attn (higher variance)
grass_rank_mlp_fc = 48      # 192/4 for mlp.c_fc
grass_scale_c_attn = 4.6293    # Operator norm from baseline
grass_scale_mlp_fc = 10.1372   # Operator norm from baseline

# Legacy unified rank
grass_rank = 48
grass_lr = 8e-3
grass_a = 1.0
grass_b = 0.0
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
