# Phase 2.5 with Per-Block Rank and Scale Matching
# Uses exact measured values for each block from 10-seed baseline analysis

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-per-block'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'phase2.5-per-block'

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

# Grassmann config - PER-BLOCK EXACT MATCHING
use_grassmann = True
grassmann_phase = 'phase2.5'

# Per-block configuration (6 values, one per transformer block)
# From logs/561374_compute_config.out - exact stable ranks and operator norms

# c_attn: blocks 0-5
grass_rank_c_attn = [28, 24, 14, 20, 25, 35]
grass_scale_c_attn = [4.8777, 3.3336, 5.8826, 5.2249, 4.7100, 3.7471]

# mlp.c_fc: blocks 0-5
grass_rank_mlp_fc = [11, 8, 9, 10, 11, 12]
grass_scale_mlp_fc = [8.7494, 10.5926, 10.7750, 10.7579, 10.3549, 9.5935]

# Legacy (not used when lists provided)
grass_rank = 24
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
