# OpenWebText Phase 2.5 - LR 2× (VERY CONSERVATIVE)
# Minimal boost - tests if even 2× helps

# I/O
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-lr2x'
eval_interval = 1000
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'openwebtext'
wandb_run_name = 'grassmann-6L-384d-lr2x'

# Data
dataset = 'openwebtext'
data_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
gradient_accumulation_steps = 40
batch_size = 64
block_size = 1024

# Model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# AdamW optimizer
learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000

# Grassmann config - stable_rank_x2 with lr2x
use_grassmann = True
grassmann_phase = 'phase2.5'

grass_rank_c_attn = 48
grass_rank_mlp_fc = 20
grass_scale_c_attn = 4.6293
grass_scale_mlp_fc = 10.1372

grass_rank = 48
grass_lr = 2e-3  # 2× (very conservative)
grass_a = 1.0
grass_b = 0.0
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
