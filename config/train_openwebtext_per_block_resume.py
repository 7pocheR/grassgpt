# Resume OpenWebText per_block from checkpoint
# Continuing from step 12000

# I/O
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-per-block'
eval_interval = 1000
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

# CRITICAL: Resume from checkpoint
init_from = 'resume'

wandb_log = False
wandb_project = 'openwebtext'
wandb_run_name = 'grassmann-6L-384d-per-block-resume'

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

# Grassmann config - per_block
use_grassmann = True
grassmann_phase = 'phase2.5'

# Per-block configuration
grass_rank_c_attn = [28, 24, 14, 20, 25, 35]
grass_scale_c_attn = [4.8777, 3.3336, 5.8826, 5.2249, 4.7100, 3.7471]

grass_rank_mlp_fc = [11, 8, 9, 10, 11, 12]
grass_scale_mlp_fc = [8.7494, 10.5926, 10.7750, 10.7579, 10.3549, 9.5935]

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
