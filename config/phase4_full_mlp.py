# Phase 4: Full MLP with Skip-Compatible Projection
# Coverage: ~62% of parameters (same as Phase 3, but mlp.c_proj now Grassmann)
# Test if mlp.c_proj needs manifold constraint despite skip

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-phase4-full-mlp'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'phase4-full-mlp'

# Dataset
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Baby GPT architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Match baseline (nanoGPT standard for Shakespeare)
bias = False

# AdamW for non-Grassmann params
learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 1e-1  # Match baseline
beta1 = 0.9
beta2 = 0.99  # Match baseline (standard for small token count)
grad_clip = 1.0

decay_lr = True
warmup_iters = 100

# Grassmann config
use_grassmann = True
grassmann_phase = 'phase4'

# Legacy params
grass_lr = 1e-3  # Note: mlp.c_proj uses 1e-3 (skip-compatible), others use 8e-3
grass_a = 0.0  # mlp.c_proj uses G_{0,-1,r}
grass_b = -1.0
grass_rank = 192
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
