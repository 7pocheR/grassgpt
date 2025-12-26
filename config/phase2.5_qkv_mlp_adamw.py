# Phase 2.5: c_attn + mlp.c_fc (derived from phase3, attn.c_proj uses AdamW)
# Coverage: ~58% of parameters (c_attn + mlp.c_fc Grassmann, attn.c_proj AdamW)
# Test if combined benefits persist while keeping attn.c_proj standard

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-qkv-mlp-adamw'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'phase2.5-qkv-mlp-adamw'

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
grassmann_phase = 'phase2.5'  # .5 = skip attn.c_proj (use AdamW)

# Legacy params
grass_lr = 8e-3
grass_a = 1.0
grass_b = 0.0
grass_rank = 192
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
