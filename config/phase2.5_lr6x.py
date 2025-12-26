# Phase 2.5 LR Sweep: 6× multiplier (.006)
# Base: stable_rank_x2 (r=48,20, scale=4.63,10.14)

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-phase2.5-lr6x'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'phase2.5-lr6x'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

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

use_grassmann = True
grassmann_phase = 'phase2.5'

grass_rank_c_attn = 48
grass_rank_mlp_fc = 20
grass_scale_c_attn = 4.6293
grass_scale_mlp_fc = 10.1372

grass_rank = 48
grass_lr = .006  # 6× multiplier
grass_a = 1.0
grass_b = 0.0
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

device = 'cuda'
dtype = 'bfloat16'
compile = True
