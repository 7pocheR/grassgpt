# Phase 2 Corrected: mlp.c_fc + attn.c_proj (34% coverage)
# CRITICAL TEST: Does corrected mlp.c_fc operator norm fix divergence?
# Previous experiments over-scaled mlp.c_fc by 35% (used 10.14, actual 7.52)

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-phase2-corrected'
eval_interval = 1000
eval_iters = 200
log_interval = 100
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'openwebtext-phase2-corrected'

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

# Grassmann Phase 2: mlp.c_fc + attn.c_proj
use_grassmann = True
grassmann_phase = 'phase2'

# CORRECTED operator norms - CRITICAL FIX for mlp.c_fc!
grass_scale_mlp_fc = 7.52  # OpenWebText measured (was 10.14, 35% over-scaling!)
grass_scale_c_proj = 2.66  # attn.c_proj

# Moderate ranks (85% variance)
# mlp.c_fc: rank@50%=58, rank@85%=187, rank@95%=271, rank@99%=338
# attn.c_proj: rank@50%=37, rank@85%=118, rank@95%=179, rank@99%=246
grass_rank_mlp_fc = 187  # 85% variance (was 20, severely under-ranked!)
grass_rank_c_proj = 118  # 85% variance

# Mixed LR strategy (no-skip: 8×, skip: 1×)
grass_lr = 8e-3  # For mlp.c_fc (no direct skip)
grass_lr_c_proj = 1e-3  # For attn.c_proj (faces skip connection)

# No-skip parametrization for mlp.c_fc
grass_a = 1.0    # Identity-centered (no skip)
grass_b = 0.0    # Identity-centered (no skip)

# Dual ascent parameters
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# Dropout for Grassmann components
grassmann_dropout = None  # Use automatic dropout/2 = 0.1

# System
dtype = 'bfloat16'
