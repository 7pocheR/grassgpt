# Small-scale GPT: QKV-only Grassmann, r=256 (50% of width)
# Proof of concept: Does Grassmann help attention at small scale?
# ~50M params, 6 layers, d=512

# Architecture (small-scale)
n_layer = 6
n_head = 8
n_embd = 512
block_size = 1024
dropout = 0.1
bias = False

# Enable Grassmann ONLY for c_attn (QKV), NO gating
use_grassmann = True
use_grassmann_c_attn = True   # Grassmann for attention
use_grassmann_c_fc = False    # AdamW for MLP
use_full_block_decomp = False

# Grassmann configuration: G_{1,0,r=256}
grass_rank = 256  # 50% of n_embd
grass_scale = 10.0
grass_a = 1.0
grass_b = 0.0
grass_lr = 2e-3

# NO gating (simplest test)

# Embedding LR
embed_lr = 3e-4

# Training hyperparameters
learning_rate = 6e-4
max_iters = 10000
warmup_iters = 1000
lr_decay_iters = 10000
min_lr = 6e-5

# Batch configuration
batch_size = 32
gradient_accumulation_steps = 12

# Regularization
weight_decay = 1e-1

# Optimizer
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Evaluation
eval_interval = 500
eval_iters = 200
log_interval = 10

# Dataset
dataset = 'openwebtext'
data_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
always_save_checkpoint = True

# Output directory
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-small-qkv-grass-r256'

# Compile
compile = True

# System
dtype = 'bfloat16'
device = 'cuda'

# Phase identifier
grassmann_phase = 'small_qkv_r256'
