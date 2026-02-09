# Gated AdamW baseline with batch=32
# For fair comparison with Grassmann variants (same batch size)

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False

# Disable Grassmann, enable Gating
use_grassmann = False
use_gating = True
gate_type = 'headwise'  # Match Grassmann experiments

# Training hyperparameters
learning_rate = 6e-4
max_iters = 13000  # Match Grassmann runs (~20B tokens)
warmup_iters = 2000
lr_decay_iters = 13000
min_lr = 6e-5

# Batch configuration - MATCH GRASSMANN (batch=32)
# Tokens/iter = 32 × 12 × 1024 × 4 = 1,572,864
batch_size = 32
gradient_accumulation_steps = 12

# Regularization
weight_decay = 1e-1

# Optimizer
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Evaluation
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Dataset
dataset = 'openwebtext'
data_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
always_save_checkpoint = True

# Output
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-baseline-gated-batch32'

# Compile
compile = True

# System
dtype = 'bfloat16'
device = 'cuda'

# Phase identifier
grassmann_phase = 'baseline_gated_batch32'
