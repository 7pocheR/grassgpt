# OpenWebText Baseline (AdamW only)
# Same architecture as Shakespeare: 6L x 384d (~10M params)
# Only difference: dataset (9B tokens vs 1.1M)

# I/O
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-baseline'
eval_interval = 1000
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'openwebtext'
wandb_run_name = 'baseline-6L-384d'

# Data
dataset = 'openwebtext'
data_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
gradient_accumulation_steps = 40  # Simulate batch size of ~2.6M tokens
batch_size = 64
block_size = 1024  # OpenWebText uses 1024 context (vs 256 for Shakespeare)

# Model - SAME as Shakespeare: 6L x 384d (~10M params)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Same as Shakespeare baseline
bias = False

# AdamW optimizer - Same as Shakespeare
learning_rate = 1e-3
max_iters = 100000  # ~100K iters for meaningful training on 9B tokens
lr_decay_iters = 100000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99  # Same as Shakespeare
weight_decay = 1e-1
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000  # Longer warmup for larger dataset

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True

# No Grassmann
use_grassmann = False
