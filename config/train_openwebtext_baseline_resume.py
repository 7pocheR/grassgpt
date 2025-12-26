# Resume OpenWebText Baseline from checkpoint
# Continuing from step 13000

# I/O
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-baseline'
eval_interval = 1000
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

# CRITICAL: Resume from checkpoint
init_from = 'resume'

wandb_log = False
wandb_project = 'openwebtext'
wandb_run_name = 'baseline-6L-384d-resume'

# Data
dataset = 'openwebtext'
data_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
gradient_accumulation_steps = 40
batch_size = 64
block_size = 1024

# Model - SAME as Shakespeare: 6L x 384d (~10M params)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# AdamW optimizer - LR decay will continue from step 13000
learning_rate = 1e-3  # Max LR (for reference, decay continues)
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000  # Already past warmup

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True

# No Grassmann
use_grassmann = False
