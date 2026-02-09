# COMPONENT ISOLATION BASELINE: Pure AdamW (no Grassmann)
# Control group for component isolation experiments

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1  # Unified with Grassmann experiments
bias = False  # Keep consistent with Grassmann version

# Disable Grassmann entirely
use_grassmann = False

# Training hyperparameters
learning_rate = 6e-4  # Standard AdamW LR
max_iters = 13000  # ~20B tokens @ 1.57M tokens/iter (match component experiments)
warmup_iters = 2000  # Unified
lr_decay_iters = 13000
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# Match Grassmann experiments: 32 × 12 × 4 GPUs = 1,572,864 tokens/iter
batch_size = 32  # Per GPU
gradient_accumulation_steps = 12  # MUST be divisible by 4 GPUs

# Regularization
weight_decay = 1e-1  # Unified

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

# Output directory
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-baseline-component-test'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'
device = 'cuda'
