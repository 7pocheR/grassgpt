# Baseline: AdamW + Gating at d=1024
# Control experiment to compare against Grassmann + Gating

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False  # Keep consistent with Grassmann version

# Disable Grassmann, enable Gating
use_grassmann = False
use_gating = True

# Gating configuration
gate_type = 'headwise'  # 16K params/layer (efficient)

# Training hyperparameters
learning_rate = 6e-4  # Standard AdamW LR
max_iters = 100000  # Match Grassmann run
warmup_iters = 2000  # Standard warmup
lr_decay_iters = 100000  # Cosine decay to min_lr
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# Effective batch = 64 × 12 × 4 GPUs = 3072 sequences × 1024 seq_len = 3,145,728 tokens
# grad_accum MUST be divisible by 4 GPUs (DDP requirement)
batch_size = 64  # Per GPU
gradient_accumulation_steps = 12  # Total 768 sequences per GPU, 786K tokens

# Regularization
weight_decay = 1e-1  # Standard for GPT-2

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

# CRITICAL: Save to scratch (home has strict disk quota!)
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-baseline-gate'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'  # More stable than float16
device = 'cuda'
