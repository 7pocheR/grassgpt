# Phase 3: Grassmann + Gating at d=1024
# Test skip-avoidance strategy with uniform x=10 scaling

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False  # Grassmann requires no bias

# Enable both Grassmann and Gating
use_grassmann = True
use_gating = True

# Grassmann configuration
# Non-skip-facing layers (c_attn, mlp.c_fc): G_{1.0, 0.0, r}
grass_rank = 512  # 50% of n_embd
grass_scale = 10.0  # Uniform x=10 scaling
grass_a = 1.0  # Identity-centered for non-skip layers
grass_b = 0.0
grass_lr = 8e-3  # 8× boost (proven on CIFAR-10)

# Gating configuration
gate_type = 'headwise'  # 16K params/layer (efficient)

# Training hyperparameters
learning_rate = 6e-4  # Base LR for AdamW components
max_iters = 100000  # Initial test (can extend to 600k later)
warmup_iters = 2000  # Standard warmup
lr_decay_iters = 100000  # Cosine decay to min_lr
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# Effective batch = 64 × 12 × 4 GPUs = 3072 sequences × 1024 seq_len = 3,145,728 tokens
# This matches successful d=384 runs (64% MFU) - previous 15×4 gave only 2.57% MFU!
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
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-gate-x10'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'  # More stable than float16 for Grassmann
device = 'cuda'

# Phase identifier (for configure_optimizers_grassmann_phases)
grassmann_phase = 'phase3'  # c_attn + mlp.c_fc (non-skip-facing only)
