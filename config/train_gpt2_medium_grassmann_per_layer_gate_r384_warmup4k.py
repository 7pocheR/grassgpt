# Grassmann + Per-Layer Gating at d=1024, r=384 (37.5% rank ratio)
# Addresses scaling rigidity: Grassmann provides direction (||W||=1), gating provides adaptive magnitude

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False  # Grassmann requires no bias

# Enable Grassmann (gating is now automatic per-layer)
use_grassmann = True

# Grassmann configuration
# Non-skip-facing layers (c_attn, mlp.c_fc): G_{1.0, 0.0, r}
grass_rank = 384  # 37.5% of n_embd (3×128, GPU-aligned)
grass_scale = 10.0  # Uniform x=10 scaling (base range for sigmoid gates)
grass_a = 1.0  # Identity-centered for non-skip layers
grass_b = 0.0
grass_lr = 2e-3  # Conservative LR (2e-3 performed same as 8e-3)

# Training hyperparameters
learning_rate = 6e-4  # Base LR for AdamW components
max_iters = 100000  # Initial test (can extend to 600k later)
warmup_iters = 2000  # Standard warmup
lr_decay_iters = 100000  # Cosine decay to min_lr
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# Reduced from baseline (64×12) due to 50% more params with gates (530M vs 354M)
# Effective batch = 32 × 12 × 4 GPUs = 1536 sequences × 1024 seq_len = 1,572,864 tokens
# grad_accum MUST be divisible by 4 GPUs (DDP requirement)
batch_size = 32  # Per GPU (logits gradient alone: 3.07 GiB @ batch=32 vs 4.61 GiB @ batch=48)
gradient_accumulation_steps = 12  # Total 384 sequences per GPU, 393K tokens

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
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-perlayergate-r384'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'  # More stable than float16 for Grassmann
device = 'cuda'

# Phase identifier (for configure_optimizers_grassmann_phases)
grassmann_phase = 'phase3'  # c_attn + mlp.c_fc (non-skip-facing only)
