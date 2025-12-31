# Grassmann + Hybrid Gating at d=1024, r=384 (37.5% rank ratio)
# Hybrid gating architecture: Head-level (48) + Block-level (4) = 52 gates/layer
# Total gate params: 1.28M (137× reduction from 176M elementwise gating)

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False  # Grassmann requires no bias

# Enable Grassmann with hybrid gating
use_grassmann = True

# Grassmann configuration
# Non-skip-facing layers (c_attn, mlp.c_fc): G_{1.0, 0.0, r}
grass_rank = 384  # 37.5% of n_embd (3×128, GPU-aligned)
grass_scale = 10.0  # Uniform x=10 scaling (base range for sigmoid gates)
grass_a = 1.0  # Identity-centered for non-skip layers
grass_b = 0.0
grass_lr = 2e-3  # Conservative LR (matches previous r=384 run)

# Gating architecture:
# - c_attn: 48 head-level gates (16 heads × 3 QKV), linear + sigmoid
# - mlp.c_fc: 4 block-level gates (4 MLP blocks), linear + sigmoid
# Gate LR should be 2× base for adaptive learning
gate_lr = 1.2e-3  # 2× learning_rate for gates
embed_lr = 3e-4   # 0.5× learning_rate for embeddings

# Training hyperparameters
learning_rate = 6e-4  # Base LR for AdamW components
max_iters = 100000  # Initial test (can extend to 600k later)
warmup_iters = 2000  # Standard warmup
lr_decay_iters = 100000  # Cosine decay to min_lr
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# Matches previous r=384 run exactly for fair comparison
# Effective batch = 32 × 12 × 4 GPUs = 1536 sequences × 1024 seq_len = 1,572,864 tokens
# grad_accum MUST be divisible by 4 GPUs (DDP requirement)
batch_size = 32  # Per GPU
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
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r384'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'  # More stable than float16 for Grassmann
device = 'cuda'

# Phase identifier (for configure_optimizers_grassmann_phases)
grassmann_phase = 'phase3'  # c_attn + mlp.c_fc (non-skip-facing only)
