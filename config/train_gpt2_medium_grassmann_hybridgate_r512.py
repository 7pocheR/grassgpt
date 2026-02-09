# TIER 1 EXPERIMENT 2: 24L, r=512, hybrid gating
# Question: Can 24L break the 3.1 ceiling with higher rank?
# Hypothesis: If ceiling breaks, rank was limiting expressivity

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False  # Grassmann requires no bias

# Enable Grassmann with hybrid gating
use_grassmann = True
use_full_block_decomp = False  # Hybrid gating (not full block decomp)

# Grassmann configuration
# INCREASED rank from 384 → 512 (50% of n_embd, very high rank)
grass_rank = 512  # CRITICAL: Higher rank to test if ceiling is due to rank constraint
grass_scale = 10.0  # Uniform x=10 scaling
grass_a = 1.0  # Identity-centered for non-skip layers
grass_b = 0.0
grass_lr = 2e-3  # Same as other 24L experiments

# Hybrid gating architecture:
# - c_attn: 48 head-level gates (16 heads × 3 QKV), linear + sigmoid
# - mlp.c_fc: 4 block-level gates (4 MLP blocks), linear + sigmoid
gate_lr = 1.2e-3  # 2× learning_rate for gates
embed_lr = 3e-4   # 0.5× learning_rate for embeddings

# Training hyperparameters
learning_rate = 6e-4  # Base LR for AdamW components
max_iters = 100000  # Same as other 24L experiments
warmup_iters = 2000  # Standard warmup
lr_decay_iters = 100000  # Cosine decay to min_lr
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# Same as other 24L: 32 × 12 × 4 GPUs = 1536 sequences × 1024 seq_len = 1,572,864 tokens
# grad_accum MUST be divisible by 4 GPUs (DDP requirement)
batch_size = 32  # Per GPU
gradient_accumulation_steps = 12  # Total 384 sequences per GPU, 393K tokens/iter

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
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r512'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'  # More stable than float16 for Grassmann
device = 'cuda'

# Phase identifier (for configure_optimizers_grassmann_phases)
grassmann_phase = 'phase3'  # c_attn + mlp.c_fc (non-skip-facing only)
