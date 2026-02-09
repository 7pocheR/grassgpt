# TIER 1 EXPERIMENT 3: 24L, r=384, FULL block decomposition
# Question: Does fine-grained decomposition help 24L like it helps 36L?
# Hypothesis: If full block outperforms hybrid, block granularity matters

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False  # Grassmann requires no bias

# Enable Grassmann with FULL block decomposition
use_grassmann = True
use_full_block_decomp = True  # Use 16×16 grid instead of 3-block hybrid

# Full block decomposition configuration
full_block_size = 64  # 1024 / 64 = 16 blocks per dimension → 16×16 grid = 256 blocks
full_block_gating_mode = 'per_head'  # Options: 'per_head' (16 gates), 'per_block' (256 gates)

# Grassmann configuration
# Note: grass_rank is used for mlp.c_fc blocks, not for c_attn full blocks
# c_attn blocks use rank = full_block_size // 2 automatically (32 for 64×64 blocks)
grass_rank = 384  # For mlp.c_fc (4 blocks), matching other 24L experiments
grass_scale = 10.0  # Uniform x=10 scaling
grass_a = 1.0  # Identity-centered for non-skip layers
grass_b = 0.0
grass_lr = 2e-3  # Same as other 24L experiments

# Gating LRs
# For full block decomp: per_head mode has 16 gates/layer
gate_lr = 1.2e-3  # 2× learning_rate for gates
embed_lr = 3e-4   # 0.5× learning_rate for embeddings

# Training hyperparameters
learning_rate = 6e-4  # Base LR for AdamW components
max_iters = 100000  # Same as other 24L experiments
warmup_iters = 2000  # Standard warmup
lr_decay_iters = 100000  # Cosine decay to min_lr
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# REDUCED for full block decomposition (768 blocks/layer uses more memory)
# 16 × 24 × 4 GPUs = 1536 sequences × 1024 seq_len = 1,572,864 tokens (same total)
# grad_accum MUST be divisible by 4 GPUs (DDP requirement)
batch_size = 16  # Per GPU (reduced from 32 due to memory constraints)
gradient_accumulation_steps = 24  # Doubled to maintain same tokens/iter (393K tokens)

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
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-fullblock-r384'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'  # More stable than float16 for Grassmann
device = 'cuda'

# Phase identifier (for configure_optimizers_grassmann_phases)
grassmann_phase = 'phase3'  # c_attn + mlp.c_fc (non-skip-facing only)
