# Optimal "Wide-Deep" Architecture: 36 layers, r=256, hybrid gating
# Based on OPTIMAL_GRASSMANN_EXPERIMENT.md design
# Key improvements: Deeper (36L), stronger constraint (r=256), proper training (100 tokens/param)

# GPT-2 Large-ish architecture (36 layers instead of GPT-2 Large's 36 layers @ d=1280)
n_layer = 36  # 50% more depth than current (24 → 36)
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.05  # Reduced from 0.1 (manifold provides regularization)
bias = False  # Grassmann requires no bias

# Enable Grassmann with hybrid gating
use_grassmann = True
use_full_block_decomp = False  # Hybrid gating (not full block decomp)

# Grassmann configuration
# Non-skip-facing layers (c_attn, mlp.c_fc): G_{1.0, 0.0, r}
grass_rank = 256  # STRONGER constraint: 25% of n_embd (vs 37.5% before)
grass_scale = 10.0  # Uniform x=10 scaling
grass_a = 1.0  # Identity-centered for non-skip layers
grass_b = 0.0
grass_lr = 3e-3  # Conservative for deeper network (was 2e-3 for 24L)

# Hybrid gating architecture:
# - c_attn: 48 head-level gates (16 heads × 3 QKV), linear + sigmoid
# - mlp.c_fc: 4 block-level gates (4 MLP blocks), linear + sigmoid
# Gate LR should be 2.5× base for adaptive learning (was 2× before)
gate_lr = 1.5e-3  # 2.5× learning_rate for gates
embed_lr = 3e-4   # 0.5× learning_rate for embeddings

# Training hyperparameters - TARGET: 100 tokens/param
# Total params: ~660M → 66B tokens → 168k iters @ 393K tokens/iter
learning_rate = 6e-4  # Base LR for AdamW components
max_iters = 168000  # 100 tokens/param (vs 100k before, which was only 3.8 tokens/param!)
warmup_iters = 6000  # Longer for deeper network (vs 2000 before)
lr_decay_iters = 150000  # CRITICAL: < max_iters for fine-tuning region!
min_lr = 6e-5  # 0.1 × learning_rate

# Multi-GPU batch configuration
# Same as before: 32 × 12 × 4 GPUs = 1536 sequences × 1024 seq_len = 1,572,864 tokens
# grad_accum MUST be divisible by 4 GPUs (DDP requirement)
batch_size = 32  # Per GPU
gradient_accumulation_steps = 12  # Total 384 sequences per GPU, 393K tokens/iter

# Regularization
weight_decay = 1e-1  # Standard for GPT-2 (applies to AdamW params only)

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
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r256'

# Compile for performance
compile = True

# System
dtype = 'bfloat16'  # More stable than float16 for Grassmann
device = 'cuda'

# Phase identifier (for configure_optimizers_grassmann_phases)
grassmann_phase = 'phase3'  # c_attn + mlp.c_fc (non-skip-facing only)
