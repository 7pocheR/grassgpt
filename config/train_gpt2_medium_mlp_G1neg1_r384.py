# MLP-only Grassmann with G_{1,-1,384}
# Testing full-rank Grassmann manifold (eigenvalues {1, -1} instead of {1, 0})
# HYPOTHESIS: Full rank → no information loss → better performance

# GPT-2 Medium architecture
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.1
bias = False

# Enable Grassmann ONLY for c_fc (MLP expansion)
use_grassmann = True
use_grassmann_c_attn = False  # DISABLE for c_attn (use AdamW instead)
use_grassmann_c_fc = True     # Enable for c_fc
use_full_block_decomp = False

# Grassmann configuration: G_{1,-1,384}
# KEY DIFFERENCE: b=-1.0 instead of b=0.0
# • Eigenvalues: {1, -1} instead of {1, 0}
# • FULL RANK (no zero eigenvalues)
# • Nullspace reflected (negated) instead of zeroed
# • Condition number = 1 instead of ∞
grass_rank = 384  # 37.5% of n_embd
grass_scale = 10.0  # Uniform x=10 scaling
grass_a = 1.0
grass_b = -1.0  # FULL RANK parametrization
grass_lr = 2e-3

# Gating LRs
gate_lr = 1.2e-3  # 2× base LR
embed_lr = 3e-4   # 0.5× base LR

# Training hyperparameters
learning_rate = 6e-4
max_iters = 13000  # ~20B tokens
warmup_iters = 2000
lr_decay_iters = 13000
min_lr = 6e-5

# Batch configuration
batch_size = 32
gradient_accumulation_steps = 12

# Regularization
weight_decay = 1e-1

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
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-mlp-G1neg1-r384'

# Compile
compile = True

# System
dtype = 'bfloat16'
device = 'cuda'

# Phase identifier
grassmann_phase = 'mlp_G1neg1_r384'
