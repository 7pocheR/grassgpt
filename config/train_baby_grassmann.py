# Baby GPT with Grassmann Muon optimizer for attention c_proj weights
# Based on CIFAR-10 breakthrough: G_{0.0, -1.0, r} works with skip connections
# Key finding: lr=5e-4 (8× lower than no-skip), rank=160 optimal (62.5% of width)

out_dir = 'out-baby-grassmann'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False  # set to True if using wandb
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'baby-grassmann-skip'

# dataset
dataset = 'shakespeare_char'  # use 'wikitext-2' for larger experiments
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Baby GPT model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0  # 0 for pretraining
bias = False

# AdamW optimizer (for non-manifold parameters)
learning_rate = 1e-3  # for AdamW parameters
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule (cosine with warmup - proven to work in CIFAR-10)
decay_lr = True
warmup_iters = 100

# Grassmann Muon optimizer
use_grassmann = True
grass_lr = 5e-4  # 8× lower than no-skip (4e-3) based on CIFAR-10 findings
grass_a = 0.0    # First eigenvalue (0.0 for skip connections - CRITICAL!)
grass_b = -1.0   # Second eigenvalue (-1.0 for skip connections - CRITICAL!)
grass_rank = 192 # 50% of n_embd=384 (n/2 for simplicity, CIFAR-10 optimal was 31%)
grass_alpha = 0.01  # Dual ascent step size
grass_steps = 10    # Max dual iterations (from CIFAR-10 experiments)
grass_tol = 1e-6    # Convergence tolerance

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True

# NOTE: This config applies Grassmann Muon ONLY to square attention c_proj weights
# Other weights (c_attn, MLP, embeddings) use AdamW
# To apply to Q/K/V and MLP, we would need to modify the architecture
