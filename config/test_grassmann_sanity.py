# Sanity check config: verify Grassmann Muon implementation works
# Run just 10 iterations to check for errors

out_dir = 'out-test-sanity'
eval_interval = 5
eval_iters = 2
log_interval = 1
always_save_checkpoint = False
eval_only = False

wandb_log = False

# dataset
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 8  # small batch for quick test
block_size = 128  # small context

# Small model for quick test
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
bias = False

# AdamW optimizer
learning_rate = 1e-3
max_iters = 10  # just 10 iterations
lr_decay_iters = 10
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
decay_lr = True
warmup_iters = 2

# Grassmann Muon optimizer
use_grassmann = True
grass_lr = 5e-4
grass_a = 0.0
grass_b = -1.0
grass_rank = 32  # 50% of n_embd=64 (n/2)
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# system
device = 'cuda'
dtype = 'bfloat16'
compile = False  # disable compile for faster startup in test
