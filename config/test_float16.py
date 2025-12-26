# Test config for float16 mixed precision (validate bug fix #2)

out_dir = 'out-test-float16'
eval_interval = 5
eval_iters = 2
log_interval = 1
always_save_checkpoint = False

wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 128

# Small model
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
bias = False

# AdamW
learning_rate = 1e-3
max_iters = 20  # 20 iterations to test gradient accumulation fix
lr_decay_iters = 20
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2

# Grassmann
use_grassmann = True
grass_lr = 5e-4
grass_a = 0.0
grass_b = -1.0
grass_rank = 40
grass_alpha = 0.01
grass_steps = 10
grass_tol = 1e-6

# CRITICAL: Use float16 to test mixed precision fix
device = 'cuda'
dtype = 'float16'  # Test the float16 bug fix
compile = False
