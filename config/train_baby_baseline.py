# Baby GPT baseline with AdamW optimizer
# Based on CIFAR-10 experiments, this is for comparison with Grassmann Muon

out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False  # set to True if using wandb
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'baby-adamw-baseline'

# dataset
dataset = 'shakespeare_char'  # use 'wikitext-2' for larger experiments
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Baby GPT model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # nanoGPT standard for Shakespeare (small dataset)
bias = False

# AdamW optimizer
learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99  # nanoGPT standard (bigger for small token count)
grad_clip = 1.0

# learning rate schedule (cosine with warmup)
decay_lr = True
warmup_iters = 100

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Grassmann disabled for baseline
use_grassmann = False
