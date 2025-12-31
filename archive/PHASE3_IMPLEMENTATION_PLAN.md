# Phase 3 Implementation Plan: Grassmann + Gating at d=1024

**Date:** December 26, 2024
**Goal:** Test Grassmann manifold optimization with gating on GPT-2 Medium architecture
**Status:** Ready to implement

---

## Configuration Summary

### Architecture
- **Width:** d=1024 (GPT-2 Medium)
- **Layers:** 24
- **Heads:** 16
- **Parameters:** ~350M total

### Layer Assignment Strategy

**Principle:** Use Grassmann ONLY for layers NOT facing skip connections

| Layer | Type | Shape | Optimizer | Parametrization | Scaling | LR | Reason |
|-------|------|-------|-----------|-----------------|---------|-----|--------|
| **c_attn** | Vertical (3n,n) | 3×(1024,1024) | Grassmann | G_{1.0, 0.0, 512} | x=10 | 8e-3 | Not skip-facing ✓ |
| **attn.c_proj** | Square | (1024,1024) | AdamW | - | - | 6e-4 | Faces skip ✗ |
| **mlp.c_fc** | Vertical (4n,n) | 4×(1024,1024) | Grassmann | G_{1.0, 0.0, 512} | x=10, frozen 0.5 | 8e-3 | Not skip-facing ✓ |
| **mlp.c_proj** | Square | (1024,4n) | AdamW | - | - | 6e-4 | Faces skip ✗ |

**Coverage:**
- Grassmann: c_attn (3.15M) + mlp.c_fc (4.19M) = 7.34M/layer × 24 = **176M params (58%)**
- AdamW: attn.c_proj (1.05M) + mlp.c_proj (4.19M) = 5.24M/layer × 24 = **126M params (42%)**

### Key Design Choices

1. **x=10 Uniform Scaling**
   - All Grassmann layers: `output = 10 × (W @ x)` where ||W|| = 1
   - Simple, uniform across all blocks
   - Will be gated afterward (effective ~1.2× on average if gate_mean≈0.12)

2. **Block Decomposition**
   - c_attn: 3 blocks (Q, K, V) - NO frozen scaling (components split immediately)
   - mlp.c_fc: 4 blocks - WITH frozen 0.5× scaling (full vector used by GELU)

3. **Parametrization**
   - All Grassmann: G_{1.0, 0.0, r} (identity-centered, no skip)
   - Rank: r=512 (50% of width)

4. **Gating**
   - Type: Head-wise (16K params/layer)
   - Position: G1 (after SDPA, before c_proj)
   - Applied to AdamW c_proj output

5. **Learning Rates**
   - AdamW base: 6e-4
   - Grassmann: 8e-3 (8× boost, proven on CIFAR-10)

---

## Implementation Steps

### Step 1: Implement Block Decomposition Module ✓

File: `manifold/block_decomposition.py`

```python
import torch
import torch.nn as nn

class BlockDecomposedLinear(nn.Module):
    """
    Decomposes rectangular (kN, N) matrix into k square (N, N) blocks.

    Used for:
    - c_attn: (3n, n) → 3 blocks (Q, K, V)
    - mlp.c_fc: (4n, n) → 4 blocks (expansion)

    Each block will use Grassmann constraint independently.
    """

    def __init__(self, in_features, out_features, n_blocks,
                 frozen_scale=None, bias=False):
        """
        Args:
            in_features: Input dimension (N)
            out_features: Output dimension (kN)
            n_blocks: Number of blocks (k)
            frozen_scale: Optional frozen scalar multiplier (e.g., 0.5 for mlp.c_fc)
            bias: Whether to include bias (should be False for Grassmann)
        """
        super().__init__()

        assert out_features == n_blocks * in_features, \
            f"out_features ({out_features}) must equal n_blocks × in_features ({n_blocks * in_features})"

        self.in_features = in_features
        self.out_features = out_features
        self.n_blocks = n_blocks

        # Create k independent (N, N) blocks
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, in_features, bias=bias)
            for _ in range(n_blocks)
        ])

        # Frozen scaling (for mlp.c_fc only)
        if frozen_scale is not None:
            self.register_buffer('frozen_scale',
                               torch.ones(out_features) * frozen_scale)
        else:
            self.frozen_scale = None

    def forward(self, x):
        """
        Args:
            x: (B, T, in_features)
        Returns:
            y: (B, T, out_features) = (B, T, n_blocks × in_features)
        """
        # Apply each block independently
        outputs = [block(x) for block in self.blocks]

        # Concatenate vertically: [W1@x; W2@x; ...; Wk@x]
        y = torch.cat(outputs, dim=-1)  # (B, T, k*in_features)

        # Apply frozen scaling if configured (mlp.c_fc needs 0.5×)
        if self.frozen_scale is not None:
            y = y * self.frozen_scale

        return y

    def get_grassmann_params(self):
        """Return list of block weights for Grassmann optimizer."""
        return [block.weight for block in self.blocks]
```

---

### Step 2: Modify model.py for Block Decomposition

**Changes needed:**

1. Import block decomposition:
```python
from manifold.block_decomposition import BlockDecomposedLinear
```

2. Add config parameters:
```python
@dataclass
class GPTConfig:
    # ... existing ...
    # Grassmann configuration
    use_grassmann: bool = False
    grass_rank: int = None  # If None, use 50% of n_embd
    grass_scale: float = 10.0  # Uniform scaling factor (x)
    grass_lr: float = 8e-3  # Learning rate for Grassmann layers
    grass_a: float = 1.0  # G_{a,b,r} parametrization
    grass_b: float = 0.0
```

3. Modify CausalSelfAttention:
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V projections - use block decomposition if Grassmann enabled
        if getattr(config, 'use_grassmann', False):
            # Decompose into 3 blocks (Q, K, V)
            self.c_attn = BlockDecomposedLinear(
                config.n_embd,
                3 * config.n_embd,
                n_blocks=3,
                frozen_scale=None,  # NO scaling (components split immediately)
                bias=config.bias
            )
            self.c_attn_is_grassmann = True
        else:
            # Standard combined linear
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_attn_is_grassmann = False

        # Output projection - ALWAYS AdamW (faces skip connection)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # ... rest unchanged ...
```

4. Modify MLP:
```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Expansion layer - use block decomposition if Grassmann enabled
        if getattr(config, 'use_grassmann', False):
            # Decompose into 4 blocks with 0.5× frozen scaling
            self.c_fc = BlockDecomposedLinear(
                config.n_embd,
                4 * config.n_embd,
                n_blocks=4,
                frozen_scale=0.5,  # CRITICAL: 0.5× to normalize √4 → √1
                bias=config.bias
            )
            self.c_fc_is_grassmann = True
        else:
            # Standard linear
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_fc_is_grassmann = False

        # Contraction layer - ALWAYS AdamW (faces skip connection)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
```

5. Update configure_optimizers_grassmann:
```python
def configure_optimizers_grassmann(self, weight_decay, learning_rate,
                                  grass_lr, betas, device_type):
    """
    Configure hybrid optimizer: Grassmann for c_attn/mlp.c_fc blocks, AdamW for rest.
    """
    # Separate parameters
    grassmann_params = []
    adamw_params = []

    for name, module in self.named_modules():
        if isinstance(module, BlockDecomposedLinear):
            # Grassmann: all block weights
            grassmann_params.extend(module.get_grassmann_params())
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            # AdamW: everything else
            adamw_params.append(module.weight)

    # AdamW optimizer
    optim_groups = [
        {'params': adamw_params, 'weight_decay': weight_decay, 'lr': learning_rate}
    ]
    optimizer = torch.optim.AdamW(optim_groups, betas=betas)

    return optimizer, grassmann_params
```

---

### Step 3: Update train.py for Grassmann Updates

**Modify training loop:**

```python
# After AdamW step
if use_grassmann:
    # Get grassmann params
    grassmann_params = []
    for name, module in model.named_modules():
        if isinstance(module, BlockDecomposedLinear):
            grassmann_params.extend(module.get_grassmann_params())

    # Apply Grassmann updates to each block
    with torch.no_grad():
        for param in grassmann_params:
            if param.grad is not None:
                # Unscale gradient if using mixed precision
                if dtype == 'float16':
                    scaler.unscale_(optimizer)

                # Grassmann update: projector manifold
                grassmann_muon_update(
                    param.data,
                    param.grad,
                    eta=grass_lr * lr_schedule_value,
                    a=grass_a,
                    b=grass_b,
                    r=grass_rank,
                    alpha=0.01,  # Dual ascent step
                    steps=10,    # Dual ascent iterations
                    tol=1e-6
                )
```

---

### Step 4: Add Grassmann Scaling in Forward Pass

**Modify forward passes to apply x=10 scaling:**

```python
# In CausalSelfAttention.forward():
if self.c_attn_is_grassmann:
    # Block decomposition outputs are already scaled by weights
    # Apply x=10 scaling uniformly
    qkv = self.c_attn(x) * self.grass_scale  # Multiply by 10
    q, k, v = qkv.split(self.n_embd, dim=2)
else:
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

# In MLP.forward():
if self.c_fc_is_grassmann:
    # Apply x=10 scaling (frozen 0.5× already applied in BlockDecomposedLinear)
    x = self.c_fc(x) * self.grass_scale  # Multiply by 10
else:
    x = self.c_fc(x)
```

---

### Step 5: Initialize Grassmann Weights

**Modify init_grassmann_weights:**

```python
def init_grassmann_weights(self, a=1.0, b=0.0, rank=512):
    """
    Initialize Grassmann block weights on manifold.

    Args:
        a, b: Eigenvalues for G_{a,b,r} (use 1.0, 0.0 for non-skip layers)
        rank: Projector rank (512 for d=1024)
    """
    from manifold import initialize_on_grassmann

    for name, module in self.named_modules():
        if isinstance(module, BlockDecomposedLinear):
            # Initialize each block independently
            for block in module.blocks:
                if hasattr(block, 'weight'):
                    initialize_on_grassmann(block.weight, a, b, rank)
                    print(f"Initialized {name} block on G_{{{a},{b},{rank}}}")
```

---

### Step 6: Create Config File

File: `config/train_gpt2_medium_grassmann_gated.py`

```python
# Phase 3: Grassmann + Gating at d=1024
# Test skip-avoidance strategy with uniform x=10 scaling

# GPT-2 Medium architecture
n_layer = 24
n_embd = 1024
n_head = 16
block_size = 1024

# Enable both Grassmann and Gating
use_grassmann = True
use_gating = True

# Grassmann configuration
grass_rank = 512  # 50% of n_embd
grass_scale = 10.0  # Uniform x=10 scaling
grass_a = 1.0  # G_{1.0, 0.0, r} for non-skip layers
grass_b = 0.0
grass_lr = 8e-3  # 8× boost (proven on CIFAR-10)

# Gating configuration
gate_type = 'headwise'  # 16K params/layer

# Training hyperparameters
learning_rate = 6e-4  # Base LR for AdamW components
max_iters = 100000
batch_size = 12  # Per GPU
gradient_accumulation_steps = 5  # Effective batch = 12 × 5 × 4 GPUs = 240

# Regularization
dropout = 0.1
weight_decay = 1e-1

# Evaluation
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Dataset
dataset = 'openwebtext'
always_save_checkpoint = True

# CRITICAL: Save to scratch!
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-gate-x10'

# Compile
compile = True

# Multi-GPU
# Will be handled by DDP wrapper in train.py
```

---

### Step 7: Multi-GPU Training Setup

**Create SLURM script for 4 GPUs:**

File: `submit_gpt2m_grassmann_gated.sh`

```bash
#!/bin/bash
#SBATCH --job-name=gpt2m_grass_gate
#SBATCH --output=logs/%j_gpt2m_grass_gate.out
#SBATCH --error=logs/%j_gpt2m_grass_gate.err
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4  # 4 H100 GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1

# Phase 3: Grassmann + Gating at d=1024
# Expected duration: ~2 days for 100k iters on 4×H100

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run multi-GPU training with DDP
cd /home/junyuren/nanoGPT

# PyTorch DDP with 4 GPUs
torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_medium_grassmann_gated.py

echo "Training complete. Checkpoint saved to /net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-gate-x10/ckpt.pt"
```

---

## Expected Results

**Success criteria:**
- ✅ Training converges (loss decreases to ~3.0-3.5)
- ✅ No NaN/Inf (gating should provide stability)
- ✅ Perplexity competitive with baseline (within 5%)
- ✅ Better train/val gap (Grassmann generalization)

**Failure modes:**
- ❌ Loss plateaus early → x=10 too small, try x=15
- ❌ Loss explodes → LR too high, reduce grass_lr
- ❌ Slow convergence → Width still too small, try d=1536

---

## Timeline

- **Implementation:** 2-3 hours
- **Training:** ~48 hours (100k iters on 4×H100)
- **Total:** ~2.5 days

---

## Next Steps After This Implementation

1. If successful → Scale to full training (600k iters)
2. If x=10 too small → Retry with x=15, x=20
3. If width insufficient → Scale to d=1536
4. If skip-facing needed → Revisit AdamW layers

---

**Status:** Ready to implement
**Start:** Step 1 - Create block_decomposition.py
