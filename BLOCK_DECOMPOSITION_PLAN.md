# Block Decomposition for Grassmann Muon in nanoGPT

**Date:** 2025-11-17
**Status:** Planning Complete - Ready for Implementation

---

## Executive Summary

Extend Grassmann Muon optimizer to rectangular weight matrices via block decomposition, enabling manifold optimization on Q, K, V projections and MLP layers. Test 5 phases ranging from 28% to 82% Grassmann coverage.

**Key Innovation:** Decompose rectangular (kN, N) and (N, kN) matrices into k square (N, N) blocks, each on Grassmann manifold G_{a,b,r}.

**CIFAR-10 Evidence:**
- Grassmann (no-skip): 65.34% test, lr=4e-3
- AdamW ResNet: 62.15% test, lr=5e-4
- **8× LR ratio**: Grassmann enables 8× higher learning rate

---

## Mathematical Foundation

### Block Decomposition Strategy

#### Type 1: (kN, N) Matrices - Vertical Stacking
**Examples:** c_attn (1152, 384), mlp.c_fc (1536, 384)

```
W ∈ ℝ^{kN×N} = [P₁]    where each Pᵢ ∈ ℝ^{N×N} on G_{a,b,r}
                 [P₂]
                 [⋮ ]
                 [Pₖ]

Forward: y = (1/√k) × W @ x
         = (1/√k) × [P₁@x; P₂@x; ...; Pₖ@x]

Norm analysis: ‖y‖² = (1/k) × Σᵢ ‖Pᵢ@x‖² ≈ ‖x‖²
               (assuming ‖Pᵢ‖ ≈ 1)
```

**Scaling factor:** 1/√k (post-multiply after concatenation)

#### Type 2: (N, kN) Matrices - Horizontal Concatenation
**Example:** mlp.c_proj (384, 1536)

```
W ∈ ℝ^{N×kN} = [P₁ | P₂ | ⋯ | Pₖ]  where each Pᵢ ∈ ℝ^{N×N} on G_{a,b,r}

Split input: x = [x₁; x₂; ...; xₖ] where xᵢ ∈ ℝ^{N}

Forward: y = Σᵢ Pᵢ @ xᵢ

Norm analysis: Each ‖xᵢ‖² ≈ ‖x‖²/k (variance split)
               ‖y‖² ≈ Σᵢ ‖Pᵢ@xᵢ‖² ≈ k × (‖x‖²/k) = ‖x‖²
```

**Scaling factor:** 1.0 (no scaling needed, naturally norm-preserving)

#### Type 3: (N, N) Matrices - No Decomposition
**Example:** c_proj (384, 384)

Already square, apply Grassmann directly (if using manifold optimization).

---

## nanoGPT Architecture Inventory

### Baby GPT (n_layer=6, n_embd=384, ~10.7M total params)

```python
# Per transformer block:
├── ln_1: LayerNorm(384)                      → AdamW (always)
├── attn.c_attn.weight: (1152, 384)          → 3 blocks of (384, 384)
│                                               [Q; K; V] vertically stacked
├── attn.c_proj.weight: (384, 384)           → Already square
├── ln_2: LayerNorm(384)                      → AdamW (always)
├── mlp.c_fc.weight: (1536, 384)             → 4 blocks of (384, 384)
│                                               4× expansion
└── mlp.c_proj.weight: (384, 1536)           → 4 blocks of (384, 384)
                                                Contraction

# Shared across model:
├── wte (token embeddings): (50257, 384)      → AdamW (always)
├── wpe (position embeddings): (256, 384)     → AdamW (always)
└── lm_head: (50257, 384)                     → AdamW (always)
```

**Skip connections:**
```python
# In Block.forward():
x = x + self.attn(self.ln_1(x))  # ← SKIP around entire attention
x = x + self.mlp(self.ln_2(x))   # ← SKIP around entire MLP
```

**Q, K, V have NO skip around them** (intermediate computations)
**c_proj, mlp.c_proj DO face skip** (final projections)

---

## Grassmann Parametrization Strategy

### Skip Connection Compatibility (CIFAR-10 Finding)

**Without skip connections:**
- Use G_{1.0, 0.0, r} (identity-centered manifold)
- Can use aggressive lr (8× higher than AdamW)
- W = I + P where P is rank-r projector

**With skip connections:**
- Use G_{0.0, -1.0, r} (pure projector manifold)
- Must use conservative lr (same as AdamW)
- W = -I + P = P - I
- The identity component cancels with skip: x + Wx = x + (P-I)x = Px

### Per-Weight Decision Table

| Weight | Shape | Blocks | Direct Skip? | Manifold | LR (relative to AdamW) |
|--------|-------|--------|--------------|----------|------------------------|
| **c_attn (Q,K,V)** | (1152, 384) | 3 | ❌ No | G_{1.0, 0.0, 240} | **8× (8e-3)** |
| **c_proj (O)** | (384, 384) | 1 | ✅ Yes | AdamW or G_{0,-1,192} | 1× (1e-3) |
| **mlp.c_fc** | (1536, 384) | 4 | ❌ No | G_{1.0, 0.0, 960} | **8× (8e-3)** |
| **mlp.c_proj** | (384, 1536) | 4 | ✅ Yes | AdamW or G_{0,-1,960} | 1× (1e-3) |

**Rank selection:** r = 62.5% of dimension (optimal from CIFAR-10)
- For 384: r = 240
- For 1536: r = 960 (62.5% of 1536)

---

## Experimental Phases

### Baseline: Pure AdamW
**Purpose:** Establish reference performance

```yaml
Optimizer: AdamW everywhere
LR: 1e-3 (nanoGPT Baby GPT default)
Weight decay: 1e-1
Architecture: Standard with skip connections
Coverage: 0% Grassmann
```

**Expected:** ~2.0-2.5 val loss (typical for Baby GPT on Shakespeare)

---

### Phase 1: QKV Only
**Purpose:** Test if attention Q, K, V benefit from manifold constraint

**Coverage:** ~28% of params (6 × 3 × 384² = 2.65M out of ~10.7M)

```yaml
c_attn (Q, K, V):
  - Decompose into 3 blocks of (384, 384)
  - Each block: G_{1.0, 0.0, 240}
  - LR: 8e-3 (8× boost, no skip around Q/K/V)
  - Scaling: 1/√3 post-multiply
  - Dual ascent: steps=10, alpha=0.01

All other weights:
  - Optimizer: AdamW
  - LR: 1e-3
```

**Implementation:**
```python
# In configure_optimizers_grassmann():
if 'attn.c_attn.weight' in name:
    grassmann_params.append({
        'name': name,
        'param': param,
        'k': 3,  # 3 blocks (Q, K, V)
        'r': 240,
        'a': 1.0, 'b': 0.0,
        'lr': 8e-3,
        'scale': 1/math.sqrt(3),
        'type': 'vertical'  # (kN, N) stacking
    })
```

**Expected:** Modest improvement (2-5%) if attention benefits from manifold

---

### Phase 2: MLP Expansion Only
**Purpose:** Test if MLP c_fc (expansion) benefits from manifold

**Coverage:** ~34% of params (6 × 4 × 384² = 3.54M out of ~10.7M)

```yaml
mlp.c_fc:
  - Decompose into 4 blocks of (384, 384)
  - Each block: G_{1.0, 0.0, 960}
  - LR: 8e-3 (8× boost, no skip around c_fc)
  - Scaling: 1/√4 = 1/2 post-multiply
  - Dual ascent: steps=10, alpha=0.01

All other weights:
  - Optimizer: AdamW
  - LR: 1e-3
```

**Implementation:**
```python
if 'mlp.c_fc.weight' in name:
    grassmann_params.append({
        'name': name,
        'param': param,
        'k': 4,  # 4 blocks (4× expansion)
        'r': 960,  # 62.5% of 1536
        'a': 1.0, 'b': 0.0,
        'lr': 8e-3,
        'scale': 0.5,  # 1/√4
        'type': 'vertical'
    })
```

**Expected:** Similar magnitude as Phase 1 (ablation to separate effects)

---

### Phase 3: QKV + MLP Expansion
**Purpose:** Test if benefits compose additively

**Coverage:** ~62% of params (Phase 1 + Phase 2)

```yaml
Combine Phase 1 + Phase 2:
  - c_attn: 3 blocks, G_{1,0,240}, lr=8e-3, scale=1/√3
  - mlp.c_fc: 4 blocks, G_{1,0,960}, lr=8e-3, scale=1/2
  - All others: AdamW, lr=1e-3
```

**Expected:**
- If additive: ~2× improvement of Phase 1 or 2
- If synergistic: >2× improvement
- If conflicting: <2× improvement

---

### Phase 4: Full MLP with Skip-Compatible Projection
**Purpose:** Test if mlp.c_proj needs manifold constraint despite skip

**Coverage:** ~62% of params (same as Phase 3, but c_proj now Grassmann)

```yaml
c_attn: Same as Phase 3 (G_{1,0,240}, lr=8e-3)

mlp.c_fc: Same as Phase 3 (G_{1,0,960}, lr=8e-3, scale=1/2)

mlp.c_proj: NEW!
  - Decompose into 4 blocks of (384, 384)
  - Each block: G_{0.0, -1.0, 960}  ← Skip-compatible!
  - LR: 1e-3 (SAME as AdamW, no 8× boost)
  - Scaling: 1.0 (no scaling for horizontal concat)
  - Type: horizontal (N, kN) concatenation

c_proj: Still AdamW, lr=1e-3
```

**Key difference:** mlp.c_proj uses skip-compatible G_{0,-1,r} at lr=1e-3

**Implementation:**
```python
if 'mlp.c_proj.weight' in name:
    grassmann_params.append({
        'name': name,
        'param': param,
        'k': 4,
        'r': 960,
        'a': 0.0, 'b': -1.0,  # Skip-compatible!
        'lr': 1e-3,  # No 8× boost
        'scale': 1.0,  # No scaling for (N, kN)
        'type': 'horizontal'
    })
```

**Expected:** Test if skip-facing projections benefit from manifold (likely small effect)

---

### Phase 5: No-Skip Architecture (All Grassmann)
**Purpose:** Test if manifold can REPLACE skip connections

**Coverage:** ~82% of params (all attention + MLP weights)

**Architecture change:** Modify Block.forward() to remove skip connections

```python
# Original (with skip):
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

# Phase 5 (no skip):
def forward(self, x):
    x = self.attn(self.ln_1(x))  # NO skip
    x = self.mlp(self.ln_2(x))   # NO skip
    return x
```

```yaml
ALL weights use G_{1.0, 0.0, r} with lr=8e-3:

c_attn: 3 blocks, G_{1,0,240}, lr=8e-3, scale=1/√3
c_proj: 1 block, G_{1,0,192}, lr=8e-3  ← NOW Grassmann (no skip!)
mlp.c_fc: 4 blocks, G_{1,0,960}, lr=8e-3, scale=1/2
mlp.c_proj: 4 blocks, G_{1,0,960}, lr=8e-3, scale=1.0

Rank: r = n/2 everywhere (n_embd/2 = 192 for square weights)
```

**Rationale:**
- CIFAR-10: Grassmann (no-skip) 65.34% >> AdamW ResNet 62.15%
- Manifold constraint provides implicit regularization
- May enable deeper networks without gradient issues

**Expected:** Best performance if manifold can substitute for skip connections

---

## Implementation Architecture

### File Structure

```
nanoGPT/
├── manifold/
│   ├── __init__.py
│   ├── grassmann_muon.py          (existing)
│   ├── grassmann_ops.py           (existing)
│   ├── msign.py                   (existing)
│   └── block_decomposition.py     (NEW - block logic)
├── model.py                        (modify)
├── train.py                        (modify)
└── config/
    ├── train_baby_baseline.py     (existing)
    ├── phase1_qkv.py              (NEW)
    ├── phase2_mlp_fc.py           (NEW)
    ├── phase3_qkv_mlp.py          (NEW)
    ├── phase4_full_mlp.py         (NEW)
    └── phase5_no_skip.py          (NEW)
```

### Core Components

#### 1. Block Decomposition Module (`manifold/block_decomposition.py`)

```python
def decompose_weight_vertical(W, k):
    """
    Split (kN, N) weight into k blocks of (N, N).

    Args:
        W: (kN, N) tensor
        k: number of blocks
    Returns:
        blocks: (k, N, N) tensor
    """
    kN, N = W.shape
    assert kN == k * N, f"Shape mismatch: {kN} != {k} * {N}"
    return W.view(k, N, N)

def decompose_weight_horizontal(W, k):
    """
    Split (N, kN) weight into k blocks of (N, N).

    Args:
        W: (N, kN) tensor
        k: number of blocks
    Returns:
        blocks: (k, N, N) tensor
    """
    N, kN = W.shape
    assert kN == k * N, f"Shape mismatch: {kN} != {k} * {N}"
    return W.view(N, k, N).transpose(0, 1).contiguous()

def compose_weight_vertical(blocks, scale=1.0):
    """
    Combine k blocks into (kN, N) weight with scaling.

    Args:
        blocks: (k, N, N) tensor
        scale: scaling factor (1/√k for vertical)
    Returns:
        W: (kN, N) tensor
    """
    k, N, _ = blocks.shape
    W = blocks.view(k * N, N)
    return W * scale

def compose_weight_horizontal(blocks, scale=1.0):
    """
    Combine k blocks into (N, kN) weight.

    Args:
        blocks: (k, N, N) tensor
        scale: scaling factor (1.0 for horizontal)
    Returns:
        W: (N, kN) tensor
    """
    k, N, _ = blocks.shape
    W = blocks.transpose(0, 1).reshape(N, k * N)
    return W * scale

def apply_grassmann_blocks(W, G, eta, a, b, r, k, decomp_type, scale):
    """
    Apply Grassmann Muon update to block-decomposed weight.

    Args:
        W: weight tensor (kN, N) or (N, kN)
        G: gradient tensor (same shape as W)
        eta: learning rate
        a, b: Grassmann eigenvalues
        r: rank
        k: number of blocks
        decomp_type: 'vertical' or 'horizontal'
        scale: scaling factor
    Returns:
        W_new: updated weight tensor
    """
    from .grassmann_muon import grassmann_muon_update

    # Decompose
    if decomp_type == 'vertical':
        blocks_W = decompose_weight_vertical(W, k)
        blocks_G = decompose_weight_vertical(G, k)
    else:
        blocks_W = decompose_weight_horizontal(W, k)
        blocks_G = decompose_weight_horizontal(G, k)

    # Update each block
    for i in range(k):
        blocks_W[i] = grassmann_muon_update(
            blocks_W[i], blocks_G[i], eta, a, b, r
        )

    # Recompose
    if decomp_type == 'vertical':
        return compose_weight_vertical(blocks_W, scale)
    else:
        return compose_weight_horizontal(blocks_W, scale)
```

#### 2. Model Modifications (`model.py`)

**Add block initialization:**

```python
def init_grassmann_weights_blocks(self, grassmann_config):
    """
    Initialize block-decomposed weights on Grassmann manifold.

    Args:
        grassmann_config: list of dicts with keys:
            - name: parameter name
            - k: number of blocks
            - a, b: eigenvalues
            - r: rank
            - decomp_type: 'vertical' or 'horizontal'
    """
    from manifold.block_decomposition import decompose_weight_vertical, decompose_weight_horizontal
    from manifold import initialize_on_grassmann

    for cfg in grassmann_config:
        param = dict(self.named_parameters())[cfg['name']]

        # Decompose
        if cfg['decomp_type'] == 'vertical':
            blocks = decompose_weight_vertical(param.data, cfg['k'])
        else:
            blocks = decompose_weight_horizontal(param.data, cfg['k'])

        # Initialize each block on manifold
        with torch.no_grad():
            for i in range(cfg['k']):
                blocks[i] = initialize_on_grassmann(
                    blocks[i], a=cfg['a'], b=cfg['b'], r=cfg['r']
                )

        # Recompose (no scaling at init)
        if cfg['decomp_type'] == 'vertical':
            param.data = blocks.view(cfg['k'] * blocks.size(1), blocks.size(2))
        else:
            param.data = blocks.transpose(0,1).reshape(blocks.size(1), cfg['k']*blocks.size(2))

        print(f"Initialized {cfg['name']} as {cfg['k']} blocks on G_{{{cfg['a']},{cfg['b']},{cfg['r']}}}")
```

**Extend configure_optimizers_grassmann:**

```python
def configure_optimizers_grassmann(self, weight_decay, learning_rate, betas, device_type, phase):
    """
    Configure hybrid optimizer with block decomposition support.

    Args:
        phase: 'phase1', 'phase2', 'phase3', 'phase4', or 'phase5'
    """
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

    grassmann_params = []
    adamw_params_decay = []
    adamw_params_nodecay = []

    for pn, p in param_dict.items():
        is_grassmann = False

        # Phase-specific logic
        if phase in ['phase1', 'phase3', 'phase4', 'phase5']:
            if 'attn.c_attn.weight' in pn:
                grassmann_params.append({
                    'name': pn, 'param': p,
                    'k': 3, 'r': 240, 'a': 1.0, 'b': 0.0,
                    'lr': 8e-3, 'scale': 1/math.sqrt(3),
                    'decomp_type': 'vertical'
                })
                is_grassmann = True

        if phase in ['phase2', 'phase3', 'phase4', 'phase5']:
            if 'mlp.c_fc.weight' in pn:
                grassmann_params.append({
                    'name': pn, 'param': p,
                    'k': 4, 'r': 960, 'a': 1.0, 'b': 0.0,
                    'lr': 8e-3, 'scale': 0.5,
                    'decomp_type': 'vertical'
                })
                is_grassmann = True

        if phase == 'phase4':
            if 'mlp.c_proj.weight' in pn:
                grassmann_params.append({
                    'name': pn, 'param': p,
                    'k': 4, 'r': 960, 'a': 0.0, 'b': -1.0,
                    'lr': 1e-3, 'scale': 1.0,
                    'decomp_type': 'horizontal'
                })
                is_grassmann = True

        if phase == 'phase5':
            if 'attn.c_proj.weight' in pn and p.dim() == 2 and p.shape[0] == p.shape[1]:
                grassmann_params.append({
                    'name': pn, 'param': p,
                    'k': 1, 'r': 192, 'a': 1.0, 'b': 0.0,
                    'lr': 8e-3, 'scale': 1.0,
                    'decomp_type': None  # Already square
                })
                is_grassmann = True
            if 'mlp.c_proj.weight' in pn:
                grassmann_params.append({
                    'name': pn, 'param': p,
                    'k': 4, 'r': 960, 'a': 1.0, 'b': 0.0,
                    'lr': 8e-3, 'scale': 1.0,
                    'decomp_type': 'horizontal'
                })
                is_grassmann = True

        # AdamW for non-Grassmann params
        if not is_grassmann:
            if p.dim() >= 2:
                adamw_params_decay.append(p)
            else:
                adamw_params_nodecay.append(p)

    # Create AdamW optimizer
    optim_groups = [
        {'params': adamw_params_decay, 'weight_decay': weight_decay},
        {'params': adamw_params_nodecay, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    return optimizer, grassmann_params
```

#### 3. Training Loop Modifications (`train.py`)

**Update optimizer step:**

```python
# After scaler.step(optimizer) and scaler.update()

if use_grassmann and len(grassmann_params) > 0:
    from manifold.block_decomposition import apply_grassmann_blocks

    for cfg in grassmann_params:
        param = cfg['param']
        if param.grad is not None:
            # Apply Grassmann update with block decomposition
            if cfg.get('decomp_type') is None:
                # Square matrix, no decomposition
                param.data = grassmann_muon_update(
                    param.data, param.grad, cfg['lr'],
                    cfg['a'], cfg['b'], cfg['r']
                )
            else:
                # Rectangular, use block decomposition
                param.data = apply_grassmann_blocks(
                    W=param.data,
                    G=param.grad,
                    eta=cfg['lr'],
                    a=cfg['a'], b=cfg['b'], r=cfg['r'],
                    k=cfg['k'],
                    decomp_type=cfg['decomp_type'],
                    scale=cfg['scale']
                )
```

**For Phase 5, modify model architecture:**

```python
# In train.py, after model creation
if phase == 'phase5':
    # Monkey-patch Block.forward to remove skip connections
    original_forward = model.transformer.h[0].__class__.forward

    def forward_no_skip(self, x):
        x = self.attn(self.ln_1(x))  # No skip
        x = self.mlp(self.ln_2(x))   # No skip
        return x

    for block in model.transformer.h:
        block.forward = forward_no_skip.__get__(block, block.__class__)

    print("Phase 5: Removed skip connections from all transformer blocks")
```

---

## Configuration Files

### Phase 1: QKV Only

```python
# config/phase1_qkv.py

out_dir = 'out-phase1-qkv'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt-manifold'
wandb_run_name = 'phase1-qkv'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Baby GPT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# AdamW for non-Grassmann params
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100

# Grassmann config
use_grassmann = True
grassmann_phase = 'phase1'

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
```

### Phase 2: MLP.c_fc Only

```python
# config/phase2_mlp_fc.py

# Same as phase1 except:
out_dir = 'out-phase2-mlp-fc'
wandb_run_name = 'phase2-mlp-fc'
grassmann_phase = 'phase2'
```

### Phase 3: QKV + MLP.c_fc

```python
# config/phase3_qkv_mlp.py

# Same as phase1 except:
out_dir = 'out-phase3-qkv-mlp'
wandb_run_name = 'phase3-qkv-mlp'
grassmann_phase = 'phase3'
```

### Phase 4: Full MLP

```python
# config/phase4_full_mlp.py

# Same as phase1 except:
out_dir = 'out-phase4-full-mlp'
wandb_run_name = 'phase4-full-mlp'
grassmann_phase = 'phase4'
```

### Phase 5: No-Skip

```python
# config/phase5_no_skip.py

# Same as phase1 except:
out_dir = 'out-phase5-no-skip'
wandb_run_name = 'phase5-no-skip'
grassmann_phase = 'phase5'
```

---

## Testing & Validation

### Sanity Checks (Before Full Training)

**Test 1: Block decomposition correctness**
```python
# Verify decompose → compose is identity
W_orig = torch.randn(1152, 384)
blocks = decompose_weight_vertical(W_orig, k=3)
W_recon = compose_weight_vertical(blocks, scale=1.0)
assert torch.allclose(W_orig, W_recon)
```

**Test 2: Scaling verification**
```python
# Verify norm preservation
x = torch.randn(64, 384)
W = torch.randn(1152, 384)
y_unscaled = W @ x
y_scaled = (1/math.sqrt(3)) * W @ x

# Check: ‖y_scaled‖ ≈ ‖y_unscaled‖ / √3
```

**Test 3: Gradient flow**
```python
# Run 10 iters, verify loss decreases
# For each phase, check no NaNs, no divergence
```

### Full Validation

**Run baseline first:**
```bash
cd /home/junyuren/nanoGPT
sbatch submit_baseline.sh  # Pure AdamW reference
```

**Then run phases sequentially:**
```bash
sbatch submit_phase1.sh
sbatch submit_phase2.sh
sbatch submit_phase3.sh
sbatch submit_phase4.sh
sbatch submit_phase5.sh
```

**Monitor for:**
- Loss decreases monotonically after warmup
- No NaN or Inf values
- Validation loss improves over baseline
- Training time reasonable (<2× baseline)

---

## Expected Outcomes

### Performance Predictions

| Phase | Coverage | Expected Val Loss | Relative to Baseline | Rationale |
|-------|----------|-------------------|----------------------|-----------|
| **Baseline** | 0% | 2.0-2.5 | 0% (reference) | Standard nanoGPT |
| **Phase 1** | 28% | 1.9-2.3 | -2% to -5% | QKV manifold helps attention |
| **Phase 2** | 34% | 1.9-2.3 | -2% to -5% | MLP manifold helps features |
| **Phase 3** | 62% | 1.8-2.2 | -5% to -10% | Synergy if additive |
| **Phase 4** | 62% | 1.8-2.1 | -5% to -10% | Test c_proj manifold |
| **Phase 5** | 82% | 1.7-2.0 | **-10% to -15%** | Best if manifold replaces skip |

### Success Criteria

**Phase 1-4 (with skip):**
- ✅ Training stable (no divergence)
- ✅ Val loss ≤ baseline
- ✅ Train/val gap < baseline (better generalization)

**Phase 5 (no skip):**
- ✅ Training stable (critical test!)
- ✅ Val loss < all other phases
- ✅ Demonstrates manifold can replace architectural tricks

**If Phase 5 fails (unstable):**
- Still valuable negative result
- Shows skip connections remain necessary
- Fall back to Phase 3 or 4 as best approach

---

## Timeline & Resources

### Implementation (1-2 days)

**Day 1:**
- [ ] Implement `block_decomposition.py`
- [ ] Test decompose/compose functions
- [ ] Modify `model.py` (init + optimizer config)
- [ ] Create 5 phase configs

**Day 2:**
- [ ] Modify `train.py` (update loop + Phase 5 arch)
- [ ] Run sanity checks (10 iters each phase)
- [ ] Fix any bugs
- [ ] Create SLURM submission scripts

### Experimentation (6-8 hours)

**Each phase:** ~1-1.5 hours on H100 (5000 iters, Baby GPT)

**Parallel execution:** Submit all 6 jobs at once
- **Total wall time:** ~1.5 hours (limited by longest job)
- **Total GPU hours:** ~9 hours (6 jobs × 1.5 hrs)

### Analysis (0.5 days)

- Plot loss curves (train/val)
- Compare final metrics
- Compute train/val gaps
- Identify best phase
- Update documentation

**Total:** 3-4 days end-to-end

---

## Risk Mitigation

### Risk 1: Block Decomposition Bugs
**Probability:** Medium (30%)
**Mitigation:**
- Comprehensive unit tests before training
- Verify with small manual examples
- Check gradient norms match expectations

### Risk 2: Phase 5 Unstable (No Skip)
**Probability:** Medium-High (40%)
**Mitigation:**
- Use lower LR (4e-3 instead of 8e-3) as backup
- Increase warmup to 200 iters
- If fails, document and use Phase 3/4

### Risk 3: Scaling Errors
**Probability:** Low (20%)
**Mitigation:**
- Test norm preservation explicitly
- Compare against unscaled version
- Monitor activation magnitudes

### Risk 4: Slower Training
**Probability:** High (60%)
**Mitigation:**
- Accept 1.5-2× slowdown as cost
- Profile to identify bottlenecks
- Consider reducing dual ascent steps (10→5)

---

## References

### Internal Documentation
- `/home/junyuren/manifold_muon/HYPERPARAMETER_TUNING_RESULTS.md` - CIFAR-10 results
- `/home/junyuren/manifold_muon/NANOGPT_SCALING_PLAN.md` - Initial nanoGPT plan
- `/home/junyuren/nanoGPT/BLOCK_DECOMPOSITION_PLAN.md` - This document

### Key Results
- **CIFAR-10 Grassmann (no-skip):** 65.34% test, lr=4e-3 (Job 552298)
- **CIFAR-10 AdamW ResNet:** 62.15% test, lr=5e-4 (Job 553255)
- **8× LR ratio:** Grassmann enables 8× higher LR than AdamW

### External
- nanoGPT: https://github.com/karpathy/nanoGPT
- Modular Manifolds: https://thinkingmachines.ai/blog/modular-manifolds/

---

## Approval & Sign-Off

**Plan Status:** ✅ COMPLETE - Ready for Implementation

**Next Steps:**
1. Review plan thoroughly
2. Implement block decomposition module
3. Modify model.py and train.py
4. Create configs
5. Run sanity checks
6. Submit full experiments

**Implementation starts:** Upon approval
