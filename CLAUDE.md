# CLAUDE.md - nanoGPT Grassmann Manifold Optimization

## Current Status

**Active Development:** Hybrid Gating Architecture (Head-Level + Block-Level)

**Configuration:**
- Architecture: GPT-2 Medium (d=1024, 24 layers, 16 heads, ~530M params)
- Grassmann layers: c_attn, mlp.c_fc (non-skip-facing)
- AdamW layers: attn.c_proj, mlp.c_proj (skip-facing)
- **Gating Strategy:**
  - c_attn: 48 head-level gates (16 heads × 3 QKV), linear + sigmoid
  - c_fc: 4 block-level gates (4 MLP blocks), linear + sigmoid
  - Total: 52 gates/layer, 1.28M gate params
- Scaling: Uniform x=10 for all Grassmann layers
- Rank: r=384 (37.5% of width)
- Learning rates: 2e-3 (Grassmann), 6e-4 (AdamW base), 1.2e-3 (gates), 3e-4 (embeddings)

**Key Empirical Findings:**
- ❌ **Skip-facing layers incompatible with Grassmann** (even with G_{0,-1,r})
- ✅ **Non-skip-facing layers work well** with G_{1,0,r} (from CIFAR-10)
- ✅ **r=384 > r=512** (lower rank is better: -10.2% vs baseline @ 786M tokens)
- ✅ **Grassmann shows better generalization** (train/val gap 0.006 vs 0.012)
- ⚠️ **Advantage shrinks over training** (undertraining + expressivity bottleneck)

---

## Why This Configuration?

### 1. Width Scaling (d=1024)
Previous d=384 experiments failed due to width saturation:
- Effective rank: 330/384 = 86% of width
- No capacity gap for low-rank constraint
- Solution: Scale to d=1024 → ~50% effective rank ratio

### 2. Skip Connection Strategy
**Empirical observation:** Skip-facing layers fail with Grassmann regardless of parametrization

**Layer Assignment:**
| Layer | Skip-Facing? | Optimizer | Parametrization | LR | Why |
|-------|--------------|-----------|-----------------|-----|-----|
| c_attn (Q,K,V) | ❌ No | Grassmann | G_{1.0, 0.0, 512} | 8e-3 | No skip, use aggressive LR |
| attn.c_proj | ✅ Yes | AdamW | - | 6e-4 | Faces skip, avoid Grassmann |
| mlp.c_fc | ❌ No | Grassmann | G_{1.0, 0.0, 512} | 8e-3 | No skip, use aggressive LR |
| mlp.c_proj | ✅ Yes | AdamW | - | 6e-4 | Faces skip, avoid Grassmann |

### 3. Hybrid Gating Mechanism

**Design Principle:** Gating granularity matches natural structure
- **c_attn:** Head-level gates (respects multi-head attention structure)
- **c_fc:** Block-level gates (respects 4-block MLP structure)

**c_attn (Head-Level Gating):**
```python
# Grassmann projection (NO gating yet)
qkv = self.c_attn(x) * grass_scale  # 3 blocks (Q/K/V), scale=10.0

# Split into 16 heads
q, k, v = qkv.split(n_embd, dim=2)
q = q.view(B, T, 16, 64).transpose(1, 2)  # (B, 16, T, 64)
k = k.view(B, T, 16, 64).transpose(1, 2)
v = v.view(B, T, 16, 64).transpose(1, 2)

# Compute 48 head gates (16 heads × 3 QKV)
gates = sigmoid(head_gates(x))  # Linear(1024 → 48)
gate_q, gate_k, gate_v = gates.split(16, dim=-1)

# Apply per-head gating
q = q * gate_q.reshape(B, 16, T, 1)  # Each head independently gated
k = k * gate_k.reshape(B, 16, T, 1)
v = v * gate_v.reshape(B, 16, T, 1)
```

**c_fc (Block-Level Gating):**
```python
# Inside BlockDecomposedLinear
for i, block in enumerate(blocks):
    out_i = block(x)  # Grassmann 1024×1024, ||W||_op=1
    gate_i = sigmoid(block_gates[i](x))  # Linear(1024 → 1)
    out_i = out_i * gate_i * frozen_scale  # 0.5 for c_fc
    outputs.append(out_i)

output = cat(outputs) * grass_scale  # Concatenate 4 blocks, scale=10.0
```

**Why this gating strategy:**
- **Input-dependent magnitude** for fixed ||W||_op=1 (Grassmann rigidity)
- **Head diversity** restored (each head gets independent adaptive scaling)
- **Parameter efficient:** 1.28M params (vs 176M elementwise, 137× reduction)
- **Interpretable:** gate_Q[h] = "how much to use head h's query"
- **Linear gating:** Proven sufficient (Qwen NeurIPS 2025, LSTM/GRU gates)

### 4. Block Decomposition
Rectangular matrices decomposed into square blocks for Grassmann:

**c_attn (3n, n) → 3 blocks:**
```python
self.c_attn = BlockDecomposedLinear(
    in_features=1024,
    out_features=3072,
    n_blocks=3,
    frozen_scale=None  # NO scaling (Q,K,V split immediately)
)
# Forward: qkv = self.c_attn(x) * 10.0  # x=10 uniform scaling
```

**mlp.c_fc (4n, n) → 4 blocks:**
```python
self.c_fc = BlockDecomposedLinear(
    in_features=1024,
    out_features=4096,
    n_blocks=4,
    frozen_scale=0.5  # CRITICAL: 0.5× to normalize √4 → √1
)
# Forward: x = self.c_fc(x) * 10.0  # x=10 uniform scaling
```

**Why 0.5× frozen scale for mlp.c_fc?**
- Each block outputs norm ≈ σ/√2 (inherent to rank=n/2)
- Stacked: ||[P₁x; P₂x; P₃x; P₄x]|| = √4 · (σ/√2) = √2·σ
- Need σ/√2 for GELU: 0.5 × √2·σ = σ/√2 ✓

---

## Implementation Details

### Files Modified
1. **model.py** - Block decomposition integration
   - Import BlockDecomposedLinear
   - Modified CausalSelfAttention and MLP classes
   - Added gating at G1 position
   - Added init_grassmann_block_weights() method

2. **train.py** - Grassmann update loop
   - Hybrid optimizer: AdamW + Grassmann
   - Grassmann updates applied to block weights after AdamW step
   - Fixed gradient zeroing, float16 scaling, checkpoint resume bugs

3. **manifold/block_decomposition.py** - NEW
   - BlockDecomposedLinear class
   - Handles vertical stacking (kN, N) → k blocks (N, N)
   - Optional frozen scaling for norm preservation

4. **config/train_gpt2_medium_grassmann_gated.py**
   - GPT-2 Medium configuration
   - Grassmann + gating enabled
   - 4-GPU DDP setup (batch=15/GPU, grad_accum=4)

### Current Training Job
- **Job ID:** 603340 (corrected batch size + divisibility)
- **GPUs:** 4×H100 (12-hour limit)
- **Data:** OpenWebText at `/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext/`
- **Output:** `/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-gate-x10/`
- **Monitor:** `tail -f logs/603340_gpt2m_grass_gate.out`

**Previous attempts:**
- 603324: Batch too small (15×4=60 seqs/GPU) → MFU 2.57%, canceled
- 603335: Fixed batch size but grad_accum=10 not divisible by 4 GPUs → AssertionError

**Final correction (603340):**
- Batch: 64×12 = 768 sequences/GPU (786K tokens/GPU)
- **DDP constraint:** grad_accum MUST be divisible by num_gpus
- Expected MFU: 60-70%

---

## Critical Lessons Learned

### 1. Batch Size Dominates MFU (Not Grassmann Coverage)

**Finding (Dec 26, 2024):**
- Job 603324: batch=15, grad_accum=4 → **MFU 2.57%** (60 sequences/GPU)
- Job 603335: batch=64, grad_accum=10 → **Expected 60-70% MFU** (640 sequences/GPU)

**Key Insight:**
- ❌ **Small batch kills MFU regardless of architecture**
- ✅ **Grassmann coverage does NOT hurt MFU** (d=384 Grassmann had 70% vs baseline 64%)
- **Rule:** Match tokens/iteration to successful runs (~2.6M tokens for 4 GPUs)

**Tokens per GPU comparison:**
| Run | Params | Batch×GradAccum | Sequences/GPU | Tokens/GPU | MFU |
|-----|--------|-----------------|---------------|------------|-----|
| d=384 Baseline | 30M | 64×40 | 2560 | 2,621,440 | **64%** |
| d=384 Grassmann | 30M | 64×40 | 2560 | 2,621,440 | **70%** |
| d=1024 Failed | 354M | 15×4 | 60 | 61,440 | **2.57%** ❌ |
| d=1024 Fixed | 354M | 64×10 | 640 | 655,360 | **60-70%** ✓ |

**Why small batch fails:**
1. GPU compute idle most of the time
2. Memory bandwidth underutilized
3. Launch overhead dominates (kernel launches, syncs)
4. Training also unstable (gradient noise too high)

**Recommendation:** Always target 2-3M tokens/iteration for multi-GPU setups.

### 2. DDP Divisibility Constraint

**Critical:** `gradient_accumulation_steps % num_gpus == 0`

**Why:** PyTorch DDP splits gradient accumulation across GPUs. If not divisible, assertion fails.

**Example (4 GPUs):**
- ✅ grad_accum = 4, 8, 12, 16, 20...
- ❌ grad_accum = 10 (10 % 4 = 2) → AssertionError

**Job 603335 failed** because grad_accum=10 with 4 GPUs.

---

## Previous Attempts (Archived)

**d=384 Experiments (Failed):**
- Effective rank too high (86% of width)
- All skip-compatible parametrizations failed
- Width too small for Grassmann constraint

**Phase 0-5 Plans (Obsolete):**
- Original plan assumed skip-facing layers could use G_{0,-1,r}
- Empirical results proved this wrong
- See `archive/MASTER_IMPLEMENTATION_PLAN.md` for details

---

## CIFAR-10 Transfer Learning (Validated Findings)

### Proven Results
- **No-skip MLP:** 64.97% test (G_{1.0, 0.0, 256}, lr=4e-3)
- **ResNet-MLP with skip:** 60.96% test (G_{0.0, -1.0, 160}, lr=5e-4)

### Key Insights Transferred to nanoGPT
1. **8× LR boost** for non-skip Grassmann layers (4e-3 vs 5e-4 on CIFAR-10)
2. **Rank ~50% of width** works well (256/512 = 50% on CIFAR-10)
3. **Manifold constraint >> low-rank** (+4% improvement)
4. **Excellent generalization** (~7× better train/test gap)

### What DIDN'T Transfer
- ❌ Skip-compatible G_{0,-1,r} worked on CIFAR-10 ResNet-MLP
- ❌ But FAILED on transformers at d=384
- **Hypothesis:** Transformer skip connections behave differently than ResNet

---

## Cluster Workflow

### SLURM Principles
- **Never run `.py` files on login node** - always submit via SLURM
- **Always save checkpoints to `/net/scratch2/`** - home has strict quota
- **Prioritize H100/H200 GPUs** - 35% faster than A100

### Standard Training Command
```bash
# Submit job
sbatch submit_gpt2m_grassmann_gated.sh

# Monitor
squeue -u $USER
tail -f logs/<JOBID>_gpt2m_grass_gate.out

# Check errors
tail -f logs/<JOBID>_gpt2m_grass_gate.err
```

### Cluster Configuration
- **Partitions:** dev (10 min), general (12 hour max)
- **GPUs:** H100 (best) > A100 > A40
- **Environment:** `source /home/junyuren/.conda/envs/manifold_muon/bin/activate`

---

## Hyperparameter Reference

### Current Configuration (d=1024)
```python
# Architecture
n_layer = 24
n_embd = 1024
n_head = 16

# Grassmann
grass_rank = 512          # 50% of width
grass_scale = 10.0        # Uniform scaling
grass_a = 1.0            # G_{1.0, 0.0, r}
grass_b = 0.0
grass_lr = 8e-3          # 8× boost

# Gating
use_gating = True
gate_type = 'headwise'    # 16 params/head

# Training
learning_rate = 6e-4      # AdamW base
batch_size = 64           # Per GPU (CRITICAL for MFU!)
gradient_accumulation_steps = 12  # MUST divide by 4 GPUs! Total 768 seqs/GPU
weight_decay = 1e-1
dropout = 0.1
```

### Parametrization by Layer
| Layer | Shape | Parametrization | Scaling | LR |
|-------|-------|----------------|---------|-----|
| c_attn blocks | 3×(1024,1024) | G_{1.0, 0.0, 512} | x=10 | 8e-3 |
| attn.c_proj | (1024,1024) | AdamW | - | 6e-4 |
| mlp.c_fc blocks | 4×(1024,1024) | G_{1.0, 0.0, 512} | 0.5×, then x=10 | 8e-3 |
| mlp.c_proj | (1024,4096) | AdamW | - | 6e-4 |

---

## Expected Results

**Success Criteria:**
- ✅ Training converges (loss → ~3.0-3.5)
- ✅ No NaN/Inf (gating provides stability)
- ✅ Competitive with baseline (within 5%)
- ✅ Better train/val gap (Grassmann generalization)

**Failure Modes:**
- ❌ Loss plateaus → x=10 too small, try x=15
- ❌ Loss explodes → LR too high, reduce grass_lr
- ❌ Slow convergence → Width insufficient, try d=1536

---

## Key Concepts

### Grassmann Manifold G_{a,b,r}
**Definition:** {bI + cP : P is rank-r projector, c = a - b}

**G_{1.0, 0.0, r}** (identity-centered):
- W = I + P (eigenvalues: 1 or 0)
- Use for NON-skip-facing layers
- Can use aggressive LR (8× boost)

**G_{0.0, -1.0, r}** (pure projector):
- W = P - I (eigenvalues: 0 or -1)
- Designed for skip connections
- **DOESN'T WORK on transformers** (empirical finding)

### Manifold Operations
```python
# Initialize on manifold (once at start)
initialize_on_grassmann(W, a=1.0, b=0.0, r=512)

# Update (every training step)
grassmann_muon_update(
    W=param.data,
    G=param.grad,
    eta=grass_lr * lr_schedule,
    a=1.0, b=0.0, r=512,
    alpha=0.01,  # Dual ascent step
    steps=10,    # Dual ascent iterations
    tol=1e-6
)
```

---

## File Structure

```
nanoGPT/
├── model.py                          # Grassmann + hybrid gating integration
├── train.py                          # Hybrid optimizer training loop
├── manifold/
│   ├── block_decomposition.py       # BlockDecomposedLinear with block gating
│   ├── grassmann_muon.py            # Dual ascent optimizer
│   ├── grassmann_ops.py             # Manifold operations
│   └── msign.py                     # Polar-Express matrix sign
├── config/
│   ├── train_gpt2_medium_grassmann_per_layer_gate_r384.py  # Current Grassmann config
│   └── train_gpt2_medium_baseline_gated.py                 # Baseline for comparison
├── docs/                             # Documentation
│   ├── GRASSMANN_GATING_ARCHITECTURE.md   # Hybrid gating design (CURRENT)
│   ├── OPTIMAL_GRASSMANN_EXPERIMENT.md    # Scaling law analysis & optimal design
│   └── HYPERPARAMETER_TUNING_TODO.md      # Tuning roadmap
├── archive/                          # Obsolete docs
│   ├── MASTER_IMPLEMENTATION_PLAN.md
│   ├── PHASE3_IMPLEMENTATION_PLAN.md
│   └── PHASE1_IMPLEMENTATION_LOG.md
├── submit_gpt2m_grassmann_*.sh      # SLURM scripts
└── CLAUDE.md                         # This file (high-level overview)
```

---

## Quick Commands

**Check job status:**
```bash
squeue -u $USER
scontrol show job <JOBID>
```

**Monitor training:**
```bash
tail -f logs/603324_gpt2m_grass_gate.out
tail -f logs/603324_gpt2m_grass_gate.err
```

**Kill job if needed:**
```bash
scancel <JOBID>
```

---

## Next Steps

**Immediate (In Progress):**
1. Implement hybrid gating in BlockDecomposedLinear (block-level gates for c_fc)
2. Implement head-level gating in CausalSelfAttention (48 gates for Q/K/V heads)
3. Create config files for parallel baseline vs Grassmann comparison
4. Submit parallel training jobs with checkpoint continuation support

**After Initial Training:**
1. Monitor gate statistics (mean, sparsity, head diversity)
2. Analyze whether Grassmann advantage is maintained throughout training
3. Compare token-matched loss curves (baseline vs Grassmann)
4. Evaluate train/val gap (generalization improvement)

**If Linear Gating Insufficient:**
1. Check gate diversity: std(gates, dim=heads) should be > 0.1
2. Check gate sparsity: mean(gates) should be ≈ 0.1-0.3 (not 0.5)
3. If gates stuck or uniform: Switch to MLP gating (see GRASSMANN_GATING_ARCHITECTURE.md)

**If Scaling Needed:**
1. Try deeper model (36 layers) with lower rank (r=256) - see OPTIMAL_GRASSMANN_EXPERIMENT.md
2. Train to Chinchilla optimal (100 tokens/param) or beyond (1000+ tokens/param)
3. Consider full 16×16 block decomposition for true head independence

---

## References

**Internal:**
- `/home/junyuren/manifold_muon/` - CIFAR-10 experiments
- `archive/` - Previous plans and attempts

**External:**
- Original blog: https://thinkingmachines.ai/blog/modular-manifolds/
- Official repo: https://github.com/thinking-machines-lab/manifolds/
- nanoGPT: https://github.com/karpathy/nanoGPT
- Qwen gating: NeurIPS 2025 paper

---

**Last Updated:** December 31, 2024
**Current Phase:** Implementation of hybrid gating architecture (head-level + block-level)
**Status:** Ready to implement - see GRASSMANN_GATING_ARCHITECTURE.md for complete design
**Key Innovation:** Gating granularity matches natural structure (heads for attention, blocks for MLP)
