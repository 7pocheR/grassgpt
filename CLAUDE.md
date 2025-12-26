# CLAUDE.md - nanoGPT Manifold Muon Implementation

This file provides guidance to Claude Code when working with the Grassmann Muon optimizer integration in nanoGPT.

## Project Overview

This project scales the **Manifold Muon optimizer** from CIFAR-10 MLP experiments (65% test accuracy) to transformer architectures. The implementation uses **Grassmann manifold constraints** on weight matrices to improve generalization while maintaining training stability.

**Current Status:** ✅ Core implementation complete, sanity check passed, block decomposition in progress

### Why nanoGPT (not nanochat)?

**Decision:** Use nanoGPT despite being marked "deprecated"

**Rationale:**
- **"Deprecated" ≠ "broken"** - Code is stable and widely tested
- **Simpler architecture** - GPT-2 is closer to CIFAR-10 MLPs (fewer confounding variables)
- **Work already done** - Phase 0 complete with 3 critical bugs fixed
- **Architecture-agnostic research** - If Grassmann works on GPT-2, it should work on Llama3 too
- **Faster iteration** - Can start experiments immediately

**nanochat differences (Llama3 architecture):**
- RMSNorm instead of LayerNorm
- SwiGLU instead of GELU
- RoPE instead of learned positional embeddings
- Grouped Query Attention instead of standard attention

**Plan:** Validate approach on nanoGPT first, then port to nanochat if results warrant it.

---

## Cluster Workflow - The Slurm Principle

**IMPORTANT**: We work on a cluster and are usually on the login node.

### The Slurm Principle
- **Never run Python scripts (`.py` files) directly on the login node**
- Always create a shell script (`.sh`) and submit it as a SLURM job
- This applies to all training scripts, experiments, and long-running processes

### Example SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=nanogpt_grassmann
#SBATCH --output=logs/%j_experiment.out  # %j = job ID (prefix for chronological sorting)
#SBATCH --error=logs/%j_experiment.err
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:1          # Prioritize H100 for best performance
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate conda environment
source /home/junyuren/.conda/envs/manifold_muon/bin/activate

# Run training
cd /home/junyuren/nanoGPT
python train.py config/train_baby_grassmann.py
```

**Log Naming Convention:**
- Format: `logs/%j_<experiment>.{out,err}`
- Example: `logs/559444_baseline.out`, `logs/559444_baseline.err`
- **Job ID as prefix** ensures chronological sorting (job IDs increment over time)
- Standard `.out`/`.err` extensions (not `.txt`)

**⚠️ CRITICAL: Disk Quota Management**
- **Home directory (`/home/junyuren/`) has STRICT disk quota**
- **ALL checkpoints MUST be saved to `/net/scratch2/junyuren/`**
- **Config setting:** `out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-<experiment>'`
- Checkpoint files are typically ~100-200MB each
- Failure to use scratch will fill home quota and block all work!

### Cluster Configuration

**Available Partitions:**
- `dev`: 10 minute limit, A40 GPUs (for testing)
- `general`: 12 hour max, A40/A100/L40S/H100/H200 GPUs

**GPU Performance Hierarchy** (fastest to slowest):
1. **H100/H200** - 🏆 BEST: ~50-60s/epoch (Hopper architecture)
2. **A100** - Good: ~75-86s/epoch (Ampere architecture)
3. **A40** - Slower: ~100-107s/epoch (Ampere architecture)

**Recommendation**: Always request H100/H200 explicitly for training jobs.

**Environment:**
- Python 3.12.3 (via conda)
- PyTorch 2.0.1+cu118 in `manifold_muon` environment
- Activate with: `source /home/junyuren/.conda/envs/manifold_muon/bin/activate`

**GPU Resources:**
- **PRIORITY**: Use H100/H200 for best performance (35% faster than A100)
- Request format: `--gres=gpu:h100:1` or `--gres=gpu:h200:1` (preferred)
- Fallback: `--gres=gpu:a100:1` (good performance)
- Avoid: A40 GPUs (35% slower than A100)

---

## CIFAR-10 Key Findings (Transfer Learning Base)

### Proven Results

**Best Configurations:**
1. **No-skip MLP**: 64.97% test (G_{1.0, 0.0, 256}, lr=4e-3)
2. **ResNet-MLP with skip**: 60.96% test (G_{0.0, -1.0, 160}, lr=5e-4) ✅ **BREAKTHROUGH**

### Critical Insights for nanoGPT

#### 1. Skip Connections ARE Compatible! 🎉
- ✅ **Validated on CIFAR-10**: 60.96% test accuracy WITH skip connections
- ✅ **Key discovery**: Different Grassmannian parametrization required
  - ❌ G_{1.0, 0.0, r} + skip = FAILURE (45% test, loss increases)
  - ✅ **G_{0.0, -1.0, r} + skip = SUCCESS** (61% test, stable training)
- Only 4% gap vs no-skip (60.96% vs 64.97%)
- **Enables standard transformer architecture** (no modifications needed!)

#### 2. Parametrization Strategy

**G_{a, b, r}**: The Grassmann manifold {bI + cP : P rank-r projector, c = a - b}

| Scenario | Use | a | b | Why |
|----------|-----|---|---|-----|
| **With skip connections** | Layers facing skip (attn.c_proj, mlp.c_proj) | 0.0 | -1.0 | Pure projector, orthogonal to skip's identity path |
| **No skip connections** | Standalone layers (Q, K, V, mlp.c_fc) | 1.0 | 0.0 | Identity-centered, best performance |

**CRITICAL**: Using wrong (a, b) causes catastrophic failure!

#### 3. Optimal Hyperparameters (from CIFAR-10)

| Parameter | With Skip (G_{0,-1,r}) | No Skip (G_{1,0,r}) | Notes |
|-----------|----------------------|---------------------|-------|
| **Learning Rate** | **5e-4** | 4e-3 | **8× ratio** (skip requires lower LR) |
| **Rank** | **160** | 256 | Lower rank more stable with skip |
| **Dual ascent steps** | 10 | 5-10 | steps=5 is 2× faster, equivalent accuracy |
| **Dual ascent alpha** | 0.01 | 0.01 | Default, worked well |
| **Tolerance** | 1e-6 | 1e-6 | Default |
| **Weight decay** | 5e-4 | 5e-4 | Optimal from sweep |
| **Dropout** | 0.05 | 0.05 | 0.0 failed, 0.10 hurt |
| **Warmup** | 5 epochs | 5 epochs | Can increase for safety |

#### 4. Rank Selection
- ✅ **Rank 128-256 all work equivalently** (~64-65% without skip)
- Lower ranks (128-160): 15-20% faster training, hint of better generalization
- **CIFAR-10 optimal:** rank=160 for width=512 → **31% of width**
- Higher ranks (384+): significant degradation (-4% to -8%)
- **nanoGPT default: 50% of width (n/2)** for simplicity
  - Example: rank=192 for n_embd=384
  - More aggressive than CIFAR-10, but easier to implement uniformly

#### 5. Manifold Constraint >> Low-Rank Structure
- Low-rank W=UV (AdamW): 60.83% test
- Grassmann (manifold-constrained low-rank): 64.97% test
- **+4.14% improvement from manifold constraint**

#### 6. Excellent Generalization
- ResNet-MLP (G_{0.0, -1.0, 160}): 4.64% train/test gap
- No-skip (G_{1.0, 0.0, 256}): 4.39% gap
- AdamW baseline: 34.87% gap
- **~7× better generalization with Grassmann**

---

## nanoGPT Architecture Analysis

### Current Implementation Status

**✅ PHASE 0 COMPLETE (8% coverage):**
- Only `attn.c_proj.weight` uses Grassmann Muon (~8% of parameters)
- Q, K, V: rectangular (n_embd, 3×n_embd) combined in c_attn → AdamW
- MLP: rectangular (n_embd, 4×n_embd expansion) → AdamW

**🚧 PHASES 1-5 IN PROGRESS (up to 82% coverage):**
- **Block decomposition strategy** enables Grassmann on Q, K, V, MLP
- See `BLOCK_DECOMPOSITION_PLAN.md` for full specification

### Weight Matrix Inventory

| Layer | Shape | Current Optimizer | Block Decomposition Plan |
|-------|-------|------------------|-------------------------|
| **Attention** |
| c_attn (Q,K,V combined) | (3×n_embd, n_embd) | AdamW | ✅ Split into 3 blocks of (n_embd, n_embd) |
| c_proj (output) | (n_embd, n_embd) | ✅ Grassmann | ✅ Already square |
| **MLP** |
| c_fc (expand) | (4×n_embd, n_embd) | AdamW | ✅ Split into 4 blocks of (n_embd, n_embd) |
| c_proj (contract) | (n_embd, 4×n_embd) | AdamW | ✅ Split into 4 blocks of (n_embd, n_embd) |

### Skip Connection Architecture

```
Input (d_model)
  ↓
LayerNorm → Attention (Q, K, V, proj) → Residual Add  ← SKIP HERE
  ↓
LayerNorm → MLP (c_fc, c_proj) → Residual Add  ← SKIP HERE
  ↓
Output (d_model)
```

**Key Insight:**
- Skip connections exist at **Block level** (around entire Attention, around entire MLP)
- But only **final projections** (attn.c_proj, mlp.c_proj) directly output to skip
- **Q, K, V have NO direct skip connection** (only indirect via Block-level skip)

---

## Block Decomposition Strategy

### Mathematical Foundation

**Problem:** Rectangular matrices (N, kN) or (kN, N) are not square, cannot use Grassmann directly

**Solution:** Decompose into k square (N, N) blocks with appropriate scaling

#### CRITICAL: Grassmann Projectors Are NOT Norm-Preserving!

**For rank r = n/2 projectors:** ||Px|| ≈ √(r/n) · ||x|| = (1/√2) · ||x|| ≈ **0.707σ**

This is 30% smaller than input - it's REGULARIZATION, not a bug!

#### Case 1: Vertical Stacking (kN, N) - Split Immediately (c_attn)

```
W = [W_q]  where each Wᵢ ∈ ℝ^(N×N), rank r=n/2
    [W_k]
    [W_v]

Forward: x → [P_q·x; P_k·x; P_v·x] → IMMEDIATELY SPLIT → q, k, v
Concatenated (1 line of code): ||[q;k;v]|| = √3 · (σ/√2)
Components (actually used): ||q|| = ||k|| = ||v|| = σ/√2
```

**NO SCALING NEEDED!**
- Concatenated norm is temporary, irrelevant (exists for 1 line)
- Per-component norms (σ/√2) are what matter for attention
- Adding 1/√3 would make queries/keys √3× smaller → overly flat attention

#### Case 2: Vertical Stacking (kN, N) - Full Vector Used (mlp.c_fc)

```
W = [W₁]  where each Wᵢ ∈ ℝ^(N×N), rank r=n/2
    [W₂]
    [W₃]
    [W₄]

Forward: x → [P₁x; P₂x; P₃x; P₄x] → GELU (full 4n vector)
Each block: ||Pᵢx|| = σ/√2
Stacked: ||output|| = √4 · (σ/√2) = √2·σ ≈ 1.41σ (41% too large!)
```

**NEEDS 0.5× SCALING!**
- Full vector used by GELU, needs consistent activation magnitudes
- Apply 0.5×I: 0.5 · √2·σ = σ/√2 (matches network's natural Grassmann scale)

#### Case 3: Horizontal Concat (N, kN) - mlp.c_proj

**NO SCALING NEEDED!** Horizontal decomposition naturally balances input splitting vs output summation.

### Parametrization by Layer

**Different layers need different Grassmannian parametrizations:**

| Layer | Shape | Blocks | Scaling | Parametrization | Reason |
|-------|-------|--------|---------|----------------|--------|
| **c_attn (Q,K,V)** | (3n, n_embd) | 3×(n,n) | **None** | **G_{1.0, 0.0, r}** | Components split immediately, per-head norms matter |
| **attn.c_proj** | (n_embd, n_embd) | 1×(n,n) | None | **G_{0.0, -1.0, r}** | Square matrix, faces skip |
| **mlp.c_fc** | (4n, n_embd) | 4×(n,n) | **0.5** (frozen) | **G_{1.0, 0.0, r}** | Full 4n vector used, need √2·σ → σ/√2 |
| **mlp.c_proj** | (n_embd, 4n) | 4×(n,n) | **None** | **G_{0.0, -1.0, r}** | Horizontal naturally balances |

### Learning Rate Strategy

**From CIFAR-10: 8× LR ratio (Grassmann no-skip : AdamW)**
- Grassmann no-skip (G_{1.0, 0.0, r}): lr = 4e-3
- AdamW ResNet baseline: lr = 5e-4
- Ratio: 4e-3 / 5e-4 = **8×**

**For nanoGPT (recommended at our parameter count ~10M):**
- AdamW baseline: 1e-3 (from nanoGPT scaling laws)
- Grassmann no-skip (Q, K, V, mlp.c_fc): **8e-3** (8× boost)
- Grassmann skip-compatible (attn.c_proj, mlp.c_proj): **1e-3** (same as AdamW)

**Rationale:**
- Layers NOT facing skip can use aggressive LR (no gradient amplification risk)
- Layers facing skip need conservative LR (proven stable at 5e-4 on CIFAR-10)

---

## Experimental Plan: 5 Phases + Baseline

See `BLOCK_DECOMPOSITION_PLAN.md` for full implementation details.

### Phase Coverage Summary

| Phase | Description | Grassmann Params | Coverage | Parametrization | Expected LR |
|-------|-------------|------------------|----------|----------------|-------------|
| **Baseline** | AdamW only | 0 | 0% | N/A | 1e-3 |
| **Phase 0** | Current (c_proj only) | c_proj | 8% | G_{0,-1,r} | 1e-3 |
| **Phase 1** | Add QKV | QKV + c_proj | 28% | Mixed | Q,K,V: 8e-3<br>c_proj: 1e-3 |
| **Phase 2** | Add MLP.c_fc only | c_fc + c_proj | 34% | Mixed | c_fc: 8e-3<br>c_proj: 1e-3 |
| **Phase 3** | QKV + MLP.c_fc | All except mlp.c_proj | 62% | Mixed | No-skip: 8e-3<br>c_proj: 1e-3 |
| **Phase 4** | Full with skip | All weights | 62% | All G_{0,-1,r} | 1e-3 (uniform) |
| **Phase 5** | Full no-skip | All weights | 82% | All G_{1,0,r} | 8e-3 (uniform) |

**Key Differences:**
- **Phase 4 vs 5:** Same coverage, different parametrization and LR
  - Phase 4: Skip-compatible (G_{0,-1,r}), conservative LR
  - Phase 5: Maximum performance (G_{1,0,r}), aggressive LR, **no skip connections**

---

## Implementation Status

### ✅ Completed (Phase 0: 8% coverage)

**Files Modified:**
1. `/home/junyuren/nanoGPT/model.py` (lines 18-19, 173-199, 292-373)
   - Added `configure_optimizers_grassmann()` method
   - Added `init_grassmann_weights()` method
   - Hybrid optimizer: Grassmann for square attn.c_proj, AdamW for rest

2. `/home/junyuren/nanoGPT/train.py` (lines 31, 64-72, 209-228, 336-376)
   - Added Grassmann config parameters
   - Modified optimizer initialization to support Grassmann
   - Modified training loop to apply Grassmann updates after AdamW step
   - ✅ **BUG FIXES APPLIED:**
     - Fixed gradient zeroing: `model.zero_grad()` instead of `optimizer.zero_grad()`
     - Fixed float16 scaling: Manual unscaling for Grassmann gradients
     - Fixed checkpoint resume: Only init weights on 'scratch', not 'resume'

3. `/home/junyuren/nanoGPT/manifold/` directory
   - `grassmann_muon.py`: Dual ascent optimizer
   - `grassmann_ops.py`: Manifold retraction operations
   - `msign.py`: Polar-Express matrix sign function
   - `__init__.py`: Clean API exports

**Config Files:**
- `config/train_baby_baseline.py`: AdamW baseline (Baby GPT)
- `config/train_baby_grassmann.py`: Grassmann Phase 0 (Baby GPT)
- `config/test_grassmann_sanity.py`: Quick sanity check
- `config/test_float16.py`: Float16 bug fix validation

**Sanity Check Results (Job 557184):**
- ✅ Loss decreased monotonically (4.21 → 3.78)
- ✅ Grassmann weights initialized correctly (2 layers, 8K params)
- ✅ No errors or crashes
- ⚠️ Only 8% coverage (c_proj only)

### 🚧 In Progress (Phases 1-5: Block Decomposition)

**Next Steps (from BLOCK_DECOMPOSITION_PLAN.md):**
1. Implement `block_decomposition.py` module
   - `BlockDecomposedLinear` class for rectangular layers
   - Support for both (N, kN) and (kN, N) cases
   - Appropriate scaling factors (1.0 or 1/√k)

2. Modify `model.py` for block decomposition
   - Replace c_attn with BlockDecomposedLinear(n, 3n, k=3)
   - Replace mlp.c_fc with BlockDecomposedLinear(n, 4n, k=4)
   - Replace mlp.c_proj with BlockDecomposedLinear(4n, n, k=4)
   - Update `configure_optimizers_grassmann()` to handle block params

3. Modify `train.py` for block updates
   - Loop through block components in Grassmann update
   - Apply correct (a, b) based on layer type
   - Apply differential LRs (8× boost for no-skip layers)

4. Create 5 phase config files
   - `config/train_phase1_qkv.py`
   - `config/train_phase2_mlp_fc.py`
   - `config/train_phase3_qkv_mlp_fc.py`
   - `config/train_phase4_full_skip.py`
   - `config/train_phase5_full_noskip.py`

5. Run unit tests and sanity checks
   - Verify block shapes match original
   - Verify scaling preserves gradient norms
   - Quick 10-iter sanity check per phase

6. Create SLURM submission scripts and run experiments

---

## Key Concepts

### Grassmann Manifold G_{a,b,r}

**Definition:** {bI + cP : P is rank-r projector, c = a - b}

**Parametrizations:**
- **G_{1.0, 0.0, r}**: Identity-centered, W = I + P (eigenvalues: 1 or 0)
  - Use for layers WITHOUT skip connections
  - Best performance on CIFAR-10 (64.97%)
  - Can use aggressive LR (8× boost)

- **G_{0.0, -1.0, r}**: Pure projector, W = P - I (eigenvalues: 0 or -1)
  - Use for layers facing skip connections
  - Stable on CIFAR-10 with skip (60.96%)
  - Requires conservative LR (same as AdamW)

### Manifold Operations

**Initialization:** `initialize_on_grassmann(W, a, b, rank)`
- Projects weight matrix W onto G_{a,b,r} manifold
- Used once at training start (or when resuming from 'scratch')

**Update:** `grassmann_muon_update(W, G, eta, a, b, r, alpha, steps, tol)`
- W: Current weight matrix
- G: Gradient
- eta: Step size (learning rate)
- a, b: Grassmannian eigenvalues
- r: Rank
- alpha: Dual ascent step size (default 0.01)
- steps: Max dual ascent iterations (default 10)
- tol: Convergence tolerance (default 1e-6)

**Algorithm:**
1. Initializes dual variable Λ
2. Iteratively solves for update direction A via dual ascent
3. Ensures A is in tangent space of Grassmann manifold
4. Applies update and retracts back to manifold using matrix sign function

**Retraction:** Uses Polar-Express algorithm (msign.py) for fast matrix sign computation

---

## Training Workflow

### Standard Training Command

```bash
# Submit SLURM job (ALWAYS use this on cluster!)
sbatch submit_baby_grassmann.sh

# Monitor job
squeue -u $USER
tail -f logs/JOBID_out.txt
```

### Config File Structure

All experiments use Python config files in `config/` directory:

```python
# Example: config/train_baby_grassmann.py

# Model architecture
n_layer = 6
n_head = 6
n_embd = 384

# Training
learning_rate = 1e-3
max_iters = 5000
batch_size = 64

# Grassmann Muon configuration
use_grassmann = True
grass_lr = 1e-3        # For skip-compatible layers
grass_a = 0.0          # First eigenvalue
grass_b = -1.0         # Second eigenvalue
grass_rank = 192       # 50% of n_embd (n/2)
grass_alpha = 0.01     # Dual ascent step
grass_steps = 10       # Dual ascent iterations
grass_tol = 1e-6       # Convergence tolerance

# Other hyperparameters
dropout = 0.05         # Optimal from CIFAR-10
weight_decay = 5e-4    # Optimal from CIFAR-10
grad_clip = 1.0        # Standard
```

### Hyperparameter Guidelines

**From nanoGPT scaling laws (at ~10M params):**
- Base LR: 1e-3 (AdamW)
- Batch size: 64-256
- Warmup: 5-10 epochs
- Decay: Cosine to 0.1 × base_lr

**From CIFAR-10 experiments:**
- Grassmann no-skip: 8× LR boost (8e-3 for nanoGPT)
- Grassmann skip-compatible: Same as AdamW (1e-3)
- Rank: 50% of width, n/2 (192 for n_embd=384)
- Weight decay: 5e-4 (CIFAR-10), 1e-1 (Shakespeare baseline)
- Dropout: 0.05 (half of AdamW's optimal 0.10)

**For Shakespeare (component-specific hyperparameters):**
- Baseline AdamW: dropout = 0.2 (nanoGPT standard)
- **Hybrid optimizer uses component-specific dropout:**
  - **Grassmann components** (c_attn, c_proj with Grassmann): dropout = 0.1 (half of baseline)
  - **AdamW components** (embedding, layernorm, non-Grassmann layers): dropout = 0.2 (match baseline)
- Implementation: `setup_component_dropout()` sets dropout.p based on grassmann_configs
- Config: Set `grassmann_dropout = 0.1` or leave None for automatic dropout/2

**Critical rules:**
- ⚠️ NEVER use G_{1.0, 0.0, r} with skip connections → catastrophic failure
- ⚠️ ALWAYS use G_{0.0, -1.0, r} for skip-compatible layers
- ⚠️ Use 8× LR only for layers NOT facing skip connections

---

## Debugging and Monitoring

### Sanity Checks Before Full Training

**1. Quick 10-iteration test:**
```python
# config/test_sanity.py
n_layer = 2
n_embd = 64
max_iters = 10
log_interval = 1
```

**Expected behavior:**
- Loss decreases (even if slowly)
- No NaN or Inf
- Grassmann weights initialized correctly
- No crashes

**2. Float16 test (if using dtype='float16'):**
```python
# config/test_float16.py
dtype = 'float16'
max_iters = 20
```

**Expected behavior:**
- Gradients unscaled correctly
- Loss decreases (no gradient scaling issues)

### Common Issues and Fixes

**Issue 1: Loss increases or plateaus early**
- **Cause:** Wrong parametrization (G_{1,0,r} with skip) or LR too high
- **Fix:** Check (a, b) values, reduce LR, increase warmup
- **Prevention:** Follow parametrization table strictly

**Issue 2: Gradients explode/vanish**
- **Cause:** Block scaling wrong, LR too high, or skip interaction
- **Fix:** Verify scaling factors (1.0 vs 1/√k), reduce LR
- **Prevention:** Run unit tests on BlockDecomposedLinear

**Issue 3: Grassmann gradients not zeroed**
- **Cause:** Using `optimizer.zero_grad()` instead of `model.zero_grad()`
- **Fix:** ✅ Already fixed in train.py:376
- **Prevention:** Always use `model.zero_grad(set_to_none=True)`

**Issue 4: Float16 instability**
- **Cause:** Grassmann gradients not unscaled
- **Fix:** ✅ Already fixed in train.py:340-343
- **Prevention:** Use bfloat16 (default) or apply manual unscaling

**Issue 5: Checkpoint resume destroys learned weights**
- **Cause:** `init_grassmann_weights()` called on resume
- **Fix:** ✅ Already fixed in train.py:224 (conditional on init_from=='scratch')
- **Prevention:** Always check init_from before manifold initialization

---

## Performance Expectations

### Based on CIFAR-10 Transfer

**Conservative estimate (Phase 1-4):**
- Training stability: ✅ High confidence (validated on CIFAR-10)
- Generalization improvement: ~2-4× better train/val gap
- Absolute performance: Within ±5% of AdamW baseline

**Optimistic estimate (Phase 5, if skip removal works):**
- Generalization improvement: ~7× better train/val gap (like CIFAR-10)
- Absolute performance: +5-10% vs baseline (perplexity reduction)
- Training speed: 15-20% faster (lower rank, fewer params to optimize)

**Risks:**
- ⚠️ Language modeling may need higher rank than vision (test 128-320)
- ⚠️ Deep transformers may be more sensitive to skip removal than MLP
- ⚠️ Block decomposition adds complexity (more hyperparams to tune)

---

## Reference Material

### Internal Documentation
- `BLOCK_DECOMPOSITION_PLAN.md`: Full implementation specification for Phases 1-5
- `NANOGPT_SCALING_PLAN.md`: Original planning document with CIFAR-10 analysis
- `/home/junyuren/manifold_muon/CLAUDE.md`: Cluster workflow reference
- `/home/junyuren/manifold_muon/README.md`: CIFAR-10 best results summary
- `/home/junyuren/manifold_muon/HYPERPARAMETER_TUNING_RESULTS.md`: Full sweep details
- `/home/junyuren/manifold_muon/ABLATION_EXPERIMENTS.md`: Rank and skip connection analysis

### Key Scripts
- `/home/junyuren/nanoGPT/manifold/grassmann_muon.py`: Optimizer implementation
- `/home/junyuren/nanoGPT/manifold/grassmann_ops.py`: Manifold operations
- `/home/junyuren/nanoGPT/manifold/msign.py`: Matrix sign function
- `/home/junyuren/nanoGPT/model.py`: Model with Grassmann integration
- `/home/junyuren/nanoGPT/train.py`: Training loop with hybrid optimizer

### External
- Original blog post: https://thinkingmachines.ai/blog/modular-manifolds/
- Official repository: https://github.com/thinking-machines-lab/manifolds/
- nanoGPT: https://github.com/karpathy/nanoGPT
- Polar-Express paper: https://arxiv.org/abs/2505.16932

---

## Quick Reference Tables

### Parametrization Lookup

| Layer Type | Skip Connection? | Use a | Use b | Use LR |
|------------|-----------------|-------|-------|--------|
| Q, K, V (c_attn blocks) | ❌ No | 1.0 | 0.0 | 8e-3 (8×) |
| Attention proj (c_proj) | ✅ Yes | 0.0 | -1.0 | 1e-3 (1×) |
| MLP expand (c_fc blocks) | ❌ No | 1.0 | 0.0 | 8e-3 (8×) |
| MLP contract (c_proj blocks) | ✅ Yes | 0.0 | -1.0 | 1e-3 (1×) |

### Scaling Factor Lookup

| Layer | Shape | Decomposition | Frozen Scale | Output Norm | Why |
|-------|-------|---------------|--------------|-------------|-----|
| c_attn | (3n, n) | 3 blocks (n,n) vertical | **None** | σ/√2 per component | Components split immediately, only per-head norms matter |
| attn.c_proj | (n, n) | No decomposition | None | σ/√2 | Square, rank=n/2 naturally gives σ/√2 |
| mlp.c_fc | (4n, n) | 4 blocks (n,n) vertical | **0.5×I** | σ/√2 | Full 4n vector used, 0.5 × √2·σ = σ/√2 |
| mlp.c_proj | (n, 4n) | 4 blocks (n,n) horizontal | **None** | σ/√2 | Horizontal naturally balances |

**Key insight:** All Grassmann outputs have norm ≈ σ/√2 (inherent to rank=n/2). This is REGULARIZATION!

### Rank Selection

| n_embd | Rank (50%) - nanoGPT default | Rank (31%) - CIFAR-10 optimal |
|--------|------------------------------|-------------------------------|
| 64 | 32 | 20 |
| 128 | 64 | 40 |
| 384 | 192 | 120 |
| 512 | 256 | 160 |
| 768 | 384 | 240 |

**Recommendation:** Start with 50% (n/2) uniformly for all layers
- CIFAR-10 optimal was 31% (160/512), but 50% is simpler to implement
- More aggressive rank allows more expressiveness
- Can ablate to 31% if needed for speed/memory

---

## Implementation Checklist

### Phase 0 (COMPLETED ✅)
- [x] Port manifold code to nanoGPT
- [x] Integrate hybrid optimizer (Grassmann + AdamW)
- [x] Fix gradient zeroing bug
- [x] Fix float16 scaling bug
- [x] Fix checkpoint resume bug
- [x] Run sanity check (Job 557184)
- [x] Create baseline and Grassmann configs

### Phases 1-5 (IN PROGRESS 🚧)
- [ ] Implement `block_decomposition.py` module
- [ ] Modify `model.py` for block decomposition
- [ ] Modify `train.py` for block updates
- [ ] Create 5 phase config files
- [ ] Run unit tests and sanity checks
- [ ] Create SLURM submission scripts
- [ ] Submit all 6 experiments (baseline + 5 phases)
- [ ] Analyze results and compare phases
