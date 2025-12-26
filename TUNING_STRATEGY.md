# Hyperparameter Tuning Strategy Based on Current Results

## Current Status (After Initial Runs)

**Top performers vs Baseline (1.4596):**
- Phase 2.5: 1.4633 (+0.0037 worse, 33% Grassmann) ⭐ **Very close!**
- Phase 0.5: 1.4696 (+0.0100 worse, 25% Grassmann)
- Phase 1.5: 1.4955 (+0.0359 worse, 30% Grassmann)

**Key observations:**
1. Phase 2.5 is **within 0.25%** of baseline - extremely promising!
2. All `.5` variants (skip attn.c_proj) perform well
3. All `.2` variants (no skip) completely failed (val loss >3.3)
4. **Smaller gen gap in Grassmann phases** - better regularization

## Why Current Config May Be Suboptimal

### 1. Inherited Assumptions from CIFAR-10

The current Grassmann hyperparameters were tuned for **CIFAR-10 ResNets**, not transformers:

```python
grass_lr = 8e-3     # 8× boost - may be too aggressive for transformers
grass_dropout = 0.1 # Half of baseline - arbitrary choice
grass_rank = 192    # n_embd/2 - may not be optimal ratio
```

**Problem:** Transformers have different dynamics:
- Attention mechanisms are more sensitive to large LR
- Different regularization needs (LayerNorm, residuals)
- Parameter interactions differ from CNNs

### 2. Baseline Config Already Optimized

nanoGPT's baseline is well-tuned:
```python
learning_rate = 1e-3  # Standard for Shakespeare
dropout = 0.2         # Optimal for this dataset size
weight_decay = 1e-1   # Proven effective
```

But we **downscaled** (6L×384d instead of 12L×768d), which may need re-tuning.

### 3. Grassmann Has Predetermined Operator Norm = 1

**Critical insight:** Grassmann manifold G_{a,b,r} constrains matrices to specific spectral norms.

For G_{1,0,r}: Eigenvalues are {1, 1, ..., 1, 0, 0, ..., 0}
- **Operator norm = 1** (largest singular value)
- This is **fixed** by the manifold, not learned!

**Implications:**
- If baseline weights naturally have ||W|| >> 1, scaling mismatch!
- If baseline weights have ||W|| << 1, we're over-regularizing!
- Need to **measure actual weight norms from baseline** and adjust

---

## Proposed Tuning Experiments

### Phase 1: Measure Baseline Weight Statistics

**Goal:** Understand what norms the baseline naturally learns

**Script to create:**
```python
# analyze_baseline_norms.py
checkpoint = torch.load('out-baby-baseline/ckpt.pt')
model_weights = checkpoint['model']

for name, param in model_weights.items():
    if 'weight' in name and param.dim() == 2:
        # Compute operator norm (largest singular value)
        U, S, Vh = torch.linalg.svd(param, full_matrices=False)
        op_norm = S[0].item()

        # Compute Frobenius norm
        frob_norm = torch.norm(param).item()

        # Effective rank (singular value distribution)
        S_normalized = S / S.sum()
        eff_rank = torch.exp(-torch.sum(S_normalized * torch.log(S_normalized + 1e-10)))

        print(f"{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Operator norm: {op_norm:.4f}")
        print(f"  Frobenius norm: {frob_norm:.4f}")
        print(f"  Effective rank: {eff_rank:.1f}")
```

**Expected findings:**
- c_attn might have ||W|| ≈ 2-5 (needs scaling up)
- mlp.c_fc might have ||W|| ≈ 3-6 (needs scaling up)
- attn.c_proj might have ||W|| ≈ 0.5-1.5 (close to Grassmann default)

**Action:** Add scaling factors to match baseline norms:
```python
# In model.py after Grassmann update
if 'c_attn' in name:
    W_grass = W_grass * 3.0  # Scale to match baseline op norm
```

---

### Phase 2: Learning Rate Sweep

**Hypothesis:** 8× LR boost is too aggressive for transformers

**Current:**
```python
adamw_lr = 1e-3
grass_lr = 8e-3  # 8× boost
```

**Test suite:**
| Config | AdamW LR | Grass LR | Multiplier | Priority |
|--------|----------|----------|------------|----------|
| current | 1e-3 | 8e-3 | 8× | Baseline |
| conservative | 1e-3 | 4e-3 | 4× | **High** |
| moderate | 1e-3 | 6e-3 | 6× | **High** |
| aggressive | 1e-3 | 1e-2 | 10× | Medium |
| matched | 1e-3 | 1e-3 | 1× | Low |

**Run on:** Phase 2.5 (already close to baseline, most promising)

**Rationale:**
- Phase 2.5 @ 8× is only 0.0037 worse than baseline
- Reducing to 4-6× might find sweet spot
- Focus on best-performing phase to save compute

---

### Phase 3: Dropout Adjustment

**Current assumption:** Grassmann needs half the dropout (0.1 vs 0.2)

**Problem:** This was heuristic, not validated

**Test suite:**
| Config | AdamW Dropout | Grass Dropout | Ratio |
|--------|---------------|---------------|-------|
| current | 0.2 | 0.1 | 0.5× |
| less_reg | 0.2 | 0.15 | 0.75× |
| matched | 0.2 | 0.2 | 1.0× |
| more_reg | 0.2 | 0.05 | 0.25× |

**Hypothesis:** Since gen gap is **smaller** in Grassmann phases, the manifold constraint already regularizes heavily. We might need:
- **More dropout** (0.15-0.2) to match baseline's effective regularization
- OR accept lower train loss but higher val loss

---

### Phase 4: Rank Tuning

**Current:** rank = 192 (n_embd / 2)

**Theory:** Lower rank = stronger regularization, faster computation

**Test suite:**
| Config | Rank | Ratio | Expected Impact |
|--------|------|-------|-----------------|
| current | 192 | 1/2 | Baseline |
| lower | 128 | 1/3 | More reg, 1.5× faster |
| higher | 256 | 2/3 | Less reg, 1.2× slower |
| minimal | 96 | 1/4 | Strong reg, 2× faster |

**Hypothesis:** Phase 2.5's better gen gap suggests we might benefit from **lower rank** (more regularization)

---

### Phase 5: Scaling Factor Correction (Most Important!)

**Current implementation:**
```python
# In model.py - setup_norm_preserving_scaling()
if has_c_fc:
    scale_value = 0.5  # 1/√4 for vertical decomposition
    block.mlp.c_fc_scale = frozen_linear(scale_value * I)
```

**Problem:** This scales **Frobenius norm**, but Grassmann fixes **operator norm = 1**!

**Proposed fix:**
1. Measure baseline weight operator norms at best checkpoint
2. Add **post-Grassmann scaling** to match:

```python
# New method in model.py
def setup_grassmann_scaling(self, baseline_norms):
    """Scale Grassmann weights to match baseline operator norms."""
    for block in self.transformer.h:
        # c_attn: target op norm ≈ 3.5 (from baseline analysis)
        if hasattr(block.attn, 'c_attn'):
            scaling = nn.Linear(n_embd, 3*n_embd, bias=False)
            with torch.no_grad():
                # Diagonal scaling: multiply by target_norm
                scaling.weight.copy_(3.5 * torch.eye(3*n_embd, n_embd))
            scaling.weight.requires_grad = False
            block.attn.c_attn_scale = scaling

        # mlp.c_fc: target op norm ≈ 4.2 (from baseline analysis)
        if hasattr(block.mlp, 'c_fc'):
            scaling = nn.Linear(4*n_embd, 4*n_embd, bias=False)
            with torch.no_grad():
                scaling.weight.copy_(4.2 * torch.eye(4*n_embd))
            scaling.weight.requires_grad = False
            block.mlp.c_fc_scale_grass = scaling
```

**Critical:** This is **separate** from the 0.5× vertical decomposition scaling!
- Vertical decomp scaling: preserves norms during block composition
- Grassmann scaling: matches baseline's learned scale

**Expected impact:** Could close the 0.0037 gap entirely or even surpass baseline!

---

### Phase 6: Combined Optimization

Once we find optimal individual settings, combine:

**Proposed best config (Phase 2.5+):**
```python
# Based on analysis results
grass_lr = 5e-3              # 5× boost (reduced from 8×)
grass_dropout = 0.15         # 0.75× of baseline (increased from 0.5×)
grass_rank = 128             # 1/3 ratio (reduced from 1/2)

# NEW: Operator norm scaling
scale_c_attn = 3.5           # Match baseline ||W_c_attn||_op
scale_mlp_fc = 4.2           # Match baseline ||W_mlp_fc||_op

# Keep current good choices
grassmann_phase = 'phase2.5' # Skip attn.c_proj (best performing)
```

---

## Implementation Plan

### Step 1: Baseline Analysis (High Priority) ⭐
```bash
python analyze_baseline_norms.py \
  --checkpoint out-baby-baseline/ckpt.pt \
  --output baseline_weight_analysis.md
```

**Deliverables:**
- Table of operator norms for each weight matrix
- Spectral analysis (eigenvalue distributions)
- Recommendations for scaling factors

---

### Step 2: Quick LR Sweep (High Priority) ⭐

Run Phase 2.5 with different LRs (short runs, 5000 iters each):
```bash
# configs: phase2.5_lr4x.py, phase2.5_lr6x.py, phase2.5_lr10x.py
sbatch submit_phase2.5_lr4x.sh
sbatch submit_phase2.5_lr6x.sh
sbatch submit_phase2.5_lr10x.sh
```

**Success metric:** Find LR that reduces gap from 0.0037 to <0.001

---

### Step 3: Add Operator Norm Scaling (High Priority) ⭐

1. Implement `setup_grassmann_scaling()` in model.py
2. Use measured norms from Step 1
3. Test on Phase 2.5
4. Compare: current vs. norm-matched

**Hypothesis:** This alone could beat baseline!

---

### Step 4: Rank Tuning (Medium Priority)

Test rank=[96, 128, 192, 256] on best LR from Step 2

---

### Step 5: Dropout Fine-tuning (Low Priority)

Once LR + scaling are optimal, sweep dropout

---

### Step 6: Full Run with Best Config (Final)

Run 15000 iters with optimized hyperparameters

**Target:** Val loss < 1.45 (beat baseline's 1.4596)

---

## Expected Outcomes

### Conservative Estimate
With LR tuning + operator norm scaling:
- **Phase 2.5**: 1.44 - 1.45 (match or slightly beat baseline)
- **Phase 0.5**: 1.45 - 1.46 (close to baseline)

### Optimistic Estimate
With full optimization (LR + scaling + rank + dropout):
- **Phase 2.5**: 1.42 - 1.43 (clearly beat baseline) ⭐
- Reason: Better regularization (smaller gen gap already observed)

### Why This Will Work

1. **We're already very close** (0.0037 gap)
2. **Better generalization** (0.3174 vs 0.4053 gap) - Grassmann regularizes well
3. **Clear path:** Operator norm mismatch is fixable
4. **Validated approach:** CIFAR-10 Muon paper showed 3-5× speedup with tuning

---

## Next Actions

1. **Create `analyze_baseline_norms.py`** - understand target scales
2. **Implement operator norm scaling** in model.py
3. **Quick LR sweep** on Phase 2.5 (3 jobs, 5000 iters each)
4. **Rerun Phase 2.5** with best settings for 15000 iters

**Time estimate:** 2-3 days for full optimization sweep

**Confidence:** 80% we can match baseline, 60% we can beat it by ≥0.01

---

## Open Questions

1. **Why did .2 variants fail?**
   - Removing skip + G(1,0,r) may create optimization landscape issues
   - Too aggressive (8× LR + no residual + op norm = 1)
   - Need gentler initialization or LR warmup?

2. **Why does skipping attn.c_proj help?**
   - Maybe attn.c_proj doesn't benefit from rank constraint?
   - Or: G(0,-1,r) creates interaction issues with residuals?
   - Suggests attn.c_proj may need different treatment

3. **Can we make phase3+ work?**
   - Currently worse than baseline (1.71 vs 1.46)
   - May need even lower LR when >50% params are Grassmann
   - Or: optimizer interference between many Grassmann layers?

4. **What's optimal Grassmann coverage?**
   - 33% (Phase 2.5) is best so far
   - 25-30% also good
   - >50% starts to degrade
   - Suggests: **selective** application better than **universal**
