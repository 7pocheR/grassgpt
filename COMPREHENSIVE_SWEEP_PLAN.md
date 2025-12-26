# Comprehensive Grassmann Hyperparameter Sweep Plan

## Current Status

**Jobs running (with operator norm scaling):**
- 561589: stable_rank (r=24,10) - best so far: 1.4877
- 561590: rank99 (r=345,345) - best so far: 1.4668
- 561591: stable_rank_x2 (r=48,20) - best so far: **1.4590** ⭐ (matches baseline!)

**Key finding**: Operator norm scaling works! All variants performing well.

---

## Issues to Address

### Issue 1: Per-Category vs Per-Layer Rank/Scale ⚠️

**What I did**: Used category averages
```python
grass_rank_c_attn = 24  # Average of [28, 24, 14, 20, 25, 35]
grass_rank_mlp_fc = 10  # Average of [11, 8, 9, 10, 11, 12]
```

**What you asked for**: Per-layer (per-block) configuration
```python
# Block 0
grass_rank_c_attn_0 = 28
grass_scale_c_attn_0 = 4.8777

# Block 1
grass_rank_c_attn_1 = 24
grass_scale_c_attn_1 = 3.3336

# ... for all 6 blocks
```

**Why this matters**:
- c_attn variance: 26.7% CV across blocks (rank 14-35, scale 3.3-5.9)
- mlp.c_fc variance: 13.2% CV (rank 8-12, scale 8.7-10.8)
- Per-block matching would be **more accurate**

**Action**: Implement per-block rank/scale configuration in model.py

---

### Issue 2: Grassmann LR and Dropout Not Tuned ⚠️

**Current fixed values**:
```python
grass_lr = 8e-3      # Inherited from CIFAR-10 (8× boost)
dropout = 0.2        # Using baseline's value
grass_dropout = ???  # Not implemented separately!
```

**Problems**:
1. 8× LR boost came from CIFAR-10, may not be optimal for transformers
2. Dropout is same for Grassmann and AdamW layers (should they differ?)
3. No systematic tuning has been done

**Hypothesis**:
- Grassmann may need different dropout (manifold constraint already regularizes)
- LR multiplier 4-6× might be better than 8×

---

## Proposed Comprehensive Sweep

### Axis 1: Rank Matching Strategy
Based on current results, best candidates:
- [x] stable_rank (exact match) - running
- [x] stable_rank_x2 (conservative) - running, **best so far!**
- [ ] **Per-block matching** (most accurate) - NOT YET DONE

### Axis 2: Grassmann Learning Rate
Test multipliers relative to AdamW's 1e-3:
- [ ] 4× = 4e-3 (conservative)
- [ ] 6× = 6e-3 (moderate)
- [x] 8× = 8e-3 (current, from CIFAR-10) - running
- [ ] 10× = 1e-2 (aggressive)

### Axis 3: Grassmann Dropout
Test different dropout for Grassmann layers:
- [ ] 0.1 (0.5× baseline, original hypothesis)
- [ ] 0.15 (0.75× baseline)
- [x] 0.2 (1.0× baseline, current) - running

---

## Immediate Next Steps

### Step 1: Let current jobs finish (~2 more hours)
- Confirm stable_rank_x2 is best rank strategy
- Use that as baseline for LR/dropout sweep

### Step 2: Implement per-block rank/scale configuration
Model.py needs to support:
```python
# Config specifies per-block
grass_ranks_c_attn = [28, 24, 14, 20, 25, 35]  # List of 6 values
grass_scales_c_attn = [4.88, 3.33, 5.88, 5.22, 4.71, 3.75]

grass_ranks_mlp_fc = [11, 8, 9, 10, 11, 12]
grass_scales_mlp_fc = [8.75, 10.59, 10.78, 10.76, 10.35, 9.59]
```

Model.py extracts per-block in configure_optimizers:
```python
for block_idx in range(n_layer):
    if 'h.{block_idx}.attn.c_attn.weight' in pn:
        r = grass_ranks_c_attn[block_idx]
        s = grass_scales_c_attn[block_idx]
```

### Step 3: LR Sweep (using best rank strategy)
Run 4 jobs with stable_rank_x2 (r=48,20) since it's currently winning:
- phase2.5_lr4x (4e-3)
- phase2.5_lr6x (6e-3)
- phase2.5_lr8x (8e-3) - baseline
- phase2.5_lr10x (1e-2)

### Step 4: Dropout Sweep
Run 3 jobs with best LR from Step 3:
- phase2.5_drop01 (0.1 for Grassmann)
- phase2.5_drop015 (0.15 for Grassmann)
- phase2.5_drop02 (0.2 for Grassmann) - baseline

### Step 5: Best Combined Config
Combine:
- Best rank strategy (likely per-block or stable_rank_x2)
- Best LR multiplier
- Best dropout
- Run final validation

---

## Time Estimate

- Current jobs: ~2 hours remaining
- Per-block implementation: 1 hour
- LR sweep (4 jobs × 6 hours): 6 hours (parallel)
- Dropout sweep (3 jobs × 6 hours): 6 hours (parallel)
- Final validation: 6 hours

**Total**: ~1 day for complete characterization

---

## Expected Outcomes

### Conservative Estimate
- Per-block matching: 1.44-1.46
- Optimal LR: 1.43-1.45
- Optimal dropout: 1.42-1.44
- **Combined**: 1.41-1.43 (clearly beat baseline 1.4596)

### Optimistic Estimate
- Grassmann's better generalization (gap 0.317 vs 0.405) + all optimizations
- **Combined**: 1.39-1.41 (significantly beat baseline)

---

## Priority Order

1. **High**: Per-block rank/scale (most accurate matching)
2. **High**: LR sweep (8× may be too aggressive)
3. **Medium**: Dropout sweep (manifold may need less)
4. **Low**: Other hyperparams (weight decay, warmup, etc.)

Should I:
1. Wait for current jobs to finish
2. Implement per-block configuration
3. Start LR sweep immediately with per-block?
