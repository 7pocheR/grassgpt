# Phase 2.5 Rank Matching Sweep

## Goal
Test different strategies for matching baseline weight statistics using Grassmann manifold optimization.

Phase 2.5 uses Grassmann for: **c_attn + mlp.c_fc** (skip attn.c_proj)

## Baseline Statistics (10-seed average)

### c_attn
- **Stable rank**: 24.18 ± 6.55
- **Rank@99%**: 345.22 ± 9.30 (captures 99% of energy)
- **Operator norm**: 4.6293 ± 0.8613

### mlp.c_fc
- **Stable rank**: 9.87 ± 1.40
- **Rank@99%**: 345.08 ± 1.91
- **Operator norm**: 10.1372 ± 0.7381

---

## Strategy Variants

### Variant 1: Match Stable Rank + Operator Norm ⭐ (Most Theoretically Sound)
**Rationale**: Grassmann Gr(1,0,r) has stable rank = r exactly. Direct match.

```python
# c_attn
grass_rank_c_attn = 24
scale_c_attn = 4.6293

# mlp.c_fc
grass_rank_mlp_fc = 10
scale_mlp_fc = 10.1372
```

**Expected**: Exact match of both stable rank and operator norm
**Risk**: Low - most principled approach

---

### Variant 2: Match Rank@99% + Operator Norm (Rank-Based)
**Rationale**: Rank@99% captures "effective rank" based on energy, more intuitive than stable rank.

```python
# c_attn
grass_rank_c_attn = 345  # ← WAY higher!
scale_c_attn = 4.6293

# mlp.c_fc
grass_rank_mlp_fc = 345  # ← WAY higher!
scale_mlp_fc = 10.1372
```

**Expected**: Much less regularization (high rank), may overfit
**Risk**: High - defeats purpose of low-rank constraint
**Note**: This is basically full rank (max possible is 384)

---

### Variant 3: Match Stable Rank × 2 + Operator Norm (Conservative)
**Rationale**: Account for possibility that Grassmann's effective rank is lower than stable rank suggests.

```python
# c_attn
grass_rank_c_attn = 48  # 24 × 2
scale_c_attn = 4.6293

# mlp.c_fc
grass_rank_mlp_fc = 20  # 10 × 2
scale_mlp_fc = 10.1372
```

**Expected**: Slightly less regularization than Variant 1, still reasonable
**Risk**: Low-Medium - still much lower than current rank=192

---

### Variant 4: Match Stable Rank × 3 + Operator Norm (Very Conservative)
**Rationale**: Further hedge against under-estimating needed rank.

```python
# c_attn
grass_rank_c_attn = 72  # 24 × 3
scale_c_attn = 4.6293

# mlp.c_fc
grass_rank_mlp_fc = 30  # 10 × 3
scale_mlp_fc = 10.1372
```

**Expected**: Even less regularization, closer to current approach
**Risk**: Medium - might not provide enough constraint

---

## Comparison Table

| Variant | c_attn rank | mlp.c_fc rank | Regularization | Theoretical Soundness | Expected Performance |
|---------|-------------|---------------|----------------|----------------------|---------------------|
| **Current** | 192 | 192 | Very weak | Poor (mismatch) | 1.4633 (good but 0.0037 worse) |
| **Var1: stable_rank** | 24 | 10 | **Strong** | ⭐ **Excellent** | Best gen, may beat baseline |
| **Var2: rank_99** | 345 | 345 | Very weak | Poor (near full rank) | Similar to current or worse |
| **Var3: stable_rank×2** | 48 | 20 | Medium | Good | Safe middle ground |
| **Var4: stable_rank×3** | 72 | 30 | Weak-Medium | Okay | Conservative |

---

## Predictions

### Most Likely Best: Variant 1 (Stable Rank Match)
- **Why**: Exact theoretical match, strong regularization
- **Evidence**: Current Phase 2.5 already shows better gen gap (0.317 vs 0.405)
- **Prediction**: Val loss **1.42-1.45** (match or beat baseline 1.4596)

### Safe Bet: Variant 3 (Stable Rank × 2)
- **Why**: Conservative hedge, still much better than current rank=192
- **Prediction**: Val loss **1.44-1.46** (close to baseline)

### Likely Worst: Variant 2 (Rank@99%)
- **Why**: Almost full rank (345/384), defeats purpose of manifold constraint
- **Prediction**: Val loss **1.46-1.48** (worse than current)

### Middle Ground: Variant 4 (Stable Rank × 3)
- **Why**: Still provides some constraint but quite conservative
- **Prediction**: Val loss **1.45-1.47**

---

## Implementation Notes

All variants share:
- Same optimizer settings (8× LR for Grassmann, 0.5× dropout)
- Same layers (c_attn + mlp.c_fc)
- Same training schedule (15000 iters)
- **Only difference**: Grassmann rank and scaling factors

This allows clean comparison of rank matching strategy.

---

## File Naming Convention

Configs:
- `config/phase2.5_stable_rank.py` (Variant 1)
- `config/phase2.5_rank99.py` (Variant 2)
- `config/phase2.5_stable_rank_x2.py` (Variant 3)
- `config/phase2.5_stable_rank_x3.py` (Variant 4)

Output directories:
- `out-phase2.5-stable-rank/`
- `out-phase2.5-rank99/`
- `out-phase2.5-stable-rank-x2/`
- `out-phase2.5-stable-rank-x3/`

SLURM scripts:
- `submit_phase2.5_stable_rank.sh`
- `submit_phase2.5_rank99.sh`
- `submit_phase2.5_stable_rank_x2.sh`
- `submit_phase2.5_stable_rank_x3.sh`
