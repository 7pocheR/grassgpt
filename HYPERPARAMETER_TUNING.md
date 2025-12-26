# Hyperparameter Tuning Plan

## Current Baseline Configuration

```yaml
AdamW (non-Grassmann):
  learning_rate: 1e-3
  weight_decay: 1e-1
  beta1: 0.9
  beta2: 0.99
  dropout: 0.2

Grassmann (manifold layers):
  grass_lr: 8e-3          # 8× boost over AdamW
  grass_dropout: 0.1      # Half of AdamW dropout
  grass_rank: 192         # n_embd / 2 = 384 / 2
  grass_a: 1.0            # For G(1,0,r) - no skip
  grass_b: 0.0            # For G(1,0,r) - no skip
  grass_alpha: 0.01       # Dual ascent step size
  grass_steps: 10         # Dual ascent iterations
  grass_tol: 1e-6         # Convergence tolerance
  msign_steps: 10         # Polar Express iterations (hardcoded in msign.py)
```

## Performance Issues

### Current Bottleneck: MFU Analysis
- **Phase 0.5** (c_attn only): MFU = 24.51% ✓ Acceptable
- **Phase 1.5** (mlp.c_fc only): MFU = 0.89% ✗ Very slow!
- **Baseline** (AdamW): MFU = ~30-40% (expected)

### Root Cause
Each Grassmann weight update requires:
- **10 dual ascent steps** (sequential, not parallelizable)
- **10 msign steps per dual ascent** (also sequential)
- **Total: ~100 matrix operations** per weight update
- Compare to AdamW: 1-2 operations per weight update

**Result:** 6 hours for 15000 iterations (phases with multiple Grassmann layers)

---

## Tuning Plan A: Speed Optimizations

### 1. Reduce Dual Ascent Steps
**Current:** `grass_steps = 10`

**Test:**
```yaml
grass_steps: 5   # 2× speedup
grass_steps: 3   # 3.3× speedup
grass_steps: 1   # 10× speedup (likely too aggressive)
```

**Rationale:** Manifold projection doesn't need to be perfect every iteration. Approximate projection may be sufficient for optimization.

**Expected impact:**
- Steps=5: 2× faster, minimal accuracy loss
- Steps=3: 3× faster, moderate accuracy loss possible
- Steps=1: Very risky, may diverge

---

### 2. Reduce msign Steps
**Current:** `msign(G, steps=10)` (hardcoded in msign.py)

**Test:** Modify msign.py to accept `steps` parameter from config
```yaml
msign_steps: 5   # 2× speedup for msign component
msign_steps: 3   # 3.3× speedup
```

**Implementation:**
```python
# In manifold/msign.py
def msign(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    # Use passed steps instead of hardcoded 10
```

**Expected impact:**
- Steps=5: 2× faster msign, likely safe (Polar Express converges fast)
- Steps=3: 3× faster, may lose accuracy

---

### 3. Reduce Projection Frequency
**Current:** Project to manifold every iteration

**Test:** Project every N iterations
```yaml
projection_frequency: 1   # Current (every iter)
projection_frequency: 2   # 2× speedup
projection_frequency: 3   # 3× speedup
projection_frequency: 5   # 5× speedup
```

**Implementation:** In train.py, add counter:
```python
if iter % projection_frequency == 0:
    grassmann_update()
else:
    standard_adamw_update()
```

**Expected impact:**
- Freq=2-3: Likely safe, 2-3× speedup
- Freq=5: May drift off manifold, test carefully

---

### 4. Reduce Rank
**Current:** `grass_rank = 192` (n_embd / 2)

**Test:**
```yaml
grass_rank: 96    # n_embd / 4, faster SVD
grass_rank: 128   # n_embd / 3, moderate reduction
```

**Rationale:** Lower rank = faster SVD operations, less expressive but may still work.

**Expected impact:**
- rank=128: 1.5× faster, moderate capacity reduction
- rank=96: 2× faster, significant capacity reduction

---

### 5. Combined Speedup Strategy

**Conservative (2-3 hours instead of 6):**
```yaml
grass_steps: 5        # 2× speedup
msign_steps: 5        # 2× speedup
projection_frequency: 1
grass_rank: 192
```
**Expected: 4× total speedup**

**Moderate (1-2 hours):**
```yaml
grass_steps: 3        # 3.3× speedup
msign_steps: 5        # 2× speedup
projection_frequency: 2  # 2× speedup
grass_rank: 192
```
**Expected: 13× total speedup**

**Aggressive (<1 hour):**
```yaml
grass_steps: 3
msign_steps: 3
projection_frequency: 3
grass_rank: 96
```
**Expected: ~36× speedup, but may hurt performance**

---

## Tuning Plan B: Learning Rate Adjustments

### Issue: 8× LR Boost May Be Too Aggressive

**Current reasoning for 8× boost:**
- G(1,0,r) uses identity-centered parametrization (no skip connection)
- Requires larger LR to make progress
- Based on CIFAR-10 experiments

**Problem:** Transformer dynamics may differ from CIFAR-10 ResNets

### Test Learning Rate Scaling

**Current:**
```yaml
adamw_lr: 1e-3
grass_lr: 8e-3   # 8× boost
```

**Test suite:**
```yaml
# More conservative
grass_lr: 4e-3   # 4× boost
grass_lr: 2e-3   # 2× boost
grass_lr: 1e-3   # 1× (same as AdamW)

# Intermediate
grass_lr: 6e-3   # 6× boost
grass_lr: 5e-3   # 5× boost

# More aggressive
grass_lr: 1e-2   # 10× boost
grass_lr: 1.5e-2 # 15× boost
```

**Test phases:**
- Focus on **Phase 1.2, 2.2, 3.2** (use G(1,0,r) + no skip)
- These are the phases that rely on 8× LR boost

**Hypothesis:**
- 8× may cause instability or overshooting
- 4-6× may be sweet spot
- 2× may be too conservative (slow convergence)

---

## Tuning Plan C: Dropout Adjustment

**Current:**
```yaml
adamw_dropout: 0.2      # Standard for Shakespeare
grass_dropout: 0.1      # Half of AdamW
```

**Rationale:** Grassmann constraint acts as implicit regularization

**Alternative hypothesis:** Half may be too little for transformer

**Test suite:**
```yaml
grass_dropout: 0.15     # 0.75× of AdamW
grass_dropout: 0.2      # Same as AdamW
grass_dropout: 0.05     # Quarter of AdamW
```

---

## Recommended Testing Order

### Phase 1: Speed (Urgent - Enable Rapid Iteration)
1. **Test conservative speedup** (grass_steps=5, msign_steps=5)
   - Run on Phase 1.5 (single Grassmann layer, fast to validate)
   - Compare val loss with baseline Phase 1.5
   - If <5% degradation, adopt for all phases

2. **Test moderate speedup** (grass_steps=3, projection_freq=2)
   - If conservative works, push further
   - Monitor training stability

### Phase 2: Learning Rate (After Speed Fixes)
1. **Sweep LR multiplier** on Phase 1.2
   - Test: [2×, 4×, 6×, 8×, 10×]
   - Plot val loss vs LR multiplier
   - Find optimal multiplier

2. **Validate on other phases**
   - Use best LR from Phase 1.2
   - Test on Phase 2.2, 3.2

### Phase 3: Fine-tuning (Optional)
1. **Rank reduction** (if still too slow)
2. **Dropout adjustment** (if overfitting/underfitting observed)

---

## Success Metrics

### Speed
- **Target:** <3 hours for 15000 iterations (down from 6 hours)
- **Minimum:** <4 hours acceptable
- **Stretch:** <2 hours ideal

### Performance
- **Validation loss:** Should not degrade >5% compared to baseline config
- **Training stability:** No NaN/Inf, smooth loss curves
- **Final checkpoint:** Comparable to baseline quality

### Efficiency
- **MFU:** Target >5% for Grassmann phases (up from 0.89%)
- **Throughput:** Target >3 iters/sec (up from ~1 iter/sec)

---

## Implementation Notes

### Files to Modify

**For speed optimizations:**
```
manifold/grassmann_muon.py     # grass_steps parameter (already exists)
manifold/msign.py              # Add steps parameter
config/*.py                    # Update grass_steps, add msign_steps
train.py                       # Add projection_frequency logic (if needed)
```

**For LR sweep:**
```
config/*.py                    # Change grass_lr values
```

### Config Naming Convention
```
phase1.2_lr4x.py              # LR multiplier in name
phase1.5_fast3x.py            # grass_steps=3 in name
phase2.2_lr6x_fast5x.py       # Both LR and speed config
```

---

## Experimental Matrix (If Resources Allow)

| Phase | grass_steps | msign_steps | grass_lr | Priority |
|-------|-------------|-------------|----------|----------|
| 1.5   | 5           | 5           | 8e-3     | High     |
| 1.5   | 3           | 5           | 8e-3     | High     |
| 1.2   | 5           | 5           | 4e-3     | High     |
| 1.2   | 5           | 5           | 6e-3     | High     |
| 1.2   | 5           | 5           | 8e-3     | Baseline |
| 2.2   | 5           | 5           | 6e-3     | Medium   |
| 3.2   | 5           | 5           | 6e-3     | Medium   |
| 1.5   | 3           | 3           | 8e-3     | Low      |

---

## Questions to Answer

1. **Speed vs Accuracy tradeoff:** How much can we reduce grass_steps before performance degrades?
2. **LR sensitivity:** Is 8× actually optimal, or empirically tuned for CIFAR-10?
3. **Manifold drift:** How often do we need to project? Can we skip projections?
4. **Rank sufficiency:** Is rank=192 overkill? Would rank=96 work?
5. **Dropout interaction:** Does Grassmann constraint interact with dropout rate?

---

## Next Steps

1. Implement `grass_steps=5, msign_steps=5` as default
2. Run Phase 1.5 with new config to validate speedup
3. If successful, update all running jobs with faster config
4. Begin LR sweep experiments on Phase 1.2
5. Document results in this file
