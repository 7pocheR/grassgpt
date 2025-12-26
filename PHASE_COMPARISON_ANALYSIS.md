# Phase Comparison Analysis: OpenWebText Strategy

**Date:** November 27, 2024
**Context:** Phase 2.5 (QKV+MLP, 62% coverage) failed on OpenWebText with wrong operator norms and ranks

---

## Executive Summary

**Phase 2.5 Failure Analysis:**
- ❌ **Operator norm mismatch**: mlp.c_fc over-scaled by 35% (used 10.14, actual 7.52)
- ❌ **Rank too low**: Used r=48,20 (below 50% variance threshold of r=66,58)
- ❌ **Result**: All variants 10-17% worse than baseline, plateau/divergence after 35-65k iters

**Corrected Baseline Measurements:**
```
OpenWebText (6L×384d, iter 69k):
  c_attn:   op_norm=4.78, ranks: @50%=66, @85%=182, @95%=262, @99%=332
  mlp.c_fc: op_norm=7.52, ranks: @50%=58, @85%=187, @95%=271, @99%=338
```

**Recommended Strategy:**
1. ✅ Start with **lower coverage** phases (Phase 0, 1) for stability
2. ✅ Use **corrected operator norms** (4.78, 7.52 instead of 4.63, 10.14)
3. ✅ Use **moderate ranks** (85% variance: r=182,187) instead of aggressive compression
4. ⚠️ Avoid Phase 2.5 until lower phases succeed

---

## Rank Analysis: Variance Thresholds

| Layer | 50% Var | 85% Var | 95% Var | 99% Var | Full Rank | Effective Rank |
|-------|---------|---------|---------|---------|-----------|----------------|
| **c_attn** | 66 | 182 | 262 | 332 | 384 | 329 |
| **mlp.c_fc** | 58 | 187 | 271 | 338 | 384 | 331 |
| **attn.c_proj** | 37 | 118 | 179 | 246 | 384 | 256 |
| **mlp.c_proj** | 39 | 147 | 237 | 324 | 384 | 306 |

**Key Insights:**
- **OpenWebText layers are nearly full rank** (effective rank ~256-331 out of 384)
- **50% variance requires r=58-66** (first 50% of energy in <20% of dimensions)
- **85% variance requires r=182-187** (reasonable middle ground)
- **95% variance requires r=262-271** (close to full rank, minimal compression)
- **99% variance requires r=332-338** (almost no compression)

**Our Previous Experiments:**
| Experiment | c_attn rank | mlp.c_fc rank | Variance Coverage | Status |
|-----------|-------------|---------------|-------------------|--------|
| stable_rank_x2 | 48 | 20 | **<50%** | ❌ 16.8% worse, too compressed |
| rank99 | 96 | 48 | **~50-60%** | ❌ 10.6% worse, plateaued |
| per_block | Variable | Variable | **~50-60%** | ❌ 11.2% worse, diverged |

**Conclusion:** Our ranks were **severely under-ranked** for OpenWebText!

---

## Phase Coverage Table

| Phase | Layers | Coverage | Grassmann Params | Parametrization | LR Strategy |
|-------|--------|----------|------------------|----------------|-------------|
| **Baseline** | None | 0% | None | N/A | 1e-3 (AdamW) |
| **Phase 0** | attn.c_proj | 8% | c_proj only | G_{0,-1,r} | 1e-3 |
| **Phase 1** | c_attn, attn.c_proj | 28% | QKV blocks + c_proj | Mixed | Q,K,V: 8e-3<br>c_proj: 1e-3 |
| **Phase 2** | mlp.c_fc, attn.c_proj | 34% | c_fc + c_proj | Mixed | c_fc: 8e-3<br>c_proj: 1e-3 |
| **Phase 2.5** | c_attn, mlp.c_fc | 62% | QKV + c_fc (no c_proj) | All G_{1,0,r} | 8e-3 (uniform) |
| **Phase 3** | c_attn, mlp.c_fc, attn.c_proj | 62% | All except mlp.c_proj | Mixed | No-skip: 8e-3<br>c_proj: 1e-3 |

**Note:** Phase 2.5 used in failed experiments is actually Phase 3 without attn.c_proj in some configs.

---

## Recommended Experiments (Priority Order)

### Priority 1: Phase 0 Corrected (8% coverage)

**Motivation:**
- Minimal risk, only attn.c_proj (8% coverage)
- **attn.c_proj has MOST LOW-RANK structure** (rank@85%=118, effective rank=256)
- Best candidate for low-rank benefit
- Validate corrected operator norms work

**Config:**
```python
# config/train_openwebtext_phase0_corrected.py
use_grassmann = True
grassmann_phase = 'phase0'

# CORRECTED operator norm (was not measured before, Phase 0 used defaults)
# Phase 0 uses attn.c_proj only
grass_scale_c_proj = 2.66  # From analysis (not c_attn!)

# Moderate rank (85% variance for attn.c_proj)
grass_rank_c_proj = 118  # 85% variance (was using defaults before)

# Conservative LR (skip-compatible)
grass_lr = 1e-3
grass_a = 0.0
grass_b = -1.0
```

**Expected Outcome:**
- ✅ Stable training (only 8% of params, skip-compatible)
- ✅ Tests corrected operator norm methodology
- ⚠️ Unlikely to beat baseline (too small coverage)
- ✓ **Success criteria: No divergence, within 5% of baseline**

---

### Priority 2: Phase 1 Corrected (28% coverage)

**Motivation:**
- Low-medium risk, QKV + attn.c_proj (28% coverage)
- QKV are natural candidates for low-rank (attention compression)
- c_attn has moderate low-rank structure (rank@85%=182)
- Good balance: enough coverage to show benefit, not too aggressive

**Config:**
```python
# config/train_openwebtext_phase1_corrected.py
use_grassmann = True
grassmann_phase = 'phase1'

# CORRECTED operator norms (measured from OpenWebText baseline)
grass_scale_c_attn = 4.78  # Was 4.63 from Shakespeare
grass_scale_c_proj = 2.66  # attn.c_proj (different from c_attn!)

# Moderate ranks (85% variance - not too aggressive)
grass_rank_c_attn = 182  # 85% variance (was 48, way too low!)
grass_rank_c_proj = 118  # 85% variance for attn.c_proj

# Mixed LR (no-skip: 8×, skip: 1×)
grass_lr_c_attn = 8e-3  # QKV blocks, no skip
grass_lr_c_proj = 1e-3  # attn.c_proj, faces skip
grass_a = 1.0
grass_b = 0.0
```

**Expected Outcome:**
- ✅ Higher chance of success than Phase 2.5 (lower coverage, correct norms)
- ✅ Tests if attention benefits from manifold constraint
- ⚠️ May not beat baseline but should be stable
- ✓ **Success criteria: Within 2-3% of baseline, no plateau**

---

### Priority 3: Phase 2 Corrected (34% coverage)

**Motivation:**
- Medium risk, mlp.c_fc + attn.c_proj (34% coverage)
- **Critical test of mlp.c_fc with CORRECTED norm** (was 35% over-scaled!)
- mlp.c_fc has moderate low-rank structure (rank@85%=187)
- Orthogonal to Phase 1 (MLP vs attention)

**Config:**
```python
# config/train_openwebtext_phase2_corrected.py
use_grassmann = True
grassmann_phase = 'phase2'

# CORRECTED operator norms (mlp.c_fc was severely wrong!)
grass_scale_mlp_fc = 7.52  # Was 10.14 (35% over-scaling!) ← CRITICAL FIX
grass_scale_c_proj = 2.66  # attn.c_proj

# Moderate ranks (85% variance)
grass_rank_mlp_fc = 187  # Was 20 (way too low!)
grass_rank_c_proj = 118

# Mixed LR
grass_lr_mlp_fc = 8e-3  # No skip
grass_lr_c_proj = 1e-3  # Faces skip
grass_a = 1.0
grass_b = 0.0
```

**Expected Outcome:**
- ✅ **Most important test**: Does corrected mlp.c_fc norm fix the divergence?
- ✅ If stable → confirms operator norm hypothesis
- ❌ If still diverges → deeper issue beyond norms
- ✓ **Success criteria: No divergence (most important!), within 5% of baseline**

---

### Priority 4: Phase 3 Corrected (62% coverage) - ONLY IF 1-3 SUCCEED

**Motivation:**
- High risk, same coverage as failed Phase 2.5
- Combines QKV + mlp.c_fc + attn.c_proj
- Only try if lower phases show promise

**Config:**
```python
# config/train_openwebtext_phase3_corrected.py
use_grassmann = True
grassmann_phase = 'phase3'

# CORRECTED operator norms
grass_scale_c_attn = 4.78
grass_scale_mlp_fc = 7.52  # CRITICAL CORRECTION
grass_scale_c_proj = 2.66

# Moderate ranks (85% variance)
grass_rank_c_attn = 182
grass_rank_mlp_fc = 187
grass_rank_c_proj = 118

# Mixed LR
grass_lr_c_attn = 8e-3
grass_lr_mlp_fc = 8e-3
grass_lr_c_proj = 1e-3
grass_a = 1.0
grass_b = 0.0
```

**Expected Outcome:**
- ⚠️ Still high risk (62% coverage)
- ✅ If Phases 1-2 work, might show additive benefits
- ❌ If Phases 1-2 fail, Phase 3 will also fail
- ✓ **Success criteria: Match or beat baseline by 1-2%**

---

## Conservative vs Aggressive Rank Strategies

### Conservative (50% variance): r=66,58

**Pros:**
- Strong compression (83% parameter reduction in Grassmann blocks)
- Faster training (fewer dual ascent iterations)
- Maximum regularization benefit

**Cons:**
- ❌ Too aggressive for OpenWebText (proven by Phase 2.5 failure)
- ❌ May lose critical information
- ❌ Likely to plateau or diverge

**Verdict:** ❌ **Do NOT use** - proven too compressed

---

### Moderate (85% variance): r=182,187 ← **RECOMMENDED**

**Pros:**
- ✅ Balanced compression (~50% parameter reduction)
- ✅ Retains 85% of singular value energy
- ✅ Still provides regularization benefit
- ✅ Less likely to plateau

**Cons:**
- Slower than 50% variance (more dual ascent work)
- Less aggressive compression

**Verdict:** ✅ **Use this** - best balance for first corrected experiments

---

### Aggressive (95% variance): r=262,271

**Pros:**
- Minimal information loss (95% energy retained)
- High expressiveness
- Unlikely to plateau

**Cons:**
- ❌ Minimal compression (32% reduction, approaching full rank)
- ❌ Less regularization benefit
- ⚠️ May not show improvement over AdamW

**Verdict:** ⚠️ **Only if 85% variance fails** - too close to full rank

---

## Phase Dependency Strategy

**DO NOT run all phases in parallel!** Use incremental validation:

### Stage 1: Validate Corrected Norms (Week 1)
```
Submit: Phase 0 corrected + Phase 1 corrected
Wait: Monitor for 20-30k iters (~2-3 days)
Check: Stability, no divergence
```

**Success criteria:**
- No divergence or plateau
- Loss within 5% of baseline
- Training smooth (no oscillations)

**If Stage 1 succeeds → Stage 2**
**If Stage 1 fails → Re-evaluate methodology**

---

### Stage 2: Test Critical MLP Fix (Week 2)
```
Submit: Phase 2 corrected (mlp.c_fc with correct norm)
Wait: Monitor for 30-40k iters (~3-4 days)
Check: Does mlp.c_fc stabilize with correct norm?
```

**Success criteria:**
- **No divergence** (most critical!)
- mlp.c_fc gradients stable
- Loss within 5% of baseline

**If Stage 2 succeeds → Operator norm hypothesis CONFIRMED**
**If Stage 2 fails → Deeper issue, need architectural changes**

---

### Stage 3: Full Integration (Week 3)
```
Submit: Phase 3 corrected (QKV + mlp.c_fc + c_proj)
Wait: Monitor for 50-60k iters (~5-6 days)
Check: Do benefits compose?
```

**Success criteria:**
- Match or beat baseline by 1-2%
- No divergence or plateau
- Better generalization gap than baseline

**If Stage 3 succeeds → Grassmann validated on OpenWebText!**
**If Stage 3 fails → Coverage may be too high, try different combinations**

---

## Risk Assessment

| Phase | Risk Level | Why? | Recommended? |
|-------|-----------|------|--------------|
| **Phase 0** | 🟢 Low | Only 8%, skip-compatible, attn.c_proj is most low-rank | ✅ Yes (validation) |
| **Phase 1** | 🟡 Medium-Low | 28% coverage, attention natural for low-rank | ✅ Yes (promising) |
| **Phase 2** | 🟡 Medium | 34% coverage, critical mlp.c_fc test | ✅ Yes (critical test) |
| **Phase 3** | 🟠 Medium-High | 62% coverage, same as failed Phase 2.5 | ⚠️ Only if 1-2 succeed |
| **Phase 2.5** | 🔴 High | 62% coverage, proven unstable | ❌ No (skip) |

---

## Operator Norm Correction Impact

**What Changed:**

| Layer | Shakespeare | OpenWebText | Difference | Previous Impact |
|-------|------------|-------------|------------|-----------------|
| **c_attn** | 4.63 | 4.78 | +3.2% | 3% under-scaling (minor) |
| **mlp.c_fc** | 10.14 | 7.52 | **-25.8%** | **35% over-scaling (CRITICAL)** |

**Why This Matters:**
- Grassmann manifold constrains W to have operator norm = 1
- We scale outputs by measured norm to match baseline magnitudes
- **Over-scaling by 35% → gradients 35% too large → training instability**

**Evidence This Caused Failure:**
1. ✅ All Phase 2.5 variants (including mlp.c_fc) plateaued or diverged
2. ✅ Phase 0 (only attn.c_proj, no mlp.c_fc) was more stable historically
3. ✅ Plateau occurred at 35-65k iters (when cumulative gradient error accumulates)
4. ✅ Baseline trained smoothly (no operator norm mismatch)

**Hypothesis Test:**
- **H0:** Correcting operator norms fixes instability
- **H1:** Instability is fundamental to Grassmann + high coverage

**Experiment:** Run Phase 2 with corrected mlp.c_fc norm (7.52 instead of 10.14)
- **If stable → H0 confirmed, corrected norms are the fix**
- **If diverges → H1 confirmed, need architectural changes**

---

## Rank Selection Rationale

**Why 85% variance (r=182,187)?**

1. **Not too aggressive:**
   - 50% variance (r=66,58) proven too compressed → Phase 2.5 failed
   - Need to capture more structure

2. **Not too conservative:**
   - 95% variance (r=262,271) too close to full rank (384)
   - Minimal regularization benefit
   - 99% variance (r=332,338) essentially full rank

3. **Optimal middle ground:**
   - 85% variance captures most important structure
   - Still provides ~50% compression (regularization)
   - Standard threshold in PCA/dimensionality reduction

4. **Supported by literature:**
   - Vision: 50-70% variance often sufficient
   - Language: 80-90% variance more common (richer structure)
   - Our 85% is conservative for language

**Backup Plan:**
- If 85% variance fails → try 95% variance (r=262,271)
- If 95% variance fails → Grassmann may not be suitable for OpenWebText at 62% coverage

---

## Success Criteria Summary

### Phase 0 (8% coverage):
- ✅ No divergence or plateau
- ✅ Within 5% of baseline val loss
- ✓ Validates corrected operator norm methodology

### Phase 1 (28% coverage):
- ✅ No divergence or plateau
- ✅ Within 2-3% of baseline val loss
- ✓ Shows attention benefits from manifold constraint

### Phase 2 (34% coverage):
- ✅ **No divergence** (CRITICAL - tests corrected mlp.c_fc norm)
- ✅ Within 5% of baseline val loss
- ✓ **Confirms operator norm hypothesis**

### Phase 3 (62% coverage):
- ✅ No divergence or plateau
- ✅ **Match or beat baseline by 1-2%**
- ✅ Better generalization gap than baseline
- ✓ **Full validation of Grassmann on OpenWebText**

---

## Timeline Estimate

**Assuming H100 GPUs, 12h time limit, dependency chains:**

| Stage | Experiments | Iterations | Wall Time | Calendar Time |
|-------|------------|------------|-----------|---------------|
| **Stage 1** | Phase 0 + Phase 1 | 30k each | 2-3 days each | 1 week |
| **Stage 2** | Phase 2 | 40k | 3-4 days | + 4 days |
| **Stage 3** | Phase 3 | 60k | 5-6 days | + 6 days |
| **Total** | 4 experiments | 160k total | ~15-20 days | **3 weeks** |

**Resource Requirements:**
- 4 H100 GPUs (1 per experiment, run in sequence)
- ~60-80 GPU-days total
- ~100-200GB checkpoint storage

---

## Recommended Action Plan

### This Week (Stage 1):

1. **Create 2 corrected configs:**
   - `config/train_openwebtext_phase0_corrected.py`
   - `config/train_openwebtext_phase1_corrected.py`

2. **Submit Stage 1 experiments:**
   ```bash
   sbatch submit_openwebtext_phase0_corrected.sh
   sbatch submit_openwebtext_phase1_corrected.sh
   ```

3. **Monitor for 2-3 days:**
   - Check for divergence (most important!)
   - Compare loss trajectories to baseline
   - Verify operator norms in logs

4. **Decision point (3 days):**
   - ✅ If stable → proceed to Stage 2
   - ❌ If diverges → re-evaluate rank selection (try 95% variance)

### Next Week (Stage 2):

5. **If Stage 1 succeeds, create Phase 2 config:**
   - `config/train_openwebtext_phase2_corrected.py`
   - **Critical test of corrected mlp.c_fc norm**

6. **Submit and monitor:**
   ```bash
   sbatch submit_openwebtext_phase2_corrected.sh
   ```

7. **Decision point (7 days total):**
   - ✅ If stable → **Operator norm hypothesis CONFIRMED**
   - ❌ If diverges → Deeper issue, may need to abandon high-coverage Grassmann

### Week 3 (Stage 3):

8. **If Stage 2 succeeds, create Phase 3 config:**
   - `config/train_openwebtext_phase3_corrected.py`

9. **Submit and monitor:**
   ```bash
   sbatch submit_openwebtext_phase3_corrected.sh
   ```

10. **Final evaluation (21 days total):**
    - Compare all phases to baseline
    - Determine optimal coverage for OpenWebText
    - Document findings (positive or negative)

---

## Open Questions

1. **Why is OpenWebText nearly full rank?**
   - Shakespeare: stable_rank ~24,10 (highly compressible)
   - OpenWebText: stable_rank ~332,338 (nearly full rank)
   - **Hypothesis:** Larger dataset diversity → more orthogonal features
   - **Implication:** Grassmann may be better for small, homogeneous datasets

2. **Can we use adaptive rank selection?**
   - Instead of fixed r=182, start with r=182 and grow if needed
   - Monitor gradient norms, increase rank if too compressed
   - **Risk:** More complex, harder to debug

3. **Should we try lower coverage permanently?**
   - If Phase 1 (28%) works but Phase 3 (62%) fails
   - **Insight:** Grassmann may work best at 20-30% coverage
   - **Trade-off:** Less total benefit, but stable training

4. **Is operator norm scaling the right approach?**
   - Current: Fix manifold at norm=1, scale outputs by measured norm
   - Alternative: Let manifold have norm=measured, different retraction
   - **Worth exploring if corrected norms still fail**

---

## Conclusion

**What We Learned:**
- ✅ Operator norm measurement is CRITICAL (35% error caused failure)
- ✅ OpenWebText is nearly full-rank (effective rank ~330 out of 384)
- ✅ Our ranks were severely under-ranked (r=48,20 vs needed r=182,187)

**What We'll Test:**
- ✅ Phase 0-1-2 with corrected norms and moderate ranks (85% variance)
- ✅ Incremental validation (don't jump to high coverage)
- ✅ Critical test: Does corrected mlp.c_fc norm fix divergence?

**Predicted Outcomes:**
- **Optimistic:** Phase 1-2 stable, Phase 3 beats baseline by 1-2%
- **Realistic:** Phase 1 stable, Phase 2-3 marginal or neutral
- **Pessimistic:** Even corrected norms fail → Grassmann unsuitable for OpenWebText

**Timeline:** 3 weeks for full validation (4 experiments in sequence)

---

**Next Step:** Create Phase 0 and Phase 1 corrected configs and submit Stage 1 experiments.
