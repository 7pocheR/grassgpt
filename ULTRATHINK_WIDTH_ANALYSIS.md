# UltraThink Analysis: Is d=768 Sufficient?

**Date:** December 25, 2024
**Question:** What width do we need for Grassmann to work on OpenWebText?

---

## TL;DR: d=768 is BORDERLINE, d=1024 is SAFER

**Conservative recommendation:** **d=1024** (GPT-2 Medium)
**Aggressive option:** d=768 with empirical validation
**Scientific approach:** Measure R_eff(d) scaling law first

---

## The Saturation Problem

### Current State (d=384)
```
Layer Type      R_eff    Width    Saturation    Our Rank    Coverage
────────────────────────────────────────────────────────────────────
c_attn          329.2    384      85.7%         48          14.6%
mlp.c_fc        330.9    384      86.2%         20          6.0%
attn.c_proj     256.2    384      66.7%         48          18.8%
mlp.c_proj      306.4    384      79.7%         20          6.5%
```

**Problem:** Model using 66-86% of width → nearly saturated → no room for low-rank constraint

**CIFAR-10 success:** Used rank=160 / width=512 = 31% of width (but R_eff unknown!)

---

## Effective Rank Scaling Laws

### Four Scenarios

**Scenario 1: Linear (Pessimistic)**
```python
R_eff(d) = α × d, where α ≈ 0.86
R_eff(768) = 0.86 × 768 = 660
R_eff(1024) = 0.86 × 1024 = 880
R_eff(1536) = 0.86 × 1536 = 1321
```
**Implication:** Always saturated, need d >> R_eff to create gap

**Scenario 2: Square Root (Medium)**
```python
R_eff(d) = β × √d, where β ≈ 329/√384 ≈ 16.8
R_eff(768) = 16.8 × √768 ≈ 465
R_eff(1024) = 16.8 × √1024 ≈ 538
R_eff(1536) = 16.8 × √1536 ≈ 659
```
**Implication:** Sublinear growth, creating increasing gap

**Scenario 3: Data-Limited (Optimistic)**
```python
R_eff(d) ≈ D_intrinsic ≈ 350-420 (constant or slow growth)
R_eff(768) ≈ 380
R_eff(1024) ≈ 400
R_eff(1536) ≈ 420
```
**Implication:** OpenWebText has intrinsic ~400 dimensions

**Scenario 4: Power Law (Conservative)**
```python
R_eff(d) = β × d^α, where α ≈ 0.4-0.6
R_eff(768) ≈ 420-490
R_eff(1024) ≈ 470-580
R_eff(1536) ≈ 530-690
```
**Implication:** Sublinear but faster than √d

---

## Evidence: Which Scenario is Reality?

### Evidence FOR Linear Scaling (Scenario 1)
1. **High saturation:** 86% suggests model uses all capacity
2. **Layer-wise trend:** Later layers MORE saturated (layer 5: 91%)
3. **Deep learning precedent:** Models often scale to use all parameters

### Evidence FOR Sublinear Scaling (Scenarios 2-4)
1. **Data is finite:** OpenWebText has fixed 9B tokens
2. **Intrinsic dimensionality:** Natural language has structure
3. **GPT-2 research:** Attention heads learn redundant patterns
4. **Per-layer variation:** attn.c_proj only 67% saturated

### Evidence FROM Block Decomposition
**Critical:** We measure full (3n×n) matrix, but decompose into (n×n) blocks!

**Full c_attn matrix:** (1152, 384), R_eff = 329
**After Q/K/V split:** 3 blocks of (384, 384), each R_eff ≈ ???

**Hypothesis:** Per-block R_eff < full matrix R_eff
- Full: R_eff = 329 (85.7% of 384)
- Per-block: R_eff ≈ 250-280 (65-73% of 384)

**At d=768:**
- Full: R_eff = 465-660 (scenario dependent)
- Per-block: R_eff ≈ 350-500
- Our rank r=384
- **Coverage: 77-109% of per-block R_eff**

This makes d=768 more viable!

---

## Width Requirements by Target

### Target A: r ≈ R_eff (Minimal Compression)

**Goal:** Use Grassmann for regularization, not compression

**For r = 0.95 × R_eff and r = d/2:**
```
If R_eff ≈ 350-420 (data-limited):
  Need d = 2 × R_eff / 0.95 = 737-884  →  d=768 or d=1024 ✓

If R_eff(d) ∝ √d:
  R_eff = 16.8√d, r = d/2
  0.95 × 16.8√d = d/2
  d = (0.95 × 16.8 × 2)^2 ≈ 1024  →  d=1024 ✓

If R_eff(d) ∝ d:
  R_eff = 0.86d, r = d/2
  0.95 × 0.86d = d/2
  0.817d = 0.5d  →  IMPOSSIBLE! ❌
```

**Conclusion:** Linear scaling requires NO compression (r ≈ d), defeating the purpose!

### Target B: Match CIFAR-10 Ratio (r/d = 31%)

**CIFAR-10:** r=160 / d=512 = 31%

**For r/d = 0.31 and r = 0.85 × R_eff (85% variance):**
```
If R_eff ≈ 400 (data-limited):
  r = 0.85 × 400 = 340
  d = 340 / 0.31 = 1097  →  d=1024 ✓

If R_eff ∝ √d:
  r = 0.85 × 16.8√d = 0.31d
  √d = 0.31d / (0.85 × 16.8)
  d ≈ 1600  →  d=1536 or d=1600 ✓✓

If R_eff ∝ d:
  r = 0.85 × 0.86d = 0.31d
  0.731d = 0.31d  →  IMPOSSIBLE! ❌
```

**Conclusion:** If aiming for CIFAR-10 ratio, need d=1024-1600!

### Target C: Capture 85% Variance (Conservative)

**Measured:** Rank@85% ≈ 182 at d=384

**At d=768 (if rank@85% scales as √d):**
```
Rank@85% = 182 × √2 ≈ 257
r = 257 (set to match)
r/d = 257/768 = 33% ✓ (close to CIFAR-10!)
```

**At d=1024:**
```
Rank@85% = 182 × √(1024/384) ≈ 298
r = 298
r/d = 298/1024 = 29% ✓✓ (better than CIFAR-10!)
```

---

## Width Recommendations

### Option 1: d=768 (GPT-2 Small) ⚠️ RISKY

**Grassmann config:**
```python
n_embd = 768
grass_rank = 384  # 50% of width

# If R_eff ≈ 400 (data-limited): 384/400 = 96% coverage ✓
# If R_eff ≈ 465 (√d scaling): 384/465 = 82% coverage ✓
# If R_eff ≈ 660 (linear): 384/660 = 58% coverage ⚠️
```

**Pros:**
- Standard architecture (GPT-2 Small)
- 4× parameters of current (124M vs 30M)
- Works IF effective rank is data-limited or √d

**Cons:**
- Fails IF effective rank scales linearly
- Still high compression if R_eff > 500
- High risk, need empirical validation

**Verdict:** **Borderline, proceed with caution**

---

### Option 2: d=1024 (GPT-2 Medium) ✅ SAFER

**Grassmann config:**
```python
n_embd = 1024
grass_rank = 512  # 50% of width

# If R_eff ≈ 400 (data-limited): 512/400 = 128% ✓✓ (over-parameterized!)
# If R_eff ≈ 538 (√d scaling): 512/538 = 95% coverage ✓✓
# If R_eff ≈ 880 (linear): 512/880 = 58% coverage ⚠️ (still compressed)
```

**Pros:**
- Works under ALL scenarios except linear scaling
- Creates 2.67× gap vs current (1024 vs 384)
- Reduces rank ratio to 50% even with √d scaling
- Standard architecture (GPT-2 Medium)

**Cons:**
- 350M parameters (11× current) → 2.8× slower training
- Higher compute cost

**Verdict:** **Conservative choice, high success probability**

---

### Option 3: d=1536 (Custom Wide) ✅✅ SAFEST

**Grassmann config:**
```python
n_embd = 1536
n_layer = 12  # Keep depth reasonable
grass_rank = 768  # 50% of width

# If R_eff ≈ 400 (data-limited): 768/400 = 192% ✓✓✓ (huge gap!)
# If R_eff ≈ 659 (√d scaling): 768/659 = 116% ✓✓✓
# If R_eff ≈ 1321 (linear): 768/1321 = 58% coverage ⚠️
```

**Pros:**
- Works under ALL scenarios except worst-case linear
- 4× gap vs current (1536 vs 384)
- Rank ratio: 768/1536 = 50%
- Almost guaranteed to work

**Cons:**
- ~550M parameters (18× current)
- 4.5× slower training than d=768
- Non-standard architecture (between GPT-2 Medium and Large)

**Verdict:** **Maximum confidence, highest cost**

---

## The Gating Wildcard

**From Qwen's Gated Attention paper (NeurIPS 2025):**

Gating adds **sparsity-inducing mechanism** that may REDUCE effective rank!

**Hypothesis:** Gating + width scaling could be synergistic:
- Gating: Induces sparsity → lowers effective rank
- Width scaling: Creates capacity gap
- Together: Larger gap than width alone

**Example at d=768 with gating:**
```
Without gating: R_eff ≈ 465
With gating: R_eff ≈ 350 (sparsity reduces by ~25%)
Rank r=384
Coverage: 384/350 = 109% ✓✓
```

**Implication:** d=768 + gating might work even if d=768 alone fails!

**Risk mitigation:** Try gating at d=768 before scaling to d=1024

---

## Recommended Experimental Strategy

### Stage 0: Quick Empirical Test (3 days)

**Measure R_eff scaling on smaller models:**

```python
# config/test_rank_scaling.py
# Train 3 small models to convergence:
n_layer = 4
n_embd = [256, 384, 512]  # 3 widths
max_iters = 15000  # Quick convergence

# Measure:
# - R_eff(256), R_eff(384), R_eff(512)
# - Fit scaling law: R_eff = β × d^α
# - Extrapolate to d=768, d=1024
```

**Decision rule:**
```
If α ≈ 1.0 (linear): Use d=1536
If α ≈ 0.5 (√d): Use d=1024
If α < 0.3 (data-limited): Use d=768
```

**Timeline:** 3 days × 3 jobs = 1 week

---

### Stage 1A: Baseline at Target Width (7-10 days)

**Based on Stage 0 results, run ONE baseline:**

**If Stage 0 suggests d=768:**
```bash
sbatch submit_openwebtext_baseline_gpt2small.sh  # d=768
```

**If Stage 0 suggests d=1024:**
```bash
sbatch submit_openwebtext_baseline_gpt2medium.sh  # d=1024
```

**If Stage 0 suggests d=1536:**
```bash
sbatch submit_openwebtext_baseline_custom_wide.sh  # d=1536
```

**Measure:**
- Effective rank at target width
- Operator norms for Grassmann scaling
- Confirm scaling law hypothesis

---

### Stage 1B: Gating Baseline (Parallel, 7-10 days)

**Run in parallel with Stage 1A:**

```bash
sbatch submit_openwebtext_baseline_gated.sh  # d=384 + gating
```

**Goal:** Test if gating alone helps at current width

---

### Stage 2: Grassmann at Validated Width (7-10 days)

**Only after Stage 1A confirms R_eff:**

```python
# config/train_openwebtext_grassmann_gpt2{small,medium,wide}.py
# Use measured R_eff and operator norms

# Set rank based on empirical R_eff:
grass_rank = min(d/2, measured_R_eff * 0.95)  # 95% coverage
```

---

### Stage 3: Grassmann + Gating (7-10 days)

**If Stage 2 works, add gating:**

```bash
sbatch submit_openwebtext_grassmann_gated_gpt2X.sh
```

**If Stage 2 fails, skip this (gating won't save it)**

---

## Decision Tree

```
START
  │
  ├─> Run Stage 0 (rank scaling measurement) [3 days]
  │     │
  │     ├─> α ≈ 1.0? → Use d=1536 (safest)
  │     ├─> α ≈ 0.5? → Use d=1024 (safe)
  │     └─> α < 0.3? → Use d=768 (risky but viable)
  │
  ├─> Run Stage 1A (baseline at chosen d) [7-10 days]
  │   + Stage 1B (gated baseline at d=384) [parallel]
  │     │
  │     ├─> Measure R_eff at new width
  │     └─> Confirm r/R_eff ≥ 0.85? → Proceed
  │                                 → Else: Scale up further
  │
  ├─> Run Stage 2 (Grassmann at chosen d) [7-10 days]
  │     │
  │     ├─> Success? → Done! 🎉
  │     ├─> Marginal? → Try Stage 3 (add gating)
  │     └─> Failure? → Width insufficient, need other approach
  │
  └─> [Optional] Stage 3 (Grassmann + Gating) [7-10 days]
        │
        └─> Success? → Best result! 🎉🎉
```

**Total timeline:** 3-5 weeks (with empirical validation)

---

## Conservative vs Aggressive Paths

### Aggressive Path: Skip Stage 0, use d=768
**Timeline:** 2-3 weeks
**Risk:** High (50% failure if R_eff scales badly)
**Reward:** Faster results, lower compute

### Conservative Path: Full staged approach with d=1024
**Timeline:** 4-5 weeks
**Risk:** Low (20% failure probability)
**Reward:** High confidence, more learnings

### Hybrid Path: Stage 0 → informed decision
**Timeline:** 3-4 weeks
**Risk:** Medium (30% failure)
**Reward:** Best risk/reward ratio ✓✓

---

## Final Recommendations

### 1. Primary Plan: Hybrid Path with d=1024

```bash
# Week 1: Measure scaling law
sbatch submit_rank_scaling_test.sh  # Test d=256,384,512

# Week 2-3: Baseline at d=1024 (conservative, likely sufficient)
sbatch submit_openwebtext_baseline_gpt2medium.sh
sbatch submit_openwebtext_baseline_gated.sh  # Parallel

# Week 3-4: Grassmann at d=1024
# (Use measured R_eff and norms from baseline)
sbatch submit_openwebtext_grassmann_gpt2medium.sh

# Week 4-5: Add gating if needed
sbatch submit_openwebtext_grassmann_gated_gpt2medium.sh
```

**Justification:**
- d=1024 safe under √d and data-limited scenarios
- Provides 2.67× width expansion (strong gap creation)
- Standard GPT-2 architecture (no surprises)
- Only fails under worst-case linear scaling

### 2. Backup Plan: If d=1024 Still Saturated

**If R_eff(1024) > 800 → scale to d=1536:**
```python
n_embd = 1536
n_layer = 12
grass_rank = 768
```

### 3. Quick Validation: Gating at d=384 First

**Before committing to large-scale training:**
```bash
sbatch submit_openwebtext_grassmann_gated.sh  # d=384 + gating
```

**If this succeeds (gating fixes instability at current width):**
- Proceed with d=768 + gating (cheaper!)
- Skip d=1024 (not needed)

**If this fails (gating doesn't help at d=384):**
- Width scaling is necessary
- Proceed with d=1024 plan

---

## Open Questions

1. **What is the per-block effective rank?**
   - Full c_attn: R_eff = 329
   - Per Q/K/V block: R_eff = ???
   - **Need to measure this!**

2. **Does gating reduce effective rank?**
   - Hypothesis: Sparsity → lower R_eff
   - Magnitude: 10-30% reduction?
   - **Needs empirical test**

3. **What's the optimal r/R_eff ratio for Grassmann?**
   - CIFAR-10 used unknown ratio (R_eff not measured)
   - We've been guessing 85-95%
   - **Might need to test 70%, 85%, 95%**

4. **Is there a ceiling effect?**
   - Maybe R_eff can't exceed ~400 regardless of d
   - Would make d=1024 definitely work
   - **Stage 0 will answer this**

---

## Conclusion

**Is d=768 sufficient?**

**Short answer:** **MAYBE** (50-70% confidence)

**Long answer:**
- ✅ Yes, IF R_eff is data-limited (~400) or scales as √d
- ⚠️ Borderline IF R_eff scales as d^0.6
- ❌ No, IF R_eff scales linearly with d

**Recommendation:** **Use d=1024** (GPT-2 Medium) as primary target
- Higher success probability (70-85%)
- Standard architecture
- Only 2.8× slower than d=768
- Safe under most scaling scenarios

**Insurance:** Run Stage 0 first (3 days) to measure scaling law empirically

**Alternative:** Try gating at d=384 first - if it works, might avoid width scaling entirely!

---

**Status:** Analysis complete, ready for decision
**Next step:** Choose path (Aggressive d=768, Conservative d=1024, or Hybrid with Stage 0)

