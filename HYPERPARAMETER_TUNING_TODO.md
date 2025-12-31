# Hyperparameter Tuning TODO for Grassmann + Gating

**Created:** 2024-12-30
**Current Baseline:** Job 604747 (r=512), Job 605987 (r=384)
**Goal:** Match or exceed baseline terminal accuracy with better sample efficiency

---

## Critical Finding: LR Scheduler Issue

**Problem:** Current scheduler has `lr_decay_iters = max_iters = 100000`
- LR stays at 99% of max until iter 20k
- Only reaches 56% at iter 50k (halfway)
- No constant min_lr region for fine-tuning
- Loss improvements slowing (0.03/1000 iters @ iter 8000) but LR not decaying to help

**Impact:** Late training fluctuates (~±4%) due to high LR preventing fine-tuning

**Fix:** See "LR Schedule Modifications" below

---

## Priority 1: LR Schedule Modifications

### Current Issues
```python
warmup_iters = 2000        # Too short for Grassmann
lr_decay_iters = 100000    # Same as max_iters (bad!)
```

### Recommended Changes

**Option A: Longer Warmup + Moderate Decay** ⭐ **RECOMMENDED**
```python
warmup_iters = 4000        # 2× longer (gentler for manifold)
lr_decay_iters = 85000     # Finish decay before max_iters
```
- **Benefit:** Allows higher grass_lr without instability + 15k iters for fine-tuning
- **Cost:** Zero
- **Expected gain:** +2-4% from faster Grassmann learning + better convergence
- **Priority:** 🔴 HIGH - Try FIRST

**Option B: Aggressive Decay**
```python
warmup_iters = 2000
lr_decay_iters = 80000     # Reach min_lr by iter 80k
```
- **Benefit:** 20k iters at min_lr for fine-tuning
- **Risk:** May undershoot optimal loss
- **Priority:** 🟡 MEDIUM - Try if Option A doesn't help

**Option C: Extended Warmup Only**
```python
warmup_iters = 6000        # 3× longer
lr_decay_iters = 100000    # Keep same
```
- **Benefit:** Maximum stability for high grass_lr
- **Drawback:** Still no fine-tuning region
- **Priority:** 🟢 LOW - Only if instability persists

### Experiments to Run

- [ ] **Exp 1:** warmup=4000, lr_decay=85000, grass_lr=4e-3, r=384
- [ ] **Exp 2:** warmup=4000, lr_decay=85000, grass_lr=3e-3, r=384 (if Exp 1 unstable)
- [ ] **Exp 3:** warmup=6000, lr_decay=80000, grass_lr=5e-3, r=384 (aggressive)

---

## Priority 2: Grassmann Learning Rate

### Current Status
```python
grass_lr = 2e-3   # Conservative (was 8e-3 but caused instability)
```

### Analysis
- **8e-3** worked on d=384 but unstable on d=1024
- **2e-3** is very conservative (4× reduction)
- Likely cause: warmup too short (2000 iters)

### Recommended Changes

**Paired with longer warmup:**
```python
warmup_iters = 4000
grass_lr = 4e-3    # 2× current (still 2× less than d=384)
```

**Aggressive (if warmup=6000):**
```python
warmup_iters = 6000
grass_lr = 6e-3    # 3× current
```

### Experiments to Run

- [ ] **grass_lr=3e-3** with warmup=4000 (conservative test)
- [ ] **grass_lr=4e-3** with warmup=4000 (target)
- [ ] **grass_lr=5e-3** with warmup=6000 (stretch goal)

---

## Priority 3: Dropout Reduction

### Current Status
```python
dropout = 0.1   # Standard GPT-2
```

### Hypothesis
Grassmann manifold constraint provides strong regularization:
- **Operator norm:** ||W||_op = 1 (fixed)
- **Frobenius norm:** ||W||_F = sqrt(r) (fixed)
- **Rank constraint:** rank(P) = r (fixed)
- **DOF reduction:** 70% fewer parameters (r=384 vs full rank)

Dropout may waste capacity that's already constrained.

### Recommended Changes
```python
dropout = 0.05    # Half (moderate reduction)
dropout = 0.0     # None (aggressive - trust manifold)
```

### Experiments to Run

- [ ] **dropout=0.05** with r=384, warmup=4000
- [ ] **dropout=0.0** with r=384, warmup=4000
- [ ] Compare train/val gap vs baseline

**Expected gain:** +1-2% from less capacity waste

---

## Priority 4: Scaling Factor (grass_scale)

### Current Status
```python
grass_scale = 10.0   # Uniform across all layers
```

### Hypothesis
- Grassmann outputs ||W||=1 → scale by 10× before gating
- Gate is sigmoid(Linear(x)) → learns input-dependent magnitude
- Optimal scaling depends on gate initialization and gradient flow
- Current x=10 is a guess, not optimized

### Recommended Changes
```python
grass_scale = 8.0     # Lower (tighter initial range)
grass_scale = 12.0    # Higher (wider initial range)
grass_scale = 15.0    # Much higher (if gates learn to compress)
```

### Experiments to Run

- [ ] **grass_scale=8** with r=384, warmup=4000
- [ ] **grass_scale=12** with r=384, warmup=4000
- [ ] **grass_scale=15** with r=384, warmup=4000
- [ ] Analyze gate statistics (mean sigmoid output) for each

**Expected gain:** +1-3% from better gradient flow

---

## Priority 5: Gate Learning Rate ⭐ **CRITICAL FIX**

### Current Issue (FIXED)
```python
# OLD: All AdamW params shared same LR
learning_rate = 6e-4   # For gates, c_proj, embeddings (ALL THE SAME!)
```

**Problem:** Gates have fundamentally different requirements:
- **Gates (176M params, 49.7%):** NEW components, sigmoid activation, need fast adaptation
- **c_proj (126M params, 35.5%):** Standard projections, skip-facing, need stability
- **Embeddings (53M params, 14.8%):** Conservative, avoid disrupting representations

### Fix Implemented ✅

Modified `model.py` to split AdamW into **3 separate optimizer groups**:

```python
# NEW: Different LRs for different components
gate_lr = 1.2e-3       # 2× base (faster gate learning)
learning_rate = 6e-4   # Base (for c_proj)
embed_lr = 3e-4        # 0.5× base (conservative for embeddings)

# Optimizer groups:
# Group 1: Gates      (176M params, lr=1.2e-3, wd=1e-1)
# Group 2: Projections (126M params, lr=6e-4,  wd=1e-1)
# Group 3: Embeddings  ( 53M params, lr=3e-4,  wd=0.0)
```

### Why This Matters

**Gates learn input-dependent scaling:**
```python
# In forward pass:
qkv = self.c_attn(x) * grass_scale  # Grassmann output × 10
gate = torch.sigmoid(self.c_attn_gate(x))  # Learn scaling
qkv = qkv * gate  # Adaptive magnitude
```

- If gates stuck at initialization → gradients don't flow to Grassmann
- Higher LR → gates adapt faster → unlock Grassmann capacity earlier
- **This could be THE missing piece**

### Recommended Changes

**Conservative (default):**
```python
gate_lr = 1.2e-3   # 2× base LR (auto-default)
embed_lr = 3e-4    # 0.5× base LR (auto-default)
```

**Aggressive:**
```python
gate_lr = 2.4e-3   # 4× base LR
embed_lr = 3e-4    # 0.5× base LR
```

### Experiments to Run

- [ ] **gate_lr=1.2e-3** (default 2×) with r=384, warmup=4000
- [ ] **gate_lr=1.8e-3** (3×) with r=384, warmup=4000
- [ ] **gate_lr=2.4e-3** (4×) with r=384, warmup=4000
- [ ] Monitor gate statistics: mean(sigmoid(c_attn_gate(x)))

**Expected gain:** +2-5% from proper gate adaptation
**Risk:** Low (gates are new components, won't destabilize baseline behavior)

---

## Medium Priority: Rank Exploration

### Current Tests
- Job 604747: r=512 (50% of d=1024)
- Job 605987: r=384 (37.5% of d=1024) ⏳ RUNNING

### Analysis
Stronger constraint (lower r) → better inductive bias, but less capacity.

### Further Experiments

- [ ] **r=320** (31.25%, 5×64 GPU-aligned)
- [ ] **r=256** (25%, 2×128 GPU-aligned)
- [ ] **r=448** (43.75%, 7×64) if r=384 too constrained

**Decision point:** Wait for r=384 results before proceeding

---

## Low Priority: Batch Configuration

### Current Status
```python
batch_size = 32                      # Per GPU
gradient_accumulation_steps = 12     # Total 384 seqs/GPU
# Effective: 32 × 12 × 4 GPUs = 1,536 sequences, 1.57M tokens/iter
```

### Hypothesis
With r=384 (smaller than r=512), may fit larger batch_size.

### Recommended Changes
```python
batch_size = 48                      # 50% larger
gradient_accumulation_steps = 8      # Keep same 384 seqs/GPU (divisible by 4!)
```

### Experiments to Run

- [ ] Test if batch=48 fits with r=384 (OOM check)
- [ ] Compare stability vs batch=32

**Expected gain:** +0-1% from lower gradient noise

---

## Low Priority: Optimizer Parameters

### Current Status
```python
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
weight_decay = 1e-1   # AdamW only (Grassmann has NO weight decay ✓)
```

### Potential Tweaks

**beta2:** GPT-3 uses 0.95, some work uses 0.999
```python
beta2 = 0.98    # Slower second moment decay
beta2 = 0.999   # Much slower (BERT-style)
```

**grad_clip:** May need adjustment for Grassmann
```python
grad_clip = 0.5    # Tighter
grad_clip = 2.0    # Looser
```

### Experiments to Run

- [ ] Only if other tweaks don't work
- [ ] Monitor gradient norms first to inform decision

---

## Very Low Priority: Grassmann Projection Accuracy

### Current Status
```python
grass_alpha = 0.01     # Dual ascent step size
grass_steps = 10       # Max dual iterations
grass_tol = 1e-6       # Convergence tolerance
```

### Analysis
These control the accuracy of manifold projection via dual ascent.

**Check logs:** Are we hitting tolerance before 10 steps?
- If YES: Current settings are fine (tolerance reached early)
- If NO: May need more steps or different alpha

### Experiments to Run

- [ ] **Log dual ascent convergence** (add prints to grassmann_muon.py)
- [ ] If not converging: Try grass_steps=15 or grass_alpha=0.02
- [ ] Compare projection accuracy vs terminal loss

**Expected impact:** Minimal (<0.5% expected)

---

## Recommended Experiment Sequence

### Phase 1: Fix LR Scheduler + Increase grass_lr + Split gate_lr ⭐

**Experiment 1A:** Moderate (RECOMMENDED FIRST)
```python
# Architecture
grass_rank = 384

# LR Schedule
warmup_iters = 4000
lr_decay_iters = 85000

# Learning Rates (SPLIT!)
grass_lr = 3e-3         # 1.5× current Grassmann LR
learning_rate = 6e-4    # Base LR (for c_proj)
gate_lr = 1.2e-3        # 2× base (NEW - gate-specific)
embed_lr = 3e-4         # 0.5× base (conservative)

# Keep other params default
```

**Experiment 1B:** Aggressive
```python
# Same as 1A but:
grass_lr = 4e-3         # 2× current Grassmann LR
gate_lr = 1.8e-3        # 3× base LR for gates
```

**Expected:** +4-7% over r=512 baseline from combined fixes

---

### Phase 2: Add Dropout Reduction

**Experiment 2A:** Moderate
```python
# Best from Phase 1 +
dropout = 0.05
```

**Experiment 2B:** Aggressive
```python
# Best from Phase 1 +
dropout = 0.0
```

**Expected:** +1-2% over Phase 1

---

### Phase 3: Scaling Factor Sweep

**Experiment 3A/B/C:** Sweep grass_scale
```python
# Best from Phase 2 +
grass_scale = 8.0   # or 12.0, or 15.0
```

**Expected:** Find optimal, +1-2% over Phase 2

---

### Phase 4: AdamW LR Boost

**Experiment 4:**
```python
# Best from Phase 3 +
learning_rate = 8e-4   # (was 6e-4)
```

**Expected:** +0-1% from faster gate adaptation

---

## Combined Best Guess (High-Risk/High-Reward)

If confident after r=384 shows promise, try everything at once:

```python
# Architecture
grass_rank = 384

# LR Schedule
warmup_iters = 4000
lr_decay_iters = 85000

# Learning Rates (SPLIT BY COMPONENT!)
grass_lr = 4e-3         # 2× current Grassmann LR
learning_rate = 6e-4    # Base LR (for c_proj)
gate_lr = 1.8e-3        # 3× base (gates need fast adaptation)
embed_lr = 3e-4         # 0.5× base (embeddings conservative)

# Regularization
dropout = 0.05

# Scaling
grass_scale = 12.0

# Batch (if fits)
batch_size = 48
gradient_accumulation_steps = 8
```

**Expected cumulative gain:** +7-12% over current r=384
**Risk:** Multiple changes make ablation harder

**Key insight:** gate_lr split is CRITICAL - gates are 50% of AdamW params and need fast learning to unlock Grassmann

---

## Success Metrics

### Sample Efficiency (Primary Goal)
- **Baseline:** val_loss=3.61 @ 3.1B tokens (iter 4000)
- **Current Grassmann r=512:** val_loss=3.64 @ 3.1B tokens (+0.8% worse)
- **Target:** val_loss ≤ 3.55 @ 3.1B tokens (match or beat baseline)

### Terminal Accuracy (Secondary Goal)
- **Baseline:** val_loss ≈ 3.0-3.1 @ 100k iters (extrapolated)
- **Target:** val_loss ≤ 3.1 @ 100k iters

### Training Stability
- No NaN/Inf
- Loss fluctuation < ±5%
- Smooth val loss curve

---

## Current Experiment Status

- [x] **604747:** r=512, warmup=2000, grass_lr=2e-3 (COMPLETED @ iter 8000+)
- [x] **605987:** r=384, warmup=2000, grass_lr=2e-3 (RUNNING @ iter 240)
- [ ] **Next:** r=384, warmup=4000, grass_lr=4e-3 (READY TO SUBMIT)

---

## Notes

### Findings from Analysis

1. **Loss fluctuation @ iter 7640-7990 is NORMAL**
   - ±4% variance is expected gradient noise
   - Val loss decreasing smoothly
   - NOT a sign of instability

2. **LR scheduler issue is REAL**
   - LR stays at 99% until iter 20k
   - Prevents fine-tuning in late training
   - Should set lr_decay_iters < max_iters

3. **Weight decay is CORRECT**
   - Grassmann params: NO weight decay ✓
   - AdamW params (gates, etc.): weight_decay=1e-1 ✓
   - Manifold constraint provides regularization

### CIFAR-10 Lessons

From `/home/junyuren/manifold_muon/`:
- **8× LR boost** worked for non-skip Grassmann (4e-3 vs 5e-4)
- **~50% rank** (256/512) was optimal
- **Longer warmup** helped stability
- **Manifold >> low-rank** for generalization

### Open Questions

- [ ] Does r=384 extend sample efficiency advantage?
- [ ] What's the optimal grass_scale for d=1024?
- [ ] Can dropout=0 work with Grassmann regularization?
- [ ] Is warmup=4000 enough or need 6000?

---

**Last Updated:** 2024-12-30
**Next Action:** Wait for r=384 (job 605987) to reach iter 1000, then decide on warmup=4000 experiment
