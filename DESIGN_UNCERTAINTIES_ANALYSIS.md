# Design Uncertainties: Deep Analysis

**Date:** December 25, 2024
**Context:** Before implementing Grassmann + Gating, analyze all uncertain design choices

---

## Critical Uncertainties Identified

### 1. Gating Granularity: Element-wise vs Head-wise vs Shared

**The Trade-off:**

| Type | Parameters @ d=1024 | PPL (from paper) | Performance | Efficiency |
|------|---------------------|------------------|-------------|------------|
| **Element-wise** | 1M per layer (1024²) | 5.761 | ✅ **Best** | ❌ Expensive |
| **Head-wise** | 16K per layer (1024×16) | 5.792 | ⚠️ -0.03 PPL | ✅ **64× fewer params** |
| **Head-shared** | 1M per layer | 5.801 | ❌ -0.04 PPL | ❌ Worse |

**Analysis:**

Element-wise gate shape: `(batch, seq, n_head, d_head)`
```python
g = sigmoid(W_gate @ x)  # (B,T,1024) → (B,T,1024)
g = g.view(B, T, n_head, d_head)  # Reshape
y = attn_out * g  # Full element-wise control
```

Head-wise gate shape: `(batch, seq, n_head, 1)`
```python
g = sigmoid(W_gate @ x)  # (B,T,1024) → (B,T,16)
g = g.unsqueeze(-1)  # (B,T,16,1)
y = attn_out * g  # One scalar per head
```

**Key insight:** Performance difference is **tiny** (0.03 PPL), but params differ by **64×**!

**For our case:**
- 24 layers × 1M params = **24M additional parameters** (element-wise)
- 24 layers × 16K params = **384K additional parameters** (head-wise)

At 350M total model params, element-wise adds 7%, head-wise adds 0.1%.

**Recommendation:** **Start with head-wise** (16K params per layer)
- Efficiency: 64× fewer parameters
- Performance: Only 0.03 PPL worse than element-wise
- Can upgrade to element-wise if needed

---

### 2. Should W_gate Use Grassmann Constraint?

**Question:** Apply Grassmann to gate parameters themselves?

**Arguments FOR:**
```python
class GrassmannGate(nn.Module):
    def __init__(self, n_embd):
        self.W_gate = GrassmannLinear(n_embd, n_head, rank=n_head//2)
        # Gate parameters also on manifold
```

**Pros:**
- Consistency: all weights on manifold
- Regularization: gate benefits from constraint
- Theoretical elegance

**Arguments AGAINST:**
```python
# Gate purpose: produce sparse scores in [0,1]
g = sigmoid(W_gate @ x)

# If ||W_gate|| = 1 (Grassmann constraint):
# - Output depends heavily on ||x||
# - Hard to control sparsity level
# - Gate becomes scale-dependent (bad!)
```

**Cons:**
- Gate needs to produce specific statistics (mean ≈ 0.12)
- Manifold constraint might interfere with learning sparsity
- Paper doesn't mention constraining gates (would have said so)
- Gates are small (16K params), regularization less critical

**From paper:** No mention of special gate initialization or constraints. Standard linear layer implied.

**Critical test:**
```python
# Standard gate (what paper uses)
gate_out = sigmoid(W_gate @ x)  # Mean ≈ 0.12 (learned)

# Grassmann gate (what we might try)
gate_out = sigmoid(scale × (W_grass @ x))  # Mean = ???
# Sparsity depends on scale, not learned organically
```

**Recommendation:** **NO Grassmann on gates**
- Keep W_gate as standard learned weights
- Let sparsity emerge naturally through training
- Gates have different purpose than transformation matrices

---

### 3. Grassmann on Q, K, V vs Only on Output Projection

**Current plan:** Block-decompose c_attn into Q, K, V, each Grassmann-constrained

**Re-think this:**

**Paper's key insight (Section 4.1):** Gating breaks the **V → O bottleneck**
```
o_i = Σ_j S_ij · X_j (W_V W_O)
              └────────────┘
         Low-rank bottleneck!
```

**Observation:** The bottleneck is between `W_V` and `W_O`, NOT in Q, K projections.

**Q, K projections:**
- Feed into attention computation (softmax), not a linear bottleneck
- Don't benefit from gating's non-linearity (gating is AFTER SDPA)
- Still benefit from Grassmann regularization, but less critical

**Phased approach:**

**Phase 3a (Minimal Grassmann):**
```python
use_grassmann_on = ['attn.c_proj', 'mlp.c_proj']  # Only output projections
# Skip Q, K, V for now
```
- Simplest implementation
- Focuses on layers facing skip connections
- Lower risk

**Phase 3b (Add V projection):**
```python
use_grassmann_on = ['attn.v', 'attn.c_proj', 'mlp.c_proj']
# Add V to benefit from V→O non-linearity
```
- V is in the gated path (V → SDPA → gate → O)
- More coverage, still manageable

**Phase 3c (Full coverage):**
```python
use_grassmann_on = ['attn.q', 'attn.k', 'attn.v', 'attn.c_proj',
                     'mlp.c_fc', 'mlp.c_proj']
# Full block decomposition
```
- Maximum coverage (82%)
- Most complex

**Recommendation:** **Start with Phase 3a** (only output projections)
- Simplest: no block decomposition needed
- Targets the layers most likely to benefit (skip-facing)
- Can expand to 3b/3c if 3a succeeds

---

### 4. Amplification Factor Uncertainty

**The Critical Problem:**

We're guessing `gate_mean ≈ 0.15` based on paper's 15B MoE model.

**But what if it's different for our 350M dense model?**

**Sensitivity analysis:**

| Actual gate_mean | Our amplification (1/0.15) | Effective scale after gating | Error |
|------------------|----------------------------|------------------------------|-------|
| 0.10 (very sparse) | 6.67× | 6.67 × 0.10 = **0.67×** | **-33% too small** ❌ |
| 0.12 (paper value) | 6.67× | 6.67 × 0.12 = **0.80×** | -20% too small |
| 0.15 (our guess) | 6.67× | 6.67 × 0.15 = **1.00×** | ✓ Perfect |
| 0.20 (less sparse) | 6.67× | 6.67 × 0.20 = **1.33×** | **+33% too large** ❌ |

**This is a ±33% scaling error - NOT acceptable!**

**Solutions:**

**Option A: Measure-then-fix (Recommended)**
```python
# Phase 1: Train with gating (no Grassmann)
# After 5k iters, measure actual gate_mean
gate_stats = measure_gate_statistics(model)
measured_mean = gate_stats['mean']  # e.g., 0.17

# Phase 3: Use measured value
amplification = 1.0 / measured_mean
```

**Option B: Adaptive amplification (Advanced)**
```python
class AdaptiveGatedGrassmann(nn.Module):
    def __init__(self, config):
        # Learnable amplification (log-space for positivity)
        init_amp = 1.0 / config.gate_mean_estimate
        self.log_amp = nn.Parameter(torch.log(torch.tensor(init_amp)))

    def forward(self, x):
        amp = torch.exp(self.log_amp)  # Always positive
        scale = self.baseline_scale * amp
        # ... rest of forward ...
```

**Pros:** Automatically adjusts to correct scale
**Cons:** One more thing to learn, might be unstable

**Option C: Warm-up ramping**
```python
def get_amplification(iter_num, target_amp=6.67, warmup_iters=5000):
    """Ramp from 1.0× to target over warmup period."""
    if iter_num < warmup_iters:
        return 1.0 + (target_amp - 1.0) * (iter_num / warmup_iters)
    else:
        return target_amp
```

**Handles initialization issue:** Gates start near 0.5, gradually become sparse

**Recommendation:** **Combination approach**
1. **Phase 1:** Measure actual gate_mean after 5k iters
2. **Phase 3:** Use measured value with ±10% safety margin
3. **Warm-up:** Ramp amplification from 1.0× to target over 5k iters
4. **Monitoring:** Log gate_mean every 1k iters, warn if drifts >20%

---

### 5. Per-Layer Amplification

**Question:** Should each layer have different amplification?

**From paper (Figure 7):** Gate means vary slightly by layer
- Layer 0: mean ≈ 0.12
- Layer 10: mean ≈ 0.15
- Layer 20: mean ≈ 0.11

**Variance:** ~±20% across layers

**Options:**

**Uniform amplification (simpler):**
```python
amplification = 1.0 / overall_mean  # e.g., 1/0.13 = 7.7×
# Use same amp for all layers
```

**Per-layer amplification (precise):**
```python
amplification = [
    1.0 / measure_gate_mean(layer=0),  # 1/0.12 = 8.3×
    1.0 / measure_gate_mean(layer=1),  # 1/0.15 = 6.7×
    # ... for each layer
]
```

**Analysis:**
- Per-layer is more accurate (each layer perfectly calibrated)
- But adds 24 hyperparameters (one per layer)
- And requires layer-wise measurement

**Trade-off:**
- Error from uniform: max ±20% (based on paper's variance)
- Benefit from per-layer: eliminates this ±20% error
- Cost: 24× more complex

**Recommendation:** **Start uniform, upgrade if needed**
- Use overall average gate_mean
- Monitor per-layer gate statistics
- Only go per-layer if we see >30% variance

---

### 6. Should We Gate the MLP?

**Paper focuses on attention gating.** But we also have MLP:
```
x → LayerNorm → W_fc (expand 4×) → GELU → W_proj (contract) → residual
```

**Potential gating positions:**
- After W_fc (before GELU)
- After GELU (before W_proj)
- After W_proj (before residual)

**Arguments FOR MLP gating:**
- Consistency: gate both attention and MLP
- More coverage
- MLP also has W_fc → W_proj bottleneck

**Arguments AGAINST:**
- MLP already has GELU (non-linearity between W_fc and W_proj)
- Paper tested 5 positions, all in attention (didn't show MLP gating)
- Attention benefits from gating because V→O is purely linear
- MLP doesn't have this issue (GELU breaks linearity)

**From paper Table 3:** Adding non-linearity (gating, RMSNorm) helps attention. MLP already has this.

**Recommendation:** **NO MLP gating initially**
- Attention needs it more (no non-linearity in V→O path)
- MLP already has GELU
- Start simple, add MLP gating only if attention gating alone insufficient

---

### 7. Gate Initialization Strategy

**Question:** How to initialize W_gate?

**Standard init:** `Normal(0, 0.01)` → `sigmoid(0) = 0.5` (50% open initially)

**But paper shows final gate_mean = 0.12 (very sparse).**

**Bootstrap problem:**

Phase 1 (no amplification):
- Gates start at 0.5 (standard init)
- Learn to become 0.12 (sparse)
- ✓ Works fine

Phase 3 (with amplification for gate_mean = 0.12):
- Gates start at 0.5 (standard init)
- Amplification set for 0.12
- Effective scale = (1/0.12) × 0.5 = 4.2× **too large!** ❌
- Training unstable initially

**Solutions:**

**Option A: Initialize sparse**
```python
# Initialize W_gate to produce mean ≈ 0.12 from start
W_gate.data = torch.randn_like(W_gate) * 0.01 - 2.0
# sigmoid(-2.0) ≈ 0.12
```

**Pros:** Matches expected gate_mean from start
**Cons:** Might hurt gradient flow early on

**Option B: Warm-up amplification (Recommended)**
```python
# Iteration 0-5000: ramp amplification 1.0× → target
# Gates can learn sparsity during warmup
# Amplification tracks actual gate_mean
```

**Pros:** Robust to initialization
**Cons:** Adds warm-up complexity

**Option C: Two-stage training**
```python
# Stage 1 (0-10k iters): No amplification, let gates stabilize
# Measure gate_mean at 10k iters
# Stage 2 (10k+): Enable amplification based on measurement
```

**Recommendation:** **Option B - Warm-up with monitoring**
```python
def get_amplification(iter_num, measured_gate_mean):
    """Ramp amplification over first 5k iters."""
    warmup_iters = 5000
    target_amp = 1.0 / measured_gate_mean

    if iter_num < warmup_iters:
        # Ramp from 1.0 to target
        progress = iter_num / warmup_iters
        return 1.0 + (target_amp - 1.0) * progress
    else:
        return target_amp
```

---

### 8. Grassmann Rank Selection at d=1024

**Current plan:** r = 512 (50% of width)

**But if effective rank R_eff ≈ 500:**
- r = 512 → capturing 102% of effective rank (no compression!)

**From CIFAR-10 success:** r/width = 31% (r=160, width=512)

**Options:**

| Rank | Ratio | Compression | Matches |
|------|-------|-------------|---------|
| r = 317 | 31% | Aggressive | CIFAR-10 ratio |
| r = 384 | 37.5% | Moderate | Middle ground |
| r = 512 | 50% | Conservative | Current plan |

**Analysis:**

**If R_eff = 500:**
- r=317: Capturing 63% of effective rank (37% loss)
- r=384: Capturing 77% of effective rank (23% loss)
- r=512: Capturing 102% of effective rank (no loss)

**Trade-off:**
- Lower rank: More regularization, more compression, **might lose critical info**
- Higher rank: Less regularization, safer, **might not benefit from constraint**

**From Phase 2 measurements, we'll know R_eff@85% variance:**

If R_eff@85% = 400:
→ Use r = 400 (captures 85% of energy, proven threshold)

If R_eff@85% = 300:
→ Use r = 350 (slightly above 85% for safety)

**Recommendation:** **Data-driven rank selection**
```python
# After Phase 2, measure R_eff at different variance thresholds
R_eff_50 = ...  # Rank capturing 50% of variance
R_eff_85 = ...  # Rank capturing 85% of variance
R_eff_95 = ...  # Rank capturing 95% of variance

# Start with 85% variance (proven in PCA/dimensionality reduction)
target_rank = R_eff_85

# Add 10% safety margin
grass_rank = int(target_rank * 1.1)
```

Don't guess - measure!

---

### 9. Block Decomposition Scaling with Gating

**If we use Grassmann on Q, K, V (block decomposed):**

```python
# Forward pass
q = scale_q × (W_q @ x)  # Block 1
k = scale_k × (W_k @ x)  # Block 2
v = scale_v × (W_v @ x)  # Block 3

# SDPA
attn_out = attention(q, k, v)  # Inherits ~scale_v

# Gate (reduces by ~0.12)
gated_out = gate × attn_out  # Now ~0.12 × scale_v

# Output (needs amplification)
final = scale_o_amp × (W_o @ gated_out)
```

**Key insight:** Only W_O needs amplification!

Q, K, V are "upstream" of gating - they don't need amplification.

**But there's a subtlety:**

The SDPA output has magnitude ~scale_v (from V projection).
Gate multiplies by 0.12.
So gated_out has magnitude ~0.12 × scale_v.

For final output to have correct magnitude ~scale_o:
```python
scale_o_amp = scale_o / gate_mean
final = scale_o_amp × (W_o @ gated_out)
      = (scale_o / gate_mean) × (W_o @ (gate_mean × scale_v × stuff))
      = scale_o × (W_o @ (scale_v × stuff))
```

**Wait, this doesn't work!** The scale_v is still in there.

**Re-thinking the scaling...**

Actually, in standard attention without Grassmann:
```python
v_out = W_v @ x  # magnitude ~scale_v
attn_out = attention(q, k, v_out)  # magnitude ~scale_v
final = W_o @ attn_out  # magnitude ~scale_v × scale_o

# Total effective operator norm: scale_v × scale_o
```

With Grassmann but no gating:
```python
v_out = scale_v × (W_v_grass @ x)  # Explicit scaling
attn_out = attention(...)  # magnitude ~scale_v
final = scale_o × (W_o_grass @ attn_out)  # magnitude ~scale_v × scale_o ✓
```

With Grassmann + gating:
```python
v_out = scale_v × (W_v_grass @ x)
attn_out = attention(...)  # magnitude ~scale_v
gated = gate × attn_out  # magnitude ~0.12 × scale_v (REDUCED!)

# Need to compensate in W_o scaling
final = scale_o_amp × (W_o_grass @ gated)
      = (scale_o / 0.12) × (W_o @ (0.12 × scale_v × stuff))
      = scale_o × (W_o @ (scale_v × stuff))  ✓ Correct!
```

**Conclusion:** Amplification only needed on the layer AFTER gating (W_o), not layers before.

**Simpler approach:** Only apply Grassmann to W_o (output projection), keep Q, K, V as standard AdamW.

---

### 10. Training Dynamics: How Does gate_mean Evolve?

**Uncertainty:** Paper shows final gate statistics, not evolution over training.

**Possible scenarios:**

**Scenario A: Start open, learn sparsity**
```
Iter 0:     gate_mean = 0.50 (random init)
Iter 10k:   gate_mean = 0.30 (learning to filter)
Iter 50k:   gate_mean = 0.15 (increasingly sparse)
Iter 100k:  gate_mean = 0.12 (final sparsity)
```

**Scenario B: Start sparse, stay sparse**
```
Iter 0:     gate_mean = 0.12 (if we init sparse)
Iter 100k:  gate_mean = 0.12 (stable)
```

**Impact on amplification:**

If we set `amp = 1/0.12` but gate_mean starts at 0.50:
- Initial effective scale = (1/0.12) × 0.50 = 4.2× (too large!)
- Training might diverge

If gate_mean drifts during training:
- Fixed amplification becomes incorrect
- Scale mismatch accumulates

**Solutions:**

**Option A: EMA-based adaptive amp**
```python
class EMAAmplification:
    def __init__(self, initial_gate_mean=0.15, momentum=0.999):
        self.gate_mean_ema = initial_gate_mean
        self.momentum = momentum

    def update(self, batch_gate_mean):
        self.gate_mean_ema = (self.momentum * self.gate_mean_ema +
                             (1 - self.momentum) * batch_gate_mean)

    def get_amplification(self):
        return 1.0 / self.gate_mean_ema
```

**Option B: Staged approach**
```python
# Stage 1 (0-5k): amp = 1.0 (no amplification)
# Measure gate_mean at 5k iters
# Stage 2 (5k+): amp = 1/measured_gate_mean (fixed)
```

**Recommendation:** **Option B (staged)** for simplicity and stability
- No amplification during warm-up
- Measure after stabilization
- Fix for remainder of training
- Monitor gate_mean, warn if drift >20%

---

## Recommended Configuration Matrix

### Conservative Configuration (Recommended for Phase 3a)

```python
# Architecture
n_layer = 24
n_embd = 1024
n_head = 16

# Gating
use_gating = True
gate_type = 'headwise'  # 16K params per layer (not 1M)
gate_on_mlp = False     # Only attention, not MLP

# Grassmann (minimal - only output projections)
use_grassmann = True
grassmann_layers = ['attn.c_proj', 'mlp.c_proj']  # Only skip-facing
grass_rank = measured_R_eff_85  # From Phase 2, e.g., 384
grass_on_gates = False  # Standard weights for gates

# Amplification
gate_mean_estimate = measured_from_phase1  # e.g., 0.15
amplification_warmup = 5000  # Ramp over 5k iters
use_adaptive_amp = False  # Fixed after measurement

# Learning rates
learning_rate = 6e-4  # Base
grass_lr = 4.8e-3     # 8× (if stable with gating)
```

**Rationale:**
- Head-wise gating: 64× fewer params, 0.03 PPL worse
- Minimal Grassmann: Only output projections (easiest to implement)
- No block decomposition: Avoid complexity initially
- Fixed amplification: After measuring in Phase 1
- Conservative rank: Based on measured R_eff@85%

### Aggressive Configuration (Phase 3c - if conservative succeeds)

```python
# Same architecture...

# Gating
gate_type = 'elementwise'  # Full expressiveness
gate_on_mlp = True         # Gate both attention and MLP

# Grassmann (full coverage)
grassmann_layers = ['attn.q', 'attn.k', 'attn.v', 'attn.c_proj',
                    'mlp.c_fc', 'mlp.c_proj']  # All layers
grass_rank = int(measured_R_eff_85 * 0.9)  # More aggressive compression

# Amplification
use_adaptive_amp = True  # Learn optimal amplification
per_layer_amp = True     # Layer-specific amplification

# Learning rates
grass_lr = 6e-3  # 10× base (push limits with gating stability)
```

**Rationale:**
- Element-wise gating: Maximum performance
- Full Grassmann: Maximum coverage (82% of params)
- Adaptive amp: Handles any gate drift
- Aggressive rank: Maximize compression benefit

---

## Implementation Priority

### Must-Have (Phase 3a)
- ✅ Head-wise gating
- ✅ Grassmann on c_proj only
- ✅ Measured amplification (from Phase 1)
- ✅ Warm-up ramping
- ✅ Gate statistics monitoring

### Nice-to-Have (Phase 3b)
- ⚠️ Element-wise gating (if head-wise insufficient)
- ⚠️ Grassmann on V projection
- ⚠️ Per-layer amplification (if >30% variance)

### Advanced (Phase 3c)
- 🔬 Full block decomposition (Q, K, V, MLP)
- 🔬 Adaptive amplification (learnable)
- 🔬 MLP gating
- 🔬 Grassmann on gates (probably not useful)

---

## Critical Measurements Needed

### From Phase 1 (Gating at d=384):
```python
measure_and_log = {
    'gate_mean': [],         # Expected: 0.10-0.20
    'gate_std': [],
    'gate_sparsity': [],     # % of values < 0.1
    'per_layer_gate_mean': [], # Check variance across layers
    'gate_evolution': [],    # How does it change over training?
}
```

### From Phase 2 (Baseline at d=1024):
```python
measure_and_log = {
    'effective_rank': [],          # Expected: ~500-600
    'rank_at_50_pct_var': [],      # For rank selection
    'rank_at_85_pct_var': [],      # Recommended target
    'rank_at_95_pct_var': [],
    'operator_norms': {},          # For Grassmann scaling
    'rank_width_ratio': [],        # Confirm < 60%
}
```

### From Phase 3 (Grassmann + Gating):
```python
monitor_and_log = {
    'effective_scale_post_gating': [],  # Should match baseline
    'gate_mean_ema': [],                # Track drift
    'amplification_used': [],           # Actual amp at each iter
    'per_layer_scales': [],             # Verify all layers correct
}
```

---

## Decision Tree

```
START
  │
  ├─> Phase 1: Gating at d=384
  │     │
  │     ├─> Measure gate_mean
  │     │   ├─> If 0.10-0.20: ✓ Expected
  │     │   └─> If outside: ⚠️ Investigate
  │     │
  │     └─> Measure gate evolution
  │         ├─> If stable: ✓ Use fixed amp
  │         └─> If drifts: ⚠️ Need adaptive amp
  │
  ├─> Phase 2: Baseline at d=1024
  │     │
  │     ├─> Measure R_eff
  │     │   ├─> If ~500-600: ✓ Width sufficient
  │     │   └─> If >700: ⚠️ Need d=1536
  │     │
  │     └─> Measure R_eff@85%
  │         └─> Set grass_rank = R_eff@85%
  │
  └─> Phase 3: Choose configuration
        │
        ├─> Start with Conservative (3a)
        │   ├─> Headwise gating
        │   ├─> c_proj only Grassmann
        │   └─> Fixed amplification
        │
        ├─> If successful → Try Aggressive (3c)
        │   ├─> Elementwise gating
        │   ├─> Full Grassmann coverage
        │   └─> Adaptive amplification
        │
        └─> If fails → Debug and iterate
```

---

## Open Questions Requiring Empirical Answers

1. **Does gate_mean differ significantly between dense and MoE models?**
   - Paper: 15B MoE, gate_mean = 0.116
   - Ours: 350M dense, gate_mean = ???
   - **Answer from Phase 1**

2. **How much does gate_mean vary across layers at d=1024?**
   - Paper shows ~±20% variance
   - Need to measure if per-layer amp worth it
   - **Answer from Phase 1**

3. **What is R_eff at d=1024 really?**
   - Estimated ~500-600
   - Could be higher (worse) or lower (better)
   - **Answer from Phase 2**

4. **Is head-wise gating sufficient or do we need element-wise?**
   - Paper: element-wise 0.03 PPL better
   - Our case: worth 64× more parameters?
   - **Answer from Phase 3a vs 3b comparison**

5. **Can we skip Q, K, V Grassmann entirely?**
   - Only c_proj is in gated path
   - Q, K, V might not benefit from gating synergy
   - **Answer from Phase 3a vs 3c comparison**

---

**Status:** Uncertainties documented, decision tree established
**Next:** Implement conservative Phase 3a, iterate based on measurements
