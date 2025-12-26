# Gated Attention + Grassmann: Implementation Plan

**Date:** December 25, 2024
**Paper:** "Gated Attention for Large Language Models" (Qwen Team, NeurIPS 2025 Oral)
**arXiv:** 2505.06708

---

## Executive Summary

**Critical Discovery:** Gating scores are **sparse** (mean = 0.116, concentrated near 0). This creates a **scaling problem** for Grassmann:

- Grassmann constrains ||W|| = 1, then scales output by measured norm
- Gating multiplies output by ~0.1-0.2 on average
- **Result:** Effective scale too small, condition number remains small
- **Solution:** Pre-amplify Grassmann scaling by **~6-10×** before gating

---

## Paper Key Findings

### 1. Optimal Gating Configuration

**Position:** G1 - After SDPA output, before output projection
**Formula (Eq. 5):**
```python
Y' = Y ⊙ σ(X W_θ)
```

**Specification:**
- **Head-specific:** Each attention head has independent gate parameters
- **Multiplicative:** Gate multiplies SDPA output (not additive)
- **Activation:** Sigmoid (outputs in [0, 1])
- **Granularity:** Element-wise or head-wise (both work, element-wise slightly better)

**Architecture:**
```
Input → Pre-Norm → Q, K, V → SDPA → [GATE HERE] → W_O → Output
```

### 2. Gating Score Statistics (Table 4)

| Method | Gate Mean | Sparsity | F-Attn | PPL | Performance |
|--------|-----------|----------|--------|-----|-------------|
| Baseline | - | - | 0.467 | 6.026 | Reference |
| **SDPA Element Gate** | **0.116** | High | **0.048** | **5.761** | ✅ Best |
| SDPA Head Gate | 0.172 | High | 0.073 | 5.792 | ✅ Good |
| SDPA Head-shared | 0.271 | Medium | 0.301 | 5.801 | ⚠️ Worse |
| Value Gate | 0.221 | Medium | 0.297 | 5.820 | ⚠️ Worse |
| Input-Independent | 0.335 | Low | 0.364 | 5.917 | ❌ Poor |
| NS-sigmoid [0.5,1.0] | 0.653 | Very Low | 0.451 | 5.900 | ❌ Poor |

**Key Insights:**
- ✅ **Lower gate mean = better performance** (sparsity is crucial!)
- ✅ Head-specific gating matters (head-shared worse)
- ✅ Query-dependent sparsity essential (input-independent fails)
- ❌ Reducing sparsity (NS-sigmoid) hurts performance

### 3. Why Gating Works

**A. Non-linearity (Section 4.1):**

In multi-head attention:
```
o_i = Σ_j S_ij · X_j (W_V W_O)
      └─────────────┘
      Low-rank bottleneck!
```

Without gating: `W_V` and `W_O` merge into one low-rank mapping (d_k < d_model)

With gating: Breaks the low-rank bottleneck
```
o_i = Σ_j S_ij · σ(X_j W_gate) ⊙ (X_j W_V) W_O
                 └──────────────┘
                 Non-linearity!
```

**B. Sparsity (Section 4.2):**

- Gate mean = 0.116 → **88% of values near 0**
- Query-dependent filtering of irrelevant context
- Input-dependent sparsity (not fixed)

**C. Attention Sink Elimination (Section 4.3):**

| Metric | Baseline | With Gating | Improvement |
|--------|----------|-------------|-------------|
| First token attention | 46.7% | **4.8%** | ✅ 90% reduction |
| Massive activations (M-Act) | 1053 | **94** | ✅ 91% reduction |

**D. Training Stability (Table 2):**

- Eliminates loss spikes
- Tolerates **2× higher LR** (8e-3 vs 4e-3)
- Better scaling to deeper models (48L stable)

### 4. Implementation Details

**Gate Parameters:**
```python
# For head-specific element-wise gating
# Input shape: (batch, seq, n_embd)
# Output shape: (batch, seq, n_head, d_head)

self.gate = nn.Linear(n_embd, n_head * d_head)  # or (n_embd, n_head) for head-wise
```

**Parameter count:**
- Element-wise: n_embd × (n_head × d_head) = 384 × 384 = **147K** for d=384
- Head-wise: n_embd × n_head = 384 × 6 = **2.3K** for d=384
- **Negligible overhead!**

**Computational cost:**
- Wall-time increase: **< 2%**
- Memory increase: **< 1%**

---

## The Critical Scaling Problem for Grassmann

### Problem Statement

**Grassmann constraint:** ||W_grassmann|| = 1 (operator norm = 1)

**Current scaling approach:**
```python
# Measure baseline operator norm
baseline_norm = measure_operator_norm(baseline_checkpoint)  # e.g., 4.78

# Grassmann forward pass
W_normalized = project_to_grassmann(W)  # ||W|| = 1
output = baseline_norm × (W_normalized @ x)  # Scale to match baseline
```

**With gating:**
```python
attn_out = attention(q, k, v)
gate = sigmoid(gate_proj(x))  # Mean ≈ 0.116 !
gated_out = attn_out * gate   # Effective scale ≈ 0.116 × baseline_norm
output = c_proj(gated_out)     # Too small!
```

**Result:**
- Intended scale: 4.78
- Actual scale after gating: 4.78 × 0.116 = **0.55** (88% too small!)
- Gradients too small
- Condition number collapses
- Training instability

### Solution: Pre-Amplification

**Key insight:** Gate values are in [0, 1] with mean ≈ 0.1-0.2

**Amplification strategy:**
```python
expected_gate_mean = 0.116  # From Table 4
amplification_factor = 1.0 / expected_gate_mean  # ≈ 8.6

# BEFORE gating is applied
grassmann_scale_amplified = baseline_norm × amplification_factor

# Forward pass
attn_out_scaled = grassmann_scale_amplified × (W_grassmann @ x)
gated_out = attn_out_scaled * gate
# Effective scale = (baseline_norm × 8.6) × 0.116 ≈ baseline_norm ✓
```

**Result:**
- Pre-amplified scale: 4.78 × 8.6 = **41.1**
- After gating: 41.1 × 0.116 = **4.77** ✓ (matches baseline!)

### Empirical Amplification Factors

Based on Table 4 statistics:

| Gating Type | Gate Mean | Recommended Amplification |
|-------------|-----------|---------------------------|
| SDPA Element-wise | 0.116 | **8.6×** |
| SDPA Head-wise | 0.172 | **5.8×** |
| Value Element-wise | 0.221 | **4.5×** |

**Conservative approach:** Use **6-8× amplification** universally

---

## Implementation Plan

### Phase 1: Gating Module (No Grassmann)

**Goal:** Validate gating works standalone before combining with Grassmann

**A. Implement Gated Attention Class:**

```python
# model.py
class GatedCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Standard attention components
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Gating mechanism (head-specific, element-wise)
        if config.use_gating:
            # Option 1: Element-wise (paper's best)
            if config.gate_granularity == 'elementwise':
                self.gate = nn.Linear(config.n_embd, config.n_embd, bias=False)
            # Option 2: Head-wise (fewer params)
            elif config.gate_granularity == 'headwise':
                self.gate = nn.Linear(config.n_embd, config.n_head, bias=False)
            else:
                raise ValueError(f"Unknown gate_granularity: {config.gate_granularity}")
        else:
            self.gate = None

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        # Q, K, V projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, hs)

        # APPLY GATING HERE (G1 position - after SDPA, before output proj)
        if self.gate is not None:
            # Compute gate from input (query-dependent)
            gate_input = x  # Use pre-norm hidden states

            if hasattr(self, 'gate'):
                if self.gate.out_features == self.n_embd:
                    # Element-wise gating
                    gate_scores = torch.sigmoid(self.gate(gate_input))  # (B, T, C)
                    gate_scores = gate_scores.view(B, T, self.n_head, C // self.n_head)
                    gate_scores = gate_scores.transpose(1, 2)  # (B, nh, T, hs)
                else:
                    # Head-wise gating
                    gate_scores = torch.sigmoid(self.gate(gate_input))  # (B, T, nh)
                    gate_scores = gate_scores.transpose(1, 2).unsqueeze(-1)  # (B, nh, T, 1)

                # Apply gate (multiplicative)
                y = y * gate_scores

        # Concatenate heads and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.c_proj(y))
        return y
```

**B. Config Parameters:**

```python
# config/train_openwebtext_baseline_gated.py
# Test gating alone (no width scaling, no Grassmann)

n_layer = 6
n_embd = 384
n_head = 6

# Gating config
use_gating = True
gate_granularity = 'elementwise'  # 'elementwise' or 'headwise'

# Training
learning_rate = 1e-3  # Standard baseline LR
max_iters = 100000
```

**C. Expected Results (from paper):**

At d=384, 400B tokens:
- Baseline: val PPL ≈ 7.5
- With gating: val PPL ≈ 7.4 (0.1 reduction)
- Better stability, fewer loss spikes

---

### Phase 2: Grassmann + Gating Integration

**A. Modified Grassmann Forward Pass:**

```python
# manifold/grassmann_ops.py

def grassmann_forward_with_gating(x, W, scale, gate_mean_estimate=0.15):
    """
    Forward pass for Grassmann-constrained weight with gating compensation.

    Args:
        x: Input tensor
        W: Weight matrix (already on Grassmann manifold, ||W|| = 1)
        scale: Measured operator norm from baseline
        gate_mean_estimate: Expected mean of gate values (default 0.15)

    Returns:
        Pre-amplified output (will be scaled down by gating)
    """
    # Amplification factor to compensate for gating
    amplification = 1.0 / gate_mean_estimate

    # Amplified scale
    scale_amplified = scale * amplification

    # Forward pass with amplified scale
    output = scale_amplified * (W @ x)

    return output
```

**B. Integration in Model:**

```python
class GatedGrassmannAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... standard setup ...

        # Grassmann config
        self.use_grassmann = config.use_grassmann
        self.use_gating = config.use_gating

        if self.use_grassmann:
            # Store operator norm scale
            self.register_buffer('grassmann_scale',
                               torch.tensor(config.grass_scale_c_attn))

            # Gate mean estimate for amplification
            self.gate_mean_estimate = config.get('gate_mean_estimate', 0.15)

    def forward(self, x):
        # ... Q, K, V, attention ...

        # After SDPA
        attn_out = att @ v  # (B, nh, T, hs)

        # Apply gating (G1 position)
        if self.use_gating:
            gate = torch.sigmoid(self.gate(x))
            # ... reshape gate ...
            attn_out = attn_out * gate

        # Concatenate heads
        y = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # Grassmann output projection with amplified scaling
        if self.use_grassmann:
            # Amplification factor
            amp = 1.0 / self.gate_mean_estimate if self.use_gating else 1.0
            scale_amplified = self.grassmann_scale * amp

            # Apply Grassmann projection with amplified scale
            y = grassmann_linear(y, self.c_proj.weight, scale_amplified)
        else:
            y = self.c_proj(y)

        return self.dropout(y)
```

**C. Config for Grassmann + Gating:**

```python
# config/train_openwebtext_grassmann_gated_gpt2medium.py

# GPT-2 Medium architecture
n_layer = 24
n_embd = 1024
n_head = 16

# Both innovations enabled
use_grassmann = True
use_gating = True

# Grassmann config (will be measured from baseline)
grass_rank_c_attn = 512  # 50% of width
grass_scale_c_attn = 4.78  # TBD: measure from d=1024 baseline
grass_a = 1.0  # G_{1,0,r} for no-skip layers
grass_b = 0.0

# Gating config
gate_granularity = 'elementwise'
gate_mean_estimate = 0.15  # Conservative estimate

# CRITICAL: Amplification for Grassmann scaling
# This compensates for gate mean ≈ 0.15
# Effective scale = (scale × 6.67) × 0.15 ≈ scale ✓

# Learning rate
learning_rate = 6e-4  # GPT-2 Medium baseline
grass_lr = 4.8e-3      # 8× boost for Grassmann (if no gating instability)
```

---

### Phase 3: Empirical Tuning

**A. Measure Actual Gate Statistics:**

```python
# After training for a few thousand iterations
def measure_gate_statistics(model, dataloader, n_batches=100):
    """Measure actual gate value distributions."""
    gate_values = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            # Forward pass and collect gate values
            # (Add hooks to capture gate activations)
            ...
            gate_values.append(gates)

    gate_values = torch.cat(gate_values)

    stats = {
        'mean': gate_values.mean().item(),
        'std': gate_values.std().item(),
        'median': gate_values.median().item(),
        'p10': gate_values.quantile(0.1).item(),
        'p90': gate_values.quantile(0.9).item(),
    }

    return stats

# Adjust amplification factor based on measurements
measured_mean = measure_gate_statistics(...)['mean']
optimal_amplification = 1.0 / measured_mean
```

**B. Grid Search Amplification:**

If empirical measurement shows different gate mean than expected:

| Measured Gate Mean | Amplification Factor | Expected Effective Scale |
|--------------------|---------------------|--------------------------|
| 0.10 | 10.0× | 1.0× ✓ |
| 0.15 | 6.7× | 1.0× ✓ |
| 0.20 | 5.0× | 1.0× ✓ |
| 0.25 | 4.0× | 1.0× ✓ |

**C. Adaptive Amplification (Advanced):**

```python
class AdaptiveGatedGrassmann(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... setup ...

        # Learnable amplification (initialized to 1/expected_mean)
        init_amp = 1.0 / config.gate_mean_estimate
        self.log_amplification = nn.Parameter(torch.log(torch.tensor(init_amp)))

    def forward(self, x):
        # ... attention ...

        # Dynamic amplification
        amplification = torch.exp(self.log_amplification)
        scale_amplified = self.grassmann_scale * amplification

        # ... rest of forward ...
```

---

## Experimental Timeline

### Week 1: Gating Validation (Phase 1)

**Day 1-2:**
- Implement `GatedCausalSelfAttention`
- Create configs: `train_openwebtext_baseline_gated.py`
- Run sanity check (10k iters)

**Day 3-5:**
- Submit: `sbatch submit_openwebtext_baseline_gated.sh`
- Monitor gate statistics
- Verify: PPL improvement, no divergence

**Success Criteria:**
- ✅ Gate mean ≈ 0.1-0.2 (matches paper)
- ✅ PPL improvement over baseline
- ✅ Training stable

---

### Week 2-3: Baseline at d=1024 (Phase 2)

**Day 1:**
- Create GPT-2 Medium config: `train_openwebtext_baseline_gpt2medium.py`
- Verify parameter count (~350M)

**Day 2:**
- Submit baseline: `sbatch submit_openwebtext_baseline_gpt2medium.sh`

**Day 3-14:**
- Monitor training (100k iters, ~10-14 days on H100)
- Measure effective rank at d=1024
- Measure operator norms for Grassmann scaling

**Measurements needed:**
```python
# analyze_gpt2medium_baseline.py
# At convergence (~100k iters):
# 1. Effective rank per layer
# 2. Operator norms: c_attn, mlp.c_fc, c_proj
# 3. Rank/width ratio (confirm if < 50%)
```

**Success Criteria:**
- ✅ Baseline converges to val loss ~6.0-6.5
- ✅ Effective rank measured
- ✅ Rank/width ratio confirms width is sufficient

---

### Week 3-4: Grassmann + Gating (Phase 3)

**Day 1-2:**
- Implement gating with amplified Grassmann scaling
- Create config: `train_openwebtext_grassmann_gated_gpt2medium.py`
- Use measured norms from baseline

**Day 3:**
- Run quick validation (5k iters)
- Check gate statistics match expectations
- Verify amplification is correct

**Day 4:**
- Submit full training: `sbatch submit_openwebtext_grassmann_gated_gpt2medium.sh`

**Day 5-14:**
- Monitor training
- Track: gate mean, effective scales, stability
- Compare to baseline

**Success Criteria:**
- ✅ No divergence or plateau
- ✅ Match or beat baseline
- ✅ Better training stability (fewer spikes)

---

## Risk Mitigation

### Risk 1: Gate Values Differ from Paper

**Symptom:** Measured gate mean ≠ 0.15

**Mitigation:**
1. Log gate statistics every 1k iters
2. Adjust amplification factor dynamically
3. Use adaptive amplification (learnable)

### Risk 2: Amplification Too Aggressive

**Symptom:** Gradients explode, training unstable

**Mitigation:**
1. Start with conservative amplification (4-5×)
2. Gradually increase if stable
3. Add gradient clipping (clip_norm=1.0)

### Risk 3: Gating Conflicts with Grassmann

**Symptom:** Performance worse than either alone

**Mitigation:**
1. Test gating alone first (Phase 1)
2. Test Grassmann alone at d=1024
3. Only combine if both work independently

---

## Success Metrics

### Minimum Success:
- ✅ Gating improves baseline by 0.1 PPL
- ✅ Grassmann at d=1024 stable (no divergence)
- ✅ Combined approach matches baseline

### Strong Success:
- ✅ Combined approach beats baseline by 1-2%
- ✅ Better generalization gap
- ✅ Stable with 8× LR boost

### Excellent Success:
- ✅ Beats baseline by 2-5%
- ✅ Eliminates attention sink (like paper)
- ✅ Better long-context performance

---

## Implementation Checklist

### Code Changes:

- [ ] **model.py:**
  - [ ] Add `GatedCausalSelfAttention` class
  - [ ] Add gating to `configure_optimizers_grassmann`
  - [ ] Implement amplified Grassmann scaling

- [ ] **train.py:**
  - [ ] Add gating config parameters
  - [ ] Add gate statistics logging
  - [ ] Add amplification factor tracking

- [ ] **manifold/grassmann_ops.py:**
  - [ ] Add `grassmann_forward_with_gating` function
  - [ ] Add amplification parameter

### Configs:

- [ ] `config/train_openwebtext_baseline_gated.py` (Phase 1)
- [ ] `config/train_openwebtext_baseline_gpt2medium.py` (Phase 2)
- [ ] `config/train_openwebtext_grassmann_gated_gpt2medium.py` (Phase 3)

### Analysis Scripts:

- [ ] `analyze_gate_statistics.py` - measure gate distributions
- [ ] `analyze_gpt2medium_baseline.py` - measure ranks/norms at d=1024
- [ ] `verify_amplification.py` - verify effective scales match baseline

### Submission Scripts:

- [ ] `submit_openwebtext_baseline_gated.sh`
- [ ] `submit_openwebtext_baseline_gpt2medium.sh`
- [ ] `submit_openwebtext_grassmann_gated_gpt2medium.sh`

---

## Key Equations Reference

### 1. Gating Mechanism (Eq. 5)
```
Y' = Y ⊙ σ(X W_θ)

where:
  Y = SDPA output
  X = hidden states (after pre-norm)
  σ = sigmoid
  W_θ = learnable gate parameters
```

### 2. Grassmann Scaling (Modified)
```
output = scale_amp × (W_grass @ x)

where:
  scale_amp = baseline_norm × (1 / gate_mean)
  W_grass = Grassmann-constrained weight (||W|| = 1)
  gate_mean ≈ 0.15 (empirical)

Effective scale after gating:
  = scale_amp × gate_mean
  = baseline_norm × (1 / gate_mean) × gate_mean
  = baseline_norm ✓
```

### 3. Amplification Factor Selection
```
amplification = 1.0 / measured_gate_mean

Conservative (initial): amp = 6.0  (assumes gate_mean = 0.167)
Moderate (paper):       amp = 8.6  (assumes gate_mean = 0.116)
Aggressive (sparse):    amp = 10.0 (assumes gate_mean = 0.100)
```

---

## References

**Primary Paper:**
- Qiu et al. "Gated Attention for Large Language Models" (NeurIPS 2025)
- arXiv:2505.06708

**Key Tables:**
- Table 1: Gating variant performance (page 4)
- Table 2: Training stability results (page 5)
- Table 4: Gate statistics (page 7)

**Key Figures:**
- Figure 1: Gating positions and performance (page 2)
- Figure 2: Attention sink elimination (page 3)
- Figure 3: Gate score distributions (page 7)

---

**Status:** Implementation plan complete, ready to code
**Next Step:** Implement `GatedCausalSelfAttention` in model.py
