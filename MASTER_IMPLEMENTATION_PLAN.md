# Master Implementation Plan: Grassmann + Gating + Width Scaling

**Date:** December 25, 2024
**Goal:** Scale Grassmann manifold optimization to OpenWebText using gating + width scaling
**Status:** Ready to implement

---

## Problem Summary

**Current state (d=384):**
- Effective rank: 330 (86% of width) - nearly saturated
- Grassmann experiments failed (9-17% worse than baseline)
- Root causes:
  1. ❌ Width too small (rank ratio 86% vs CIFAR-10's 31%)
  2. ❌ Wrong operator norms (mlp.c_fc over-scaled by 35%)
  3. ❌ Training instability (plateau after 35k iters)

**Solution:**
1. ✅ Scale to **d=1024** (GPT-2 Medium) - creates rank/width gap
2. ✅ Add **gating mechanism** - improves stability, allows higher LR
3. ✅ **Amplify Grassmann scaling** - compensate for sparse gates (mean ≈ 0.12)

---

## The Three-Part Solution

### Part 1: Width Scaling (d → 1024)

**Why d=1024:**
```
Current d=384:  R_eff=330, ratio=86% (saturated!)
Target d=1024:  R_eff≈500, ratio≈49% (matches CIFAR-10's 31-50%)
                r=512 (50% of width)
```

**Expected improvement:**
- Creates capacity gap for low-rank constraint
- Rank ratio drops from 86% to ~50%

### Part 2: Gating Mechanism

**What (from Qwen NeurIPS 2025 paper):**
```python
attn_out = attention(q, k, v)          # SDPA
gate = sigmoid(W_gate @ x)              # Query-dependent gate
gated_out = attn_out * gate             # Multiplicative gating (G1 position)
output = W_O @ gated_out
```

**Why it helps:**
- ✅ Breaks low-rank bottleneck (W_V → W_O becomes non-linear)
- ✅ Induces sparsity (gate mean ≈ 0.12, 88% near zero)
- ✅ Eliminates attention sink (46% → 5% first-token attention)
- ✅ **Training stability** (tolerates 2× higher LR, no loss spikes)

**Key statistics (from paper Table 4):**
| Metric | Value |
|--------|-------|
| Gate mean | **0.116** |
| Sparsity | **88% near 0** |
| Performance | **-0.2 PPL** |

### Part 3: Amplified Grassmann Scaling

**The critical problem:**
```python
# Grassmann constraint: ||W|| = 1
# Standard scaling: output = baseline_norm × (W @ x)
# With gating: output = gate × baseline_norm × (W @ x)
#            = 0.116 × 4.78 × (W @ x)  # Only 0.55! (88% too small)
```

**The solution:**
```python
# PRE-AMPLIFY before gating
amplification = 1.0 / gate_mean_estimate  # 1/0.12 ≈ 8.3×
scale_amp = baseline_norm × amplification  # 4.78 × 8.3 = 39.7

output = scale_amp × (W_grassmann @ x)     # Amplified
gated = gate × output                       # 0.116 × 39.7 ≈ 4.6 ✓
```

**Result:** Effective scale matches baseline after gating!

---

## Phased Implementation (4 Weeks)

### Phase 1: Baseline + Gating Validation (Week 1)

**Goal:** Prove gating works standalone before combining with Grassmann

**Tasks:**
1. Implement gated attention (no Grassmann, no width scaling)
2. Test at current d=384
3. Verify gate statistics match paper

**Implementation:**

```python
# model.py - Add gating to existing CausalSelfAttention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # ... existing code ...

        # ADD: Gating mechanism
        if config.use_gating:
            if config.gate_type == 'elementwise':
                self.gate = nn.Linear(config.n_embd, config.n_embd, bias=False)
            elif config.gate_type == 'headwise':
                self.gate = nn.Linear(config.n_embd, config.n_head, bias=False)
        else:
            self.gate = None

    def forward(self, x):
        B, T, C = x.size()

        # Q, K, V (unchanged)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention (unchanged)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, hs)

        # **NEW: Apply gating at G1 position (after SDPA)**
        if self.gate is not None:
            gate_input = x  # Use pre-norm hidden states

            if self.gate.out_features == self.n_embd:
                # Element-wise gating
                g = torch.sigmoid(self.gate(gate_input))  # (B, T, C)
                g = g.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            else:
                # Head-wise gating
                g = torch.sigmoid(self.gate(gate_input))  # (B, T, nh)
                g = g.transpose(1, 2).unsqueeze(-1)  # (B, nh, T, 1)

            y = y * g  # Multiplicative gating

        # Output projection (unchanged)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.c_proj(y))
        return y
```

**Config:**
```python
# config/test_gating_d384.py
n_layer = 6
n_embd = 384
n_head = 6

use_gating = True
gate_type = 'elementwise'  # or 'headwise'

learning_rate = 1e-3
max_iters = 20000  # Quick test
eval_interval = 1000
```

**Validation script:**
```python
# analyze_gate_stats.py
def measure_gate_statistics(model, dataloader, n_batches=100):
    """Measure gate value distribution."""
    gate_values = []

    for i, (x, y) in enumerate(dataloader):
        if i >= n_batches:
            break

        # Hook to capture gate values
        def hook(module, input, output):
            if hasattr(module, 'gate') and module.gate is not None:
                # Capture gate values during forward
                gate_input = input[0]  # x
                g = torch.sigmoid(module.gate(gate_input))
                gate_values.append(g.detach().cpu())

        handle = model.transformer.h[0].attn.register_forward_hook(hook)
        _ = model(x)
        handle.remove()

    gates = torch.cat(gate_values)

    print(f"Gate statistics:")
    print(f"  Mean: {gates.mean():.4f}")
    print(f"  Std:  {gates.std():.4f}")
    print(f"  P10:  {gates.quantile(0.1):.4f}")
    print(f"  P50:  {gates.quantile(0.5):.4f}")
    print(f"  P90:  {gates.quantile(0.9):.4f}")

    # Expected: mean ≈ 0.12-0.20
    return gates.mean().item()
```

**Commands:**
```bash
# Create config
vim config/test_gating_d384.py

# Run quick test
python train.py config/test_gating_d384.py

# After 5k iters, measure gates
python analyze_gate_stats.py --checkpoint out-test-gating/ckpt.pt

# Expected output:
#   Gate mean: 0.12-0.20 ✓
#   PPL improvement: ~0.05-0.1
```

**Success criteria:**
- ✅ Gate mean ≈ 0.10-0.20 (matches paper)
- ✅ Training stable, no divergence
- ✅ Small PPL improvement over baseline

**Timeline:** 2-3 days

---

### Phase 2: GPT-2 Medium Baseline (Week 2-3)

**Goal:** Train baseline at d=1024, measure effective ranks and operator norms

**Config:**
```python
# config/train_gpt2_medium_baseline.py
# GPT-2 Medium architecture
n_layer = 24
n_embd = 1024
n_head = 16
n_embd = 1024

# Standard settings (from GPT-2 paper)
learning_rate = 6e-4
max_iters = 100000
batch_size = 8
gradient_accumulation_steps = 80  # Total batch ~640K tokens

# Dataset
dataset = 'openwebtext'
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2-medium-baseline'
```

**Measurement script:**
```python
# analyze_gpt2_medium_norms.py
import torch
import numpy as np

def analyze_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = ckpt['model']

    results = {}

    for name, param in model.items():
        if 'weight' in name and param.ndim == 2:
            # SVD analysis
            U, S, Vh = torch.linalg.svd(param, full_matrices=False)

            # Effective rank
            S_normalized = S / S.sum()
            entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
            eff_rank = torch.exp(entropy).item()

            # Rank at variance thresholds
            cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
            rank_50 = (cumvar < 0.50).sum().item()
            rank_85 = (cumvar < 0.85).sum().item()
            rank_95 = (cumvar < 0.95).sum().item()

            # Operator norm
            op_norm = S[0].item()

            results[name] = {
                'shape': param.shape,
                'eff_rank': eff_rank,
                'rank_50': rank_50,
                'rank_85': rank_85,
                'rank_95': rank_95,
                'op_norm': op_norm,
                'rank_width_ratio': eff_rank / param.shape[0]
            }

            print(f"\n{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Effective rank: {eff_rank:.1f} ({eff_rank/param.shape[0]:.1%} of width)")
            print(f"  Rank@50%: {rank_50}, @85%: {rank_85}, @95%: {rank_95}")
            print(f"  Operator norm: {op_norm:.4f}")

    # Average by layer type
    for layer_type in ['c_attn', 'mlp.c_fc', 'attn.c_proj', 'mlp.c_proj']:
        matching = [v for k, v in results.items() if layer_type in k]
        if matching:
            avg_eff_rank = np.mean([m['eff_rank'] for m in matching])
            avg_op_norm = np.mean([m['op_norm'] for m in matching])
            avg_ratio = np.mean([m['rank_width_ratio'] for m in matching])

            print(f"\n{layer_type} (averaged):")
            print(f"  Effective rank: {avg_eff_rank:.1f} ({avg_ratio:.1%} of width)")
            print(f"  Operator norm: {avg_op_norm:.4f}")

    return results

# Run analysis
results = analyze_checkpoint('/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2-medium-baseline/ckpt.pt')

# Expected for d=1024:
#   c_attn effective rank: ~500-600 (49-59% of width) ✓
#   This is much better than d=384's 86%!
```

**Commands:**
```bash
# Create config
vim config/train_gpt2_medium_baseline.py

# Submit (long job - 10-14 days)
sbatch submit_gpt2_medium_baseline.sh

# Monitor
tail -f logs/*gpt2_medium_baseline*.out

# When done, analyze
python analyze_gpt2_medium_norms.py
```

**Success criteria:**
- ✅ Converges to val loss ~6.0-6.5
- ✅ Effective rank ~500-600 at d=1024
- ✅ Rank/width ratio ~50-60% (better than 86% at d=384!)

**Timeline:** 10-14 days training + 1 day analysis

---

### Phase 3: Grassmann + Gating Integration (Week 4)

**Goal:** Combine all three components with proper amplification

**Implementation:**

```python
# manifold/grassmann_ops.py - ADD AMPLIFICATION

def grassmann_linear_gated(x, W, scale, gate_mean=0.15):
    """
    Grassmann linear layer with gating compensation.

    Args:
        x: Input (B, T, C)
        W: Grassmann weight (C, C_out), ||W|| = 1
        scale: Measured operator norm from baseline
        gate_mean: Expected gate mean (default 0.15)

    Returns:
        Output with amplified scale (will be reduced by gating)
    """
    # Amplification to compensate for gating
    amplification = 1.0 / gate_mean
    scale_amplified = scale * amplification

    # Linear projection with amplified scale
    return scale_amplified * (x @ W)
```

```python
# model.py - MODIFIED CausalSelfAttention

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Standard components
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Gating
        self.use_gating = config.use_gating
        if self.use_gating:
            self.gate = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Grassmann
        self.use_grassmann = config.use_grassmann
        if self.use_grassmann:
            # Store operator norm scale (measured from baseline)
            self.register_buffer('grass_scale', torch.tensor(config.grass_scale_c_attn))

            # Gate mean for amplification
            self.gate_mean_estimate = config.gate_mean_estimate

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

        # GATING (G1 position - after SDPA)
        if self.use_gating:
            g = torch.sigmoid(self.gate(x))  # (B, T, C)
            g = g.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            y = y * g  # Apply gate

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # OUTPUT PROJECTION with Grassmann + Amplification
        if self.use_grassmann:
            from manifold.grassmann_ops import grassmann_linear_gated

            # Use amplified scaling if gating is enabled
            gate_mean = self.gate_mean_estimate if self.use_gating else 1.0
            y = grassmann_linear_gated(y, self.c_proj.weight,
                                      self.grass_scale, gate_mean)
        else:
            y = self.c_proj(y)

        return self.dropout(y)
```

**Config:**
```python
# config/train_gpt2_medium_grassmann_gated.py

# GPT-2 Medium
n_layer = 24
n_embd = 1024
n_head = 16

# Enable both
use_grassmann = True
use_gating = True

# Gating config
gate_type = 'elementwise'
gate_mean_estimate = 0.15  # From Phase 1 measurement (or paper's 0.116)

# Grassmann config (use Phase 2 measurements)
grass_rank_c_attn = 512  # 50% of width

# CRITICAL: Use measured operator norms from Phase 2
grass_scale_c_attn = 5.2   # EXAMPLE - replace with actual measurement
grass_scale_mlp_fc = 8.1   # EXAMPLE - replace with actual measurement
grass_scale_c_proj = 3.0   # EXAMPLE - replace with actual measurement

# Parametrization
grass_a = 1.0  # G_{1,0,r} for no-skip layers (c_attn, mlp.c_fc)
grass_b = 0.0

# Learning rates
learning_rate = 6e-4      # Base LR for AdamW components
grass_lr = 4.8e-3          # 8× for Grassmann (may work with gating stability)

# Output
out_dir = '/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2-medium-grassmann-gated'
```

**Validation:**
```python
# verify_amplification.py
"""Verify effective scales match baseline after gating."""

def verify_scales(model, dataloader):
    """Check that gating + amplification produces correct scales."""

    # Collect activations
    attn_outs_baseline = []  # From baseline checkpoint
    attn_outs_gated = []      # From our model

    # ... collect activations ...

    # Measure effective scales
    baseline_scale = attn_outs_baseline.std().item()
    gated_scale = attn_outs_gated.std().item()

    print(f"Baseline scale: {baseline_scale:.4f}")
    print(f"Gated scale:    {gated_scale:.4f}")
    print(f"Ratio:          {gated_scale/baseline_scale:.4f}")

    # Should be close to 1.0
    assert abs(gated_scale / baseline_scale - 1.0) < 0.1, "Scale mismatch!"
```

**Commands:**
```bash
# After Phase 2 completes, update config with measured norms
vim config/train_gpt2_medium_grassmann_gated.py
# Replace grass_scale_* with actual measurements

# Quick sanity check (5k iters)
python train.py config/train_gpt2_medium_grassmann_gated.py --max_iters 5000

# Verify amplification
python verify_amplification.py

# If verification passes, submit full training
sbatch submit_gpt2_medium_grassmann_gated.sh
```

**Success criteria:**
- ✅ Training stable, no divergence
- ✅ Scales match baseline (verified)
- ✅ Match or beat baseline by 1-2%
- ✅ Better train/val gap

**Timeline:** 10-14 days

---

## File Structure

```
nanoGPT/
├── model.py                           # MODIFY: Add gating to CausalSelfAttention
├── train.py                           # MODIFY: Add config params
├── manifold/
│   ├── grassmann_ops.py              # MODIFY: Add grassmann_linear_gated
│   └── ...
├── config/
│   ├── test_gating_d384.py           # NEW: Phase 1 test
│   ├── train_gpt2_medium_baseline.py # NEW: Phase 2 baseline
│   └── train_gpt2_medium_grassmann_gated.py  # NEW: Phase 3 full solution
├── analyze_gate_stats.py              # NEW: Measure gate distributions
├── analyze_gpt2_medium_norms.py       # NEW: Measure ranks/norms at d=1024
└── verify_amplification.py            # NEW: Verify scales match
```

---

## Quick Reference

### Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Width** | 1024 | GPT-2 Medium |
| **Rank** | 512 | 50% of width |
| **Gate mean** | 0.12-0.20 | Measure in Phase 1 |
| **Amplification** | 5-8× | 1/gate_mean |
| **Base LR** | 6e-4 | GPT-2 paper |
| **Grassmann LR** | 4.8e-3 | 8× base (if stable) |

### Critical Equations

**Gating:**
```python
y' = y * sigmoid(W_gate @ x)
```

**Amplified Grassmann:**
```python
amp = 1.0 / gate_mean_estimate
scale_amp = baseline_norm × amp
output = scale_amp × (W_grassmann @ x)
# After gating: scale_amp × gate_mean ≈ baseline_norm ✓
```

---

## Risk Mitigation

### If Phase 1 fails (gating doesn't work):
- Check gate statistics (should be sparse, mean 0.1-0.2)
- Try head-wise instead of element-wise
- Verify implementation matches paper

### If Phase 2 shows R_eff still high (>70% of width):
- May need d=1536 instead
- Or proceed with lower rank target (r=256 instead of 512)

### If Phase 3 diverges:
- Reduce amplification factor (start with 4-5× instead of 8×)
- Check gate statistics in actual training
- Verify scales with `verify_amplification.py`
- Add gradient clipping

---

## Timeline Summary

| Phase | Duration | Blocking? | Tasks |
|-------|----------|-----------|-------|
| **Phase 1** | 2-3 days | No | Implement + test gating at d=384 |
| **Phase 2** | 10-14 days | Yes | Train baseline at d=1024, measure norms |
| **Phase 3** | 10-14 days | Yes | Train full solution |
| **Total** | **~4 weeks** | | End-to-end |

---

## Next Steps (Right Now)

1. **Implement Phase 1** (gating at d=384):
   ```bash
   # 1. Modify model.py - add gating to CausalSelfAttention
   vim model.py

   # 2. Create config
   vim config/test_gating_d384.py

   # 3. Run quick test
   python train.py config/test_gating_d384.py
   ```

2. **While Phase 1 runs, prepare Phase 2:**
   ```bash
   # Create baseline config for d=1024
   vim config/train_gpt2_medium_baseline.py

   # Create submission script
   vim submit_gpt2_medium_baseline.sh
   ```

3. **Prepare analysis scripts:**
   ```bash
   # Create measurement scripts
   vim analyze_gate_stats.py
   vim analyze_gpt2_medium_norms.py
   ```

---

**Status:** Ready to start Phase 1
**Action:** Implement gating in `model.py`
