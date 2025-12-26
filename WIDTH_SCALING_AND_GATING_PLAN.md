# Width Scaling + Gating: Improvement Plan

**Date:** December 25, 2024
**Context:** OpenWebText Grassmann experiments failed (9-17% worse than baseline)
**New Hypothesis:** Model width too small for effective rank; gating may improve stability

---

## Analysis of Current Failure

### Current Model Specs
```
n_embd = 384 (width)
n_layer = 6
n_head = 6
~30M parameters
```

### Effective Rank Problem

**OpenWebText measurements (from baseline @ 69k iters):**
```
Layer          Effective Rank    Rank/Width    Our Target Rank    Coverage
─────────────────────────────────────────────────────────────────────────
c_attn         329               85.7%         48                 14.6%
mlp.c_fc       331               86.2%         20                  6.0%
attn.c_proj    256               66.7%         48                 18.8%
mlp.c_proj     306               79.7%         20                  6.5%
```

**CIFAR-10 MLP (successful configuration):**
```
Layer          Width    Effective Rank    Rank/Width    Target Rank    Coverage
──────────────────────────────────────────────────────────────────────────────
Hidden         512      ~280              54.7%         160            31.2%
```

### Key Insight: Width Mismatch

**On CIFAR-10/Shakespeare:** Grassmann worked when effective rank was 30-50% of width
- CIFAR-10: rank 160 / width 512 = **31.2%**
- Shakespeare char: smaller model, similar ratio

**On OpenWebText:** Effective rank is 66-86% of width
- c_attn: 329 / 384 = **85.7%** (nearly full rank!)
- Using r=48 captures only 14.6% of effective structure

**Hypothesis:** OpenWebText data has richer structure (9B tokens vs 1.1M), requiring wider models for low-rank constraints to be effective.

---

## Solution 1: Width Scaling

### Mathematical Framework

**Goal:** Make effective rank R_eff = 30-50% of width d

Given:
- R_eff ≈ 330 (measured from OpenWebText)
- Want: R_eff / d = 0.3 to 0.5

Solve for d:
```
d = R_eff / target_ratio
d = 330 / 0.5 = 660 (conservative)
d = 330 / 0.3 = 1100 (aggressive)
```

**Recommended width: d = 768** (2× current, standard GPT-2 size)

### Width Scaling Projection

**If we use d = 768:**
- Effective rank (scaled): ~500-600 (empirical scaling R_eff ∝ √d)
- Target Grassmann rank: r = 384 (50% of 768)
- Rank/width ratio: 384/768 = **50%** ✓ (matches CIFAR-10 success!)

**If we use d = 1024:**
- Effective rank (scaled): ~600-700
- Target Grassmann rank: r = 512 (50% of 1024)
- Rank/width ratio: 512/1024 = **50%** ✓

**If we use d = 1536:**
- Effective rank (scaled): ~700-800
- Target Grassmann rank: r = 768 (50% of 1536)
- Rank/width ratio: 768/1536 = **50%** ✓

### Proposed Configurations

#### Config A: GPT-2 Small (Recommended)
```python
n_layer = 12       # 2× current
n_embd = 768       # 2× current
n_head = 12        # 2× current
# ~124M parameters (4× current)

# Grassmann config
grass_rank_c_attn = 384    # 50% of 768
grass_rank_mlp_fc = 384    # 50% of 768
grass_rank_c_proj = 384    # uniform for simplicity
grass_rank_mlp_proj = 384
```

**Rationale:**
- Standard GPT-2 Small size (well-tested architecture)
- 2× width allows 2× rank while maintaining 50% ratio
- 4× parameters compensates for parameter reduction from low-rank
- Effective rank should be ~500-600, so r=384 captures 64-77% (reasonable)

#### Config B: GPT-2 Medium (Conservative)
```python
n_layer = 24
n_embd = 1024
n_head = 16
# ~350M parameters

# Grassmann config
grass_rank = 512    # 50% of 1024 (uniform)
```

**Rationale:**
- Even wider, ensuring effective rank << total width
- Allows for aggressive low-rank constraint
- Standard GPT-2 Medium size

#### Config C: Custom Wide-Shallow
```python
n_layer = 6        # Keep depth same
n_embd = 1536      # 4× current width
n_head = 12
# ~120M parameters

# Grassmann config
grass_rank = 768    # 50% of 1536
```

**Rationale:**
- Maximizes width without exploding depth
- Maintains similar training time to current setup
- Tests "wide-shallow" hypothesis for Grassmann

---

## Solution 2: Gating Mechanism

### From "Gated Attention" (NeurIPS 2025 Oral, Qwen Team)

**Key Finding:** Adding a head-specific sigmoid gate after SDPA improves:
- Training stability (↓ loss spikes, tolerates higher LR)
- Final performance (lower loss)
- Long-context extrapolation
- Eliminates attention sink

**Architecture Modification:**
```python
# Standard attention
attn_output = softmax(Q @ K^T / √d) @ V

# Gated attention
attn_weights = softmax(Q @ K^T / √d)
gate = sigmoid(W_gate @ Q)  # Head-specific, query-dependent
attn_output = (attn_weights @ V) * gate  # Element-wise gating
```

**Relevance to Grassmann:**
1. **Sparsity:** Gate induces sparsity → naturally low-rank
2. **Non-linearity:** Adds non-linearity on top of low-rank mapping
3. **Stability:** May counteract Grassmann's gradient issues
4. **Higher LR:** Qwen team achieved stable training with higher LR (aligns with our 8× LR goal!)

### Integration with Grassmann

**Hypothesis:** Gating + Grassmann may be synergistic:
- **Gating:** Induces sparsity and non-linearity
- **Grassmann:** Enforces low-rank manifold constraint
- **Together:** Sparse low-rank structure with stable training

**Proposed Hybrid Architecture:**
```python
class GatedGrassmannAttention(nn.Module):
    def __init__(self, n_embd, n_head, grass_rank):
        # Q, K, V projections: Grassmann-constrained
        self.c_attn = BlockDecomposedLinear(n_embd, 3*n_embd,
                                            optimizer='grassmann',
                                            rank=grass_rank)

        # Gating: Learned sigmoid gate per head
        self.gate = nn.Linear(n_embd // n_head, 1)  # Per-head gate

        # Output projection: Grassmann-constrained
        self.c_proj = nn.Linear(n_embd, n_embd)  # Grassmann

    def forward(self, x):
        q, k, v = self.c_attn(x).split(n_embd, dim=-1)

        # Standard attention
        attn = F.softmax(q @ k.T / sqrt(d_head), dim=-1)
        attn_out = attn @ v

        # Apply head-specific gate
        gate = torch.sigmoid(self.gate(q))  # (batch, seq, n_head, 1)
        attn_out = attn_out * gate

        # Grassmann output projection
        return self.c_proj(attn_out)
```

**Benefits:**
- Gating adds only ~n_embd parameters (negligible)
- May stabilize Grassmann training (addresses our divergence issue!)
- Allows higher LR (addresses our 8× LR plateau)
- Qwen team validated on 3.5T tokens (massive scale)

---

## Proposed Experiments

### Experiment Set A: Width Scaling (No Gating)

**A1: Baseline at GPT-2 Small Scale**
```python
# config/train_openwebtext_baseline_gpt2small.py
n_layer = 12
n_embd = 768
n_head = 12
learning_rate = 6e-4  # Standard GPT-2 Small LR
max_iters = 100000
```

**A2: Grassmann at GPT-2 Small Scale**
```python
# config/train_openwebtext_grassmann_gpt2small.py
n_layer = 12
n_embd = 768
n_head = 12

# Grassmann Phase 2.5 (QKV + MLP)
use_grassmann = True
grassmann_phase = 'phase2.5'

# Width-scaled ranks (50% of width)
grass_rank_c_attn = 384
grass_rank_mlp_fc = 384

# CORRECTED operator norms (measure from A1 baseline first!)
# Placeholder: will measure from A1
grass_scale_c_attn = TBD
grass_scale_mlp_fc = TBD

# Standard LR (8× boost)
grass_lr = 4.8e-3  # 8× of 6e-4
```

**Expected Outcome:**
- ✅ If rank ratio matters: Grassmann should match/beat baseline
- ❌ If width doesn't help: Still fails despite correct ratio

---

### Experiment Set B: Gating (No Width Scaling)

**B1: Baseline + Gating (current width)**
```python
# config/train_openwebtext_baseline_gated.py
n_layer = 6
n_embd = 384
n_head = 6

# Add gating to attention
use_gating = True
gate_type = 'sigmoid'  # Per Qwen paper
```

**B2: Grassmann + Gating (current width)**
```python
# config/train_openwebtext_grassmann_gated.py
n_layer = 6
n_embd = 384
n_head = 6

# Both Grassmann AND gating
use_grassmann = True
use_gating = True

# Same ranks as before (may still be too low, but test stability)
grass_rank_c_attn = 182  # 85% variance
grass_rank_mlp_fc = 187

# Higher LR (gating should stabilize)
grass_lr = 8e-3  # Test if gating prevents plateau
```

**Expected Outcome:**
- ✅ If gating helps: Training stable, less divergence
- ✅ May allow higher LR without plateau
- ⚠️ Rank still low, but stability improved

---

### Experiment Set C: Width Scaling + Gating (Full Solution)

**C1: GPT-2 Small + Gating + Grassmann**
```python
# config/train_openwebtext_grassmann_gated_gpt2small.py
n_layer = 12
n_embd = 768
n_head = 12

# Both innovations
use_grassmann = True
use_gating = True

# Width-scaled ranks (50%)
grass_rank = 384  # Uniform

# Corrected norms (measure from A1)
grass_scale_c_attn = TBD
grass_scale_mlp_fc = TBD

# High LR (gating should stabilize)
grass_lr = 4.8e-3
```

**Expected Outcome:**
- ✅✅ Best case: Combines width scaling (rank ratio) + gating (stability)
- ✅ Should match or beat baseline
- ✅ Tests full hypothesis: width + stability = success

---

## Phased Implementation Plan

### Phase 1: Measure Baselines (Week 1)

**1A: Run GPT-2 Small Baseline**
```bash
sbatch submit_openwebtext_baseline_gpt2small.sh
# ~7-10 days to 100k iters
```

**1B: Analyze effective ranks**
```python
# analyze_gpt2small_baseline_norms.py
# Measure:
# - Effective rank at d=768
# - Operator norms
# - Confirm rank scaling hypothesis
```

**Success Criteria:**
- Effective rank ~500-600 (validates √d scaling)
- Rank/width ratio ~65-80% (better than 85% at d=384)
- Operator norms measured for Grassmann scaling

---

### Phase 2: Test Width Scaling (Week 2-3)

**Only proceed if Phase 1 confirms rank scaling**

**2A: Grassmann at GPT-2 Small (no gating)**
```bash
sbatch submit_openwebtext_grassmann_gpt2small.sh
# Use corrected norms from 1B
# Use r=384 (50% of width)
```

**Success Criteria:**
- No divergence or plateau (main goal!)
- Within 5% of baseline (acceptable)
- Better generalization gap than baseline

**If 2A succeeds:**
- ✅ Width scaling hypothesis CONFIRMED
- Proceed to Phase 3 (add gating)

**If 2A fails:**
- ❌ Width scaling insufficient
- Skip Phase 3, re-evaluate approach

---

### Phase 3: Add Gating (Week 4)

**Only proceed if Phase 2 succeeds**

**3A: Implement gating mechanism**
```python
# model.py
class GatedAttention:
    # Add head-specific sigmoid gate
    # Per Qwen NeurIPS 2025 paper
```

**3B: GPT-2 Small + Gating + Grassmann**
```bash
sbatch submit_openwebtext_grassmann_gated_gpt2small.sh
```

**Success Criteria:**
- Matches or beats baseline by 1-2%
- Stable with higher LR (test up to 10× boost)
- Better long-context performance

---

### Phase 4: Ablation Studies (Week 5)

**Test contributions independently:**

**4A: Gating only (no width scaling, no Grassmann)**
- Baseline at d=384 + gating
- Tests: Does gating help AdamW baseline?

**4B: Width scaling only (no gating)**
- Already done in Phase 2

**4C: Grassmann + Gating at d=384 (no width scaling)**
- Tests: Can gating fix instability without width scaling?

**Ablation Matrix:**

| Config | Width | Gating | Grassmann | Expected Performance |
|--------|-------|--------|-----------|---------------------|
| Baseline (current) | 384 | ❌ | ❌ | 3.57 val loss (reference) |
| A1 (GPT-2 baseline) | 768 | ❌ | ❌ | ~3.3-3.4 (better model) |
| A2 (Width + Grass) | 768 | ❌ | ✅ | Test width hypothesis |
| B1 (Baseline + Gate) | 384 | ✅ | ❌ | Test gating alone |
| B2 (Gate + Grass) | 384 | ✅ | ✅ | Test if gating fixes d=384 |
| C1 (Full solution) | 768 | ✅ | ✅ | **Best expected** |

---

## Resource Requirements

### Computational Cost

**Per experiment (GPT-2 Small, 100k iters):**
- Time: ~7-10 days on H100
- GPU-hours: ~168-240 hours
- Storage: ~1GB checkpoints

**Total for full plan:**
- Phase 1: 1 job (baseline)
- Phase 2: 1 job (width scaling test)
- Phase 3: 1 job (add gating)
- Phase 4: 3 jobs (ablations)
- **Total: 6 jobs × 7-10 days = 6-8 weeks**

**Parallelization:**
- Phase 1 + B1 (baseline + gated baseline): Can run in parallel (2 jobs)
- Phase 2 + B2: Can run after Phase 1 results (2 jobs in parallel)
- Phase 3: After Phase 2 succeeds
- Phase 4: All ablations in parallel (3 jobs)

**Optimized timeline: ~4 weeks** (with parallelization)

---

## Implementation Tasks

### Task 1: Gating Implementation (2-3 days)

**Files to modify:**
1. `model.py`: Add `GatedAttention` class
2. `train.py`: Add `use_gating` config flag
3. Config files: Create gated variants

**Code snippet:**
```python
# model.py
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Add gating (if enabled)
        if config.use_gating:
            self.gate = nn.Linear(config.n_embd // config.n_head, 1)
        else:
            self.gate = None

    def forward(self, x):
        # ... standard attention code ...

        # Apply gating
        if self.gate is not None:
            # Reshape to (B, T, n_head, d_head)
            y_heads = y.view(B, T, self.n_head, -1)

            # Compute head-specific gates
            gate = torch.sigmoid(self.gate(y_heads))  # (B, T, n_head, 1)

            # Apply gate
            y = (y_heads * gate).view(B, T, C)

        return self.c_proj(y)
```

---

### Task 2: Width Scaling Configs (1 day)

**Create config files:**
```python
# config/train_openwebtext_baseline_gpt2small.py
# config/train_openwebtext_grassmann_gpt2small.py
# config/train_openwebtext_grassmann_gated_gpt2small.py
```

**Template:**
```python
# GPT-2 Small base config
n_layer = 12
n_embd = 768
n_head = 12
dropout = 0.1

# Training config (from GPT-2 paper)
learning_rate = 6e-4
max_iters = 100000
lr_decay_iters = 100000
min_lr = 6e-5

# Dataset
dataset = 'openwebtext'
batch_size = 8  # Reduced for larger model
gradient_accumulation_steps = 80  # Total batch ~640K tokens

# Grassmann config (if enabled)
use_grassmann = True
grass_rank_c_attn = 384
grass_rank_mlp_fc = 384
grass_lr = 4.8e-3  # 8× base LR
```

---

### Task 3: Measurement Scripts (1 day)

**Create analysis scripts:**
```python
# analyze_gpt2small_baseline_norms.py
# Measures:
# - Effective rank vs width
# - Operator norms for Grassmann scaling
# - Validates rank/width hypothesis

import torch
import numpy as np

def analyze_effective_rank(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model = ckpt['model']

    for name, param in model.items():
        if 'weight' in name and param.ndim == 2:
            U, S, Vh = torch.svd(param)

            # Effective rank (entropy-based)
            S_normalized = S / S.sum()
            entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
            eff_rank = torch.exp(entropy).item()

            # Variance thresholds
            cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
            rank_50 = (cumvar < 0.50).sum().item()
            rank_85 = (cumvar < 0.85).sum().item()
            rank_95 = (cumvar < 0.95).sum().item()

            # Operator norm
            op_norm = S[0].item()

            print(f"{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Effective rank: {eff_rank:.1f}")
            print(f"  Rank@50%: {rank_50}, Rank@85%: {rank_85}, Rank@95%: {rank_95}")
            print(f"  Operator norm: {op_norm:.4f}")
            print(f"  Rank/Width: {eff_rank/param.shape[0]:.1%}")
```

---

## Risk Assessment

### High Risk: Width Scaling Fails
- **Probability:** 30-40%
- **Impact:** Width scaling alone insufficient, need other solutions
- **Mitigation:** Test gating in parallel (Experiment Set B)

### Medium Risk: Gating Doesn't Help Grassmann
- **Probability:** 20-30%
- **Impact:** Gating helps baseline but not Grassmann
- **Mitigation:** Focus on width scaling only

### Low Risk: Both Solutions Fail
- **Probability:** 10-15%
- **Impact:** Grassmann may not be suitable for OpenWebText at any scale
- **Mitigation:** Document negative result, explore other manifolds (Stiefel, etc.)

---

## Success Criteria

### Minimum Success (Phase 2):
- ✅ GPT-2 Small + Grassmann trains stably (no divergence)
- ✅ Within 5% of baseline
- ✓ Validates width scaling hypothesis

### Strong Success (Phase 3):
- ✅ GPT-2 Small + Grassmann + Gating matches baseline
- ✅ Stable with 8× LR
- ✓ Proves Grassmann viable at scale with proper architecture

### Excellent Success (Phase 4):
- ✅ Beats baseline by 1-2%
- ✅ Better generalization gap
- ✅ Ablations show both width and gating contribute
- ✓ Ready for production use

---

## Open Questions

1. **How does effective rank scale with width?**
   - Hypothesis: R_eff ∝ √d or R_eff ∝ d^α where α ≈ 0.5-0.7
   - Measure from baseline at d=384 vs d=768

2. **Is 50% rank ratio optimal?**
   - CIFAR-10 used 31% (r=160/512)
   - Should we try 30% (r=230/768) instead of 50% (r=384/768)?

3. **Does gating induce sparsity in attention?**
   - Measure gate activation statistics
   - Compare sparsity: baseline vs gated vs Grassmann vs both

4. **Can we predict rank/width ratio from data?**
   - Measure effective rank on various datasets
   - Develop formula: r_target = f(R_eff, d)

---

## Recommended Action

### Immediate (This Week):

1. **Implement gating mechanism** (Task 1)
   - Add to `model.py`
   - Create gated configs
   - Test on small sanity check

2. **Create GPT-2 Small configs** (Task 2)
   - Baseline, Grassmann, Gated, Full
   - Verify parameter counts

3. **Submit Phase 1 experiments:**
   ```bash
   sbatch submit_openwebtext_baseline_gpt2small.sh
   sbatch submit_openwebtext_baseline_gated.sh  # Parallel
   ```

### Next Week:

4. **Monitor Phase 1 progress**
   - Check for stable training
   - Measure effective ranks from baseline

5. **If Phase 1 looks good, submit Phase 2:**
   ```bash
   sbatch submit_openwebtext_grassmann_gpt2small.sh
   sbatch submit_openwebtext_grassmann_gated.sh  # Parallel
   ```

### Week 3-4:

6. **Analyze Phase 2 results**
   - Does width scaling work?
   - Does gating help?

7. **Submit Phase 3 if Phase 2 succeeds:**
   ```bash
   sbatch submit_openwebtext_grassmann_gated_gpt2small.sh
   ```

8. **Run ablations (Phase 4)**

---

## Alternative: Quick Validation

**If concerned about compute budget, run quick validation first:**

**Quick Test (2-3 days):**
```python
# config/test_width_scaling.py
n_layer = 6
n_embd = 768  # 2× width
n_head = 6
max_iters = 10000  # Short run

# Grassmann with width-scaled rank
grass_rank = 384
```

**Goal:** See if stability improves in first 10k iters
- If yes → proceed with full experiments
- If no → reconsider approach

---

## Conclusion

**Two promising solutions identified:**

1. **Width Scaling:** Increase model width to make effective rank a smaller fraction of total width (target 30-50% like CIFAR-10)

2. **Gating:** Add Qwen's gating mechanism for training stability and higher LR tolerance

**Best approach: Combine both** (Experiment Set C)

**Timeline:** 4 weeks with parallelization, 6-8 weeks sequential

**Next step:** Implement gating + create GPT-2 Small configs + submit Phase 1

---

**Document Status:** Active planning, ready for implementation
**Last Updated:** December 25, 2024
