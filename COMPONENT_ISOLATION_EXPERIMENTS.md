# Component Isolation Experiments

**Goal:** Determine where Grassmann manifold constraints provide benefit in transformer architecture.

**Key Insight:** Success = selective improvement in specific components, not maximum coverage.

---

## Experiment Design

| Experiment | c_attn (QKV) | c_fc (MLP) | Job ID | Config |
|------------|--------------|------------|--------|--------|
| **QKV-only** | Grassmann (16×16) | AdamW | 613766 | `train_gpt2_medium_grassmann_qkv_only_r384.py` |
| **MLP-only** | AdamW | Grassmann (4-block) | 613767 | `train_gpt2_medium_grassmann_mlp_only_r384.py` |
| **Baseline** | AdamW | AdamW | 613768 | `train_gpt2_medium_baseline_component_test.py` |
| **Full (reference)** | Grassmann (16×16) | Grassmann (4-block) | 610970 | Best from Tier 1: 3.1014 @ 45.6B tokens |

---

## Unified Hyperparameters

All experiments use identical training configuration:

```python
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.1
learning_rate = 6e-4
grass_lr = 2e-3
gate_lr = 1.2e-3
warmup_iters = 2000
batch_size = 32
gradient_accumulation_steps = 12
tokens_per_iter = 1,572,864
max_iters = 13000  # ~20B tokens
```

---

## Architecture Details

### QKV-only (Job 613766)
- **c_attn:** Full 16×16 block decomposition
  - 256 Grassmann blocks per layer (64×64 each, rank=32)
  - Per-head gating (16 gates)
  - Total: 6,144 Grassmann blocks
- **c_fc:** Standard AdamW nn.Linear
- **Hypothesis:** Multi-head attention benefits from manifold constraint (head diversity, orthogonality)

### MLP-only (Job 613767)
- **c_attn:** Standard AdamW nn.Linear
- **c_fc:** 4-block decomposition
  - 4 Grassmann blocks per layer (1024×1024 each, rank=384)
  - Block-level gating (4 gates)
  - Total: 96 Grassmann blocks
- **Hypothesis:** Feedforward benefits from low-rank regularization

### Baseline (Job 613768)
- **c_attn:** Standard AdamW nn.Linear
- **c_fc:** Standard AdamW nn.Linear
- **Purpose:** Control group

---

## Expected Outcomes

### Scenario 1: QKV-only ≈ Full > MLP-only ≈ Baseline
**Interpretation:** Grassmann helps primarily in attention
- Multi-head structure benefits from geometric constraints
- MLP doesn't need manifold constraint
- **Recommendation:** Use Grassmann only for c_attn in production

### Scenario 2: MLP-only ≈ Full > QKV-only ≈ Baseline
**Interpretation:** Grassmann helps primarily in MLP
- Feedforward benefits from low-rank regularization
- Attention doesn't need manifold constraint
- **Recommendation:** Use Grassmann only for c_fc in production

### Scenario 3: QKV-only ≈ MLP-only > Baseline, but both < Full
**Interpretation:** Grassmann needs both components
- Constraints work synergistically
- Removing either component degrades performance
- **Recommendation:** Use full Grassmann architecture

### Scenario 4: Baseline ≈ QKV-only ≈ MLP-only ≈ Full
**Interpretation:** Grassmann provides marginal benefit
- Constraint doesn't significantly help
- Consider simpler alternatives

### Scenario 5: Baseline > All Grassmann variants
**Interpretation:** Grassmann is net negative for transformers at scale
- Constraint harms performance
- **Recommendation:** Abandon Grassmann approach for this domain

---

## Success Criteria

**Primary:**
- Identify which component(s) benefit from Grassmann
- Quantify magnitude of benefit (% improvement over baseline)

**Secondary:**
- Understand sample efficiency (early stage performance)
- Assess long-term scaling behavior
- Compare parameter efficiency

---

## Monitoring

**Quick checkpoint (1.57B tokens, ~1k iters):**
```bash
# Check early progress
for job in 613766 613767 613768; do
  echo "=== Job $job ===" && grep "step 1000:" logs/${job}_*.out
done
```

**Mid-checkpoint (6.29B tokens, ~4k iters):**
```bash
# Check convergence trends
for job in 613766 613767 613768; do
  echo "=== Job $job ===" && grep "step 4000:" logs/${job}_*.out
done
```

**Final checkpoint (20.45B tokens, ~13k iters):**
```bash
# Compare final performance
python compare_component_experiments.py
```

---

## Code Changes

Modified `model.py` to support selective Grassmann application:

```python
# New config options
use_grassmann_c_attn = True/False  # Enable Grassmann for c_attn
use_grassmann_c_fc = True/False    # Enable Grassmann for c_fc

# CausalSelfAttention checks use_grassmann_c_attn
# MLP checks use_grassmann_c_fc
```

This allows component isolation without changing model architecture.

---

## Timeline

- **Start:** Jan 5, 2026 11:00 (jobs submitted)
- **Quick check:** Jan 5, 14:00 (~1.5B tokens)
- **Mid check:** Jan 6, 02:00 (~6B tokens)
- **Final:** Jan 6, 23:00 (~20B tokens)

Each job: 12 hours × ~2 sessions = ~24 hours total

---

## References

- **Tier 1 results:** 24L full block best at 3.1014 @ 45.6B tokens
- **Elementwise results:** 3.3081 @ 7.86B tokens (but 176M extra params)
- **Baseline (old):** 2.8618 @ 75.5B tokens (incomplete, different batch size)

---

**Last Updated:** Jan 5, 2026 11:05
**Status:** Running
**Next Action:** Monitor at ~1.5B tokens for early signal
