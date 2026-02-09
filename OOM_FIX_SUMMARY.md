# OOM Issue Resolution

**Date:** Jan 5, 2026
**Issue:** QKV-only component experiment (job 613766) failed with OOM error

---

## Root Cause Analysis

### 1. **GPU Type Mismatch**

| Script Type | GPU Request | Memory | Status |
|-------------|-------------|--------|--------|
| **Tier 1 scripts** | `--gres=gpu:h100:4` | 80GB per GPU | ✅ Worked with batch=16 |
| **Component scripts (original)** | `--gres=gpu:4` | Unknown (could be A100 40GB) | ❌ Failed with batch=32 |

**Problem:** Component scripts didn't specify GPU type, so SLURM allocated whatever was available (likely A100 40GB on node g004 instead of H100 80GB).

### 2. **Memory Requirements**

**16×16 Full Block Decomposition (QKV-only):**
- 3 separate FullBlockDecomposedLinear layers (Q, K, V)
- Each: 16×16 = 256 blocks of 64×64 weights
- Total: 768 blocks per layer × 24 layers = **18,432 Grassmann blocks**
- Plus: 16 gates per layer for gating
- **Very memory intensive!**

**Comparison:**
- Baseline: Standard nn.Linear, ~1GB per GPU
- MLP-only: 4 blocks per layer, minimal memory increase
- **QKV-only: 768 blocks per layer, ~2-3GB extra per GPU**

### 3. **Batch Size Impact**

| Config | Batch | Grad Accum | Memory per GPU | Status |
|--------|-------|------------|----------------|--------|
| Tier 1 full block | 16 | 24 | ~42GB | ✅ Worked on H100 |
| QKV-only (failed) | 32 | 12 | **~80GB** | ❌ OOM on g004 |
| QKV-only (fixed) | 16 | 24 | ~42GB | ⏳ To test |

---

## Solutions Applied

### ✅ 1. Specify H100 GPU Type
Updated all component scripts:
```bash
# Before
#SBATCH --gres=gpu:4

# After
#SBATCH --gres=gpu:h100:4
```

**Files modified:**
- `submit_component_qkv_only.sh`
- `submit_component_mlp_only.sh`
- `submit_component_baseline.sh`

### ✅ 2. Reduce Batch Size for QKV-only
Updated config:
```python
# Before
batch_size = 32
gradient_accumulation_steps = 12

# After (maintains same tokens/iter)
batch_size = 16  # Halved to reduce memory
gradient_accumulation_steps = 24  # Doubled to maintain 1.57M tokens/iter
```

**File modified:** `config/train_gpt2_medium_grassmann_qkv_only_r384.py`

---

## Why H100/H200?

**H100 Advantages:**
1. **80GB memory** vs A100 40GB (2× more capacity)
2. **~35% faster** than A100 (from CLAUDE.md)
3. **More stable** for large models with many blocks

**Verification:**
```bash
# Check GPU type for running jobs
scontrol show node [nodename] | grep Gres
```

---

## Remaining Issue: MLP-only Slowness

**Observation:**
- MLP-only (613767): **MFU 23%**, very slow progress
- Baseline (613768): **MFU 124%**, normal speed
- **10× performance difference!**

**Possible causes:**
1. **Compilation issue** with BlockDecomposedLinear
2. **Grassmann update overhead** (dual ascent iterations)
3. **Node-specific issue** (j003-ds vs o001)
4. **Memory bandwidth** bottleneck

**Investigation needed:**
- Check if Grassmann updates are taking too long
- Verify compilation succeeded
- Compare hardware specs of j003-ds vs o001

**Recommendation:** Cancel 613767 and restart on H100 to see if performance improves.

---

## Summary of Changes

✅ **Tier 1 experiments canceled** (610974, 611029)
✅ **Component scripts updated** to request H100
✅ **QKV-only config updated** to use batch=16
✅ **Ready to resubmit** QKV-only experiment

**Next steps:**
1. Resubmit QKV-only (should work with batch=16 + H100)
2. Investigate MLP-only slowness (cancel and restart on H100?)
3. Let baseline continue (progressing well)
