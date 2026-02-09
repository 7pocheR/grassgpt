# Component Isolation Experiments - Corrected Resubmission

## Bug Fixed

**Root cause:** Config flags `use_grassmann_c_attn` and `use_grassmann_c_fc` were not being passed from config files to the model via `model_args` in train.py.

**Result:** Both QKV-only and MLP-only were accidentally running full Grassmann (c_attn + c_fc), making them identical to 24L Hybrid.

**Fix applied:**
1. Added component isolation flags to GPTConfig dataclass in model.py
2. Added flags to model_args dictionary in train.py  
3. Fixed None-handling logic in CausalSelfAttention and MLP classes

See `COMPONENT_ISOLATION_BUG_FIX.md` for complete details.

## Jobs Submitted (with FIX)

| Job ID | Config | Expected Behavior | Status |
|--------|--------|-------------------|--------|
| 620042 | QKV-only | c_attn: Grassmann, c_fc: AdamW | Pending (QOSMaxJobsPerUserLimit) |
| 620043 | MLP-only | c_attn: AdamW, c_fc: Grassmann | Pending (QOSMaxJobsPerUserLimit) |

## Previous Jobs (BUGGY - Ignore Results)

| Job ID | Config | Actual Behavior | Issue |
|--------|--------|-----------------|-------|
| 615947 | QKV-only | c_attn: Grassmann, c_fc: Grassmann | BUG: Ran full hybrid, timed out at 2k steps |
| 615948 | MLP-only | c_attn: Grassmann, c_fc: Grassmann | BUG: Ran full hybrid, reached 5k steps |

**Both buggy experiments were identical to 24L Hybrid!**

## Reference Jobs (For Comparison)

| Job ID | Config | Behavior | Status |
|--------|--------|----------|--------|
| 613768 | Baseline | Pure AdamW | ✅ Complete: 2.9581 @ 20.45B tokens |
| 608059 | 24L Hybrid | c_attn + c_fc Grassmann | Running: 3.3017 @ 12.58B tokens |

## Verification Plan

Once jobs 620042 and 620043 start, verify the fix by checking logs:

### QKV-only (620042) - SHOULD initialize:
```
✅ attn.c_attn.blocks.0-2   (Grassmann)
❌ mlp.c_fc.blocks          (should NOT appear - using AdamW)
```

### MLP-only (620043) - SHOULD initialize:
```
❌ attn.c_attn.blocks       (should NOT appear - using AdamW)  
✅ mlp.c_fc.blocks.0-3      (Grassmann)
```

## Expected Results (After Fix)

If component isolation works correctly, we should see:

### Hypothesis 1: c_fc Grassmann helps, c_attn hurts
- **Prediction:** MLP-only < 24L Hybrid < QKV-only < Baseline
- **Interpretation:** Use c_fc Grassmann only

### Hypothesis 2: c_attn Grassmann helps, c_fc hurts  
- **Prediction:** QKV-only < 24L Hybrid < MLP-only < Baseline
- **Interpretation:** Use c_attn Grassmann only

### Hypothesis 3: Both hurt independently, but combine better
- **Prediction:** 24L Hybrid < QKV-only ≈ MLP-only < Baseline
- **Interpretation:** Need both together (synergy)

### Hypothesis 4: Both hurt independently and together
- **Prediction:** Baseline < QKV-only ≈ MLP-only ≈ 24L Hybrid
- **Interpretation:** Don't use Grassmann at all

**Based on previous buggy results** (where both = full hybrid):
- Baseline always won (2.9581 vs 3.39+)
- This suggests Hypothesis 4 is likely
- But need corrected experiments to be sure!

## Timeline

- **2026-01-07 (current):** Fixed bug, resubmitted experiments
- **Next:** Wait for jobs to start (currently blocked by job limit)
- **Then:** Monitor initialization logs to verify fix
- **Finally:** Compare performance at 5B, 10B, 20B tokens

---
**Status:** Awaiting job start to verify fix worked correctly
**Confidence in fix:** High (tested logic, checked all code paths)
