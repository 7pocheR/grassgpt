# Component Isolation Experiments - Progress Update

## ✅ FIX VERIFIED - Component Isolation Working!

**Jobs:**
- 625870 (QKV-only): Running on m002 (H100)
- 625871 (MLP-only): Running on m001 (H100)

**Initialization verification:**

| Config | c_attn Grassmann? | c_fc Grassmann? | Status |
|--------|-------------------|-----------------|---------|
| QKV-only (625870) | ✅ YES (72 blocks) | ❌ NO (AdamW) | ✅ Correct |
| MLP-only (625871) | ❌ NO (AdamW) | ✅ YES (96 blocks) | ✅ Correct |

**Previous buggy experiments (615947, 615948):**
- Both initialized c_attn AND c_fc with Grassmann
- Were identical to 24L Hybrid despite different configs
- Results INVALID

---

## Training Progress (as of ~1.5 hours runtime)

### QKV-only (625870)
- **Progress:** 690 iters | 1.09B tokens
- **Speed:** 7.0 sec/iter | MFU 10.9%
- **Node:** m002 (H100)

### MLP-only (625871)
- **Progress:** 320 iters | 0.50B tokens  
- **Speed:** 3.9 sec/iter | MFU 38.6%
- **Node:** m001 (H100)

---

## Performance Analysis

**MLP-only is 1.8× FASTER than QKV-only:**
- MLP-only: 3.9 sec/iter, MFU 38.6%
- QKV-only: 7.0 sec/iter, MFU 10.9%

**Why QKV-only slower?**
1. Attention computation more complex than MLP
2. 48 head-level gates vs 4 block-level gates
3. Multi-head attention requires reshaping/transpose ops

---

## Timeline Projections

**12-hour timeout limit:**

| Config | Current | Target | Hours Needed | Will Complete? |
|--------|---------|--------|--------------|----------------|
| QKV-only | 690 iters | 13000 iters | ~24 hours | ❌ NO (~6100 iters max) |
| MLP-only | 320 iters | 13000 iters | ~14 hours | ❌ NO (~11000 iters max) |

**Expected at timeout:**
- QKV-only: ~6141 iters (**9.7B tokens**)
- MLP-only: ~11000 iters (**17.3B tokens**)

---

## Comparison with References

**Baseline (613768):**
- ✅ Complete: 2.9581 @ 20.45B tokens
- MFU: ~120% (pure AdamW, no Grassmann overhead)

**24L Hybrid (608059):**
- Reference: 3.3017 @ 12.58B tokens
- Both c_attn + c_fc Grassmann

**At comparable token counts, we can compare:**
- Baseline @ 9.7B: ~3.04 (interpolated)
- QKV-only @ 9.7B: TBD (will know in ~10.5 hours)
- MLP-only @ 17.3B: TBD (will know in ~11.5 hours)

---

## Next Steps

**Option 1: Let them run to timeout**
- QKV-only will reach ~9.7B tokens (enough for early comparison)
- MLP-only will reach ~17.3B tokens (good coverage)
- Can compare with baseline and hybrid at matched token counts

**Option 2: Reduce max_iters in configs**
- Set target to achievable iters (e.g., 6000 for QKV, 11000 for MLP)
- But jobs already running, can't change mid-flight

**Option 3: Request continuation jobs**
- Submit dependent jobs to continue from checkpoints
- But need to verify results are worth continuing first

**Recommendation: Let current jobs complete to timeout**
- Will get data at 9.7B and 17.3B tokens
- Enough to evaluate component effects
- Can decide whether to continue based on results

---

**Status:** Both experiments running correctly with verified component isolation
**ETA:** ~10.5 hours for timeout
**Data quality:** Valid (fix verified)
