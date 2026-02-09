# Training Progress Summary
**Generated:** Jan 5, 2026 08:45

---

## Component Isolation Experiments (New)

| Job | Experiment | Status | Progress | Val Loss | Issue |
|-----|------------|--------|----------|----------|-------|
| 613766 | QKV-only | ❌ FAILED | 2 min | - | **OOM**: 16×16 full block needs reduced batch size |
| 613767 | MLP-only | 🐌 SLOW | iter 660 @ 1.0B tokens | 5.3 | **10× slower than baseline** (MFU 23% vs 124%) |
| 613768 | Baseline | ✅ RUNNING | iter 6600 @ 10.4B tokens | 3.14 | Progressing normally |

### Analysis
1. **QKV-only OOM:** Need to reduce batch_size from 32→16, grad_accum 12→24 (like Tier 1 full block)
2. **MLP-only VERY slow:** Unexpected - 4-block MLP should not be this slow. Possible compile issue?
3. **Baseline progressing well:** Will reach 20B tokens in ~24 hours as planned

---

## Tier 1 Experiments (Ongoing)

| Job | Experiment | Status | Progress | Latest Val | Previous Best | Trend |
|-----|------------|--------|----------|------------|---------------|-------|
| 610974 | 36L r=384 hybrid | ✅ RUNNING | iter 18220 @ 28.7B tokens | 3.1097 @ 18k | 3.1263 @ 17k | ⬇️ **Improving** |
| 611029 | 24L r=512 hybrid | ✅ RUNNING | iter 25190 @ 39.6B tokens | 3.1554 @ 25k | 3.1560 @ 24k | ➡️ **Plateaued** |

### Full Tier 1 Timeline

**36L r=384 hybrid (job chain 610968→610974):**
- 5k iters: 3.3542
- 10k iters: 3.1947
- 15k iters: 3.1476
- 17k iters: 3.1263
- 18k iters: 3.1097 ← **Current, still improving**

**24L r=512 hybrid (job chain 610969→611029):**
- 4k iters: 3.5024
- 10k iters: 3.2888
- 15k iters: 3.2211
- 20k iters: 3.1891
- 24k iters: 3.1560
- 25k iters: 3.1554 ← **Current, plateaued**

**24L r=384 full block (completed at 611038):**
- 29k iters: 3.1014 @ 45.6B tokens ← **Best Tier 1 result**

---

## Comparison: Current Best Results

| Configuration | Tokens | Val Loss | Notes |
|---------------|--------|----------|-------|
| **24L r=384 full block** | 45.6B | **3.1014** | Best Tier 1, completed |
| 36L r=384 hybrid | 28.7B | 3.1097 | Still improving slowly |
| 24L r=512 hybrid | 39.6B | 3.1554 | Plateaued (higher rank hurts) |
| Baseline (component test) | 10.4B | 3.14 | Early, for component comparison |

---

## Recommendations

### 1. Component Experiments

**Immediate actions:**
- ✅ **Keep baseline running** (613768) - progressing well
- ❌ **QKV-only needs restart** (613766) - fix batch size (32→16, grad_accum 12→24)
- ⚠️ **MLP-only investigate** (613767) - why 10× slower? Check compile, consider canceling if no fix

**Alternative approach:**
Since QKV needs OOM fix and MLP is unexpectedly slow, consider simpler test:
- Use **hybrid gating** (3-block) for QKV instead of full 16×16
- Reduces memory, faster, still tests attention vs MLP

### 2. Tier 1 Experiments

**36L r=384 hybrid (610974):**
- **KEEP RUNNING** - still improving at 28.7B tokens (3.1097)
- Currently at iter 18k, could reach iter 20-25k before timeout
- Projected: ~3.08-3.09 @ 30-35B tokens (continuing downward trend)

**24L r=512 hybrid (611029):**
- **CONSIDER CANCELING** - plateaued at 3.1554 for 5k+ iters
- No improvement from 24k→25k iters (1.57B tokens)
- Confirms: **higher rank (r=512) hurts performance**
- Could free GPU for component experiments

---

## Key Findings Update

1. **24L r=384 full block remains best** (3.1014 @ 45.6B)
2. **36L r=384 hybrid improving** (3.1097 @ 28.7B, may reach ~3.08-3.09)
3. **r=512 is definitively worse** than r=384 (3.1554 vs 3.1014/3.1097)
4. **Component experiments hit issues:**
   - QKV-only: OOM (fixable with batch size reduction)
   - MLP-only: Unexpectedly slow (needs investigation)
   - Baseline: Progressing normally

---

## Next Actions

**Option A: Cancel slow jobs, fix component experiments**
```bash
scancel 611029  # Cancel plateaued 24L r=512
scancel 613767  # Cancel slow MLP-only
# Restart QKV-only with reduced batch size
# Restart MLP-only with investigation/fixes
```

**Option B: Let everything finish**
- 36L continues (interesting to see final result)
- 24L r=512 continues (waste of compute but confirms plateau)
- Component baseline finishes (useful control group)
- Fix component experiments separately

**My recommendation: Option A**
- 24L r=512 is clearly plateaued, won't improve
- MLP-only has issues, better to restart with fixes
- Free GPUs for fixed component experiments
- Keep 36L running (still showing improvement)
