# TIER 1 EXPERIMENTS - Disentangling Depth, Rank, and Block Decomposition

**Started:** January 2, 2026
**Status:** ✅ All 3 experiments running with automatic continuation chains

---

## Experimental Design

### Critical Confounds to Resolve

Previous experiments confounded multiple variables:
- 36L models used r=256 (25% of width)
- 24L models used r=384 (37.5% of width)
- Full block decomposition only tested on 36L
- Can't tell if 36L is slow due to **depth** or **low rank**

### Total Nullspace Hypothesis

**Key insight:** Depth × (width - rank) = total nullspace capacity

| Model | Per-layer nullspace | Total nullspace | Comparison |
|-------|---------------------|-----------------|------------|
| 24L, r=384 | 1024 - 384 = 640 | 24 × 640 = **15,360** | Baseline |
| 36L, r=256 | 1024 - 256 = 768 | 36 × 768 = **27,648** | **+78% MORE!** |
| 36L, r=384 | 1024 - 384 = 640 | 36 × 640 = **23,040** | **+50% MORE** |

**Hypothesis:** 36L r=256 might have MORE effective capacity than 24L r=384!

---

## Tier 1 Experiments

### Experiment 1: 36L, r=384, Hybrid Gating

**Job:** 610883 → 610889...610944 (56 sessions total)
**Config:** `config/train_gpt2_large_grassmann_hybridgate_r384.py`
**Output:** `/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r384`

**Question:** Is 36L slow because of depth or low rank?

**Hypothesis:**
- If faster convergence than 36L r=256 → **rank was bottleneck**
- If still slow → **depth needs more training**

**Parameters:**
- Depth: 36 layers
- Rank: **r=384** (↑ from r=256, +50% increase)
- Decomposition: Hybrid (3 blocks for c_attn, 4 for mlp.c_fc)
- Gating: 52 gates/layer (48 head + 4 block)
- Training: 168k iters (100 tokens/param, ~66B tokens)
- Est. time: 27.5 days (~660 GPU-hours)

**Success criteria:**
- Converges faster than 36L r=256 (which reached val 3.26 @ 14.2B tokens)
- Reaches val < 3.2 by 15B tokens
- Final val loss < 3.0 (breaks ceiling from r=256)

---

### Experiment 2: 24L, r=512, Hybrid Gating

**Job:** 610884 → 610945...610954 (11 sessions total)
**Config:** `config/train_gpt2_medium_grassmann_hybridgate_r512.py`
**Output:** `/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r512`

**Question:** Can 24L break the 3.1 ceiling with higher rank?

**Hypothesis:**
- If breaks ceiling → **rank was limiting expressivity**
- If ceiling persists → **bottleneck is elsewhere** (gating, training, or fundamental)

**Parameters:**
- Depth: 24 layers
- Rank: **r=512** (↑ from r=384, +33% increase, 50% of width!)
- Decomposition: Hybrid (3 blocks for c_attn, 4 for mlp.c_fc)
- Gating: 52 gates/layer
- Training: 100k iters (~157B tokens)
- Est. time: 5 days (~120 GPU-hours)

**Success criteria:**
- Breaks 3.1 ceiling observed in 24L r=384 (which plateaued at val 3.1168 @ 40.9B tokens)
- Reaches val < 3.0 by 50B tokens
- Final val loss < 2.9 (approaching baseline's 2.86)

**Early evaluation points:**
- 3B tokens: Should match or beat r=384 (4.6 vs 4.6)
- 7B tokens: Should start diverging if rank helps (3.4 vs 3.5?)
- 15B tokens: Clear signal if ceiling is breaking (3.0 vs 3.1+)

---

### Experiment 3: 24L, r=384, Full Block Decomposition

**Job:** 610885 → 610955...610964 (11 sessions total)
**Config:** `config/train_gpt2_medium_grassmann_fullblock_r384.py`
**Output:** `/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-fullblock-r384`

**Question:** Does fine-grained decomposition help 24L like it helps 36L?

**Hypothesis:**
- If outperforms hybrid → **block granularity matters**
- If similar → **gating strategy doesn't matter much**

**Parameters:**
- Depth: 24 layers
- Rank: r=384 (same as hybrid baseline)
- Decomposition: **Full block** (16×16 grid = 768 blocks for c_attn)
  - Block size: 64×64, rank=32 per block
  - Total c_attn blocks: 768 per layer (vs 3 for hybrid)
- Gating: 52 gates/layer (16 per-head gates × 3 QKV + 4 block gates for MLP)
- Training: 100k iters (~157B tokens)
- Batch: 16 per GPU (reduced from 32 due to memory)
- Est. time: 5 days (~120 GPU-hours)

**Success criteria:**
- Outperforms 24L hybrid r=384 by >2% at same tokens
- Shows better convergence rate than hybrid
- Final val loss < 3.05 (vs 3.1168 for hybrid)

**Comparison with 36L full block:**
- 36L full r=256: val 3.1724 @ 20.4B tokens (better than 36L hybrid 3.26)
- This 24L full r=384: Should show if fine-grained helps 24L too

---

## Expected Outcomes & Decision Tree

### Scenario A: Rank is the Key Bottleneck

**Evidence:**
- 36L r=384 converges much faster than 36L r=256 ✓
- 24L r=512 breaks 3.1 ceiling ✓
- Full block similar to hybrid (granularity doesn't matter)

**Conclusion:** Higher rank = better performance
**Next steps:** Test 36L r=512, try r=768, investigate rank scheduling

---

### Scenario B: Depth Helps, Rank Limits Ceiling

**Evidence:**
- 36L r=384 still slower than 24L r=384 early on
- 36L r=384 eventually reaches lower loss than 24L r=384
- 24L r=512 still hits ceiling (just higher, ~2.95 instead of 3.1)

**Conclusion:** Depth provides capacity, but rank limits final performance
**Next steps:** Test hybrid strategy (Grassmann→Baseline switch), try 48L models

---

### Scenario C: Block Granularity Matters Most

**Evidence:**
- 24L full r=384 significantly outperforms 24L hybrid r=384
- 36L r=384 similar to 36L r=256 (rank doesn't help much)
- 24L r=512 similar to 24L r=384 (rank doesn't help much)

**Conclusion:** Fine-grained decomposition is key, not just rank
**Next steps:** Test 36L full r=384, optimize block decomposition strategy

---

### Scenario D: Fundamental Limitation (Negative Result)

**Evidence:**
- All three Tier 1 experiments show marginal improvement (<2%)
- Ceiling persists across all variants
- Baseline still outperforms all Grassmann variants

**Conclusion:** Grassmann constraint fundamentally incompatible with LM
**Next steps:** Document negative result, pivot to hybrid training strategy or alternative approaches

---

## Monitoring Schedule

**Daily (Days 1-7):**
- Check 24L r=512 and 24L full block (fast iterations)
- Evaluate at 3B, 6B, 9B, 12B, 15B tokens
- Early signal on whether rank/blocks help

**Weekly (Days 7-28):**
- Check 36L r=384 progress
- Compare convergence rate to 36L r=256
- Evaluate at 15B, 30B, 45B, 60B tokens

**Critical evaluation points:**

| Tokens | 24L hybrid r=384 | 24L r=512 | 24L full | 36L r=384 | Comparison |
|--------|------------------|-----------|----------|-----------|------------|
| 3B | 3.60 | ? | ? | ? | Early advantage test |
| 7B | 3.33 | ? | ? | ? | Crossover point |
| 15B | 3.15 | ? | ? | ? | Clear signal |
| 40B | 3.12 | ? | ? | ? | Final performance |

---

## Compute Budget

**Total Tier 1 compute:**
- 36L r=384: 56 sessions × 12h = 672 GPU-hours
- 24L r=512: 11 sessions × 12h = 132 GPU-hours
- 24L full: 11 sessions × 12h = 132 GPU-hours
- **Total: 936 GPU-hours (~39 GPU-days)**

**Cost-benefit:**
- Resolves 3 major confounds with single set of experiments
- Informs whether to continue Grassmann research or pivot
- Even negative results are valuable (save community from repeating)

---

## Files Created

**Config files:**
- `/home/junyuren/nanoGPT/config/train_gpt2_large_grassmann_hybridgate_r384.py`
- `/home/junyuren/nanoGPT/config/train_gpt2_medium_grassmann_hybridgate_r512.py`
- `/home/junyuren/nanoGPT/config/train_gpt2_medium_grassmann_fullblock_r384.py`

**Submit scripts:**
- `/home/junyuren/nanoGPT/submit_tier1_36L_r384_hybrid.sh`
- `/home/junyuren/nanoGPT/submit_tier1_24L_r512_hybrid.sh`
- `/home/junyuren/nanoGPT/submit_tier1_24L_r384_fullblock.sh`

**Continuation scripts:**
- `/home/junyuren/nanoGPT/submit_tier1_36L_r384_continue.sh`
- `/home/junyuren/nanoGPT/submit_tier1_24L_r512_continue.sh`
- `/home/junyuren/nanoGPT/submit_tier1_24L_fullblock_continue.sh`

**Automation:**
- `/home/junyuren/nanoGPT/setup_tier1_continuations.sh`

---

## Current Status

**All 3 experiments RUNNING:**
- Job 610883 (36L r=384): Started 2026-01-02 11:12, running on m001
- Job 610884 (24L r=512): Started 2026-01-02 11:12, running on m002
- Job 610885 (24L full): Started 2026-01-02 11:12, running on n001

**Continuation chains:** 75 jobs queued with dependencies

**Monitoring:**
```bash
# Check status
squeue -u $USER | grep tier1

# Monitor logs
tail -f logs/610883_tier1_36L_r384.out
tail -f logs/610884_tier1_24L_r512.out
tail -f logs/610885_tier1_24L_fb384.out

# Check latest iteration
grep "step [0-9]*:" logs/610883_tier1_36L_r384.out | tail -1
grep "step [0-9]*:" logs/610884_tier1_24L_r512.out | tail -1
grep "step [0-9]*:" logs/610885_tier1_24L_fb384.out | tail -1
```

---

**Last Updated:** 2026-01-02 11:15 UTC
**Next Review:** 2026-01-03 (check 24L experiments @ ~3B tokens)
