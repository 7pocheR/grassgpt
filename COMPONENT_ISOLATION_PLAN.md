# Component Isolation Experiments (CORRECTED)

## Objective
Test whether Grassmann helps in c_attn (QKV) or c_fc (MLP) or both, using the SAME hybrid architecture.

## Experimental Design

### Configurations (All use IDENTICAL architecture for fair comparison):

1. **Baseline** (Pure AdamW)
   - c_attn: AdamW
   - c_fc: AdamW
   - Config: `train_gpt2_medium_baseline_component_test.py`

2. **QKV-only** (c_attn Grassmann, c_fc AdamW)
   - c_attn: 3-block decomposition (Q/K/V), r=384, 48 head-level gates
   - c_fc: AdamW (NO Grassmann)
   - Config: `train_gpt2_medium_grassmann_qkv_only_r384.py` ✅ FIXED

3. **MLP-only** (c_attn AdamW, c_fc Grassmann)
   - c_attn: AdamW (NO Grassmann)
   - c_fc: 4-block decomposition, r=384, 4 block-level gates
   - Config: `train_gpt2_medium_grassmann_mlp_only_r384.py`

4. **24L Hybrid** (Both Grassmann) - REFERENCE
   - c_attn: 3-block decomposition (Q/K/V), r=384, 48 head-level gates
   - c_fc: 4-block decomposition, r=384, 4 block-level gates
   - Config: `train_gpt2_medium_grassmann_hybridgate_r384.py`
   - Already running: job 608059

## Key Changes from Previous Run

**BEFORE (WRONG):**
- QKV-only used 16×16 full block decomposition (256 blocks, r=32/block)
- Very slow (13× slower than baseline)
- Unfair comparison (different architecture)

**NOW (CORRECT):**
- QKV-only uses SAME hybrid architecture as 24L (3-block, r=384, 48 gates)
- Same speed as 24L hybrid
- Fair comparison - only difference is whether c_fc uses Grassmann

## Expected Outcomes

1. **If QKV-only > 24L Hybrid**: c_fc Grassmann HURTS (remove it!)
2. **If MLP-only > 24L Hybrid**: c_attn Grassmann HURTS (remove it!)
3. **If 24L Hybrid > both**: Synergy between c_attn + c_fc Grassmann
4. **If Baseline > all**: Grassmann doesn't help anywhere

## Hypothesis from User

User suggests 24L hybrid shows sample efficiency vs baseline at 5-7B tokens.
- If true, we want to see if QKV-only or MLP-only can improve FURTHER
- Goal: Find which component contributes to sample efficiency

## Training Budget

- max_iters: 13,000 (~20B tokens)
- This covers the critical 5-10B token range where sample efficiency matters
- Can extend if needed

## Next Steps

1. ✅ Fix QKV-only config (use hybrid gating, not full block)
2. Submit all 3 component experiments in parallel
3. Compare at 5-7B tokens (sample efficiency region)
4. Compare at 20B tokens (final performance)
