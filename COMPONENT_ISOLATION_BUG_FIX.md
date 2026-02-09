# Component Isolation Bug Fix

## Problem

The component isolation experiments (QKV-only and MLP-only) were BOTH running with full Grassmann (c_attn + c_fc), despite the config files correctly setting:

**QKV-only:**
```python
use_grassmann_c_attn = True   # Enable for c_attn
use_grassmann_c_fc = False    # DISABLE for c_fc
```

**MLP-only:**
```python
use_grassmann_c_attn = False  # DISABLE for c_attn
use_grassmann_c_fc = True     # Enable for c_fc
```

## Root Cause

**train.py line 163-167**: The `model_args` dictionary was **missing** the component isolation flags!

```python
# BEFORE (BUGGY):
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  use_grassmann=use_grassmann, grass_rank=grass_rank, grass_scale=grass_scale,
                  grass_a=grass_a, grass_b=grass_b, grass_lr=grass_lr,
                  grassmann_dropout=grassmann_dropout, gate_lr=gate_lr, embed_lr=embed_lr)
                  # ❌ use_grassmann_c_attn and use_grassmann_c_fc NOT INCLUDED!
```

This meant that even though the config files set these variables, they were **never passed to GPTConfig**, so the flags didn't exist on the config object.

Then in `model.py`:
- CausalSelfAttention: `self.use_grassmann_c_attn = getattr(config, 'use_grassmann_c_attn', self.use_grassmann)`
  - Since `config.use_grassmann_c_attn` didn't exist, it defaulted to `self.use_grassmann = True`
  - **Result: ALWAYS used Grassmann for c_attn when use_grassmann=True**

- MLP: `self.use_grassmann_c_fc = getattr(config, 'use_grassmann_c_fc', self.use_grassmann)`
  - Since `config.use_grassmann_c_fc` didn't exist, it defaulted to `self.use_grassmann = True`
  - **Result: ALWAYS used Grassmann for c_fc when use_grassmann=True**

## Evidence

From log 615947 (QKV-only):
```
Initializing attn.c_attn.blocks.0-2   ← Expected
Initializing mlp.c_fc.blocks.0-3      ← BUG! Should be AdamW
```

From log 615948 (MLP-only):
```
Initializing attn.c_attn.blocks.0-2   ← BUG! Should be AdamW  
Initializing mlp.c_fc.blocks.0-3      ← Expected
```

**Both experiments were identical to 24L Hybrid!**

## Fix

### 1. Added component isolation flags to GPTConfig dataclass (model.py:268-269)

```python
# Component isolation (selective Grassmann application)
use_grassmann_c_attn: bool = None  # If None, defaults to use_grassmann
use_grassmann_c_fc: bool = None    # If None, defaults to use_grassmann
```

### 2. Added flags to model_args in train.py (line 168-170)

```python
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  use_grassmann=use_grassmann, grass_rank=grass_rank, grass_scale=grass_scale,
                  grass_a=grass_a, grass_b=grass_b, grass_lr=grass_lr,
                  grassmann_dropout=grassmann_dropout, gate_lr=gate_lr, embed_lr=embed_lr,
                  # Component isolation flags
                  use_grassmann_c_attn=globals().get('use_grassmann_c_attn', None),
                  use_grassmann_c_fc=globals().get('use_grassmann_c_fc', None),
                  # Full block decomposition flags (BONUS FIX)
                  use_full_block_decomp=globals().get('use_full_block_decomp', False),
                  full_block_size=globals().get('full_block_size', 64),
                  full_block_gating_mode=globals().get('full_block_gating_mode', 'per_head'))
```

### 3. Fixed flag handling to properly handle None (model.py:41-42, 194-195)

**CausalSelfAttention:**
```python
# BEFORE:
self.use_grassmann_c_attn = getattr(config, 'use_grassmann_c_attn', self.use_grassmann)

# AFTER:
c_attn_flag = getattr(config, 'use_grassmann_c_attn', None)
self.use_grassmann_c_attn = c_attn_flag if c_attn_flag is not None else self.use_grassmann
```

**MLP:**
```python
# BEFORE:
self.use_grassmann_c_fc = getattr(config, 'use_grassmann_c_fc', self.use_grassmann)

# AFTER:
c_fc_flag = getattr(config, 'use_grassmann_c_fc', None)
self.use_grassmann_c_fc = c_fc_flag if c_fc_flag is not None else self.use_grassmann
```

**Why this matters:** If `config.use_grassmann_c_attn = None`, the old code would return `None` instead of falling back to `self.use_grassmann`. The new code explicitly checks for `None` and uses `self.use_grassmann` as the default.

## Expected Behavior After Fix

### QKV-only:
- use_grassmann_c_attn = True → c_attn uses Grassmann ✅
- use_grassmann_c_fc = False → c_fc uses AdamW ✅

### MLP-only:
- use_grassmann_c_attn = False → c_attn uses AdamW ✅
- use_grassmann_c_fc = True → c_fc uses Grassmann ✅

### 24L Hybrid (both None):
- use_grassmann_c_attn = None → defaults to use_grassmann = True ✅
- use_grassmann_c_fc = None → defaults to use_grassmann = True ✅

### Baseline:
- use_grassmann = False → no Grassmann anywhere ✅

## Bonus Fixes

Also added missing full block decomposition flags to model_args:
- `use_full_block_decomp`
- `full_block_size`
- `full_block_gating_mode`

These were in GPTConfig but not being passed from config files to the model!

## Files Modified

1. **model.py**:
   - Added `use_grassmann_c_attn` and `use_grassmann_c_fc` to GPTConfig dataclass
   - Fixed flag handling in CausalSelfAttention and MLP to properly handle None

2. **train.py**:
   - Added component isolation flags to model_args dictionary
   - Added full block decomposition flags to model_args dictionary

## Next Steps

✅ Fix complete - ready to resubmit component isolation experiments
- QKV-only will NOW actually test c_attn Grassmann only
- MLP-only will NOW actually test c_fc Grassmann only
- Valid comparison possible!

---
**Date:** 2026-01-07
**Status:** Fixed and ready to test
