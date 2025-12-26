# Baseline Weight Matrix Analysis

**Checkpoint:** /net/scratch2/junyuren/nanoGPT-manifold/out-openwebtext-baseline/ckpt.pt
**Iteration:** 13000
**Best val loss:** 3.7837696075439453

## Weight Matrix Statistics

| Layer | Shape | Op Norm | Frob Norm | Eff Rank | Rank@99% | Condition |
|-------|-------|---------|-----------|----------|----------|-----------|
| _orig_mod.lm_head.weight | 50304×384 | 80.1334 | 359.35 | 343.7 | 353 | 50.5 |
| _orig_mod.transformer.h.0.attn.c_attn.weight | 1152×384 | 3.9236 | 32.49 | 332.5 | 333 | 12.4 |
| _orig_mod.transformer.h.0.attn.c_proj.weight | 384×384 | 3.4274 | 10.17 | 229.2 | 233 | 7221.2 |
| _orig_mod.transformer.h.0.mlp.c_fc.weight | 1536×384 | 10.0682 | 33.59 | 334.8 | 338 | 52.7 |
| _orig_mod.transformer.h.0.mlp.c_proj.weight | 384×1536 | 4.0774 | 17.26 | 332.8 | 340 | 39.0 |
| _orig_mod.transformer.h.1.attn.c_attn.weight | 1152×384 | 3.7922 | 27.46 | 316.2 | 324 | 18.0 |
| _orig_mod.transformer.h.1.attn.c_proj.weight | 384×384 | 3.5826 | 12.66 | 242.6 | 232 | 16855.1 |
| _orig_mod.transformer.h.1.mlp.c_fc.weight | 1536×384 | 8.1418 | 29.27 | 342.5 | 347 | 81.4 |
| _orig_mod.transformer.h.1.mlp.c_proj.weight | 384×1536 | 4.3697 | 22.02 | 341.8 | 349 | 114.6 |
| _orig_mod.transformer.h.2.attn.c_attn.weight | 1152×384 | 5.7594 | 33.05 | 323.5 | 331 | 32.6 |
| _orig_mod.transformer.h.2.attn.c_proj.weight | 384×384 | 1.8172 | 11.10 | 263.2 | 250 | 4306.5 |
| _orig_mod.transformer.h.2.mlp.c_fc.weight | 1536×384 | 7.7484 | 27.80 | 333.8 | 343 | 89.9 |
| _orig_mod.transformer.h.2.mlp.c_proj.weight | 384×1536 | 4.1491 | 20.45 | 330.4 | 334 | 156.5 |
| _orig_mod.transformer.h.3.attn.c_attn.weight | 1152×384 | 7.9522 | 41.66 | 301.6 | 315 | 69.2 |
| _orig_mod.transformer.h.3.attn.c_proj.weight | 384×384 | 1.6164 | 10.79 | 258.9 | 238 | 13180.5 |
| _orig_mod.transformer.h.3.mlp.c_fc.weight | 1536×384 | 6.4766 | 27.15 | 330.4 | 338 | 49.4 |
| _orig_mod.transformer.h.3.mlp.c_proj.weight | 384×1536 | 3.6272 | 19.94 | 326.5 | 336 | 80.0 |
| _orig_mod.transformer.h.4.attn.c_attn.weight | 1152×384 | 5.4994 | 31.66 | 333.8 | 340 | 34.7 |
| _orig_mod.transformer.h.4.attn.c_proj.weight | 384×384 | 3.6223 | 15.17 | 291.0 | 274 | 41747.5 |
| _orig_mod.transformer.h.4.mlp.c_fc.weight | 1536×384 | 6.5419 | 27.90 | 336.1 | 347 | 27.3 |
| _orig_mod.transformer.h.4.mlp.c_proj.weight | 384×1536 | 5.9208 | 24.82 | 329.9 | 345 | 65.8 |
| _orig_mod.transformer.h.5.attn.c_attn.weight | 1152×384 | 3.3205 | 30.21 | 353.4 | 355 | 18.3 |
| _orig_mod.transformer.h.5.attn.c_proj.weight | 384×384 | 5.9736 | 20.99 | 284.9 | 268 | 5267.0 |
| _orig_mod.transformer.h.5.mlp.c_fc.weight | 1536×384 | 6.2774 | 28.86 | 340.6 | 350 | 76.4 |
| _orig_mod.transformer.h.5.mlp.c_proj.weight | 384×1536 | 8.9783 | 27.73 | 328.8 | 342 | 83.4 |
| _orig_mod.transformer.wpe.weight | 1024×384 | 29.7138 | 60.50 | 49.4 | 17 | 504.5 |
| _orig_mod.transformer.wte.weight | 50304×384 | 80.1334 | 359.35 | 343.7 | 353 | 50.5 |

## Summary by Layer Type

### c_attn

- **Count:** 6
- **Operator norm:** mean=5.0412, std=1.5795, range=[3.3205, 7.9522]
- **Frobenius norm:** mean=32.75, std=4.38
- **Effective rank:** mean=326.8, std=16.0

### attn.c_proj

- **Count:** 6
- **Operator norm:** mean=3.3399, std=1.4359, range=[1.6164, 5.9736]
- **Frobenius norm:** mean=13.48, std=3.73
- **Effective rank:** mean=261.6, std=21.7

### mlp.c_fc

- **Count:** 6
- **Operator norm:** mean=7.5424, std=1.3240, range=[6.2774, 10.0682]
- **Frobenius norm:** mean=29.09, std=2.13
- **Effective rank:** mean=336.4, std=4.1

### mlp.c_proj

- **Count:** 6
- **Operator norm:** mean=5.1871, std=1.8404, range=[3.6272, 8.9783]
- **Frobenius norm:** mean=22.04, std=3.41
- **Effective rank:** mean=331.7, std=4.9

### other

- **Count:** 3
- **Operator norm:** mean=63.3269, std=23.7680, range=[29.7138, 80.1334]
- **Frobenius norm:** mean=259.73, std=140.88
- **Effective rank:** mean=245.6, std=138.7

## Recommendations for Grassmann Scaling

Since Grassmann manifold fixes operator norm = 1, we need to scale outputs to match baseline:

- **c_attn**: Scale by 5.0412 (mean operator norm)
- **mlp.c_fc**: Scale by 7.5424 (mean operator norm)
- **attn.c_proj**: Scale by 3.3399 (mean operator norm)

### Implementation

```python
# In model.py, after Grassmann update:
# c_attn: multiply output by 5.0412
block.c_attn_scale = 5.0412
# mlp.c_fc: multiply output by 7.5424
block.mlp_c_fc_scale = 7.5424
```

## Singular Value Analysis

Top 10 singular values for key layers:

**_orig_mod.transformer.h.0.attn.c_attn.weight:** [3.924, 3.663, 3.604, 3.525, 3.282, 3.171, 3.128, 3.029, 3.023, 2.978, ...]

**_orig_mod.transformer.h.0.mlp.c_fc.weight:** [10.068, 5.145, 4.519, 4.287, 3.995, 3.760, 3.713, 3.524, 3.272, 3.189, ...]

**_orig_mod.transformer.h.0.attn.c_proj.weight:** [3.427, 2.232, 2.098, 1.916, 1.829, 1.810, 1.772, 1.655, 1.639, 1.546, ...]
