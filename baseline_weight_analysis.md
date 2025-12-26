# Baseline Weight Matrix Analysis

**Checkpoint:** /net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline/ckpt.pt
**Iteration:** 2000
**Best val loss:** 1.4785162210464478

## Weight Matrix Statistics

| Layer | Shape | Op Norm | Frob Norm | Eff Rank | Rank@99% | Condition |
|-------|-------|---------|-----------|----------|----------|-----------|
| _orig_mod.lm_head.weight | 65×384 | 4.2226 | 10.59 | 57.5 | 60 | 10.1 |
| _orig_mod.transformer.h.0.attn.c_attn.weight | 1152×384 | 5.3932 | 27.23 | 277.0 | 321 | 20.7 |
| _orig_mod.transformer.h.0.attn.c_proj.weight | 384×384 | 0.8322 | 5.89 | 255.5 | 257 | 5821.2 |
| _orig_mod.transformer.h.0.mlp.c_fc.weight | 1536×384 | 9.3816 | 29.72 | 297.4 | 340 | 24.4 |
| _orig_mod.transformer.h.0.mlp.c_proj.weight | 384×1536 | 3.0045 | 22.09 | 329.8 | 349 | 9.6 |
| _orig_mod.transformer.h.1.attn.c_attn.weight | 1152×384 | 3.4025 | 17.08 | 336.0 | 356 | 12.7 |
| _orig_mod.transformer.h.1.attn.c_proj.weight | 384×384 | 2.2186 | 8.34 | 238.0 | 239 | 3575.9 |
| _orig_mod.transformer.h.1.mlp.c_fc.weight | 1536×384 | 11.2323 | 31.02 | 306.4 | 345 | 25.9 |
| _orig_mod.transformer.h.1.mlp.c_proj.weight | 384×1536 | 3.8905 | 23.01 | 325.8 | 348 | 12.7 |
| _orig_mod.transformer.h.2.attn.c_attn.weight | 1152×384 | 5.8526 | 22.44 | 315.1 | 344 | 20.2 |
| _orig_mod.transformer.h.2.attn.c_proj.weight | 384×384 | 1.8055 | 10.68 | 243.7 | 238 | 7719.6 |
| _orig_mod.transformer.h.2.mlp.c_fc.weight | 1536×384 | 11.2857 | 33.18 | 305.3 | 342 | 25.5 |
| _orig_mod.transformer.h.2.mlp.c_proj.weight | 384×1536 | 4.1952 | 27.29 | 327.1 | 348 | 11.2 |
| _orig_mod.transformer.h.3.attn.c_attn.weight | 1152×384 | 5.3705 | 24.35 | 316.5 | 343 | 18.0 |
| _orig_mod.transformer.h.3.attn.c_proj.weight | 384×384 | 1.8096 | 11.84 | 247.1 | 239 | 2784.3 |
| _orig_mod.transformer.h.3.mlp.c_fc.weight | 1536×384 | 11.2352 | 34.82 | 307.2 | 341 | 24.9 |
| _orig_mod.transformer.h.3.mlp.c_proj.weight | 384×1536 | 4.7017 | 30.30 | 316.0 | 344 | 12.2 |
| _orig_mod.transformer.h.4.attn.c_attn.weight | 1152×384 | 4.7699 | 24.84 | 322.0 | 344 | 14.8 |
| _orig_mod.transformer.h.4.attn.c_proj.weight | 384×384 | 1.9642 | 12.35 | 242.9 | 234 | 2899.1 |
| _orig_mod.transformer.h.4.mlp.c_fc.weight | 1536×384 | 10.8116 | 35.34 | 314.3 | 344 | 23.0 |
| _orig_mod.transformer.h.4.mlp.c_proj.weight | 384×1536 | 6.0701 | 31.63 | 286.9 | 330 | 17.2 |
| _orig_mod.transformer.h.5.attn.c_attn.weight | 1152×384 | 3.8579 | 23.77 | 324.0 | 346 | 12.2 |
| _orig_mod.transformer.h.5.attn.c_proj.weight | 384×384 | 2.3620 | 11.68 | 221.5 | 216 | 6067.0 |
| _orig_mod.transformer.h.5.mlp.c_fc.weight | 1536×384 | 10.0655 | 35.07 | 318.3 | 346 | 20.6 |
| _orig_mod.transformer.h.5.mlp.c_proj.weight | 384×1536 | 7.2719 | 31.68 | 232.2 | 296 | 28.3 |
| _orig_mod.transformer.wpe.weight | 256×384 | 4.9391 | 14.70 | 150.8 | 171 | 80.2 |
| _orig_mod.transformer.wte.weight | 65×384 | 4.2226 | 10.59 | 57.5 | 60 | 10.1 |

## Summary by Layer Type

### c_attn

- **Count:** 6
- **Operator norm:** mean=4.7744, std=0.8778, range=[3.4025, 5.8526]
- **Frobenius norm:** mean=23.29, std=3.12
- **Effective rank:** mean=315.1, std=18.3

### attn.c_proj

- **Count:** 6
- **Operator norm:** mean=1.8320, std=0.4914, range=[0.8322, 2.3620]
- **Frobenius norm:** mean=10.13, std=2.30
- **Effective rank:** mean=241.4, std=10.4

### mlp.c_fc

- **Count:** 6
- **Operator norm:** mean=10.6687, std=0.7142, range=[9.3816, 11.2857]
- **Frobenius norm:** mean=33.19, std=2.15
- **Effective rank:** mean=308.2, std=6.7

### mlp.c_proj

- **Count:** 6
- **Operator norm:** mean=4.8556, std=1.4219, range=[3.0045, 7.2719]
- **Frobenius norm:** mean=27.67, std=3.91
- **Effective rank:** mean=303.0, std=34.8

### other

- **Count:** 3
- **Operator norm:** mean=4.4614, std=0.3378, range=[4.2226, 4.9391]
- **Frobenius norm:** mean=11.96, std=1.94
- **Effective rank:** mean=88.6, std=44.0

## Recommendations for Grassmann Scaling

Since Grassmann manifold fixes operator norm = 1, we need to scale outputs to match baseline:

- **c_attn**: Scale by 4.7744 (mean operator norm)
- **mlp.c_fc**: Scale by 10.6687 (mean operator norm)
- **attn.c_proj**: Scale by 1.8320 (mean operator norm)

### Implementation

```python
# In model.py, after Grassmann update:
# c_attn: multiply output by 4.7744
block.c_attn_scale = 4.7744
# mlp.c_fc: multiply output by 10.6687
block.mlp_c_fc_scale = 10.6687
```

## Singular Value Analysis

Top 10 singular values for key layers:

**_orig_mod.transformer.h.0.attn.c_attn.weight:** [5.393, 5.323, 5.149, 5.111, 4.986, 4.912, 4.711, 4.703, 4.352, 4.311, ...]

**_orig_mod.transformer.h.0.mlp.c_fc.weight:** [9.382, 4.171, 4.088, 3.897, 3.746, 3.683, 3.554, 3.523, 3.461, 3.457, ...]

**_orig_mod.transformer.h.0.attn.c_proj.weight:** [0.832, 0.804, 0.785, 0.777, 0.773, 0.768, 0.760, 0.758, 0.752, 0.747, ...]
