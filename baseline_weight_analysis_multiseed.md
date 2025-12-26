# Baseline Weight Matrix Analysis (Multi-Seed)

**Number of seeds:** 10
**Checkpoint pattern:** `/net/scratch2/junyuren/nanoGPT-manifold/out-baby-baseline-seed*/ckpt.pt`

## Checkpoint Details

| Seed | Iteration | Best Val Loss |
|------|-----------|---------------|
| out-baby-baseline-seed42 | 1750 | 1.4694 |
| out-baby-baseline-seed43 | 1750 | 1.4717 |
| out-baby-baseline-seed44 | 1500 | 1.4700 |
| out-baby-baseline-seed45 | 1750 | 1.4686 |
| out-baby-baseline-seed46 | 1750 | 1.4752 |
| out-baby-baseline-seed47 | 1750 | 1.4704 |
| out-baby-baseline-seed48 | 1500 | 1.4671 |
| out-baby-baseline-seed49 | 1750 | 1.4733 |
| out-baby-baseline-seed50 | 1750 | 1.4737 |
| out-baby-baseline-seed51 | 1750 | 1.4705 |

## Per-Layer Weight Matrix Statistics (Averaged Across Seeds)

| Layer | Shape | Op Norm | Frob Norm | Nuclear Norm | L1 Norm | L∞ Norm | Stable Rank | Eff Rank | Rank@99% | Condition |
|-------|-------|---------|-----------|--------------|---------|---------|-------------|----------|----------|-----------|
| _orig_mod.lm_head.weight | 65×384 | 3.9801±0.0926 | 9.95±0.23 | 70.5±1.67 | 4.46±0.103 | 26.46±0.817 | 6.2±0.06 | 57.4 | 60.3 | 10.1 |
| _orig_mod.transformer.h.0.attn.c_attn.weight | 1152×384 | 4.8777±0.1868 | 25.69±0.60 | 370.2±4.63 | 40.04±1.072 | 21.05±0.591 | 27.8±0.90 | 284.9 | 326.8 | 18.4 |
| _orig_mod.transformer.h.0.attn.c_proj.weight | 384×384 | 0.7733±0.0203 | 5.48±0.15 | 77.9±2.14 | 6.23±0.262 | 6.60±0.483 | 50.2±1.09 | 257.2 | 259.0 | 14190.5 |
| _orig_mod.transformer.h.0.mlp.c_fc.weight | 1536×384 | 8.7494±0.2539 | 28.41±0.52 | 433.4±6.17 | 52.95±1.198 | 22.26±0.536 | 10.6±0.24 | 300.6 | 342.4 | 23.4 |
| _orig_mod.transformer.h.0.mlp.c_proj.weight | 384×1536 | 2.7757±0.0590 | 20.48±0.60 | 344.3±11.31 | 15.73±0.555 | 48.89±2.782 | 54.4±1.13 | 326.6 | 347.3 | 10.0 |
| _orig_mod.transformer.h.1.attn.c_attn.weight | 1152×384 | 3.3336±0.0998 | 16.37±0.55 | 282.4±6.96 | 25.70±0.773 | 12.63±1.788 | 24.1±0.83 | 340.6 | 358.0 | 11.9 |
| _orig_mod.transformer.h.1.attn.c_proj.weight | 384×384 | 2.1078±0.0390 | 7.69±0.22 | 100.1±3.23 | 8.48±0.391 | 8.93±0.420 | 13.3±0.47 | 236.9 | 239.3 | 29849.7 |
| _orig_mod.transformer.h.1.mlp.c_fc.weight | 1536×384 | 10.5926±0.2309 | 29.45±0.56 | 453.5±7.09 | 55.68±0.951 | 18.45±0.410 | 7.7±0.06 | 309.6 | 347.5 | 24.9 |
| _orig_mod.transformer.h.1.mlp.c_proj.weight | 384×1536 | 3.5434±0.1214 | 20.79±0.78 | 344.8±13.73 | 13.17±0.524 | 39.02±1.520 | 34.4±0.67 | 323.6 | 347.1 | 12.3 |
| _orig_mod.transformer.h.2.attn.c_attn.weight | 1152×384 | 5.8826±0.0427 | 21.69±0.38 | 348.3±4.94 | 44.64±1.094 | 15.01±0.513 | 13.6±0.47 | 317.6 | 346.2 | 20.1 |
| _orig_mod.transformer.h.2.attn.c_proj.weight | 384×384 | 1.7225±0.0425 | 9.95±0.30 | 135.7±4.54 | 10.32±0.418 | 9.62±0.469 | 33.4±0.91 | 241.7 | 236.8 | 22506.5 |
| _orig_mod.transformer.h.2.mlp.c_fc.weight | 1536×384 | 10.7750±0.2096 | 31.82±0.55 | 490.0±7.38 | 57.46±0.855 | 16.36±0.380 | 8.7±0.06 | 306.6 | 344.4 | 24.9 |
| _orig_mod.transformer.h.2.mlp.c_proj.weight | 384×1536 | 3.8672±0.1215 | 25.01±0.90 | 415.6±16.52 | 13.92±0.805 | 45.57±1.823 | 41.8±0.86 | 323.9 | 346.9 | 11.4 |
| _orig_mod.transformer.h.3.attn.c_attn.weight | 1152×384 | 5.2249±0.0462 | 23.29±0.38 | 378.3±5.53 | 44.47±1.306 | 14.30±0.383 | 19.9±0.45 | 318.9 | 345.0 | 16.9 |
| _orig_mod.transformer.h.3.attn.c_proj.weight | 384×384 | 1.8070±0.0321 | 11.06±0.31 | 152.3±4.84 | 10.73±0.457 | 10.60±0.339 | 37.4±1.01 | 242.9 | 235.5 | 31045.5 |
| _orig_mod.transformer.h.3.mlp.c_fc.weight | 1536×384 | 10.7579±0.1940 | 33.33±0.59 | 519.3±8.52 | 59.05±1.077 | 16.15±0.464 | 9.6±0.04 | 308.6 | 343.4 | 24.0 |
| _orig_mod.transformer.h.3.mlp.c_proj.weight | 384×1536 | 4.4099±0.1059 | 27.74±0.99 | 441.3±18.03 | 14.86±0.777 | 49.65±2.230 | 39.6±1.50 | 311.0 | 341.4 | 12.8 |
| _orig_mod.transformer.h.4.attn.c_attn.weight | 1152×384 | 4.7100±0.0535 | 23.48±0.46 | 388.8±7.44 | 43.10±0.882 | 13.57±0.386 | 24.9±0.68 | 323.4 | 346.4 | 14.8 |
| _orig_mod.transformer.h.4.attn.c_proj.weight | 384×384 | 1.8425±0.0530 | 11.22±0.41 | 153.3±6.12 | 10.85±0.451 | 11.53±0.606 | 37.1±1.06 | 239.9 | 232.3 | 12780.1 |
| _orig_mod.transformer.h.4.mlp.c_fc.weight | 1536×384 | 10.3549±0.1810 | 33.70±0.63 | 536.7±10.05 | 59.37±1.174 | 16.27±0.365 | 10.6±0.07 | 314.5 | 345.4 | 22.2 |
| _orig_mod.transformer.h.4.mlp.c_proj.weight | 384×1536 | 5.6531±0.1751 | 28.72±1.10 | 408.8±17.65 | 16.44±0.489 | 62.31±3.051 | 25.8±0.64 | 281.4 | 326.8 | 18.3 |
| _orig_mod.transformer.h.5.attn.c_attn.weight | 1152×384 | 3.7471±0.0559 | 22.12±0.48 | 368.7±7.61 | 36.66±0.370 | 13.27±0.203 | 34.8±1.10 | 325.9 | 348.9 | 12.0 |
| _orig_mod.transformer.h.5.attn.c_proj.weight | 384×384 | 2.2033±0.0667 | 10.36±0.39 | 130.5±5.48 | 11.01±0.446 | 15.88±0.931 | 22.1±1.03 | 220.5 | 218.9 | 33211.8 |
| _orig_mod.transformer.h.5.mlp.c_fc.weight | 1536×384 | 9.5935±0.1838 | 33.29±0.66 | 537.6±10.86 | 62.27±0.966 | 16.48±0.207 | 12.0±0.05 | 318.2 | 347.4 | 20.2 |
| _orig_mod.transformer.h.5.mlp.c_proj.weight | 384×1536 | 6.5697±0.3006 | 28.27±1.22 | 339.0±13.77 | 19.17±1.650 | 77.75±3.827 | 18.5±0.26 | 234.7 | 298.1 | 27.8 |
| _orig_mod.transformer.wpe.weight | 256×384 | 4.6970±0.0855 | 13.88±0.29 | 138.5±2.41 | 12.46±0.255 | 26.37±0.620 | 8.7±0.08 | 152.6 | 171.5 | 75.5 |
| _orig_mod.transformer.wte.weight | 65×384 | 3.9801±0.0926 | 9.95±0.23 | 70.5±1.67 | 4.46±0.103 | 26.46±0.817 | 6.2±0.06 | 57.4 | 60.3 | 10.1 |

## Summary by Layer Type

### c_attn

- **Count:** 6
- **Operator norm (mean):** 4.6293 ± 0.0808
- **Operator norm (range):** [3.3336, 5.8826]

### attn.c_proj

- **Count:** 6
- **Operator norm (mean):** 1.7427 ± 0.0423
- **Operator norm (range):** [0.7733, 2.2033]

### mlp.c_fc

- **Count:** 6
- **Operator norm (mean):** 10.1372 ± 0.2089
- **Operator norm (range):** [8.7494, 10.7750]

### mlp.c_proj

- **Count:** 6
- **Operator norm (mean):** 4.4698 ± 0.1473
- **Operator norm (range):** [2.7757, 6.5697]

### other

- **Count:** 3
- **Operator norm (mean):** 4.2191 ± 0.0902
- **Operator norm (range):** [3.9801, 4.6970]

## Recommendations for Grassmann Scaling

Since Grassmann manifold fixes operator norm = 1, we need to scale outputs to match baseline:

- **c_attn**: Scale by **4.6293 ± 0.0808** (mean operator norm across seeds)
- **mlp.c_fc**: Scale by **10.1372 ± 0.2089** (mean operator norm across seeds)
- **attn.c_proj**: Scale by **1.7427 ± 0.0423** (mean operator norm across seeds)

### Implementation

```python
# In model.py, multiply Grassmann weight outputs by these factors:

# c_attn: scale = 4.6293
self.c_attn_scale = 4.6293  # Robust across 10 seeds (±0.0808)

# mlp.c_fc: scale = 10.1372
self.mlp_c_fc_scale = 10.1372  # Robust across 10 seeds (±0.2089)

# attn.c_proj: scale = 1.7427
self.attn_c_proj_scale = 1.7427  # Robust across 10 seeds (±0.0423)
```
