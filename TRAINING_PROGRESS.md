# Training Progress Summary

**Total phases tracked:** 30

| Phase | Grass% | Best Val | Best Train | Gen Gap | Step @ Best | Current Iter | Status |
|-------|--------|----------|------------|---------|-------------|--------------|--------|
| phase2.5_stable_rank_x2 | N/A | 1.4590 | 1.0907 | 0.3683 | 7000 | 7930 | Running 52% |
| phase2.5_lr4x | N/A | 1.4616 | 1.0908 | 0.3708 | 7000 | 15000 | ✓ Complete |
| phase2.5 | 33% | 1.4633 | 1.1459 | 0.3174 | 3000 | 15000 | ✓ Complete |
| phase2.5_rank99 | N/A | 1.4668 | 1.0526 | 0.4142 | 6000 | 8340 | Running 55% |
| baseline_seed6 | N/A | 1.4671 | 1.1569 | 0.3102 | 1500 | 15000 | ✓ Complete |
| baseline_seed3 | N/A | 1.4686 | 1.1140 | 0.3546 | 1750 | 15000 | ✓ Complete |
| phase2.5_lr10x | N/A | 1.4686 | 1.0934 | 0.3752 | 7000 | 15000 | ✓ Complete |
| baseline_seed0 | N/A | 1.4694 | 1.1148 | 0.3546 | 1750 | 15000 | ✓ Complete |
| phase0.5 | 25% | 1.4696 | 1.0658 | 0.4038 | 2000 | 15000 | ✓ Complete |
| baseline_seed2 | N/A | 1.4700 | 1.1571 | 0.3129 | 1500 | 15000 | ✓ Complete |
| baseline_seed5 | N/A | 1.4704 | 1.1152 | 0.3552 | 1750 | 15000 | ✓ Complete |
| baseline_seed9 | N/A | 1.4705 | 1.1135 | 0.3570 | 1750 | 15000 | ✓ Complete |
| baseline_seed1 | N/A | 1.4717 | 1.1139 | 0.3578 | 1750 | 15000 | ✓ Complete |
| baseline_seed7 | N/A | 1.4733 | 1.1148 | 0.3585 | 1750 | 15000 | ✓ Complete |
| baseline_seed8 | N/A | 1.4737 | 1.1124 | 0.3613 | 1750 | 15000 | ✓ Complete |
| baseline_seed4 | N/A | 1.4752 | 1.1090 | 0.3662 | 1750 | 15000 | ✓ Complete |
| phase2.5_lr6x | N/A | 1.4756 | 1.0944 | 0.3812 | 7000 | 15000 | ✓ Complete |
| phase2.5_drop15 | N/A | 1.4758 | 1.0434 | 0.4324 | 6500 | 15000 | ✓ Complete |
| phase2.5_per_block | N/A | 1.4851 | 1.0866 | 0.3985 | 9500 | 15000 | ✓ Complete |
| phase2.5_stable_rank | N/A | 1.4877 | 1.1492 | 0.3385 | 7500 | 8200 | Running 54% |
| phase1.5 | 30% | 1.4955 | 1.1379 | 0.3576 | 2000 | 15000 | ✓ Complete |
| phase2.5_drop10 | N/A | 1.5055 | 1.0880 | 0.4175 | 4500 | 15000 | ✓ Complete |
| phase2 | 34% | 1.5568 | 1.2623 | 0.2945 | 11500 | 11810 | Running 78% |
| phase4 | 90% | 1.6550 | 1.4342 | 0.2208 | 3500 | 3960 | Running 26% |
| phase3 | 62% | 1.6583 | 1.4501 | 0.2082 | 7500 | 8470 | Running 56% |
| phase1 | 28% | 1.6762 | 1.4134 | 0.2628 | 4500 | 4780 | Running 31% |
| phase5 | 100% | 1.6988 | 1.4653 | 0.2335 | 12500 | 12720 | Running 84% |
| phase3.2 | 62% | 3.0923 | 3.0899 | 0.0024 | 500 | 690 | Running 4% |
| phase2.2 | 34% | 3.1976 | 3.1661 | 0.0315 | 15000 | 15000 | ✓ Complete |
| phase1.2 | 28% | 3.3442 | 3.3065 | 0.0377 | 8000 | 11880 | Running 79% |

## Phase Descriptions

- **baseline**: AdamW only (no Grassmann)
- **phase0.5**: c_attn (Grassmann), attn.c_proj (AdamW)
- **phase1**: c_attn + attn.c_proj (both Grassmann G(0,-1,r) or G(1,0,r))
- **phase1.2**: c_attn + attn.c_proj (G(1,0,r)) + **no skip**
- **phase1.5**: mlp.c_fc (Grassmann), attn.c_proj (AdamW)
- **phase2**: mlp.c_fc + attn.c_proj (both Grassmann)
- **phase2.2**: mlp.c_fc + attn.c_proj (G(1,0,r)) + **no skip**
- **phase2.5**: c_attn + mlp.c_fc (Grassmann), attn.c_proj (AdamW)
- **phase3**: c_attn + mlp.c_fc + attn.c_proj (all Grassmann)
- **phase3.2**: c_attn + mlp.c_fc + attn.c_proj (G(1,0,r)) + **no skip**
- **phase4**: Full MLP (c_fc + c_proj) + attn layers
- **phase5**: All layers G(1,0,r), no skip connections
