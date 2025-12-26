# Comparison: Muon/Modular Duality vs Our Grassmann Approach

## Summary of Findings

Based on analysis of the Muon optimizer literature and related papers (arXiv 2410.21265 "Modular Duality in Deep Learning", arXiv 2506.15054 "Muon Optimizes Under Spectral Norm Constraints", arXiv 2510.03871 optimal scaling), here's how their approach compares to ours:

---

## 1. Muon Optimizer Approach

### Core Mechanism
- **Orthogonalization**: Uses Newton-Schulz iteration to find nearest semi-orthogonal matrix to gradient update
- **Spectral constraint**: Implicitly enforces `||W||_op ≤ 1/λ` where λ is weight decay
- **Update process**:
  1. Compute momentum update G
  2. Normalize: `G = G / ||G||_F` (Frobenius norm)
  3. Orthogonalize via Newton-Schulz iteration
  4. Apply to weights

### Norm Handling
- **Spectral norm constraint**: Controlled by weight decay parameter λ
- **Pre-orthogonalization**: Frobenius norm rescaling
- **Post-orthogonalization**: **NO explicit scaling to match baseline norms**
- **Key insight**: Constraint is `||W||_op ≤ 1/λ`, not fixed at 1

### Applied To
- 2D parameter matrices (hidden layers)
- Transformer Q, K, V matrices separately
- Conv filters (flattened)

---

## 2. Modular Duality Approach (arXiv 2410.21265)

### Core Mechanism
- **Duality mapping**: Maps gradients from dual space to primal space before weight update
- **Operator norm assignment**: Assigns norms to layers "based on semantics of each layer"
- **Recursive construction**: Builds full network duality map from layer-wise norms

### Norm Handling
- Mentions "operator norms to layers" but specifics not clear from abstract
- Unifies μP (maximal update parameterization) and Shampoo
- Focus on creating theoretically principled gradient transformations

### Key Difference from Muon
- More general theoretical framework
- Not necessarily orthogonalization-based
- Emphasizes correct gradient space transformations

---

## 3. Our Grassmann Manifold Approach

### Core Mechanism
- **Grassmann manifold**: G(a,b,r) = {bI + c·P : P rank-r projector, c = a-b}
- **Fixed operator norm**: G(1,0,r) has eigenvalues {1,1,...,1,0,...,0} → **||W||_op = 1 exactly**
- **Optimization**: Use specialized Grassmann optimizer (dual ascent + matrix sign)

### Current Implementation (Phase 2.5)
✓ Grassmann optimization on c_attn and mlp.c_fc
✓ **NO scaling applied** - using raw Grassmann weights with ||W||_op = 1
✗ **Missing**: Post-Grassmann scaling to match baseline norms

### Baseline Norms from 10-Seed Analysis
From our multi-seed analysis:
- **c_attn**: 4.63 ± 0.08 operator norm
- **mlp.c_fc**: 10.14 ± 0.21 operator norm
- **mlp.c_proj**: 4.47 ± 0.15 operator norm
- **attn.c_proj**: 1.74 ± 0.04 operator norm

### The Gap
Our Grassmann weights have ||W||_op = 1, but baseline learns:
- c_attn: **4.6× larger**
- mlp.c_fc: **10.1× larger**
- mlp.c_proj: **4.5× larger**

---

## 4. Key Differences: Muon vs Our Approach

| Aspect | Muon | Our Grassmann Approach |
|--------|------|------------------------|
| **Norm constraint** | `||W||_op ≤ 1/λ` (weight decay) | `||W||_op = 1` exactly (manifold) |
| **Flexibility** | Adjustable via λ | Fixed by manifold choice |
| **Applied to** | Gradient **updates** | Weight **matrices** themselves |
| **Scaling** | Controlled by weight decay | **Need explicit post-scaling** |
| **Theoretical basis** | Constrained optimization | Riemannian manifold optimization |
| **Empirical tuning** | λ controls effective norm | Need to measure baseline & scale |

### Critical Insight
**Muon doesn't need explicit norm matching because:**
1. Weight decay λ controls the spectral bound `1/λ`
2. They optimize in update space, not weight space
3. Effective norm is tunable via λ

**We DO need explicit norm matching because:**
1. Grassmann manifold **fixes** ||W||_op = 1
2. We optimize weights directly on manifold
3. No degree of freedom to adjust effective norm
4. **Solution**: Multiply Grassmann output by baseline norm

---

## 5. Our Proposed Solution

### What We're Missing
Currently, Phase 2.5 uses raw Grassmann weights:
```python
# Current (WRONG scale)
x = self.c_attn(x)  # Output has ||W||_op = 1
```

### What We Should Add
Post-Grassmann scaling to match baseline operator norms:
```python
# Proposed (CORRECT scale)
class CausalSelfAttention:
    def __init__(self):
        # ... existing code ...
        self.c_attn_scale = 4.6293  # From 10-seed analysis

    def forward(self, x):
        if using_grassmann:
            x = self.c_attn(x) * self.c_attn_scale  # Scale by baseline op norm
        else:
            x = self.c_attn(x)  # AdamW learns its own scale
```

Similarly for MLP:
```python
class MLP:
    def __init__(self):
        # ... existing code ...
        self.c_fc_scale = 10.1372  # From 10-seed analysis

    def forward(self, x):
        if using_grassmann:
            x = self.c_fc(x) * self.c_fc_scale  # Scale by baseline op norm
        else:
            x = self.c_fc(x)
```

### Why This Should Work
1. **Grassmann provides regularization** (rank constraint, manifold structure)
2. **Scaling restores magnitude** to match baseline's learned scale
3. **Best of both worlds**:
   - Manifold structure for better optimization geometry
   - Correct output scale for network functionality

### Expected Impact
Current Phase 2.5: **1.4633** (baseline: 1.4596, gap: **+0.0037**)

With operator norm scaling:
- **Conservative**: 1.45-1.46 (match baseline)
- **Optimistic**: 1.42-1.44 (beat baseline due to better regularization)

Reasoning: Phase 2.5 already shows **better generalization** (gap 0.317 vs 0.405), suggesting manifold constraint helps. The 0.0037 gap is likely just scale mismatch.

---

## 6. Block-Level Variance Considerations

From our variance analysis, we found:

### Uniform Layers (Can Use Single Scale)
- **mlp.c_fc**: CV = 7.28% across blocks → **Use global scale 10.14**
- **c_attn**: CV = 18.60% across blocks → **Use global scale 4.63** (acceptable)

### Variable Layers (May Need Per-Block Scales)
- **mlp.c_proj**: CV = 28.74%, range [2.78, 6.57] → Consider depth-dependent scaling
- **attn.c_proj**: CV = 26.71%, Block 0 outlier (0.77 vs 1.7-2.2) → Special case first block?

### Recommendation
**Start simple**: Use global scales for all layers (Phase 2.5 only uses c_attn + mlp.c_fc)

**If needed**: Add per-block scaling based on our analysis (Table in logs/561359_analyze_variance.out)

---

## 7. Implementation Priority

### High Priority ⭐
1. Add operator norm scaling for c_attn (× 4.63)
2. Add operator norm scaling for mlp.c_fc (× 10.14)
3. Test Phase 2.5 with scaling vs without

### Medium Priority
4. Test if scaling helps other phases (0.5, 1.5)
5. Experiment with per-block vs global scaling

### Low Priority
6. Investigate mlp.c_proj depth-dependent scaling
7. Compare Frobenius norm vs operator norm scaling

---

## 8. Comparison Summary

**Muon's approach**: Let weight decay control effective norm → Works because they optimize updates

**Our approach**: Fix norm via manifold, then scale explicitly → Necessary because we optimize weights

**Key lesson**: Both approaches recognize importance of controlling spectral norms, but implementation differs based on whether you optimize in weight space (us) or update space (Muon).

**Our advantage**: Direct manifold optimization may provide better geometry and implicit regularization (as evidenced by Phase 2.5's smaller gen gap).

**Next step**: Implement scaling and validate that it closes the 0.0037 gap with baseline!
