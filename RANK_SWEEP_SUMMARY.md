# Phase 2.5 Rank Matching Sweep - SUBMITTED

## Jobs Submitted

| Job ID | Variant | c_attn rank | mlp.c_fc rank | c_attn scale | mlp.c_fc scale | Strategy |
|--------|---------|-------------|---------------|--------------|----------------|----------|
| 561589 | **stable_rank** | 24 | 10 | 4.6293 | 10.1372 | Match stable rank exactly ⭐ |
| 561590 | **rank99** | 345 | 345 | 4.6293 | 10.1372 | Match rank@99% (nearly full) |
| 561591 | **stable_rank_x2** | 48 | 20 | 4.6293 | 10.1372 | 2× stable rank (conservative) |
| 561592 | **stable_rank_x3** | 72 | 30 | 4.6293 | 10.1372 | 3× stable rank (very conservative) |

## Current Baseline for Comparison

- **Baseline AdamW**: Val loss 1.4596 (10-seed average: 1.4710 ± 0.0024)
- **Phase 2.5 (original, rank=192)**: Val loss 1.4633 (+0.0037 worse than baseline)

## Predictions

### Most Likely Best: Variant 1 (stable_rank)
- **Rank**: c_attn=24, mlp.c_fc=10
- **Regularization**: Strong (8-19× reduction from current rank=192)
- **Theory**: Perfect match - Grassmann stable rank = r
- **Expected**: 1.42-1.45 (match or beat baseline)
- **Reasoning**:
  - Exact theoretical match of both stable rank and operator norm
  - Current Phase 2.5 shows better gen gap (0.317 vs 0.405)
  - With correct scaling + stronger regularization → likely to win

### Conservative Hedge: Variant 3 (stable_rank_x2)
- **Rank**: c_attn=48, mlp.c_fc=20
- **Regularization**: Medium (4-10× reduction from current)
- **Expected**: 1.44-1.46
- **Reasoning**: Still much better than rank=192, safer choice

### Likely Worst: Variant 2 (rank99)
- **Rank**: c_attn=345, mlp.c_fc=345
- **Regularization**: Minimal (90% of full rank 384)
- **Expected**: 1.46-1.48 (worse than current)
- **Reasoning**: Defeats purpose of manifold constraint, almost no regularization

### Middle: Variant 4 (stable_rank_x3)
- **Rank**: c_attn=72, mlp.c_fc=30
- **Regularization**: Weak-Medium
- **Expected**: 1.45-1.47

## Key Changes vs Original Phase 2.5

**Original (Job 560532)**:
```python
grass_rank = 192  # Uniform for all layers
scale = 1.0       # No scaling
```

**New Variants**:
```python
# Per-layer rank matching baseline stable rank
grass_rank_c_attn = 24      # vs 192 (8× lower!)
grass_rank_mlp_fc = 10      # vs 192 (19× lower!)

# Per-layer scaling matching baseline operator norm
grass_scale_c_attn = 4.6293    # vs 1.0 (4.6× higher output!)
grass_scale_mlp_fc = 10.1372   # vs 1.0 (10× higher output!)
```

## Implementation Details

Modified `model.py` to support:
1. Per-layer rank configuration via `grass_rank_c_attn`, `grass_rank_mlp_fc`, etc.
2. Per-layer scaling via `grass_scale_c_attn`, `grass_scale_mlp_fc`, etc.
3. Falls back to `grass_rank` if per-layer not specified (backward compatible)

The scaling is applied in the Grassmann optimizer's scale parameter, which multiplies the output after manifold projection.

## Timeline

- **Start**: ~6 hours per job (15000 iterations)
- **Expected completion**: ~6 hours from submission
- **Check status**: `squeue -u junyuren | grep phase2.5`
- **Monitor logs**: `tail -f logs/561589_phase2.5_stable_rank.out`

## Next Steps

1. Wait for jobs to complete (~6 hours)
2. Run `python3 track_progress.py` to compare results
3. Analyze which rank matching strategy works best
4. If stable_rank wins (as predicted), apply to other phases (0.5, 1.5)
5. Consider per-block configuration if needed (variance analysis showed some layers vary by depth)

## Success Criteria

**Minimal success**: Any variant ≤ 1.4633 (match current Phase 2.5)
**Good success**: Any variant ≤ 1.4596 (match/beat baseline)
**Excellent success**: Variant 1 < 1.45 (clearly beat baseline with better generalization)

Given Phase 2.5 already shows 0.317 vs 0.405 generalization gap, we have strong evidence that Grassmann regularizes better. The 0.0037 gap is likely just scaling mismatch, which we've now fixed!
