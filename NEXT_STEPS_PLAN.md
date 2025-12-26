# Next Steps: Validation and Scaling Plan

## Current Status

**Shakespeare char-level (1.1M tokens):**
- ✅ All configs converge to val loss ~1.46
- ✅ Best: stable_rank_x2 (1.4590) and lr4x (1.4616)
- ⚠️ **Dataset too small** - all models hit ceiling
- ⚠️ No test set - only train/val split

**Key insight:** Differences < 0.02 are likely noise on this tiny dataset

---

## Available Datasets

| Dataset | Train Size | Val Size | Total Tokens | Vocabulary | Use Case |
|---------|------------|----------|--------------|------------|----------|
| **shakespeare_char** | 1.0M | 111K | 1.1M | 65 chars | ✅ Done - too small |
| **openwebtext** | **9B** | 4.4M | 9B | GPT-2 BPE | 🎯 **Best for validation** |
| shakespeare (word) | Small | Small | ~1M | Words | Skip - similar to char |

**WikiText-2/103 status:** Not in data/ directory, would need to download

---

## Immediate Actions (Before Scaling Up)

### 1. Compute Perplexity from Val Loss ✅ Easy

**Formula:** `perplexity = exp(val_loss)`

Create script to convert all val losses to perplexity:
```python
# compute_metrics.py
import numpy as np

results = {
    'stable_rank_x2': 1.4590,
    'lr4x': 1.4616,
    'baseline': 1.4596,
}

for name, val_loss in results.items():
    perplexity = np.exp(val_loss)
    print(f"{name}: val_loss={val_loss:.4f}, perplexity={perplexity:.2f}")
```

**Expected output:**
- baseline: perplexity ≈ 4.30
- stable_rank_x2: perplexity ≈ 4.30
- lr4x: perplexity ≈ 4.31

**Still tiny differences!**

---

### 2. Additional Metrics to Compute

#### A. Sample Quality (Qualitative)
Generate text from each checkpoint and compare:
```bash
python sample.py --out_dir=out-phase2.5-stable-rank-x2 --num_samples=5
python sample.py --out_dir=out-baby-baseline --num_samples=5
```

**Metrics:**
- Coherence
- Grammar
- Diversity
- Repetition rate

#### B. Bits-per-Character/Byte
```python
bpc = val_loss / np.log(2)  # Convert nats to bits
```

#### C. Calibration (if we had test set)
- Expected Calibration Error (ECE)
- Reliability diagrams

#### D. Computational Efficiency
Already have from logs:
- MFU (Model FLOPs Utilization)
- Time per iteration
- Tokens/second

#### E. Generalization Gap (already tracking)
```python
gen_gap = val_loss - train_loss
```
- stable_rank_x2: 0.368
- baseline: 0.405
- **Grassmann wins by 9%!**

---

## Scaling to Larger Datasets

### Option 1: OpenWebText (9B tokens) 🎯 **RECOMMENDED**

**Pros:**
- ✅ Already available in data/
- ✅ **2000× larger** than Shakespeare
- ✅ Same as GPT-2 training
- ✅ BPE tokenization (more realistic)
- ✅ Large enough to differentiate architectures

**Cons:**
- Requires more compute (~100× longer training)
- Need to scale model size appropriately

**Estimated training time:**
- Shakespeare: 6 hours @ 15K iters
- OpenWebText (scaled): ~100-200 hours for meaningful training

**Recommended approach:**
1. Start with **Baby GPT scaled up**: 12L×768d (~40M params)
2. Train for 50K-100K iterations
3. Compare best Grassmann config vs baseline

---

### Option 2: WikiText-103 (100M tokens)

**Pros:**
- Medium size (100× Shakespeare)
- Word-level benchmark
- Established baseline scores

**Cons:**
- Need to download/prepare
- Still relatively small

**Estimated training time:** ~20-40 hours

---

### Option 3: WikiText-2 (2M tokens)

**Pros:**
- Only 2× Shakespeare, quick to validate
- Standard benchmark

**Cons:**
- May still hit ceiling quickly
- Not much larger than Shakespeare

---

## Recommended Next Steps

### Phase 1: Quick Validation (1-2 days)

**A. Compute all metrics on Shakespeare** ✓ Easy
```bash
# Create evaluation script
python compute_metrics.py \
  --checkpoints out-phase2.5-stable-rank-x2,out-phase2.5-lr4x,out-baby-baseline \
  --metrics perplexity,bpc,samples
```

**Metrics to compute:**
1. Perplexity
2. Bits-per-character
3. Sample 100 sequences per model
4. Compare sample quality

---

### Phase 2: Scale to OpenWebText (1-2 weeks)

**A. Prepare scaled model config**
```python
# config/train_openwebtext_baseline.py
n_layer = 12       # 2× current
n_embd = 768       # 2× current
n_head = 12
# ~40M parameters (vs 10M on Shakespeare)

dataset = 'openwebtext'
batch_size = 12    # Reduce due to larger model
max_iters = 100000 # More data needs more training
eval_interval = 1000
```

**B. Train baseline first** (validate setup)
```bash
sbatch submit_openwebtext_baseline.sh
# ~5-7 days on H100
```

**C. Apply winning Grassmann config**
```bash
# config/train_openwebtext_grassmann.py
# Use stable_rank_x2 or lr4x settings
# Scale ranks proportionally: r_c_attn = 96 (vs 48), r_mlp_fc = 40 (vs 20)

sbatch submit_openwebtext_grassmann.sh
```

**D. Compare on held-out validation set**
- Perplexity
- Sample quality
- Training efficiency (tokens/second)
- Generalization gap

---

### Phase 3: Advanced Evaluation (optional)

**A. Transfer learning**
- Fine-tune both on downstream task
- Compare few-shot performance

**B. Scaling laws**
- Train multiple sizes (10M, 40M, 100M)
- Plot Grassmann vs baseline scaling

**C. Ablation studies**
- Per-layer contribution analysis
- Rank sensitivity on larger dataset

---

## Resource Estimation

### Shakespeare Metrics (immediate)
- **Time:** 1-2 hours
- **Compute:** CPU only
- **Storage:** Negligible

### OpenWebText Baseline
- **Time:** 5-7 days (100K iters @ 3-4s/iter)
- **Compute:** 1 H100 GPU
- **Storage:** ~500MB checkpoints

### OpenWebText Grassmann
- **Time:** 5-7 days per config
- **Compute:** 1 H100 GPU
- **Storage:** ~500MB checkpoints

**Total for full validation:** ~2-3 weeks

---

## Decision Matrix

| Dataset | Size | Time | Differentiation | Recommendation |
|---------|------|------|----------------|----------------|
| Shakespeare | 1.1M | ✓ Done | ❌ Too small | Compute metrics only |
| WikiText-2 | 2M | 1 day | ⚠️ Marginal | Skip |
| WikiText-103 | 100M | 1 week | ✓ Good | Consider |
| **OpenWebText** | **9B** | **2 weeks** | **✓✓ Best** | **Do this** |

---

## Immediate TODO

### This Week:
1. ✅ Create `compute_metrics.py` for perplexity/BPC/samples
2. ✅ Generate sample text from top 3 checkpoints
3. ✅ Compare sample quality manually
4. ✅ Create OpenWebText baseline config
5. ✅ Submit OpenWebText baseline training

### Next Week:
6. Monitor OpenWebText baseline
7. Apply winning Grassmann config to OpenWebText
8. Compare results on 9B token dataset

---

## Success Criteria

### Weak Success:
- Grassmann **matches** baseline on OpenWebText
- Proves approach scales beyond tiny datasets

### Strong Success:
- Grassmann **beats** baseline by >1% perplexity on OpenWebText
- Better generalization gap (as on Shakespeare)
- Competitive or better training efficiency

### Excellent Success:
- Grassmann beats baseline by >5% perplexity
- Demonstrates scaling law advantages
- Opens path to larger models (GPT-2 scale)

---

## Questions to Answer

1. **Does Grassmann advantage hold on 9B tokens?**
   - Shakespeare too small to tell
   - OpenWebText will definitively answer

2. **Does lower generalization gap translate to better test performance?**
   - Shakespeare: Grassmann has 9% better gap
   - Will this show as perplexity improvement on OpenWebText?

3. **Is computational overhead worth it?**
   - Grassmann adds ~10-20% training time (dual ascent)
   - If >10-20% perplexity improvement → worth it

4. **How do ranks scale?**
   - Shakespeare: r=48,20 optimal
   - OpenWebText (4× model size): r=96,40? r=192,80?

Should I start by creating the metrics computation script?
