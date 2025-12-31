# Phase 1 Implementation Log: Gating at d=384

**Date Started:** December 26, 2024
**Goal:** Implement and validate head-wise gating mechanism before Grassmann integration
**Status:** In Progress

---

## Design Decisions (from paper analysis)

### 1. Gate Initialization ✓
- **Decision:** `bias=False`, default PyTorch initialization
- **Source:** Paper Section 2.2, Formula 5: `σ(XW_θ)` (no bias term shown)
- **Source:** Paper Section 3.1: "Other hyperparameters follow default AdamW"
- **Implementation:** `nn.Linear(n_embd, n_head, bias=False)`
- **Initial gate values:** sigmoid(small_random) ≈ 0.5 with variance
- **Expected after training:** mean ≈ 0.116 (sparse)

### 2. Dataset ✓
- **Decision:** OpenWebText (not shakespeare_char)
- **Reason:** Need realistic gate statistics for Phase 3 amplification factor
- **Risk:** If gate_mean differs by 33%, amplification will be off → Test on real data first

### 3. Gate Configuration ✓
- **Type:** Head-wise (not element-wise)
- **Parameters:** 16K per layer (n_embd × n_head = 384 × 6)
- **Comparison:** Element-wise would be 1M params (only 0.03 PPL better, not worth it)
- **Position:** G1 (after SDPA, before c_proj)
- **Activation:** Sigmoid (sparse in [0,1])
- **Gate input:** Pre-norm hidden states `x` (before attention)

### 4. Training Hyperparameters ✓
- **Learning rate:** Same as base (6e-4 for Baby GPT, will scale for GPT-2 Medium)
- **Logging frequency:** Every 500 iters
- **Gate statistics:** Per-layer mean/std/sparsity

### 5. Success Criteria
- ✅ **Gate mean ∈ [0.08, 0.25]** (matches paper's 0.116)
- ✅ **Sparsity >70%** (values <0.1)
- ✅ **Loss decreases** (no catastrophic failure)
- ✅ **Statistics converge** (not drifting)

**Failure recovery:**
- If gate_mean >0.45 → Not learning sparsity, retry with bias=-2.0 init
- If gate_mean <0.02 → Too sparse, may block gradients
- If loss plateaus → Gating destabilizing, reduce LR or remove

---

## Implementation Steps

### Step 1: Modify model.py - Add gating to CausalSelfAttention ✓
- [x] Add `use_gating` and `gate_type` config parameters
- [x] Add `self.gate = nn.Linear()` in __init__
- [x] Modify forward() to apply gate after SDPA
- [x] Handle head-wise gating shape: (B, T, nh) → (B, nh, T, 1)

**Status:** Complete (lines 134-136, 55-69, 93-105 in model.py)

### Step 2: Create config/train_openwebtext_gating_d384.py ✓
- [x] n_layer=6, n_embd=384, n_head=6 (Baby GPT)
- [x] use_gating=True, gate_type='headwise'
- [x] dataset='openwebtext'
- [x] out_dir='/net/scratch2/junyuren/nanoGPT-manifold/out-owt-gating-d384'
- [x] max_iters=10000 (quick test)
- [x] eval_interval=500

**Status:** Complete (config/train_openwebtext_gating_d384.py)

### Step 3: Create submit_openwebtext_gating.sh ✓
- [x] SLURM config: H100 GPU, 8 CPUs, 32GB RAM
- [x] Time limit: 4 hours (conservative for 10k iters)
- [x] Activate manifold_muon conda env
- [x] Run train.py with config

**Status:** Complete (submit_openwebtext_gating.sh)

### Step 4: Create analyze_gate_stats.py ✓
- [x] Load checkpoint from /net/scratch2/
- [x] Hook into forward pass to capture gate values
- [x] Compute mean, std, sparsity per layer
- [x] Visualize distribution
- [x] Print statistics

**Status:** Complete (analyze_gate_stats.py)

### Step 5: Run and validate ⏳ IN PROGRESS
- [x] Submit SLURM job → **Job 603162** submitted
- [ ] Monitor training loss
- [ ] After 10k iters, analyze gate statistics
- [ ] Compare to paper's target (mean=0.116)
- [ ] Document actual gate_mean for Phase 3

**Status:** Job queued, scheduled start at 11:22:52 on node l001 (H100)

---

## Critical Reminders

### 🚨 NEVER DO:
1. ❌ Run `python train.py` directly (use sbatch!)
2. ❌ Save checkpoints to /home/junyuren/ (disk quota!)
3. ❌ Use element-wise gating (64× more params, negligible gain)

### ✅ ALWAYS DO:
1. ✅ Save checkpoints to `/net/scratch2/junyuren/nanoGPT-manifold/`
2. ✅ Use `sbatch submit_*.sh` for all training
3. ✅ Request H100 GPU with `--gres=gpu:h100:1`
4. ✅ Log job ID in filename: `logs/%j_gating_test.out`

---

## Expected Timeline

- **Step 1-4 (Implementation):** 1-2 hours
- **Step 5 (SLURM job):** 3 hours (10k iters on H100)
- **Analysis:** 30 minutes
- **Total:** 4-5 hours

**Next Phase:** Once gate_mean measured, proceed to Phase 2 (GPT-2 Medium baseline)

---

## Implementation Notes

### Gate Forward Pass Logic
```python
# After SDPA (y is attention output)
y = att @ v  # (B, nh, T, hs)

# Apply head-wise gating
if self.gate is not None:
    g = torch.sigmoid(self.gate(x))  # (B, T, nh) using pre-norm x
    g = g.transpose(1, 2).unsqueeze(-1)  # (B, nh, T, 1)
    y = y * g  # Multiplicative gating

# Continue to output projection
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.c_proj(y)
```

### SLURM Job Naming Convention
```bash
#SBATCH --output=logs/%j_gating_d384.out
#SBATCH --error=logs/%j_gating_d384.err
```

### Measurement Protocol
```python
# Every 500 iters during training
if iter_num % 500 == 0:
    for layer in model.transformer.h:
        if hasattr(layer.attn, 'gate_stats'):
            stats = layer.attn.gate_stats
            print(f"Layer {i}: gate_mean={stats['mean']:.4f}, sparsity={stats['sparsity']:.2%}")
```

---

## Job Submission Log

### Job 603162 - Phase 1 Gating Validation
**Submitted:** 2025-12-26 08:47:16
**Status:** PENDING (waiting for resources)
**Scheduled Start:** 2025-12-26 11:22:52
**Node:** l001
**GPU:** H100
**Resources:** 8 CPUs, 32GB RAM
**Time Limit:** 4 hours
**Output:** `/home/junyuren/nanoGPT/logs/603162_gating_d384.out`
**Error:** `/home/junyuren/nanoGPT/logs/603162_gating_d384.err`
**Checkpoint:** `/net/scratch2/junyuren/nanoGPT-manifold/out-owt-gating-d384/ckpt.pt`

**Expected Completion:** ~3 hours after start (~14:22:52)

### Monitoring Commands
```bash
# Check job status
squeue -u $USER
scontrol show job 603162

# Monitor output (when running)
tail -f logs/603162_gating_d384.out

# Check for completion
ls -lh /net/scratch2/junyuren/nanoGPT-manifold/out-owt-gating-d384/
```

### After Completion
```bash
# Analyze gate statistics
python analyze_gate_stats.py --ckpt /net/scratch2/junyuren/nanoGPT-manifold/out-owt-gating-d384/ckpt.pt

# View plot
open plots/gate_statistics.png
```

---

**Current Status:** Phase 1 implementation complete, job submitted
**Next Action:** Wait for job completion, then analyze gate statistics
