# Grassmann Gating Architecture: Block-Level + Head-Level Hybrid

**Date:** December 31, 2025
**Status:** Ready for Implementation
**Motivation:** Provide input-dependent adaptive magnitude for Grassmann-constrained layers while respecting multi-head attention structure

---

## Core Design Principle

**Grassmann Constraint:** ||W||_op = 1 (operator norm = 1)
- Provides strong regularization and direction
- But fixed magnitude is too restrictive

**Solution:** Input-dependent gating f(x) ∈ [0,1] via sigmoid
- **Grassmann provides direction** (||W||_op = 1, rank = r)
- **Gating provides adaptive magnitude** (f(x) × scale)
- **Formula:** output = W_grassmann(x) × f(x) × scale

**Key Insight:** Gating granularity should match natural structure
- c_attn: Multi-head attention → gate each head independently
- c_fc: Block decomposition → gate each block independently

---

## Architecture Overview

### Current Implementation (Phase 3)

**GPT-2 Medium:**
- n_layer = 24
- n_embd = 1024
- n_head = 16
- d_head = 64 (1024 / 16)

**Grassmann Layers:**
- c_attn: 3 square blocks (1024×1024) for Q, K, V
- c_fc: 4 square blocks (1024×1024)

**AdamW Layers (skip-facing):**
- attn.c_proj: 1024×1024
- mlp.c_proj: 1024×4096

---

## Gating Strategy

### For c_attn: Head-Level Gates Only

**Purpose:** Gate each attention head independently (respects multi-head structure)

**Why not block-level gates for c_attn?**
- Block gates (Q/K/V) + head gates would be redundant
- Each head would get effective scaling of `gate_Q_block × gate_Q[h]`
- Head-level gates alone provide sufficient flexibility

**Implementation:**
```python
# Step 1: Grassmann projection (NO gating yet)
qkv = self.c_attn(x) * self.grass_scale  # 3 blocks, scale=10.0
q, k, v = qkv.split(self.n_embd, dim=2)  # Each (B, T, 1024)

# Step 2: Split into 16 heads
Q = q.view(B, T, 16, 64).transpose(1, 2)  # (B, 16, T, 64)
K = k.view(B, T, 16, 64).transpose(1, 2)
V = v.view(B, T, 16, 64).transpose(1, 2)

# Compute per-head gates (48 total: 16 heads × 3 QKV)
gates = sigmoid(W_gates(x))  # W_gates: Linear(1024 → 48)
gate_Q, gate_K, gate_V = gates.split(16, dim=-1)  # Each (B, T, 16)

# Reshape gates: (B, T, 16) → (B, 16, T, 1)
gate_Q = gate_Q.transpose(1, 2).unsqueeze(-1)
gate_K = gate_K.transpose(1, 2).unsqueeze(-1)
gate_V = gate_V.transpose(1, 2).unsqueeze(-1)

# Apply per-head gating
Q = Q × gate_Q  # Each head independently modulated
K = K × gate_K
V = V × gate_V
```

**Parameters:** 48 gates × 1024 params = **49,152 params**

**Interpretation:**
- gate_Q[h]: "How much to use head h's query for this input"
- gate_K[h]: "How much to use head h's key for this input"
- gate_V[h]: "How much to use head h's value for this input"

---

### For c_fc: Block-Level Gates Only

**Purpose:** Gate each of the 4 MLP hidden blocks independently

**Why block-level gates for c_fc?**
- c_fc has no multi-head structure (just 4 expansion blocks)
- Block-level granularity is natural for 4×1024 → 4096 expansion
- Gating applied inside BlockDecomposedLinear

**Implementation:**
```python
# Inside BlockDecomposedLinear with use_block_gating=True
H_1 = W_1(x) × sigmoid(v_1^T x) × frozen_scale  # frozen_scale=0.5
H_2 = W_2(x) × sigmoid(v_2^T x) × frozen_scale
H_3 = W_3(x) × sigmoid(v_3^T x) × frozen_scale
H_4 = W_4(x) × sigmoid(v_4^T x) × frozen_scale

output = [H_1; H_2; H_3; H_4]  # Concatenate to R^4096

# Then in MLP.forward:
h = self.c_fc(x) * self.grass_scale  # scale=10.0, already block-gated
x = self.gelu(h)
```

**Parameters:** 4 gates × 1024 params = **4,096 params**

**Interpretation:**
- gate_i: "How much to use hidden block i for this input"
- Each block can specialize (e.g., block 1 for certain features)

---

### Total Gating Parameters

**Per Layer:**
- c_attn: 48 gates (head-level)
- c_fc: 4 gates (block-level)
- **Total:** 52 gates/layer
- **Params/layer:** 48×1024 + 4×1024 = **53,248 params**

**All 24 Layers:**
- **Total gates:** 24 × 52 = 1,248 gates
- **Total params:** 24 × 53,248 = **1,277,952 params** (~1.28M)

**Comparison to alternatives:**
- Current (elementwise G0): 176M params (137× more!)
- Qwen headwise G1: 394K params (0.31×, but doesn't respect Grassmann structure)
- Pure block-wise (7 gates): 168K params (0.13×, but no head diversity)
- Redundant (block+head for c_attn): 1.35M params (1.05×, wasteful)

---

## Implementation Details

### File: `manifold/block_decomposition.py`

**Add to `BlockDecomposedLinear.__init__`:**
```python
def __init__(self, in_features, out_features, n_blocks,
             frozen_scale=None, use_block_gating=False):
    # ... existing code ...

    # Block-level gating
    if use_block_gating:
        self.block_gates = nn.ModuleList([
            nn.Linear(in_features, 1, bias=False)
            for _ in range(n_blocks)
        ])
    else:
        self.block_gates = None
```

**Add to `BlockDecomposedLinear.forward`:**
```python
def forward(self, x):
    block_outputs = []
    for i, block in enumerate(self.blocks):
        out = block(x)  # Grassmann projection, ||W||_op=1

        # Apply block-level gate if enabled
        if self.block_gates is not None:
            gate = torch.sigmoid(self.block_gates[i](x))  # (B, T, 1)
            out = out * gate

        block_outputs.append(out)

    output = torch.cat(block_outputs, dim=-1)

    # Apply frozen scaling (e.g., 0.5 for c_fc)
    if self.frozen_scale is not None:
        output = output * self.frozen_scale

    return output
```

---

### File: `model.py`

**Add to `CausalSelfAttention.__init__`:**
```python
if self.use_grassmann:
    # Block decomposition (3 blocks for Q/K/V, NO block gating)
    self.c_attn = BlockDecomposedLinear(
        in_features=config.n_embd,
        out_features=3 * config.n_embd,
        n_blocks=3,
        frozen_scale=None,
        use_block_gating=False  # NO block gates (use head gates instead)
    )
    self.grass_scale = config.grass_scale  # e.g., 10.0

    # Head-level gating (48 gates: 16 heads × 3 QKV)
    self.head_gates = nn.Linear(config.n_embd, 3 * config.n_head, bias=False)
else:
    # Standard linear (no gating)
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
```

**Modify `CausalSelfAttention.forward`:**
```python
def forward(self, x):
    B, T, C = x.size()

    if self.use_grassmann:
        # Step 1: Grassmann projection (NO gating yet)
        qkv = self.c_attn(x)  # (B, T, 3072), NO block gating

        # Step 2: Apply global scaling
        qkv = qkv * self.grass_scale  # Scale by 10.0

        # Step 3: Split into Q, K, V
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each (B, T, 1024)

        # Step 4: Reshape to heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, 16, T, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Step 5: Compute and apply per-head gates
        gates = torch.sigmoid(self.head_gates(x))  # (B, T, 48)
        gate_q, gate_k, gate_v = gates.split(self.n_head, dim=-1)  # Each (B, T, 16)

        # Reshape gates: (B, T, 16) → (B, 16, T, 1)
        gate_q = gate_q.transpose(1, 2).unsqueeze(-1)
        gate_k = gate_k.transpose(1, 2).unsqueeze(-1)
        gate_v = gate_v.transpose(1, 2).unsqueeze(-1)

        # Apply per-head gating
        q = q * gate_q
        k = k * gate_k
        v = v * gate_v
    else:
        # Standard forward (no Grassmann, no gating)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # Continue with attention computation (same for both paths)
    # ... SDPA, output projection, etc ...
```

**Add to `MLP.__init__`:**
```python
if self.use_grassmann:
    # Block decomposition (4 blocks for 4× expansion)
    self.c_fc = BlockDecomposedLinear(
        in_features=config.n_embd,
        out_features=4 * config.n_embd,
        n_blocks=4,
        frozen_scale=0.5,  # Normalize 4 blocks → √1
        use_block_gating=True  # Enable block-level gates
    )
    self.grass_scale = config.grass_scale  # e.g., 10.0
else:
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
```

**Modify `MLP.forward`:**
```python
def forward(self, x):
    if self.use_grassmann:
        # Grassmann: block-gated inside BlockDecomposedLinear
        h = self.c_fc(x) * self.grass_scale  # Already has block gates + frozen_scale
        x = self.gelu(h)
    else:
        x = self.gelu(self.c_fc(x))

    x = self.c_proj(x)  # AdamW projection (skip-facing)
    x = self.dropout(x)
    return x
```

---

### File: `train.py`

**Add config parameter:**
```python
use_block_gating = True  # Enable block-level gates in BlockDecomposedLinear
use_head_gating = True   # Enable head-level gates in CausalSelfAttention
```

**Update `model_args`:**
```python
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, vocab_size=None,
    dropout=dropout,
    use_grassmann=use_grassmann,
    grass_rank=grass_rank,
    grass_scale=grass_scale,
    grass_a=grass_a,
    grass_b=grass_b,
    grass_lr=grass_lr,
    grassmann_dropout=grassmann_dropout,
    gate_lr=gate_lr,
    embed_lr=embed_lr,
    use_block_gating=use_block_gating,  # NEW
    use_head_gating=use_head_gating      # NEW
)
```

---

### Optimizer Groups

**Split into 4 groups with different learning rates:**

1. **Grassmann params:** grass_lr (e.g., 2e-3)
   - All BlockDecomposedLinear.blocks[*].weight

2. **Gate params:** gate_lr (e.g., 1.2e-3, 2× base)
   - BlockDecomposedLinear.block_gates[*].weight
   - CausalSelfAttention.head_gates.weight

3. **Projection params:** learning_rate (e.g., 6e-4)
   - c_proj.weight (attn and mlp)

4. **Embedding params:** embed_lr (e.g., 3e-4, 0.5× base)
   - wte.weight, wpe.weight

**Rationale:**
- Gates are new components learning from scratch → need higher LR
- Grassmann can use aggressive LR (8× boost validated on CIFAR-10)
- Projections are skip-facing → conservative LR
- Embeddings converge slowly → moderate LR

---

## Expected Benefits

### 1. Grassmann Advantage Maintained

**Problem:** Grassmann advantage shrinks over training (from -10.2% to -4.8%)

**Hypothesis:** Fixed scaling (x=10.0) becomes suboptimal as model matures

**Solution:** Input-dependent gating adapts magnitude dynamically
- Early training: Gates ≈ 0.5 (moderate scaling)
- Late training: Gates become sparse (selective scaling)

**Expected:** Advantage maintained or improved throughout training

---

### 2. Head Diversity Restored

**Problem:** All 16 heads share same Grassmann structure (1024×1024 block)

**Solution:** Per-head gates allow independent magnitude modulation
- Head 1: Specializes on local patterns → high gate values
- Head 2: Specializes on long-range → low gate values for irrelevant inputs
- Each head can adapt independently

**Expected:** Better multi-head utilization, improved attention quality

---

### 3. Parameter Efficiency

**Current (elementwise):** 176M gate params (49.7% of AdamW budget)

**Hybrid (head + block):** 1.28M gate params (0.72% of AdamW budget)

**Savings:** 174.72M params freed for other components!

---

### 4. Interpretability

**Block-level gates (4 per layer, c_fc only):**
- "How much to use each MLP hidden block for this input"
- Each block can specialize for different feature types

**Head-level gates (48 per layer, c_attn only):**
- "How much to activate head h's query/key/value"
- Can analyze: which heads are used for which inputs
- Natural for multi-head attention structure

**Expected:** Clear attention patterns, easier debugging

---

## Experimental Protocol

### Baseline vs Grassmann Comparison

**Baseline (AdamW + head-level gating):**
- Standard full-rank layers (no Grassmann)
- Same head-level gating (48 gates for c_attn)
- Fair comparison: both have head-gated multi-head attention

**Grassmann (Grassmann + hybrid gating):**
- BlockDecomposedLinear with rank=384
- Block-level gates (4 per layer, c_fc only)
- Head-level gates (48 per layer, c_attn only)
- Total: 52 gates per layer

**Controlled variables:**
- Same architecture: 24L, 1024d, 16 heads
- Same batch size and learning schedule
- Same number of tokens per iteration
- Only difference: Grassmann constraint + gating strategy

**Metrics:**
- Token-matched loss comparison
- Train/val gap (generalization)
- Gate statistics (mean, sparsity)
- Head utilization diversity

---

## Gating Function: Linear vs MLP (Design Choice)

### Current Design: Linear + Sigmoid

```python
gate_h = sigmoid(v_h^T x)  # Linear projection + sigmoid activation
```

**Why linear is sufficient:**

1. **Empirical precedent:** All successful gating mechanisms use linear
   - LSTM/GRU gates: σ(W[h,x] + b)
   - Qwen NeurIPS 2025 (best paper): σ(XW_θ)
   - Mixture-of-Experts routers: Linear

2. **Task separation:** Gating selects, Grassmann transforms
   - Gating task: "How much to use this component?" (binary-ish decision)
   - Feature transformation: Already done by Grassmann projection
   - Linear separability often sufficient for selection

3. **Sigmoid provides non-linearity:**
   - Steep transition around 0 (soft threshold)
   - Saturates at extremes (robust)
   - Maps (-∞, +∞) → (0, 1)

4. **Diversity through projections:**
   - 48 different learned directions (v_Q[1]...v_Q[16], v_K[1]...v_V[16])
   - Each v_h is a "feature detector" for when to activate head h
   - Collectively spans diverse input patterns

**Parameter efficiency:**
- Linear: 48 × 1024 = 49,152 params per layer
- Total: 1.28M params (24 layers)

---

### Alternative: MLP Gating (Future Consideration)

```python
# Shared MLP backbone
hidden = ReLU(W1 @ x)      # 1024 → 128
gates = sigmoid(W2 @ hidden)  # 128 → 48

# Parameters:
# W1: 1024 × 128 = 131K
# W2: 128 × 48 = 6K
# Total per layer: 137K (vs 49K linear)
# All 24 layers: 3.3M (vs 1.18M linear)
```

**When to consider MLP:**
- ❌ Gates don't learn diverse patterns (all v_h similar)
- ❌ Gates stuck at uniform values (e.g., all ≈0.5, not learning)
- ❌ Grassmann advantage doesn't materialize with linear gates

**Pros:**
- ✅ Can learn non-linear decision boundaries
- ✅ Can capture high-order interactions (x_i × x_j)
- ✅ Shared ReLU features across gates (efficiency)

**Cons:**
- ❌ No empirical evidence it's necessary for gating task
- ❌ 2.5× more parameters
- ❌ 2.5× slower gating computation
- ❌ Less interpretable (v_h is "activation direction", but ReLU features are opaque)

**Recommendation:** Start with linear (proven, efficient). Only switch to MLP if empirical results show linear is insufficient.

**Metrics to monitor:**
1. Gate diversity: `std(gates, dim=heads)` should be > 0.1
2. Gate sparsity: `mean(gates)` should be ≈ 0.1-0.3 (Qwen achieves 0.12)
3. Input-dependence: `correlation(gates[sample_i], gates[sample_j])` should vary

---

## Alternative: Full Block Decomposition (Future Work)

### Motivation

Current hybrid still shares Grassmann structure across heads:
- All 16 heads use slices of same 1024×1024 Grassmann matrix
- Heads can't have truly independent Grassmann structures

**Alternative:** Decompose 1024×1024 into 16×16 grid of 64×64 blocks

---

### Architecture: 16×16 Block Grid

**For Q projection (1024 → 1024):**

```
Input x: R^1024 (split into 16 segments of 64 dims)
x = [x_1, x_2, ..., x_16]  where x_i ∈ R^64

Output Q: R^1024 (split into 16 heads of 64 dims)
Q = [Q_1, Q_2, ..., Q_16]  where Q_i ∈ R^64

For each output head h ∈ {1..16}:
  For each input segment i ∈ {1..16}:
    W_{h,i}: Grassmann block (64×64), ||W_{h,i}||_op=1, rank=32
    contribution_{h,i} = W_{h,i} @ x_i  # Maps 64 → 64

  Q_h = sum_{i=1..16} contribution_{h,i}  # Aggregate 16 contributions
```

**Total blocks per projection:** 16×16 = **256 blocks**

**Each block:**
- Dimension: 64×64 (square, Grassmann-compatible!)
- Constraint: ||W_{h,i}||_op = 1
- Rank: 32 (50% of 64)
- Parameters per block: 64×64 = 4,096

**Total parameters for Q projection:**
- Standard: 1024×1024 = 1,048,576
- Block decomposition: 256 × (64×64) = 1,048,576 (same!)

---

### Gating Strategy for 16×16 Blocks

**Option 1: Per-head gates (16 gates)**
```python
For each head h:
  gate_h = sigmoid(v_h^T x)  # Single gate for head h
  Q_h = gate_h × sum_{i=1..16} W_{h,i} @ x_i
```
**Parameters:** 16 gates × 1024 = 16,384 params

**Option 2: Per-block gates (256 gates)**
```python
For each head h:
  For each input segment i:
    gate_{h,i} = sigmoid(v_{h,i}^T x)
    contribution_{h,i} = gate_{h,i} × W_{h,i} @ x_i
  Q_h = sum_{i=1..16} contribution_{h,i}
```
**Parameters:** 256 gates × 1024 = 262,144 params

**Option 3: Per-block from segment (256 gates, efficient)**
```python
For each head h:
  For each input segment i:
    gate_{h,i} = sigmoid(v_{h,i}^T x_i)  # Only look at segment i!
    contribution_{h,i} = gate_{h,i} × W_{h,i} @ x_i
  Q_h = sum_{i=1..16} contribution_{h,i}
```
**Parameters:** 256 gates × 64 = 16,384 params (same as Option 1!)

---

### Mathematical Formulation

**Standard 1024×1024 matrix:**
```
Q = W @ x  where W ∈ R^{1024×1024}
```

**16×16 block decomposition:**
```
W = [W_{h,i}]  where h,i ∈ {1..16}, W_{h,i} ∈ R^{64×64}

For output head h:
  Q_h = sum_{i=1}^{16} W_{h,i} @ x_i

In matrix form:
  [Q_1]   [W_{1,1} W_{1,2} ... W_{1,16}]   [x_1]
  [Q_2] = [W_{2,1} W_{2,2} ... W_{2,16}] @ [x_2]
  [...]   [............................]   [...]
  [Q_16]  [W_{16,1} W_{16,2} ... W_{16,16}] [x_16]
```

**Each block W_{h,i} is Grassmann-constrained:**
- ||W_{h,i}||_op = 1
- rank(W_{h,i}) = 32 (50% of 64)

---

### Implementation Sketch

**File: `manifold/block_decomposition.py`**

```python
class FullBlockDecomposedLinear(nn.Module):
    """
    Decomposes (out_features, in_features) into grid of square blocks.

    For 1024×1024:
      - n_blocks_out = 16 (output heads)
      - n_blocks_in = 16 (input segments)
      - block_size = 64
      - Total: 16×16 = 256 blocks of 64×64
    """
    def __init__(self, in_features, out_features, block_size=64,
                 use_block_gating=False, gating_mode='per_head'):
        super().__init__()
        assert in_features % block_size == 0
        assert out_features % block_size == 0

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.n_blocks_in = in_features // block_size  # 16
        self.n_blocks_out = out_features // block_size  # 16

        # Create grid of blocks (16×16 = 256 blocks of 64×64)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(block_size, block_size, bias=False)
                for _ in range(self.n_blocks_in)
            ]) for _ in range(self.n_blocks_out)
        ])

        # Gating
        if use_block_gating:
            if gating_mode == 'per_head':
                # 16 gates (one per output head)
                self.gates = nn.Linear(in_features, self.n_blocks_out, bias=False)
            elif gating_mode == 'per_block':
                # 256 gates (one per block)
                self.gates = nn.Linear(in_features,
                    self.n_blocks_out * self.n_blocks_in, bias=False)
            elif gating_mode == 'per_block_segment':
                # 256 gates, but each only sees its input segment
                self.gates = nn.ModuleList([
                    nn.ModuleList([
                        nn.Linear(block_size, 1, bias=False)
                        for _ in range(self.n_blocks_in)
                    ]) for _ in range(self.n_blocks_out)
                ])
            self.gating_mode = gating_mode
        else:
            self.gates = None

    def forward(self, x):
        # x: (B, T, 1024)
        B, T, _ = x.size()

        # Split input into 16 segments of 64
        x_segments = x.view(B, T, self.n_blocks_in, self.block_size)
        # x_segments: (B, T, 16, 64)

        # Compute output for each head
        outputs = []
        for h in range(self.n_blocks_out):
            # Aggregate contributions from all input segments
            head_output = torch.zeros(B, T, self.block_size, device=x.device)

            for i in range(self.n_blocks_in):
                # Block projection
                contribution = self.blocks[h][i](x_segments[:, :, i])  # (B, T, 64)

                # Apply gating
                if self.gates is not None:
                    if self.gating_mode == 'per_head':
                        gate_all = torch.sigmoid(self.gates(x))  # (B, T, 16)
                        gate = gate_all[:, :, h:h+1]  # (B, T, 1)
                    elif self.gating_mode == 'per_block':
                        gate_all = torch.sigmoid(self.gates(x))  # (B, T, 256)
                        gate = gate_all[:, :, h*self.n_blocks_in + i:h*self.n_blocks_in + i+1]
                    elif self.gating_mode == 'per_block_segment':
                        gate = torch.sigmoid(self.gates[h][i](x_segments[:, :, i]))  # (B, T, 1)

                    contribution = contribution * gate

                head_output = head_output + contribution

            outputs.append(head_output)

        # Concatenate all heads: (B, T, 16, 64) → (B, T, 1024)
        output = torch.cat(outputs, dim=-1)
        return output
```

---

### Comparison: Current vs Full Block Decomposition

| Property | Current (3 blocks) | Full Decomposition (256 blocks) |
|----------|-------------------|--------------------------------|
| **Block size** | 1024×1024 | 64×64 |
| **Blocks per layer** | 3 (Q/K/V) | 3×256 = 768 (Q/K/V grids) |
| **Head independence** | ❌ Shared structure | ✅ Independent per head |
| **Grassmann rank** | 384 (37.5% of 1024) | 32 (50% of 64) |
| **Parameters** | Same | Same |
| **Flexibility** | Lower (heads coupled) | Higher (heads decoupled) |
| **Complexity** | Simple | High |
| **Optimizer overhead** | Low (3 matrices) | High (768 matrices) |

---

### When to Use Full Block Decomposition

**Advantages:**
1. ✅ **True head independence:** Each head has own Grassmann structure
2. ✅ **Finer-grained control:** 64×64 blocks easier to optimize than 1024×1024
3. ✅ **Stronger rank constraint:** 50% rank ratio (32/64) vs 37.5% (384/1024)
4. ✅ **Mixture-of-experts flavor:** Each head selectively combines input segments

**Disadvantages:**
1. ❌ **Complexity:** 256× more blocks to manage
2. ❌ **Memory overhead:** 256 separate Grassmann optimizers
3. ❌ **Implementation complexity:** Nested loops, harder to debug
4. ❌ **No literature precedent:** Completely novel approach

**Recommendation:**
- **Start with current hybrid** (simple, proven to work)
- **If successful, try full decomposition** (maximum flexibility)
- **Likely use case:** Models where head diversity is critical (e.g., long-context)

---

## Next Steps

1. ✅ Implement hybrid gating (block + head) in `model.py` and `manifold/block_decomposition.py`
2. ✅ Create config files for baseline vs Grassmann comparison
3. ✅ Submit parallel SLURM jobs with checkpoint continuation
4. ⏳ Monitor gate statistics (mean, sparsity, per-head diversity)
5. ⏳ Analyze whether Grassmann advantage is maintained
6. 📋 (Future) If hybrid works well, implement full 16×16 block decomposition

---

**Status:** Ready for implementation
**Expected Implementation Time:** 2-3 hours
**Expected Training Time:** 12 hours (may need continuation jobs)
