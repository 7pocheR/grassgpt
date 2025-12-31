"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Manifold Muon optimizer support
from manifold import grassmann_muon_update, initialize_on_grassmann, BlockDecomposedLinear

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V projections - use block decomposition if Grassmann enabled
        self.use_grassmann = getattr(config, 'use_grassmann', False)
        if self.use_grassmann:
            # Decompose into 3 blocks (Q, K, V) with NO frozen scaling, NO block gating
            self.c_attn = BlockDecomposedLinear(
                config.n_embd,
                3 * config.n_embd,
                n_blocks=3,
                frozen_scale=None,  # Components split immediately, no norm issue
                use_block_gating=False,  # Use head-level gates instead
                bias=False  # Grassmann requires bias=False
            )
            self.grass_scale = config.grass_scale  # x=10 uniform scaling

            # Head-level gating (48 gates: 16 heads × 3 QKV)
            # Provides input-dependent magnitude for each head independently
            self.head_gates = nn.Linear(config.n_embd, 3 * config.n_head, bias=False)
        else:
            # Standard combined linear
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection - ALWAYS AdamW (faces skip connection)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Q, K, V projections with Grassmann scaling and head-level gating
        if self.use_grassmann:
            # Step 1: Grassmann projection (NO gating yet)
            qkv = self.c_attn(x)  # (B, T, 3*n_embd), BlockDecomposedLinear with NO block gating

            # Step 2: Apply global scaling
            qkv = qkv * self.grass_scale  # Scale by 10.0

            # Step 3: Split into Q, K, V
            q, k, v = qkv.split(self.n_embd, dim=2)  # Each (B, T, n_embd)

            # Step 4: Reshape to heads
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            # Step 5: Compute and apply per-head gates
            gates = torch.sigmoid(self.head_gates(x))  # (B, T, 3*n_head) = (B, T, 48)
            gate_q, gate_k, gate_v = gates.split(self.n_head, dim=-1)  # Each (B, T, n_head)

            # Reshape gates: (B, T, n_head) → (B, n_head, T, 1)
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
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Expansion layer - use block decomposition if Grassmann enabled
        self.use_grassmann = getattr(config, 'use_grassmann', False)
        if self.use_grassmann:
            # Decompose into 4 blocks with frozen 0.5× scaling and block-level gating
            self.c_fc = BlockDecomposedLinear(
                config.n_embd,
                4 * config.n_embd,
                n_blocks=4,
                frozen_scale=0.5,  # CRITICAL: 0.5× to normalize √4 → 1
                use_block_gating=True,  # 4 block-level gates (one per MLP block)
                bias=False  # Grassmann requires bias=False
            )
            self.grass_scale = config.grass_scale  # x=10 uniform scaling
            # Note: Block-level gating happens inside BlockDecomposedLinear
        else:
            # Standard linear
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        self.gelu = nn.GELU()

        # Contraction layer - ALWAYS AdamW (faces skip connection)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Apply c_fc with Grassmann scaling if enabled
        if self.use_grassmann:
            # Grassmann: ||W||_op=1, block gating (inside BlockDecomposedLinear), frozen 0.5×, then grass_scale
            # Block gating provides input-dependent magnitude per MLP block
            h = self.c_fc(x) * self.grass_scale  # (B, T, 4*n_embd), already block-gated inside
        else:
            h = self.c_fc(x)

        x = self.gelu(h)
        x = self.c_proj(x)  # AdamW layer (faces skip)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.remove_attn_skip = False  # Can be set to True for .2 phase variants

    def forward(self, x):
        if self.remove_attn_skip:
            x = self.attn(self.ln_1(x))  # No residual connection
        else:
            x = x + self.attn(self.ln_1(x))  # Standard residual
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    grassmann_dropout: float = None  # Dropout for Grassmann components (if None, use dropout/2)
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # Grassmann manifold optimization with hybrid gating
    use_grassmann: bool = False  # Enable Grassmann for c_attn, mlp.c_fc (not skip-facing)
    grass_rank: int = None  # Rank for G_{a,b,r} (if None, use 50% of n_embd)
    grass_scale: float = 10.0  # Uniform x=10 scaling factor
    grass_a: float = 1.0  # G_{a,b,r} first eigenvalue (1.0 for non-skip layers)
    grass_b: float = 0.0  # G_{a,b,r} second eigenvalue (0.0 for non-skip layers)
    grass_lr: float = 8e-3  # Learning rate for Grassmann layers (8× boost)
    gate_lr: float = None  # Learning rate for gating networks (None = 2× learning_rate)
    embed_lr: float = None  # Learning rate for embeddings (None = 0.5× learning_rate)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_grassmann_weights(self, a=0.0, b=-1.0, rank=None):
        """
        Initialize square attention c_proj weights on the Grassmann manifold G_{a,b,r}.

        This should be called AFTER model initialization when using Grassmann optimizer.

        Args:
            a, b: Two eigenvalues defining the manifold
                  Use (0.0, -1.0) for skip connections (recommended)
            rank: Projector rank (if None, use 50% of n_embd for simplicity)
                  CIFAR-10 optimal was 31% (160/512), but 50% is easier to implement uniformly
        """
        if rank is None:
            rank = int(0.5 * self.config.n_embd)  # 50% (n/2) for simplicity

        print(f"\n=== Initializing Grassmann Weights ===")
        print(f"Manifold: G_{{{a}, {b}, {rank}}}")

        for name, module in self.named_modules():
            # Initialize square attention c_proj weights on manifold
            if isinstance(module, nn.Linear) and 'attn.c_proj' in name:
                if module.weight.shape[0] == module.weight.shape[1]:
                    with torch.no_grad():
                        W_init = initialize_on_grassmann(module.weight, a=a, b=b, r=rank)
                        module.weight.copy_(W_init)
                    print(f"Initialized {name}.weight {tuple(module.weight.shape)} on G_{{{a},{b},{rank}}}")

        print(f"Grassmann initialization complete.")

    def init_grassmann_weights_blocks(self, grassmann_configs):
        """
        Initialize block-decomposed weights on Grassmann manifold for all phases.

        This method handles:
        - Phase 0: Only attn.c_proj (square, no decomposition)
        - Phase 1-5: Various combinations of c_attn, mlp.c_fc, mlp.c_proj with block decomposition

        Args:
            grassmann_configs: list of dicts with keys:
                - name: parameter name pattern (e.g., 'attn.c_attn.weight')
                - k: number of blocks (1 for square, 3 for c_attn, 4 for MLP)
                - a, b: eigenvalues
                - r: rank
                - decomp_type: 'vertical', 'horizontal', or None for square
        """
        from manifold.block_decomposition import (
            decompose_weight_vertical,
            decompose_weight_horizontal
        )

        print(f"\n=== Initializing Grassmann Weights (Block Decomposition) ===")
        print(f"Number of weight types to initialize: {len(grassmann_configs)}")

        for cfg in grassmann_configs:
            pattern = cfg['name']
            k = cfg['k']
            a, b, r = cfg['a'], cfg['b'], cfg['r']
            decomp_type = cfg.get('decomp_type', None)

            print(f"\nInitializing {pattern}: k={k}, G_{{{a},{b},{r}}}, type={decomp_type}")

            for name, param in self.named_parameters():
                if pattern in name and 'weight' in name:
                    with torch.no_grad():
                        if decomp_type is None:
                            # Square matrix, no decomposition
                            if param.shape[0] == param.shape[1]:
                                W_init = initialize_on_grassmann(param, a=a, b=b, r=r)
                                param.copy_(W_init)
                                print(f"  Initialized {name} {tuple(param.shape)} on G_{{{a},{b},{r}}}")
                        else:
                            # Rectangular, use block decomposition
                            if decomp_type == 'vertical':
                                blocks = decompose_weight_vertical(param, k)
                            else:  # horizontal
                                blocks = decompose_weight_horizontal(param, k)

                            # Initialize each block on manifold
                            for i in range(k):
                                blocks[i] = initialize_on_grassmann(blocks[i], a=a, b=b, r=r)

                            # Recompose (no scaling at init)
                            if decomp_type == 'vertical':
                                param.data = blocks.view(k * blocks.size(1), blocks.size(2))
                            else:  # horizontal
                                param.data = blocks.transpose(0, 1).reshape(blocks.size(1), k * blocks.size(2))

                            print(f"  Initialized {name} {tuple(param.shape)} as {k} blocks of {tuple(blocks[0].shape)} on G_{{{a},{b},{r}}}")

        print(f"\nGrassmann block initialization complete.")

    def setup_component_dropout(self, grassmann_configs):
        """
        Set component-specific dropout rates based on which layers use Grassmann.

        From CIFAR-10: Grassmann layers use half the dropout of AdamW layers.
        - AdamW components: config.dropout (e.g., 0.2 for Shakespeare)
        - Grassmann components: config.grassmann_dropout or config.dropout/2 (e.g., 0.1)

        Args:
            grassmann_configs: List of dicts with Grassmann layer configs
        """
        # Determine Grassmann dropout rate
        grassmann_dropout = self.config.grassmann_dropout
        if grassmann_dropout is None:
            grassmann_dropout = self.config.dropout / 2

        adamw_dropout = self.config.dropout

        print(f"\n=== Setting up component-specific dropout ===")
        print(f"AdamW components: dropout = {adamw_dropout}")
        print(f"Grassmann components: dropout = {grassmann_dropout}")

        for block in self.transformer.h:
            # Check if c_attn uses Grassmann
            uses_grassmann_attn = any('attn.c_attn' in cfg['name'] for cfg in grassmann_configs)
            dropout_attn = grassmann_dropout if uses_grassmann_attn else adamw_dropout
            block.attn.attn_dropout.p = dropout_attn
            block.attn.resid_dropout.p = dropout_attn

            # Check if MLP uses Grassmann
            uses_grassmann_mlp = any('mlp.c_fc' in cfg['name'] or 'mlp.c_proj' in cfg['name']
                                     for cfg in grassmann_configs)
            dropout_mlp = grassmann_dropout if uses_grassmann_mlp else adamw_dropout
            block.mlp.dropout.p = dropout_mlp

        print("Component-specific dropout setup complete.")

    def setup_norm_preserving_scaling(self, grassmann_phase, grassmann_configs):
        """
        Setup frozen 0.5×I scaling layer for mlp.c_fc vertical decomposition.

        Critical insight from NORM_SCALING_ANALYSIS.md:
        - c_attn: NO scaling needed (components split immediately, only per-head norms matter)
        - mlp.c_fc: NEEDS 0.5× scaling (full 4n vector used by GELU, must scale √2·σ → σ/√2)
        - mlp.c_proj: NO scaling needed (horizontal concat naturally balances)

        Grassmann with rank=n/2 naturally produces outputs with norm ≈ σ/√2.
        This is REGULARIZATION, not a bug. Accept it as the network's natural scale.

        Args:
            grassmann_phase: Which phase we're running
            grassmann_configs: List of dicts with Grassmann layer configs
        """
        if grassmann_phase == 'phase0':
            return  # No block decomposition in phase 0

        print(f"\n=== Setting up norm-preserving scaling (Phase {grassmann_phase}) ===")
        n_embd = self.config.n_embd

        # Check if mlp.c_fc is using block decomposition
        has_c_fc = any('mlp.c_fc' in cfg['name'] for cfg in grassmann_configs)

        if has_c_fc:
            # Get device from model parameters
            device = next(self.parameters()).device

            for block in self.transformer.h:
                # Add c_fc scaling: 4 blocks stacked → √2·σ, need 0.5× → σ/√2
                scale_value = 0.5  # 1.0 / sqrt(4)
                scaling = nn.Linear(4 * n_embd, 4 * n_embd, bias=False)
                with torch.no_grad():
                    scaling.weight.copy_(scale_value * torch.eye(4 * n_embd))
                scaling.weight.requires_grad = False
                scaling = scaling.to(device)  # Move to correct device
                block.mlp.c_fc_scale = scaling
                print(f"  Added frozen {scale_value}×I scaling after mlp.c_fc (device={device})")

        print("Norm-preserving scaling setup complete.")

    def setup_skip_removal(self, grassmann_phase):
        """Remove attention skip connections for .2 phase variants."""
        if '.' in grassmann_phase and grassmann_phase.endswith('.2'):
            print(f"\n=== Removing attention skip connections (phase {grassmann_phase}) ===")
            for block in self.transformer.h:
                block.remove_attn_skip = True
            print(f"  Removed skip connections from {len(self.transformer.h)} blocks")

    def init_grassmann_block_weights(self, a=1.0, b=0.0, rank=None):
        """
        Initialize BlockDecomposedLinear weights on Grassmann manifold.

        Args:
            a, b: Grassmann eigenvalues (use 1.0, 0.0 for non-skip layers)
            rank: Projector rank (if None, uses 50% of n_embd)
        """
        if rank is None:
            rank = int(0.5 * self.config.n_embd)

        print(f"\n=== Initializing Grassmann Block Weights ===")
        print(f"Manifold: G_{{{a}, {b}, {rank}}}")

        count = 0
        for name, module in self.named_modules():
            if isinstance(module, BlockDecomposedLinear):
                for i, block in enumerate(module.blocks):
                    with torch.no_grad():
                        W_init = initialize_on_grassmann(block.weight, a=a, b=b, r=rank)
                        block.weight.copy_(W_init)
                    count += 1
                print(f"  Initialized {name} ({module.n_blocks} blocks) on G_{{{a},{b},{rank}}}")

        print(f"Total blocks initialized: {count}\n")

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def configure_optimizers_grassmann(self, weight_decay, learning_rate, betas, device_type, grass_config):
        """
        Configure hybrid optimizer: Grassmann Muon for square attention weights, AdamW for others.

        NOTE: Current implementation only applies Grassmann Muon to square attention c_proj layers.
        - c_attn (combined QKV): (3*n_embd, n_embd) - rectangular, uses AdamW
        - c_proj (attention output): (n_embd, n_embd) - square, uses Grassmann Muon
        - MLP layers: rectangular, uses AdamW

        To apply manifold optimization to Q/K/V and MLP, we would need to either:
        1. Split c_attn into separate Q, K, V projections (make them square)
        2. Make MLP layers square (remove 4× expansion)
        3. Extend to Stiefel manifold for rectangular matrices

        Args:
            weight_decay: Weight decay for AdamW
            learning_rate: Learning rate for AdamW (Grassmann uses grass_config['lr'])
            betas: AdamW betas
            device_type: 'cuda' or 'cpu'
            grass_config: Dict with Grassmann parameters:
                - 'a': First eigenvalue (0.0 for skip connections)
                - 'b': Second eigenvalue (-1.0 for skip connections)
                - 'rank': Projector rank (e.g., 240 for d_model=384)
                - 'lr': Grassmann learning rate (eta)
                - 'alpha': Dual ascent step size (default 0.01)
                - 'steps': Max dual iterations (default 10)
                - 'tol': Convergence tolerance (default 1e-6)

        Returns:
            optimizer: AdamW optimizer for non-manifold parameters
            grassmann_params: List of (name, param) tuples for Grassmann Muon updates
            grass_config: Grassmann configuration dict
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Identify square attention c_proj weights for Grassmann Muon
        grassmann_params = []
        adamw_params_decay = []
        adamw_params_nodecay = []

        for pn, p in param_dict.items():
            # Check if this is a square attention c_proj weight
            is_square_attn = (
                'attn.c_proj.weight' in pn and
                p.dim() == 2 and
                p.shape[0] == p.shape[1]
            )

            if is_square_attn:
                grassmann_params.append((pn, p))
                print(f"Grassmann Muon: {pn} {tuple(p.shape)}")
            else:
                # AdamW params: 2D+ get weight decay, <2D don't
                if p.dim() >= 2:
                    adamw_params_decay.append(p)
                else:
                    adamw_params_nodecay.append(p)

        # Create AdamW optimizer for non-manifold parameters
        optim_groups = [
            {'params': adamw_params_decay, 'weight_decay': weight_decay},
            {'params': adamw_params_nodecay, 'weight_decay': 0.0}
        ]

        num_grassmann = sum(p.numel() for _, p in grassmann_params)
        num_adamw_decay = sum(p.numel() for p in adamw_params_decay)
        num_adamw_nodecay = sum(p.numel() for p in adamw_params_nodecay)

        print(f"\n=== Hybrid Optimizer Configuration ===")
        print(f"Grassmann Muon parameters: {len(grassmann_params)} tensors, {num_grassmann:,} params")
        print(f"AdamW decay parameters: {len(adamw_params_decay)} tensors, {num_adamw_decay:,} params")
        print(f"AdamW no-decay parameters: {len(adamw_params_nodecay)} tensors, {num_adamw_nodecay:,} params")
        print(f"Grassmann config: a={grass_config['a']}, b={grass_config['b']}, "
              f"rank={grass_config['rank']}, lr={grass_config['lr']}")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer, grassmann_params, grass_config

    def configure_optimizers_grassmann_phases(self, weight_decay, learning_rate, betas, device_type, grassmann_phase='phase0'):
        """
        Configure hybrid optimizer with block decomposition support for all experimental phases.

        Phases:
        - phase0: Only attn.c_proj (square, current implementation)
        - phase1: c_attn (QKV) + attn.c_proj G(0,-1,r)
        - phase2: mlp.c_fc + attn.c_proj G(0,-1,r)
        - phase3: c_attn + mlp.c_fc + attn.c_proj G(0,-1,r)
        - phase4: c_attn + mlp.c_fc + mlp.c_proj + attn.c_proj
        - phase5: All above with G_{1,0,r} (no skip connections)

        Phase variants:
        - .5 suffix (e.g., phase0.5, phase1.5, phase2.5): Skip attn.c_proj (use AdamW instead)
        - .2 suffix (e.g., phase1.2, phase2.2, phase3.2): attn.c_proj uses G(1,0,r) + remove attn skip

        Args:
            weight_decay: Weight decay for AdamW
            learning_rate: Learning rate for AdamW
            betas: AdamW betas
            device_type: 'cuda' or 'cpu'
            grassmann_phase: E.g., 'phase1', 'phase1.5', 'phase1.2'

        Returns:
            optimizer: AdamW optimizer for non-manifold parameters
            grassmann_params: List of dicts with Grassmann configuration
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        grassmann_params = []
        adamw_params_decay = []
        adamw_params_nodecay = []

        # Get rank configuration from config (supports per-block or per-category)
        import sys
        config_module = sys.modules['__main__']

        # Helper to get per-block or scalar value
        def get_rank_or_list(name, default):
            val = getattr(config_module, name, None)
            if val is None:
                val = getattr(config_module, 'grass_rank', default)
            # If scalar, convert to list of n_layer copies
            if not isinstance(val, (list, tuple)):
                return [val] * self.config.n_layer
            return list(val)

        def get_scale_or_list(name, default):
            val = getattr(config_module, name, default)
            if not isinstance(val, (list, tuple)):
                return [val] * self.config.n_layer
            return list(val)

        # Get per-block ranks (list of n_layer values)
        ranks_c_attn = get_rank_or_list('grass_rank_c_attn', 192)
        ranks_mlp_fc = get_rank_or_list('grass_rank_mlp_fc', 192)
        ranks_attn_c_proj = get_rank_or_list('grass_rank_attn_c_proj', 192)
        ranks_mlp_c_proj = get_rank_or_list('grass_rank_mlp_c_proj', 192)

        # Get per-block scales (list of n_layer values)
        scales_c_attn = get_scale_or_list('grass_scale_c_attn', 1.0)
        scales_mlp_fc = get_scale_or_list('grass_scale_mlp_fc', 1.0)
        scales_attn_c_proj = get_scale_or_list('grass_scale_attn_c_proj', 1.0)
        scales_mlp_c_proj = get_scale_or_list('grass_scale_mlp_c_proj', 1.0)

        # Track which parameters are used by BlockDecomposedLinear (to skip in parameter loop)
        block_decomposed_params = set()

        # First pass: Handle BlockDecomposedLinear modules
        # These are used when use_grassmann=True to decompose rectangular matrices
        import re
        for module_name, module in self.named_modules():
            if isinstance(module, BlockDecomposedLinear):
                # Extract block index from module name (e.g., "transformer.h.2.attn.c_attn")
                match = re.search(r'\.h\.(\d+)\.', module_name)
                block_idx = int(match.group(1)) if match else 0

                # Determine layer type and configuration
                if 'attn.c_attn' in module_name:
                    # Q, K, V blocks - non-skip-facing
                    rank = ranks_c_attn[block_idx]
                    a, b = 1.0, 0.0  # G_{1.0, 0.0, r}
                    lr = 8e-3  # 8× boost
                    layer_type = 'c_attn'
                elif 'mlp.c_fc' in module_name:
                    # MLP expansion blocks - non-skip-facing
                    rank = ranks_mlp_fc[block_idx]
                    a, b = 1.0, 0.0  # G_{1.0, 0.0, r}
                    lr = 8e-3  # 8× boost
                    layer_type = 'mlp.c_fc'
                else:
                    continue  # Skip other layers (shouldn't happen in Phase 3)

                # Add each block's weight as a separate Grassmann parameter
                for i, block in enumerate(module.blocks):
                    param_name = f"{module_name}.blocks.{i}.weight"
                    if hasattr(block, 'weight'):
                        grassmann_params.append({
                            'name': param_name,
                            'param': block.weight,
                            'k': 1,  # Each block is already square
                            'r': rank,
                            'a': a,
                            'b': b,
                            'lr': lr,
                            'scale': 1.0,  # Scaling handled by frozen_scale in module
                            'decomp_type': None,  # Already square, no decomposition
                            'alpha': 0.01,
                            'steps': 10,
                            'tol': 1e-6
                        })
                        block_decomposed_params.add(id(block.weight))

                # Also mark gate params to skip (they'll be handled as AdamW gate params)
                if hasattr(module, 'block_gates') and module.block_gates is not None:
                    for i, gate in enumerate(module.block_gates):
                        if hasattr(gate, 'weight'):
                            block_decomposed_params.add(id(gate.weight))

        # Parse phase variants
        base_phase = grassmann_phase
        skip_attn_c_proj = False
        attn_c_proj_no_skip = False

        if '.' in grassmann_phase:
            base_phase, variant = grassmann_phase.rsplit('.', 1)
            if variant == '5':
                skip_attn_c_proj = True  # .5 = skip attn.c_proj (AdamW)
            elif variant == '2':
                attn_c_proj_no_skip = True  # .2 = G(1,0,r) + remove skip

        print(f"\n=== Phase {grassmann_phase} Grassmann Configuration ===")
        if skip_attn_c_proj:
            print("  Variant .5: attn.c_proj uses AdamW (not Grassmann)")
        if attn_c_proj_no_skip:
            print("  Variant .2: attn.c_proj uses G(1,0,r) with attn skip removed")

        print(f"\nPer-block configuration ({self.config.n_layer} blocks):")
        print(f"  c_attn ranks:       {ranks_c_attn}")
        print(f"  c_attn scales:      {[f'{s:.2f}' for s in scales_c_attn]}")
        print(f"  mlp.c_fc ranks:     {ranks_mlp_fc}")
        print(f"  mlp.c_fc scales:    {[f'{s:.2f}' for s in scales_mlp_fc]}")

        for pn, p in param_dict.items():
            # Skip parameters already handled by BlockDecomposedLinear
            if id(p) in block_decomposed_params:
                continue

            is_grassmann = False

            # Phase 0: Only attn.c_proj (backward compatibility)
            if base_phase == 'phase0':
                if 'attn.c_proj.weight' in pn and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    if not skip_attn_c_proj:  # .5 variant skips this
                        # Extract block index from parameter name (e.g., "h.2.attn.c_proj.weight")
                        import re
                        match = re.search(r'\.h\.(\d+)\.', pn)
                        block_idx = int(match.group(1)) if match else 0

                        # Phase 0.2 uses G(1,0,r), base phase 0 uses G(0,-1,r)
                        a_val = 1.0 if attn_c_proj_no_skip else 0.0
                        b_val = 0.0 if attn_c_proj_no_skip else -1.0
                        lr_val = 8e-3 if attn_c_proj_no_skip else learning_rate

                        grassmann_params.append({
                            'name': pn,
                            'param': p,
                            'k': 1,
                            'r': ranks_attn_c_proj[block_idx],
                            'a': a_val, 'b': b_val,
                            'lr': lr_val,
                            'scale': scales_attn_c_proj[block_idx],
                            'decomp_type': None,  # Square, no decomposition
                            'alpha': 0.01,
                            'steps': 10,
                            'tol': 1e-6
                        })
                        is_grassmann = True

            # Phase 1: c_attn (QKV) + attn.c_proj
            if base_phase in ['phase1', 'phase3', 'phase4', 'phase5']:
                if 'attn.c_attn.' in pn and '.weight' in pn:
                    # Extract block index from parameter name (e.g., "h.2.attn.c_attn.weight")
                    import re
                    match = re.search(r'\.h\.(\d+)\.', pn)
                    block_idx = int(match.group(1)) if match else 0

                    grassmann_params.append({
                        'name': pn,
                        'param': p,
                        'k': 3,
                        'r': ranks_c_attn[block_idx],
                        'a': 1.0, 'b': 0.0,  # No-skip, identity-centered
                        'lr': 8e-3,  # 8× boost
                        'scale': scales_c_attn[block_idx],  # Per-block operator norm scaling
                        'decomp_type': 'vertical',
                        'alpha': 0.01,
                        'steps': 10,
                        'tol': 1e-6
                    })
                    is_grassmann = True

            # Phase 2: mlp.c_fc + attn.c_proj
            if base_phase in ['phase2', 'phase3', 'phase4', 'phase5']:
                if 'mlp.c_fc.' in pn and '.weight' in pn:
                    # Extract block index
                    import re
                    match = re.search(r'\.h\.(\d+)\.', pn)
                    block_idx = int(match.group(1)) if match else 0

                    grassmann_params.append({
                        'name': pn,
                        'param': p,
                        'k': 4,
                        'r': ranks_mlp_fc[block_idx],
                        'a': 1.0, 'b': 0.0,  # No-skip
                        'lr': 8e-3,  # 8× boost
                        'scale': scales_mlp_fc[block_idx],  # Per-block operator norm scaling
                        'decomp_type': 'vertical',
                        'alpha': 0.01,
                        'steps': 10,
                        'tol': 1e-6
                    })
                    is_grassmann = True

            # Phase 4: Add mlp.c_proj (skip-compatible)
            if grassmann_phase == 'phase4':
                if 'mlp.c_proj.weight' in pn:
                    # Extract block index
                    import re
                    match = re.search(r'\.h\.(\d+)\.', pn)
                    block_idx = int(match.group(1)) if match else 0

                    grassmann_params.append({
                        'name': pn,
                        'param': p,
                        'k': 4,
                        'r': ranks_mlp_c_proj[block_idx],
                        'a': 0.0, 'b': -1.0,  # Skip-compatible
                        'lr': learning_rate,  # No boost (1e-3)
                        'scale': 0.5,  # Gradient dampening
                        'decomp_type': 'horizontal',
                        'alpha': 0.01,
                        'steps': 10,
                        'tol': 1e-6
                    })
                    is_grassmann = True

            # Phase 5: All G_{1,0,r} (no skip)
            if grassmann_phase == 'phase5':
                # attn.c_proj now uses G_{1,0,r} instead of G_{0,-1,r}
                if 'attn.c_proj.weight' in pn and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    # Extract block index
                    import re
                    match = re.search(r'\.h\.(\d+)\.', pn)
                    block_idx = int(match.group(1)) if match else 0

                    grassmann_params.append({
                        'name': pn,
                        'param': p,
                        'k': 1,
                        'r': ranks_attn_c_proj[block_idx],
                        'a': 1.0, 'b': 0.0,  # No-skip (changed from 0,-1)
                        'lr': 8e-3,  # 8× boost (changed from 1e-3)
                        'scale': 1.0,
                        'decomp_type': None,
                        'alpha': 0.01,
                        'steps': 10,
                        'tol': 1e-6
                    })
                    is_grassmann = True

                # mlp.c_proj now uses G_{1,0,r} with 8× LR
                if 'mlp.c_proj.weight' in pn:
                    # Extract block index
                    import re
                    match = re.search(r'\.h\.(\d+)\.', pn)
                    block_idx = int(match.group(1)) if match else 0

                    grassmann_params.append({
                        'name': pn,
                        'param': p,
                        'k': 4,
                        'r': ranks_mlp_c_proj[block_idx],
                        'a': 1.0, 'b': 0.0,  # No-skip (changed from 0,-1)
                        'lr': 8e-3,  # 8× boost (changed from 1e-3)
                        'scale': 1.0,  # No dampening (changed from 0.5)
                        'decomp_type': 'horizontal',
                        'alpha': 0.01,
                        'steps': 10,
                        'tol': 1e-6
                    })
                    is_grassmann = True

            # attn.c_proj for phases 1-2 ONLY (phase3 uses AdamW for skip-facing layers)
            if base_phase in ['phase1', 'phase2']:
                if 'attn.c_proj.weight' in pn and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    if not skip_attn_c_proj:  # .5 variant skips attn.c_proj
                        # Extract block index
                        import re
                        match = re.search(r'\.h\.(\d+)\.', pn)
                        block_idx = int(match.group(1)) if match else 0

                        # .2 variant uses G(1,0,r) + 8× LR, base uses G(0,-1,r) + 1× LR
                        a_val = 1.0 if attn_c_proj_no_skip else 0.0
                        b_val = 0.0 if attn_c_proj_no_skip else -1.0
                        lr_val = 8e-3 if attn_c_proj_no_skip else learning_rate

                        grassmann_params.append({
                            'name': pn,
                            'param': p,
                            'k': 1,
                            'r': ranks_attn_c_proj[block_idx],
                            'a': a_val, 'b': b_val,
                            'lr': lr_val,
                            'scale': 1.0,
                            'decomp_type': None,
                            'alpha': 0.01,
                            'steps': 10,
                            'tol': 1e-6
                        })
                        is_grassmann = True

            # AdamW for non-Grassmann params
            if not is_grassmann:
                if p.dim() >= 2:
                    adamw_params_decay.append(p)
                else:
                    adamw_params_nodecay.append(p)

        # Split AdamW params into groups with different learning rates
        # Gates need higher LR for fast adaptation, embeddings need lower LR
        gate_params = []
        proj_params = []
        embed_params = []
        other_decay_params = []

        # First, collect block_gates from BlockDecomposedLinear modules
        for module_name, module in self.named_modules():
            if isinstance(module, BlockDecomposedLinear):
                if hasattr(module, 'block_gates') and module.block_gates is not None:
                    for i, gate in enumerate(module.block_gates):
                        if hasattr(gate, 'weight'):
                            gate_params.append(gate.weight)

        for pn, p in param_dict.items():
            if id(p) in block_decomposed_params:
                continue  # Skip Grassmann params and block_gates (already collected)

            # Classify non-Grassmann 2D params
            if p.dim() >= 2:
                if 'gate' in pn and '.weight' in pn:
                    # Gating networks (head_gates, block_gates)
                    gate_params.append(p)
                elif 'c_proj.weight' in pn:
                    # Projection layers (attn.c_proj, mlp.c_proj)
                    proj_params.append(p)
                elif 'wte.weight' in pn or 'wpe.weight' in pn:
                    # Embeddings (token + position)
                    embed_params.append(p)
                else:
                    # Other 2D params (shouldn't be many)
                    other_decay_params.append(p)

        # Determine gate learning rate
        # If gate_lr is provided in config, use it; otherwise use 2× base LR
        gate_lr = getattr(self.config, 'gate_lr', None)
        if gate_lr is None:
            gate_lr = learning_rate * 2.0  # Default: 2× base LR

        # Determine embedding learning rate
        embed_lr = getattr(self.config, 'embed_lr', None)
        if embed_lr is None:
            embed_lr = learning_rate * 0.5  # Default: 0.5× base LR

        # Create optimizer groups with different LRs
        optim_groups = [
            {'params': gate_params, 'lr': gate_lr, 'weight_decay': weight_decay},
            {'params': proj_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': embed_params, 'lr': embed_lr, 'weight_decay': 0.0},  # No decay for embeddings
            {'params': other_decay_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': adamw_params_nodecay, 'lr': learning_rate, 'weight_decay': 0.0}
        ]

        num_grassmann = sum(p['param'].numel() for p in grassmann_params)
        num_gate = sum(p.numel() for p in gate_params)
        num_proj = sum(p.numel() for p in proj_params)
        num_embed = sum(p.numel() for p in embed_params)
        num_other_decay = sum(p.numel() for p in other_decay_params)
        num_adamw_nodecay = sum(p.numel() for p in adamw_params_nodecay)

        print(f"Grassmann Muon parameters: {len(grassmann_params)} weight matrices, {num_grassmann:,} params")
        print(f"AdamW optimizer groups:")
        print(f"  Gates:      {len(gate_params)} tensors, {num_gate:,} params, lr={gate_lr:.2e}, wd={weight_decay}")
        print(f"  Projections: {len(proj_params)} tensors, {num_proj:,} params, lr={learning_rate:.2e}, wd={weight_decay}")
        print(f"  Embeddings: {len(embed_params)} tensors, {num_embed:,} params, lr={embed_lr:.2e}, wd=0.0")
        if num_other_decay > 0:
            print(f"  Other decay: {len(other_decay_params)} tensors, {num_other_decay:,} params, lr={learning_rate:.2e}, wd={weight_decay}")
        print(f"  No decay:   {len(adamw_params_nodecay)} tensors, {num_adamw_nodecay:,} params, lr={learning_rate:.2e}, wd=0.0")

        for cfg in grassmann_params:
            print(f"  {cfg['name']}: k={cfg['k']}, G_{{{cfg['a']},{cfg['b']},{cfg['r']}}}, "
                  f"lr={cfg['lr']}, scale={cfg['scale']}, type={cfg['decomp_type']}")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer, grassmann_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
