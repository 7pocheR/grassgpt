"""
Block decomposition for applying Grassmann Muon to rectangular weight matrices.

Decomposes (kN, N) and (N, kN) matrices into k square (N, N) blocks,
each constrained to Grassmann manifold G_{a,b,r}.

Two approaches:
1. Functional API: decompose/compose functions for in-place weight updates
2. Module API: BlockDecomposedLinear nn.Module for cleaner integration
"""

import torch
import torch.nn as nn
import math


def decompose_weight_vertical(W, k):
    """
    Split (kN, N) weight into k blocks of (N, N).

    Used for vertically stacked matrices like:
    - c_attn: (1152, 384) → 3 blocks of (384, 384)
    - mlp.c_fc: (1536, 384) → 4 blocks of (384, 384)

    Args:
        W: (kN, N) tensor
        k: number of blocks
    Returns:
        blocks: (k, N, N) tensor
    """
    kN, N = W.shape
    assert kN == k * N, f"Shape mismatch: {kN} != {k} * {N}"
    return W.view(k, N, N)


def decompose_weight_horizontal(W, k):
    """
    Split (N, kN) weight into k blocks of (N, N).

    Used for horizontally concatenated matrices like:
    - mlp.c_proj: (384, 1536) → 4 blocks of (384, 384)

    Args:
        W: (N, kN) tensor
        k: number of blocks
    Returns:
        blocks: (k, N, N) tensor
    """
    N, kN = W.shape
    assert kN == k * N, f"Shape mismatch: {kN} != {k} * {N}"
    # Reshape to (N, k, N) then transpose to (k, N, N)
    return W.view(N, k, N).transpose(0, 1).contiguous()


def compose_weight_vertical(blocks, scale=1.0):
    """
    Combine k blocks into (kN, N) weight.

    Args:
        blocks: (k, N, N) tensor
        scale: DEPRECATED - kept for API compatibility, always use 1.0
    Returns:
        W: (kN, N) tensor
    """
    k, N, _ = blocks.shape
    return blocks.view(k * N, N)


def compose_weight_horizontal(blocks, scale=1.0):
    """
    Combine k blocks into (N, kN) weight.

    Args:
        blocks: (k, N, N) tensor
        scale: DEPRECATED - kept for API compatibility, always use 1.0
    Returns:
        W: (N, kN) tensor
    """
    k, N, _ = blocks.shape
    # Transpose (k, N, N) to (N, k, N) then reshape to (N, kN)
    return blocks.transpose(0, 1).reshape(N, k * N)


@torch.no_grad()
def apply_grassmann_blocks(W, G, eta, a, b, r, k, decomp_type, scale,
                           alpha=0.01, steps=10, tol=1e-6):
    """
    Apply Grassmann Muon update to block-decomposed weight.

    This function:
    1. Decomposes W and G into k square blocks
    2. Applies grassmann_muon_update to each block independently
    3. Recomposes blocks back into original shape (NO scaling here)

    Note: Norm preservation handled by frozen (1/√k)×I layers in forward pass:
    - Vertical decomposition (c_attn, c_fc): Frozen scaling after layer
    - Horizontal decomposition (c_proj): No scaling (natural cancellation)

    Args:
        W: weight tensor (kN, N) or (N, kN)
        G: gradient tensor (same shape as W)
        eta: learning rate
        a, b: Grassmann eigenvalues
        r: rank
        k: number of blocks
        decomp_type: 'vertical' or 'horizontal'
        scale: DEPRECATED - kept for API compatibility, ignored
        alpha: Dual ascent step size (default 0.01)
        steps: Max dual iterations (default 10)
        tol: Convergence tolerance (default 1e-6)
    Returns:
        W_new: updated weight tensor
    """
    from .grassmann_muon import grassmann_muon_update

    # Decompose
    if decomp_type == 'vertical':
        blocks_W = decompose_weight_vertical(W, k)
        blocks_G = decompose_weight_vertical(G, k)
    elif decomp_type == 'horizontal':
        blocks_W = decompose_weight_horizontal(W, k)
        blocks_G = decompose_weight_horizontal(G, k)
    else:
        raise ValueError(f"Unknown decomp_type: {decomp_type}")

    # Update each block independently on the manifold
    # Note: 'scale' parameter IGNORED - norm preservation via frozen layers in forward pass
    blocks_W_new = torch.zeros_like(blocks_W)
    for i in range(k):
        blocks_W_new[i] = grassmann_muon_update(
            blocks_W[i],
            blocks_G[i],
            eta,
            a, b, r,
            alpha=alpha,
            steps=steps,
            tol=tol
        )

    # Recompose (no scaling - pure block reassembly)
    if decomp_type == 'vertical':
        return compose_weight_vertical(blocks_W_new, 1.0)
    else:  # horizontal
        return compose_weight_horizontal(blocks_W_new, 1.0)


class BlockDecomposedLinear(nn.Module):
    """
    Linear layer decomposed into multiple square blocks for Grassmann optimization.

    Replaces nn.Linear for rectangular (kN, N) weight matrices by decomposing
    into k independent (N, N) blocks, each constrained to Grassmann manifold.

    Forward: y = [W_1@x; W_2@x; ...; W_k@x]  (vertical stacking)

    Used for:
    - c_attn: (3n, n) → 3 blocks (Q, K, V)
    - mlp.c_fc: (4n, n) → 4 blocks with frozen 0.5× scaling
    """

    def __init__(self, in_features, out_features, n_blocks,
                 frozen_scale=None, use_block_gating=False, bias=False):
        """
        Args:
            in_features (int): Input dimension N
            out_features (int): Output dimension kN (must equal n_blocks × in_features)
            n_blocks (int): Number of blocks k
            frozen_scale (float, optional): Frozen scalar multiplier applied to output
                                           Use 0.5 for mlp.c_fc
            use_block_gating (bool): Whether to add per-block gating (for c_fc)
            bias (bool): Whether to include bias (should be False for Grassmann)
        """
        super().__init__()

        # Validate dimensions
        assert out_features == n_blocks * in_features, \
            f"Output features ({out_features}) must equal n_blocks ({n_blocks}) × in_features ({in_features})"

        assert not bias, "Block decomposition for Grassmann should not use bias"

        self.in_features = in_features
        self.out_features = out_features
        self.n_blocks = n_blocks

        # Create k independent (N, N) blocks
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, in_features, bias=False)
            for _ in range(n_blocks)
        ])

        # Block-level gating (for c_fc)
        # Each gate: Linear(in_features → 1) projects input to scalar per block
        if use_block_gating:
            self.block_gates = nn.ModuleList([
                nn.Linear(in_features, 1, bias=False)
                for _ in range(n_blocks)
            ])
        else:
            self.block_gates = None

        # Frozen scaling (non-trainable)
        # For mlp.c_fc: 0.5× to normalize √4 → 1
        if frozen_scale is not None:
            self.register_buffer('frozen_scale',
                               torch.tensor(frozen_scale, dtype=torch.float32))
        else:
            self.frozen_scale = None

    def forward(self, x):
        """
        Forward pass through block-decomposed layer.

        Args:
            x: (B, T, in_features)

        Returns:
            y: (B, T, out_features) = (B, T, n_blocks × in_features)
        """
        # Apply each block independently with optional gating
        block_outputs = []
        for i, block in enumerate(self.blocks):
            out = block(x)  # (B, T, in_features), Grassmann projection ||W||_op=1

            # Apply block-level gate if enabled
            if self.block_gates is not None:
                gate = torch.sigmoid(self.block_gates[i](x))  # (B, T, 1)
                out = out * gate  # Input-dependent scaling

            block_outputs.append(out)

        # Concatenate along feature dimension
        y = torch.cat(block_outputs, dim=-1)  # (B, T, k × in_features)

        # Apply frozen scaling if configured
        if self.frozen_scale is not None:
            y = y * self.frozen_scale

        return y

    def get_grassmann_params(self):
        """
        Get list of block weight parameters for Grassmann optimizer.

        Returns:
            List[torch.Tensor]: Weight tensors, each (in_features, in_features)
        """
        return [block.weight for block in self.blocks]

    def extra_repr(self):
        """String representation for debugging."""
        s = f'{self.in_features}, {self.out_features}, n_blocks={self.n_blocks}'
        if self.frozen_scale is not None:
            s += f', frozen_scale={self.frozen_scale.item():.2f}'
        if self.block_gates is not None:
            s += f', block_gating=True'
        return s


class FullBlockDecomposedLinear(nn.Module):
    """
    Full block decomposition: Decomposes (out_features, in_features) into grid of square blocks.

    For 1024×1024 matrix:
      - Splits into 16×16 grid = 256 blocks of 64×64
      - Each block is independently Grassmann-constrained
      - Enables true head independence in multi-head attention

    Forward computation:
      For each output head h:
        head_h = sum_{i=1}^{n_blocks_in} W_{h,i} @ x_i

    Used for c_attn in alternative architecture (768 blocks total for Q/K/V).
    """

    def __init__(self, in_features, out_features, block_size=64,
                 use_block_gating=False, gating_mode='per_head', bias=False):
        """
        Args:
            in_features (int): Input dimension (e.g., 1024)
            out_features (int): Output dimension (e.g., 1024)
            block_size (int): Size of each square block (default: 64)
            use_block_gating (bool): Whether to add gating
            gating_mode (str): 'per_head' (16 gates), 'per_block' (256 gates),
                              or 'per_block_segment' (256 gates, efficient)
            bias (bool): Whether to include bias (should be False for Grassmann)
        """
        super().__init__()

        assert in_features % block_size == 0, \
            f"in_features ({in_features}) must be divisible by block_size ({block_size})"
        assert out_features % block_size == 0, \
            f"out_features ({out_features}) must be divisible by block_size ({block_size})"
        assert not bias, "Full block decomposition for Grassmann should not use bias"

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.n_blocks_in = in_features // block_size      # 16 for 1024/64
        self.n_blocks_out = out_features // block_size    # 16 for 1024/64

        # Create grid of blocks: n_blocks_out × n_blocks_in (e.g., 16×16 = 256 blocks)
        # blocks[h][i] corresponds to W_{h,i} in mathematical notation
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(block_size, block_size, bias=False)
                for _ in range(self.n_blocks_in)
            ]) for _ in range(self.n_blocks_out)
        ])

        # Gating
        self.gating_mode = gating_mode if use_block_gating else None
        if use_block_gating:
            if gating_mode == 'per_head':
                # One gate per output head: 16 gates for 1024d
                # gate_h = sigmoid(v_h^T x)
                self.gates = nn.Linear(in_features, self.n_blocks_out, bias=False)
            elif gating_mode == 'per_block':
                # One gate per block: 256 gates for 16×16 grid
                # gate_{h,i} = sigmoid(v_{h,i}^T x)
                self.gates = nn.Linear(in_features,
                    self.n_blocks_out * self.n_blocks_in, bias=False)
            elif gating_mode == 'per_block_segment':
                # One gate per block, but only sees its input segment: 256 gates
                # gate_{h,i} = sigmoid(v_{h,i}^T x_i)  [most efficient]
                self.gates = nn.ModuleList([
                    nn.ModuleList([
                        nn.Linear(block_size, 1, bias=False)
                        for _ in range(self.n_blocks_in)
                    ]) for _ in range(self.n_blocks_out)
                ])
            else:
                raise ValueError(f"Unknown gating_mode: {gating_mode}")
        else:
            self.gates = None

    def forward(self, x):
        """
        Forward pass through full block decomposition.

        Args:
            x: (B, T, in_features)  [e.g., (B, T, 1024)]

        Returns:
            y: (B, T, out_features)  [e.g., (B, T, 1024)]
        """
        B, T, _ = x.size()

        # Split input into n_blocks_in segments of block_size
        # x_segments: (B, T, 16, 64) for 1024d with block_size=64
        x_segments = x.view(B, T, self.n_blocks_in, self.block_size)

        # Compute output for each head
        outputs = []
        for h in range(self.n_blocks_out):
            # Aggregate contributions from all input segments
            head_output = torch.zeros(B, T, self.block_size,
                                     device=x.device, dtype=x.dtype)

            for i in range(self.n_blocks_in):
                # Block projection: W_{h,i} @ x_i
                # x_segments[:, :, i]: (B, T, 64)
                contribution = self.blocks[h][i](x_segments[:, :, i])  # (B, T, 64)

                # Apply gating if enabled
                if self.gates is not None:
                    if self.gating_mode == 'per_head':
                        # Compute all head gates once (cache if needed)
                        if i == 0:  # Only compute once per head
                            gate_all = torch.sigmoid(self.gates(x))  # (B, T, n_blocks_out)
                        gate = gate_all[:, :, h:h+1]  # (B, T, 1)
                    elif self.gating_mode == 'per_block':
                        # Compute all block gates from full input
                        if h == 0 and i == 0:  # Only compute once
                            self.gate_cache = torch.sigmoid(self.gates(x))  # (B, T, 256)
                        gate_idx = h * self.n_blocks_in + i
                        gate = self.gate_cache[:, :, gate_idx:gate_idx+1]  # (B, T, 1)
                    elif self.gating_mode == 'per_block_segment':
                        # Compute gate from input segment only (most efficient)
                        gate = torch.sigmoid(
                            self.gates[h][i](x_segments[:, :, i])
                        )  # (B, T, 1)

                    contribution = contribution * gate

                head_output = head_output + contribution

            outputs.append(head_output)

        # Concatenate all heads: [(B, T, 64)] × 16 → (B, T, 1024)
        output = torch.cat(outputs, dim=-1)
        return output

    def get_grassmann_params(self):
        """
        Get list of all block weight parameters for Grassmann optimizer.

        Returns:
            List[torch.Tensor]: Weight tensors, each (block_size, block_size)
                               Total: n_blocks_out × n_blocks_in tensors
        """
        params = []
        for h in range(self.n_blocks_out):
            for i in range(self.n_blocks_in):
                params.append(self.blocks[h][i].weight)
        return params

    def extra_repr(self):
        """String representation for debugging."""
        total_blocks = self.n_blocks_out * self.n_blocks_in
        s = (f'{self.in_features}, {self.out_features}, '
             f'block_size={self.block_size}, '
             f'blocks={self.n_blocks_out}×{self.n_blocks_in}={total_blocks}')
        if self.gating_mode is not None:
            s += f', gating={self.gating_mode}'
        return s


def test_block_decomposition():
    """
    Unit tests for block decomposition functions.

    Tests:
    1. Vertical decompose/compose roundtrip
    2. Horizontal decompose/compose roundtrip
    3. Shape correctness
    """
    print("=== Testing Block Decomposition ===")

    # Test 1: Vertical stacking (c_attn example)
    print("\nTest 1: Vertical stacking (1152, 384) → 3 blocks")
    W_vert = torch.randn(1152, 384)
    blocks = decompose_weight_vertical(W_vert, k=3)
    print(f"  Original shape: {W_vert.shape}")
    print(f"  Blocks shape: {blocks.shape}")
    assert blocks.shape == (3, 384, 384), f"Expected (3, 384, 384), got {blocks.shape}"

    W_recon = compose_weight_vertical(blocks, scale=1.0)
    print(f"  Reconstructed shape: {W_recon.shape}")
    assert W_recon.shape == W_vert.shape, f"Shape mismatch: {W_recon.shape} != {W_vert.shape}"
    assert torch.allclose(W_vert, W_recon), "Vertical roundtrip failed!"
    print("  ✓ Vertical roundtrip passed")

    # Test 2: Horizontal concatenation (mlp.c_proj example)
    print("\nTest 2: Horizontal concat (384, 1536) → 4 blocks")
    W_horiz = torch.randn(384, 1536)
    blocks = decompose_weight_horizontal(W_horiz, k=4)
    print(f"  Original shape: {W_horiz.shape}")
    print(f"  Blocks shape: {blocks.shape}")
    assert blocks.shape == (4, 384, 384), f"Expected (4, 384, 384), got {blocks.shape}"

    W_recon = compose_weight_horizontal(blocks, scale=1.0)
    print(f"  Reconstructed shape: {W_recon.shape}")
    assert W_recon.shape == W_horiz.shape, f"Shape mismatch: {W_recon.shape} != {W_horiz.shape}"
    assert torch.allclose(W_horiz, W_recon), "Horizontal roundtrip failed!"
    print("  ✓ Horizontal roundtrip passed")

    # Test 3: Scaling verification
    print("\nTest 3: Scaling verification")
    W_scaled = compose_weight_vertical(blocks, scale=0.5)
    W_expected = blocks.view(4 * 384, 384) * 0.5
    assert torch.allclose(W_scaled, W_expected), "Scaling failed!"
    print("  ✓ Scaling works correctly")

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_block_decomposition()
