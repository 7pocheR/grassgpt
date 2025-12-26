"""
Block decomposition for applying Grassmann Muon to rectangular weight matrices.

Decomposes (kN, N) and (N, kN) matrices into k square (N, N) blocks,
each constrained to Grassmann manifold G_{a,b,r}.
"""

import torch
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
