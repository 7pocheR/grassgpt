import math
import torch
from .msign import msign
from .grassmann_ops import (
    tangent_operator_LX,
    retract_to_grassmann,
    initialize_on_grassmann
)


@torch.no_grad()
def grassmann_muon(X, G, eta=0.1, a=1.0, b=0.0, r=None,
                   alpha=0.01, steps=100, tol=1e-6):
    """
    Grassmann Muon update on the two-eigenvalue Grassmannian G_{a,b,r}.

    The manifold is: G_{a,b,r} = {bI + cP : P projector, rank(P) = r}
    where c = a - b ≠ 0.

    Algorithm:
    1. Symmetrize gradient
    2. Dual ascent to find tangent direction (enforce L_X(A) = 0)
    3. Take primal step
    4. Retract to manifold via sign-based projection

    Args:
        X: Weight matrix on manifold (n×n, symmetric)
        G: Gradient matrix (n×n)
        eta: Step size (learning rate)
        a, b: Two eigenvalues defining the manifold (a ≠ b)
        r: Target rank of projector P (if None, auto-detect from X)
        alpha: Dual ascent step size
        steps: Max dual ascent iterations
        tol: Convergence tolerance for tangent constraint

    Returns:
        X_new: Updated weight on the manifold (n×n, symmetric)
    """
    n = X.size(0)
    device = X.device
    dtype = X.dtype

    # Compute c = a - b
    c = a - b
    c_squared = c * c

    # Auto-detect rank if not provided
    if r is None:
        # Extract projector: P = (X - bI) / c
        I = torch.eye(n, device=device, dtype=dtype)
        P = (X - b * I) / c
        # Rank is approximately trace(P)
        r = int(torch.round(P.trace()).item())
        r = max(1, min(r, n - 1))  # Clamp to [1, n-1]

    # Symmetrize gradient (ensure G is symmetric)
    G = 0.5 * (G + G.T)

    # Initialize dual variable (symmetric, zero initialization)
    Lambda = torch.zeros_like(X)

    # Dual ascent loop (enforce tangency constraint)
    for step in range(steps):
        # Compute M = G + L_X(Lambda)
        LX_Lambda = tangent_operator_LX(X, Lambda, c_squared)
        M = G + LX_Lambda

        # Compute sign(M) - use Polar Express from existing msign
        S = msign(M)

        # Ensure S is symmetric (should be, but enforce numerically)
        S = 0.5 * (S + S.T)

        # Compute subgradient: H = -eta * L_X(S)
        LX_S = tangent_operator_LX(X, S, c_squared)
        H = -eta * LX_S

        # Check stopping criterion: ||L_X(S)||_F / sqrt(n²) < tol
        norm_LX_S = torch.norm(LX_S) / math.sqrt(n * n)
        if norm_LX_S < tol:
            break

        # Update dual variable with decaying step size
        Lambda = Lambda + alpha * (1 - step / steps) * H

    # Primal step: A_opt = -eta * S
    A_opt = -eta * S

    # Apply step and symmetrize
    X_plus = X + A_opt
    X_plus = 0.5 * (X_plus + X_plus.T)

    # Retract to manifold G_{a,b,r} (use more iterations for accuracy)
    X_new = retract_to_grassmann(X_plus, a, b, r, bisect_iters=30, ns_iters=10)

    return X_new


@torch.no_grad()
def grassmann_muon_update(W, G, eta=0.1, a=1.0, b=0.0, r=None,
                          alpha=0.01, steps=100, tol=1e-6):
    """
    Wrapper for grassmann_muon that handles weight matrices.

    For now, only supports square 2D tensors (n×n).
    First/last layers of networks should use different optimizers.

    Args:
        W: Weight tensor (must be 2D and square)
        G: Gradient tensor (same shape as W)
        eta: Step size
        a, b: Two eigenvalues for Grassmannian
        r: Target rank (if None, auto-detect or use n//2)
        alpha: Dual ascent step size
        steps: Max dual ascent iterations
        tol: Convergence tolerance

    Returns:
        Updated weight tensor (same shape as input)
    """
    if W.ndim != 2:
        raise ValueError(f"grassmann_muon_update only supports 2D tensors, got {W.ndim}D")

    if W.shape[0] != W.shape[1]:
        raise ValueError(
            f"grassmann_muon_update requires square matrices, "
            f"got shape {W.shape}. Use AdamW or Stiefel for rectangular layers."
        )

    # Apply Grassmann Muon
    X_new = grassmann_muon(W, G, eta, a, b, r, alpha, steps, tol)

    return X_new
