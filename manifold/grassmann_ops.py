import torch

@torch.no_grad()
def tangent_operator_LX(X, Z, c_squared):
    """
    Compute the tangent space operator L_X(Z) = [X, [X, Z]] - c²Z
    where [A, B] = AB - BA is the commutator.

    This operator characterizes the tangent space: A ∈ T_X G ⟺ L_X(A) = 0

    Args:
        X: Point on manifold (n×n symmetric)
        Z: Input matrix (n×n)
        c_squared: (a-b)², where a, b are the two eigenvalues

    Returns:
        L_X(Z): Result (n×n)

    Cost: 4 matrix multiplications
    """
    # First commutator: [X, Z] = XZ - ZX
    XZ = X @ Z
    ZX = Z @ X
    comm1 = XZ - ZX

    # Second commutator: [X, [X, Z]] = X·comm1 - comm1·X
    X_comm1 = X @ comm1
    comm1_X = comm1 @ X
    comm2 = X_comm1 - comm1_X

    # L_X(Z) = [X, [X, Z]] - c²Z
    return comm2 - c_squared * Z


@torch.no_grad()
def ns_sign_symmetric(A, max_iters=10, tol=1e-6):
    """
    Newton-Schulz iteration for matrix sign function (symmetric version).
    More numerically stable than Polar Express for symmetric matrices.

    Iteration: X_{k+1} = 0.5 * X_k * (3I - X_k²)

    Args:
        A: Symmetric matrix (n×n)
        max_iters: Maximum NS iterations
        tol: Relative change tolerance for early stopping

    Returns:
        sign(A): Matrix sign function result (n×n)
    """
    n = A.size(0)
    device = A.device
    dtype = A.dtype

    # Scale by Frobenius norm (upper bound on spectral norm)
    mu = A.norm() + 1e-8
    X = A / mu

    I = torch.eye(n, device=device, dtype=dtype)

    for i in range(max_iters):
        X_prev = X

        # X_{k+1} = 0.5 * X_k * (3I - X_k²)
        X2 = X @ X
        X = 0.5 * X @ (3 * I - X2)

        # Symmetrize to maintain symmetry under numerical errors
        X = 0.5 * (X + X.T)

        # Check convergence
        rel_change = (X - X_prev).norm() / (X.norm() + 1e-8)
        if rel_change < tol:
            break

    return X


@torch.no_grad()
def power_iteration_bounds(S, num_iters=5):
    """
    Estimate λ_max and λ_min of symmetric matrix using power iteration.

    Args:
        S: Symmetric matrix (n×n)
        num_iters: Number of power iterations

    Returns:
        lambda_max: Approximate largest eigenvalue
        lambda_min: Approximate smallest eigenvalue
    """
    n = S.size(0)
    device = S.device
    dtype = S.dtype

    # Random starting vector for λ_max
    v = torch.randn(n, 1, device=device, dtype=dtype)
    v = v / v.norm()

    # Power iteration for λ_max
    for _ in range(num_iters):
        v = S @ v
        v = v / (v.norm() + 1e-12)
    lambda_max = (v.T @ S @ v).item()

    # Random starting vector for λ_min (use -S)
    v = torch.randn(n, 1, device=device, dtype=dtype)
    v = v / v.norm()

    # Power iteration for λ_min
    for _ in range(num_iters):
        v = (-S) @ v
        v = v / (v.norm() + 1e-12)
    lambda_min = -(v.T @ (-S) @ v).item()

    return lambda_max, lambda_min


@torch.no_grad()
def find_threshold_bisection(S, target_rank, a, b, bisect_iters=20, ns_iters=5):
    """
    Find threshold τ ∈ (λ_r, λ_{r+1}) via bisection.

    Uses the rank-counting property:
    #{λ_i > τ} = 0.5 * (n + trace(sign(S - τI)))

    Args:
        S: Symmetric matrix to threshold (n×n)
        target_rank: Desired rank r of the projector
        a, b: Two eigenvalues defining the manifold (a ≠ b)
        bisect_iters: Number of bisection iterations
        ns_iters: NS iterations per sign computation

    Returns:
        tau: Threshold value
        Y: sign(S - τI) at the final τ
    """
    n = S.size(0)
    device = S.device
    dtype = S.dtype

    sigma = 1.0 if a > b else -1.0  # sign(a - b)

    # Get eigenvalue bounds via power iteration
    lambda_max, lambda_min = power_iteration_bounds(S, num_iters=5)

    # Initialize bisection bounds (add margin for numerical safety)
    lower = lambda_min - 0.1 * abs(lambda_min + 1e-6)
    upper = lambda_max + 0.1 * abs(lambda_max + 1e-6)

    Y = None
    tau = None

    I = torch.eye(n, device=device, dtype=dtype)

    for iter_idx in range(bisect_iters):
        tau = 0.5 * (lower + upper)

        # Compute sign(S - τI)
        S_shifted = S - tau * I
        Y = ns_sign_symmetric(S_shifted, max_iters=ns_iters)

        # Count eigenvalues > τ using: k(τ) = 0.5 * (n + trace(Y))
        k_tau = 0.5 * (n + Y.trace().item())

        # Bisection update
        if a > b:
            # Want top-r eigenvalues = a
            if k_tau > target_rank:
                lower = tau  # Too many above → raise threshold
            else:
                upper = tau  # Too few above → lower threshold
        else:
            # Want bottom-r eigenvalues = a (flip logic)
            if k_tau < n - target_rank:
                upper = tau
            else:
                lower = tau

        # Early termination if rank is close enough
        if abs(k_tau - target_rank) < 0.5:
            break

    return tau, Y


@torch.no_grad()
def retract_to_grassmann_evd(X_plus, a, b, r):
    """
    Project X_plus back to G_{a,b,r} using EVD (more stable for b=0 case).

    Algorithm:
    1. Symmetrize: S = 0.5(X_plus + X_plus^T)
    2. Compute eigendecomposition: S = Q Λ Q^T
    3. Select top-r eigenvectors to form rank-r projector
    4. Return: X_new = bI + c*P_r

    Args:
        X_plus: Matrix after primal step (n×n)
        a, b: Two eigenvalues defining the manifold
        r: Target rank of projector

    Returns:
        X_new: Projected matrix on the manifold (n×n)
    """
    n = X_plus.size(0)
    device = X_plus.device
    dtype = X_plus.dtype

    c = a - b

    # Symmetrize
    S = 0.5 * (X_plus + X_plus.T)

    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(S)

    # Sort in descending order if a > b, ascending if a < b
    if a > b:
        # Want top-r eigenvalues
        indices = torch.argsort(eigenvalues, descending=True)
    else:
        # Want bottom-r eigenvalues
        indices = torch.argsort(eigenvalues, descending=False)

    # Select top/bottom r eigenvectors
    Q_r = eigenvectors[:, indices[:r]]

    # Form rank-r projector: P_r = Q_r Q_r^T
    P_r = Q_r @ Q_r.T

    # Project back to manifold: X_new = bI + cP_r
    I = torch.eye(n, device=device, dtype=dtype)
    X_new = b * I + c * P_r

    # Symmetrize to ensure numerical symmetry
    X_new = 0.5 * (X_new + X_new.T)

    return X_new


@torch.no_grad()
def retract_to_grassmann(X_plus, a, b, r, bisect_iters=20, ns_iters=5, use_evd=None):
    """
    Project X_plus back to the Grassmannian G_{a,b,r} via sign-based retraction.

    Algorithm:
    1. Symmetrize: S = 0.5(X_plus + X_plus^T)
    2. Find threshold τ via bisection
    3. Compute Y = sign(S - τI)
    4. Form projector: P* = 0.5(I + σY), σ = sign(a-b)
    5. Return: X_new = bI + cP*

    This is the Frobenius-nearest point in G_{a,b,r}.

    Args:
        X_plus: Matrix after primal step (n×n)
        a, b: Two eigenvalues defining the manifold
        r: Target rank of projector
        bisect_iters: Number of bisection iterations for threshold
        ns_iters: NS iterations per sign computation
        use_evd: If True, use EVD method; if False, use bisection; if None, auto-select

    Returns:
        X_new: Projected matrix on the manifold (n×n)
    """
    # Auto-select method: use EVD for projector manifold (b=0)
    if use_evd is None:
        use_evd = (abs(b) < 1e-6)

    if use_evd:
        return retract_to_grassmann_evd(X_plus, a, b, r)

    n = X_plus.size(0)
    device = X_plus.device
    dtype = X_plus.dtype

    c = a - b
    sigma = 1.0 if a > b else -1.0

    # Symmetrize
    S = 0.5 * (X_plus + X_plus.T)

    # Find threshold and compute sign(S - τI)
    tau, Y = find_threshold_bisection(S, r, a, b, bisect_iters, ns_iters)

    # Form rank-r projector: P* = 0.5(I + σY)
    I = torch.eye(n, device=device, dtype=dtype)
    P_star = 0.5 * (I + sigma * Y)

    # Project back to manifold: X_new = bI + cP*
    X_new = b * I + c * P_star

    # Ensure symmetry (should already be symmetric, but enforce numerically)
    X_new = 0.5 * (X_new + X_new.T)

    return X_new


@torch.no_grad()
def initialize_on_grassmann(W, a=1.0, b=0.0, r=None):
    """
    Initialize a weight matrix on the Grassmannian G_{a,b,r}.

    Creates X = bI + cP where P is a random rank-r projector.

    Args:
        W: Weight matrix (n×n)
        a, b: Two eigenvalues
        r: Rank of projector (if None, use n//2)

    Returns:
        X: Initialized matrix on the manifold (n×n)
    """
    assert W.ndim == 2 and W.shape[0] == W.shape[1], "W must be square"

    n = W.size(0)
    device = W.device
    dtype = W.dtype

    if r is None:
        r = n // 2  # Default: half rank

    c = a - b

    # Create random orthonormal matrix Q
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))

    # Create rank-r projector: P = Q @ diag([1]*r, [0]*(n-r)) @ Q.T
    D = torch.zeros(n, device=device, dtype=dtype)
    D[:r] = 1.0
    P = Q @ torch.diag(D) @ Q.T

    # X = bI + cP
    I = torch.eye(n, device=device, dtype=dtype)
    X = b * I + c * P

    # Ensure symmetry
    X = 0.5 * (X + X.T)

    return X
