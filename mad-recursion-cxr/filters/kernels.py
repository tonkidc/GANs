"""Kernels for greedy-MMD selection. Torch-based so the same code runs on CPU (unit tests) and GPU
(the real selection loop). Everything accumulates in float64 — the polynomial kernel cubes its
argument and fp32 drifts over thousands of terms.

  poly_kernel : k(x,y) = (x.y / d + 1)^degree      <- the kernel torchmetrics' KID uses
  rbf_kernel  : k(x,y) = exp(-||x-y||^2 / 2 sigma^2)
  median_sigma: the median-heuristic bandwidth, computed ONCE on the select-real half.

Both kernels stream over rows in chunks so a large M x m matrix is never materialised beyond one
chunk of rows at a time.
"""
import torch


def poly_kernel(X, Y, degree=3, chunk_size=4096):
    """(|X|, |Y|) polynomial-kernel matrix, float64. d = feature dim of X."""
    X = torch.as_tensor(X).double()
    Y = torch.as_tensor(Y).double()
    d = X.shape[1]
    out = []
    for i in range(0, X.shape[0], chunk_size):
        out.append((X[i:i + chunk_size] @ Y.T / d + 1.0).pow(degree))
    return torch.cat(out, 0) if out else torch.empty(0, Y.shape[0], dtype=torch.float64)


def poly_diag(X, degree=3):
    """k(x,x) for the polynomial kernel = (||x||^2 / d + 1)^degree, as a length-|X| vector."""
    X = torch.as_tensor(X).double()
    d = X.shape[1]
    return (X.pow(2).sum(1) / d + 1.0).pow(degree)


def rbf_kernel(X, Y, sigma, chunk_size=4096):
    """(|X|, |Y|) RBF-kernel matrix, float64."""
    X = torch.as_tensor(X).double()
    Y = torch.as_tensor(Y).double()
    two_s2 = 2.0 * float(sigma) ** 2
    out = []
    for i in range(0, X.shape[0], chunk_size):
        d2 = torch.cdist(X[i:i + chunk_size], Y).pow(2)
        out.append(torch.exp(-d2 / two_s2))
    return torch.cat(out, 0) if out else torch.empty(0, Y.shape[0], dtype=torch.float64)


def median_sigma(X, max_n=2000, seed=0):
    """Median-heuristic bandwidth: median pairwise distance over a random subsample of X.
    Compute this ONCE on the select-real half and pass sigma explicitly thereafter."""
    X = torch.as_tensor(X).double()
    n = min(max_n, len(X))
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:n]
    d = torch.pdist(X[idx])
    med = float(d.median())
    return med if med > 0 else 1.0
