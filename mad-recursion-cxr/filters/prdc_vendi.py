"""Diversity / coverage metrics for the filter ablation — the columns the selector does NOT optimise,
so they can judge it fairly (KID is the selector's own objective and is used only as a sanity unit-test).

  coverage  (Naeem et al. 2020) — fraction of REAL samples that have at least one fake inside their
             k-NN ball. This is manifold RECALL: did the kept set cover the real distribution's modes,
             tails included? -> the study's DECISION metric.
  density   (Naeem et al. 2020) — how many real k-NN balls each fake lands in (fidelity-ish, robust
             to outliers vs precision).
  vendi     (Friedman & Dieng 2022) — the "effective number of distinct samples": exp(Shannon entropy
             of the eigenvalues of the normalised similarity kernel). Pure internal diversity of the
             kept set, independent of the real set.

All operate on precomputed FEATURES (extract once, in a fixed backbone, so every ablation row is
comparable). Torch-based; float64 for the eigen/entropy step.
"""
import numpy as np
import torch


def _pairwise(A, B):
    return torch.cdist(torch.as_tensor(A, dtype=torch.float64),
                       torch.as_tensor(B, dtype=torch.float64))


def coverage_density(real_feat, fake_feat, k=5):
    """Return {'coverage','density'} (Naeem et al. 2020). k = nearest-neighbour count for real balls."""
    R = torch.as_tensor(real_feat, dtype=torch.float64)
    F = torch.as_tensor(fake_feat, dtype=torch.float64)
    nr, nf = R.shape[0], F.shape[0]
    kk = min(k, nr - 1)
    # radius of each real point = distance to its k-th nearest OTHER real point
    d_rr = _pairwise(R, R)
    d_rr.fill_diagonal_(float("inf"))
    radii = torch.kthvalue(d_rr, kk, dim=1).values          # (nr,) k-th smallest per row
    d_rf = _pairwise(R, F)                                    # (nr, nf)
    inside = d_rf <= radii[:, None]                          # fake j inside real i's ball
    coverage = float((inside.any(1).sum()) / nr)            # reals with >=1 fake nearby
    density = float(inside.sum() / (kk * nf))               # avg real-balls each fake falls in
    return {"coverage": round(coverage, 4), "density": round(density, 4), "k": int(kk)}


def recall(real_feat, fake_feat, k=5):
    """Generative RECALL (Kynkaanniemi et al. 2019): fraction of REAL points that fall inside the FAKE
    manifold (a real point is covered if it lies within some fake's k-NN sphere). This is the direct
    'did the fakes cover the real distribution's variety?' metric — the one a realism gate could hurt."""
    R = torch.as_tensor(real_feat, dtype=torch.float64)
    F = torch.as_tensor(fake_feat, dtype=torch.float64)
    nf = F.shape[0]
    kk = min(k, nf - 1)
    d_ff = _pairwise(F, F)
    d_ff.fill_diagonal_(float("inf"))
    fake_radii = torch.kthvalue(d_ff, kk, dim=1).values      # each fake's k-th nearest fake distance
    d_fr = _pairwise(F, R)                                    # (nf, nr)
    covered = (d_fr <= fake_radii[:, None]).any(0)           # real j inside any fake sphere
    return {"recall": round(float(covered.float().mean()), 4), "k": int(kk)}


def vendi_score(feat, kernel="cosine", sigma=None):
    """Effective number of distinct samples (Friedman & Dieng 2022). Higher = more internally diverse.
    kernel='cosine' (unit-normalised dot product) or 'rbf' (needs/derives sigma)."""
    X = torch.as_tensor(feat, dtype=torch.float64)
    n = X.shape[0]
    if n == 0:
        return {"vendi": float("nan"), "n": 0}
    if kernel == "cosine":
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        S = Xn @ Xn.T
    elif kernel == "rbf":
        if sigma is None:
            d = torch.pdist(X)
            sigma = float(d.median()) if len(d) else 1.0
            sigma = sigma if sigma > 0 else 1.0
        S = torch.exp(-torch.cdist(X, X).pow(2) / (2.0 * sigma ** 2))
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    # Vendi = exp(H(eigenvalues of S/n)); eigenvalues of S/n sum to 1 (trace = n/n).
    w = torch.linalg.eigvalsh(S / n).clamp_min(0)
    w = w / w.sum().clamp_min(1e-12)
    ent = -(w * torch.log(w.clamp_min(1e-12))).sum()
    return {"vendi": round(float(torch.exp(ent)), 4), "n": int(n), "kernel": kernel}
