"""Fréchet Distance + KID in ARBITRARY feature spaces (DINOv2, XRV DenseNet-121, ...).

torchmetrics' FID/KID are hard-wired to Inception-v3, so evaluating in another space needs its own
implementation. These reproduce the standard definitions on precomputed features:

  FD  = ||mu_r - mu_f||^2 + Tr(Sr + Sf - 2 (Sr Sf)^{1/2})       ("FID" when the space is Inception)
  KID = unbiased MMD^2 with the polynomial kernel (x.y/d + 1)^3, averaged over random subsets
        (the same estimator torchmetrics uses; reported with its subset std)

Naming: call it FD-<space> (e.g. FD-DINOv2), not FID -- FID means the Inception one specifically.

Tr((Sr Sf)^{1/2}) is computed WITHOUT scipy.sqrtm: with A = Sr^{1/2} (symmetric psd root via eigh),
A Sf A is symmetric psd and shares eigenvalues with Sr Sf, so the trace is sum(sqrt(eigvals)).
That is numerically stabler than a general matrix square root and keeps the dependency list unchanged.
"""
import numpy as np
import torch


def _psd_sqrt(S, eps=1e-12):
    """Symmetric positive-semidefinite square root via eigendecomposition."""
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(eps)
    return (V * w.sqrt()) @ V.T


def frechet_distance(real_feat, fake_feat, eps=1e-6):
    """Fréchet Distance between two feature sets. Space-agnostic (Inception -> this is FID)."""
    R = torch.as_tensor(np.asarray(real_feat), dtype=torch.float64)
    F = torch.as_tensor(np.asarray(fake_feat), dtype=torch.float64)
    mu_r, mu_f = R.mean(0), F.mean(0)
    Sr = torch.cov(R.T) + eps * torch.eye(R.shape[1], dtype=torch.float64)
    Sf = torch.cov(F.T) + eps * torch.eye(F.shape[1], dtype=torch.float64)
    A = _psd_sqrt(Sr)
    # eigenvalues of (A Sf A) == eigenvalues of (Sr Sf); A Sf A is symmetric psd
    w = torch.linalg.eigvalsh(A @ Sf @ A).clamp_min(0)
    tr_cross = w.sqrt().sum()
    d2 = ((mu_r - mu_f) ** 2).sum() + torch.trace(Sr) + torch.trace(Sf) - 2 * tr_cross
    return float(d2.clamp_min(0))


def _poly(X, Y, degree=3):
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** degree


def _mmd2_unbiased(X, Y, degree=3):
    m, n = X.shape[0], Y.shape[0]
    Kxx, Kyy, Kxy = _poly(X, X, degree), _poly(Y, Y, degree), _poly(X, Y, degree)
    sxx = (Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
    syy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
    sxy = Kxy.mean()
    return float(sxx + syy - 2 * sxy)


def kid_from_feats(real_feat, fake_feat, subset_size=1000, subsets=100, degree=3, seed=0):
    """KID = mean/std of unbiased poly-kernel MMD^2 over `subsets` random subsets of `subset_size`.
    Matches torchmetrics' estimator, but on features from any backbone."""
    R = torch.as_tensor(np.asarray(real_feat), dtype=torch.float64)
    F = torch.as_tensor(np.asarray(fake_feat), dtype=torch.float64)
    ss = int(max(2, min(subset_size, R.shape[0], F.shape[0])))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(subsets):
        ri = torch.as_tensor(rng.choice(R.shape[0], ss, replace=False))
        fi = torch.as_tensor(rng.choice(F.shape[0], ss, replace=False))
        vals.append(_mmd2_unbiased(R[ri], F[fi], degree))
    v = np.asarray(vals)
    return {"kid": float(v.mean()), "kid_std": float(v.std()), "subset_size": ss, "subsets": subsets}


def evaluate_space(real_feat, fake_feat, k=5, n_vendi=2000, subset_size=1000, subsets=100, seed=0):
    """FD + KID + coverage/density/recall/vendi for one feature space. Returns a flat dict."""
    from prdc_vendi import coverage_density, recall, vendi_score
    out = {"fd": round(frechet_distance(real_feat, fake_feat), 3)}
    out.update(kid_from_feats(real_feat, fake_feat, subset_size, subsets, seed=seed))
    cd = coverage_density(real_feat, fake_feat, k=k)
    out.update({"coverage": cd["coverage"], "density": cd["density"]})
    out.update({"recall": recall(real_feat, fake_feat, k=k)["recall"]})
    rng = np.random.default_rng(seed)
    F = np.asarray(fake_feat)
    vi = rng.choice(len(F), min(n_vendi, len(F)), replace=False)
    out["vendi"] = vendi_score(F[vi], kernel="cosine")["vendi"]
    return out
