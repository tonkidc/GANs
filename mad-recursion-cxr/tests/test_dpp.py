"""Unit tests for the k-DPP (diversity) selector. Run before any real data:

    python tstr/tests/test_dpp.py         # standalone, prints PASS/FAIL
    pytest  tstr/tests/test_dpp.py         # also works

All tests run on CPU with synthetic data. The core claims: DPP repels near-duplicates, spreads wider
than random, actually MAXIMISES the log-det it targets, and the real-affinity quality term biases picks
toward the real set.
"""
import os, sys
import numpy as np
import torch

FILTERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "filters")
sys.path.insert(0, os.path.abspath(FILTERS))
import kernels as K
import dpp as D


def _three_gaussians():
    """Real weighted toward component 0; pool ~ uniform over the 3 components."""
    rng = np.random.default_rng(0)
    mu = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    w = np.array([0.6, 0.25, 0.15])
    comp_r = rng.choice(3, size=2000, p=w)
    real = mu[comp_r] + rng.normal(size=(2000, 2))
    comp_p = rng.integers(0, 3, size=3000)
    pool = mu[comp_p] + rng.normal(size=(3000, 2))
    return real.astype(np.float32), pool.astype(np.float32), mu, w


def _mean_nn_dist(X):
    """Mean over rows of the distance to the nearest OTHER row — a spread/coverage measure."""
    X = torch.as_tensor(X, dtype=torch.float64)
    Dm = torch.cdist(X, X)
    Dm.fill_diagonal_(float("inf"))
    return float(Dm.min(1).values.mean())


def _logdet_rbf(sub, sigma):
    """log det of the RBF Gram of a selected set (the pure-diversity DPP objective, q=1)."""
    sub = torch.as_tensor(sub, dtype=torch.float64)
    S = K.rbf_kernel(sub, sub, sigma) + 1e-9 * torch.eye(len(sub), dtype=torch.float64)
    return float(2.0 * torch.log(torch.diagonal(torch.linalg.cholesky(S))).sum())


def test_duplicate_rejection():
    """100 identical copies of the origin: once one is picked the rest add zero volume -> repelled."""
    rng = np.random.default_rng(2)
    real = rng.normal(scale=2.0, size=(500, 2)).astype(np.float32)
    dup = np.zeros((100, 2), np.float32)
    varied = rng.normal(scale=2.0, size=(200, 2)).astype(np.float32)
    pool = np.concatenate([dup, varied])                 # copies are indices 0..99
    idx, _ = D.greedy_dpp_select(pool, real, n_keep=50, kernel="rbf", device="cpu", log_every=50)
    n_dup = int((idx < 100).sum())
    assert n_dup <= 2, f"selected {n_dup} of the duplicated point (>2) — DPP repulsion not working"


def test_beats_random_diversity():
    """Pure-diversity DPP (uniform quality) must spread wider than random selection."""
    real, pool, mu, w = _three_gaussians()
    idx, _ = D.greedy_dpp_select(pool, real, n_keep=200, kernel="rbf", quality="uniform",
                                 device="cpu", log_every=200)
    dpp_nn = _mean_nn_dist(pool[idx])
    rng = np.random.default_rng(1)
    rand_nn = np.mean([_mean_nn_dist(pool[rng.choice(len(pool), 200, replace=False)]) for _ in range(10)])
    assert dpp_nn > rand_nn, f"DPP spread {dpp_nn:.4g} not > random {rand_nn:.4g}"


def test_maximises_logdet():
    """DPP is defined as the subset maximising log-det(L_S) — it must beat random subsets at it."""
    real, pool, mu, w = _three_gaussians()
    sigma = K.median_sigma(torch.as_tensor(pool, dtype=torch.float64))
    idx, diag = D.greedy_dpp_select(pool, real, n_keep=150, kernel="rbf", quality="uniform",
                                    kernel_kwargs={"sigma": sigma}, device="cpu", log_every=150)
    ld_dpp = _logdet_rbf(pool[idx], sigma)
    rng = np.random.default_rng(3)
    ld_rand = np.mean([_logdet_rbf(pool[rng.choice(len(pool), 150, replace=False)], sigma) for _ in range(10)])
    assert ld_dpp > ld_rand, f"DPP log-det {ld_dpp:.4g} not > random {ld_rand:.4g}"
    assert diag["min_gain"] > 0, f"a greedy gain went non-positive ({diag['min_gain']}) — numerics broke"


def test_quality_biases_toward_real():
    """real_affinity quality should raise the kept set's affinity to real vs pure-diversity (uniform)."""
    real, pool, mu, w = _three_gaussians()
    sigma = K.median_sigma(torch.as_tensor(real, dtype=torch.float64))
    Fp = torch.as_tensor(pool, dtype=torch.float64)
    Fr = torch.as_tensor(real, dtype=torch.float64)
    aff = K.rbf_kernel(Fp, Fr, sigma).mean(1)            # per-pool-item affinity to real
    kw = {"sigma": sigma}
    idx_q, _ = D.greedy_dpp_select(pool, real, n_keep=200, kernel="rbf", quality="real_affinity",
                                   kernel_kwargs=kw, device="cpu", log_every=200)
    idx_u, _ = D.greedy_dpp_select(pool, real, n_keep=200, kernel="rbf", quality="uniform",
                                   kernel_kwargs=kw, device="cpu", log_every=200)
    assert float(aff[idx_q].mean()) > float(aff[idx_u].mean()), \
        "real_affinity quality did not increase mean affinity to real vs uniform"


def test_normalised_kernel_unit_diag():
    """The DPP similarity S is normalised to unit diagonal — check that construction on the poly kernel."""
    X = torch.randn(30, 8, dtype=torch.float64)
    Kmat = K.poly_kernel(X, X, degree=3)
    d = K.poly_diag(X, degree=3)
    S = Kmat / torch.sqrt(d[:, None] * d[None, :])
    assert torch.allclose(torch.diagonal(S), torch.ones(30, dtype=torch.float64), atol=1e-9), \
        "normalised similarity diagonal != 1"
    assert torch.allclose(S, S.T, atol=1e-9), "normalised similarity not symmetric"


if __name__ == "__main__":
    tests = [test_duplicate_rejection, test_beats_random_diversity, test_maximises_logdet,
             test_quality_biases_toward_real, test_normalised_kernel_unit_diag]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests)-fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
