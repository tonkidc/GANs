"""Unit tests for the greedy-MMD (herding) selector. Run before any real data:

    python tstr/tests/test_herding.py        # standalone, prints PASS/FAIL
    pytest  tstr/tests/test_herding.py        # also works

All tests run on CPU with synthetic data. Test 1 (mixture recovery) is the whole hypothesis in
miniature — if it fails, nothing downstream matters.
"""
import os, sys
import numpy as np
import torch

FILTERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "filters")
sys.path.insert(0, os.path.abspath(FILTERS))
import kernels as K
import herding as H


def _mmd2_rbf(S, R, sigma):
    S = torch.as_tensor(S, dtype=torch.float64)
    R = torch.as_tensor(R, dtype=torch.float64)
    kss = K.rbf_kernel(S, S, sigma).mean()
    krr = K.rbf_kernel(R, R, sigma).mean()
    ksr = K.rbf_kernel(S, R, sigma).mean()
    return float(kss + krr - 2.0 * ksr)


def _three_gaussians():
    """Real ~ weights 0.5/0.3/0.2 over 3 well-separated 2-D Gaussians; pool ~ uniform 1/3 each."""
    rng = np.random.default_rng(0)
    mu = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    w = np.array([0.5, 0.3, 0.2])
    comp_r = rng.choice(3, size=2000, p=w)
    real = mu[comp_r] + rng.normal(size=(2000, 2))
    comp_p = rng.integers(0, 3, size=3000)              # wrong proportions: ~1/3 each
    pool = mu[comp_p] + rng.normal(size=(3000, 2))
    return real.astype(np.float32), pool.astype(np.float32), mu, w


def test_mixture_recovery():
    real, pool, mu, w = _three_gaussians()
    sigma = K.median_sigma(torch.as_tensor(real, dtype=torch.float64))
    idx, _ = H.greedy_mmd_select(pool, real, n_keep=300, kernel="rbf",
                                 kernel_kwargs={"sigma": sigma}, device="cpu", log_every=50)
    sel = pool[idx]
    assign = np.argmin(((sel[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)
    props = np.array([(assign == c).mean() for c in range(3)])
    err = np.abs(props - w).max()
    assert err < 0.06, f"selected proportions {props} off target {w} by {err:.3f} (>0.06)"


def test_beats_random():
    real, pool, mu, w = _three_gaussians()
    sigma = K.median_sigma(torch.as_tensor(real, dtype=torch.float64))
    idx, _ = H.greedy_mmd_select(pool, real, n_keep=300, kernel="rbf",
                                 kernel_kwargs={"sigma": sigma}, device="cpu", log_every=300)
    mmd_herd = _mmd2_rbf(pool[idx], real, sigma)
    rng = np.random.default_rng(1)
    rand = np.mean([_mmd2_rbf(pool[rng.choice(len(pool), 300, replace=False)], real, sigma)
                    for _ in range(10)])
    assert mmd_herd < 0.5 * rand, f"herd MMD^2 {mmd_herd:.4g} not < 0.5*random {0.5*rand:.4g}"


def test_duplicate_rejection():
    rng = np.random.default_rng(2)
    real = rng.normal(scale=2.0, size=(500, 2)).astype(np.float32)
    dup = np.zeros((100, 2), np.float32)                 # 100 identical copies of the origin
    varied = rng.normal(scale=2.0, size=(200, 2)).astype(np.float32)
    pool = np.concatenate([dup, varied])                 # copies are indices 0..99
    sigma = K.median_sigma(torch.as_tensor(real, dtype=torch.float64))
    idx, _ = H.greedy_mmd_select(pool, real, n_keep=50, kernel="rbf",
                                 kernel_kwargs={"sigma": sigma}, device="cpu", log_every=50)
    n_dup = int((idx < 100).sum())
    assert n_dup <= 3, f"selected {n_dup} of the duplicated point (>3) — repulsion not working"


def test_kernel_sanity():
    X = torch.randn(20, 8, dtype=torch.float64)
    Kmat = K.poly_kernel(X, X, degree=3)
    assert torch.allclose(Kmat, Kmat.T, atol=1e-9), "poly kernel not symmetric"
    assert torch.allclose(torch.diagonal(Kmat), K.poly_diag(X, degree=3), atol=1e-9), \
        "poly diagonal != (||x||^2/d + 1)^3"


def test_monotone_trace():
    real, pool, mu, w = _three_gaussians()
    sigma = K.median_sigma(torch.as_tensor(real, dtype=torch.float64))
    _, diag = H.greedy_mmd_select(pool, real, n_keep=300, kernel="rbf",
                                  kernel_kwargs={"sigma": sigma}, device="cpu", log_every=25)
    assert diag["mmd2_end"] < diag["mmd2_start"], \
        f"MMD^2 did not decrease: start {diag['mmd2_start']:.4g} -> end {diag['mmd2_end']:.4g}"


if __name__ == "__main__":
    tests = [test_mixture_recovery, test_beats_random, test_duplicate_rejection,
             test_kernel_sanity, test_monotone_trace]
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
