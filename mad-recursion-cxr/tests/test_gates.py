"""Unit tests for the Q1 validity gate + the pluggable-q DPP path. CPU, synthetic.

Q2 (segmentation) needs the TorchXRayVision model and is exercised at pipeline run-time, not here;
we test Q1's image logic and that apply_gates composes masks correctly with Q2 off.
"""
import os, sys
import numpy as np
import torch

FILTERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "filters")
sys.path.insert(0, os.path.abspath(FILTERS))


def _imgs():
    """4 fake 1x32x32 uint8 images: [0]=normal noise, [1]=constant, [2]=blown white, [3]=normal noise."""
    rng = np.random.default_rng(0)
    normal1 = rng.integers(40, 210, size=(1, 32, 32), dtype=np.uint8)
    const   = np.full((1, 32, 32), 128, dtype=np.uint8)
    blown   = np.full((1, 32, 32), 255, dtype=np.uint8)
    normal2 = rng.integers(30, 220, size=(1, 32, 32), dtype=np.uint8)
    return torch.from_numpy(np.stack([normal1, const, blown, normal2]))   # (4,1,32,32)


def test_q1_drops_constant_and_blown():
    import gates as G
    keep = G.q1_validity(_imgs(), term_A=None)
    assert keep.tolist() == [True, False, False, True], f"got {keep.tolist()} (expect drop const+blown)"


def test_q1_drops_bottom_by_termA():
    import gates as G
    imgs = _imgs()
    imgs = imgs[[0, 3, 0, 3]]                                  # all valid images
    term_A = np.array([0.01, 5.0, 6.0, 7.0])                   # index 0 is the least realistic
    keep = G.q1_validity(imgs, term_A=term_A, drop_frac_A=0.25)
    assert keep[0] == False and keep[1:].all(), f"bottom-A image not dropped: {keep.tolist()}"


def test_apply_gates_q2_off_composes():
    import gates as G
    keep, rep = G.apply_gates(_imgs(), use_q1=True, use_q2=False)
    assert keep.tolist() == [True, False, False, True]
    assert rep["q1_dropped"] == 2 and rep["n_kept"] == 2, rep


def test_dpp_vector_quality_biases_picks():
    import dpp as D
    rng = np.random.default_rng(1)
    real = rng.normal(size=(300, 6))
    pool = rng.normal(size=(500, 6))
    q = np.ones(len(pool)); q[:50] = 20.0                     # first 50 have huge quality
    idx, diag = D.greedy_dpp_select(pool, real, n_keep=60, kernel="rbf",
                                    quality="vector", q_vector=q, device="cpu", log_every=60)
    n_top = int((idx < 50).sum())
    assert n_top >= 25, f"vector quality ignored: only {n_top}/50 high-q picked"
    assert diag["quality"] == "vector"


def test_dpp_vector_rejects_negative():
    import dpp as D
    real = np.random.default_rng(2).normal(size=(100, 6))
    pool = np.random.default_rng(3).normal(size=(200, 6))
    q = np.ones(len(pool)); q[0] = -1.0                       # illegal (a distance, not sign-flipped)
    try:
        D.greedy_dpp_select(pool, real, n_keep=20, quality="vector", q_vector=q, device="cpu")
        raise SystemExit("FAILNOEXC")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [test_q1_drops_constant_and_blown, test_q1_drops_bottom_by_termA,
             test_apply_gates_q2_off_composes, test_dpp_vector_quality_biases_picks,
             test_dpp_vector_rejects_negative]
    fails = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1; print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            fails += 1; print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests)-fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
