"""Unit tests for coverage/density (Naeem 2020) and the Vendi score. CPU, synthetic, no real data."""
import os, sys
import numpy as np
import torch

FILTERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "filters")
sys.path.insert(0, os.path.abspath(FILTERS))
import prdc_vendi as PV


def test_coverage_high_when_fakes_match_real():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(400, 8))
    fake = real + rng.normal(scale=0.01, size=real.shape)     # fakes sit on the real manifold
    cd = PV.coverage_density(real, fake, k=5)
    assert cd["coverage"] > 0.9, f"coverage {cd['coverage']} should be ~1 when fakes match real"


def test_coverage_low_when_fakes_far():
    rng = np.random.default_rng(1)
    real = rng.normal(size=(400, 8))
    fake = rng.normal(size=(400, 8)) + 50.0                   # far away -> covers almost no real balls
    cd = PV.coverage_density(real, fake, k=5)
    assert cd["coverage"] < 0.1, f"coverage {cd['coverage']} should be ~0 when fakes are far"


def test_coverage_drops_on_mode_collapse():
    rng = np.random.default_rng(2)
    real = rng.normal(size=(400, 8))
    spread = rng.normal(size=(300, 8))                        # covers the space
    collapsed = np.zeros((300, 8)) + rng.normal(scale=0.01, size=(300, 8))   # one point
    assert PV.coverage_density(real, spread, k=5)["coverage"] > \
           PV.coverage_density(real, collapsed, k=5)["coverage"], \
           "collapsed fakes should cover less of real than spread fakes"


def test_vendi_duplicates_is_one():
    X = np.ones((100, 8))                                     # all identical -> 1 effective sample
    v = PV.vendi_score(X, kernel="cosine")["vendi"]
    assert abs(v - 1.0) < 0.05, f"Vendi of identical set should be ~1, got {v}"


def test_vendi_orthogonal_is_n():
    X = np.eye(10)                                            # 10 orthogonal -> ~10 effective samples
    v = PV.vendi_score(X, kernel="cosine")["vendi"]
    assert v > 9.0, f"Vendi of 10 orthogonal vectors should be ~10, got {v}"


def test_vendi_rewards_diversity():
    rng = np.random.default_rng(3)
    diverse = rng.normal(size=(200, 16))
    tight = rng.normal(scale=0.001, size=(200, 16)) + rng.normal(size=(1, 16))   # near-duplicates
    assert PV.vendi_score(diverse)["vendi"] > PV.vendi_score(tight)["vendi"], \
        "Vendi should be higher for the diverse set"


if __name__ == "__main__":
    tests = [test_coverage_high_when_fakes_match_real, test_coverage_low_when_fakes_far,
             test_coverage_drops_on_mode_collapse, test_vendi_duplicates_is_one,
             test_vendi_orthogonal_is_n, test_vendi_rewards_diversity]
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
