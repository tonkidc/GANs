"""Deterministic select/eval split of the REAL images (leakage guard), per class.

The single most important correctness property of the herding bake-off: the real images used to
SELECT (herd against) must never be the real images used to SCORE (compute KID/FID). This module
splits each class's real training images into two disjoint halves with a fixed seed and writes them
as JSON path manifests, so the guarantee is structural, not by convention.

  select[_<cls>].json -> herding's F_real (the distribution it matches)
  eval[_<cls>].json   -> the held-out real half every filter's kept set is scored against

cardiomegaly uses the plain names (select.json / eval.json) for backward compatibility; other classes
use the suffixed names. The locked real TEST set (manifest split="test", read by the probe) is a
different set entirely and is never touched here.
"""
import os, glob, json
import numpy as np

REAL_ROOT = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_full\train"   # FULL dataset; was data_cardio_mad
OUT_DIR   = r"C:\Users\Tonkid\Downloads\tstr\filters\splits"
SEED      = 0


def _fname(which, cls):
    return f"{which}.json" if cls == "cardiomegaly" else f"{which}_{cls}.json"


def _all_paths(cls):
    root = os.path.join(REAL_ROOT, cls)
    return sorted(glob.glob(os.path.join(root, "*.png")) + glob.glob(os.path.join(root, "*.jpg")))


def make_split(cls="cardiomegaly", out_dir=OUT_DIR, seed=SEED):
    """Split a class's real paths 50/50 into select/eval, deterministically. Writes two manifests."""
    paths = _all_paths(cls)
    n = len(paths)
    if n < 4:
        raise RuntimeError(f"only {n} real {cls} images — need the preprocessed train tree")
    perm = np.random.default_rng(seed).permutation(n)
    sel = sorted(paths[i] for i in perm[:n // 2])
    ev  = sorted(paths[i] for i in perm[n // 2:])
    assert set(sel).isdisjoint(ev), "BUG: select/eval overlap"
    os.makedirs(out_dir, exist_ok=True)
    for which, subset in (("select", sel), ("eval", ev)):
        json.dump({"cls": cls, "seed": seed, "n_total": n, "n": len(subset), "paths": subset},
                  open(os.path.join(out_dir, _fname(which, cls)), "w"), indent=1)
    return sel, ev


def load_split(which, cls="cardiomegaly", out_dir=OUT_DIR):
    """Return the path list for 'select'/'eval' of a class, building the split on first use."""
    assert which in ("select", "eval")
    p = os.path.join(out_dir, _fname(which, cls))
    if not os.path.exists(p):
        make_split(cls, out_dir=out_dir)
    return json.load(open(p))["paths"]


def assert_disjoint(cls="cardiomegaly", out_dir=OUT_DIR):
    """Hard guard used before any scoring. Raises if the halves overlap."""
    s, e = set(load_split("select", cls, out_dir)), set(load_split("eval", cls, out_dir))
    inter = s & e
    if inter:
        raise AssertionError(f"LEAKAGE ({cls}): {len(inter)} paths in BOTH select and eval")
    return len(s), len(e)


if __name__ == "__main__":
    for cls in ("cardiomegaly", "normal"):
        make_split(cls)
        ns, ne = assert_disjoint(cls)
        print(f"{cls:13s}: select={ns}  eval={ne}  disjoint OK")
    print(f"-> {OUT_DIR}")
