"""The 4-stage filter pipeline:  50k pool -> Q1 validity -> Q2 anatomy -> selector -> n_keep.

Gates (Q1, Q2) only remove; the selector (herding or DPP) makes the actual set-level choice at the end.
Q5 (label gate) is intentionally skipped in v1 (classifier gate vs classifier judge = circular).

run_pipeline() works on a class's pool PATHS plus precomputed POOL/SELECT features (extract once, fixed
backbone). It loads raw images only in chunks for the gates, so 50k never sits in RAM at once. Returns the
kept GLOBAL pool indices plus a per-stage drop report — every number re-derivable from the seed.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import cardio_metrics as M
import kernels as K
import gates as G
import herding as H
import dpp as DPP


def _term_A(pool_feat, sel_feat, degree=3):
    """Herding's attraction A(x) = mean_j k_poly(x, real_select) — the realism score Q1 thresholds on."""
    Fp = torch.as_tensor(pool_feat, dtype=torch.float64)
    Fr = torch.as_tensor(sel_feat, dtype=torch.float64)
    return K.poly_kernel(Fp, Fr, degree=degree).mean(1).cpu().numpy()


def _q1_mask_chunked(pool_paths, term_A, chunk=2000, **q1thr):
    """Q1 over the whole pool, loading images chunk-by-chunk (image stats) + term_A (feature-based)."""
    N = len(pool_paths)
    keep = np.zeros(N, dtype=bool)
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        imgs = M.load_paths_uint8(pool_paths[sl])
        keep[sl] = G.q1_validity(imgs, term_A=term_A[sl], **q1thr)
    return keep


def _q2_mask_survivors(pool_paths, keep1, chunk=512, device="cuda", **q2thr):
    """Q2 (segmentation) only on Q1 survivors, chunked. Returns a full-length mask."""
    N = len(pool_paths)
    keep2 = np.ones(N, dtype=bool)
    idx = np.where(keep1)[0]
    for i in range(0, len(idx), chunk):
        gidx = idx[i:i + chunk]
        imgs = M.load_paths_uint8([pool_paths[j] for j in gidx])
        keep2[gidx] = G.q2_anatomy(imgs, device=device, **q2thr)
    return keep2


def _select(selector, pool_feat, sel_feat, sub_idx, n_keep, q_vector=None, seed=0, log_every=500):
    """Run the chosen selector on the GATED subset; return kept GLOBAL indices + diag."""
    Fp = np.asarray(pool_feat)[sub_idx]
    if len(sub_idx) < n_keep:
        raise RuntimeError(f"only {len(sub_idx)} survive the gates < n_keep {n_keep} — loosen gates")
    if selector == "herding":
        loc, diag = H.greedy_mmd_select(Fp, sel_feat, n_keep, kernel="poly",
                                        kernel_kwargs={"degree": 3}, log_every=log_every)
    elif selector == "dpp":
        sigma = K.median_sigma(torch.as_tensor(sel_feat, dtype=torch.float64))
        quality = "vector" if q_vector is not None else "real_affinity"
        qv = None if q_vector is None else np.asarray(q_vector)[sub_idx]
        loc, diag = DPP.greedy_dpp_select(Fp, sel_feat, n_keep, kernel="rbf",
                                          kernel_kwargs={"sigma": sigma}, quality=quality,
                                          q_vector=qv, log_every=log_every)
    elif selector == "random":
        loc = np.random.default_rng(seed).choice(len(sub_idx), size=n_keep, replace=False)
        diag = {"kind": "random"}
    else:
        raise ValueError(f"unknown selector: {selector}")
    return np.asarray(sub_idx)[loc], diag


def run_pipeline(pool_paths, pool_feat, sel_feat, sel_paths=None, selector="herding", n_keep=3300,
                 use_mem=False, mem_pool_feat=None, mem_real_feat=None, mem_pct=1,
                 use_maha=False, maha_pool_feat=None, maha_real_feat=None, maha_drop=0.05,
                 use_xrv=True, use_wire=True, use_q1=False, use_q2=False, q_vector=None,
                 device="cuda", seed=0, xrv_drop=0.05, wire_drop=0.05, q1thr=None, q2thr=None):
    """Full staged pipeline for one class:  pool -> [memorization] -> XRV-realism -> wire-density -> herding.
    Default gates are the validated pair (XRV drops crinkle, wire drops tangles); the old Q1/Q2 remain
    available but off by default (Q1 was Inception-blind, Q2 barely dropped anything). The memorization
    gate (use_mem) runs FIRST and needs its own features: mem_pool_feat + mem_real_feat (XRV, anatomy-aware)
    — it is the only gate that removes HIGH-fidelity near-copies rather than low-quality junk. Returns
    (kept_global_idx ndarray, report dict). sel_paths (real select image paths) is required for use_xrv."""
    import wire_gate as WG
    N = len(pool_paths)
    report = {"n_pool": int(N), "selector": selector,
              "gates": {"mem": use_mem, "xrv": use_xrv, "wire": use_wire, "q1": use_q1, "q2": use_q2}}
    term_A = _term_A(pool_feat, sel_feat)

    keep = np.ones(N, dtype=bool)
    if use_mem:                                             # privacy: drop near-copies of real patients
        if mem_pool_feat is None or mem_real_feat is None:
            raise ValueError("use_mem=True needs mem_pool_feat and mem_real_feat (XRV features)")
        mem_keep, mem_info = G.q0_memorization(mem_pool_feat, mem_real_feat, pct=mem_pct, device=device)
        keep &= mem_keep
        report["after_mem"] = int(keep.sum())
        report["mem"] = mem_info
    if use_maha:                                            # PCA-Mahalanobis: drop outlier/broken frames
        if maha_pool_feat is None or maha_real_feat is None:
            raise ValueError("use_maha=True needs maha_pool_feat and maha_real_feat (inception features)")
        maha_keep, maha_info = G.q_maha(maha_pool_feat, maha_real_feat, drop_frac=maha_drop)
        keep &= maha_keep
        report["after_maha"] = int(keep.sum())
        report["maha"] = maha_info
    if use_xrv:                                              # chest-aware realism -> drops crinkle
        if sel_paths is None:
            raise ValueError("use_xrv=True needs sel_paths (real select image paths)")
        xrv_keep, _ = G.q_realism_xrv(pool_paths, sel_paths, drop_frac=xrv_drop)
        keep &= xrv_keep
        report["after_xrv"] = int(keep.sum())
    if use_wire:                                             # wire density -> drops tangles
        cov = WG.score_paths(pool_paths, fn=WG.wire_coverage)
        keep &= (cov < np.quantile(cov, 1 - wire_drop))
        report["after_wire"] = int(keep.sum())
    if use_q1:
        keep &= _q1_mask_chunked(pool_paths, term_A, **(q1thr or {}))
        report["after_q1"] = int(keep.sum())
    if use_q2:
        keep &= _q2_mask_survivors(pool_paths, keep, device=device, **(q2thr or {}))
        report["after_q2"] = int(keep.sum())

    sub_idx = np.where(keep)[0]
    kept_idx, diag = _select(selector, pool_feat, sel_feat, sub_idx, n_keep,
                             q_vector=q_vector, seed=seed)
    report["n_gated"] = int(len(sub_idx))
    report["n_kept"] = int(len(kept_idx))
    report["term_A_drop_thr"] = None
    report["selector_diag"] = diag
    return kept_idx, report
