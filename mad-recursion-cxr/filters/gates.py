"""Q1 (validity) and Q2 (anatomy) gates — the cheap, per-sample, HIGH-RECALL stages that drop the
indefensible before the set-level selector makes the real choice. Gates only ever REMOVE clearly-bad
images; they never rank or pick. Returns boolean keep-masks so the pipeline can compose them.

Q1 validity  — kill broken images: near-constant intensity, blown-out histograms, and the bottom
                few % by term A (real-affinity realism score, already computed for herding). ~free.
Q2 anatomy   — run the TorchXRayVision chest segmenter (reused from cardio_metrics._seg_model) and keep
                images where two lungs + a cardiac silhouette are found with plausible geometry.
                HARD RULE: gate on segmentation SUCCESS / geometry only — NEVER on the CTR value, which
                is an eval metric (gating on it would be circular). No CTR is computed here.

All thresholds are explicit arguments (fixed across ablation rows); nothing is hand-tuned per run.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import cardio_metrics as M


# ----------------------------- Q1: validity -----------------------------
def q1_validity(imgs_uint8, term_A=None, drop_frac_A=0.03,
                std_min=6.0, blown_frac=0.35):
    """Keep-mask (bool, len N). Drops:
      * near-constant images   (per-image intensity std < std_min, on 0..255)
      * blown-out histograms   (> blown_frac of pixels at the 0 or 255 rails)
      * bottom drop_frac_A by term_A, if term_A given (A = real-affinity; LOW = least realistic)
    High recall by design: only clearly-broken images fall.
    """
    N = len(imgs_uint8)
    x = imgs_uint8.float() if torch.is_tensor(imgs_uint8) else torch.as_tensor(imgs_uint8).float()
    x = x.view(N, -1)                                         # (N, H*W*C) on 0..255
    std = x.std(1)
    rail = ((x <= 1) | (x >= 254)).float().mean(1)           # fraction of pixels pinned at a rail
    keep = (std >= std_min) & (rail <= blown_frac)
    if term_A is not None and drop_frac_A > 0:
        a = np.asarray(term_A, dtype=np.float64)
        thr = np.quantile(a, drop_frac_A)                    # bottom drop_frac_A by realism
        keep = keep & torch.as_tensor(a > thr)
    return keep.cpu().numpy().astype(bool)


# ----------------------------- Q2: anatomy ------------------------------
@torch.no_grad()
def _seg_areas(imgs_uint8, dev, batch=24):        # was 8: bigger batch feeds the GPU (Q2 was CPU/IO-starved)
    """Per-image (heart_frac, leftlung_frac, rightlung_frac) mask-area fractions from the XRV PSPNet.
    Reuses the exact segmenter cardio_metrics uses for CTR, but reads AREAS/geometry, not the CTR ratio."""
    import torchxrayvision as xrv, torchvision
    m = M._seg_model(dev)
    H  = m.targets.index("Heart")
    LL = m.targets.index("Left Lung")
    RL = m.targets.index("Right Lung")
    resize = torchvision.transforms.Compose([xrv.datasets.XRayResizer(512)])
    xs = []
    for i in range(len(imgs_uint8)):
        a = imgs_uint8[i, 0].numpy().astype(np.float32)
        a = xrv.datasets.normalize(a, 255)[None, ...]
        xs.append(resize(a))
    X = torch.from_numpy(np.stack(xs))                       # (N,1,512,512)
    out = []
    for i in range(0, len(X), batch):
        p = torch.sigmoid(m(X[i:i + batch].to(dev)))         # (b,14,512,512)
        area = (p > 0.5).float().mean(dim=(2, 3)).cpu().numpy()   # fraction of frame per channel
        for j in range(p.shape[0]):
            out.append((float(area[j, H]), float(area[j, LL]), float(area[j, RL])))
    if dev != "cpu":
        torch.cuda.empty_cache()
    return np.array(out, dtype=np.float32)                   # (N,3)


def q2_anatomy(imgs_uint8, heart_min=0.005, lung_min=0.02, lung_ratio_max=6.0, device="cuda"):
    """Keep-mask (bool). Keeps images where the segmenter finds a heart AND both lungs with plausible
    geometry: each area above a floor, and the two lungs not wildly asymmetric (ratio <= lung_ratio_max).
    Gates on segmentation SUCCESS + geometry only — no CTR. Robust: on any seg failure keeps the image
    (high recall: a gate should not delete a chest just because the segmenter choked).
    """
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    try:
        areas = _seg_areas(imgs_uint8, dev)
    except Exception as e:
        print(f"  [q2_anatomy] segmenter unavailable ({e!r}) -> keeping all (high recall)", flush=True)
        return np.ones(len(imgs_uint8), dtype=bool)
    heart, ll, rl = areas[:, 0], areas[:, 1], areas[:, 2]
    has_heart = heart >= heart_min
    has_lungs = (ll >= lung_min) & (rl >= lung_min)
    lo = np.minimum(ll, rl); hi = np.maximum(ll, rl)
    plausible = (hi / np.clip(lo, 1e-6, None)) <= lung_ratio_max
    return (has_heart & has_lungs & plausible)


def q_realism_xrv(pool_paths, sel_paths, drop_frac=0.05):
    """Chest-aware realism gate. Scores each fake's affinity to REAL chests in XRV-DenseNet (chest-trained)
    feature space — the space that actually 'sees' broken anatomy — and drops the bottom `drop_frac`.
    This catches the crinkle / bad-wire artifacts that Inception-space Q1 is blind to. Runs at XRV's 224px
    (far faster than Q2's 512px segmenter). Returns (keep_mask bool, xrv_affinity scores).
    """
    import features as FE, kernels as K
    pool_x = FE.extract_features(pool_paths, backbone="xrv")
    sel_x = FE.extract_features(sel_paths, backbone="xrv")
    sigma = K.median_sigma(torch.as_tensor(sel_x, dtype=torch.float64))
    A = K.rbf_kernel(torch.as_tensor(pool_x, dtype=torch.float64),
                     torch.as_tensor(sel_x, dtype=torch.float64), sigma).mean(1).cpu().numpy()
    thr = float(np.quantile(A, drop_frac))               # drop the least-chest-like fraction
    return (A > thr), A


def q0_memorization(pool_feat, real_feat, pct=1, chunk=512, device="cuda"):
    """PRIVACY gate — drop fakes that are near-copies of a real patient.

    tau = the pct-th percentile of REAL->REAL nearest-neighbour distances (the natural spacing between
    real patients); a fake whose nearest real neighbour is closer than tau is 'too similar' -> dropped.
    Run in an anatomy-aware space (XRV) so identity, not ImageNet texture, drives the match.

    This is the one gate that is NOT redundant with herding's realism term A: term A rewards closeness
    to real (high fidelity), so herding *prefers* near-copies; this gate removes exactly those. It cuts
    against the selector instead of agreeing with it. Insurance, not quality. Returns (keep_mask, info).
    """
    fr = torch.as_tensor(real_feat, dtype=torch.float32)
    ff = torch.as_tensor(pool_feat, dtype=torch.float32)
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    fr = fr.to(dev)
    R = fr.shape[0]
    # real->real NN (exclude self) -> calibrate tau
    d_ref = []
    for i in range(0, R, chunk):
        blk = fr[i:i + chunk]
        d = torch.cdist(blk, fr)
        rows = torch.arange(blk.shape[0], device=dev)
        cols = torch.arange(i, i + blk.shape[0], device=dev)
        d[rows, cols] = float("inf")                         # mask self
        d_ref.append(d.min(1).values.cpu())
    tau = float(np.percentile(torch.cat(d_ref).numpy(), pct))
    # fake->real NN
    d_fake = []
    for i in range(0, ff.shape[0], chunk):
        d = torch.cdist(ff[i:i + chunk].to(dev), fr)
        d_fake.append(d.min(1).values.cpu())
    d_fake = torch.cat(d_fake).numpy()
    if dev != "cpu":
        torch.cuda.empty_cache()
    keep = d_fake >= tau                                      # drop the too-close (memorized) fakes
    info = {"tau": round(tau, 4), "pct": pct, "n_in": int(len(keep)),
            "n_dropped": int((~keep).sum()), "drop_frac": round(float((~keep).mean()), 4)}
    return keep.astype(bool), info


def q_maha(pool_feat, real_feat, drop_frac=0.05, k=50, seed=0):
    """PCA-Mahalanobis OUTLIER gate. Fits PCA on the real features, then drops the `drop_frac` of fakes
    farthest (diagonal Mahalanobis distance) from the real distribution's centre. Removes broken/atypical
    frames (genuine outliers) that herding might otherwise pick for diversity. Does NOT remove in-
    distribution devices/framing (those sit close to the real centre). Works on cached features (no
    re-extraction). Returns (keep_mask, info)."""
    from sklearn.decomposition import PCA
    rf = np.asarray(real_feat, dtype=np.float64)
    ff = np.asarray(pool_feat, dtype=np.float64)
    pca = PCA(n_components=min(k, rf.shape[1], len(rf) - 1), random_state=seed).fit(rf)
    Zr, Zf = pca.transform(rf), pca.transform(ff)
    std = Zr.std(0) + 1e-6
    d = np.sqrt((((Zf - Zr.mean(0)) / std) ** 2).sum(1))     # per-fake Mahalanobis distance
    thr = float(np.quantile(d, 1.0 - drop_frac))             # keep the closest (1 - drop_frac)
    keep = d <= thr
    info = {"drop_frac": drop_frac, "k": k, "n_dropped": int((~keep).sum()), "thr": round(thr, 3),
            "d_median": round(float(np.median(d)), 3), "d_max": round(float(d.max()), 3)}
    return keep.astype(bool), info


def apply_gates(imgs_uint8, term_A=None, use_q1=True, use_q2=True, device="cuda", **thr):
    """Compose the enabled gates -> (keep_mask, report dict of how many each stage dropped)."""
    N = len(imgs_uint8)
    keep = np.ones(N, dtype=bool)
    report = {"n_in": int(N)}
    if use_q1:
        m1 = q1_validity(imgs_uint8, term_A=term_A,
                         drop_frac_A=thr.get("drop_frac_A", 0.03),
                         std_min=thr.get("std_min", 6.0),
                         blown_frac=thr.get("blown_frac", 0.35))
        report["q1_dropped"] = int((~m1).sum())
        keep &= m1
    if use_q2:
        idx = np.where(keep)[0]                               # only segment survivors (seg is heavy)
        m2_sub = q2_anatomy(imgs_uint8[torch.as_tensor(idx)] if torch.is_tensor(imgs_uint8)
                            else imgs_uint8[idx],
                            heart_min=thr.get("heart_min", 0.005),
                            lung_min=thr.get("lung_min", 0.02),
                            lung_ratio_max=thr.get("lung_ratio_max", 6.0), device=device)
        m2 = np.ones(N, dtype=bool)
        m2[idx] = m2_sub
        report["q2_dropped"] = int((keep & ~m2).sum())
        keep &= m2
    report["n_kept"] = int(keep.sum())
    return keep, report
