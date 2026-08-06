"""FID/KID + copy-rate (memorization) for the cardio MAD recursion.

Two references, deliberately different:
  * FID / KID  -> fakes vs the ORIGINAL real set (held FIXED every generation) = drift from reality.
  * copy rate  -> fakes vs the data THIS generation was TRAINED on = memorization of its inputs.

Same torchmetrics InceptionV3 (feature=2048, uint8 in) as metrics_hygd, so numbers are comparable
to the HYGD study. FID is biased upward at small n (trust KID there); MIMIC's real side is larger,
so FID is steadier here, but we report both.

Copy rate: a fake whose nearest real neighbour in Inception space is closer than the p-th percentile
of genuine real-real neighbour distances is flagged "too close to be novel". Reported as a fraction,
with the calibration distances (tau, medians) so it's transparent rather than a magic number.

NOTE (eval-mode BatchNorm): Inception runs in eval mode, which uses the inference BN path and does
NOT hit the ROCm MIOpen BatchNorm-*train* JIT bug that forces the DenseNet probe to disable cudnn.
So metrics run fine in-process with cudnn ON (same as metrics_hygd already does on this box).
"""
import os, glob
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate incompletely-downloaded jpgs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["cardiomegaly", "normal"]
METRIC_N = 5000          # cap images per side of a comparison (bounds RAM/time; FID/KID plenty at 5k)
CTR_N    = 300           # images per class for the CTR distribution (seg is heavy; 300 is plenty)


def load_uint8(root, cls=None, limit=METRIC_N, seed=0):
    """Load PNG/JPG from root/<cls>/ (or all CLASSES) -> uint8 (N,3,256,256). Grayscale->RGB."""
    subs = [cls] if cls else CLASSES
    fs = []
    for c in subs:
        fs += sorted(glob.glob(os.path.join(root, c, "*.png")) +
                     glob.glob(os.path.join(root, c, "*.jpg")))
    if limit and len(fs) > limit:
        rng = np.random.default_rng(seed)
        fs = sorted(rng.choice(fs, size=limit, replace=False).tolist())
    if not fs:
        return torch.empty(0, 3, 256, 256, dtype=torch.uint8)
    arr = np.stack([np.asarray(Image.open(f).convert("RGB"), np.uint8) for f in fs])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()   # (N,3,H,W) uint8


def _new_fid():
    from torchmetrics.image.fid import FrechetInceptionDistance
    return FrechetInceptionDistance(feature=2048, normalize=False).to(DEV)


@torch.no_grad()
def _feed(metric, imgs, real, batch=64):
    for i in range(0, len(imgs), batch):
        metric.update(imgs[i:i + batch].to(DEV), real=real)


@torch.no_grad()
def _features(fid_obj, imgs, batch=64):
    """2048-d Inception features via the FID object's frozen inception net."""
    out = []
    for i in range(0, len(imgs), batch):
        out.append(fid_obj.inception(imgs[i:i + batch].to(DEV)).cpu())
    return torch.cat(out) if out else torch.empty(0, 2048)


@torch.no_grad()
def fid_kid(real_imgs, fake_imgs):
    """FID + KID (real vs fake). Returns nan-safe dict."""
    nr, nf = len(real_imgs), len(fake_imgs)
    if nr < 2 or nf < 2:
        return {"fid": float("nan"), "kid": float("nan"), "kid_std": float("nan"),
                "n_real": nr, "n_fake": nf}
    from torchmetrics.image.kid import KernelInceptionDistance
    fid = _new_fid()
    _feed(fid, real_imgs, True); _feed(fid, fake_imgs, False)
    f = float(fid.compute()); del fid; torch.cuda.empty_cache()

    ss = max(10, min(1000, nr // 2, nf // 2))
    kid = KernelInceptionDistance(feature=2048, subset_size=ss, subsets=100,
                                  normalize=False).to(DEV)
    _feed(kid, real_imgs, True); _feed(kid, fake_imgs, False)
    km, ks = kid.compute(); del kid; torch.cuda.empty_cache()
    return {"fid": round(f, 3), "kid": round(float(km), 6), "kid_std": round(float(ks), 6),
            "n_real": nr, "n_fake": nf, "kid_subset": ss}


@torch.no_grad()
def inception_score(fake_imgs):
    """Inception Score (fake-only). CAVEAT: IS uses ImageNet class logits, which don't describe
    chest X-rays, so absolute IS is near-meaningless in this domain (HYGD: real fundus scored ~1.58,
    fakes scored HIGHER). Only trust it RELATIVE to the real-set IS reference / across generations."""
    n = len(fake_imgs)
    if n < 4:
        return {"is_mean": float("nan"), "is_std": float("nan"), "n": n}
    from torchmetrics.image.inception import InceptionScore
    splits = 10 if n >= 20 else max(1, n // 2)
    m = InceptionScore(feature="logits_unbiased", splits=splits, normalize=False).to(DEV)
    for i in range(0, n, 64):
        m.update(fake_imgs[i:i + 64].to(DEV))
    a, b = m.compute(); del m; torch.cuda.empty_cache()
    return {"is_mean": round(float(a), 4), "is_std": round(float(b), 4), "n": n}


@torch.no_grad()
def copy_rate(ref_imgs, fake_imgs, pct=5, chunk=512):
    """Inception-space nearest-neighbour memorization check.
    tau = p-th percentile of real-real NN distances; copy_rate = frac of fakes with a real
    neighbour closer than tau. Baseline expectation ~ pct%; a value well above it = memorization."""
    nr, nf = len(ref_imgs), len(fake_imgs)
    if nr < 3 or nf < 1:
        return {"copy_rate": float("nan"), "tau": float("nan"), "n_ref": nr, "n_fake": nf}
    fid = _new_fid()
    fr = _features(fid, ref_imgs).to(DEV)       # (R,2048)
    ff = _features(fid, fake_imgs)              # (F,2048) on cpu, streamed to gpu in chunks
    del fid; torch.cuda.empty_cache()

    # real->real NN distance (exclude self) for calibration
    d_ref = []
    for i in range(0, nr, chunk):
        blk = fr[i:i + chunk]
        d = torch.cdist(blk, fr)                          # (c,R)
        rows = torch.arange(blk.shape[0], device=DEV)
        cols = torch.arange(i, i + blk.shape[0], device=DEV)
        d[rows, cols] = float("inf")                     # mask self
        d_ref.append(d.min(1).values.cpu())
    d_ref = torch.cat(d_ref)
    tau = float(np.percentile(d_ref.numpy(), pct))

    # fake->real NN distance
    d_fake = []
    for i in range(0, nf, chunk):
        d = torch.cdist(ff[i:i + chunk].to(DEV), fr)     # (c,R)
        d_fake.append(d.min(1).values.cpu())
    d_fake = torch.cat(d_fake)
    del fr; torch.cuda.empty_cache()

    cr = float((d_fake < tau).float().mean())
    return {"copy_rate": round(cr, 4), "tau": round(tau, 3), "pct": pct,
            "med_fake_nn": round(float(d_fake.median()), 3),
            "med_ref_nn": round(float(d_ref.median()), 3),
            "n_ref": nr, "n_fake": nf}


# ---------------------------------------------------------------------------
# Disease size/shape — cardiothoracic ratio (CTR) via torchxrayvision segmentation
# ---------------------------------------------------------------------------
# CTR = max transverse cardiac width / max internal thoracic (lung-to-lung) width. Clinically
# CTR > 0.5 = cardiomegaly, so this DIRECTLY measures the diagnostic feature: if recursion shrinks
# the hearts, the fake cardiomegaly CTR distribution drifts down toward normal = the disease dying.
# Needs a segmenter (the "extra data"): torchxrayvision's chestx_det PSPNet (Heart + L/R Lung).
_SEG = {}

def _seg_model(dev):
    if dev not in _SEG:
        import torchxrayvision as xrv
        _SEG[dev] = xrv.baseline_models.chestx_det.PSPNet().eval().to(dev)
    return _SEG[dev]


def _max_width(mask):
    """Widest horizontal run of a binary mask (max over rows of right-left extent)."""
    rows = np.where(mask.any(1))[0]
    best = 0
    for r in rows:
        c = np.where(mask[r])[0]
        best = max(best, int(c.max() - c.min() + 1))
    return best


@torch.no_grad()
def _ctr_run(imgs_uint8, dev, batch=8):
    import torchxrayvision as xrv
    import torchvision
    m = _seg_model(dev)
    H  = m.targets.index("Heart")
    LL = m.targets.index("Left Lung")
    RL = m.targets.index("Right Lung")
    resize = torchvision.transforms.Compose([xrv.datasets.XRayResizer(512)])
    xs = []
    for i in range(len(imgs_uint8)):
        a = imgs_uint8[i, 0].numpy().astype(np.float32)      # grayscale channel, 0..255
        a = xrv.datasets.normalize(a, 255)[None, ...]        # (1,H,W) -> [-1024,1024]
        xs.append(resize(a))                                 # (1,512,512)
    X = torch.from_numpy(np.stack(xs))                       # (N,1,512,512)
    out = []
    for i in range(0, len(X), batch):
        p = torch.sigmoid(m(X[i:i + batch].to(dev)))         # (b,14,512,512)
        for j in range(p.shape[0]):
            heart = (p[j, H] > 0.5).cpu().numpy()
            lungs = ((p[j, LL] > 0.5) | (p[j, RL] > 0.5)).cpu().numpy()
            tw = _max_width(lungs); cw = _max_width(heart)
            out.append(cw / tw if tw else np.nan)
    if dev != "cpu":
        torch.cuda.empty_cache()
    return np.array(out, dtype=np.float32)


def ctr_values(imgs_uint8, batch=8):
    """Per-image cardiothoracic ratio (nan where no heart/lung found). Robust: tries the GPU seg
    model, falls back to CPU, and if both fail returns all-nan rather than crashing the run.
    Returns [] only if torchxrayvision isn't installed."""
    if len(imgs_uint8) == 0:
        return np.array([])
    try:
        import torchxrayvision, torchvision  # noqa: F401
    except Exception as e:
        print(f"  [CTR skipped: {e!r} -> pip install torchxrayvision imageio scikit-image "
              f"lazy_loader tifffile pandas]", flush=True)
        return np.array([])
    for dev in ([DEV, "cpu"] if DEV != "cpu" else ["cpu"]):
        try:
            return _ctr_run(imgs_uint8, dev, batch)
        except Exception as e:
            print(f"  [CTR on {dev} failed: {e!r}]", flush=True)
    return np.full(len(imgs_uint8), np.nan, dtype=np.float32)


def mean_ctr(imgs_uint8):
    """Mean CTR + fraction over the clinical 0.5 threshold, nan-safe."""
    v = ctr_values(imgs_uint8)
    v = v[~np.isnan(v)] if len(v) else v
    if len(v) == 0:
        return {"ctr_mean": float("nan"), "ctr_frac_gt0.5": float("nan"), "n": 0}
    return {"ctr_mean": round(float(v.mean()), 4),
            "ctr_frac_gt0.5": round(float((v > 0.5).mean()), 4), "n": int(len(v))}


# ---------------------------------------------------------------------------
# Quality filter — the HYGD filter_fakes.py method, generalized & per-class
# ---------------------------------------------------------------------------
# A good fake sits INSIDE the real distribution in Inception feature space; an artifact is an
# OUTLIER. Fit PCA on the REAL images of a class, score each fake by its normalized (diagonal-
# Mahalanobis) distance from the real centre; low = realistic, high = junk. Threshold = the real
# images' own keep_pct-th percentile ("drop a fake if it's a bigger outlier than keep_pct% of reals").
def load_paths_uint8(paths):
    """Load an explicit list of image paths -> uint8 (N,3,256,256)."""
    if not paths:
        return torch.empty(0, 3, 256, 256, dtype=torch.uint8)
    arr = np.stack([np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def fit_real_filter(real_dir, k=50, keep_pct=95, limit=METRIC_N):
    """Fit the per-class PCA+Mahalanobis quality gate on REAL images.
    Returns {cls: (mean, comps, std, thr)} — pass to filter_scores()."""
    fid = _new_fid()
    out = {}
    for cls in CLASSES:
        rf = _features(fid, load_uint8(real_dir, cls=cls, limit=limit)).numpy()   # (n,2048)
        mean = rf.mean(0)
        Uc = rf - mean
        _, _, Vh = np.linalg.svd(Uc, full_matrices=False)
        comps = Vh[:k]                                          # (k,2048)
        Zr = Uc @ comps.T
        std = Zr.std(0) + 1e-6
        sr = np.sqrt(((Zr / std) ** 2).sum(1))
        thr = float(np.percentile(sr, keep_pct))
        out[cls] = (mean.astype(np.float32), comps.astype(np.float32),
                    std.astype(np.float32), thr)
    del fid; torch.cuda.empty_cache()
    return out


@torch.no_grad()
def filter_scores(scorer, imgs_uint8):
    """Mahalanobis distance from the real centre for each fake (lower = more real). Returns
    (scores, thr) where thr is the real keep_pct gate from fit_real_filter."""
    mean, comps, std, thr = scorer
    if len(imgs_uint8) == 0:
        return np.array([]), thr
    fid = _new_fid()
    ff = _features(fid, imgs_uint8).numpy()
    del fid; torch.cuda.empty_cache()
    Z = (ff - mean) @ comps.T
    return np.sqrt(((Z / std) ** 2).sum(1)), thr


# ---------------------------------------------------------------------------
# Real-vs-fake separability: C2ST (one number) + PCA/t-SNE/UMAP (a picture)
# ---------------------------------------------------------------------------
def _subsample(imgs, n, seed=0):
    if len(imgs) <= n:
        return imgs
    g = torch.Generator().manual_seed(seed)
    return imgs[torch.randperm(len(imgs), generator=g)[:n]]


@torch.no_grad()
def c2st(real_imgs, fake_imgs, k=50, seed=0):
    """Classifier two-sample test: RandomForest 5-fold CV accuracy at telling real from fake in
    Inception-PCA space. 0.5 = indistinguishable (great fakes), 1.0 = fully separable (bad)."""
    nr, nf = len(real_imgs), len(fake_imgs)
    if nr < 10 or nf < 10:
        return {"c2st": float("nan"), "n_real": nr, "n_fake": nf}
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    fid = _new_fid()
    rf = _features(fid, real_imgs).numpy()
    ff = _features(fid, fake_imgs).numpy()
    del fid; torch.cuda.empty_cache()
    X = np.concatenate([rf, ff]); y = np.array([0] * nr + [1] * nf)
    kk = min(k, X.shape[1], len(X) - 1)
    Xp = PCA(n_components=kk, random_state=seed).fit_transform(X)
    acc = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=seed),
                          Xp, y, cv=5).mean()
    return {"c2st": round(float(acc), 4), "n_real": nr, "n_fake": nf}


@torch.no_grad()
def rank_by_method(real_imgs, fake_imgs, method="maha", k=50, seed=0):
    """Per-fake realism score, LOWER = more real (so 'keep the lowest N'). Methods:
       'maha' PCA+Mahalanobis distance to the real centre (global density),
       'knn'  mean distance to the k nearest REAL images in PCA space (local density),
       'disc' out-of-fold RandomForest P(fake) — the C2ST classifier used per image."""
    fid = _new_fid()
    rf = _features(fid, real_imgs).numpy()
    ff = _features(fid, fake_imgs).numpy()
    del fid; torch.cuda.empty_cache()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(k, rf.shape[1], len(rf) - 1), random_state=seed).fit(rf)
    Zr, Zf = pca.transform(rf), pca.transform(ff)
    if method == "maha":
        std = Zr.std(0) + 1e-6
        return np.sqrt((((Zf - Zr.mean(0)) / std) ** 2).sum(1))
    if method == "knn":
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(Zr))).fit(Zr)
        return nn.kneighbors(Zf)[0].mean(1)
    if method == "disc":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict
        X = np.concatenate([Zr, Zf]); y = np.array([0] * len(Zr) + [1] * len(Zf))
        proba = cross_val_predict(RandomForestClassifier(n_estimators=200, random_state=seed),
                                  X, y, cv=5, method="predict_proba")
        return proba[len(Zr):, 1]                       # P(fake) for the fakes, out-of-fold
    if method == "umap":
        # fit UMAP on the REAL PCA features, project fakes in (out-of-sample transform), score by
        # mean distance to the nearest real points IN the UMAP embedding.
        import umap
        from sklearn.neighbors import NearestNeighbors
        reducer = umap.UMAP(n_components=2, random_state=seed).fit(Zr)
        Er, Ef = reducer.embedding_, reducer.transform(Zf)
        nn = NearestNeighbors(n_neighbors=min(5, len(Er))).fit(Er)
        return nn.kneighbors(Ef)[0].mean(1)
    if method == "tsne":
        # t-SNE has no out-of-sample transform, so embed real+fake JOINTLY and score each fake by
        # its distance to the nearest real points in that shared t-SNE map.
        from sklearn.manifold import TSNE
        from sklearn.neighbors import NearestNeighbors
        Z = np.concatenate([Zr, Zf])
        E = TSNE(n_components=2, init="pca", perplexity=min(30, (len(Z) - 1) // 3),
                 random_state=seed).fit_transform(Z)
        Er, Ef = E[:len(Zr)], E[len(Zr):]
        nn = NearestNeighbors(n_neighbors=min(5, len(Er))).fit(Er)
        return nn.kneighbors(Ef)[0].mean(1)
    raise ValueError(f"unknown filter method: {method}")


def compare_filters(real_imgs, fake_imgs, keep_frac=0.66,
                    methods=("maha", "knn", "disc", "umap", "tsne"), seed=0):
    """Bake-off: for each filter, keep the top keep_frac of the fake pool and score that KEPT set's
    realism (FID/KID vs real + C2ST). Best filter = kept set closest to real (lowest FID/KID, C2ST
    nearest 0.5). Also reports the unfiltered pool as a baseline. Returns {name: {...}}."""
    n_keep = max(2, int(round(keep_frac * len(fake_imgs))))
    out = {}
    base = fid_kid(real_imgs, fake_imgs)
    out["(none)"] = {"n_keep": len(fake_imgs), "fid": base["fid"], "kid": base["kid"],
                     "c2st": c2st(real_imgs, fake_imgs)["c2st"]}
    for name in methods:
        s = rank_by_method(real_imgs, fake_imgs, method=name, seed=seed)
        keep_idx = torch.from_numpy(np.argsort(s)[:n_keep].copy())
        kept = fake_imgs[keep_idx]
        fk = fid_kid(real_imgs, kept)
        out[name] = {"n_keep": n_keep, "fid": fk["fid"], "kid": fk["kid"],
                     "c2st": c2st(real_imgs, kept)["c2st"]}
    return out


@torch.no_grad()
def save_projection(real_imgs, fake_imgs, out_png, title="", cap=500, seed=0):
    """PCA / t-SNE / UMAP scatter of real vs fake in Inception space, with the C2ST number in the
    title. Saves a PNG. UMAP is optional (skipped cleanly if umap-learn isn't installed)."""
    real_imgs, fake_imgs = _subsample(real_imgs, cap, seed), _subsample(fake_imgs, cap, seed)
    nr, nf = len(real_imgs), len(fake_imgs)
    if nr < 10 or nf < 10:
        return None
    fid = _new_fid()
    rf = _features(fid, real_imgs).numpy(); ff = _features(fid, fake_imgs).numpy()
    del fid; torch.cuda.empty_cache()
    X = np.concatenate([rf, ff]); y = np.array([0] * nr + [1] * nf)

    from sklearn.decomposition import PCA
    p50 = PCA(n_components=min(50, len(X) - 1), random_state=seed).fit_transform(X)
    embeds = {"PCA": p50[:, :2]}
    from sklearn.manifold import TSNE
    embeds["t-SNE"] = TSNE(n_components=2, init="pca",
                           perplexity=min(30, (len(X) - 1) // 3), random_state=seed).fit_transform(p50)
    try:
        import umap
        embeds["UMAP"] = umap.UMAP(n_components=2, random_state=seed).fit_transform(p50)
    except Exception as e:
        print(f"  [UMAP skipped: {e}]", flush=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    acc = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=seed),
                          p50, y, cv=5).mean()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(embeds)
    fig, ax = plt.subplots(1, n, figsize=(6.2 * n, 6))
    ax = np.atleast_1d(ax)
    for a, (name, e) in zip(ax, embeds.items()):
        a.scatter(e[y == 0, 0], e[y == 0, 1], s=8, c="#2b6cb0", alpha=.45, label=f"real ({nr})")
        a.scatter(e[y == 1, 0], e[y == 1, 1], s=8, c="#e53e3e", alpha=.45, label=f"fake ({nf})")
        a.set_title(name); a.set_xticks([]); a.set_yticks([]); a.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title}   |   C2ST = {acc:.2f}  (0.5 = indistinguishable)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png
