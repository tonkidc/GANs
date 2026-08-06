"""Feature extraction + cache for the herding filter.

The whole point of Step 1: herd in the SAME feature space KID scores in. We do NOT build a fresh
Inception — we reuse cardio_metrics._new_fid().inception, which IS torchmetrics' own frozen
Inception-v3 (the net FID/KID use with normalize=False, uint8 in). So features extracted here and
KID computed by torchmetrics are in the same space by construction; verify_matches_kid() spot-checks
that to <1e-4.

Cache is keyed by a hash of (backbone, exact sorted file list) so nothing re-extracts needlessly.
"""
import os, sys, glob, json, hashlib
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import cardio_metrics as M

CACHE_DIR = r"C:\Users\Tonkid\Downloads\tstr\filters\feat_cache"


def load_paths_uint8(paths):
    """Explicit path list -> uint8 (N,3,256,256), RGB, matching cardio_metrics.load_uint8."""
    if not paths:
        return torch.empty(0, 3, 256, 256, dtype=torch.uint8)
    arr = np.stack([np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def _key(backbone, paths):
    h = hashlib.sha1(backbone.encode())
    for p in paths:
        h.update(b"\0"); h.update(p.encode())
    return h.hexdigest()[:16]


def extract_features(paths, backbone="inception", out_path=None, batch_size=64, cache_dir=CACHE_DIR):
    """Return (N, d) float32 features for `paths`. Cached to .npy keyed by (backbone, file list)."""
    paths = list(paths)
    if not paths:
        raise ValueError("extract_features: empty path list")
    if out_path is None:
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, f"{backbone}_{_key(backbone, paths)}.npy")
    listp = out_path + ".files.json"
    if os.path.exists(out_path) and os.path.exists(listp) and json.load(open(listp))["paths"] == paths:
        return np.load(out_path)

    if backbone == "inception":
        F = _inception_features(paths, batch_size)
    elif backbone == "xrv":
        F = _xrv_features(paths, batch_size)
    elif backbone == "inception_v4":
        F = _inception_v4_features(paths, batch_size)
    elif backbone == "dinov2":
        F = _dinov2_features(paths, batch_size)
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    np.save(out_path, F)
    json.dump({"backbone": backbone, "n": len(paths), "paths": paths}, open(listp, "w"))
    return F


@torch.no_grad()
def _inception_features(paths, batch_size):
    """2048-d features via torchmetrics' own Inception (the exact net KID uses)."""
    fid = M._new_fid()
    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = load_paths_uint8(paths[i:i + batch_size])
        feats.append(M._features(fid, imgs, batch=batch_size).numpy())
    del fid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(feats).astype(np.float32)


@torch.no_grad()
def _xrv_features(paths, batch_size):
    """1024-d TorchXRayVision DenseNet-121 features (device-robust, medical-domain). Uses XRV's own
    preprocessing: grayscale -> normalize to [-1024,1024] -> resize 224. Different from Inception's
    (uint8, 299, ImageNet) on purpose — the point of the cross-space experiment."""
    import torchxrayvision as xrv
    import torchvision
    import torch.nn.functional as Fn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval().to(dev)
    resize = torchvision.transforms.Compose([xrv.datasets.XRayResizer(224)])
    feats = []
    for i in range(0, len(paths), batch_size):
        xs = []
        for p in paths[i:i + batch_size]:
            a = np.asarray(Image.open(p).convert("L"), np.float32)     # 0..255 grayscale
            a = xrv.datasets.normalize(a, 255)[None, ...]              # (1,H,W) -> [-1024,1024]
            xs.append(resize(a))                                       # (1,224,224)
        X = torch.from_numpy(np.stack(xs)).to(dev)                     # (b,1,224,224)
        try:
            f = model.features2(X)                                     # (b,1024)
        except AttributeError:
            c = Fn.relu(model.features(X))
            f = Fn.adaptive_avg_pool2d(c, (1, 1)).flatten(1)
        feats.append(f.cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(feats).astype(np.float32)


@torch.no_grad()
def _inception_v4_features(paths, batch_size):
    """1536-d Inception-v4 (ImageNet, timm) global-pool features. A DEEPER ImageNet net than the v3 KID
    uses — used ONLY as an alternative herding SELECTION space (never as the metric). Same domain as v3
    (ImageNet, not chest), so this row tests whether depth alone buys a better selection. Uses timm's own
    eval transform (resize 299 + inception normalize)."""
    import timm
    from timm.data import resolve_model_data_config, create_transform
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("inception_v4", pretrained=True, num_classes=0).eval().to(dev)
    tf = create_transform(**resolve_model_data_config(model), is_training=False)
    feats = []
    for i in range(0, len(paths), batch_size):
        xs = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]
        X = torch.stack(xs).to(dev)
        feats.append(model(X).cpu().numpy())               # (b,1536) pooled features
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(feats).astype(np.float32)


@torch.no_grad()
def _dinov2_features(paths, batch_size):
    """768-d DINOv2 ViT-B/14 (self-supervised, LVD-142M) CLS features, via timm.

    Used as an EVALUATION space, not a selection space. Two reasons it earns its place:
      1) Inception-based FID tracks human judgement poorly (Stein et al., NeurIPS 2023); DINOv2
         features are a markedly better perceptual space for generative evaluation.
      2) ANTI-GAMING CONTROL. Herding minimises MMD in Inception-v3 — the same space Inception-KID
         scores in — so 'it wins on KID' is partly teaching to the test. DINOv2 is a space the
         selector never optimised in, so a win here is evidence the gain is real.
    Uses timm's own eval transform for this checkpoint (resize + ImageNet normalize)."""
    import timm
    from timm.data import resolve_model_data_config, create_transform
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0).eval().to(dev)
    tf = create_transform(**resolve_model_data_config(model), is_training=False)
    feats = []
    for i in range(0, len(paths), batch_size):
        xs = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]
        X = torch.stack(xs).to(dev)
        feats.append(model(X).float().cpu().numpy())        # (b,768) CLS/pooled features
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(feats).astype(np.float32)


@torch.no_grad()
def verify_matches_kid(paths, n=200):
    """Sanity gate: our cached features must be the SAME features KID/FID consume. We prove it directly
    — extract_features() vs a fresh cardio_metrics._features() call on the same images must be identical
    (both are the FID object's frozen Inception). Identical features => identical KID by construction,
    which is a stronger guarantee than matching a resampled KID to 1e-4."""
    sub = list(paths)[:n]
    fid = M._new_fid()
    direct = M._features(fid, load_paths_uint8(sub)).numpy()
    del fid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ours = extract_features(sub)
    diff = float(np.abs(ours - direct).max())
    print(f"feature max|diff| vs torchmetrics inception = {diff:.2e}  "
          f"{'OK (same KID space)' if diff < 1e-4 else 'MISMATCH'}", flush=True)
    return diff
