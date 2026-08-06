"""12-filter ranked comparison on the NEW (cropped, kimg1000) GAN. Cardiomegaly.

Reference design:
  select = the 7,047 cropped images the GAN trained on  (herding's target distribution)
  eval   = 9,617 device-free cardiomegaly train images that were NOT used for training,
           autocropped identically. The generator has never seen these, so FD/KID/coverage
           measure generalisation rather than reproduction of the training images.
           (Image-disjoint, not patient-disjoint -- the validate split would have been
           patient-disjoint but its source JPGs are not downloaded, only files/p10.)

Ranked by COVERAGE (herding does not optimise it); KID reported second (herding DOES optimise it).
"""
import os, sys, glob, csv, json, time, gc
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr\filters")
import numpy as np, torch
from PIL import Image
import pandas as pd

CKPT      = r"C:\Users\Tonkid\Downloads\runs\cardio_final_crop_p06\gen0\checkpoints\final.pt"
CKPT_FULL = r"C:\Users\Tonkid\Downloads\runs\cardio_final_crop_p06\gen0\checkpoints\latest.pt"   # has D
TRAIN_CROP= r"C:\Users\Tonkid\Downloads\tstr\data_cardio_final_crop\train\cardiomegaly"
FULL_TRAIN= r"C:\Users\Tonkid\Downloads\tstr\data_cardio_full\train\cardiomegaly"
EVAL_DIR  = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_eval_crop\cardiomegaly"
POOL_DIR  = r"C:\Users\Tonkid\Downloads\runs\cardio_final_crop_p06\gen0\pool100k\cardiomegaly"
OUT       = r"C:\Users\Tonkid\Downloads\results\filter_ranking_newgan.json"
N_POOL, N_KEEP, SEED = 100000, 7047, 0
N_VENDI, K_NN, MAHA_DROP, PSI = 2000, 5, 0.10, 0.7

from build_final_cropped import autocrop
import features as FE, herding as H, dpp as DPP, kernels as K
import eval_spaces as ES
from prdc_vendi import coverage_density, recall, vendi_score
from filter_bakeoff import load_G, gen_pool

# ---------------- Stage 1: build the held-out cropped eval reference ----------------
# The 9,617 device-free cardiomegaly images left over from 1:1 balancing -- the GAN never
# trained on them. Cropped identically so the only difference from the fakes is real-vs-fake.
if len(glob.glob(os.path.join(EVAL_DIR, "*.png"))) < 9000:
    print("building held-out eval reference ...", flush=True)
    used = {os.path.basename(p) for p in glob.glob(os.path.join(TRAIN_CROP, "*.png"))}
    kw = set()
    with open(r"C:\Users\Tonkid\Downloads\results\device_filter\report_device_hits.csv", newline="") as f:
        for r in csv.DictReader(f):
            kw.add(str(r["study_id"]))
    cx = pd.read_csv(r"C:\Users\Tonkid\Downloads\mimic_cxr_cardiomegaly\mimic-cxr-2.0.0-chexpert.csv.gz")
    nb = pd.read_csv(r"C:\Users\Tonkid\Downloads\mimic_cxr_cardiomegaly\mimic-cxr-2.0.0-negbio.csv.gz")
    drop = kw | set(cx[cx["Support Devices"] == 1.0]["study_id"].astype(str)) \
              | set(nb[nb["Support Devices"] == 1.0]["study_id"].astype(str))
    held = [p for p in sorted(glob.glob(os.path.join(FULL_TRAIN, "*.png")))
            if os.path.basename(p).split("_")[1] not in drop and os.path.basename(p) not in used]
    os.makedirs(EVAL_DIR, exist_ok=True)
    for i, src in enumerate(held):
        g = Image.open(src).convert("L")
        autocrop(g).resize((256, 256), Image.LANCZOS).save(os.path.join(EVAL_DIR, os.path.basename(src)))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(held)}", flush=True)
    print(f"eval reference built: {len(held)} imgs", flush=True)

# select = all 7,047 cropped reals the GAN trained on (herding's target distribution)
# eval   = 9,617 device-free cardiomegaly the GAN NEVER trained on (balancing leftovers).
# Scoring against images the generator has never seen measures generalisation, not reproduction.
sel_paths  = sorted(glob.glob(os.path.join(TRAIN_CROP, "*.png")))
eval_paths = sorted(glob.glob(os.path.join(EVAL_DIR, "*.png")))
assert not ({os.path.basename(p) for p in sel_paths} &
            {os.path.basename(p) for p in eval_paths}), "LEAKAGE: select/eval overlap"
print(f"select {len(sel_paths)} (GAN-trained) | eval {len(eval_paths)} (held out, unseen) "
      f"- disjoint OK", flush=True)

# ---------------- Stage 2: generate the 100k pool ----------------
os.makedirs(POOL_DIR, exist_ok=True)
have = len(glob.glob(os.path.join(POOL_DIR, "*.png")))
if have < N_POOL:
    print(f"generating pool: have {have}, need {N_POOL} ...", flush=True)
    G, classes, c_dim, cfg = load_G(CKPT)
    ci = classes.index("cardiomegaly")
    t0 = time.time()
    B = 250
    for start in range(have, N_POOL, B):
        n = min(B, N_POOL - start)
        imgs = gen_pool(G, cfg, c_dim, ci, n, batch=50, seed=1000 + start)
        for j in range(n):
            Image.fromarray(imgs[j].permute(1, 2, 0).numpy()).convert("L").save(
                os.path.join(POOL_DIR, f"{start + j:06d}.png"))
        if (start // B) % 20 == 0:
            el = time.time() - t0
            print(f"  {start + n}/{N_POOL}  ({el/60:.1f} min)", flush=True)
    del G; torch.cuda.empty_cache()
pool_paths = sorted(glob.glob(os.path.join(POOL_DIR, "*.png")))[:N_POOL]
print(f"pool ready: {len(pool_paths)}", flush=True)

# ---------------- Stage 3: features ----------------
t0 = time.time()
pool_feat = FE.extract_features(pool_paths, backbone="inception")
sel_feat  = FE.extract_features(sel_paths,  backbone="inception")
eval_feat = FE.extract_features(eval_paths, backbone="inception")
print(f"features ready in {(time.time()-t0)/60:.1f} min  {pool_feat.shape}", flush=True)

# ---------------- Stage 4: discriminator scores (rows DRS + top-k) ----------------
print("scoring pool with D ...", flush=True)
from stylegan2_ada_cond.models.networks import Discriminator
ckf = torch.load(CKPT_FULL, map_location="cpu", weights_only=False)
c = ckf["config"]
D = Discriminator(img_resolution=c["resolution"], img_channels=ckf["img_channels"],
                  channel_base=c["channel_base"], channel_max=c["channel_max"],
                  c_dim=int(ckf["c_dim"])).eval().cuda()
D.load_state_dict(ckf["D"]); ci_lbl = list(ckf["classes"]).index("cardiomegaly")
d_logits = np.empty(len(pool_paths), np.float32)
with torch.no_grad():
    for i in range(0, len(pool_paths), 128):
        # load_paths_uint8 returns (N,3,H,W); D was built with ckf["img_channels"] channels
        # (3 here, not 1) -- slice to match rather than assuming grayscale.
        batch = (FE.load_paths_uint8(pool_paths[i:i + 128])[:, :int(ckf["img_channels"])]
                 .float().cuda() / 127.5 - 1.0)
        lab = torch.zeros(batch.shape[0], int(ckf["c_dim"]), device="cuda"); lab[:, ci_lbl] = 1.0
        d_logits[i:i + batch.shape[0]] = D(batch, lab).squeeze(1).float().cpu().numpy()
        if i % 25600 == 0: print(f"  D {i}/{len(pool_paths)}", flush=True)
del D, ckf; torch.cuda.empty_cache()
print("D scores done", flush=True)

# ---------------- Stage 5: build all rows ----------------
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=SEED).fit(sel_feat)
Zr, Zf = pca.transform(sel_feat), pca.transform(pool_feat)
rng = np.random.default_rng(SEED)
kept, extra_feat = {}, {}

kept["none (unfiltered)"] = rng.choice(len(pool_paths), min(25000, len(pool_paths)), replace=False)
kept["random (control)"]  = np.random.default_rng(SEED).choice(len(pool_paths), N_KEEP, replace=False)

std = Zr.std(0) + 1e-6
maha = np.sqrt((((Zf - Zr.mean(0)) / std) ** 2).sum(1))
kept["PCA-Mahalanobis"] = np.argsort(maha)[:N_KEEP]

from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5).fit(Zr)
kept["PCA-kNN"] = np.argsort(nn.kneighbors(Zf)[0].mean(1))[:N_KEEP]

def kcenter(F, n_keep, seed=SEED):
    X = torch.as_tensor(F, dtype=torch.float32, device="cuda")
    sel = [int(np.random.default_rng(seed).integers(X.shape[0]))]
    d = torch.cdist(X, X[sel[0]:sel[0] + 1]).squeeze(1)
    for _ in range(n_keep - 1):
        j = int(torch.argmax(d)); sel.append(j)
        d = torch.minimum(d, torch.cdist(X, X[j:j + 1]).squeeze(1))
    del X; torch.cuda.empty_cache()
    return np.asarray(sel)
t = time.time(); kept["k-center greedy"] = kcenter(pool_feat, N_KEEP)
print(f"  k-center ({time.time()-t:.0f}s)", flush=True)

from sklearn.cluster import MiniBatchKMeans
t = time.time()
km = MiniBatchKMeans(n_clusters=200, random_state=SEED, n_init=3, batch_size=4096).fit(Zf)
per = N_KEEP // 200; pick = []
for c_ in range(200):
    idx = np.where(km.labels_ == c_)[0]
    pick.append(rng.choice(idx, min(per, len(idx)), replace=False))
pick = np.concatenate(pick)
if len(pick) < N_KEEP:
    rest = np.setdiff1d(np.arange(len(pool_paths)), pick)
    pick = np.concatenate([pick, rng.choice(rest, N_KEEP - len(pick), replace=False)])
kept["k-means balanced"] = pick[:N_KEEP]
print(f"  k-means ({time.time()-t:.0f}s)", flush=True)

t = time.time()
sig = K.median_sigma(torch.as_tensor(sel_feat, dtype=torch.float64))
loc, _ = DPP.greedy_dpp_select(pool_feat, sel_feat, N_KEEP, kernel="rbf",
                               kernel_kwargs={"sigma": sig}, quality="real_affinity", log_every=3000)
kept["DPP"] = np.asarray(loc); print(f"  DPP ({time.time()-t:.0f}s)", flush=True)

t = time.time()
loc, dh = H.greedy_mmd_select(pool_feat, sel_feat, N_KEEP, kernel="poly",
                              kernel_kwargs={"degree": 3}, device="cuda", log_every=3000)
kept["herding"] = np.asarray(loc); print(f"  herding ({time.time()-t:.0f}s)", flush=True)

# DRS -- density-ratio resampling (documented simplification of Azadi et al.)
w = np.exp(d_logits - d_logits.max()); w = w / w.sum()
kept["DRS (density-ratio)"] = rng.choice(len(pool_paths), N_KEEP, replace=False, p=w)
kept["top-k by D score"] = np.argsort(-d_logits)[:N_KEEP]

t = time.time()
loc, _ = H.greedy_mmd_select(pool_feat[surv := np.argsort(maha)[:int(len(maha) * (1 - MAHA_DROP))]],
                             sel_feat, N_KEEP, kernel="poly", kernel_kwargs={"degree": 3},
                             device="cuda", log_every=3000)
kept["PCA-Mahalanobis 10% + herding"] = surv[np.asarray(loc)]
print(f"  maha10+herding ({time.time()-t:.0f}s)", flush=True)

# truncation psi=0.7 -- generated fresh, NOT a subset of the pool
t = time.time()
G, classes, c_dim, cfg = load_G(CKPT)
ci = classes.index("cardiomegaly")
import torch as _t
outs = []
with _t.no_grad():
    for i in range(0, N_KEEP, 50):
        n = min(50, N_KEEP - i)
        z = _t.randn(n, cfg["z_dim"], device="cuda")
        lab = _t.zeros(n, c_dim, device="cuda"); lab[:, ci] = 1.0
        im = G(z, lab, truncation_psi=PSI, noise_mode="random")
        outs.append(((im.clamp(-1, 1) + 1) * 127.5).round().to(_t.uint8).cpu())
trunc_imgs = _t.cat(outs)[:N_KEEP]
del G, outs; _t.cuda.empty_cache()
import cardio_metrics as M
fo = M._new_fid(); extra_feat["truncation psi=0.7"] = M._features(fo, trunc_imgs).numpy()
del fo, trunc_imgs; _t.cuda.empty_cache()
print(f"  truncation ({time.time()-t:.0f}s)", flush=True)

# ---------------- Stage 6: score ----------------
def copy_rate_feat(ref, fake, pct=5):
    R = torch.as_tensor(ref, dtype=torch.float32, device="cuda")
    F = torch.as_tensor(np.asarray(fake), dtype=torch.float32, device="cuda")
    d = torch.cdist(R, R); d.fill_diagonal_(float("inf"))
    tau = torch.quantile(d.min(1).values.float(), pct / 100.0)
    out = float((torch.cdist(F, R).min(1).values < tau).float().mean())
    del R, F, d; torch.cuda.empty_cache()
    return out

print("\nscoring rows ...", flush=True)
rows = []
names = list(kept.keys()) + list(extra_feat.keys())
for name in names:
    if name in extra_feat:
        F = extra_feat[name]; n = len(F)
    else:
        idx = np.asarray(kept[name])
        if len(idx) > N_KEEP:
            idx = idx[np.random.default_rng(SEED).choice(len(idx), N_KEEP, replace=False)]
        F = pool_feat[idx]; n = len(idx)
    fd = ES.frechet_distance(eval_feat, F)
    kd = ES.kid_from_feats(eval_feat, F, subset_size=1000, subsets=100, seed=SEED)
    cd = coverage_density(eval_feat, F, k=K_NN)
    rc = recall(eval_feat, F, k=K_NN)
    vi = np.random.default_rng(SEED).choice(len(F), min(N_VENDI, len(F)), replace=False)
    vd = vendi_score(F[vi], kernel="cosine")
    cr = copy_rate_feat(eval_feat, F)
    rows.append({"name": name, "n": int(n), "fd": round(fd, 3), "kid": kd["kid"],
                 "kid_std": kd["kid_std"], "coverage": cd["coverage"], "density": cd["density"],
                 "recall": rc["recall"], "vendi": vd["vendi"], "copy_rate": round(cr, 4)})
    print(f"  {name:32s} FD {fd:7.2f}  KID {kd['kid']:+.5f}  cov {cd['coverage']:.4f}  "
          f"copy {cr:.3f}", flush=True)
    gc.collect()

rows.sort(key=lambda r: -r["coverage"])
print(f"\n=== filter ranking | NEW cropped GAN kimg1000 | cardiomegaly | n={N_KEEP} | by COVERAGE ===")
print(f"{'#':>2} {'filter':32s} {'coverage':>9s} {'KID':>10s} {'FD':>7s} {'density':>8s} "
      f"{'recall':>7s} {'vendi':>7s} {'copy':>6s}")
print("-" * 102)
for i, r in enumerate(rows, 1):
    print(f"{i:2d} {r['name']:32s} {r['coverage']:9.4f} {r['kid']:+10.5f} {r['fd']:7.2f} "
          f"{r['density']:8.4f} {r['recall']:7.4f} {r['vendi']:7.1f} {r['copy_rate']:6.3f}")

json.dump({"gan": CKPT, "class": "cardiomegaly", "n_keep": N_KEEP, "n_pool": len(pool_paths),
           "select_n": len(sel_paths), "eval_n": len(eval_paths), "rank_by": "coverage",
           "eval_note": "held-out device-free train images the GAN never trained on (image-disjoint)",
           "rows": rows}, open(OUT, "w", encoding="utf-8"), indent=2)
print("\nsaved ->", OUT)
