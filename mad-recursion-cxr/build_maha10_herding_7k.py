"""PCA-Mahalanobis (drop 10%) -> herding, keep 7,047/class, BOTH classes.

7,047 matches the real device-filtered set size exactly (data_cardio_final_crop), so gen-1 trains on
the same per-class count as gen-0 did.

Exports the kept images as hardlinks into a train/ folder (ready for training) and saves a 100-image
montage per class.
"""
import os, sys, glob, time, shutil
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr\filters")
import numpy as np
from PIL import Image
import features as FE, real_split, gates as G, herding as H

POOL_DIR  = r"C:\Users\Tonkid\Downloads\runs\cardio_mad_A\gen0\pool100k"
OUTROOT   = r"C:\Users\Tonkid\Downloads\tstr\data_gen1_maha10_herding\train"
MAHA_DROP = 0.10
N_KEEP    = 7047          # match the real set size per class
CLASSES   = ["cardiomegaly", "normal"]
SEED      = 0

summary = {}
for cls in CLASSES:
    print(f"\n=== {cls} ===", flush=True)
    pool_paths = sorted(glob.glob(os.path.join(POOL_DIR, cls, "*.png")))
    real_split.assert_disjoint(cls)
    sel_paths = real_split.load_split("select", cls)
    print(f"pool {len(pool_paths)} | real select {len(sel_paths)}", flush=True)

    pool_feat = FE.extract_features(pool_paths, backbone="inception")   # cached
    sel_feat  = FE.extract_features(sel_paths,  backbone="inception")   # cached

    # --- PCA-Mahalanobis gate: drop the 10% farthest from the real mean ---
    keep, info = G.q_maha(pool_feat, sel_feat, drop_frac=MAHA_DROP, seed=SEED)
    surv = np.where(keep)[0]
    print(f"  maha 10%: dropped {(~keep).sum()} -> {len(surv)} survive", flush=True)

    # --- herding on the survivors ---
    t0 = time.time()
    loc, diag = H.greedy_mmd_select(pool_feat[surv], sel_feat, N_KEEP, kernel="poly",
                                    kernel_kwargs={"degree": 3}, device="cuda", log_every=2000)
    kept_idx = surv[np.asarray(loc)]
    print(f"  herding: kept {len(kept_idx)}  mmd2 {diag.get('mmd2_start')} -> "
          f"{diag.get('mmd2_end')}  ({time.time()-t0:.0f}s)", flush=True)

    # --- export as hardlinks (ready to train on) ---
    d = os.path.join(OUTROOT, cls)
    os.makedirs(d, exist_ok=True)
    for j in np.sort(kept_idx):
        src = pool_paths[j]
        dst = os.path.join(d, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    print(f"  exported -> {d}", flush=True)

    # --- 100-image montage ---
    rng = np.random.default_rng(5)
    pick = [pool_paths[j] for j in rng.choice(kept_idx, 100, replace=False)]
    CELL, GAP = 200, 4
    W = 10 * CELL + 11 * GAP
    out = Image.new("L", (W, W), 32)
    for i, p in enumerate(pick):
        im = Image.open(p).convert("L").resize((CELL, CELL))
        r, c = divmod(i, 10)
        out.paste(im, (GAP + c * (CELL + GAP), GAP + r * (CELL + GAP)))
    dst = rf"C:\Users\Tonkid\Downloads\MAHA10_HERDING_7k_{cls}_100.png"
    out.save(dst)
    print(f"  montage -> {dst}", flush=True)
    summary[cls] = {"kept": int(len(kept_idx)), "survivors": int(len(surv)),
                    "mmd2_end": diag.get("mmd2_end")}
    np.save(rf"C:\Users\Tonkid\Downloads\results\maha10_herding_7k_{cls}_idx.npy", kept_idx)

print("\n=== done ===")
for cls, s in summary.items():
    print(f"  {cls:14s} kept {s['kept']}  (from {s['survivors']} survivors)  mmd2_end {s['mmd2_end']}")
print("\ntraining set ->", OUTROOT)
