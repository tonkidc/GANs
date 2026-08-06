"""Capped (p_max=0.6) vs fully-adaptive (p_max=1.0) ADA, matched at kimg 600.

Identical data (data_cardio_final_crop, 7,047/class) and identical config except p_max, so this
isolates the augmentation cap. Scored against the CROPPED real set (the distribution both models
were trained to match) at n=2,500/class.

Reports FID + KID (Inception) and coverage/density/recall/vendi, so a quality change and a
diversity change can be told apart -- the leak shows up as edge/translation artifacts that FID may
partly absorb but coverage reads differently.
"""
import os, sys, glob, json, gc
os.environ.setdefault("MIOPEN_FIND_MODE", "NORMAL")
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr\filters")
import numpy as np, torch
import cardio_metrics as M
import features as FE
from filter_bakeoff import load_G, gen_pool
from prdc_vendi import coverage_density, recall, vendi_score

DATA  = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_final_crop\train"
KIMG, FID_N, K_NN, N_VENDI, SEED = 600, 2500, 5, 2000, 0
RUNS = {
    "capped  (p_max=0.6)":   rf"C:\Users\Tonkid\Downloads\runs\cardio_final_crop_p06\gen0\checkpoints\kimg{KIMG:05d}.pt",
    "adaptive (p_max=1.0)":  rf"C:\Users\Tonkid\Downloads\runs\cardio_final_crop_padaptive\gen0\checkpoints\kimg{KIMG:05d}.pt",
}
OUT = r"C:\Users\Tonkid\Downloads\results\capped_vs_adaptive_kimg600.json"

# ---- real reference: the CROPPED set both models were trained on ----
real_paths = []
for c in ("cardiomegaly", "normal"):
    real_paths += sorted(glob.glob(os.path.join(DATA, c, "*.png")))[:FID_N]
real_imgs = FE.load_paths_uint8(real_paths)
print(f"real reference (CROPPED): {len(real_imgs)} imgs", flush=True)

fid_obj = M._new_fid()
real_feat = M._features(fid_obj, real_imgs).numpy()

rows = []
for name, ck in RUNS.items():
    assert os.path.exists(ck), f"missing {ck}"
    G, classes, c_dim, cfg = load_G(ck)
    fake = torch.cat([gen_pool(G, cfg, c_dim, ci, FID_N) for ci in range(len(classes))])
    del G; torch.cuda.empty_cache()

    fk = M.fid_kid(real_imgs, fake)
    fake_feat = M._features(fid_obj, fake).numpy()
    cd = coverage_density(real_feat, fake_feat, k=K_NN)
    rc = recall(real_feat, fake_feat, k=K_NN)
    vi = np.random.default_rng(SEED).choice(len(fake_feat), min(N_VENDI, len(fake_feat)), replace=False)
    vd = vendi_score(fake_feat[vi], kernel="cosine")

    rows.append({"name": name, "kimg": KIMG, "n": int(len(fake)), "fid": fk["fid"],
                 "kid": fk["kid"], "kid_std": fk["kid_std"], "coverage": cd["coverage"],
                 "density": cd["density"], "recall": rc["recall"], "vendi": vd["vendi"]})
    print(f"  {name:22s} FID {fk['fid']:7.2f}  KID {fk['kid']:+.5f} +/-{fk['kid_std']:.5f}  "
          f"coverage {cd['coverage']:.4f}  density {cd['density']:.4f}  vendi {vd['vendi']:.1f}",
          flush=True)
    del fake, fake_feat; gc.collect(); torch.cuda.empty_cache()

print(f"\n=== capped vs adaptive @ kimg {KIMG} (same data, same config, only p_max differs) ===")
print(f"{'run':22s} {'FID':>7s} {'KID':>10s} | {'coverage':>9s} {'density':>8s} {'recall':>7s} {'vendi':>8s}")
print("-" * 84)
for r in rows:
    print(f"{r['name']:22s} {r['fid']:7.2f} {r['kid']:+10.5f} | {r['coverage']:9.4f} "
          f"{r['density']:8.4f} {r['recall']:7.4f} {r['vendi']:8.1f}")

if len(rows) == 2:
    cap, ada = rows[0], rows[1]
    print(f"\n  FID      {cap['fid']:.2f} -> {ada['fid']:.2f}   ({ada['fid']-cap['fid']:+.2f})")
    print(f"  KID      {cap['kid']:+.5f} -> {ada['kid']:+.5f}   ({ada['kid']-cap['kid']:+.5f}, "
          f"err +/-{cap['kid_std']:.5f})")
    print(f"  coverage {cap['coverage']:.4f} -> {ada['coverage']:.4f}   "
          f"({ada['coverage']-cap['coverage']:+.4f})")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"kimg": KIMG, "fid_ref": f"data_cardio_final_crop (CROPPED), {FID_N}/class", "rows": rows},
          open(OUT, "w", encoding="utf-8"), indent=2)
print("\nsaved ->", OUT)
