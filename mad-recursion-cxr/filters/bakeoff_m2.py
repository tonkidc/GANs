"""Bake-off Milestone 2 — big pool + cross-space selection, with kept sets materialised for the probe.

Adds to Milestone 1:
  * PoolS = 50,000/class (generate_pool.py) -> 3,300 kept = 6.6%, so herding stays selective.
  * Cross-space herding: select on TorchXRayVision (device-robust) features, score KID on Inception.
    The selector cannot game a space it never sees; this also attacks the device confounder.
  * Materialises each kept set (BOTH classes) to a folder so the TSTR probe can train on it
    (see probe_eval.py — the downstream recall test that decides whether herding preserves the disease).

KID/FID are reported for the cardiomegaly class vs its held-out eval half. The five Milestone-1
filters are shown as reference rows (they were selected on the OLD 5k pool — not directly comparable
at 50k, but useful context).

Run (GPU inference, minutes; needs generate_pool.py first):
    python tstr/filters/bakeoff_m2.py                 # inception + xrv cross-space + random
    python tstr/filters/bakeoff_m2.py --no-xrv        # skip the cross-space row
"""
import os, sys, json, time, glob, shutil, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import cardio_metrics as M
import real_split
import features as FE
import herding as H
import dpp as DPP
import kernels as K

POOL_DIR  = r"C:\Users\Tonkid\Downloads\runs\cardio_mad_A\gen0\pool50k"
BAKE_DIR  = r"C:\Users\Tonkid\Downloads\results\filter_bakeoff"          # M1 kept sets (reference)
KEPT_ROOT = r"C:\Users\Tonkid\Downloads\results\filter_bakeoff_v2\kept"  # materialised M2 kept sets
OUT_DIR   = r"C:\Users\Tonkid\Downloads\results\filter_bakeoff_m2"
CLASSES   = ["cardiomegaly", "normal"]
PRIMARY   = "cardiomegaly"
EXISTING  = {"maha": "PCA-Mahalanobis", "knn": "PCA-kNN", "disc": "C2ST (discriminator)",
             "umap": "UMAP", "tsne": "t-SNE"}


def _pngs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))


def _materialise(paths, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for p in paths:
        d = os.path.join(dst_dir, os.path.basename(p))
        if not os.path.exists(d):
            shutil.copy2(p, d)


def _select(tag, space, pool_feats, sel_feats, pool_paths, n_keep, seed, rng):
    """Return (kept_indices, diag) for one class. space in {'random','inception','xrv'}."""
    if tag == "random":
        return rng.choice(len(pool_paths), size=n_keep, replace=False), {"kind": "random"}
    F_pool, F_sel = pool_feats[space], sel_feats[space]
    if tag.startswith("dpp"):                             # diversity objective (RBF, real-affinity quality)
        sigma = K.median_sigma(torch.as_tensor(F_sel, dtype=torch.float64))
        return DPP.greedy_dpp_select(F_pool, F_sel, n_keep, kernel="rbf",
                                     kernel_kwargs={"sigma": sigma}, quality="real_affinity", log_every=500)
    kw = {"degree": 3}                                    # herding: distribution-match objective (KID's poly)
    idx, diag = H.greedy_mmd_select(F_pool, F_sel, n_keep, kernel="poly", kernel_kwargs=kw, log_every=500)
    return idx, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=POOL_DIR)
    ap.add_argument("--n-keep", type=int, default=3300)
    ap.add_argument("--no-xrv", action="store_true", help="skip the cross-space (XRV) herding row")
    ap.add_argument("--kept-root", default=KEPT_ROOT)
    a = ap.parse_args()
    use_xrv = not a.no_xrv
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    spaces = ["inception"] + (["xrv"] if use_xrv else [])
    # rows: (tag, feature space used for selection)
    row_specs = [("herding-poly-inception", "inception"),
                 ("dpp-inception", "inception"),          # diversity counterpart, same feature space
                 ("random", "random")]
    if use_xrv:
        row_specs.insert(2, ("herding-poly-xrv", "xrv"))

    # ---- per-class: paths, splits, features ----
    pool_paths, sel_paths, eval_paths = {}, {}, {}
    pool_feats, sel_feats = {c: {} for c in CLASSES}, {c: {} for c in CLASSES}
    for cls in CLASSES:
        pool_paths[cls] = _pngs(os.path.join(a.pool_dir, cls))
        if len(pool_paths[cls]) < a.n_keep:
            raise RuntimeError(f"{cls}: pool has {len(pool_paths[cls])} < n_keep {a.n_keep} "
                               f"— run generate_pool.py first ({a.pool_dir})")
        real_split.assert_disjoint(cls)
        sel_paths[cls] = real_split.load_split("select", cls)
        eval_paths[cls] = real_split.load_split("eval", cls)
        print(f"[{cls}] pool={len(pool_paths[cls])} real select={len(sel_paths[cls])} "
              f"eval={len(eval_paths[cls])}", flush=True)
        for sp in spaces:
            print(f"  features[{sp}]...", flush=True)
            pool_feats[cls][sp] = FE.extract_features(pool_paths[cls], backbone=sp)
            sel_feats[cls][sp]  = FE.extract_features(sel_paths[cls], backbone=sp)

    FE.verify_matches_kid(pool_paths[PRIMARY], n=200)   # inception space sanity

    # ---- select + materialise each row (both classes) ----
    rng = np.random.default_rng(0)
    diagnostics = {}
    for tag, space in row_specs:
        print(f"\n=== {tag} ===", flush=True)
        for cls in CLASSES:
            idx, diag = _select(tag, space, pool_feats[cls], sel_feats[cls],
                                pool_paths[cls], a.n_keep, seed=0, rng=rng)
            kept = [pool_paths[cls][i] for i in idx]
            _materialise(kept, os.path.join(a.kept_root, tag, cls))
            if cls == PRIMARY:
                diagnostics[tag] = diag
                if diag.get("logdet_final") is not None:          # DPP: diversity objective
                    print(f"  [{cls}] logdet {diag['logdet_final']:.4g}  MMD^2_end {diag['mmd2_end']:.4g}  "
                          f"n_diverse {diag.get('n_diverse')}  ({diag['wall_time_s']}s)", flush=True)
                elif diag.get("mmd2_start") is not None:          # herding: distribution-match objective
                    print(f"  [{cls}] MMD^2 {diag['mmd2_start']:.4g} -> {diag['mmd2_end']:.4g} "
                          f"({diag['wall_time_s']}s)", flush=True)

    # ---- score PRIMARY class vs its held-out eval half ----
    real_eval = FE.load_paths_uint8(eval_paths[PRIMARY])
    print(f"\nscoring {PRIMARY} vs held-out eval (n={len(eval_paths[PRIMARY])}):", flush=True)
    rows = []

    def score(name, kind, paths):
        fk = M.fid_kid(real_eval, FE.load_paths_uint8(paths))
        print(f"  {name:26s} n={len(paths):5d}  KID {fk['kid']:+.5f} (+/-{fk['kid_std']:.5f})  "
              f"FID {fk['fid']:7.2f}", flush=True)
        return {"name": name, "kind": kind, "n": len(paths),
                "fid": fk["fid"], "kid": fk["kid"], "kid_std": fk["kid_std"]}

    # unfiltered reference (native pool n)
    rows.append(score("none (unfiltered pool)", "reference", pool_paths[PRIMARY]))
    for tag, _ in row_specs:
        kind = "control" if tag == "random" else ("dpp" if tag.startswith("dpp") else "herding")
        rows.append(score(tag, kind, _pngs(os.path.join(a.kept_root, tag, PRIMARY))))
    # M1 filters as reference (old 5k pool — noted)
    for t, pretty in EXISTING.items():
        kp = _pngs(os.path.join(BAKE_DIR, t, PRIMARY))
        if kp:
            rows.append(score(f"{pretty} (M1 pool)", "reference-m1", kp))

    ranked = sorted([r for r in rows if r["kind"] in ("herding", "dpp", "control")], key=lambda r: r["kid"])
    print("\n=== herding vs DPP vs random on the 50k pool, KID vs eval (lower better) ===")
    for r in ranked:
        print(f"  {r['name']:26s} KID {r['kid']:+.5f}  FID {r['fid']:7.2f}", flush=True)

    out = {
        "config": {"pool_dir": a.pool_dir, "n_pool": len(pool_paths[PRIMARY]), "n_keep": a.n_keep,
                   "primary": PRIMARY, "spaces": spaces, "kept_root": a.kept_root,
                   "real_eval": len(eval_paths[PRIMARY]), "wall_time_s": round(time.time() - t0, 1)},
        "note": "Kept sets for BOTH classes are materialised under kept_root/<tag>/<cls>/ for probe_eval.py "
                "(the TSTR recall test). KID/FID here are cardiomegaly vs its held-out eval half. The M1 "
                "filter rows used the OLD 5k pool and are reference only.",
        "diagnostics": diagnostics,
        "rows": rows,
        "ranked_by_kid": [r["name"] for r in ranked],
    }
    json.dump(out, open(os.path.join(OUT_DIR, "realism_rank_m2.json"), "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'realism_rank_m2.json')}"
          f"\nkept sets -> {a.kept_root}/<tag>/<cls>/  (feed to probe_eval.py)", flush=True)


if __name__ == "__main__":
    main()
