"""Bake-off v2 — does set-aware herding beat the per-sample filters and a random control on KID?

Additive: touches no existing filter and no existing results file. Writes
results/filter_bakeoff_v2/realism_rank_v2.json + a markdown table to stdout.

Setup (per the milestone):
  * Unfiltered pool  = the existing 5,000 gen-0 cardiomegaly fakes (runs/cardio_mad_A/gen0/fakes).
  * herding-poly / herding-rbf / random each select 3,300 from THIS pool.
  * The five existing filters' saved kept sets are reused as COMPARISON ROWS only.
  * Real cardiomegaly is split select/eval; herding sees only select; EVERY row is scored vs eval.

Run (GPU inference, ~minutes):
    python tstr/filters/bakeoff_v2.py
"""
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import cardio_metrics as M
import real_split
import features as FE
import herding as H
import kernels as K

POOL_DIR   = r"C:\Users\Tonkid\Downloads\runs\cardio_mad_A\gen0\fakes"          # cardiomegaly/ has 5,000
BAKE_DIR   = r"C:\Users\Tonkid\Downloads\results\filter_bakeoff"                 # existing kept sets
OUT_DIR    = r"C:\Users\Tonkid\Downloads\results\filter_bakeoff_v2"
CLS        = "cardiomegaly"
N_KEEP     = 3300
SEED       = 0
EXISTING   = {"maha": "PCA-Mahalanobis", "knn": "PCA-kNN", "disc": "C2ST (discriminator)",
              "umap": "UMAP", "tsne": "t-SNE"}


def _pngs(folder):
    import glob
    return sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))


def _score(name, kind, paths, real_eval_uint8, n_report):
    """FID/KID of a kept image set vs the held-out real eval half."""
    kept = FE.load_paths_uint8(paths)
    fk = M.fid_kid(real_eval_uint8, kept)
    print(f"  {name:22s} n={len(paths):5d}  FID {fk['fid']:7.2f}  KID {fk['kid']:+.5f} "
          f"(±{fk['kid_std']:.5f})", flush=True)
    return {"name": name, "kind": kind, "n": len(paths), "n_report": n_report,
            "fid": fk["fid"], "kid": fk["kid"], "kid_std": fk["kid_std"]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # --- leakage guard: disjoint real select/eval halves ---
    n_sel, n_eval = real_split.assert_disjoint()
    sel_paths = real_split.load_split("select")
    eval_paths = real_split.load_split("eval")
    print(f"real cardiomegaly split: select={n_sel}  eval={n_eval}  (disjoint OK)", flush=True)

    # --- feature-space sanity gate (our features == torchmetrics' inception) ---
    pool_paths = _pngs(os.path.join(POOL_DIR, CLS))
    assert len(pool_paths) >= N_KEEP, f"pool has {len(pool_paths)} < {N_KEEP}"
    FE.verify_matches_kid(pool_paths, n=200)

    # --- features: select-real + full pool (cached) ---
    print("extracting features (cached)...", flush=True)
    F_sel = FE.extract_features(sel_paths)
    F_pool = FE.extract_features(pool_paths)
    print(f"  select-real {F_sel.shape}  pool {F_pool.shape}", flush=True)

    # --- selectors ---
    diagnostics = {}
    print("herding-poly...", flush=True)
    idx_poly, diagnostics["herding-poly"] = H.greedy_mmd_select(
        F_pool, F_sel, N_KEEP, kernel="poly", kernel_kwargs={"degree": 3}, log_every=200)
    sigma = K.median_sigma(torch.from_numpy(F_sel).double())
    print(f"herding-rbf (sigma={sigma:.3f})...", flush=True)
    idx_rbf, diagnostics["herding-rbf"] = H.greedy_mmd_select(
        F_pool, F_sel, N_KEEP, kernel="rbf", kernel_kwargs={"sigma": sigma}, log_every=200)
    idx_rand = np.random.default_rng(SEED).choice(len(pool_paths), size=N_KEEP, replace=False)
    for name, d in diagnostics.items():
        print(f"  {name}: MMD^2 {d['mmd2_start']:.4g} -> {d['mmd2_end']:.4g}  "
              f"({d['wall_time_s']}s, peak {d['peak_gpu_mb']} MB)", flush=True)

    # --- score everything vs the real EVAL half ---
    real_eval = FE.load_paths_uint8(eval_paths)
    print(f"\nscoring vs held-out real eval (n={len(eval_paths)}):", flush=True)
    rows = []
    # unfiltered reference (native n=5000, NOT part of the equal-n ranking)
    rows.append(_score("none (unfiltered)", "reference", pool_paths, real_eval, n_report=len(pool_paths)))
    # equal-n ranking rows (all 3,300)
    for tag, pretty in EXISTING.items():
        kp = _pngs(os.path.join(BAKE_DIR, tag, CLS))
        if kp:
            rows.append(_score(pretty, "existing-filter", kp, real_eval, n_report=N_KEEP))
        else:
            print(f"  [skip {pretty}: no kept set at {os.path.join(BAKE_DIR, tag, CLS)}]", flush=True)
    rows.append(_score("random", "control", [pool_paths[i] for i in idx_rand], real_eval, N_KEEP))
    rows.append(_score("herding-poly", "herding", [pool_paths[i] for i in idx_poly], real_eval, N_KEEP))
    rows.append(_score("herding-rbf", "herding", [pool_paths[i] for i in idx_rbf], real_eval, N_KEEP))

    # --- rank the equal-n rows by KID ---
    ranked = sorted([r for r in rows if r["kind"] != "reference"], key=lambda r: r["kid"])
    print("\n=== ranked by KID vs eval (equal n=3,300; lower = distribution matches real) ===")
    hdr = f"| {'filter':22s} | {'kind':15s} | {'KID':>9s} | {'±std':>8s} | {'FID':>7s} |"
    print(hdr); print("|" + "-"*24 + "|" + "-"*17 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*9 + "|")
    md = [hdr, "|" + "-"*24 + "|" + "-"*17 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*9 + "|"]
    for r in ranked:
        line = f"| {r['name']:22s} | {r['kind']:15s} | {r['kid']:+9.5f} | {r['kid_std']:8.5f} | {r['fid']:7.2f} |"
        print(line); md.append(line)
    ref = next((r for r in rows if r["kind"] == "reference"), None)
    if ref:
        print(f"\n(reference) {ref['name']} at native n={ref['n']}: KID {ref['kid']:+.5f}  FID {ref['fid']:.2f}")

    out = {
        "config": {"class": CLS, "n_keep": N_KEEP, "n_pool": len(pool_paths),
                   "real_select": n_sel, "real_eval": n_eval, "seed": SEED, "rbf_sigma": sigma,
                   "wall_time_s": round(time.time() - t0, 1)},
        "caveats": [
            "Existing-filter kept sets came from the bake-off's SEPARATE pool draw and were selected "
            "using the FULL real set (known leakage). Reused here as comparison rows; SCORING is uniform "
            "(eval half), but their SELECTION provenance differs from herding/random. A fully-controlled "
            "run re-selects every filter on this pool + select half (cheap, no probe).",
            "FID on the ~2,194-image eval half is noisy; rank by KID, treat FID as secondary.",
            "'none' is the unfiltered pool at native n=5,000 and is NOT part of the equal-n ranking.",
        ],
        "diagnostics": diagnostics,
        "rows": rows,
        "ranked_by_kid": [r["name"] for r in ranked],
    }
    json.dump(out, open(os.path.join(OUT_DIR, "realism_rank_v2.json"), "w", encoding="utf-8"), indent=2)
    open(os.path.join(OUT_DIR, "table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'realism_rank_v2.json')}", flush=True)


if __name__ == "__main__":
    main()
