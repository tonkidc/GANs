"""Ablation table for the staged filter — the eval that decides whether the gates earn their place.

Rows  : each selector {herding, dpp, random} x gate config {none, +Q1, +Q1Q2}.
Cols  : KID (sanity only) | coverage (DECISION) | density | Vendi | copy_rate.
Rules the spec pins down and this honours:
  * DECISION METRIC = coverage — feature-based, and NOT what the selector optimises (the selector
    optimises KID/MMD, so KID here is only a unit-test sanity column, never the judge).
  * copy_rate REQUIRED — a fidelity term can 'win' by keeping near-copies of real patients; elevated
    copy_rate => bought the score by memorising => disqualified.
  * Select on the real SELECT half; score KID/coverage on the real EVAL half; disjoint (asserted).
  * Fixed metric params (k, n_keep, backbone) across every row; every number re-runnable from --seed.

Scores the PRIMARY class (cardiomegaly) vs its held-out eval half, like bakeoff_m2. Additive: reads the
50k pool + real halves, writes only results/filter_ablation/. Touches no existing filter/probe/test set.

    python tstr/filters/ablation.py                    # herding+dpp+random x {none,Q1,Q1Q2}
    python tstr/filters/ablation.py --selectors herding dpp     # subset
"""
import os, sys, json, time, glob, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import cardio_metrics as M
import real_split
import features as FE
import prdc_vendi as PV
import pipeline as P

POOL_DIR = r"C:\Users\Tonkid\Downloads\runs\cardio_mad_A\gen0\pool50k"
OUT_DIR  = r"C:\Users\Tonkid\Downloads\results\filter_ablation"
PRIMARY  = "cardiomegaly"
GATE_CONFIGS = [("none", False, False), ("+Q1", True, False), ("+Q1Q2", True, True)]
K_COV = 5                         # fixed k for coverage/density across all rows
N_REAL = 5000                     # cap on the real eval reference (KID + coverage), bounds RAM/time


def _pngs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=POOL_DIR)
    ap.add_argument("--n-keep", type=int, default=3300)
    ap.add_argument("--selectors", nargs="+", default=["herding", "dpp", "random"])
    ap.add_argument("--no-q2", action="store_true", help="skip the slow Q2 anatomy gate (none + Q1 only)")
    ap.add_argument("--subset", type=int, default=None, help="sample this many pool fakes (fast subset run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # ---- data: pool paths + real disjoint halves + features (fixed backbone = inception) ----
    real_split.assert_disjoint(PRIMARY)
    pool_paths = _pngs(os.path.join(a.pool_dir, PRIMARY))
    if a.subset and a.subset < len(pool_paths):          # fast subset run (deterministic sample)
        sel = np.random.default_rng(a.seed).choice(len(pool_paths), size=a.subset, replace=False)
        pool_paths = [pool_paths[i] for i in sorted(sel)]
        print(f"[subset] sampled {len(pool_paths)} of the pool (seed {a.seed})", flush=True)
    if len(pool_paths) < a.n_keep:
        raise RuntimeError(f"pool {len(pool_paths)} < n_keep {a.n_keep} — lower --n-keep or --subset")
    sel_paths = real_split.load_split("select", PRIMARY)
    eval_paths = real_split.load_split("eval", PRIMARY)[:N_REAL]        # cap eval ref (bounds coverage NxN + RAM)
    print(f"[{PRIMARY}] pool={len(pool_paths)} select={len(sel_paths)} eval={len(eval_paths)} (capped)", flush=True)

    pool_feat = FE.extract_features(pool_paths, backbone="inception")
    sel_feat  = FE.extract_features(sel_paths, backbone="inception")
    eval_feat = FE.extract_features(eval_paths, backbone="inception")
    real_eval_uint8 = M.load_paths_uint8(eval_paths)
    real_sel_uint8  = M.load_paths_uint8(sel_paths[:M.METRIC_N])       # copy-rate reference (memorisation)

    # ---- gate masks computed ONCE (independent of selector) ----
    term_A = P._term_A(pool_feat, sel_feat)
    q1 = P._q1_mask_chunked(pool_paths, term_A)
    masks = {"none": np.ones(len(pool_paths), dtype=bool), "+Q1": q1}
    configs = [c for c in GATE_CONFIGS if c[0] in ("none", "+Q1")]
    if not a.no_q2:                                        # Q2 anatomy is the slow (segmentation) gate
        masks["+Q1Q2"] = q1 & P._q2_mask_survivors(pool_paths, q1, device=a.device)
        configs = GATE_CONFIGS
    print("gates: " + "  ".join(f"{k}={int(v.sum())}" for k, v in masks.items()), flush=True)

    def score(idx):
        paths = [pool_paths[i] for i in idx]
        kept_uint8 = M.load_paths_uint8(paths)
        kept_feat = FE.extract_features(paths, backbone="inception")
        fk = M.fid_kid(real_eval_uint8, kept_uint8)
        cd = PV.coverage_density(eval_feat, kept_feat, k=K_COV)
        vd = PV.vendi_score(kept_feat, kernel="cosine")
        cr = M.copy_rate(real_sel_uint8, kept_uint8)
        return {"kid": fk["kid"], "kid_std": fk["kid_std"], "coverage": cd["coverage"],
                "density": cd["density"], "vendi": vd["vendi"], "copy_rate": cr["copy_rate"]}

    rows = []
    for selector in a.selectors:
        for tag, uq1, uq2 in configs:
            sub_idx = np.where(masks[tag])[0]
            kept_idx, diag = P._select(selector, pool_feat, sel_feat, sub_idx, a.n_keep, seed=a.seed)
            s = score(kept_idx)
            s.update({"selector": selector, "gates": tag, "n_gated": int(len(sub_idx))})
            rows.append(s)
            print(f"  {selector:8s} {tag:6s} | coverage {s['coverage']:.3f}  KID {s['kid']:+.5f}  "
                  f"density {s['density']:.3f}  Vendi {s['vendi']:.1f}  copy_rate {s['copy_rate']:.3f}",
                  flush=True)

    ranked = sorted(rows, key=lambda r: -r["coverage"])   # DECISION metric = coverage (higher better)
    print("\n=== ranked by COVERAGE (decision metric; KID is a sanity unit-test only) ===")
    print(f"  {'selector':8s} {'gates':6s} {'cover':>6s} {'KID':>9s} {'dens':>6s} {'Vendi':>6s} {'copy':>6s}")
    for r in ranked:
        flag = "  <- HIGH COPY (memorising?)" if r["copy_rate"] > 0.10 else ""
        print(f"  {r['selector']:8s} {r['gates']:6s} {r['coverage']:6.3f} {r['kid']:+9.5f} "
              f"{r['density']:6.3f} {r['vendi']:6.1f} {r['copy_rate']:6.3f}{flag}")

    out = {"config": {"pool_dir": a.pool_dir, "n_keep": a.n_keep, "primary": PRIMARY, "k_coverage": K_COV,
                      "backbone": "inception", "seed": a.seed, "n_select": len(sel_paths),
                      "n_eval": len(eval_paths), "wall_time_s": round(time.time() - t0, 1)},
           "gate_survivors": {k: int(v.sum()) for k, v in masks.items()},
           "decision_metric": "coverage",
           "note": "coverage is the decision metric (not KID, which the selector optimises). copy_rate>0.10 "
                   "flags a set that may have bought its fidelity by memorising real patients.",
           "rows": rows,
           "ranked_by_coverage": [f"{r['selector']}/{r['gates']}" for r in ranked]}
    json.dump(out, open(os.path.join(OUT_DIR, "ablation.json"), "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'ablation.json')}", flush=True)


if __name__ == "__main__":
    main()
