"""Milestone-2 Step 7 — TSTR recall per filter. The test that actually matters.

Lower KID is not the goal by itself. For each materialised kept set (herding-*, random), train the
FROZEN DenseNet-121 probe on it and score on the LOCKED real test set. Record AUROC + cardiomegaly
recall. Compare against the Milestone-1 filters' recall. We EXPECT KID and recall may disagree — that
disagreement is the result, not a bug. Nothing here is tuned to make them agree.

Reuses tstr/densenet_probe.py unchanged (same frozen recipe, same locked test) so the number is
comparable to every earlier probe run.

Run (GPU TRAINING — ~1 h per kept set; run overnight):
    python tstr/filters/probe_eval.py                    # all kept sets under kept_root
    python tstr/filters/probe_eval.py --tags herding-poly-inception random
"""
import os, sys, json, glob, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY         = sys.executable
PROBE      = r"C:\Users\Tonkid\Downloads\tstr\densenet_probe.py"
PROBE_JSON = r"C:\Users\Tonkid\Downloads\tstr\probe_densenet"
KEPT_ROOT  = r"C:\Users\Tonkid\Downloads\results\filter_bakeoff_v2\kept"
OUT_DIR    = r"C:\Users\Tonkid\Downloads\results\filter_bakeoff_m2"
EXISTING   = {"maha": "PCA-Mahalanobis", "knn": "PCA-kNN", "disc": "C2ST (discriminator)",
              "umap": "UMAP", "tsne": "t-SNE"}


def run_probe(train_dir, tag, epochs):
    """Train the frozen probe on train_dir, test on the locked real test set; return its result json."""
    for cls in ("cardiomegaly", "normal"):
        n = len(glob.glob(os.path.join(train_dir, cls, "*.png")))
        if n == 0:
            raise RuntimeError(f"{train_dir}/{cls} is empty — run bakeoff_m2.py to materialise kept sets")
    subprocess.run([PY, PROBE, "--train", train_dir, "--tag", tag,
                    "--epochs", str(epochs), "--test-split", "test"], check=True)
    return json.load(open(os.path.join(PROBE_JSON, f"{tag}.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept-root", default=KEPT_ROOT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--tags", nargs="+", default=None, help="kept-set subfolders (default: all)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    tags = a.tags or sorted(d for d in os.listdir(a.kept_root)
                            if os.path.isdir(os.path.join(a.kept_root, d)))
    if not tags:
        raise RuntimeError(f"no kept sets under {a.kept_root} — run bakeoff_m2.py first")

    rows = []
    for tag in tags:
        print(f"\n=== TSTR probe: {tag} ===", flush=True)
        r = run_probe(os.path.join(a.kept_root, tag), f"m2_{tag}", a.epochs)
        rows.append({"name": tag, "kind": "m2", "auroc": round(r["auroc"], 4),
                     "recall_cardiomegaly": round(r["recall_pos"], 4),
                     "recall_normal": round(r["recall_neg"], 4), "acc": round(r["acc"], 4),
                     "n_train": r.get("n_train")})

    # Milestone-1 filters' recall for comparison (already computed in the earlier bake-off)
    for t, pretty in EXISTING.items():
        p = os.path.join(PROBE_JSON, f"bakeoff_{t}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            rows.append({"name": f"{pretty} (M1)", "kind": "m1-ref", "auroc": round(r["auroc"], 4),
                         "recall_cardiomegaly": round(r["recall_pos"], 4),
                         "recall_normal": round(r["recall_neg"], 4), "acc": round(r["acc"], 4),
                         "n_train": r.get("n_train")})

    rows.sort(key=lambda r: -r["recall_cardiomegaly"])
    print("\n=== ranked by cardiomegaly recall (the disease signal) ===")
    print(f"  {'filter':28s} {'recall':>7s} {'AUROC':>7s} {'acc':>6s}")
    for r in rows:
        print(f"  {r['name']:28s} {r['recall_cardiomegaly']:7.3f} {r['auroc']:7.3f} {r['acc']:6.3f}", flush=True)

    out = {"note": "TSTR: probe trained on each kept set, tested on the LOCKED real test set. Compare "
                   "herding's recall against KID rank (realism_rank_m2.json) — if the KID winner is not "
                   "the recall winner, realism and disease-signal have decoupled.",
           "epochs": a.epochs, "rows": rows,
           "ranked_by_recall": [r["name"] for r in rows]}
    json.dump(out, open(os.path.join(OUT_DIR, "recall_m2.json"), "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'recall_m2.json')}", flush=True)


if __name__ == "__main__":
    main()
