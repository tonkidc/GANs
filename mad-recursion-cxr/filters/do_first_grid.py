"""The 10-minute 'do first' check from the spec — BEFORE building/trusting gates.

Runs herding on the pool once and dumps two image grids: the FIRST picks (1-50) vs the LAST picks
(n_keep-49 .. n_keep). Herding's order is quality-descending in the set-aware sense, so:
  * late grid clean  -> term A already handles validity, the gates may be unnecessary -> consider skipping.
  * late grid has broken/non-chest images -> the gates earn their place -> build/enable them.

Additive + read-only: reads the pool + real select half, writes only two PNGs. No selection is committed.

    python tstr/filters/do_first_grid.py                     # primary class (cardiomegaly)
    python tstr/filters/do_first_grid.py --cls normal
"""
import os, sys, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import real_split
import features as FE
import herding as H

POOL_DIR = r"C:\Users\Tonkid\Downloads\runs\cardio_mad_A\gen0\pool50k"
OUT_DIR  = r"C:\Users\Tonkid\Downloads\results\filter_ablation"


def _pngs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))


def _grid(paths, out_png, cols=10, cell=128, title=""):
    from PIL import Image, ImageDraw
    n = len(paths); rows = (n + cols - 1) // cols
    W, H = cols * cell, rows * cell + 20
    canvas = Image.new("L", (W, H), 0)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L").resize((cell, cell))
        canvas.paste(im, ((i % cols) * cell, 20 + (i // cols) * cell))
    ImageDraw.Draw(canvas).text((4, 4), title, fill=255)
    canvas.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=POOL_DIR)
    ap.add_argument("--cls", default="cardiomegaly")
    ap.add_argument("--n-keep", type=int, default=3300)
    ap.add_argument("--n-show", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    pool_paths = _pngs(os.path.join(a.pool_dir, a.cls))
    assert len(pool_paths) >= a.n_keep, f"pool {len(pool_paths)} < n_keep {a.n_keep} — run generate_pool.py"
    sel_paths = real_split.load_split("select", a.cls)
    pool_feat = FE.extract_features(pool_paths, backbone="inception")
    sel_feat  = FE.extract_features(sel_paths, backbone="inception")

    idx, _ = H.greedy_mmd_select(pool_feat, sel_feat, a.n_keep, kernel="poly",
                                 kernel_kwargs={"degree": 3}, log_every=1000)
    early = [pool_paths[i] for i in idx[:a.n_show]]
    late  = [pool_paths[i] for i in idx[a.n_keep - a.n_show:a.n_keep]]

    e = _grid(early, os.path.join(OUT_DIR, f"dofirst_{a.cls}_early_1-{a.n_show}.png"),
              title=f"{a.cls}  herding picks 1-{a.n_show} (best)")
    l = _grid(late, os.path.join(OUT_DIR, f"dofirst_{a.cls}_late_{a.n_keep-a.n_show}-{a.n_keep}.png"),
              title=f"{a.cls}  herding picks {a.n_keep-a.n_show}-{a.n_keep} (last kept)")
    print("wrote:\n ", e, "\n ", l,
          "\n\nEyeball the LATE grid: clean -> gates likely unnecessary; broken/non-chest -> build the gates.",
          flush=True)


if __name__ == "__main__":
    main()
