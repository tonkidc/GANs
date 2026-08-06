"""Generate a large unfiltered fake pool from the gen-0 GAN, for the Milestone-2 filter experiments.

A big pool + a small keep fraction is what lets herding stay selective (Milestone 1 showed its MMD^2
climbing once forced past ~66% of a 5k pool). Default 50,000 per class -> 3,300 kept = 6.6%.

Streams to disk one batch at a time (never holds 50k images in RAM). Resumable: skips a class whose
folder already has >= n images. Reuses filter_bakeoff.load_G so the generation matches the bake-off's.

Run (GPU, ~minutes-to-tens-of-minutes):
    python tstr/filters/generate_pool.py                 # 50k/class -> runs/.../pool50k/<cls>/
    python tstr/filters/generate_pool.py --n 20000       # smaller
"""
import os, sys, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r"C:\Users\Tonkid\Downloads\tstr")
import filter_bakeoff as FB   # reuse load_G (same checkpoint/config as the bake-off)

CKPT = r"C:\Users\Tonkid\Downloads\runs\cardio_mad_A\gen0\checkpoints\final.pt"
OUT  = r"C:\Users\Tonkid\Downloads\runs\cardio_mad_A\gen0\pool50k"


def _save(arr_chw_uint8, path):
    """Save a (C,H,W) uint8 array as a grayscale PNG (matches the existing fakes)."""
    if arr_chw_uint8.shape[0] == 1:
        Image.fromarray(arr_chw_uint8[0]).save(path)
    else:
        Image.fromarray(arr_chw_uint8.transpose(1, 2, 0)).convert("L").save(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="fakes PER CLASS")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert os.path.isfile(a.ckpt), f"no checkpoint at {a.ckpt} — train gen-0 first"

    G, classes, c_dim, cfg = FB.load_G(a.ckpt)
    dev = FB.DEV
    print(f"loaded G ({classes}, c_dim={c_dim}, z={cfg['z_dim']}) on {dev}; target {a.n}/class", flush=True)

    for ci, cls in enumerate(classes):
        d = os.path.join(a.out, cls)
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".png")])
        if have >= a.n:
            print(f"{cls}: {have} already present >= {a.n}, skipping", flush=True)
            continue
        torch.manual_seed(a.seed + ci)
        k = 0
        for start in range(0, a.n, a.batch):
            b = min(a.batch, a.n - start)
            z = torch.randn(b, cfg["z_dim"], device=dev)
            c = torch.zeros(b, c_dim, device=dev); c[:, ci] = 1.0
            img = G(z, c, truncation_psi=1.0, noise_mode="random")
            img = ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()   # (b,C,H,W)
            for j in range(b):
                _save(img[j], os.path.join(d, f"{k:06d}.png"))
                k += 1
            if start % (a.batch * 40) == 0:
                print(f"  {cls}: {k}/{a.n}", flush=True)
        print(f"{cls}: wrote {k} -> {d}", flush=True)
    print("pool generation done.", flush=True)


if __name__ == "__main__":
    main()
