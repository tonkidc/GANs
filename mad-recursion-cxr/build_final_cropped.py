"""Build the FINALIZED device-filtered + border-cropped training set.

  device filter = union of (report keywords) U (CheXpert Support Devices) U (NegBio Support Devices)
  then balance 1:1 (seed 0), auto-crop the black collimation border off each image, resize 256.

  in : tstr/data_cardio_full/train/{cardiomegaly,normal}/*.png  (256px, bordered)
  out: tstr/data_cardio_final_crop/train/{cardiomegaly,normal}/*.png  (256px, cropped, ~7047/class)
"""
import os, csv, glob, random
import numpy as np
import pandas as pd
from PIL import Image

FULL = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_full\train"
OUT  = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_final_crop\train"
HITS = r"C:\Users\Tonkid\Downloads\results\device_filter\report_device_hits.csv"
CX   = r"C:\Users\Tonkid\Downloads\mimic_cxr_cardiomegaly\mimic-cxr-2.0.0-chexpert.csv.gz"
NB   = r"C:\Users\Tonkid\Downloads\mimic_cxr_cardiomegaly\mimic-cxr-2.0.0-negbio.csv.gz"
CLASSES, SEED = ("cardiomegaly", "normal"), 0


def autocrop(gray_u8, frac=0.15, pad=2):
    """Crop black border: a border row/col has near-zero MEAN (robust to bright markers). Resize to fill."""
    a = np.asarray(gray_u8, np.float32)
    rm, cm = a.mean(1), a.mean(0)
    rows = np.where(rm > frac * rm.max())[0]
    cols = np.where(cm > frac * cm.max())[0]
    if len(rows) < 8 or len(cols) < 8:
        return gray_u8
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, a.shape[0] - 1)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, a.shape[1] - 1)
    return Image.fromarray(a[r0:r1 + 1, c0:c1 + 1].astype(np.uint8))


def main():
    # ---- finalized device drop set (union of all 3 text signals) ----
    kw = set()
    with open(HITS, newline="") as f:
        for row in csv.DictReader(f):
            kw.add(str(row["study_id"]))
    cx = pd.read_csv(CX); nb = pd.read_csv(NB)
    cx_sd = set(cx[cx["Support Devices"] == 1.0]["study_id"].astype(str))
    nb_sd = set(nb[nb["Support Devices"] == 1.0]["study_id"].astype(str))
    drop = kw | cx_sd | nb_sd
    print(f"device drop set (keyword U CheXpert U NegBio): {len(drop)} studies", flush=True)

    clean = {}
    for cls in CLASSES:
        paths = sorted(glob.glob(os.path.join(FULL, cls, "*.png")))
        kept = [p for p in paths if os.path.basename(p).split("_")[1] not in drop]
        clean[cls] = kept
        print(f"  {cls:14s}: full {len(paths):6d} -> device-clean {len(kept):6d}", flush=True)

    n = min(len(v) for v in clean.values())
    print(f"\nbalancing to {n}/class, cropping + resizing 256 ...\n", flush=True)
    rng = random.Random(SEED)
    for cls in CLASSES:
        files = clean[cls]
        if len(files) > n:
            files = rng.sample(files, n)
        d = os.path.join(OUT, cls); os.makedirs(d, exist_ok=True)
        for i, src in enumerate(files):
            g = Image.open(src).convert("L")
            cropped = autocrop(g).resize((256, 256), Image.LANCZOS)
            cropped.save(os.path.join(d, os.path.basename(src)))
            if (i + 1) % 1000 == 0:
                print(f"    {cls}: {i + 1}/{n}", flush=True)
        print(f"  {cls:14s}: wrote {len(files)} -> {d}", flush=True)
    print(f"\nDONE -> {OUT}  ({n}/class, device-filtered + border-cropped)")
    return OUT


if __name__ == "__main__":
    main()
