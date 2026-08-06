"""Materialize the cardiomegaly manifest into CLASS-FOLDER PNGs for the conditional GAN.

The DenseNet probe reads the real test set straight from the manifest (on the fly), but the
class-conditional StyleGAN2-ADA needs its training data as image files laid out in class
subfolders (LabeledImageDataset):

    data_cardio_mad/train/cardiomegaly/*.png   -> class 0 (sorted order)
    data_cardio_mad/train/normal/*.png         -> class 1

This script builds ONLY that train tree (the test set stays in the manifest, LOCKED, and is
scored by the probe). Folder names match the probe's FolderDataset exactly ("normal" /
"cardiomegaly"), so gen-0 real training data and gen-k synthetic data are interchangeable.

Label from cardiomegaly_chexpert: 1.0 -> cardiomegaly, 0.0 -> normal, -1.0/blank -> DROPPED.
Only rows whose image is actually on disk are written (works on a partial download).

  python tstr/preprocess_cardio_mad.py                 # all train rows on disk, 256px
  python tstr/preprocess_cardio_mad.py --limit 500     # cap per class (quick GAN smoke)

PREREQUISITE: the manifest + at least some pXX image chunks in Downloads. Same resolver as the
probe, so it finds files/pXX/... under either mimic_cxr_cardiomegaly/ or a bare Downloads/pXX/.
"""
import os, csv, argparse
from PIL import Image, ImageFile
# Keep LOAD_TRUNCATED_IMAGES off by default so a truncated jpg RAISES and we can FLAG it; the
# helper below then loads it anyway (so we don't lose the image) but records that it was truncated.


def load_gray_flagged(src):
    """Load `src` as grayscale. If the jpg is truncated (incomplete download), FLAG it and load it
    anyway. Returns (PIL 'L' image, truncated: bool). Raises only on a genuinely unreadable file."""
    try:
        img = Image.open(src); img.load()
        return img.convert("L"), False
    except OSError as e:
        if "truncat" in str(e).lower():
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                img = Image.open(src); img.load()
                return img.convert("L"), True
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = False
        raise

RES       = 256
MANIFEST  = r"C:\Users\Tonkid\Downloads\mimic_cxr_cardiomegaly\cardiomegaly_manifest.csv"
DATA_ROOT = r"C:\Users\Tonkid\Downloads\mimic_cxr_cardiomegaly"
DOWNLOADS = r"C:\Users\Tonkid\Downloads"
OUT       = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_mad"
SEARCH_ROOTS = [DATA_ROOT, DOWNLOADS]
LABEL_DIR = {"1.0": "cardiomegaly", "0.0": "normal"}   # -1.0 / blank -> dropped


def resolve_image(relpath):
    """Same resolver as densenet_probe: try the relpath (and the 'files/'-stripped variant)
    under each candidate root, return the first that exists on disk."""
    rp = relpath.replace("/", os.sep)
    cands = [rp]
    if rp.lower().startswith("files" + os.sep):
        cands.append(rp[len("files" + os.sep):])
    for root in SEARCH_ROOTS:
        for c in cands:
            p = os.path.join(root, c)
            if os.path.isfile(p):
                return p
    return None


def main():
    global MANIFEST, OUT, SEARCH_ROOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=RES)
    ap.add_argument("--split", default="train", choices=["train", "validate", "test"],
                    help="which manifest split to materialize (default: train — the GAN's data)")
    ap.add_argument("--limit", type=int, default=None, help="cap images PER CLASS (GAN smoke test)")
    ap.add_argument("--manifest", default=MANIFEST, help="manifest CSV (default: old partial set)")
    ap.add_argument("--data-root", default=DATA_ROOT, help="root containing files/pXX/ (relpath base)")
    ap.add_argument("--out", default=OUT, help="output root for the class-folder PNG tree")
    a = ap.parse_args()
    MANIFEST, OUT = a.manifest, a.out
    SEARCH_ROOTS = [a.data_root, DOWNLOADS]

    out_split = os.path.join(OUT, a.split)
    for sub in LABEL_DIR.values():
        os.makedirs(os.path.join(out_split, sub), exist_ok=True)

    counts = {sub: 0 for sub in LABEL_DIR.values()}
    missing = dropped = bad = 0
    trunc_list = []                                   # FLAGGED: truncated jpgs (loaded anyway)
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != a.split:
                continue
            sub = LABEL_DIR.get(r["cardiomegaly_chexpert"].strip())
            if sub is None:
                dropped += 1                      # -1.0 uncertain / blank
                continue
            if a.limit and counts[sub] >= a.limit:
                continue
            src = resolve_image(r["relpath"])
            if src is None:
                missing += 1
                continue
            try:
                img, truncated = load_gray_flagged(src)
            except Exception as e:
                print(f"  [UNREADABLE, skipped] {src}: {e}", flush=True)
                bad += 1
                continue
            if truncated:
                trunc_list.append(src)
                print(f"  [TRUNCATED #{len(trunc_list)}, loaded anyway] {src}", flush=True)
            img = img.resize((a.res, a.res), Image.LANCZOS)
            name = f"{r['subject_id']}_{r['study_id']}_{r['dicom_id']}.png"
            img.save(os.path.join(out_split, sub, name))
            counts[sub] += 1
            n = sum(counts.values())
            if n % 200 == 0:
                print(f"    ...{n} written", flush=True)

    if trunc_list:                                    # write the flagged list to disk
        flag_file = os.path.join(OUT, f"truncated_{a.split}.txt")
        with open(flag_file, "w") as fh:
            fh.write("\n".join(trunc_list))
    print(f"\n=== DONE -> {out_split} ===")
    for sub in sorted(counts):
        print(f"  {sub:13s}: {counts[sub]:5d} images")
    print(f"  TOTAL: {sum(counts.values())} written | {missing} rows not on disk | "
          f"{dropped} uncertain/blank dropped")
    print(f"  ⚠ FLAGGED {len(trunc_list)} TRUNCATED image(s) (loaded anyway"
          + (f"; list -> {flag_file}" if trunc_list else "") + f") | {bad} unreadable/skipped")
    print(f"\n(class index for the conditional GAN = sorted folder order: "
          f"cardiomegaly=0, normal=1)")


if __name__ == "__main__":
    main()
