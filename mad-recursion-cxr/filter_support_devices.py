"""Measure (and optionally exclude) 'Support Devices' contamination in the cardiomegaly training set.

Your training images are named {subject_id}_{study_id}_{dicom_id}.png, and MIMIC's CheXpert label table
(chexpert.csv) carries a 'Support Devices' flag keyed by (subject_id, study_id). So we can tag every
training image with whether its study was labelled as having a support device (pacemaker/ICD/lines/
sternal wires/ECG leads/CardioMEMS/... — the label does NOT distinguish which) and count the data loss
of filtering them out. Report-text CardioMEMS-specific filtering isn't possible here (no reports on disk).

  python tstr/filter_support_devices.py                 # MEASURE only (counts per class)
  python tstr/filter_support_devices.py --write-exclude  # also write exclusion lists (paths to drop)

CheXpert 'Support Devices' values: 1.0 = present, 0.0 = explicitly absent, blank = not mentioned,
-1.0 = uncertain. We treat 1.0 (and optionally -1.0) as 'device'.
"""
import os, sys, csv, glob, argparse

CHEXPERT = r"D:\mimic-cardiomegaly\chexpert.csv"
TRAIN    = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_full\train"
CLASSES  = ("cardiomegaly", "normal")
OUT_DIR  = r"C:\Users\Tonkid\Downloads\results\device_filter"


def load_device_flags():
    """(subject_id, study_id) -> raw 'Support Devices' string ('1.0','0.0','-1.0','')."""
    flags = {}
    with open(CHEXPERT, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            flags[(row["subject_id"], row["study_id"])] = row.get("Support Devices", "").strip()
    return flags


def classify(val, uncertain_is_device):
    if val == "1.0":
        return "device"
    if val == "-1.0":
        return "device" if uncertain_is_device else "uncertain"
    if val == "0.0":
        return "no_device"
    return "unmentioned"      # blank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-exclude", action="store_true",
                    help="write per-class exclusion lists (image paths whose study has a device)")
    ap.add_argument("--uncertain-is-device", action="store_true",
                    help="count CheXpert -1.0 (uncertain) as device too (more aggressive)")
    a = ap.parse_args()

    flags = load_device_flags()
    print(f"loaded {len(flags)} study-level Support Devices labels from chexpert.csv\n", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for cls in CLASSES:
        paths = sorted(glob.glob(os.path.join(TRAIN, cls, "*.png")))
        counts = {"device": 0, "no_device": 0, "unmentioned": 0, "uncertain": 0, "unmatched": 0}
        drop = []
        for p in paths:
            parts = os.path.basename(p).split("_")
            key = (parts[0], parts[1]) if len(parts) >= 2 else None
            if key is None or key not in flags:
                counts["unmatched"] += 1
                continue
            c = classify(flags[key], a.uncertain_is_device)
            counts[c] += 1
            if c == "device":
                drop.append(p)
        n = len(paths)
        print(f"=== {cls}  (n={n}) ===", flush=True)
        for k in ("device", "no_device", "unmentioned", "uncertain", "unmatched"):
            if counts[k]:
                print(f"  {k:12s} {counts[k]:6d}  ({100*counts[k]/max(1,n):5.1f}%)", flush=True)
        keep = n - counts["device"]
        print(f"  -> filtering 'device' would DROP {counts['device']} ({100*counts['device']/max(1,n):.1f}%), "
              f"keep {keep}", flush=True)
        if a.write_exclude:
            outp = os.path.join(OUT_DIR, f"exclude_{cls}_device.txt")
            with open(outp, "w") as f:
                f.write("\n".join(drop))
            print(f"  wrote exclusion list -> {outp}  ({len(drop)} paths)", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
