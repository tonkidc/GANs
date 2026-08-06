"""Frozen DenseNet-121 classifier probe for the cardiomegaly recursion study (TSTR).

This is a MEASUREMENT INSTRUMENT, not a model to tune. The recipe below is FROZEN on
purpose: same architecture, same hyperparameters, same locked real test set for every run.
That way any change in the score is provably caused by the TRAINING DATA (real vs synthetic
gen-k), not by the classifier. Do not "improve" it between generations.

Task   : binary cardiomegaly (1) vs normal (0), frontal PA chest X-ray, 224px, grayscale->3ch
Metric : AUROC (primary) + rare-class recall / sensitivity (secondary), on the locked REAL test
Model  : DenseNet-121, ImageNet-pretrained, fine-tuned (CheXNet standard)

Usage
-----
  # real upper-bound baseline (train on real, test on locked real test):
  python tstr/densenet_probe.py --train real --tag real_baseline

  # TSTR for a synthetic generation (train on fakes, test on the SAME locked real test):
  python tstr/densenet_probe.py --train "C:/Users/Tonkid/Downloads/.../synthetic_gen0" --tag gen0

  # quick smoke test on whatever is on disk locally (p10 only right now):
  python tstr/densenet_probe.py --train real --tag smoke --epochs 1 --test-split validate
"""
import os, sys, csv, glob, argparse, json, time
os.environ.setdefault("MIOPEN_FIND_MODE", "NORMAL")
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
import numpy as np
import torch
# ROCm/gfx1101 toolchain bug: MIOpen JIT-compiles its BatchNorm-train kernel at runtime and
# the HIP compiler can't find <type_traits> -> miopenStatusUnknownError. StyleGAN2 dodges it
# (no BatchNorm); DenseNet is full of it. Disabling cudnn/MIOpen routes BatchNorm (and conv)
# to PyTorch's native HIP kernels, which work. Small speed cost, but this is just the probe.
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # some MIMIC jpgs download incomplete; load them anyway
import torchvision.transforms as T
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

# ---------------------------------------------------------------------------
# FROZEN CONFIG  (the ruler — do not change between generations)
# ---------------------------------------------------------------------------
DEV        = "cuda" if torch.cuda.is_available() else "cpu"
RES        = 256            # match the GAN/recursion resolution (BraTS pipeline is 256); DenseNet's
                           # adaptive pool handles it, and it avoids resizing the 256px fakes
BATCH      = 32
EPOCHS     = 20
LR         = 1e-4
WEIGHT_DEC = 1e-4
SEED       = 0
CLASSES    = ["normal", "cardiomegaly"]   # index 0, 1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MANIFEST = r"C:\Users\Tonkid\Downloads\tstr\data_cardio_full\manifest_full.csv"   # FULL patient-disjoint splits
DATA_ROOT = r"D:\mimic-cardiomegaly"   # relpath (files/pXX/...) is relative to this (full download)
DOWNLOADS = r"C:\Users\Tonkid\Downloads"
# Manifest relpaths look like 'files/p10/pXXXX/sXXXX/uuid.jpg'. Chunks may live either under
# DATA_ROOT/files/... (original layout) OR be dropped straight into Downloads as p10/, p11/, ...
# so we resolve each relpath against several candidate roots and take the first that exists.
SEARCH_ROOTS = [DATA_ROOT, DOWNLOADS]


def resolve_image(relpath):
    rp = relpath.replace("/", os.sep)
    cands = [rp]
    if rp.lower().startswith("files" + os.sep):     # also try without the 'files/' prefix
        cands.append(rp[len("files" + os.sep):])
    for root in SEARCH_ROOTS:
        for c in cands:
            p = os.path.join(root, c)
            if os.path.isfile(p):
                return p
    return None


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
# NB: no horizontal flip — flipping a chest X-ray puts the heart on the wrong side
# (anatomically invalid). Mild affine + photometric only.
def make_tf(train: bool):
    ops = [T.Grayscale(num_output_channels=3), T.Resize((RES, RES))]
    if train:
        ops += [T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.1, contrast=0.1)]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
class MimicManifestDataset(Dataset):
    """Reads the cardiomegaly manifest. label from cardiomegaly_chexpert:
    1.0 -> cardiomegaly(1), 0.0 -> normal(0), -1.0/blank -> DROPPED. Only rows whose
    image file actually exists on disk are kept (so it works on a partial download)."""
    def __init__(self, split, train_aug, limit=None):
        self.tf = make_tf(train_aug)
        rows, missing = [], 0
        with open(MANIFEST, newline="") as f:
            for r in csv.DictReader(f):            # csv handles the quoted-comma fields
                if r["split"] != split:
                    continue
                lab = r["cardiomegaly_chexpert"].strip()
                if lab == "1.0":
                    y = 1
                elif lab == "0.0":
                    y = 0
                else:
                    continue                       # drop -1.0 / blank
                p = resolve_image(r["relpath"])
                if p is None:
                    missing += 1
                    continue
                rows.append((p, y))
        if limit:
            rows = rows[:limit]
        self.rows = rows
        self.missing = missing
        self.pos = sum(y for _, y in rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, y = self.rows[i]
        return self.tf(Image.open(p).convert("L")), y


class FolderDataset(Dataset):
    """For synthetic TSTR: class subfolders  <root>/normal/*.png  <root>/cardiomegaly/*.png"""
    def __init__(self, root, train_aug, limit=None):
        self.tf = make_tf(train_aug)
        rows = []
        for y, c in enumerate(CLASSES):
            fs = sorted(glob.glob(os.path.join(root, c, "*.png")) +
                        glob.glob(os.path.join(root, c, "*.jpg")))
            rows += [(f, y) for f in fs]
        if limit:
            rows = rows[:limit]
        self.rows = rows
        self.pos = sum(y for _, y in rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, y = self.rows[i]
        return self.tf(Image.open(p).convert("L")), y


def build_train_ds(train_arg, limit=None):
    if train_arg == "real":
        return MimicManifestDataset("train", train_aug=True, limit=limit)
    return FolderDataset(train_arg, train_aug=True, limit=limit)


# ---------------------------------------------------------------------------
# model / train / eval
# ---------------------------------------------------------------------------
def build_model():
    try:
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"[warn] could not fetch ImageNet weights ({e}); random init (smoke only).", flush=True)
        m = densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, len(CLASSES))
    return m.to(DEV)


def train_classifier(train_ds, epochs=EPOCHS):
    torch.manual_seed(SEED); np.random.seed(SEED)
    # class-weighted loss to protect the minority class
    n_pos = max(train_ds.pos, 1); n_neg = max(len(train_ds) - train_ds.pos, 1)
    w = torch.tensor([len(train_ds) / (2 * n_neg), len(train_ds) / (2 * n_pos)],
                     dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    model.train()
    for ep in range(epochs):
        tot, t0 = 0.0, time.time()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        print(f"  epoch {ep+1:2d}/{epochs}  loss {tot/max(len(train_ds),1):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def evaluate(model, test_ds):
    model.eval()
    dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    probs, ys = [], []
    for x, y in dl:
        p = torch.softmax(model(x.to(DEV)), 1)[:, 1].cpu().numpy()   # P(cardiomegaly)
        probs.append(p); ys.append(y.numpy())
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    pred = (probs >= 0.5).astype(int)
    out = {"n": int(len(ys)), "n_pos": int(ys.sum()), "n_neg": int((ys == 0).sum())}
    out["auroc"] = float(roc_auc_score(ys, probs)) if out["n_pos"] and out["n_neg"] else float("nan")
    out["recall_pos"] = float(recall_score(ys, pred, pos_label=1, zero_division=0))   # sensitivity
    out["recall_neg"] = float(recall_score(ys, pred, pos_label=0, zero_division=0))   # specificity
    out["acc"] = float((pred == ys).mean())
    tn, fp, fn, tp = confusion_matrix(ys, pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="'real' or path to a synthetic class-folder root")
    ap.add_argument("--tag", required=True, help="label for this run (output json name)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--test-split", default="test", choices=["test", "validate", "train"],
                    help="which manifest split to evaluate on (default: locked test)")
    ap.add_argument("--limit", type=int, default=None, help="cap #train images (smoke test)")
    a = ap.parse_args()

    print(f"device={DEV}  res={RES}  batch={BATCH}  epochs={a.epochs}  seed={SEED}", flush=True)
    train_ds = build_train_ds(a.train, limit=a.limit)
    test_ds = MimicManifestDataset(a.test_split, train_aug=False)
    print(f"TRAIN: {len(train_ds)} imgs ({train_ds.pos} cardiomegaly / "
          f"{len(train_ds)-train_ds.pos} normal)"
          + (f"  [{train_ds.missing} manifest rows skipped — not on disk]"
             if isinstance(train_ds, MimicManifestDataset) else ""), flush=True)
    print(f"TEST  ({a.test_split}): {len(test_ds)} imgs ({test_ds.pos} cardiomegaly / "
          f"{len(test_ds)-test_ds.pos} normal)  [{test_ds.missing} rows not on disk]", flush=True)
    if len(train_ds) == 0 or len(test_ds) == 0:
        print("\n[ABORT] empty train or test set — need more of the download on disk.", flush=True)
        return

    model = train_classifier(train_ds, epochs=a.epochs)
    res = evaluate(model, test_ds)
    res.update({"tag": a.tag, "train_source": a.train, "test_split": a.test_split,
                "n_train": len(train_ds), "epochs": a.epochs})

    out_dir = r"C:\Users\Tonkid\Downloads\tstr\probe_densenet"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n==============================================")
    print(f"  DenseNet probe [{a.tag}]  train={a.train}")
    print(f"  test={a.test_split}  n={res['n']} (pos {res['n_pos']}, neg {res['n_neg']})")
    print(f"  AUROC             = {res['auroc']:.4f}")
    print(f"  recall / sens (+) = {res['recall_pos']:.4f}   <- rare-class recall")
    print(f"  specificity  (-)  = {res['recall_neg']:.4f}")
    print(f"  accuracy          = {res['acc']:.4f}")
    print(f"  confusion         = {res['confusion']}")
    print("==============================================")


if __name__ == "__main__":
    main()
