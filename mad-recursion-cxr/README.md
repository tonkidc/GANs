# Model Autophagy Disorder in class-conditional chest X-ray synthesis

Code for a self-consuming ("MAD") recursion study on **MIMIC-CXR-JPG**: a class-conditional
StyleGAN2-ADA is trained on real cardiomegaly / no-finding frontal radiographs, its own filtered
output becomes the next generation's training set, and the collapse is tracked with distributional,
clinical, and downstream-classifier metrics.

The central question is not *"do the images still look real"* — it is **"is the disease still there."**

## What is in here

| path | what |
|---|---|
| [`mad_recursion.ipynb`](mad_recursion.ipynb) | **the whole pipeline**, raw MIMIC → MAD curve. One generation per run; `GEN` is the only edit. |
| [`RESULTS.md`](RESULTS.md) | every number produced so far, with the source JSON named per table |
| `filters/` | the selection methods, incl. the KID-based herding selector |
| `tests/` | unit tests for herding, DPP, PRDC/Vendi, and the anatomical gates |
| `cardio_metrics.py` | FID, KID, coverage/density/recall, Vendi, copy_rate, cardiothoracic ratio |
| `filter_ranking_newgan.py` | the 12-filter ranked bake-off |
| `results/` | aggregate metric JSONs — numbers only, no image-derived data |

## Data — read this first

**No data is included and none may be added.** MIMIC-CXR-JPG is credentialed-access under a
[PhysioNet Data Use Agreement](https://physionet.org/content/mimic-cxr-jpg/). You must complete
CITI training and sign the DUA yourself. The `.gitignore` blocks every data directory, image
extension, feature cache and CSV, because MIMIC filenames themselves embed `subject_id` and
`study_id`.

The notebook regenerates all derived sets from your own licensed copy.

## Pipeline

```
Step 0a  manifest + patient-disjoint split assertion
Step 0b  decode to 256px PNG
Step 0c  device filter -> 1:1 balance -> autocrop      -> 7,047/class
Step 0d  held-out eval set (never seen by the GAN)     -> 9,617
Step 0e  probe eval sets (validate + test)             -> 865 / 867 per class
--------------------------------------------------------------------
Stage A  generate a 100k pool from generation N-1
Stage B  PCA-Mahalanobis 10% -> herding                -> 7,047/class
Stage C  montage
Stage D  train generation N (identical config every generation)
Stage E  FID / KID / coverage / density / recall / Vendi / copy_rate
Stage E2 cardiothoracic ratio + CTR gap + segmentation failure rate
Stage F  DenseNet-121 and DINOv2 ViT-B/14 probes -> AUROC
```

### Three design decisions worth knowing

**Devices are filtered out.** Support hardware (pacemakers, ICDs, lines, sternal wires) appears in
56% of cardiomegaly films but only 47% of normals, so a classifier can score on hardware instead of
heart size. A report-text filter (keyword ∪ CheXpert ∪ NegBio) removes what it can — but it tops out
at 56.6% recall because unreported devices exist. **The training set is device-reduced, not
device-free**, and that is stated as a limitation rather than papered over.

**ADA is capped at p ≤ 0.6, flip and rotate disabled.** Uncapped, `p` runs to ~1.0 and the generator
learns the *augmented* distribution — flipped hearts, rotated films, black translation wedges. Heart
laterality is diagnostic, so flip is never valid here. A controlled A/B at 600 kimg puts capped ahead
on every metric, with a KID gap ~5× the error bar.

**Filters are ranked, not assumed.** Twelve selection methods were compared at matched *n* = 7,047
against held-out data the generator never saw. Two results drove the choice: herding wins on coverage
(0.524 → 0.731) *and* KID (−86%), and per-sample density filters **memorise** — copy_rate 0.135–0.188
against a 0.05 baseline, which is a privacy failure in a medical dataset. Full table in
[`RESULTS.md`](RESULTS.md).

> **Caveat for readers of the ranking table:** k-center greedy and DPP show high Kynkäänniemi *recall*
> (0.79–0.83) with near-zero coverage. That is an artifact, not diversity — recall measures reals
> falling inside the *fake* manifold, so a wildly dispersed set gets huge k-NN spheres that swallow
> everything. Read coverage, not recall.

## Requirements

Built and run on ROCm (RX 7800 XT / gfx1101, Windows), but nothing is AMD-specific except two
workarounds noted in the notebook.

```
torch 2.9.1          torchvision        torchmetrics 1.9.0
timm                 torchxrayvision 1.5.2
scikit-learn         numpy  scipy  pillow  matplotlib
```

Plus a class-conditional StyleGAN2-ADA package (`stylegan2_ada_cond`), a modified derivative of
[NVIDIA's StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), which carries the
NVIDIA Source Code License and is **not** redistributed here.

### ROCm notes

- `MIOPEN_FIND_MODE` must be `NORMAL` **before** the StyleGAN package is imported — it calls
  `setdefault("FAST")` at import, and MIOpen reads the variable once at init. Getting this wrong
  silently drops R1 to native convolutions: 24.4 s per firing instead of 0.71 s.
- MIOpen's **BatchNorm** kernels fail on gfx1101 (`miopenStatusUnknownError`). The DenseNet probe
  swaps in a `NativeBN` subclass that bypasses MIOpen *at the BatchNorm layer only*. Disabling the
  backend around the whole model also strips MIOpen from all 121 conv layers and the probe will not
  finish in reasonable time.

## Status

Generation 0 (real-data baseline) is complete: FID 28.67, KID 0.0262, coverage 0.435,
copy_rate 0.010, **CTR gap +0.0837 vs a real +0.0767 (109% retained)**, **DenseNet-121 AUROC 0.8527**.
Generation 1 is in progress. See [`RESULTS.md`](RESULTS.md).

## License

Code: not yet chosen — add one before making this public.
Data: not included; governed by the PhysioNet DUA.
