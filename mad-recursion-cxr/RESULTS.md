# Cardiomegaly MAD study — data snapshot, 2026-08-06 07:30

Everything on disk as of now. Source files listed per table so every number is re-derivable.

---

## 1. Datasets

| set | path | count | notes |
|---|---|---|---|
| device-filtered balanced | `tstr/data_cardio_bal_clean/train/` | 12,528 / class | keyword filter only |
| **GAN training set** | `tstr/data_cardio_final_crop/train/` | **7,047 / class** | union filter (keyword ∪ CheXpert ∪ NegBio) + autocrop |
| held-out eval (cardiomegaly) | (derived, manifest) | 9,617 | device-free, **GAN never trained on it** — 1:1 balancing leftovers |
| probe eval — validate | `tstr/data_probe/validate/` | 865 / class | device-filtered, cropped, balanced |
| probe eval — test | `tstr/data_probe/test/` | 867 / class | **locked**, untouched |
| gen-1 pool | `runs/mad_gen1/pool/cardiomegaly/` | 16,249 | partial (target 100k) |
| gen-0 pool | `runs/cardio_final_crop_p06/gen0/pool100k/cardiomegaly/` | 100,000 | used for the filter bakeoff |

Device prevalence: 53.8% overall; cardiomegaly 56% vs normal 47% → devices are a shortcut,
which is why both the training set and every probe eval set are device-filtered.
Text-based filtering ceiling is 56.6% — unreported hardware survives. State this as a limitation.

---

## 2. GAN — final model

`runs/cardio_final_crop_p06/gen0/checkpoints/final.pt`, 256px class-conditional,
ADA capped at **p_max = 0.6**, flip/rotate disabled, 1000 kimg.

### 2a. ADA cap sweep (`results/best_ada_p.json`, kimg 500)

| p_max | FID | KID |
|---|---|---|
| 0.4 | 60.66 | 0.06455 |
| **0.6** | **58.59** | **0.05969** |
| 0.8 | 61.76 | 0.06668 |
| 1.0 | 74.23 | 0.08430 |

### 2b. Controlled capped-vs-adaptive A/B (`results/capped_vs_adaptive_kimg600.json`, kimg 600, n=5000)

| run | FID | KID | coverage | density | recall | vendi |
|---|---|---|---|---|---|---|
| **capped p_max=0.6** | **37.38** | **0.03479 ±0.00154** | **0.4536** | 0.4391 | 0.1048 | 4.519 |
| adaptive p_max=1.0 | 43.21 | 0.04252 ±0.00180 | 0.3838 | 0.3722 | 0.0820 | 4.182 |

KID gap is ~5× the error bar. The cap is justified, not arbitrary.

### 2c. Convergence (cropped reference, 2500/class)

| kimg | FID | KID | file |
|---|---|---|---|
| 500 | 44.60 | 0.04261 | `crop7k_p06_kimg500_fidkid.json` |
| 800 | 29.14 | **0.02257** | `crop7k_p06_kimg800_fidkid.json` |
| 900 | 29.25 | 0.02459 | `crop7k_p06_kimg900_vs_1000.json` |
| 1000 | **27.29** | 0.02404 | `crop7k_p06_kimg1000_fidkid.json` |

Non-monotone after 800 → plateaued at ~800 kimg. 1000 kimg is the locked checkpoint.

---

## 3. Filter ranking — the definitive bakeoff

`results/filter_ranking_newgan.json`. GAN kimg1000, 100k cardiomegaly pool, **every row keeps n = 7,047**,
ranked by coverage. select = the 7,047 reals the GAN trained on; **eval = the 9,617 held-out device-free
cardiomegaly the GAN never saw** (image-disjoint) → measures generalisation, herding cannot have gamed it.

| # | filter | coverage | KID | FD (=FID) | density | recall | vendi | copy_rate |
|---|---|---|---|---|---|---|---|---|
| 1 | **PCA-Mahalanobis 10% + herding** | **0.7314** | 0.00350 | 14.27 | 1.068 | 0.218 | 4.96 | 0.057 |
| 2 | herding | 0.7310 | **0.00340** | 14.52 | 1.074 | 0.224 | 5.12 | 0.058 |
| 3 | PCA-Mahalanobis | 0.7134 | 0.03019 | 34.31 | 2.408 | 0.045 | 3.42 | **0.135** |
| 4 | PCA-kNN | 0.5702 | 0.04847 | 50.63 | 1.903 | 0.034 | 3.65 | **0.188** |
| 5 | DRS (density-ratio) | 0.5559 | 0.02688 | 27.54 | 0.750 | 0.188 | 4.57 | 0.024 |
| 6 | top-k by D score | 0.5428 | 0.03705 | 35.15 | 0.819 | 0.130 | 4.08 | 0.039 |
| 7 | none (unfiltered) | 0.5243 | 0.02600 | 27.70 | 0.638 | 0.209 | 4.85 | 0.013 |
| 8 | **random (control)** | 0.5239 | 0.02524 | 27.36 | 0.638 | 0.198 | 4.85 | 0.013 |
| 9 | k-means balanced | 0.5133 | 0.02745 | 28.95 | 0.591 | 0.220 | 4.94 | 0.011 |
| 10 | truncation ψ=0.7 | 0.4979 | 0.04385 | 40.40 | 1.068 | 0.068 | 3.77 | 0.016 |
| 11 | DPP | 0.0755 | 0.04952 | 51.78 | 0.029 | 0.785 | 6.79 | 0.000 |
| 12 | k-center greedy | 0.0601 | 0.07410 | 67.81 | 0.027 | 0.831 | 6.51 | 0.000 |

### Findings (paper-ready)

1. **Herding wins on every axis** vs random: coverage 0.524 → 0.731, KID −86%, FD −47%.
   It wins on coverage, which it does *not* optimise, on data the generator never saw → the gain is real.
2. **PCA-Mahalanobis 10% + herding ≈ herding alone** (0.7314 vs 0.7310 — statistically identical).
   The gate is kept because it removes visible junk, not because it moves the metrics.
3. **Per-sample density filters MEMORISE.** copy_rate 0.135 (Mahalanobis) / 0.188 (kNN) vs 0.05 baseline.
   "Keep images nearest the real distribution" ≈ "keep the near-copies" → disqualifying for a medical paper.
   Running Mahalanobis *before* herding removes its own problem (0.057).
4. **Diversity-only selectors collapse.** k-center greedy and DPP land at coverage 0.06–0.076, an order of
   magnitude *below* random. They maximise spread, so they select the extremes (junk); density 0.027 confirms it.
5. **Their high recall (0.79–0.83) is an artifact** — Kynkäänniemi recall measures reals inside the *fake*
   manifold, so wildly dispersed fakes get huge k-NN spheres that swallow everything. High recall + near-zero
   coverage = degenerate, not diverse. **Footnote this** or a reviewer will misread row 12 as best diversity.
6. **GAN-native baselines do nothing.** DRS barely beats random with a worse KID; top-k by D is worse;
   truncation ψ=0.7 is *below* random on coverage → truncation is not a substitute for filtering.

**Decision: PCA-Mahalanobis 10% → herding.**

Documented deviations: DRS is density-ratio resampling (prob ∝ exp(D_logit)), not exact Azadi acceptance.
C2ST, UMAP, t-SNE, TSTR excluded for compute. eval is image-disjoint, not patient-disjoint.

---

## 4. MAD recursion — generation 0

`results/mad_recursion.json`. gen-0 = the real-data baseline every later generation is measured against.

| metric | gen-0 | reference |
|---|---|---|
| FID | 28.674 | — |
| KID | 0.026193 ±0.001484 | — |
| coverage | 0.4349 | higher = better |
| density | 0.5637 | — |
| recall | 0.2273 | see caveat #5 above |
| vendi | 4.8635 | — |
| copy_rate | **0.0104** | baseline 0.05 → no memorisation |
| CTR cardiomegaly | 0.5428 | real 0.5325 |
| CTR normal | 0.4591 | real 0.4558 |
| **CTR gap** | **+0.0837** | real **+0.0767** |
| CTR gap retained | **109%** | class separation fully intact |
| CTR seg-fail rate | 0.003 | anatomy sound |
| **AUROC DenseNet-121** | **0.8527** | recall_cardio 0.765 |
| AUROC DINOv2 | *pending* | — |

CTR real reference (`results/ctr_real_reference.json`): cardiomegaly 0.5325 ±0.0937, normal 0.4558 ±0.0711,
0% segmentation failure.

DenseNet AUROC 0.8527 sits inside the published MIMIC cardiomegaly band (~0.80–0.87) → the device filter and
autocrop did not damage the disease signal, and the probe is not riding a hardware shortcut. **This is the
ceiling the MAD curve degrades away from.**

---

## 5. Status / what is next

**Done:** dataset build, device filter, autocrop, held-out + probe splits, gen-0 GAN to 1000 kimg,
convergence check, ADA cap A/B, 12-filter ranking, gen-0 metrics + CTR + DenseNet probe.

**Running:** DINOv2 feature extraction for the gen-0 probe (last step of Stage F).
Watch for `tstr/filters/feat_cache/dinov2_*.npy` (~43 MB then ~5 MB), then `saved -> results/mad_recursion.json`.

**Next:** gen-1 — pool is at 16,249 / 100,000, so resume Stage A, then filter → train → score.

**Known issue, fixed in the notebook but not yet exercised:** the DenseNet probe ran with cuDNN/MIOpen
disabled for the *whole* model, which put all 121 conv layers on the native path — 20 epochs took ~7 hours.
[mad_recursion.ipynb](mad_recursion.ipynb) Stage F now swaps in a `NativeBN` subclass so **only BatchNorm**
bypasses MIOpen (MIOpen's BN kernels genuinely fail on gfx1101 with `miopenStatusUnknownError`).
Convs keep their MIOpen kernels → expect **~45–60 min** per generation instead of 7 hours.

**Stale process:** PID 9408, an orphaned diagnostic script, is holding 0.54 GB VRAM and a CPU core.
Kill it or reboot when the current cell finishes.
