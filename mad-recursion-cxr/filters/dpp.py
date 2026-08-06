"""Determinantal Point Process (k-DPP) selection — the diversity-first counterpart to herding.

Where herding (greedy MMD) picks the subset that best MATCHES the real distribution, a DPP picks the
subset that best COVERS the feature space: it maximises det(L_S), the squared volume the chosen vectors
span, so near-duplicates repel each other determinantally. That is the property the MAD study cares
about on the other axis — preserving atypical / tail patients instead of collapsing onto the mode.

We use a quality-diversity DPP kernel (Kulesza & Taskar):

    L = diag(q) . S . diag(q)

  * S = an RBF similarity in the SAME feature space herding/KID use (same Inception embedding), so the two
    methods differ in OBJECTIVE, not in what they see. RBF (not the poly/KID kernel) because a DPP needs a
    BOUNDED, well-conditioned similarity — the poly kernel's cubic, unbounded values make det(L_S) blow up
    numerically. RBF has unit diagonal and lies in [0,1]: exactly what greedy MAP wants.
  * q_i = real-affinity quality = mean_j k_rbf(x_i, r_j), rescaled to max 1 (a global scale on q leaves the
    selected subset unchanged but keeps the Cholesky well-conditioned). So a picked fake is both realistic
    (high q) and non-redundant (S repulsion). quality="uniform" drops q -> pure diversity, a max-spread
    baseline that ignores realism.

Selection is greedy MAP inference (Chen et al. 2018): incremental Cholesky, O(M . k^2), never materialising
the full M x M kernel (one kernel row per step, like herding's B_sum update).
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as K
from herding import _kmat, _diag, _mmd2


def greedy_dpp_select(F_pool, F_real, n_keep, kernel="rbf", kernel_kwargs=None,
                      quality="real_affinity", q_vector=None, device="cuda", log_every=100):
    """Return (selected_indices ndarray, diagnostics dict). Selects n_keep of F_pool maximising
    det(L_S) for the quality-diversity kernel L. F_pool/F_real are (N,d)/(m,d) arrays or tensors.
    For kernel='rbf', sigma is taken from kernel_kwargs or the median heuristic on F_real.

    quality:
      'real_affinity' (default) — q_i = mean_j k(x_i, r_j), a realism score in the kernel space.
      'uniform'                 — q_i = 1, pure diversity (max-spread baseline, ignores realism).
      'vector'                  — q from the caller-supplied `q_vector` (len M). This is where an
                                  external REALISM score plugs in (e.g. PCA-Mahalanobis, sign-flipped so
                                  higher = more real). HARD RULE (spec): q must be a realism/density score,
                                  NEVER a classifier-confidence score (no C2ST / discriminator P(fake)).
                                  q_vector must be finite and >= 0."""
    kw = dict(kernel_kwargs or {})
    if quality not in ("real_affinity", "uniform", "vector"):
        raise ValueError(f"unknown quality: {quality}")
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    Fp = torch.as_tensor(F_pool, dtype=torch.float64, device=dev)
    Fr = torch.as_tensor(F_real, dtype=torch.float64, device=dev)
    M, m = Fp.shape[0], Fr.shape[0]
    if n_keep > M:
        raise ValueError(f"n_keep={n_keep} exceeds pool size {M}")
    if kernel == "rbf" and kw.get("sigma") is None:      # median heuristic on the real set
        kw["sigma"] = K.median_sigma(Fr)

    if dev != "cpu":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    kxx = _diag(Fp, kernel, kw).clamp_min(1e-12)          # (M,) self-sim -> normalise S to unit diagonal
    sqrt_kxx = kxx.sqrt()
    if quality == "uniform":
        q = torch.ones(M, dtype=torch.float64, device=dev)
    elif quality == "vector":
        if q_vector is None:
            raise ValueError("quality='vector' requires q_vector (a per-pool realism score, len M)")
        q = torch.as_tensor(q_vector, dtype=torch.float64, device=dev).reshape(-1)
        if q.shape[0] != M:
            raise ValueError(f"q_vector length {q.shape[0]} != pool size {M}")
        if not torch.isfinite(q).all() or float(q.min()) < 0:
            raise ValueError("q_vector must be finite and >= 0 (realism score; sign-flip a distance first)")
        q = q / q.max().clamp_min(1e-12)                          # global rescale -> keep Cholesky sane
    else:
        q = _kmat(Fp, Fr, kernel, kw).mean(1).clamp_min(0.0)      # (M,) real-affinity
        q = q / q.max().clamp_min(1e-12)                          # global rescale -> keep Cholesky sane
    krr_mean = _kmat(Fr, Fr, kernel, kw).mean()           # constant for the MMD readout

    # Greedy MAP for k-DPP (Chen et al. 2018). d2_i is the marginal gain (Schur-complement diagonal);
    # it starts at L_ii = q_i^2 (S normalised -> unit diagonal). c holds the incremental Cholesky columns.
    # GAIN_FLOOR guards saturation: if the remaining pool spans no more volume (gains -> 0, e.g. n_keep
    # exceeds the data's effective diversity) we stop the Cholesky and fill the rest by quality, so the
    # sqrt(gain) division can never explode.
    GAIN_FLOOR = 1e-10
    d2 = (q * q).clone()
    c = torch.zeros(M, n_keep, dtype=torch.float64, device=dev)
    chosen_mask = torch.zeros(M, dtype=torch.bool, device=dev)
    chosen = []
    logdet = 0.0
    logdet_trace = []
    min_gain = float("inf")
    n_diverse = n_keep

    for g in range(n_keep):
        sel = int(torch.argmax(d2).item())               # chosen items are set to -inf, never re-picked
        gain = float(d2[sel])
        if gain < GAIN_FLOOR:                             # diversity exhausted -> fill remainder by quality
            n_diverse = g
            fill_scores = q.clone()
            fill_scores[chosen_mask] = -float("inf")
            need = n_keep - g
            extra = torch.topk(fill_scores, need).indices.tolist()
            chosen.extend(int(i) for i in extra)
            break
        min_gain = min(min_gain, gain)
        logdet += float(np.log(gain))
        chosen.append(sel)
        chosen_mask[sel] = True

        raw = _kmat(Fp[sel:sel + 1], Fp, kernel, kw).squeeze(0)    # (M,) k(sel, .)
        Lrow = q[sel] * q * (raw / (sqrt_kxx[sel] * sqrt_kxx))     # (M,) row of L (normalised S + quality)
        proj = c[:, :g] @ c[sel, :g] if g > 0 else torch.zeros(M, dtype=torch.float64, device=dev)
        e = (Lrow - proj) / (gain ** 0.5)
        c[:, g] = e
        d2 = d2 - e * e
        d2[sel] = -float("inf")

        if (g + 1) % log_every == 0 or g == n_keep - 1:
            logdet_trace.append((g + 1, round(logdet, 4)))

    idx = np.array(chosen, dtype=np.int64)
    mmd2_end = _mmd2(Fp[chosen], Fr, krr_mean, kernel, kw)         # shared axis vs herding
    diag = {
        "method": "dpp",
        "quality": quality,
        "kernel": kernel,
        "sigma": kw.get("sigma"),
        "n_pool": int(M),
        "n_real": int(m),
        "n_keep": int(n_keep),
        "logdet_final": round(logdet, 4),
        "logdet_trace": logdet_trace,
        "n_diverse": int(n_diverse),    # picks made by DPP volume (rest, if any, filled by quality)
        "min_gain": (min_gain if min_gain != float("inf") else None),   # >0 confirms a valid greedy run
        "mmd2_end": mmd2_end,           # so it lines up against herding's MMD on one table
        "mean_quality_kept": round(float(q[idx].mean()), 6),
        "wall_time_s": round(time.time() - t0, 2),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if dev != "cpu" else None,
        "device": dev,
    }
    return idx, diag
