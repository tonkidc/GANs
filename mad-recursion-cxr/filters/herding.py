"""Greedy MMD selection (kernel herding). Selects the subset of a fake pool that minimises MMD to the
real set — i.e. matches the real distribution as a whole, tails included, instead of scoring each
image alone. This is the one property no per-sample filter can have.

At step t (t items already chosen) score every unchosen candidate x and take the argmax:

    score(x) = A(x) - ( B_sum(x) + 0.5 * k(x,x) ) / (t + 1)

    A(x)     = mean_j k(x, r_j)          attraction to the real set   (constant, precomputed)
    B_sum(x) = sum_{i<=t} k(x, s_i)      repulsion from what's chosen  (running accumulator)
    k(x,x)   = self-similarity           (constant vector)

This is the EXACT greedy minimiser of the biased MMD^2 estimator. The 0.5*k(x,x) term matters for the
polynomial kernel (non-constant diagonal); for RBF the diagonal is 1 and the term is a harmless
constant. Do NOT use the textbook A - (1/t)*B form — it assumes a constant diagonal.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as K


def _kmat(A, B, kernel, kw):
    if kernel == "poly":
        return K.poly_kernel(A, B, degree=kw.get("degree", 3))
    if kernel == "rbf":
        return K.rbf_kernel(A, B, sigma=kw["sigma"])
    raise ValueError(f"unknown kernel: {kernel}")


def _diag(A, kernel, kw):
    if kernel == "poly":
        return K.poly_diag(A, degree=kw.get("degree", 3))
    return torch.ones(A.shape[0], dtype=torch.float64, device=A.device)   # rbf: k(x,x)=1


def _mean_rows_chunked(Fa, Fb, kernel, kw, chunk=4096):
    """mean_j k(Fa_i, Fb_j) per row i -> (len(Fa),), WITHOUT materialising the full (Fa x Fb) matrix.
    At 100k pool the full matrix is ~15 GB (float64) and OOMs a 16 GB GPU; chunking keeps peak tiny."""
    out = torch.empty(Fa.shape[0], dtype=torch.float64, device=Fa.device)
    for i in range(0, Fa.shape[0], chunk):
        out[i:i + chunk] = _kmat(Fa[i:i + chunk], Fb, kernel, kw).mean(1)
    return out


def _grand_mean_chunked(Fa, Fb, kernel, kw, chunk=4096):
    """mean over ALL pairs k(Fa_i, Fb_j) -> scalar, chunked over Fa rows (memory-safe at 100k)."""
    s, n = 0.0, 0
    for i in range(0, Fa.shape[0], chunk):
        blk = _kmat(Fa[i:i + chunk], Fb, kernel, kw)
        s += float(blk.sum()); n += blk.numel()
    return s / n


def _mmd2(sel_feats, F_real, krr_mean, kernel, kw):
    """Biased MMD^2 of the selected set vs real: mean k(S,S) + mean k(R,R) - 2 mean k(S,R)."""
    kss = _grand_mean_chunked(sel_feats, sel_feats, kernel, kw)
    ksr = _grand_mean_chunked(sel_feats, F_real, kernel, kw)
    return float(kss + krr_mean - 2.0 * ksr)


def greedy_mmd_select(F_pool, F_real, n_keep, kernel="poly", kernel_kwargs=None,
                      device="cuda", log_every=100):
    """Return (selected_indices ndarray, diagnostics dict). Selects n_keep of F_pool minimising MMD
    to F_real. F_pool/F_real are (N,d) / (m,d) arrays or tensors."""
    kw = dict(kernel_kwargs or {})
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    Fp = torch.as_tensor(F_pool, dtype=torch.float64, device=dev)
    Fr = torch.as_tensor(F_real, dtype=torch.float64, device=dev)
    M, m = Fp.shape[0], Fr.shape[0]
    if n_keep > M:
        raise ValueError(f"n_keep={n_keep} exceeds pool size {M}")

    if dev != "cpu":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    A = _mean_rows_chunked(Fp, Fr, kernel, kw)            # (M,) attraction, precomputed once (chunked)
    kxx = _diag(Fp, kernel, kw)                           # (M,) self-similarity
    krr_mean = _grand_mean_chunked(Fr, Fr, kernel, kw)    # constant for the MMD trace (chunked)

    B_sum = torch.zeros(M, dtype=torch.float64, device=dev)
    chosen_mask = torch.zeros(M, dtype=torch.bool, device=dev)
    chosen = []
    trace = []

    for t in range(n_keep):
        score = A - (B_sum + 0.5 * kxx) / (t + 1)
        score[chosen_mask] = -float("inf")
        best = int(torch.argmax(score).item())
        chosen.append(best)
        chosen_mask[best] = True
        B_sum += _kmat(Fp, Fp[best:best + 1], kernel, kw).squeeze(1)   # one (M,) matvec/step
        if (t + 1) % log_every == 0 or t == n_keep - 1:
            trace.append((t + 1, _mmd2(Fp[chosen], Fr, krr_mean, kernel, kw)))

    diag = {
        "kernel": kernel,
        "sigma": kw.get("sigma"),
        "n_pool": int(M),
        "n_real": int(m),
        "n_keep": int(n_keep),
        "mmd2_trace": trace,
        "mmd2_start": trace[0][1] if trace else None,
        "mmd2_end": trace[-1][1] if trace else None,
        "wall_time_s": round(time.time() - t0, 2),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if dev != "cpu" else None,
        "device": dev,
    }
    if dev != "cpu":
        del Fp, Fr, A, kxx, B_sum, chosen_mask
        torch.cuda.empty_cache()
    return np.array(chosen, dtype=np.int64), diag
