"""Wire / lead detection gate — drops fakes whose image is dominated by thin bright line structures
(ECG leads, wires, and the ugly GAN 'wire-crinkle' artifacts the prof flagged).

Detector: a Sato ridge filter (skimage) responds to thin TUBULAR structures at any orientation/curvature
— the right model for wires (Hough only finds straight lines; top-hat mis-fires on texture). We take the
BRIGHT ridges at a fine scale (sigmas 1-2 px, wires are thinner than ribs), and score each image by the
FRACTION of pixels that are strong thin ridges. Threshold that fraction to drop wire-dominated images.

CAVEAT (write this in the method): wires are real in cardiomegaly films, so this gate is meant to catch
BADLY-RENDERED wire artifacts, not to erase the sick-patient look. Tune `max_wire_frac` on the validation
grids so it drops the ugly ones without deleting realistic wire films or normal anatomy (ribs/vessels).
Per-sample + high-recall by design; the set-level selector (herding) still makes the final choice.
"""
import numpy as np
from PIL import Image
from skimage.filters import sato

RES = 256
SIGMAS = (1, 2)                 # fine scales: wires are thinner than ribs
TOP_FRAC = 0.02                 # score = mean strength of the strongest 2% of ridges (absolute, comparable)


def wire_score(gray_uint8, res=RES):
    """Absolute wire-content score: mean Sato response over the strongest 2% of thin BRIGHT ridges.
    Higher = stronger/denser thin lines (wires, leads, crinkle). Comparable across images (not per-image
    normalised), so it ranks and thresholds cleanly. Fine scales downweight the broader ribs."""
    a = np.asarray(gray_uint8, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != (res, res):
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((res, res)), np.float32)
    a = a / 255.0
    r = sato(a, sigmas=SIGMAS, black_ridges=False)          # bright tubular ridges
    flat = np.sort(r.ravel())
    top = flat[int((1.0 - TOP_FRAC) * len(flat)):]
    return float(top.mean())


def wire_coverage(gray_uint8, res=RES):
    """Wire DENSITY: mean thin-bright-ridge response over the WHOLE frame. High = thin lines spread
    everywhere (dense wire tangle / crinkle); low = a few localised leads. This is the signal that
    separates 'ugly wire tangle on a coherent chest' (which XRV-realism keeps) from a clean single-lead
    film — wire_score (peak strength) can't, because both can have one strong ridge."""
    a = np.asarray(gray_uint8, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != (res, res):
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((res, res)), np.float32)
    r = sato(a / 255.0, sigmas=SIGMAS, black_ridges=False)
    return float(r.mean())


def score_paths(paths, res=RES, fn=None):
    fn = fn or wire_score
    return np.array([fn(np.asarray(Image.open(p).convert("L")), res) for p in paths], dtype=np.float32)


def wire_keep_mask(paths, max_score):
    """Keep-mask (bool): True = keep, False = drop (wire_score above max_score). High recall: only the
    most wire-dominated images fall. Returns (mask, scores) so the caller can log/inspect."""
    s = score_paths(paths)
    return (s <= max_score), s
