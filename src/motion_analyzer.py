"""
motion_analyzer.py — Real Optical Flow + Motion Feature Extraction

Extracts a 48-dimensional motion feature vector from consecutive video frame
pairs using dense Farneback optical flow + sparse Lucas-Kanade corner tracking.
Features distinguish real camera footage from AI-generated video.

  Real camera footage:
    - Natural micro-vibration / camera shake (corner > center motion)
    - Smooth, physically-consistent acceleration (low temporal jitter)
    - Coherent flow direction (objects move together)
    - No periodic grid artifacts in flow spectrum
    - Stable LK corner tracking with consistent motion

  AI-generated video (GAN / Diffusion):
    - Near-zero or unnaturally uniform motion
    - Sudden velocity jumps between frames (no inertia)
    - Periodic grid artifacts in FFT of flow magnitude
    - No camera shake (vibration_score ≈ 0)
    - Over-smoothed flow (too low spatial Laplacian variance)
    - Erratic/inconsistent corner tracking

Feature vector layout (MOTION_FEATURE_DIM = 48):
  ── Dense flow (Farneback) ──
  [0]     mean_magnitude       — average motion speed (px/frame)
  [1]     std_magnitude        — motion speed variability
  [2]     p95_magnitude        — robust peak speed
  [3]     max_magnitude        — absolute peak speed
  [4]     motion_coverage      — fraction of pixels with significant motion
  [5]     flow_coherence       — resultant vector length [0,1]
  [6]     direction_entropy    — Shannon entropy of 8-bin direction histogram
  [7]     spatial_smoothness   — Laplacian variance / mean (low=AI)
  [8]     hf_spatial_ratio     — high-frequency energy fraction in flow FFT
  [9]     vibration_score      — std/mean of flow magnitude
  [10]    corner_vs_center     — shake signature: (corner - center) / center
  [11]    temporal_accel       — |mean_mag_t - mean_mag_{t-1}|
  [12-19] direction_hist[8]    — normalized 8-bin (45°) flow direction histogram
  [20]    h_symmetry           — left–right flow symmetry [0,1]
  [21]    v_symmetry           — top–bottom flow symmetry [0,1]
  [22]    gan_grid_score       — FFT peak-to-mean in mid-freq band
  [23]    zero_motion_fraction — fraction of pixels with flow < 0.1 px/frame
  ── Sparse LK corner tracking ──
  [24]    corner_motion_std    — std of LK tracked corner displacements
  [25]    corner_motion_mean   — mean LK corner displacement
  [26]    corner_track_ratio   — fraction of corners successfully tracked
  [27]    corner_flow_coherence— resultant vector of corner motion directions
  ── Flow gradient sharpness ──
  [28]    flow_grad_x_rms      — RMS of ∂fx/∂x (horizontal sharpness)
  [29]    flow_grad_y_rms      — RMS of ∂fy/∂y (vertical sharpness)
  [30]    laplacian_fx_rms     — Laplacian RMS of x-flow (discontinuity measure)
  [31]    laplacian_fy_rms     — Laplacian RMS of y-flow
  ── Affine motion decomposition ──
  [32]    translation_x        — mean flow x (normalized)
  [33]    translation_y        — mean flow y (normalized)
  [34]    curl_mean            — mean |∂fy/∂x - ∂fx/∂y| (rotation magnitude)
  [35]    divergence_mean      — mean |∂fx/∂x + ∂fy/∂y| (zoom/contraction)
  ── Multi-scale flow consistency ──
  [36]    scale_inconsistency  — mean |mag - upsampled_coarse_mag|
  [37]    scale_inconsistency_var — variance of scale mismatch
  [38]    coarse_fine_ratio    — |coarse_mean - fine_mean| / (fine_mean + eps)
  [39]    edge_motion_ratio    — mean motion at edges / mean motion at flat regions
  ── Motion entropy and complexity ──
  [40]    flow_entropy         — entropy of flow magnitude histogram (16-bin)
  [41]    opposing_flow_frac   — fraction of pixels with flow opposing mean dir
  [42]    dominant_freq        — dominant frequency in row-mean flow FFT
  [43]    temporal_smoothness  — temporal_accel / mean_magnitude (smoothness)
  ── Frame-flow coupling ──
  [44]    edge_motion_abs      — mean flow magnitude at edge pixels (normalized)
  [45]    flat_motion_abs      — mean flow magnitude at flat pixels (normalized)
  [46]    edge_motion_std      — std of motion within edge regions (normalized)
  [47]    background_fraction  — fraction of near-zero motion in flat regions
"""

import cv2
import numpy as np
from typing import List, Union

try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

MOTION_FEATURE_DIM = 48

# Internal processing resolution
_FLOW_H = 512
_FLOW_W = 512

# Farneback parameters — tuned for _FLOW_H×_FLOW_W frames
_FB_PYR_SCALE  = 0.5
_FB_LEVELS     = 4
_FB_WINSIZE    = 15
_FB_ITERATIONS = 3
_FB_POLY_N     = 7
_FB_POLY_SIGMA = 1.5

# Minimum flow magnitude considered "motion" (px/frame at _FLOW_H×_FLOW_W scale)
_MOTION_THRESH = 0.5


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _to_gray_256(frame) -> np.ndarray:
    """Convert PIL Image or (H,W,3) uint8 ndarray to float32 grayscale 256×256."""
    if _HAS_PIL and isinstance(frame, _PIL_Image.Image):
        arr = np.array(frame.convert('RGB'))
    else:
        arr = np.asarray(frame)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (_FLOW_W, _FLOW_H)).astype(np.float32)


# ---------------------------------------------------------------------------
#  Core: single frame-pair
# ---------------------------------------------------------------------------

def extract_pair_features(prev_frame, curr_frame,
                          prev_mean_mag: float = None) -> np.ndarray:
    """
    Compute the 24-dim motion feature vector for one consecutive frame pair.

    Args:
        prev_frame      : PIL Image or (H, W, 3) uint8 ndarray
        curr_frame      : PIL Image or (H, W, 3) uint8 ndarray
        prev_mean_mag   : mean flow magnitude of the *previous* pair (for
                          computing temporal acceleration at feature[11]).
                          Pass None for the first pair in a sequence.

    Returns:
        (MOTION_FEATURE_DIM,) float32
    """
    H, W = _FLOW_H, _FLOW_W

    prev_g = _to_gray_256(prev_frame)
    curr_g = _to_gray_256(curr_frame)

    # ── Dense optical flow (Farneback) ───────────────────────────────────
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        _FB_PYR_SCALE, _FB_LEVELS, _FB_WINSIZE,
        _FB_ITERATIONS, _FB_POLY_N, _FB_POLY_SIGMA, 0,
    )   # (H, W, 2)

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)

    feats = np.zeros(MOTION_FEATURE_DIM, dtype=np.float32)

    # ── [0-3] Magnitude statistics ────────────────────────────────────────
    feats[0] = float(mag.mean())
    feats[1] = float(mag.std())
    feats[2] = float(np.percentile(mag, 95))
    feats[3] = float(mag.max())

    # ── [4] Motion coverage ───────────────────────────────────────────────
    feats[4] = float((mag > _MOTION_THRESH).mean())

    # ── [5] Flow coherence — resultant vector length [0, 1] ───────────────
    ang_rad   = np.deg2rad(ang)
    feats[5]  = float(np.sqrt(np.cos(ang_rad).mean()**2 + np.sin(ang_rad).mean()**2))

    # ── [6] Direction entropy of moving pixels ────────────────────────────
    mask = mag > _MOTION_THRESH
    if mask.sum() > 10:
        hist, _ = np.histogram(ang[mask], bins=8, range=(0.0, 360.0))
        hist     = hist.astype(np.float32) + 1e-8
        hist    /= hist.sum()
        feats[6] = float(-np.sum(hist * np.log(hist)))
    else:
        feats[6] = 0.0

    # ── [7] Spatial smoothness — Laplacian variance / mean ────────────────
    # Low value = AI-style over-smooth flow; high value = natural camera shake
    lap      = cv2.Laplacian(mag, cv2.CV_32F)
    feats[7] = float(min(lap.var() / (mag.mean() + 1e-6), 10.0))

    # ── [8] High-frequency spatial energy ratio in flow FFT ───────────────
    fft_mag    = np.abs(np.fft.fft2(mag))
    total_e    = fft_mag.sum() + 1e-8
    hf_e       = fft_mag[H // 4:, :].sum() + fft_mag[:, W // 4:].sum()
    feats[8]   = float(min(hf_e / total_e, 1.0))

    # ── [9] Vibration score (coefficient of variation) ────────────────────
    # Real camera: natural hand-shake → moderate CV; AI video → near 0
    feats[9] = float(mag.std() / (mag.mean() + 1e-6))

    # ── [10] Corner-vs-centre motion ratio ────────────────────────────────
    # Real cameras: corners shake as much as centre (rigid body motion).
    # AI generators: centre may move (foreground) while corners stay still.
    cy, cx       = H // 2, W // 2
    qy, qx       = H // 8, W // 8
    centre_mag   = mag[cy - qy:cy + qy, cx - qx:cx + qx].mean()
    corner_mag   = np.array([
        mag[:qy,   :qx  ].mean(),
        mag[:qy,   -qx: ].mean(),
        mag[-qy:,  :qx  ].mean(),
        mag[-qy:,  -qx: ].mean(),
    ]).mean()
    feats[10] = float((corner_mag - centre_mag) / (centre_mag + 1e-6))

    # ── [11] Temporal acceleration ────────────────────────────────────────
    feats[11] = float(abs(feats[0] - prev_mean_mag)) if prev_mean_mag is not None else 0.0

    # ── [12-19] 8-bin direction histogram (normalized) ────────────────────
    hist8, _ = np.histogram(ang, bins=8, range=(0.0, 360.0))
    hist8     = hist8.astype(np.float32)
    hist8    /= (hist8.sum() + 1e-8)
    feats[12:20] = hist8

    # ── [20] Horizontal (left–right) flow symmetry ────────────────────────
    l_mag    = mag[:, :W // 2].mean()
    r_mag    = mag[:, W // 2:].mean()
    feats[20] = float(1.0 - abs(l_mag - r_mag) / (l_mag + r_mag + 1e-6))

    # ── [21] Vertical (top–bottom) flow symmetry ──────────────────────────
    t_mag    = mag[:H // 2, :].mean()
    b_mag    = mag[H // 2:, :].mean()
    feats[21] = float(1.0 - abs(t_mag - b_mag) / (t_mag + b_mag + 1e-6))

    # ── [22] GAN grid artifact score ─────────────────────────────────────
    h4, w4    = H // 4, W // 4
    mid_spec  = fft_mag[h4:-h4, w4:-w4]
    spec_mean = mid_spec.mean() + 1e-8
    feats[22] = float(min(mid_spec.max() / spec_mean, 20.0) / 20.0)

    # ── [23] Zero-motion fraction ─────────────────────────────────────────
    feats[23] = float((mag < 0.1).mean())

    # ────────────────────────────────────────────────────────────────────
    #  Extended features [24-47]
    # ────────────────────────────────────────────────────────────────────

    # ── [24-27] Sparse Lucas-Kanade corner tracking ───────────────────────
    try:
        prev_u8 = np.clip(prev_g, 0, 255).astype(np.uint8)
        curr_u8 = np.clip(curr_g, 0, 255).astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(
            prev_u8, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=3,
        )
        if corners is not None and len(corners) >= 5:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_u8, curr_u8, corners, None,
                winSize=(15, 15), maxLevel=3,
            )
            good = (st.ravel() == 1)
            if good.sum() >= 5:
                diff = (p1[good] - corners[good]).reshape(-1, 2)
                cm   = np.sqrt((diff ** 2).sum(axis=1))
                feats[24] = float(np.clip(cm.std()    / 5.0, 0.0, 1.0))
                feats[25] = float(np.clip(cm.mean()   / 5.0, 0.0, 1.0))
                feats[26] = float(np.clip(good.sum()  / 200.0, 0.0, 1.0))
                ca        = np.arctan2(diff[:, 1], diff[:, 0])
                feats[27] = float(np.sqrt(np.cos(ca).mean() ** 2 + np.sin(ca).mean() ** 2))
    except Exception:
        pass  # feats[24:28] stay 0

    # ── [28-31] Flow gradient sharpness ─────────────────────────────────
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    gx_rms = float(np.sqrt(np.mean(np.gradient(fx, axis=1) ** 2)))
    gy_rms = float(np.sqrt(np.mean(np.gradient(fy, axis=0) ** 2)))
    lap_fx = float(np.sqrt(np.mean(cv2.Laplacian(fx, cv2.CV_32F) ** 2)))
    lap_fy = float(np.sqrt(np.mean(cv2.Laplacian(fy, cv2.CV_32F) ** 2)))
    feats[28] = float(np.clip(gx_rms / 0.5, 0.0, 1.0))
    feats[29] = float(np.clip(gy_rms / 0.5, 0.0, 1.0))
    feats[30] = float(np.clip(lap_fx / 1.0, 0.0, 1.0))
    feats[31] = float(np.clip(lap_fy / 1.0, 0.0, 1.0))

    # ── [32-35] Affine motion decomposition ──────────────────────────────
    feats[32] = float(np.clip(fx.mean() / 20.0 * 0.5 + 0.5, 0.0, 1.0))   # translation x
    feats[33] = float(np.clip(fy.mean() / 20.0 * 0.5 + 0.5, 0.0, 1.0))   # translation y
    curl = np.gradient(fy, axis=1) - np.gradient(fx, axis=0)
    div  = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
    feats[34] = float(np.clip(float(np.abs(curl).mean()) / 0.5, 0.0, 1.0))
    feats[35] = float(np.clip(float(np.abs(div).mean())  / 0.5, 0.0, 1.0))

    # ── [36-39] Multi-scale flow consistency + edge-motion ───────────────
    H2, W2   = H // 2, W // 2
    mag_down = cv2.resize(mag, (W2, H2))
    mag_up   = cv2.resize(mag_down, (W, H))
    diff_ms  = np.abs(mag - mag_up)
    feats[36] = float(np.clip(diff_ms.mean() / 2.0, 0.0, 1.0))
    feats[37] = float(np.clip(diff_ms.std()  / 2.0, 0.0, 1.0))
    feats[38] = float(np.clip(
        abs(float(mag_down.mean()) - float(mag.mean())) / (float(mag.mean()) + 1e-6),
        0.0, 1.0))
    # Edge-motion ratio: motion at high-gradient regions vs flat regions
    sob_x      = cv2.Sobel(curr_g, cv2.CV_32F, 1, 0, ksize=3)
    sob_y      = cv2.Sobel(curr_g, cv2.CV_32F, 0, 1, ksize=3)
    edge_map   = np.sqrt(sob_x ** 2 + sob_y ** 2)
    eth        = float(np.percentile(edge_map, 75))
    edge_mask  = edge_map > eth
    flat_mask  = ~edge_mask
    edge_mot   = float(mag[edge_mask].mean()) if edge_mask.sum() > 0 else 0.0
    flat_mot   = float(mag[flat_mask].mean()) if flat_mask.sum() > 0 else 0.0
    feats[39]  = float(np.clip(edge_mot / (flat_mot + 1e-6) / 3.0, 0.0, 1.0))

    # ── [40-43] Motion entropy and complexity ────────────────────────────
    hist_m, _ = np.histogram(mag.ravel(), bins=16,
                             range=(0.0, float(mag.max()) + 1e-6))
    hist_m    = hist_m.astype(np.float32) + 1e-8
    hist_m   /= hist_m.sum()
    feats[40] = float(np.clip(-np.sum(hist_m * np.log(hist_m)) / np.log(16), 0.0, 1.0))

    mean_a    = float(np.arctan2(np.sin(ang_rad).mean(), np.cos(ang_rad).mean()))
    ang_diff  = np.abs(np.mod(ang_rad - mean_a + np.pi, 2 * np.pi) - np.pi)
    feats[41] = float((ang_diff > np.pi / 2).mean())

    row_mf    = mag.mean(axis=1)
    fft_row   = np.abs(np.fft.rfft(row_mf))
    fft_row[0] = 0.0
    feats[42] = (float(np.argmax(fft_row) / max(len(fft_row) - 1, 1))
                 if fft_row.sum() > 1e-10 else 0.0)

    if prev_mean_mag is not None and feats[0] > 1e-6:
        feats[43] = float(np.clip(feats[11] / (feats[0] + 1e-6), 0.0, 1.0))

    # ── [44-47] Frame-flow coupling ──────────────────────────────────────
    feats[44] = float(np.clip(edge_mot / 5.0, 0.0, 1.0))
    feats[45] = float(np.clip(flat_mot / 5.0, 0.0, 1.0))
    feats[46] = (float(np.clip(mag[edge_mask].std() / 2.0, 0.0, 1.0))
                 if edge_mask.sum() > 10 else 0.0)
    feats[47] = float((mag[flat_mask] < 0.3).mean()) if flat_mask.sum() > 0 else 0.0

    return feats


# ---------------------------------------------------------------------------
#  Sequence: multiple consecutive pairs
# ---------------------------------------------------------------------------

def extract_sequence_features(
    frames: List,
    max_pairs: int = 16,
) -> np.ndarray:
    """
    Extract per-pair motion features from a list of video frames.

    Args:
        frames    : list of PIL Images or (H, W, 3) uint8 ndarrays, length ≥ 2
        max_pairs : cap on number of pairs processed (limits CPU time)

    Returns:
        (T, MOTION_FEATURE_DIM) float32  where T = min(len(frames)-1, max_pairs)
        Falls back to (1, MOTION_FEATURE_DIM) of zeros if fewer than 2 frames.
    """
    if len(frames) < 2:
        return np.zeros((1, MOTION_FEATURE_DIM), dtype=np.float32)

    n_pairs    = min(len(frames) - 1, max_pairs)
    all_feats  = []
    prev_mean  = None

    for i in range(n_pairs):
        f = extract_pair_features(frames[i], frames[i + 1], prev_mean)
        prev_mean = float(f[0])
        all_feats.append(f)

    return np.stack(all_feats, axis=0).astype(np.float32)   # (T, 24)


def extract_aggregated_features(
    frames: List,
    max_pairs: int = 16,
) -> np.ndarray:
    """
    Single-vector version: mean across all frame pairs → (MOTION_FEATURE_DIM,).
    Used when you don't need the full temporal sequence.
    """
    seq = extract_sequence_features(frames, max_pairs)
    return seq.mean(axis=0)


# ---------------------------------------------------------------------------
#  Convenience class (stateless wrapper — same API as other src/ analyzers)
# ---------------------------------------------------------------------------

class MotionAnalyzer:
    """
    Stateless wrapper around the module-level functions.

    Usage:
        analyzer = MotionAnalyzer()

        # Full sequence (for GRU branch):
        seq = analyzer.extract_sequence(frames)        # (T, 24) float32

        # Single aggregated vector (for MLP / legacy):
        vec = analyzer.extract_aggregated(frames)      # (24,) float32

        # Single pair:
        vec = analyzer.extract_pair(prev, curr)        # (24,) float32
    """

    FEATURE_DIM = MOTION_FEATURE_DIM

    def extract_pair(self, prev_frame, curr_frame,
                     prev_mean_mag: float = None) -> np.ndarray:
        return extract_pair_features(prev_frame, curr_frame, prev_mean_mag)

    def extract_sequence(self, frames: List,
                         max_pairs: int = 16) -> np.ndarray:
        return extract_sequence_features(frames, max_pairs)

    def extract_aggregated(self, frames: List,
                           max_pairs: int = 16) -> np.ndarray:
        return extract_aggregated_features(frames, max_pairs)


# ---------------------------------------------------------------------------
#  Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch

    print("MotionAnalyzer sanity check")
    print(f"  MOTION_FEATURE_DIM = {MOTION_FEATURE_DIM}")

    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 255, (_FLOW_H, _FLOW_W, 3), dtype=np.uint8)
              for _ in range(6)]

    analyzer = MotionAnalyzer()

    seq = analyzer.extract_sequence(frames)
    print(f"  extract_sequence  : {seq.shape}  dtype={seq.dtype}")
    assert seq.shape == (5, MOTION_FEATURE_DIM), f"Expected (5, 48), got {seq.shape}"

    agg = analyzer.extract_aggregated(frames)
    print(f"  extract_aggregated: {agg.shape}  dtype={agg.dtype}")
    assert agg.shape == (MOTION_FEATURE_DIM,)

    # Simulate torch tensor input
    t = torch.from_numpy(seq).unsqueeze(0)   # (1, T, 48)
    print(f"  Tensor shape for GRU: {t.shape}")

    print("  PASSED ✓")
