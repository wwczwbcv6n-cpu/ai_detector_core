"""
prnu_features.py — PRNU Feature Extractor v4

Feature vector layout — extract_prnu_features() — 8 values (fast mode):
  [0] noise_strength       — RMS of noise residual (normalised)
  [1] noise_uniformity     — spatial consistency (high = AI-like)
  [2] blockiness_score     — JPEG 8×8 block artefact strength
  [3] freq_energy_ratio    — high-freq energy ratio
  [4] noise_skewness_norm  — skewness of noise distribution
  [5] noise_kurtosis_norm  — excess kurtosis of noise
  [6] high_freq_ratio      — fraction of FFT energy in HF band
  [7] compression_quality  — estimated JPEG quality (0=none/unknown, 1=max)

Feature vector layout — extract_prnu_features_fullres() — 64 values (v4):
  ── Inherited from v3 (unchanged positions, backward-compatible) ──
  [0..7]   — reliability-weighted mean of per-tile 8-dim vectors
  [8..14]  — reliability-weighted std of per-tile 8-dim vectors (7 values)
  [15]     — inter-tile PRNU correlation (0=inconsistent=AI, 1=consistent=real)
  [16..19] — PRNU energy in 4 frequency bands (LF, mid, HF, VHF)
  [20..23] — cross-channel PRNU correlation (R-G, R-B, G-B, mean)
  [24..27] — recovery delta stats (mean_energy_delta, max_per_channel,
             spectral_shift, confidence) — zeros if recovery_net=None
  [28..30] — double-compression signature (Q1_est, sub_block_score, reenc_conf)
  [31]     — recovery net confidence score (0 if recovery_net=None)
  ── New in v4 ──
  [32..35] — extended frequency bands ×4 (EHF1/EHF2/EHF3/EHF4 > VHF)
             finer resolution at spatial frequencies where AI artifacts cluster
  [36..39] — per-channel noise RMS: R, G, B, Luminance
             AI generators often have imbalanced per-channel noise amplitude
  [40..43] — noise anisotropy: row_corr, col_corr, row_vs_col_ratio, stripe_score
             real CMOS sensors have row/column readout patterns absent in AI
  [44..47] — Bayer CFA residual: 2px_period, 4px_period, CFA_score, demosaic_artifact
             demosaicking leaves 2×2 periodic signatures invisible to AI generators
  [48..51] — phase spectrum coherence: low_freq, mid_freq, high_freq, mean
             real PRNU has coherent phase structure across channels; AI does not
  [52..55] — directional noise correlation: horizontal, vertical, diagonal, anti-diagonal
             real sensor noise is directionally structured by readout circuitry
  [56..59] — saturation features: clip_fraction, dynamic_range, tonal_entropy, snr_estimate
  [60..63] — multi-scale noise consistency: 64px↔128px, 128px↔256px, 64px↔256px, overall
             real cameras have consistent fingerprint at all scales; AI varies
"""

import io
import numpy as np
from PIL import Image
from scipy import fftpack
from scipy.ndimage import uniform_filter
try:
    from skimage.restoration import denoise_wavelet, estimate_sigma
except ImportError as e:
    print(f"[prnu_features] ERROR: {e}. Install: pip install scikit-image PyWavelets")
    def denoise_wavelet(*a, **k): raise e
    def estimate_sigma(*a, **k):  raise e

_FAST_SIZE = 256

PRNU_FAST_DIM    = 8
PRNU_FULLRES_DIM = 64    # 32 in v3; 64 in v4


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def extract_prnu_features(image_input) -> np.ndarray:
    """
    Fast 8-dim PRNU feature vector (downsampled to 128×128).
    Uses intensity-normalised noise residual.
    """
    try:
        img_array, compression_quality = _load_and_downsample(image_input)
        noise = _extract_noise_intensity_normalized(img_array)
        return _compute_features(img_array, noise, compression_quality)
    except Exception as e:
        print(f"[prnu_features] Warning: {e}")
        return np.zeros(PRNU_FAST_DIM, dtype=np.float32)


def extract_prnu_features_fullres(
    image_input,
    tile_size: int = 1024,
    recovery_net=None,
    device=None,
) -> np.ndarray:
    """
    Deep 64-dim PRNU feature extraction at full resolution (v4).

    Args:
        image_input  : bytes | PIL Image | np.ndarray (H,W,3) float64 [0,1]
        tile_size    : tile size for tiled processing (default 1024)
        recovery_net : optional PRNURecoveryNet to de-compress the image first
        device       : torch.device for recovery_net inference (default CPU)

    Returns:
        np.ndarray  shape (64,), values in [0, 1]
    """
    try:
        arr = _load_fullres(image_input)     # (H, W, 3) float64 [0,1]

        # Optional PRNU recovery (de-compression)
        recovered_arr = None
        if recovery_net is not None:
            try:
                from prnu_recovery import recover_prnu_signal
                import torch
                dev = device if device is not None else torch.device('cpu')
                with torch.no_grad():
                    recovered_arr = recover_prnu_signal(arr, recovery_net, dev)
                arr_for_prnu = recovered_arr
            except Exception as e:
                print(f"[prnu_features] Recovery net failed: {e} — using original image")
                arr_for_prnu = arr
        else:
            arr_for_prnu = arr

        h, w = arr_for_prnu.shape[:2]

        tile_features   = []   # list of 8-dim vectors
        tile_weights    = []   # reliability weight per tile
        tile_residuals  = []   # per-tile mean noise residual (for inter-tile corr)
        tile_noises_3ch = []   # per-tile 3-channel noise (for band energy / cross-corr)

        for row_start in range(0, h, tile_size):
            for col_start in range(0, w, tile_size):
                tile = arr_for_prnu[row_start:row_start + tile_size,
                                    col_start:col_start + tile_size]

                if tile.shape[0] < tile_size // 4 or tile.shape[1] < tile_size // 4:
                    continue

                weight = _compute_tile_reliability(tile)
                if weight < 1e-6:
                    continue

                pil       = Image.fromarray((tile * 255).astype(np.uint8))
                pil_small = pil.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR)
                small     = np.array(pil_small, dtype=np.float64) / 255.0

                noise = _extract_noise_intensity_normalized(small)

                tile_residuals.append(noise.mean(axis=-1).ravel())
                tile_noises_3ch.append(noise)

                feats = _compute_features(small, noise, compression_quality=None)
                tile_features.append(feats)
                tile_weights.append(weight)

        if not tile_features:
            # Fallback: single pass on full image (downsampled)
            small_pil = Image.fromarray(
                (arr_for_prnu * 255).astype(np.uint8)
            ).resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR)
            small = np.array(small_pil, dtype=np.float64) / 255.0
            noise = _extract_noise_intensity_normalized(small)
            feats = _compute_features(small, noise, compression_quality=None)
            out = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)
            out[:8]  = feats
            # std, inter_corr, and all extended features default to 0
            return out

        tile_matrix = np.stack(tile_features, axis=0)    # (N, 8)
        w_arr       = np.array(tile_weights, dtype=np.float64)
        w_arr       = w_arr / (w_arr.sum() + 1e-10)

        # Reliability-weighted mean
        tile_mean = (w_arr[:, None] * tile_matrix).sum(axis=0).astype(np.float32)

        # Reliability-weighted std
        diff     = tile_matrix - tile_mean[None, :]
        tile_std = np.sqrt((w_arr[:, None] * diff ** 2).sum(axis=0)).astype(np.float32)

        # Feature [15]: inter-tile PRNU correlation
        inter_corr = _compute_inter_tile_correlation(tile_residuals)

        # Aggregate noise for band energy and cross-channel correlation
        agg_noise = np.zeros((_FAST_SIZE, _FAST_SIZE, 3), dtype=np.float64)
        for i, noise in enumerate(tile_noises_3ch):
            agg_noise += noise * w_arr[i]

        # Features [16..19]: frequency band energy (LF, mid, HF, VHF)
        freq_bands = _compute_frequency_band_energy(agg_noise)

        # Features [20..23]: cross-channel PRNU correlation (R-G, R-B, G-B, mean)
        cross_chan = _compute_cross_channel_correlation(agg_noise)

        # Features [24..27]: recovery delta stats
        if recovered_arr is not None:
            recovery_delta = _compute_recovery_delta_stats(arr, recovered_arr)
            recovery_conf  = _compute_recovery_confidence(arr, recovered_arr)
        else:
            recovery_delta = np.zeros(4, dtype=np.float32)
            recovery_conf  = 0.0

        # Features [28..30]: double-compression signature
        double_comp = _compute_double_compression_signature(image_input, arr)

        # ── Assemble 64-dim vector ──────────────────────────────────────────
        out = np.empty(PRNU_FULLRES_DIM, dtype=np.float32)
        out[:8]    = tile_mean
        out[8:15]  = tile_std[:7]
        out[15]    = np.float32(inter_corr)
        out[16:20] = freq_bands
        out[20:24] = cross_chan
        out[24:28] = recovery_delta
        out[28:31] = double_comp
        out[31]    = np.float32(recovery_conf)

        # ── v4 extended features [32..63] ───────────────────────────────────
        out[32:36] = _compute_extended_frequency_bands(agg_noise)
        out[36:40] = _compute_per_channel_noise_rms(agg_noise)
        out[40:44] = _compute_noise_anisotropy(agg_noise)
        out[44:48] = _compute_bayer_cfa_residual(arr_for_prnu)
        out[48:52] = _compute_phase_coherence(agg_noise)
        out[52:56] = _compute_directional_correlation(agg_noise)
        out[56:60] = _compute_saturation_features(arr_for_prnu)
        out[60:64] = _compute_multiscale_consistency(arr_for_prnu)

        return out

    except Exception as e:
        print(f"[prnu_features] Warning: extract_prnu_features_fullres failed: {e}")
        return np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)


def extract_prnu_map(image_input, output_size: int = 64) -> np.ndarray:
    """
    Extract spatial PRNU noise map at fixed output resolution.

    Uses existing _extract_noise_intensity_normalized() on a 128×128 downsample,
    then resizes to output_size × output_size.

    Args:
        image_input : bytes | PIL Image | np.ndarray (H,W,3) float64 [0,1]
        output_size : spatial resolution of the returned map (default 64)

    Returns:
        np.ndarray  shape (output_size, output_size, 3) float32 in [-1, 1],
                    zero-centred noise map
    """
    try:
        arr   = _load_fullres(image_input)                         # (H,W,3) float64
        small = Image.fromarray((arr * 255).astype(np.uint8))
        small = np.array(
            small.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR), dtype=np.float64
        ) / 255.0
        noise = _extract_noise_intensity_normalized(small)   # (_FAST_SIZE, _FAST_SIZE, 3)

        # Resize from 128×128 to output_size×output_size
        pil   = Image.fromarray(
            np.clip((noise + 0.5) * 127.5, 0, 255).astype(np.uint8)
        )
        pil   = pil.resize((output_size, output_size), Image.BILINEAR)
        result = np.array(pil, dtype=np.float32) / 127.5 - 1.0    # [-1,1]
        return result                                              # (output_size, output_size, 3)
    except Exception as e:
        print(f"[prnu_features] Warning: extract_prnu_map failed: {e}")
        return np.zeros((output_size, output_size, 3), dtype=np.float32)


def extract_prnu_patch_map(
    image_input,
    tile_size: int = 256,
) -> np.ndarray:
    """
    Patch-level PRNU feature map for spatial forgery localisation.

    Returns:
        np.ndarray: shape (N_tiles, 8) — one 8-dim feature vector per tile.
    """
    try:
        arr  = _load_fullres(image_input)
        h, w = arr.shape[:2]

        patch_features = []

        for row_start in range(0, h, tile_size):
            for col_start in range(0, w, tile_size):
                tile = arr[row_start:row_start + tile_size,
                           col_start:col_start + tile_size]
                if tile.shape[0] < tile_size // 4 or tile.shape[1] < tile_size // 4:
                    continue

                pil   = Image.fromarray((tile * 255).astype(np.uint8))
                small = np.array(
                    pil.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR),
                    dtype=np.float64
                ) / 255.0

                noise = _extract_noise_intensity_normalized(small)
                feats = _compute_features(small, noise, compression_quality=None)
                patch_features.append(feats)

        if not patch_features:
            return np.zeros((1, PRNU_FAST_DIM), dtype=np.float32)

        return np.stack(patch_features, axis=0)

    except Exception as e:
        print(f"[prnu_features] Warning: extract_prnu_patch_map failed: {e}")
        return np.zeros((1, PRNU_FAST_DIM), dtype=np.float32)


# ---------------------------------------------------------------------------
#  Extended feature helpers (v3)
# ---------------------------------------------------------------------------

def _compute_frequency_band_energy(noise: np.ndarray) -> np.ndarray:
    """
    PRNU energy in 4 frequency bands: LF [0,0.1), mid [0.1,0.3), HF [0.3,0.6),
    VHF [0.6,∞).  JPEG compression destroys HF first; AI images are typically
    HF-deficient.

    Args:
        noise : (H, W, 3) float64 noise residual

    Returns:
        np.ndarray  shape (4,) in [0, 1] — fraction of total energy per band
    """
    gray  = noise.mean(axis=-1)            # (H, W)
    F     = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    Y = np.arange(h) - cy
    X = np.arange(w) - cx
    FX, FY = np.meshgrid(X, Y)
    dist = np.sqrt((FY / (cy + 1e-8)) ** 2 + (FX / (cx + 1e-8)) ** 2)

    total = float(np.sum(power)) + 1e-12
    cuts  = [0.0, 0.1, 0.3, 0.6, 1e9]
    energies = []
    for i in range(4):
        mask = (dist >= cuts[i]) & (dist < cuts[i + 1])
        energies.append(float(np.sum(power[mask])) / total)

    return np.array(energies, dtype=np.float32)


def _compute_cross_channel_correlation(noise: np.ndarray) -> np.ndarray:
    """
    Pairwise cross-channel PRNU correlation (R-G, R-B, G-B, mean).

    Real camera sensors have correlated PRNU across RGB channels (shared
    optics + pixel neighbourhood); AI generators produce independent noise.

    Returns:
        np.ndarray  shape (4,) in [0, 1]   (mapped from [-1,1] → [0,1])
    """
    ch_r = noise[:, :, 0].ravel()
    ch_g = noise[:, :, 1].ravel()
    ch_b = noise[:, :, 2].ravel()

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        sa, sb = float(np.std(a)), float(np.std(b))
        if sa < 1e-8 or sb < 1e-8:
            return 0.5
        c = float(np.corrcoef(a, b)[0, 1])
        return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(c) else 0.5

    rg   = safe_corr(ch_r, ch_g)
    rb   = safe_corr(ch_r, ch_b)
    gb   = safe_corr(ch_g, ch_b)
    mean = (rg + rb + gb) / 3.0
    return np.array([rg, rb, gb, mean], dtype=np.float32)


def _compute_recovery_delta_stats(
    original_arr: np.ndarray,
    recovered_arr: np.ndarray,
) -> np.ndarray:
    """
    Measure how much the PRNU signal changed after recovery (de-compression).

    Large delta → the image was heavily compressed and the recovery net did
    significant work.  Features: (mean_energy_delta, max_per_channel,
    spectral_shift, confidence).

    Returns:
        np.ndarray  shape (4,) in [0, 1]
    """
    try:
        # Downsample both to _FAST_SIZE for speed
        def _small(a):
            pil = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
            return np.array(
                pil.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR), dtype=np.float64
            ) / 255.0

        orig_s = _small(original_arr)
        rec_s  = _small(recovered_arr)

        orig_noise = _extract_noise_intensity_normalized(orig_s)
        rec_noise  = _extract_noise_intensity_normalized(rec_s)

        # Per-channel RMS energy
        orig_e = np.array([float(np.sqrt(np.mean(orig_noise[:, :, c] ** 2))) for c in range(3)])
        rec_e  = np.array([float(np.sqrt(np.mean(rec_noise[:, :,  c] ** 2))) for c in range(3)])
        delta  = np.abs(rec_e - orig_e)

        mean_delta = float(np.clip(delta.mean() / 0.1, 0.0, 1.0))
        max_delta  = float(np.clip(delta.max()  / 0.1, 0.0, 1.0))

        # Spectral shift: peak-frequency displacement after recovery
        def peak_freq(noise_arr):
            fft_m = np.abs(np.fft.fft2(noise_arr.mean(axis=-1)))
            fft_m[0, 0] = 0.0
            idx = np.unravel_index(np.argmax(fft_m), fft_m.shape)
            return np.array(idx, dtype=float) / np.array(fft_m.shape, dtype=float)

        p0 = peak_freq(orig_noise)
        p1 = peak_freq(rec_noise)
        spectral_shift = float(np.clip(np.linalg.norm(p1 - p0) * 4.0, 0.0, 1.0))

        confidence = float(np.clip((mean_delta + max_delta) / 2.0, 0.0, 1.0))

        return np.array([mean_delta, max_delta, spectral_shift, confidence],
                        dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


def _compute_recovery_confidence(
    original_arr: np.ndarray,
    recovered_arr: np.ndarray,
) -> float:
    """
    Recovery confidence score: mean absolute difference between original and
    recovered arrays, scaled so 0.05 difference → confidence ≈ 1.0.
    """
    try:
        diff = np.abs(recovered_arr.astype(np.float64) - original_arr.astype(np.float64))
        return float(np.clip(diff.mean() * 20.0, 0.0, 1.0))
    except Exception:
        return 0.0


def _compute_double_compression_signature(image_input, arr: np.ndarray) -> np.ndarray:
    """
    Estimate double-JPEG-compression signature.

    Returns (Q1_est, sub_block_score, reenc_confidence) — each in [0, 1].

    Q1_est            : first JPEG quality estimated from quantization tables
                        (0 if unavailable)
    sub_block_score   : blockiness at 4 px period (signs of a second JPEG)
    reenc_confidence  : overall confidence that re-encoding occurred
    """
    try:
        # Q1 from JPEG quantization tables (needs raw bytes)
        q1_est = 0.0
        if isinstance(image_input, bytes):
            q = _estimate_jpeg_quality_bytes(image_input)
            if q is not None:
                q1_est = float(np.clip(q / 100.0, 0.0, 1.0))

        # Blockiness at multiple periods using the loaded array
        gray = arr.mean(axis=-1)
        b8   = _blockiness_at_period(gray, 8)
        b4   = _blockiness_at_period(gray, 4)

        # Sub-block artefacts relative to main-block artefacts
        sub_block_score = float(np.clip(b4 / (b8 + 1e-6), 0.0, 1.0))

        # Re-encoding confidence: both 8 px and 4 px blockiness present
        reenc_confidence = float(np.clip((b8 + b4) / 2.0, 0.0, 1.0))

        return np.array([q1_est, sub_block_score, reenc_confidence], dtype=np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _blockiness_at_period(gray: np.ndarray, period: int) -> float:
    """Measure JPEG-like blockiness at a given period in pixels."""
    lim = min(gray.shape[1], 128)
    boundary_diffs, interior_diffs = [], []
    for col in range(1, lim):
        diff = float(np.mean(np.abs(gray[:lim, col] - gray[:lim, col - 1])))
        (boundary_diffs if col % period == 0 else interior_diffs).append(diff)
    if not boundary_diffs or not interior_diffs:
        return 0.0
    ratio = float(np.mean(boundary_diffs)) / (float(np.mean(interior_diffs)) + 1e-10)
    return float(np.clip((ratio - 1.0) * 5.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
#  Extended feature helpers (v4)
# ---------------------------------------------------------------------------

def _compute_extended_frequency_bands(noise: np.ndarray) -> np.ndarray:
    """
    PRNU energy in 4 high-frequency bands above VHF: [0.6-0.7), [0.7-0.8),
    [0.8-0.9), [0.9, ∞).  AI artifact signatures cluster in these bands.
    Returns (4,) in [0, 1].
    """
    gray  = noise.mean(axis=-1)
    F     = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F) ** 2
    h, w  = gray.shape
    cy, cx = h // 2, w // 2
    Y  = np.arange(h) - cy
    X  = np.arange(w) - cx
    FX, FY = np.meshgrid(X, Y)
    dist = np.sqrt((FY / (cy + 1e-8)) ** 2 + (FX / (cx + 1e-8)) ** 2)
    total  = float(power.sum()) + 1e-12
    cuts   = [0.6, 0.7, 0.8, 0.9, 1e9]
    return np.array([
        float(power[(dist >= cuts[i]) & (dist < cuts[i + 1])].sum()) / total
        for i in range(4)
    ], dtype=np.float32)


def _compute_per_channel_noise_rms(noise: np.ndarray) -> np.ndarray:
    """
    Per-channel noise RMS: R, G, B, Luminance.
    AI generators often have imbalanced per-channel noise amplitude.
    Returns (4,) in [0, 1].
    """
    rms = [float(np.sqrt(np.mean(noise[:, :, c] ** 2))) for c in range(3)]
    rms.append(float(np.sqrt(np.mean(noise.mean(axis=-1) ** 2))))
    return np.clip(np.array(rms, dtype=np.float32) / 0.1, 0.0, 1.0)


def _compute_noise_anisotropy(noise: np.ndarray) -> np.ndarray:
    """
    Noise anisotropy from row/column autocorrelation.
    Real CMOS sensors produce row/column readout patterns absent in AI images.
    Returns (4,) [row_corr, col_corr, row_vs_col_ratio, stripe_score] in [0, 1].
    """
    gray = noise.mean(axis=-1)
    std2 = float(np.std(gray)) ** 2 + 1e-10

    row_ac  = float(np.mean(gray[:, 1:] * gray[:, :-1])) / std2
    row_corr = float(np.clip((row_ac + 1.0) / 2.0, 0.0, 1.0))

    col_ac  = float(np.mean(gray[1:, :] * gray[:-1, :])) / std2
    col_corr = float(np.clip((col_ac + 1.0) / 2.0, 0.0, 1.0))

    row_vs_col = float(abs(row_corr - col_corr) / (row_corr + col_corr + 1e-6))

    row_means = gray.mean(axis=1)
    fft_rows  = np.abs(np.fft.fft(row_means))
    fft_rows[0] = 0.0
    stripe_score = (
        float(np.clip(fft_rows.max() / (fft_rows.mean() + 1e-8) / 10.0, 0.0, 1.0))
        if fft_rows.sum() > 1e-10 else 0.0
    )
    return np.array([row_corr, col_corr, row_vs_col, stripe_score], dtype=np.float32)


def _compute_bayer_cfa_residual(arr: np.ndarray) -> np.ndarray:
    """
    Bayer CFA and demosaicking artifact detection.
    Real cameras: 2×2 Bayer mosaic leaves 2px/4px periodic artifacts.
    Returns (4,) [period_2, period_4, cfa_score, demosaic] in [0, 1].
    """
    try:
        pil   = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        small = np.array(
            pil.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR), dtype=np.float64
        ) / 255.0
        H, W  = _FAST_SIZE, _FAST_SIZE
        gray  = small.mean(axis=-1)
        F_s   = np.fft.fftshift(np.fft.fft2(gray))
        power = np.abs(F_s) ** 2
        cy, cx = H // 2, W // 2
        total  = float(power.sum()) + 1e-12

        half_r = H // 4
        qtr_r  = H // 8
        rows   = np.arange(H)[:, None]
        cols   = np.arange(W)[None, :]
        p2_mask = (np.abs(rows - cy) == half_r) | (np.abs(cols - cx) == half_r)
        p4_mask = (np.abs(rows - cy) == qtr_r)  | (np.abs(cols - cx) == qtr_r)
        period_2 = float(np.clip(power[p2_mask].sum() / total * 10.0, 0.0, 1.0))
        period_4 = float(np.clip(power[p4_mask].sum() / total * 10.0, 0.0, 1.0))
        cfa_score = (period_2 + period_4) / 2.0

        # Cross-channel phase coherence at Bayer frequency
        phases = [float(np.angle(np.fft.fft2(small[:, :, c])[half_r, half_r]))
                  for c in range(3)]
        demosaic = float(np.clip(1.0 - float(np.std(phases)) / np.pi, 0.0, 1.0))

        return np.array([period_2, period_4, cfa_score, demosaic], dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


def _compute_phase_coherence(noise: np.ndarray) -> np.ndarray:
    """
    Phase spectrum coherence across RGB channels of the noise residual.
    Real PRNU has coherent phase (shared optics); AI noise is channel-independent.
    Returns (4,) [low_freq, mid_freq, high_freq, mean] in [0, 1].
    """
    try:
        H, W = noise.shape[:2]
        phases = [np.angle(np.fft.fftshift(np.fft.fft2(noise[:, :, c])))
                  for c in range(3)]
        cy, cx = H // 2, W // 2
        Y  = np.arange(H) - cy
        X  = np.arange(W) - cx
        FX, FY = np.meshgrid(X, Y)
        dist = np.sqrt((FY / (cy + 1e-8)) ** 2 + (FX / (cx + 1e-8)) ** 2)

        def _band_coh(mask):
            if mask.sum() == 0:
                return 0.5
            p = [ph[mask] for ph in phases]
            mean_cos = float(np.mean([
                np.mean(np.cos(p[i] - p[j]))
                for i, j in [(0, 1), (0, 2), (1, 2)]
            ]))
            return float(np.clip((mean_cos + 1.0) / 2.0, 0.0, 1.0))

        lf = _band_coh(dist < 0.3)
        mf = _band_coh((dist >= 0.3) & (dist < 0.6))
        hf = _band_coh(dist >= 0.6)
        return np.array([lf, mf, hf, (lf + mf + hf) / 3.0], dtype=np.float32)
    except Exception:
        return np.full(4, 0.5, dtype=np.float32)


def _compute_directional_correlation(noise: np.ndarray) -> np.ndarray:
    """
    Directional lag-1 autocorrelation: horizontal, vertical, diagonal, anti-diagonal.
    Real sensor readout noise has directional structure; AI noise is isotropic.
    Returns (4,) in [0, 1] (mapped from [-1, 1]).
    """
    gray = noise.mean(axis=-1)
    std2 = float(np.std(gray)) ** 2 + 1e-10

    def _lag1(a, b):
        c = float(np.mean(a * b)) / std2
        return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0))

    return np.array([
        _lag1(gray[:, 1:],   gray[:, :-1]),    # horizontal
        _lag1(gray[1:, :],   gray[:-1, :]),    # vertical
        _lag1(gray[1:, 1:],  gray[:-1, :-1]),  # diagonal
        _lag1(gray[1:, :-1], gray[:-1, 1:]),   # anti-diagonal
    ], dtype=np.float32)


def _compute_saturation_features(arr: np.ndarray) -> np.ndarray:
    """
    Image saturation and tonal statistics.
    clip_fraction: fraction of saturated pixels.
    dynamic_range: std of luminance (low = AI flat regions).
    tonal_entropy: Shannon entropy of 64-bin histogram.
    snr_estimate : local mean / local noise std ratio.
    Returns (4,) in [0, 1].
    """
    try:
        pil  = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        s    = np.array(
            pil.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR), dtype=np.float64
        ) / 255.0
        gray = s.mean(axis=-1)

        clip_frac  = float(np.mean((gray < 0.02) | (gray > 0.98)))
        dyn_range  = float(np.clip(np.std(gray) * 4.0, 0.0, 1.0))
        hist, _    = np.histogram(gray.ravel(), bins=64, range=(0.0, 1.0))
        hist       = hist.astype(np.float64) + 1e-8
        hist      /= hist.sum()
        tonal_ent  = float(np.clip(-np.sum(hist * np.log2(hist)) / 6.0, 0.0, 1.0))

        block = 8
        H, W  = gray.shape
        local_vars = [
            float(np.var(gray[i:i + block, j:j + block]))
            for i in range(0, H - block + 1, block)
            for j in range(0, W - block + 1, block)
        ]
        noise_rms = float(np.sqrt(np.median(local_vars))) if local_vars else 0.01
        snr       = float(np.clip(gray.mean() / (noise_rms + 0.01) / 10.0, 0.0, 1.0))

        return np.array([clip_frac, dyn_range, tonal_ent, snr], dtype=np.float32)
    except Exception:
        return np.array([0.0, 0.5, 0.5, 0.5], dtype=np.float32)


def _compute_multiscale_consistency(arr: np.ndarray) -> np.ndarray:
    """
    Multi-scale noise fingerprint consistency.
    Real cameras have a stable PRNU fingerprint at all scales.
    AI generators produce scale-inconsistent noise patterns.
    Uses fast box-blur high-pass filter (no wavelet — fast).
    Returns (4,) [64↔128, 128↔256, 64↔256, mean] in [0, 1].
    """
    try:
        def _hpf(arr_in, size):
            pil   = Image.fromarray((np.clip(arr_in, 0, 1) * 255).astype(np.uint8))
            small = np.array(
                pil.resize((size, size), Image.BILINEAR), dtype=np.float64
            ) / 255.0
            gray  = small.mean(axis=-1)
            return (gray - uniform_filter(gray, size=5)).ravel()

        n64  = _hpf(arr, 64)
        n128 = _hpf(arr, 128)
        n256 = _hpf(arr, 256)

        L = 64 * 64
        n128_r = np.interp(np.linspace(0, len(n128) - 1, L),
                           np.arange(len(n128)), n128)
        n256_r = np.interp(np.linspace(0, len(n256) - 1, L),
                           np.arange(len(n256)), n256)

        def _corr(a, b):
            sa, sb = float(np.std(a)), float(np.std(b))
            if sa < 1e-10 or sb < 1e-10:
                return 0.5
            c = float(np.corrcoef(a, b)[0, 1])
            return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(c) else 0.5

        c64_128  = _corr(n64,   n128_r)
        c128_256 = _corr(n128_r, n256_r)
        c64_256  = _corr(n64,   n256_r)
        return np.array([c64_128, c128_256, c64_256,
                         (c64_128 + c128_256 + c64_256) / 3.0], dtype=np.float32)
    except Exception:
        return np.full(4, 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
#  Internal helpers (unchanged from v2)
# ---------------------------------------------------------------------------

def _extract_noise_intensity_normalized(img_array: np.ndarray) -> np.ndarray:
    """
    Wavelet BayesShrink denoising → intensity-normalised noise residual.
    Normalisation: W ← (I - F(I)) / (I_gray + 0.01)
    Then zero-mean per row and column to remove stripe artefacts.
    """
    try:
        denoised = denoise_wavelet(
            img_array,
            method='BayesShrink',
            mode='soft',
            wavelet='db4',
            wavelet_levels=3,
            channel_axis=-1,
            rescale_sigma=True,
        )
        noise = img_array - denoised
    except Exception:
        return np.zeros_like(img_array, dtype=np.float32)

    if not np.isfinite(noise).all():
        return np.zeros_like(img_array, dtype=np.float32)

    i_gray = img_array.mean(axis=-1, keepdims=True)
    noise  = noise / (i_gray + 0.01)
    noise -= noise.mean(axis=0, keepdims=True)
    noise -= noise.mean(axis=1, keepdims=True)
    return noise.astype(np.float32)


def _compute_tile_reliability(tile: np.ndarray) -> float:
    sat_mask  = (tile > 0.95) | (tile < 0.05)
    sat_frac  = float(sat_mask.mean())
    non_sat   = max(0.0, 1.0 - sat_frac)
    local_var = float(np.var(tile))
    var_weight = float(np.clip(np.log1p(local_var * 1000) / 3.0, 0.0, 1.0))
    return non_sat * var_weight


def _compute_inter_tile_correlation(tile_residuals: list) -> float:
    if len(tile_residuals) < 2:
        return 0.5
    idx = np.linspace(0, len(tile_residuals) - 1,
                      min(10, len(tile_residuals)), dtype=int)
    residuals = [tile_residuals[i] for i in idx]
    corrs = []
    for i in range(len(residuals)):
        for j in range(i + 1, len(residuals)):
            a, b = residuals[i], residuals[j]
            sa, sb = np.std(a), np.std(b)
            if sa > 1e-8 and sb > 1e-8:
                c = float(np.corrcoef(a, b)[0, 1])
                if np.isfinite(c):
                    corrs.append(c)
    if not corrs:
        return 0.5
    return float(np.clip((float(np.mean(corrs)) + 1.0) / 2.0, 0.0, 1.0))


def _compute_features(
    img_array: np.ndarray,
    noise: np.ndarray,
    compression_quality,
) -> np.ndarray:
    """Compute 8-dim feature vector from a (H,W,3) float64 image + noise residual."""
    noise_strength_raw = float(np.sqrt(np.mean(noise ** 2)))
    noise_strength     = float(np.clip(noise_strength_raw / 0.15, 0.0, 1.0))
    noise_uniformity   = _spatial_uniformity(noise, block_size=16)
    blockiness_score   = _compute_blockiness(img_array)

    hf_ratios, skewnesses, kurtoses, hf_fracs = [], [], [], []
    for c in range(3):
        ch    = noise[:, :, c]
        F     = fftpack.fft2(ch)
        power = np.abs(F) ** 2

        total_energy = float(np.sum(power)) + 1e-12
        hf_mask      = _high_freq_mask(*power.shape, cutoff=0.5)
        hf_ratios.append(float(np.sum(power[hf_mask])) / total_energy)

        flat = ch.ravel()
        std  = float(np.std(flat)) + 1e-10
        mean = float(np.mean(flat))
        skew = float(np.mean((flat - mean) ** 3) / std ** 3)
        kurt = float(np.mean((flat - mean) ** 4) / std ** 4 - 3.0)
        if not np.isfinite(skew): skew = 0.0
        if not np.isfinite(kurt): kurt = 0.0
        skewnesses.append(skew)
        kurtoses.append(kurt)
        hf_fracs.append(float(np.mean(np.abs(ch) > 0.02)))

    freq_energy_ratio   = float(np.mean(hf_ratios))   if hf_ratios   else 0.5
    noise_skewness_norm = float(np.clip((np.mean(skewnesses) + 3.0) / 6.0,  0.0, 1.0)) if skewnesses else 0.5
    noise_kurtosis_norm = float(np.clip((np.mean(kurtoses)  + 2.0) / 12.0, 0.0, 1.0)) if kurtoses   else 0.5
    high_freq_ratio     = float(np.clip(np.mean(hf_fracs), 0.0, 1.0))                  if hf_fracs   else 0.5

    if compression_quality is not None:
        compression_quality_norm = float(np.clip(compression_quality / 100.0, 0.0, 1.0))
    else:
        compression_quality_norm = 0.0

    return np.array([
        noise_strength, noise_uniformity, blockiness_score,
        freq_energy_ratio, noise_skewness_norm, noise_kurtosis_norm,
        high_freq_ratio, compression_quality_norm,
    ], dtype=np.float32)


def _spatial_uniformity(noise: np.ndarray, block_size: int = 16) -> float:
    h, w = noise.shape[:2]
    local_variances = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            local_variances.append(float(np.var(noise[i:i + block_size, j:j + block_size])))
    if not local_variances:
        return 0.5
    var_array = np.array(local_variances)
    mean_var  = float(np.mean(var_array))
    std_var   = float(np.std(var_array))
    return float(1.0 / (1.0 + std_var / (mean_var + 1e-10)))


def _compute_blockiness(image_array: np.ndarray) -> float:
    gray = np.mean(image_array, axis=-1)
    h, w = gray.shape
    boundary_diffs, interior_diffs = [], []
    for col in range(1, w):
        diff = float(np.mean(np.abs(gray[:, col] - gray[:, col - 1])))
        (boundary_diffs if col % 8 == 0 else interior_diffs).append(diff)
    for row in range(1, h):
        diff = float(np.mean(np.abs(gray[row, :] - gray[row - 1, :])))
        (boundary_diffs if row % 8 == 0 else interior_diffs).append(diff)
    if not boundary_diffs or not interior_diffs:
        return 0.0
    ratio = float(np.mean(boundary_diffs)) / (float(np.mean(interior_diffs)) + 1e-10)
    return float(max(0.0, min(1.0, (ratio - 1.0) * 5.0)))


def _high_freq_mask(h: int, w: int, cutoff: float = 0.5) -> np.ndarray:
    Y  = np.arange(h)
    X  = np.arange(w)
    Y2 = np.minimum(Y, h - Y)
    X2 = np.minimum(X, w - X)
    FX, FY = np.meshgrid(X2, Y2)
    dist2  = np.sqrt((FY / (h / 2)) ** 2 + (FX / (w / 2)) ** 2)
    return dist2 >= cutoff


def _load_fullres(image_input) -> np.ndarray:
    """Load image at full native resolution as float64 [0,1] (H,W,3)."""
    if isinstance(image_input, bytes):
        return np.array(Image.open(io.BytesIO(image_input)).convert("RGB"),
                        dtype=np.float64) / 255.0
    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.float64) / 255.0
    if isinstance(image_input, np.ndarray):
        if image_input.dtype == np.float64:
            return image_input
        return image_input.astype(np.float64) / 255.0
    raise TypeError(f"Unsupported image_input type: {type(image_input)}")


def _load_and_downsample(image_input):
    """Load and downsample to _FAST_SIZE. Returns (array_float64, quality_or_None)."""
    quality = None
    if isinstance(image_input, bytes):
        quality = _estimate_jpeg_quality_bytes(image_input)
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype == np.float64:
            arr = image_input
            if arr.shape[0] != _FAST_SIZE or arr.shape[1] != _FAST_SIZE:
                pil = Image.fromarray((arr * 255).astype(np.uint8))
                arr = np.array(pil.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR),
                               dtype=np.float64) / 255.0
            return arr, quality
        img = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise TypeError(f"Unsupported type: {type(image_input)}")
    img = img.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float64) / 255.0, quality


def _estimate_jpeg_quality_bytes(image_data: bytes):
    """Estimate JPEG quality from raw bytes. Returns float 0-100 or None."""
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.format != 'JPEG':
            return None
        qtables = getattr(img, 'quantization', None)
        if qtables:
            luma = list(qtables[0].values()) if isinstance(qtables[0], dict) else list(qtables[0])
            avg_quant = float(np.mean(luma))
            return float(max(0.0, min(100.0, 100.0 - (avg_quant - 1.0) * 2.0)))
    except Exception:
        pass
    return None
