"""
prnu_features.py — Fast PRNU Feature Extractor for Neural Network Training

Converts an image into an 8-dimensional feature vector capturing PRNU
(Photo Response Non-Uniformity) forensic signals. Designed to be:
  - Fast: operates on 128×128 downsampled images (not full-res)
  - Batch-safe: stateless, pure numpy functions
  - Normalized: all features returned in [0, 1] for stable training

Feature vector layout (8 values):
  [0] noise_strength       — RMS of noise residual
  [1] noise_uniformity     — spatial consistency of noise (high = AI-like)
  [2] blockiness_score     — JPEG 8×8 block artifact strength
  [3] freq_energy_ratio    — high-freq energy ratio (AI = low HF)
  [4] noise_skewness_norm  — skewness of noise distribution (normalized)
  [5] noise_kurtosis_norm  — excess kurtosis of noise (normalized)
  [6] high_freq_ratio      — fraction of FFT energy in HF band
  [7] compression_quality  — estimated JPEG quality (0=none/unknown, 1=max)
"""

import io
import numpy as np
from PIL import Image
from scipy import fftpack
from skimage.restoration import denoise_wavelet, estimate_sigma

# Fast-path resolution — lower = faster, still captures noise characteristics
_FAST_SIZE = 128


def extract_prnu_features(image_input) -> np.ndarray:
    """
    Extract an 8-dimensional PRNU feature vector from an image.

    Args:
        image_input: One of:
            - bytes         — raw image file bytes (JPEG, PNG, etc.)
            - PIL.Image     — already-loaded PIL image
            - np.ndarray    — RGB float64 array in [0,1] of shape (H, W, 3)
                              OR uint8 array (H, W, 3)

    Returns:
        np.ndarray: Feature vector of shape (8,), all values in [0, 1].
                    On error returns zero vector.
    """
    try:
        img_array, compression_quality = _load_and_downsample(image_input)
        return _compute_features(img_array, compression_quality)
    except Exception as e:
        # Return neutral zero vector if anything fails — training will
        # still proceed, the model simply gets no PRNU signal for this image.
        print(f"[prnu_features] Warning: failed to extract features: {e}")
        return np.zeros(8, dtype=np.float32)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _load_and_downsample(image_input):
    """
    Load image from various input types and downsample to _FAST_SIZE × _FAST_SIZE.
    Returns (img_array_float64, compression_quality_or_None).
    """
    compression_quality = None

    if isinstance(image_input, bytes):
        compression_quality = _estimate_jpeg_quality_bytes(image_input)
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.float64:
            img = Image.fromarray(image_input.astype(np.uint8))
        else:
            # Already float64 in [0,1]
            arr = image_input
            if arr.shape[0] != _FAST_SIZE or arr.shape[1] != _FAST_SIZE:
                pil = Image.fromarray((arr * 255).astype(np.uint8))
                pil = pil.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR)
                arr = np.array(pil, dtype=np.float64) / 255.0
            return arr, compression_quality
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    img = img.resize((_FAST_SIZE, _FAST_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr, compression_quality


def _compute_features(img_array: np.ndarray, compression_quality) -> np.ndarray:
    """
    Given a float64 [0,1] array of shape (H,W,3), compute the 8 features.
    """
    # --- 1. Noise residual via wavelet denoising ---
    denoised = denoise_wavelet(
        img_array,
        method='BayesShrink',
        mode='soft',
        wavelet='db4',          # db4 is faster than db8 for the small image
        wavelet_levels=3,       # 3 levels is sufficient at 128×128
        channel_axis=-1,
        rescale_sigma=True,
    )
    noise = img_array - denoised  # shape (H, W, 3)

    # --- Feature 0: Noise strength (RMS), normalized ---
    noise_strength_raw = float(np.sqrt(np.mean(noise ** 2)))
    # Typical range for real images: 0.003–0.08; clip to [0, 0.15]
    noise_strength = float(np.clip(noise_strength_raw / 0.15, 0.0, 1.0))

    # --- Feature 1: Noise uniformity across 16×16 blocks ---
    noise_uniformity = _spatial_uniformity(noise, block_size=16)

    # --- Feature 2: Blockiness score ---
    blockiness_score = _compute_blockiness(img_array)

    # --- Feature 3: High-frequency energy ratio (per-channel average) ---
    hf_ratios = []
    skewnesses = []
    kurtoses = []
    hf_fracs = []
    for c in range(3):
        ch = noise[:, :, c]
        F = fftpack.fft2(ch)
        power = np.abs(F) ** 2

        total_energy = np.sum(power) + 1e-12
        # High-freq = outer 50% of frequencies
        h, w = power.shape
        hf_mask = _high_freq_mask(h, w, cutoff=0.5)
        hf_energy = np.sum(power[hf_mask])
        hf_ratios.append(hf_energy / total_energy)

        # Noise distribution shape
        flat = ch.ravel()
        std = np.std(flat) + 1e-10
        skewnesses.append(float(np.mean((flat - np.mean(flat)) ** 3) / std ** 3))
        kurtoses.append(float(np.mean((flat - np.mean(flat)) ** 4) / std ** 4 - 3.0))

        # Fraction of image pixels with |noise| > 0.02 (high-freq texture)
        hf_fracs.append(float(np.mean(np.abs(ch) > 0.02)))

    freq_energy_ratio = float(np.mean(hf_ratios))        # already in [0,1]

    # --- Feature 4: Noise skewness (normalize to [0,1]) ---
    raw_skew = float(np.mean(skewnesses))
    # Typical range [-3, 3] → clip and map to [0,1]
    noise_skewness_norm = float(np.clip((raw_skew + 3.0) / 6.0, 0.0, 1.0))

    # --- Feature 5: Excess kurtosis (normalize to [0,1]) ---
    raw_kurt = float(np.mean(kurtoses))
    # Typical range [-2, 10] → clip and map
    noise_kurtosis_norm = float(np.clip((raw_kurt + 2.0) / 12.0, 0.0, 1.0))

    # --- Feature 6: High-frequency texture fraction ---
    high_freq_ratio = float(np.clip(np.mean(hf_fracs), 0.0, 1.0))

    # --- Feature 7: Compression quality (normalized; 0 if unknown) ---
    if compression_quality is not None:
        compression_quality_norm = float(compression_quality / 100.0)
    else:
        compression_quality_norm = 0.0

    features = np.array([
        noise_strength,
        noise_uniformity,
        blockiness_score,
        freq_energy_ratio,
        noise_skewness_norm,
        noise_kurtosis_norm,
        high_freq_ratio,
        compression_quality_norm,
    ], dtype=np.float32)

    return features


def _spatial_uniformity(noise: np.ndarray, block_size: int = 16) -> float:
    """How uniform is the noise across the image? High = AI-like."""
    h, w = noise.shape[:2]
    local_variances = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = noise[i:i + block_size, j:j + block_size]
            local_variances.append(float(np.var(block)))
    if not local_variances:
        return 0.5
    var_array = np.array(local_variances)
    mean_var = float(np.mean(var_array))
    std_var = float(np.std(var_array))
    # Coefficient of variation inversion: high CV = low uniformity
    return float(1.0 / (1.0 + std_var / (mean_var + 1e-10)))


def _compute_blockiness(image_array: np.ndarray) -> float:
    """Detect 8×8 JPEG block boundary artifacts. Returns score in [0,1]."""
    gray = np.mean(image_array, axis=-1)
    h, w = gray.shape
    boundary_diffs = []
    interior_diffs = []
    for col in range(1, w):
        diff = float(np.mean(np.abs(gray[:, col] - gray[:, col - 1])))
        if col % 8 == 0:
            boundary_diffs.append(diff)
        else:
            interior_diffs.append(diff)
    for row in range(1, h):
        diff = float(np.mean(np.abs(gray[row, :] - gray[row - 1, :])))
        if row % 8 == 0:
            boundary_diffs.append(diff)
        else:
            interior_diffs.append(diff)
    if not boundary_diffs or not interior_diffs:
        return 0.0
    mean_boundary = float(np.mean(boundary_diffs))
    mean_interior = float(np.mean(interior_diffs))
    if mean_interior > 0:
        ratio = mean_boundary / mean_interior
        return float(max(0.0, min(1.0, (ratio - 1.0) * 5.0)))
    return 0.0


def _high_freq_mask(h: int, w: int, cutoff: float = 0.5) -> np.ndarray:
    """
    Returns a boolean mask of shape (h, w) selecting the high-frequency
    region of a centered FFT spectrum (outer `cutoff` fraction of frequencies).
    """
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((Y - cy) / (h / 2)) ** 2 + ((X - cx) / (w / 2)) ** 2)
    # FFT output is not centered; use fftshift logic:
    # For unshifted FFT, high-freq values are near the edges.
    # Simple implementation: treat the unshifted FFT as periodic and use
    # the distance from (0,0) after wrapping.
    Y2 = np.minimum(Y, h - Y)
    X2 = np.minimum(X, w - X)
    dist2 = np.sqrt((Y2 / (h / 2)) ** 2 + (X2 / (w / 2)) ** 2)
    return dist2 >= cutoff


def _estimate_jpeg_quality_bytes(image_data: bytes):
    """
    Estimate JPEG quality from raw bytes. Returns float 0-100 or None.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.format != 'JPEG':
            return None
        qtables = getattr(img, 'quantization', None)
        if qtables:
            luma_table = list(qtables[0].values()) if isinstance(qtables[0], dict) else list(qtables[0])
            avg_quant = float(np.mean(luma_table))
            quality = max(0.0, min(100.0, 100.0 - (avg_quant - 1.0) * 2.0))
            return quality
    except Exception:
        pass
    return None
