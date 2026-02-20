"""
PRNU (Photo Response Non-Uniformity) Noise Recovery Module

This module implements PRNU-based image forensics for AI image detection.
PRNU is a unique sensor noise fingerprint that exists in real camera images
but is absent or inconsistent in AI-generated images.

Key capabilities:
  - Extract noise residuals from images using wavelet-based denoising
  - Estimate camera PRNU fingerprints from multiple images
  - Recover degraded PRNU from compressed/transferred images via Wiener filtering
  - Detect AI-generated images by analyzing PRNU characteristics

References:
  - Lukas, Fridrich, Goljan (2006): "Digital Camera Identification from Sensor Pattern Noise"
  - Chen et al. (2008): "Determining Image Origin and Integrity Using Sensor Noise"
"""

import io
import numpy as np
from PIL import Image
from scipy import fftpack
from skimage.restoration import denoise_wavelet, estimate_sigma


# ---------------------------------------------------------------------------
#  Core PRNU functions
# ---------------------------------------------------------------------------

def extract_noise_residual(image_array):
    """
    Extract noise residual from an image by subtracting its denoised version.

    The noise residual W = I - F(I) where I is the image and F is a denoising
    filter. We use wavelet-based BayesShrink soft thresholding with the
    Daubechies-8 wavelet at 4 decomposition levels.

    Args:
        image_array (np.ndarray): Input image as float64 array in [0, 1], shape (H, W, 3).

    Returns:
        np.ndarray: Noise residual of same shape as input.
    """
    denoised = denoise_wavelet(
        image_array,
        method='BayesShrink',
        mode='soft',
        wavelet='db8',
        wavelet_levels=4,
        channel_axis=-1,
        rescale_sigma=True,
    )
    noise = image_array - denoised
    return noise


def estimate_prnu_pattern(noise_residuals):
    """
    Estimate the camera PRNU fingerprint from a set of noise residuals using
    Maximum Likelihood Estimation (MLE).

    K_hat = (sum Wi * Ii) / (sum Ii^2)

    For a simplified single-image scenario, the noise residual itself is
    used as the PRNU estimate after zero-mean normalisation.

    Args:
        noise_residuals (list[np.ndarray]): List of noise residuals, each (H, W, 3).

    Returns:
        np.ndarray: Estimated PRNU pattern (H, W, 3).
    """
    if not noise_residuals:
        raise ValueError("At least one noise residual is required.")

    # Stack and average (simple MLE when all images have similar intensity)
    stacked = np.stack(noise_residuals, axis=0)
    prnu = np.mean(stacked, axis=0)

    # Zero-mean normalisation per channel
    for c in range(prnu.shape[-1]):
        channel = prnu[:, :, c]
        prnu[:, :, c] = channel - np.mean(channel)

    return prnu


def recover_prnu_from_compressed(noise_residual, quality_estimate=None):
    """
    Recover / enhance PRNU signal that has been degraded by JPEG compression
    or social-media re-encoding using frequency-domain Wiener filtering.

    Compression introduces quantisation noise that masks the PRNU pattern.
    This function estimates the noise power spectral density (PSD) from the
    high-frequency content and applies a Wiener filter to suppress it while
    preserving the underlying PRNU signal.

    Args:
        noise_residual (np.ndarray): Degraded noise residual (H, W, 3).
        quality_estimate (float | None): Estimated JPEG quality factor 0-100.
            If None, it is inferred from the noise level.  Lower quality →
            stronger Wiener regularisation.

    Returns:
        np.ndarray: Enhanced noise residual with PRNU signal recovered.
    """
    recovered = np.zeros_like(noise_residual, dtype=np.float64)

    for c in range(noise_residual.shape[-1]):
        channel = noise_residual[:, :, c].astype(np.float64)

        # Estimate noise level for this channel
        sigma = estimate_sigma(channel)

        # Determine Wiener regularisation strength
        if quality_estimate is not None:
            # Lower quality → higher regularisation
            reg_strength = max(0.001, (100.0 - quality_estimate) / 100.0 * 0.1)
        else:
            # Auto: proportional to estimated noise level
            reg_strength = max(0.001, sigma * 5.0)

        # FFT-based Wiener filter
        F = fftpack.fft2(channel)
        power = np.abs(F) ** 2
        noise_power = reg_strength * np.mean(power)

        # H_wiener = |F|^2 / (|F|^2 + N)
        wiener_filter = power / (power + noise_power)
        recovered_channel = np.real(fftpack.ifft2(F * wiener_filter))

        recovered[:, :, c] = recovered_channel

    return recovered


def compute_pce(noise_residual, prnu_pattern):
    """
    Compute the Peak-to-Correlation Energy (PCE) ratio between a noise
    residual and a PRNU pattern.

    PCE is the standard metric for PRNU-based source identification.
    A high PCE (> ~50-60) strongly indicates the image was captured by the
    same camera as the PRNU pattern.  AI-generated images will typically
    have very low PCE values.

    Args:
        noise_residual (np.ndarray): Noise residual of the test image (H, W, 3).
        prnu_pattern (np.ndarray): Reference PRNU pattern (H, W, 3).

    Returns:
        float: PCE value.  Higher = more likely a real camera image.
    """
    # Ensure same shape (crop to minimum)
    min_h = min(noise_residual.shape[0], prnu_pattern.shape[0])
    min_w = min(noise_residual.shape[1], prnu_pattern.shape[1])
    nr = noise_residual[:min_h, :min_w]
    pp = prnu_pattern[:min_h, :min_w]

    pce_per_channel = []
    for c in range(nr.shape[-1]):
        nr_c = nr[:, :, c]
        pp_c = pp[:, :, c]

        # Cross-correlation via FFT
        F_nr = fftpack.fft2(nr_c)
        F_pp = fftpack.fft2(pp_c)
        cross_corr = np.real(fftpack.ifft2(F_nr * np.conj(F_pp)))

        # PCE = peak^2 / mean(energy excluding peak neighbourhood)
        peak_idx = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
        peak_value = cross_corr[peak_idx]

        # Exclude a small neighbourhood around the peak (11×11)
        mask = np.ones_like(cross_corr, dtype=bool)
        r, col = peak_idx
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                rr = (r + dr) % cross_corr.shape[0]
                cc = (col + dc) % cross_corr.shape[1]
                mask[rr, cc] = False

        noise_energy = np.mean(cross_corr[mask] ** 2)
        if noise_energy > 0:
            pce_per_channel.append((peak_value ** 2) / noise_energy)
        else:
            pce_per_channel.append(0.0)

    return float(np.mean(pce_per_channel))


def estimate_jpeg_quality(image_data):
    """
    Estimate the JPEG quality factor from raw image bytes.

    Uses quantisation table analysis when available; otherwise falls back
    to a noise-level heuristic.

    Args:
        image_data (bytes): Raw image file bytes.

    Returns:
        float | None: Estimated quality 0-100, or None if not JPEG.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.format != 'JPEG':
            return None

        # Try reading quantization tables
        qtables = img.quantization
        if qtables:
            # Average quantisation step across luminance table
            luma_table = list(qtables[0].values()) if isinstance(qtables[0], dict) else list(qtables[0])
            avg_quant = np.mean(luma_table)
            # Heuristic mapping: avg_quant ≈ 1 → q100, avg_quant ≈ 25 → q50
            quality = max(0, min(100, 100 - (avg_quant - 1) * 2))
            return float(quality)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
#  High-level API  (used by detect.py and server.py)
# ---------------------------------------------------------------------------

def analyze_prnu(image_data):
    """
    Perform a full PRNU-based forensic analysis on a single image.

    This is the main entry point called by the detection pipeline.
    It extracts the noise residual, optionally recovers it from compression
    artefacts, and computes a set of forensic metrics.

    Args:
        image_data (bytes): Raw image file data (JPEG, PNG, etc.).

    Returns:
        dict: Analysis results containing:
            - noise_strength (float): RMS magnitude of the noise residual.
                Real images have consistent, moderate noise; AI images tend
                toward very low or unnaturally uniform noise.
            - noise_uniformity (float 0-1): How uniform the noise is across
                the image.  AI-generated images often have very uniform noise
                distributions (high value), while real images have spatially
                varying sensor noise (lower value).
            - blockiness_score (float 0-1): Detects 8×8 block artefacts
                typical of JPEG compression.  Useful for judging how much
                PRNU signal may have been lost.
            - compression_quality (float | None): Estimated JPEG quality.
            - prnu_likelihood_real (float 0-1): Overall likelihood that the
                image contains a genuine camera PRNU pattern (higher = more
                likely real).
    """
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(img, dtype=np.float64) / 255.0

        # 1. Extract noise residual
        noise = extract_noise_residual(img_array)

        # 2. Attempt PRNU recovery if JPEG compressed
        quality = estimate_jpeg_quality(image_data)
        if quality is not None and quality < 95:
            noise = recover_prnu_from_compressed(noise, quality_estimate=quality)

        # 3. Compute forensic metrics
        noise_strength = float(np.sqrt(np.mean(noise ** 2)))

        # Noise uniformity: compare local noise variance across 64×64 blocks
        block_size = 64
        h, w = noise.shape[:2]
        local_variances = []
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = noise[i:i + block_size, j:j + block_size]
                local_variances.append(np.var(block))

        if local_variances:
            var_array = np.array(local_variances)
            mean_var = np.mean(var_array)
            std_var = np.std(var_array)
            # High uniformity = low coefficient of variation
            noise_uniformity = float(1.0 / (1.0 + std_var / (mean_var + 1e-10)))
        else:
            noise_uniformity = 0.5

        # Blockiness: detect 8×8 grid artefacts (JPEG compression)
        blockiness_score = _compute_blockiness(img_array)

        # 4. Compute PRNU likelihood
        # Combine metrics into a single real-vs-AI score
        prnu_likelihood_real = _compute_prnu_likelihood(
            noise_strength, noise_uniformity, blockiness_score
        )

        return {
            "noise_strength": round(noise_strength, 6),
            "noise_uniformity": round(noise_uniformity, 4),
            "blockiness_score": round(blockiness_score, 4),
            "compression_quality": round(quality, 1) if quality is not None else None,
            "prnu_likelihood_real": round(prnu_likelihood_real, 4),
        }

    except Exception as e:
        print(f"PRNU analysis error: {e}")
        return {
            "noise_strength": None,
            "noise_uniformity": None,
            "blockiness_score": None,
            "compression_quality": None,
            "prnu_likelihood_real": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _compute_blockiness(image_array):
    """
    Detect 8×8 block boundary artefacts characteristic of JPEG compression.

    Compares pixel differences across block boundaries vs. within blocks.
    Higher blockiness means more compression artefacts (and more PRNU loss).

    Returns:
        float: Blockiness score in [0, 1].
    """
    gray = np.mean(image_array, axis=-1)
    h, w = gray.shape

    # Horizontal block boundaries (columns 7, 15, 23, ...)
    boundary_diffs = []
    interior_diffs = []

    for col in range(1, w):
        diff = np.mean(np.abs(gray[:, col] - gray[:, col - 1]))
        if col % 8 == 0:
            boundary_diffs.append(diff)
        else:
            interior_diffs.append(diff)

    # Vertical block boundaries (rows 7, 15, 23, ...)
    for row in range(1, h):
        diff = np.mean(np.abs(gray[row, :] - gray[row - 1, :]))
        if row % 8 == 0:
            boundary_diffs.append(diff)
        else:
            interior_diffs.append(diff)

    if not boundary_diffs or not interior_diffs:
        return 0.0

    mean_boundary = np.mean(boundary_diffs)
    mean_interior = np.mean(interior_diffs)

    # Ratio: if boundaries are stronger than interior, blockiness is present
    if mean_interior > 0:
        ratio = mean_boundary / mean_interior
        # Normalise to [0, 1] — ratio near 1.0 = no blockiness
        blockiness = max(0.0, min(1.0, (ratio - 1.0) * 5.0))
    else:
        blockiness = 0.0

    return blockiness


def _compute_prnu_likelihood(noise_strength, noise_uniformity, blockiness_score):
    """
    Compute the overall likelihood that an image contains a genuine PRNU pattern.

    Heuristic scoring based on:
      - Real images have moderate, spatially-varying noise (not too uniform)
      - AI images tend to have very low noise or highly uniform noise
      - Heavily compressed images may have low PRNU but still show blockiness

    Returns:
        float: Score in [0, 1] where 1 = very likely real camera image.
    """
    score = 0.5  # Start neutral

    # Factor 1: Noise strength
    # Real camera images have moderate noise (0.005-0.05 RMS typically)
    # AI-generated images tend to have very low noise or no natural noise
    if 0.003 < noise_strength < 0.08:
        score += 0.2  # Good range for real images
    elif noise_strength < 0.001:
        score -= 0.3  # Suspiciously clean → likely AI
    elif noise_strength > 0.1:
        score -= 0.1  # Unusually noisy

    # Factor 2: Noise uniformity
    # Real PRNU noise is spatially varying; AI noise is more uniform
    if noise_uniformity > 0.85:
        score -= 0.2  # Too uniform → likely AI
    elif noise_uniformity < 0.6:
        score += 0.15  # Good spatial variation → likely real

    # Factor 3: Blockiness
    # Moderate blockiness suggests real JPEG (real image)
    # No blockiness with very uniform noise suggests AI
    if 0.05 < blockiness_score < 0.5:
        score += 0.1  # Some compression artefacts (typical for real photos)
    elif blockiness_score > 0.7:
        score -= 0.05  # Very heavy compression — PRNU unreliable

    return max(0.0, min(1.0, score))
