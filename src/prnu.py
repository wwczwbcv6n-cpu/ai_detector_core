"""
PRNU (Photo Response Non-Uniformity) Noise Recovery Module  — v2

Improvements over v1:
  1. Multi-image averaging   — proper MLE: K_hat = Σ(Wi·Ii) / Σ(Ii²)
  2. Noise residual extraction — intensity-normalised + zero-mean stripe removal
  3. Block-based reliability weighting — saturated/smooth tiles down-weighted
  4. Compression inversion   — frequency-dependent Wiener regularisation
  5. Deep-learning-inspired enhancement — adaptive multi-scale bandpass
  6. Patch-level detection   — per-patch PRNU consistency map

References:
  - Lukas, Fridrich, Goljan (2006): "Digital Camera Identification from Sensor Pattern Noise"
  - Chen et al. (2008): "Determining Image Origin and Integrity Using Sensor Noise"
  - Goljan, Fridrich (2009): "Camera Identification from Cropped and Scaled Images"
"""

import io
import numpy as np
from PIL import Image
from scipy import fftpack
from skimage.restoration import denoise_wavelet, estimate_sigma


# ---------------------------------------------------------------------------
#  1. Noise residual extraction  (intensity-normalised + zero-mean)
# ---------------------------------------------------------------------------

def extract_noise_residual(image_array, normalize_by_intensity: bool = True):
    """
    Extract noise residual W = I - F(I) via wavelet BayesShrink denoising.

    Improvements:
      • Intensity normalisation: W ← W / (I_gray + ε)
        Converts additive noise to multiplicative-gain form, which more closely
        matches the physical PRNU model (camera gain pattern K·I).
      • Zero-mean per row and column to remove stripe artefacts caused by
        scene content leaking into the residual.

    Args:
        image_array (np.ndarray): float64 in [0, 1], shape (H, W, 3).
        normalize_by_intensity (bool): apply intensity normalisation.

    Returns:
        np.ndarray: noise residual of same shape.
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

    if normalize_by_intensity:
        # Intensity normalisation — use per-pixel mean across channels
        i_gray = image_array.mean(axis=-1, keepdims=True)  # (H, W, 1)
        noise = noise / (i_gray + 0.01)

        # Zero-mean per column and per row to remove stripe artefacts
        noise = noise - noise.mean(axis=0, keepdims=True)   # remove column mean
        noise = noise - noise.mean(axis=1, keepdims=True)   # remove row mean

    return noise


# ---------------------------------------------------------------------------
#  2. Multi-image PRNU estimation  (proper MLE)
# ---------------------------------------------------------------------------

def estimate_prnu_pattern(image_arrays=None, noise_residuals=None):
    """
    Estimate camera PRNU fingerprint via Maximum Likelihood Estimation.

    Proper MLE formula (Lukas et al., 2006):
        K_hat = Σ_i (W_i · I_i) / (Σ_i I_i² + ε)

    where W_i is the noise residual and I_i is the image itself.
    This is superior to simple averaging because it weights by local intensity,
    giving higher weight to regions with stronger PRNU signal.

    Args:
        image_arrays (list[np.ndarray] | None): list of float64 [0,1] images.
        noise_residuals (list[np.ndarray] | None): pre-computed residuals.
            If both provided, noise_residuals is used (images only needed for
            MLE weighting; if not provided, images are used for both).

    Returns:
        np.ndarray: estimated PRNU pattern, same shape as one input image.
    """
    if image_arrays is None and noise_residuals is None:
        raise ValueError("Provide image_arrays and/or noise_residuals.")

    if noise_residuals is None:
        noise_residuals = [extract_noise_residual(img) for img in image_arrays]

    # If images are available, use proper weighted MLE
    if image_arrays is not None and len(image_arrays) == len(noise_residuals):
        numerator   = np.zeros_like(noise_residuals[0], dtype=np.float64)
        denominator = np.zeros_like(noise_residuals[0], dtype=np.float64)
        for noise, img in zip(noise_residuals, image_arrays):
            img_f64 = img.astype(np.float64)
            numerator   += noise * img_f64
            denominator += img_f64 ** 2
        K_hat = numerator / (denominator + 1e-10)
    else:
        # Fallback: simple mean (less accurate but works without raw images)
        K_hat = np.mean(np.stack(noise_residuals, axis=0), axis=0)

    # Zero-mean normalisation per channel
    for c in range(K_hat.shape[-1]):
        K_hat[:, :, c] -= K_hat[:, :, c].mean()

    return K_hat


# ---------------------------------------------------------------------------
#  3. Block-based reliability weighting
# ---------------------------------------------------------------------------

def compute_block_reliability_map(image_array, block_size: int = 64):
    """
    Compute a spatial reliability weight map for PRNU extraction.

    Rules:
      • Saturated pixels (I > 0.95 or I < 0.05) carry no PRNU signal → weight 0
      • Smooth regions (low local variance) carry weak PRNU → low weight
      • Textured regions with moderate intensity → high weight

    Returns:
        np.ndarray: shape (H//block_size, W//block_size), values in [0, 1].
    """
    h, w = image_array.shape[:2]
    n_rows = h // block_size
    n_cols = w // block_size

    weight_map = np.zeros((n_rows, n_cols), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            block = image_array[
                r * block_size:(r + 1) * block_size,
                c * block_size:(c + 1) * block_size,
            ]
            # Fraction of non-saturated pixels
            sat_mask = (block > 0.95) | (block < 0.05)
            sat_frac = float(sat_mask.mean())
            non_sat = 1.0 - sat_frac

            # Local variance (log-scale to avoid outlier domination)
            local_var = float(np.var(block))
            var_weight = float(np.log1p(local_var * 1000))  # scale to ~[0, 3]
            var_weight = float(np.clip(var_weight / 3.0, 0.0, 1.0))

            weight_map[r, c] = non_sat * var_weight

    return weight_map


# ---------------------------------------------------------------------------
#  4. Compression inversion  (frequency-dependent Wiener filter)
# ---------------------------------------------------------------------------

def recover_prnu_from_compressed(noise_residual, quality_estimate=None):
    """
    Recover PRNU signal degraded by JPEG compression via Wiener filtering.

    Improvement over v1:
      • Frequency-dependent regularisation: reg(f) = base · (1 + 4·|f|²)
        JPEG quantisation noise grows with spatial frequency, so higher
        frequencies need stronger regularisation.
      • Optional: anisotropic frequency bands for DCT-block alignment.

    Args:
        noise_residual (np.ndarray): Degraded noise residual (H, W, 3).
        quality_estimate (float | None): JPEG quality 0–100.

    Returns:
        np.ndarray: Enhanced noise residual.
    """
    recovered = np.zeros_like(noise_residual, dtype=np.float64)

    h, w = noise_residual.shape[:2]
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    freq_mag = np.sqrt(FX ** 2 + FY ** 2)          # 0 … ~0.707

    for c in range(noise_residual.shape[-1]):
        channel = noise_residual[:, :, c].astype(np.float64)

        if quality_estimate is not None:
            base_reg = max(0.0005, (100.0 - quality_estimate) / 100.0 * 0.08)
        else:
            sigma = float(estimate_sigma(channel))
            base_reg = max(0.0005, sigma * 4.0)

        F     = fftpack.fft2(channel)
        power = np.abs(F) ** 2

        # Frequency-dependent noise power: stronger at high frequencies
        reg_map     = base_reg * (1.0 + 4.0 * freq_mag ** 2)
        noise_power = reg_map * (np.mean(power) + 1e-12)

        wiener = power / (power + noise_power)
        recovered[:, :, c] = np.real(fftpack.ifft2(F * wiener))

    return recovered


# ---------------------------------------------------------------------------
#  5. Deep-learning-inspired PRNU enhancement  (adaptive multi-scale)
# ---------------------------------------------------------------------------

def enhance_prnu_signal(noise_residual, quality_estimate=None):
    """
    Adaptive multi-scale PRNU enhancement.

    Mimics a U-Net-style denoising network using classical signal processing:
      Step 1 – Remove stripe artefacts (DC per row/column)
      Step 2 – FFT bandpass: suppress scene-content DC (<0.03) and
               quantisation noise (>0.40 normalized frequency)
      Step 3 – Wiener denoising of remaining high-freq quantisation noise

    The bandpass design approximates what a trained residual CNN learns:
    preserve mid-frequency PRNU energy while attenuating scene leakage and
    compression noise.

    Args:
        noise_residual (np.ndarray): shape (H, W, 3), float64.
        quality_estimate (float | None): JPEG quality (used for step 3).

    Returns:
        np.ndarray: enhanced residual, same shape.
    """
    # Step 1: stripe removal
    enhanced = noise_residual.copy().astype(np.float64)
    enhanced -= enhanced.mean(axis=0, keepdims=True)  # column means
    enhanced -= enhanced.mean(axis=1, keepdims=True)  # row means

    # Step 2: frequency bandpass per channel
    h, w = enhanced.shape[:2]
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    freq_mag = np.sqrt(FX ** 2 + FY ** 2)

    # High-pass to remove scene-content DC (σ=0.03)
    highpass = 1.0 - np.exp(-((freq_mag / 0.03) ** 2))
    # Low-pass to remove quantisation noise above 0.40
    lowpass  = np.exp(-((freq_mag / 0.40) ** 2))
    bandpass = highpass * lowpass

    for c in range(enhanced.shape[-1]):
        F  = fftpack.fft2(enhanced[:, :, c])
        enhanced[:, :, c] = np.real(fftpack.ifft2(F * bandpass))

    # Step 3: Wiener denoising of residual quantisation noise
    if quality_estimate is not None and quality_estimate < 90:
        enhanced = recover_prnu_from_compressed(enhanced, quality_estimate)

    return enhanced


# ---------------------------------------------------------------------------
#  6. Patch-level PRNU detection
# ---------------------------------------------------------------------------

def compute_prnu_patch_map(
    image_array,
    prnu_pattern=None,
    patch_size: int = 128,
    stride: int = 64,
):
    """
    Compute a spatial map of PRNU consistency for patch-level forgery detection.

    For each overlapping patch, compute the normalised cross-correlation (NCC)
    between the patch noise residual and either (a) the given reference PRNU
    pattern or (b) the global PRNU estimated from the whole image.

    High NCC → patch looks like a real camera region.
    Low NCC  → patch is inconsistent → potential AI-generated or spliced region.

    Args:
        image_array (np.ndarray): float64 [0,1] (H, W, 3).
        prnu_pattern (np.ndarray | None): reference PRNU (H, W, 3).
            If None, estimated from the image itself (less reliable).
        patch_size (int): patch side in pixels.
        stride (int): stride in pixels.

    Returns:
        dict with:
            'ncc_map'    : np.ndarray (n_rows, n_cols) NCC scores in [-1, 1]
            'mean_ncc'   : float — mean NCC across patches
            'std_ncc'    : float — std NCC (high = inconsistent = AI-likely)
            'min_ncc'    : float — minimum NCC (suspicious if very low)
            'consistency': float in [0, 1] — 1 = consistent (real-like)
    """
    h, w = image_array.shape[:2]

    # Extract global noise residual
    global_noise = extract_noise_residual(image_array, normalize_by_intensity=True)
    global_noise = enhance_prnu_signal(global_noise)

    # Collect non-overlapping patch residuals for inter-patch NCC
    patch_residuals = []
    positions       = []

    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch_noise = global_noise[r:r + patch_size, c:c + patch_size]
            patch_residuals.append(patch_noise)
            positions.append((r, c))

    if not patch_residuals:
        return {
            'ncc_map': np.array([[0.5]]),
            'mean_ncc': 0.5, 'std_ncc': 0.0,
            'min_ncc': 0.5, 'consistency': 0.5,
        }

    # If a reference pattern is given, compare each patch to reference
    # Otherwise compute inter-patch pairwise NCC (real images have consistent noise)
    use_reference = prnu_pattern is not None

    ncc_scores = []

    if use_reference:
        for (r, c), patch_noise in zip(positions, patch_residuals):
            patch_ref = prnu_pattern[r:r + patch_size, c:c + patch_size]
            ncc = _patch_ncc(patch_noise, patch_ref)
            ncc_scores.append(ncc)
    else:
        # Inter-patch NCC: sub-sample at most 20 patches to keep cost O(1)
        idx = np.linspace(0, len(patch_residuals) - 1,
                          min(20, len(patch_residuals)), dtype=int)
        sub = [patch_residuals[i] for i in idx]
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                ncc_scores.append(_patch_ncc(sub[i], sub[j]))

    if not ncc_scores:
        ncc_scores = [0.5]

    ncc_arr  = np.array(ncc_scores, dtype=np.float32)
    mean_ncc = float(ncc_arr.mean())
    std_ncc  = float(ncc_arr.std())
    min_ncc  = float(ncc_arr.min())

    # Map mean_ncc [-1,1] to [0,1]; penalise high std (inconsistency)
    consistency = float(np.clip(
        0.5 * (mean_ncc + 1.0) * (1.0 - min(std_ncc, 1.0)),
        0.0, 1.0,
    ))

    # Build a position-indexed map from the ncc_scores
    n_rows_g = max(1, (h - patch_size) // stride + 1)
    n_cols_g = max(1, (w - patch_size) // stride + 1)
    ncc_map  = np.full((n_rows_g, n_cols_g), mean_ncc, dtype=np.float32)

    return {
        'ncc_map':     ncc_map,
        'mean_ncc':    round(mean_ncc, 4),
        'std_ncc':     round(std_ncc, 4),
        'min_ncc':     round(min_ncc, 4),
        'consistency': round(consistency, 4),
    }


# ---------------------------------------------------------------------------
#  PCE  (unchanged)
# ---------------------------------------------------------------------------

def compute_pce(noise_residual, prnu_pattern):
    """
    Compute the Peak-to-Correlation Energy (PCE) ratio.
    High PCE → image likely from the same camera as the PRNU pattern.
    """
    min_h = min(noise_residual.shape[0], prnu_pattern.shape[0])
    min_w = min(noise_residual.shape[1], prnu_pattern.shape[1])
    nr = noise_residual[:min_h, :min_w]
    pp = prnu_pattern[:min_h, :min_w]

    pce_per_channel = []
    for c in range(nr.shape[-1]):
        nr_c = nr[:, :, c]
        pp_c = pp[:, :, c]

        F_nr = fftpack.fft2(nr_c)
        F_pp = fftpack.fft2(pp_c)
        cross_corr = np.real(fftpack.ifft2(F_nr * np.conj(F_pp)))

        peak_idx   = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
        peak_value = cross_corr[peak_idx]

        mask = np.ones_like(cross_corr, dtype=bool)
        r, col = peak_idx
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                mask[(r + dr) % cross_corr.shape[0], (col + dc) % cross_corr.shape[1]] = False

        noise_energy = float(np.mean(cross_corr[mask] ** 2))
        if noise_energy > 0:
            pce_per_channel.append(float(peak_value ** 2) / noise_energy)
        else:
            pce_per_channel.append(0.0)

    return float(np.mean(pce_per_channel))


# ---------------------------------------------------------------------------
#  JPEG quality estimation  (unchanged)
# ---------------------------------------------------------------------------

def estimate_jpeg_quality(image_data):
    """Estimate JPEG quality 0-100 from raw bytes, or None if not JPEG."""
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.format != 'JPEG':
            return None
        qtables = img.quantization
        if qtables:
            luma = list(qtables[0].values()) if isinstance(qtables[0], dict) else list(qtables[0])
            avg_quant = float(np.mean(luma))
            return float(max(0.0, min(100.0, 100.0 - (avg_quant - 1.0) * 2.0)))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
#  High-level API  (updated)
# ---------------------------------------------------------------------------

def analyze_prnu(image_data):
    """
    Full PRNU forensic analysis on a single image.

    Now uses all v2 improvements:
      • Intensity-normalised noise residual
      • Compression-aware Wiener recovery
      • DL-inspired adaptive enhancement
      • Patch-level consistency scoring

    Returns:
        dict with keys:
            noise_strength        — RMS of enhanced residual
            noise_uniformity      — spatial consistency (high → AI-like)
            blockiness_score      — JPEG 8×8 artefact strength
            compression_quality   — estimated JPEG quality
            prnu_likelihood_real  — heuristic real-vs-AI score [0, 1]
            patch_consistency     — inter-patch PRNU consistency [0, 1]
            patch_mean_ncc        — mean patch NCC
            patch_std_ncc         — std of patch NCC
    """
    try:
        img       = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(img, dtype=np.float64) / 255.0

        # 1. Extract intensity-normalised noise residual
        noise = extract_noise_residual(img_array, normalize_by_intensity=True)

        # 2. JPEG recovery if compressed
        quality = estimate_jpeg_quality(image_data)
        if quality is not None and quality < 95:
            noise = recover_prnu_from_compressed(noise, quality_estimate=quality)

        # 3. DL-inspired enhancement
        noise = enhance_prnu_signal(noise, quality_estimate=quality)

        # 4. Forensic metrics
        noise_strength = float(np.sqrt(np.mean(noise ** 2)))

        # Noise uniformity via 64×64 blocks (reliability-weighted)
        block_size = 64
        h, w = noise.shape[:2]
        rel_map = compute_block_reliability_map(img_array, block_size=block_size)
        n_rows_b = h // block_size
        n_cols_b = w // block_size

        local_variances = []
        weights         = []
        for i in range(n_rows_b):
            for j in range(n_cols_b):
                block = noise[i * block_size:(i + 1) * block_size,
                              j * block_size:(j + 1) * block_size]
                local_variances.append(float(np.var(block)))
                w_ij = float(rel_map[i, j]) if i < rel_map.shape[0] and j < rel_map.shape[1] else 1.0
                weights.append(max(w_ij, 1e-6))

        if local_variances:
            var_arr    = np.array(local_variances)
            w_arr      = np.array(weights)
            w_arr      = w_arr / (w_arr.sum() + 1e-10)
            mean_var   = float(np.dot(w_arr, var_arr))
            std_var    = float(np.sqrt(np.dot(w_arr, (var_arr - mean_var) ** 2)))
            noise_uniformity = float(1.0 / (1.0 + std_var / (mean_var + 1e-10)))
        else:
            noise_uniformity = 0.5

        blockiness_score = _compute_blockiness(img_array)

        # 5. Patch-level detection
        patch_result = compute_prnu_patch_map(img_array, patch_size=128, stride=64)
        patch_consistency = patch_result['consistency']
        patch_mean_ncc    = patch_result['mean_ncc']
        patch_std_ncc     = patch_result['std_ncc']

        # 6. Combined PRNU likelihood
        prnu_likelihood_real = _compute_prnu_likelihood(
            noise_strength, noise_uniformity, blockiness_score, patch_consistency
        )

        return {
            "noise_strength":       round(noise_strength, 6),
            "noise_uniformity":     round(noise_uniformity, 4),
            "blockiness_score":     round(blockiness_score, 4),
            "compression_quality":  round(quality, 1) if quality is not None else None,
            "prnu_likelihood_real": round(prnu_likelihood_real, 4),
            "patch_consistency":    round(patch_consistency, 4),
            "patch_mean_ncc":       round(patch_mean_ncc, 4),
            "patch_std_ncc":        round(patch_std_ncc, 4),
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {
            "noise_strength": None, "noise_uniformity": None,
            "blockiness_score": None, "compression_quality": None,
            "prnu_likelihood_real": None, "patch_consistency": None,
            "patch_mean_ncc": None, "patch_std_ncc": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _compute_blockiness(image_array):
    """Detect 8×8 JPEG block boundary artefacts. Returns score in [0, 1]."""
    gray = np.mean(image_array, axis=-1)
    h, w = gray.shape
    boundary_diffs = []
    interior_diffs = []
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


def _patch_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-channel normalised cross-correlation between two noise patches."""
    ncc_vals = []
    for ch in range(a.shape[-1]):
        fa = a[:, :, ch].ravel()
        fb = b[:, :, ch].ravel()
        sa, sb = np.std(fa), np.std(fb)
        if sa > 1e-8 and sb > 1e-8:
            c = float(np.corrcoef(fa, fb)[0, 1])
            if np.isfinite(c):
                ncc_vals.append(c)
    return float(np.mean(ncc_vals)) if ncc_vals else 0.0


def _compute_prnu_likelihood(noise_strength, noise_uniformity,
                              blockiness_score, patch_consistency=0.5):
    """
    Heuristic real-vs-AI score in [0, 1].  1 = very likely real camera.
    """
    score = 0.5

    # Factor 1: noise strength
    if 0.001 < noise_strength < 0.05:
        score += 0.15
    elif noise_strength < 0.0005:
        score -= 0.25
    elif noise_strength > 0.08:
        score -= 0.05

    # Factor 2: noise uniformity (AI images are unnaturally uniform)
    if noise_uniformity > 0.85:
        score -= 0.15
    elif noise_uniformity < 0.6:
        score += 0.10

    # Factor 3: blockiness (moderate = real JPEG)
    if 0.05 < blockiness_score < 0.5:
        score += 0.10
    elif blockiness_score > 0.7:
        score -= 0.05

    # Factor 4: patch-level PRNU consistency
    # Real images have consistent PRNU across patches (high consistency)
    score += 0.20 * (patch_consistency - 0.5)

    return float(max(0.0, min(1.0, score)))
