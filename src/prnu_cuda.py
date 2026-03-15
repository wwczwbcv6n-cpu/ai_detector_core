"""
prnu_cuda.py — GPU-Accelerated PRNU Feature Extractor

Replaces scipy wavelet (~200ms/img CPU) with Gaussian high-pass approximation
on GPU (~2ms/img).  Used during training; detect.py still uses the accurate
scipy path for inference.

Architecture:
    noise = img - gaussian_blur(img)   # fixed 7×7 separable kernel, not learned
    noise = noise / (mean_gray + 0.01) # intensity normalisation
    noise -= noise.mean(dim=-1)        # stripe removal (horizontal)
    noise -= noise.mean(dim=-2)        # stripe removal (vertical)

PRNUExtractorGPU.extract_both() returns:
    (prnu_feats (B,64), prnu_map (B,3,128,128))

Scalar features layout matches extract_prnu_features_fullres() exactly:
    [0..7]   — noise stats (strength, uniformity, blockiness proxy,
               freq_energy_ratio, skewness_norm, kurtosis_norm, hf_ratio, 0)
    [8..14]  — per-channel std of noise patches (7 values)
    [15]     — spatial consistency proxy (inter-patch correlation)
    [16..19] — PRNU energy in 4 freq bands (LF/mid/HF/VHF)
    [20..23] — cross-channel correlations (R-G, R-B, G-B, mean)
    [24..31] — zeros (recovery stats skipped in fast GPU path)
    [32..35] — extended high-freq bands (EHF1-4, above VHF)
    [36..39] — per-channel noise RMS (R, G, B, Lum)
    [40..43] — noise anisotropy (row_corr, col_corr, asymmetry, stripe_ratio)
    [44..47] — Bayer CFA residual proxy (period_2, period_4, cfa_score, 0)
    [48..51] — phase coherence (low, mid, high, mean)
    [52..55] — directional correlation (H, V, diag, anti-diag)
    [56..59] — saturation features (clip_frac, dyn_range_norm, 0, snr_proxy)
    [60..63] — multi-scale consistency (64↔128, 128↔256, 64↔256, mean)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PRNUExtractorGPU(nn.Module):
    """
    Fixed Gaussian high-pass PRNU noise extractor — GPU-only, no gradients.

    Args:
        device   : torch.device to place fixed buffers on
        map_size : spatial output size for extract_map() (default 64)
        sigma    : Gaussian blur std dev (default 1.5)
    """

    def __init__(self, device: torch.device, map_size: int = 128, sigma: float = 1.5):
        super().__init__()
        self.map_size = map_size
        self.device   = device

        # Build 7×7 separable Gaussian kernel
        kernel_1d = self._gaussian_kernel_1d(7, sigma)                # (7,)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]           # (7,7)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)               # (1,1,7,7)
        # Apply same kernel to each of 3 channels independently (groups=3)
        kernel_3ch = kernel_2d.repeat(3, 1, 1, 1)                     # (3,1,7,7)

        self.register_buffer('gauss_kernel', kernel_3ch.to(device))
        self.to(device)

    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        return g / g.sum()

    def _extract_noise(self, img: torch.Tensor) -> torch.Tensor:
        """
        (B,3,H,W) float32 [0,1] → (B,3,H,W) noise residual, zero-striped.
        """
        # Gaussian blur with reflect padding
        blurred = F.conv2d(
            F.pad(img, [3, 3, 3, 3], mode='reflect'),
            self.gauss_kernel,
            groups=3,
        )
        noise = img - blurred

        # Intensity normalisation: W ← (I - F(I)) / (I_gray + 0.01)
        i_gray = img.mean(dim=1, keepdim=True)          # (B,1,H,W)
        noise  = noise / (i_gray + 0.01)

        # Stripe removal
        noise = noise - noise.mean(dim=-1, keepdim=True)   # horizontal stripes
        noise = noise - noise.mean(dim=-2, keepdim=True)   # vertical stripes

        return noise

    @torch.no_grad()
    def extract_map(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial PRNU map at fixed resolution.

        Args:
            img : (B,3,H,W) float32 [0,1] on self.device

        Returns:
            (B,3,map_size,map_size) float32  — noise map in [-1,1]
        """
        img = img.to(self.device)
        noise = self._extract_noise(img)                   # (B,3,H,W)

        # Downsample to map_size × map_size
        noise_map = F.interpolate(
            noise, size=(self.map_size, self.map_size),
            mode='bilinear', align_corners=False,
        )

        # Normalise to [-1, 1] — matches CPU extract_prnu_map() output range
        # Per-sample min-max norm so each image is independently scaled
        B = noise_map.shape[0]
        flat   = noise_map.view(B, -1)
        lo     = flat.min(dim=1)[0].view(B, 1, 1, 1)
        hi     = flat.max(dim=1)[0].view(B, 1, 1, 1)
        span   = (hi - lo).clamp(min=1e-6)
        noise_map = (noise_map - lo) / span * 2.0 - 1.0   # → [-1, 1]
        return noise_map

    @torch.no_grad()
    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract 64-dim scalar PRNU feature vector (GPU path).
        Layout matches extract_prnu_features_fullres() from prnu_features.py.

        Args:
            img : (B,3,H,W) float32 [0,1] on self.device

        Returns:
            (B,64) float32
        """
        img   = img.to(self.device)
        B     = img.shape[0]
        noise = self._extract_noise(img)                   # (B,3,H,W)

        feats = torch.zeros(B, 64, device=self.device, dtype=torch.float32)

        # ── [0] noise_strength ─────────────────────────────────────────────
        rms = noise.pow(2).mean(dim=[1, 2, 3]).sqrt()     # (B,)
        feats[:, 0] = (rms / 0.15).clamp(0.0, 1.0)

        # ── [1] noise_uniformity (spatial variance of patch energies) ──────
        # Compute 16×16 patch energies via unfold
        try:
            ph = 16
            nh = noise.shape[2] // ph
            nw = noise.shape[3] // ph
            if nh >= 2 and nw >= 2:
                # Average across channels first
                gray_noise = noise.mean(dim=1, keepdim=True)  # (B,1,H,W)
                # Patch variances using avg_pool
                patch_e = F.avg_pool2d(
                    gray_noise.pow(2),
                    kernel_size=ph, stride=ph,
                )                                              # (B,1,nh,nw)
                patch_e = patch_e.view(B, -1)                 # (B, nh*nw)
                p_mean  = patch_e.mean(dim=1, keepdim=True)
                p_std   = patch_e.std(dim=1)
                uniformity = 1.0 / (1.0 + p_std / (p_mean.squeeze(1) + 1e-10))
                feats[:, 1] = uniformity.clamp(0.0, 1.0)
        except Exception:
            feats[:, 1] = 0.5

        # ── [2] blockiness proxy (8-px period boundary diff ratio) ─────────
        try:
            gray = img.mean(dim=1)                             # (B,H,W)
            # Horizontal boundary at 8-px columns
            boundary_mask = torch.zeros(gray.shape[2], device=self.device, dtype=torch.bool)
            boundary_mask[7::8] = True                         # columns 7,15,23,...
            h_diff = (gray[:, :, 1:] - gray[:, :, :-1]).abs()  # (B,H,W-1)
            boundary_diff = h_diff[:, :, boundary_mask[:h_diff.shape[2]]].mean(dim=[1, 2])
            interior_diff = h_diff[:, :, ~boundary_mask[:h_diff.shape[2]]].mean(dim=[1, 2])
            ratio = boundary_diff / (interior_diff + 1e-10)
            feats[:, 2] = ((ratio - 1.0) * 5.0).clamp(0.0, 1.0)
        except Exception:
            feats[:, 2] = 0.0

        # ── [3] freq_energy_ratio (high-freq fraction via rfft2) ───────────
        try:
            gray_noise = noise.mean(dim=1)                     # (B,H,W)
            F_noise    = torch.fft.rfft2(gray_noise)           # (B,H,W//2+1)
            power      = F_noise.abs().pow(2)
            h, w       = power.shape[1], power.shape[2]
            # Build frequency-distance mask (normalised to [0,1])
            fy = torch.arange(h, device=self.device, dtype=torch.float32) / h
            fx = torch.arange(w, device=self.device, dtype=torch.float32) / w
            fy = torch.minimum(fy, 1.0 - fy)                  # wrap
            FX, FY = torch.meshgrid(fx, fy, indexing='xy')
            dist   = (FX.pow(2) + FY.pow(2)).sqrt()           # (h,w) broadcast

            total_e = power.sum(dim=[1, 2]).clamp(min=1e-12)
            hf_mask = dist.unsqueeze(0) >= 0.5                 # (1,h,w) → broadcasts over B
            hf_e    = (power * hf_mask).sum(dim=[1, 2])
            feats[:, 3] = (hf_e / total_e).clamp(0.0, 1.0)
        except Exception:
            feats[:, 3] = 0.5

        # ── [4..5] skewness_norm, kurtosis_norm ────────────────────────────
        try:
            flat    = noise.view(B, -1)                        # (B, C*H*W)
            n_mean  = flat.mean(dim=1, keepdim=True)
            n_std   = flat.std(dim=1, keepdim=True).clamp(min=1e-10)
            z       = (flat - n_mean) / n_std
            skew    = z.pow(3).mean(dim=1)
            kurt    = z.pow(4).mean(dim=1) - 3.0
            feats[:, 4] = ((skew + 3.0) / 6.0).clamp(0.0, 1.0)
            feats[:, 5] = ((kurt + 2.0) / 12.0).clamp(0.0, 1.0)
        except Exception:
            feats[:, 4] = 0.5
            feats[:, 5] = 0.5

        # ── [6] high_freq_ratio (fraction of |noise| > threshold) ──────────
        feats[:, 6] = (noise.abs() > 0.02).float().mean(dim=[1, 2, 3]).clamp(0.0, 1.0)

        # ── [7] compression_quality_norm — 0 in GPU fast path ──────────────
        feats[:, 7] = 0.0

        # ── [8..14] per-channel std of noise patches (7 values) ────────────
        try:
            for c in range(3):
                ch_std = noise[:, c].std(dim=[1, 2])           # (B,)
                feats[:, 8 + c] = ch_std.clamp(0.0, 1.0)
            # Pad to 7 values with cross-channel ratio proxies
            feats[:, 11] = (feats[:, 8] - feats[:, 9]).abs()
            feats[:, 12] = (feats[:, 8] - feats[:, 10]).abs()
            feats[:, 13] = (feats[:, 9] - feats[:, 10]).abs()
            feats[:, 14] = feats[:, 8:11].std(dim=1)
        except Exception:
            pass

        # ── [15] spatial consistency (inter-patch correlation proxy) ────────
        try:
            gray_n = noise.mean(dim=1)                         # (B,H,W)
            # Use two non-overlapping halves
            h = gray_n.shape[1]
            half = h // 2
            a = gray_n[:, :half].reshape(B, -1)
            b = gray_n[:, half:half*2].reshape(B, -1)
            # Batch pearson correlation
            a = a - a.mean(dim=1, keepdim=True)
            b = b - b.mean(dim=1, keepdim=True)
            num = (a * b).sum(dim=1)
            den = (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-10)
            corr = (num / den).clamp(-1.0, 1.0)
            feats[:, 15] = ((corr + 1.0) / 2.0).clamp(0.0, 1.0)
        except Exception:
            feats[:, 15] = 0.5

        # ── [16..19] PRNU energy in 4 freq bands ────────────────────────────
        try:
            gray_noise = noise.mean(dim=1)                     # (B,H,W)
            F_n = torch.fft.fft2(gray_noise)
            F_n = torch.fft.fftshift(F_n, dim=[-2, -1])
            power = F_n.abs().pow(2)                           # (B,H,W)
            h, w  = power.shape[1], power.shape[2]
            cy, cx = h // 2, w // 2
            fy = (torch.arange(h, device=self.device, dtype=torch.float32) - cy) / (cy + 1e-8)
            fx = (torch.arange(w, device=self.device, dtype=torch.float32) - cx) / (cx + 1e-8)
            FX, FY = torch.meshgrid(fx, fy, indexing='xy')
            dist   = (FX.pow(2) + FY.pow(2)).sqrt()           # (H,W)

            total_e = power.sum(dim=[1, 2]).clamp(min=1e-12)  # (B,)
            cuts    = [0.0, 0.1, 0.3, 0.6, 1e9]
            for i in range(4):
                mask = ((dist >= cuts[i]) & (dist < cuts[i + 1])).unsqueeze(0)  # (1,H,W)
                band_e = (power * mask).sum(dim=[1, 2])
                feats[:, 16 + i] = (band_e / total_e).clamp(0.0, 1.0)
        except Exception:
            pass

        # ── [20..23] cross-channel PRNU correlations ─────────────────────────
        try:
            pairs = [(0, 1), (0, 2), (1, 2)]
            corrs = []
            for c1, c2 in pairs:
                a = noise[:, c1].reshape(B, -1)
                b = noise[:, c2].reshape(B, -1)
                a = a - a.mean(dim=1, keepdim=True)
                b = b - b.mean(dim=1, keepdim=True)
                sa = a.norm(dim=1).clamp(min=1e-8)
                sb = b.norm(dim=1).clamp(min=1e-8)
                c  = (a * b).sum(dim=1) / (sa * sb)
                corrs.append(((c + 1.0) / 2.0).clamp(0.0, 1.0))
            feats[:, 20] = corrs[0]   # R-G
            feats[:, 21] = corrs[1]   # R-B
            feats[:, 22] = corrs[2]   # G-B
            feats[:, 23] = (corrs[0] + corrs[1] + corrs[2]) / 3.0
        except Exception:
            feats[:, 20:24] = 0.5

        # [24..31] — zeros (recovery stats and double-compression skipped in GPU path)

        # ── [32..35] extended high-freq bands (above VHF) ────────────────────
        try:
            gray_noise = noise.mean(dim=1)
            F_n   = torch.fft.fft2(gray_noise)
            F_n   = torch.fft.fftshift(F_n, dim=[-2, -1])
            power = F_n.abs().pow(2)
            h, w  = power.shape[1], power.shape[2]
            cy, cx = h // 2, w // 2
            fy = (torch.arange(h, device=self.device, dtype=torch.float32) - cy) / (cy + 1e-8)
            fx = (torch.arange(w, device=self.device, dtype=torch.float32) - cx) / (cx + 1e-8)
            FX, FY = torch.meshgrid(fx, fy, indexing='xy')
            dist   = (FX.pow(2) + FY.pow(2)).sqrt()
            total_e = power.sum(dim=[1, 2]).clamp(min=1e-12)
            cuts = [0.6, 0.7, 0.8, 0.9, 1e9]
            for i in range(4):
                mask = ((dist >= cuts[i]) & (dist < cuts[i + 1])).unsqueeze(0)
                feats[:, 32 + i] = ((power * mask).sum(dim=[1, 2]) / total_e).clamp(0.0, 1.0)
        except Exception:
            pass

        # ── [36..39] per-channel noise RMS (R, G, B, Lum) ───────────────────
        try:
            for c in range(3):
                feats[:, 36 + c] = noise[:, c].pow(2).mean(dim=[1, 2]).sqrt().clamp(0.0, 1.0)
            feats[:, 39] = noise.mean(dim=1).pow(2).mean(dim=[1, 2]).sqrt().clamp(0.0, 1.0)
        except Exception:
            pass

        # ── [40..43] noise anisotropy ─────────────────────────────────────────
        try:
            gray_n = noise.mean(dim=1)                         # (B,H,W)
            row_var = gray_n.var(dim=2).mean(dim=1)            # variance along columns → row pattern
            col_var = gray_n.var(dim=1).mean(dim=1)            # variance along rows → col pattern
            total_var = gray_n.var(dim=[1, 2]).clamp(min=1e-12)
            feats[:, 40] = (row_var / total_var).clamp(0.0, 2.0) / 2.0  # row_corr proxy
            feats[:, 41] = (col_var / total_var).clamp(0.0, 2.0) / 2.0  # col_corr proxy
            feats[:, 42] = ((row_var - col_var).abs() / (row_var + col_var + 1e-10)).clamp(0.0, 1.0)
            # Stripe ratio: mean of row means vs col means std
            row_means = gray_n.mean(dim=2)   # (B,H)
            col_means = gray_n.mean(dim=1)   # (B,W)
            feats[:, 43] = (row_means.std(dim=1) / (col_means.std(dim=1) + 1e-10)).clamp(0.0, 2.0) / 2.0
        except Exception:
            pass

        # ── [44..47] Bayer CFA residual proxy ────────────────────────────────
        try:
            gray_n = noise.mean(dim=1)
            # Period-2 energy: difference of even/odd rows
            even = gray_n[:, 0::2, :]
            odd  = gray_n[:, 1::2, :]
            min_h = min(even.shape[1], odd.shape[1])
            period2 = (even[:, :min_h] - odd[:, :min_h]).pow(2).mean(dim=[1, 2])
            total   = gray_n.pow(2).mean(dim=[1, 2]).clamp(min=1e-12)
            feats[:, 44] = (period2 / total).clamp(0.0, 1.0)
            # Period-4: every 4th row difference
            r0 = gray_n[:, 0::4, :]
            r2 = gray_n[:, 2::4, :]
            min_h2 = min(r0.shape[1], r2.shape[1])
            period4 = (r0[:, :min_h2] - r2[:, :min_h2]).pow(2).mean(dim=[1, 2])
            feats[:, 45] = (period4 / total).clamp(0.0, 1.0)
            feats[:, 46] = ((period2 + period4) / (2 * total)).clamp(0.0, 1.0)
            # [47] — zero (demosaic artifact, skipped in GPU fast path)
        except Exception:
            pass

        # ── [48..51] phase coherence ──────────────────────────────────────────
        try:
            gray_n = noise.mean(dim=1)
            F_n    = torch.fft.fft2(gray_n)
            phase  = torch.angle(F_n)              # (B,H,W) in [-π,π]
            h, w   = phase.shape[1], phase.shape[2]
            # Phase diff between adjacent pixels — coherent noise → small diff
            ph_diff_h = (phase[:, :, 1:] - phase[:, :, :-1]).abs()
            ph_diff_v = (phase[:, 1:, :] - phase[:, :-1, :]).abs()
            ph_diff   = torch.cat([ph_diff_h.reshape(B, -1), ph_diff_v.reshape(B, -1)], dim=1)
            ph_mean   = ph_diff.mean(dim=1)
            # Split into 3 freq bands by pixel index
            n_total = ph_diff.shape[1]
            low_ph  = ph_diff[:, :n_total // 3].mean(dim=1)
            mid_ph  = ph_diff[:, n_total // 3: 2 * n_total // 3].mean(dim=1)
            hi_ph   = ph_diff[:, 2 * n_total // 3:].mean(dim=1)
            # Coherence: 1 - normalized mean phase diff (lower diff = more coherent)
            norm = torch.tensor(3.14159, device=self.device)
            feats[:, 48] = (1.0 - low_ph / norm).clamp(0.0, 1.0)
            feats[:, 49] = (1.0 - mid_ph / norm).clamp(0.0, 1.0)
            feats[:, 50] = (1.0 - hi_ph  / norm).clamp(0.0, 1.0)
            feats[:, 51] = (1.0 - ph_mean / norm).clamp(0.0, 1.0)
        except Exception:
            pass

        # ── [52..55] directional correlation (H, V, diag, anti-diag) ─────────
        try:
            gray_n = noise.mean(dim=1)       # (B,H,W)
            flat   = gray_n.reshape(B, -1)
            flat   = flat - flat.mean(dim=1, keepdim=True)
            n      = flat.shape[1]

            def _corr(a, b):
                a = a - a.mean(dim=1, keepdim=True)
                b = b - b.mean(dim=1, keepdim=True)
                return ((a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-10)).clamp(-1, 1)

            # Horizontal: row[i] vs row[i+1]
            rows = gray_n.reshape(B, gray_n.shape[1], -1)
            h_corr = _corr(rows[:, :-1].reshape(B, -1), rows[:, 1:].reshape(B, -1))
            feats[:, 52] = ((h_corr + 1.0) / 2.0).clamp(0.0, 1.0)
            # Vertical: col[j] vs col[j+1]
            cols = gray_n.permute(0, 2, 1).reshape(B, gray_n.shape[2], -1)
            v_corr = _corr(cols[:, :-1].reshape(B, -1), cols[:, 1:].reshape(B, -1))
            feats[:, 53] = ((v_corr + 1.0) / 2.0).clamp(0.0, 1.0)
            # Diagonal: pixel vs pixel diag
            d1 = gray_n[:, :-1, :-1].reshape(B, -1)
            d2 = gray_n[:, 1:,  1:].reshape(B, -1)
            feats[:, 54] = ((_corr(d1, d2) + 1.0) / 2.0).clamp(0.0, 1.0)
            # Anti-diagonal
            ad1 = gray_n[:, :-1, 1:].reshape(B, -1)
            ad2 = gray_n[:, 1:, :-1].reshape(B, -1)
            feats[:, 55] = ((_corr(ad1, ad2) + 1.0) / 2.0).clamp(0.0, 1.0)
        except Exception:
            pass

        # ── [56..59] saturation features ─────────────────────────────────────
        try:
            clip_frac   = ((img > 0.98) | (img < 0.02)).float().mean(dim=[1, 2, 3])
            dyn_range   = img.amax(dim=[1, 2, 3]) - img.amin(dim=[1, 2, 3])
            snr = (img.mean(dim=[1, 2, 3]) / (noise.std(dim=[1, 2, 3]).clamp(min=1e-10)))
            feats[:, 56] = clip_frac.clamp(0.0, 1.0)
            feats[:, 57] = dyn_range.clamp(0.0, 1.0)
            feats[:, 58] = 0.0  # tonal entropy — skipped in GPU fast path
            feats[:, 59] = (snr / 100.0).clamp(0.0, 1.0)
        except Exception:
            pass

        # ── [60..63] multi-scale PRNU consistency ────────────────────────────
        try:
            gray_n = noise.mean(dim=1, keepdim=True)   # (B,1,H,W)
            s64  = F.adaptive_avg_pool2d(gray_n, 64).reshape(B, -1)
            s128 = F.adaptive_avg_pool2d(gray_n, 128).reshape(B, -1)
            # Downsample s128 to 64 for comparison
            s128_64 = F.adaptive_avg_pool2d(
                F.adaptive_avg_pool2d(gray_n, 128), 64
            ).reshape(B, -1)

            def _ms_corr(a, b):
                a = a - a.mean(dim=1, keepdim=True)
                b = b - b.mean(dim=1, keepdim=True)
                return ((a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-10)).clamp(-1, 1)

            c1 = (_ms_corr(s64, s128_64) + 1.0) / 2.0
            # 128 vs 256
            s256 = F.adaptive_avg_pool2d(gray_n, 256).reshape(B, -1)
            s256_128 = F.adaptive_avg_pool2d(
                F.adaptive_avg_pool2d(gray_n, 256), 128
            ).reshape(B, -1)
            c2 = (_ms_corr(s128, s256_128) + 1.0) / 2.0
            # 64 vs 256
            s256_64 = F.adaptive_avg_pool2d(
                F.adaptive_avg_pool2d(gray_n, 256), 64
            ).reshape(B, -1)
            c3 = (_ms_corr(s64, s256_64) + 1.0) / 2.0

            feats[:, 60] = c1.clamp(0.0, 1.0)
            feats[:, 61] = c2.clamp(0.0, 1.0)
            feats[:, 62] = c3.clamp(0.0, 1.0)
            feats[:, 63] = ((c1 + c2 + c3) / 3.0).clamp(0.0, 1.0)
        except Exception:
            pass

        return feats

    def extract_both(self, img: torch.Tensor):
        """
        Extract both PRNU map and scalar features in a single pass.

        Args:
            img : (B,3,H,W) float32 [0,1]

        Returns:
            (prnu_feats (B,64), prnu_map (B,3,map_size,map_size))
        """
        img   = img.to(self.device)

        # Use extract_map so normalisation to [-1,1] is always applied
        noise_map = self.extract_map(img)                  # (B,3,map_size,map_size) [-1,1]

        # Scalar features from the full-resolution noise
        feats = self.extract_features(img)

        return feats, noise_map


# ---------------------------------------------------------------------------
#  Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing PRNUExtractorGPU on {dev}...")
    ext = PRNUExtractorGPU(dev)
    img = torch.rand(2, 3, 512, 512, device=dev)
    feats, nmap = ext.extract_both(img)
    print(f"  feats: {feats.shape}  map: {nmap.shape}")  # (2,64)  (2,3,128,128)
    assert feats.shape == (2, 64), f"Expected (2,64), got {feats.shape}"
    assert nmap.shape  == (2, 3, 128, 128), f"Expected (2,3,128,128), got {nmap.shape}"
    print("  PRNUExtractorGPU sanity check PASSED ✓")
