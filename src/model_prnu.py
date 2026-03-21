"""
model_prnu.py — AI-Detector Model Library v6.1

Major enhancements over v5:

  DeepFusionNet  (image detection) — v6.1
  ─────────────────────────────────────────
  Twelve-branch architecture:
    1.  EfficientNet-B3 vision backbone         (1536-dim)  [60% frozen]
    2.  SRMBranch  — 5 fixed forensic filters   (64-dim)
    3.  FrequencyBranch — multi-scale FFT        (128-dim)
    4.  ColorForensicsBranch — YCbCr anomalies   (64-dim)
    5.  PRNUBranchV2 — 64-dim PRNU + LayerNorm   (96-dim)
    6.  SpatialCNNBranch — spatial texture CNN   (128-dim)
    7.  HallucinationBranch — MobileNetV3-Small  (128-dim)
    8.  PRNUSpatialBranch — 128×128 PRNU map CNN (128-dim)
    9.  GANDiffusionFingerprintBranch            (128-dim)  ← NEW v6
    10. CMOSCCDSensorBranch                      (96-dim)   ← NEW v6
    11. ColorChannelInconsistencyBranch          (96-dim)   ← NEW v6
    12. OpticalFlowIrregularityBranch            (96-dim)   ← NEW v6
  BranchGate (SE-style per-branch importance weighting)     ← NEW v6.1
  CrossAttentionFusion(d_model=192, n_heads=4, n_layers=3,
                       pre-norm, learnable pos_emb) → 768-dim ← NEW v6.1
  Classifier: 768→384→64→1
  Aux heads (training only): per branch.

  Training: forward(img, prnu_feats, prnu_map) returns
            (logit, *aux_logits) in train mode, just logit in eval mode.

  VRAM estimate at batch=4, 512×512, AMP FP16: ~3.4 GB (+1.5 MB for v6.1 additions)

  VideoTemporalFusionNet  (video / 4K detection)  — v4 (unchanged)

  Backward-compat aliases:
    EfficientFusionNet = DeepFusionNet
    PRNUFusionNet      = DeepFusionNet
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)

_log = logging.getLogger(__name__)


def _safe_branch(branch: nn.Module, *args, out_dim: int,
                 B: int, device, dtype, **kwargs) -> torch.Tensor:
    """
    Call ``branch(*args, **kwargs)`` and return its output.
    On ANY exception (shape mismatch, NaN, OOM, …) logs a warning and
    returns a zero tensor of shape (B, out_dim) so training continues.
    """
    try:
        out = branch(*args, **kwargs)
        if torch.isnan(out).any() or torch.isinf(out).any():
            _log.warning("[_safe_branch] %s produced NaN/Inf — zeroing",
                         branch.__class__.__name__)
            return torch.zeros(B, out_dim, device=device, dtype=dtype)
        return out
    except Exception as exc:
        _log.warning("[_safe_branch] %s failed (%s: %s) — zeroing output",
                     branch.__class__.__name__, type(exc).__name__, exc)
        return torch.zeros(B, out_dim, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
#  SRM Branch — Steganalysis Rich Model (5 fixed high-pass filters → 64-dim)
# ---------------------------------------------------------------------------

class SRMBranch(nn.Module):
    """
    Apply 5 fixed SRM high-pass filters then learn features.
    Input:  (B, 3, H, W)
    Output: (B, out_features)
    """

    _SRM_KERNELS = torch.tensor([
        [[ 0,  0, -1,  0,  0],
         [ 0, -1,  2, -1,  0],
         [-1,  2,  0,  2, -1],
         [ 0, -1,  2, -1,  0],
         [ 0,  0, -1,  0,  0]],
        [[ 0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0],
         [-1,  2, -2,  2, -1],
         [ 0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0]],
        [[ 0,  0, -1,  0,  0],
         [ 0,  0,  2,  0,  0],
         [ 0,  0, -2,  0,  0],
         [ 0,  0,  2,  0,  0],
         [ 0,  0, -1,  0,  0]],
        [[-1,  0,  0,  0,  0],
         [ 0, -1,  0,  0,  0],
         [ 0,  0,  4,  0,  0],
         [ 0,  0,  0, -1,  0],
         [ 0,  0,  0,  0, -1]],
        [[-1, -1, -1, -1, -1],
         [-1,  2,  2,  2, -1],
         [-1,  2,  8,  2, -1],
         [-1,  2,  2,  2, -1],
         [-1, -1, -1, -1, -1]],
    ], dtype=torch.float32) / 12.0

    def __init__(self, out_features: int = 64):
        super().__init__()
        kernels = self._SRM_KERNELS.unsqueeze(1).repeat(3, 1, 1, 1)   # (15, 1, 5, 5)
        self.srm_conv = nn.Conv2d(3, 15, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.srm_conv.weight.copy_(kernels.expand(-1, 3, -1, -1).contiguous())
        for p in self.srm_conv.parameters():
            p.requires_grad = False
        self.learn = nn.Sequential(
            nn.Conv2d(15, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, out_features), nn.SiLU())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4:
                raise ValueError(f"SRMBranch expects 4-D (B,C,H,W), got {img.shape}")
            residual = torch.clamp(self.srm_conv(img), -3.0, 3.0)
            out = self.head(self.learn(residual))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in SRMBranch output")
            return out
        except Exception as exc:
            _log.warning("[SRMBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, self.head[-1].out_features
                               if hasattr(self.head[-1], 'out_features') else 64,
                               device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  Color Forensics Branch — YCbCr chroma noise analysis
# ---------------------------------------------------------------------------

class ColorForensicsBranch(nn.Module):
    """
    Detect AI-generation artefacts in YCbCr chroma channels.
    Input:  (B, 3, H, W) — RGB ImageNet-normalised
    Output: (B, out_features)
    """

    _RGB_TO_YCBCR = torch.tensor([
        [ 0.299,    0.587,    0.114  ],
        [-0.16875, -0.33126,  0.5    ],
        [ 0.5,     -0.41869, -0.08131],
    ], dtype=torch.float32)

    def __init__(self, out_features: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, out_features), nn.SiLU())
        self.register_buffer('rgb_to_ycbcr', self._RGB_TO_YCBCR)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4 or img.shape[1] != 3:
                raise ValueError(f"ColorForensicsBranch expects (B,3,H,W), got {img.shape}")
            mean  = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
            std   = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
            rgb   = img * std + mean
            ycbcr = torch.einsum('ck,bkhw->bchw', self.rgb_to_ycbcr, rgb)
            chroma = ycbcr[:, 1:, :, :]
            out   = self.head(self.conv(chroma))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in ColorForensicsBranch output")
            return out
        except Exception as exc:
            _log.warning("[ColorForensicsBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 64, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  Frequency Branch — multi-scale FFT analysis
# ---------------------------------------------------------------------------

class FrequencyBranch(nn.Module):
    """
    Multi-scale frequency-domain artefact detector (full, 1/2, 1/4 scale).
    Input:  (B, 3, H, W)
    Output: (B, out_features)
    """
    _FREQ_SIZE = 64

    def __init__(self, out_features: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, out_features), nn.SiLU())

    def _fft_map(self, x: torch.Tensor, size: int) -> torch.Tensor:
        try:
            x_r = F.interpolate(x.unsqueeze(1), size=(size, size),
                                mode='bilinear', align_corners=False).squeeze(1)
            fft = torch.fft.fftshift(torch.fft.fft2(x_r), dim=(-2, -1))
            mag = torch.log1p(torch.abs(fft))
            b   = mag.shape[0]
            lo  = mag.view(b, -1).min(dim=1)[0].view(b, 1, 1)
            hi  = mag.view(b, -1).max(dim=1)[0].view(b, 1, 1)
            return (mag - lo) / (hi - lo + 1e-8)
        except Exception as exc:
            _log.warning("[FrequencyBranch._fft_map] %s — returning zeros", exc)
            B = x.shape[0] if x.dim() >= 1 else 1
            return torch.zeros(B, size, size, device=x.device, dtype=x.dtype)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4:
                raise ValueError(f"FrequencyBranch expects (B,C,H,W), got {img.shape}")
            gray = img.mean(dim=1)
            s    = self._FREQ_SIZE
            f1   = self._fft_map(gray, s)
            f2   = F.interpolate(self._fft_map(gray, s // 2).unsqueeze(1),
                                 size=(s, s), mode='bilinear', align_corners=False).squeeze(1)
            f3   = F.interpolate(self._fft_map(gray, s // 4).unsqueeze(1),
                                 size=(s, s), mode='bilinear', align_corners=False).squeeze(1)
            out  = self.head(self.conv(torch.stack([f1, f2, f3], dim=1)))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in FrequencyBranch output")
            return out
        except Exception as exc:
            _log.warning("[FrequencyBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 128, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  PRNU Branch v2 — 32-dim PRNU features with LayerNorm (NEW in v5)
# ---------------------------------------------------------------------------

class PRNUBranchV2(nn.Module):
    """
    Enhanced PRNU MLP with LayerNorm to prevent scale domination.

    LayerNorm normalises the 32-dim PRNU vector before the MLP, preventing
    cases where high absolute PRNU values overwhelm the gradient signal.

    Input:  (B, in_features)   — default 32-dim PRNU feature vector
    Output: (B, out_features)  — default 96-dim embedding
    """

    def __init__(self, in_features: int = 32, out_features: int = 96):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(256, 128),         nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(128, out_features), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 2:
                raise ValueError(f"PRNUBranchV2 expects (B, in_features), got {x.shape}")
            out = self.mlp(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in PRNUBranchV2 output")
            return out
        except Exception as exc:
            _log.warning("[PRNUBranchV2] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = x.shape[0] if x.dim() >= 1 else 1
            return torch.zeros(B, 96, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
#  Spatial CNN Branch — dedicated spatial texture detector (NEW in v5)
# ---------------------------------------------------------------------------

class SpatialCNNBranch(nn.Module):
    """
    Dedicated spatial-texture CNN for detecting JPEG block artefacts and
    GAN periodic patterns.

    Input:  (B, 3, H, W)
    Output: (B, out_features)

    Architecture:
      Conv2d(3→32, 3×3) + BN + GELU + MaxPool
      Conv2d(32→64, 3×3) + BN + GELU + MaxPool
      Conv2d(64→128, 5×5) + BN + GELU         ← 5×5 covers 8×8 JPEG blocks
      Conv2d(128→64, 3×3, dilation=2) + BN + GELU  ← dilation catches GAN periodics
      AdaptiveAvgPool2d(4,4) → Flatten → Linear(1024→128) → GELU
    """

    def __init__(self, out_features: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,  128, 5, padding=2, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_features), nn.GELU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4:
                raise ValueError(f"SpatialCNNBranch expects (B,C,H,W), got {img.shape}")
            out = self.head(self.conv(img))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in SpatialCNNBranch output")
            return out
        except Exception as exc:
            _log.warning("[SpatialCNNBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 128, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  Hallucination Branch — MobileNetV3-Small (NEW in v5)
# ---------------------------------------------------------------------------

class HallucinationBranch(nn.Module):
    """
    Detect AI hallucination artefacts using MobileNetV3-Small with a spatial
    attention gate to focus on inconsistent regions.

    MobileNetV3-Small (2.5M params) avoids OOM — EfficientNet-B3 already used.

    Input:  (B, 3, H, W) — ImageNet-normalised
    Output: (B, out_features)
    """

    _FREEZE_RATIO = 0.85

    def __init__(self, out_features: int = 128):
        super().__init__()
        net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Freeze first 85% of feature layers
        children     = list(net.features.children())
        freeze_until = int(len(children) * self._FREEZE_RATIO)
        for i, block in enumerate(children):
            if i < freeze_until:
                for p in block.parameters():
                    p.requires_grad = False

        self.backbone = net.features   # output: (B, 576, H/32, W/32)

        # Spatial attention gate: amplify informative spatial regions
        self.attention_gate = nn.Sequential(
            nn.Conv2d(576, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Head: 576*2*2 = 2304
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576 * 2 * 2, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, out_features), nn.GELU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4 or img.shape[1] != 3:
                raise ValueError(f"HallucinationBranch expects (B,3,H,W), got {img.shape}")
            feat   = self.backbone(img)
            attn   = self.attention_gate(feat)
            feat   = feat * attn
            pooled = self.pool(feat)
            out    = self.head(pooled)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in HallucinationBranch output")
            return out
        except Exception as exc:
            _log.warning("[HallucinationBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 128, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  PRNU Spatial Branch — 64×64 noise map CNN (NEW in v5)
# ---------------------------------------------------------------------------

class PRNUSpatialBranch(nn.Module):
    """
    CNN that learns structured camera fingerprints from the raw 64×64 PRNU
    noise map: column noise, hot pixels, readout patterns — all absent from
    AI-generated images.

    Dilation in the last conv catches GAN periodic artifacts (typically at
    8-px or 16-px grid frequency).

    Input:  (B, 3, 64, 64)  — PRNU noise map, values in [-1, 1]
    Output: (B, out_features)
    """

    def __init__(self, out_features: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            # Stage 1: 64×64 → 64×64 → 32×32
            nn.Conv2d(3,   32,  3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.SiLU(),
            nn.Conv2d(32,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.SiLU(),
            nn.MaxPool2d(2, 2),                                          # 32×32
            # Stage 2: 32×32 → 16×16
            nn.Conv2d(64,  128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.MaxPool2d(2, 2),                                          # 16×16
            # Stage 3: dilated conv to catch GAN periodic artifacts
            nn.Conv2d(128, 128, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
            # Global pooling to 4×4
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # 128 * 4 * 4 = 2048
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_features), nn.SiLU(),
        )

    def forward(self, prnu_map: torch.Tensor) -> torch.Tensor:
        try:
            if prnu_map is None:
                raise ValueError("prnu_map is None")
            if prnu_map.dim() != 4 or prnu_map.shape[1] != 3:
                raise ValueError(f"PRNUSpatialBranch expects (B,3,H,W), got {prnu_map.shape}")
            out = self.head(self.conv(prnu_map))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in PRNUSpatialBranch output")
            return out
        except Exception as exc:
            _log.warning("[PRNUSpatialBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = prnu_map.shape[0] if (prnu_map is not None and prnu_map.dim() >= 1) else 1
            dev = prnu_map.device if prnu_map is not None else torch.device('cpu')
            dtype = prnu_map.dtype if prnu_map is not None else torch.float32
            return torch.zeros(B, 128, device=dev, dtype=dtype)


# ---------------------------------------------------------------------------
#  GAN / Diffusion Fingerprint Branch  (NEW in v6)
# ---------------------------------------------------------------------------

class GANDiffusionFingerprintBranch(nn.Module):
    """
    Detect GAN grid-frequency peaks and Diffusion model denoising artifacts.

    Strategy:
      - Compute per-channel 2-D FFT magnitude spectrum (log-scaled).
      - Stack 3-channel log-spectrum → small CNN picks up radial spectral
        anomalies (GAN checkerboard at Nyquist) and diffusion ringing bands.
      - A secondary path applies Laplacian-of-Gaussian to catch diffusion
        step-edge smearing; features are concatenated before the head.

    Input:  (B, 3, H, W) — ImageNet-normalised
    Output: (B, out_features)
    """

    _SPEC_SIZE = 64   # resize spectrum to this square

    def __init__(self, out_features: int = 128):
        super().__init__()
        # Spectral path
        self.spec_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.spec_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.GELU(),
        )
        # LoG (Laplacian-of-Gaussian) residual path — catches diffusion smear
        self.log_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.log_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.GELU(),
        )
        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(128 + 64, out_features), nn.GELU(), nn.Dropout(0.2),
        )

        # Fixed LoG kernel (5×5, σ=1) applied per channel via grouped conv
        log_k = self._make_log_kernel(size=5, sigma=1.0)
        log_weight = log_k.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # (3,1,5,5)
        self.register_buffer('_log_kernel', log_weight)

    @staticmethod
    def _make_log_kernel(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """Create a Laplacian-of-Gaussian kernel."""
        ax = torch.arange(size, dtype=torch.float32) - size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        r2 = xx ** 2 + yy ** 2
        s2 = sigma ** 2
        k  = -(1.0 / (torch.pi * s2 ** 2)) * (1.0 - r2 / (2 * s2)) * torch.exp(-r2 / (2 * s2))
        k  = k - k.mean()
        return k

    def _log_response(self, img: torch.Tensor) -> torch.Tensor:
        """Apply LoG filter channel-wise (grouped conv)."""
        return F.conv2d(img, self._log_kernel, padding=2, groups=3)

    def _fft_spectrum(self, img: torch.Tensor) -> torch.Tensor:
        """Return log-magnitude FFT spectrum resized to (B,3,S,S)."""
        try:
            B, C, H, W = img.shape
            if B == 0:
                raise ValueError("Empty batch")
            S = self._SPEC_SIZE
            specs = []
            for c in range(C):
                ch  = img[:, c, :, :]
                fft = torch.fft.fftshift(
                    torch.fft.fft2(ch, s=(S, S)), dim=(-2, -1)
                )
                mag = torch.log1p(torch.abs(fft))
                mn  = mag.flatten(1).min(1)[0].view(B, 1, 1)
                mx  = mag.flatten(1).max(1)[0].view(B, 1, 1)
                specs.append((mag - mn) / (mx - mn + 1e-8))
            return torch.stack(specs, dim=1)
        except Exception as exc:
            _log.warning("[GANDiffusionBranch._fft_spectrum] %s — returning zeros", exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 3, self._SPEC_SIZE, self._SPEC_SIZE,
                               device=img.device, dtype=img.dtype)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4 or img.shape[1] != 3:
                raise ValueError(f"GANDiffusionBranch expects (B,3,H,W), got {img.shape}")
            spec   = self._fft_spectrum(img)
            spec_f = self.spec_fc(self.spec_conv(spec))
            log_r  = self._log_response(img)
            log_f  = self.log_fc(self.log_conv(log_r))
            out    = self.head(torch.cat([spec_f, log_f], dim=1))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in GANDiffusionBranch output")
            return out
        except Exception as exc:
            _log.warning("[GANDiffusionFingerprintBranch] %s — returning zeros. (%s)",
                         type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 128, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  CMOS / CCD Sensor Noise Branch  (NEW in v6)
# ---------------------------------------------------------------------------

class CMOSCCDSensorBranch(nn.Module):
    """
    Detect the absence of real camera sensor fingerprints.

    Real CMOS/CCD sensors produce:
      - Fixed-Pattern Noise (FPN): column-correlated dark-current variations.
      - Photo-Response Non-Uniformity (PRNU): pixel-level gain differences.
      - Row/column readout noise: periodic stripes at multiples of readout freq.

    AI generators lack all of these; this branch is trained to detect their
    absence as an indicator of synthetic origin.

    Approach:
      1. High-pass filter the image (SRM-style) to isolate noise residual.
      2. Compute column-mean and row-mean of the noise to expose FPN stripes.
      3. Run a small 1-D CNN along columns and rows separately to characterise
         the stripe pattern, then fuse.

    Input:  (B, 3, H, W) — ImageNet-normalised
    Output: (B, out_features)
    """

    # Simple 3×3 high-pass kernel applied per channel
    _HP_KERNEL = torch.tensor([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ], dtype=torch.float32) / 8.0

    def __init__(self, out_features: int = 96):
        super().__init__()
        hp_w = self._HP_KERNEL.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # (3,1,3,3)
        self.register_buffer('_hp_kernel', hp_w)

        # Column-profile CNN: input shape is (B, 3, W) treated as 1-D
        self.col_cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.col_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 64), nn.GELU(),
        )

        # Row-profile CNN
        self.row_cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.row_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 64), nn.GELU(),
        )

        # Spatial variance map CNN
        self.var_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.var_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 64, out_features), nn.GELU(), nn.Dropout(0.2),
        )

    def _high_pass(self, img: torch.Tensor) -> torch.Tensor:
        """Apply per-channel high-pass filter (grouped conv)."""
        return F.conv2d(img, self._hp_kernel, padding=1, groups=3)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4 or img.shape[1] != 3:
                raise ValueError(f"CMOSCCDSensorBranch expects (B,3,H,W), got {img.shape}")
            noise = self._high_pass(img)
            col_profile = noise.mean(dim=2)
            col_f = self.col_fc(self.col_cnn(col_profile))
            row_profile = noise.mean(dim=3)
            row_f = self.row_fc(self.row_cnn(row_profile))
            var_map = noise.abs()
            var_f = self.var_fc(self.var_conv(var_map))
            out = self.head(torch.cat([col_f, row_f, var_f], dim=1))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in CMOSCCDSensorBranch output")
            return out
        except Exception as exc:
            _log.warning("[CMOSCCDSensorBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 96, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  Color Channel Inconsistency Branch  (NEW in v6)
# ---------------------------------------------------------------------------

class ColorChannelInconsistencyBranch(nn.Module):
    """
    Detect inter-channel inconsistencies introduced by AI generators.

    Real cameras apply a Bayer CFA and demosaic algorithmically, creating
    predictable spatial correlations between R, G, B channels.  GAN and
    diffusion models synthesise each channel semi-independently, leaving
    subtle phase and spectral mis-alignments that this branch learns to spot.

    Approach:
      1. Compute pairwise channel difference maps: R-G, R-B, G-B.  (3 maps)
      2. Compute pairwise FFT cross-power spectrum phase maps.       (3 maps)
      3. Stack all 6 maps → (B, 6, H, W) → small CNN.

    Input:  (B, 3, H, W) — ImageNet-normalised
    Output: (B, out_features)
    """

    _SPEC_SIZE = 32

    def __init__(self, out_features: int = 96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, out_features), nn.GELU(), nn.Dropout(0.2),
        )

    def _cross_phase(self, a: torch.Tensor, b: torch.Tensor, S: int) -> torch.Tensor:
        """
        Cross-power spectrum phase between two (B, H, W) maps.
        Returns (B, S, S) phase map in [-π, π], scaled to [-1, 1].
        """
        fa  = torch.fft.fft2(a, s=(S, S))
        fb  = torch.fft.fft2(b, s=(S, S))
        # Cross-power normalised phase
        cp  = fa * fb.conj()
        mag = torch.abs(cp).clamp(min=1e-8)
        phase = torch.angle(cp / mag)          # (B, S, S)
        return phase / torch.pi                # normalise to [-1, 1]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4 or img.shape[1] != 3:
                raise ValueError(f"ColorChannelInconsistencyBranch expects (B,3,H,W), got {img.shape}")
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
            rgb  = img * std + mean

            R, G, B_ch = rgb[:, 0], rgb[:, 1], rgb[:, 2]

            S = self._SPEC_SIZE

            def diff_map(a, b):
                d = (a - b).unsqueeze(1)
                return F.interpolate(d, size=(S, S), mode='bilinear', align_corners=False)

            d_rg = diff_map(R, G)
            d_rb = diff_map(R, B_ch)
            d_gb = diff_map(G, B_ch)
            p_rg = self._cross_phase(R, G, S).unsqueeze(1)
            p_rb = self._cross_phase(R, B_ch, S).unsqueeze(1)
            p_gb = self._cross_phase(G, B_ch, S).unsqueeze(1)

            feat = torch.cat([d_rg, d_rb, d_gb, p_rg, p_rb, p_gb], dim=1)
            out  = self.head(self.conv(feat))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in ColorChannelInconsistencyBranch output")
            return out
        except Exception as exc:
            _log.warning("[ColorChannelInconsistencyBranch] %s — returning zeros. (%s)",
                         type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 96, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  Optical Flow Irregularity Branch  (NEW in v6)
# ---------------------------------------------------------------------------

class OpticalFlowIrregularityBranch(nn.Module):
    """
    Detect spatial warp / hallucination artifacts via pseudo-optical-flow.

    AI generators sometimes produce fine-structure inconsistencies across
    image scale — e.g., a finger that looks natural at full resolution but
    "flows" unnaturally when compared with a half-scale version.

    We approximate inter-scale warp by computing the gradient of the
    difference between the image and its blurred version at multiple scales,
    mimicking what a flow estimator would output between consecutive video
    frames.  This is a purely spatial operation — no external optical-flow
    library required.

    Inputs at 3 scales:
      full    → (B, 3, H,   W  )
      half    → (B, 3, H/2, W/2) upsampled back to H×W
      quarter → (B, 3, H/4, W/4) upsampled back to H×W

    Pseudo-flow fields: full−half, full−quarter, half−quarter  → (B, 9, H, W)
    A small CNN processes these to produce the branch embedding.

    Input:  (B, 3, H, W) — ImageNet-normalised
    Output: (B, out_features)
    """

    def __init__(self, out_features: int = 96):
        super().__init__()
        self.flow_conv = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_features), nn.GELU(), nn.Dropout(0.2),
        )

        # Fixed Gaussian blur kernel (5×5, σ=1.5) for scale-space smoothing
        gb = self._gaussian_kernel(5, 1.5)                     # (5,5)
        gb_w = gb.view(1, 1, 5, 5).repeat(3, 1, 1, 1)         # (3,1,5,5)
        self.register_buffer('_gauss_kernel', gb_w)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        ax   = torch.arange(size, dtype=torch.float32) - size // 2
        x, y = torch.meshgrid(ax, ax, indexing='ij')
        k    = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return k / k.sum()

    def _blur(self, img: torch.Tensor) -> torch.Tensor:
        return F.conv2d(img, self._gauss_kernel, padding=2, groups=3)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            if img.dim() != 4 or img.shape[1] != 3:
                raise ValueError(f"OpticalFlowIrregularityBranch expects (B,3,H,W), got {img.shape}")
            B, C, H, W = img.shape
            half_hw = (max(H // 2, 16), max(W // 2, 16))
            qtr_hw  = (max(H // 4, 8),  max(W // 4, 8))

            full   = self._blur(img)
            half_s = F.interpolate(img, size=half_hw, mode='bilinear', align_corners=False)
            half_s = F.interpolate(self._blur(half_s), size=(H, W),
                                   mode='bilinear', align_corners=False)
            qtr_s  = F.interpolate(img, size=qtr_hw, mode='bilinear', align_corners=False)
            qtr_s  = F.interpolate(self._blur(qtr_s), size=(H, W),
                                   mode='bilinear', align_corners=False)

            flow = torch.clamp(
                torch.cat([full - half_s, full - qtr_s, half_s - qtr_s], dim=1),
                -1.0, 1.0,
            )
            out = self.head(self.flow_conv(flow))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in OpticalFlowIrregularityBranch output")
            return out
        except Exception as exc:
            _log.warning("[OpticalFlowIrregularityBranch] %s — returning zeros. (%s)",
                         type(exc).__name__, exc)
            B = img.shape[0] if img.dim() >= 1 else 1
            return torch.zeros(B, 96, device=img.device, dtype=img.dtype)


# ---------------------------------------------------------------------------
#  Cross-Attention Fusion (upgraded in v5: d_model param, n_layers param)
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion for N branch embeddings.

    Each branch attends to every other branch via a Transformer encoder,
    allowing the model to learn which branches are most informative per image.

    Input:  list of (B, d_i) tensors, one per branch
    Output: (B, out_dim)
    """

    def __init__(self, branch_dims: list, out_dim: int = 256,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, d_model) for d in branch_dims])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # pre-norm: more stable gradients
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                   enable_nested_tensor=False)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, len(branch_dims), d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * len(branch_dims), out_dim),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, branch_outputs: list) -> torch.Tensor:
        try:
            if not branch_outputs:
                raise ValueError("CrossAttentionFusion received empty branch_outputs list")
            tokens   = torch.stack(
                [proj(b) for proj, b in zip(self.projs, branch_outputs)], dim=1
            )
            tokens   = tokens + self.pos_emb
            attended = self.transformer(tokens)
            out      = self.head(attended.flatten(1))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in CrossAttentionFusion output")
            return out
        except Exception as exc:
            _log.warning("[CrossAttentionFusion] %s — returning zeros. (%s)",
                         type(exc).__name__, exc)
            B = branch_outputs[0].shape[0] if branch_outputs else 1
            out_dim = self.head[0].out_features if hasattr(self.head[0], 'out_features') else 768
            dev   = branch_outputs[0].device if branch_outputs else torch.device('cpu')
            dtype = branch_outputs[0].dtype if branch_outputs else torch.float32
            return torch.zeros(B, out_dim, device=dev, dtype=dtype)


# ---------------------------------------------------------------------------
#  BranchGate — SE-style learnable per-branch importance weighting (v6.1)
# ---------------------------------------------------------------------------

class BranchGate(nn.Module):
    """
    Squeeze-and-Excitation style per-branch importance gating.

    Reads a global context (concatenation of all branch outputs) and
    produces one scalar weight per branch in (0, 1).  Branches that
    are irrelevant for a given image are suppressed without being
    zeroed out entirely (sigmoid, not hard mask).

    Parameters chosen so the gate is cheap:
      reduction=16  →  Linear(2688→168) + Linear(168→12)  ≈ 455 K params
    """
    def __init__(self, branch_dims: list, reduction: int = 16):
        super().__init__()
        total  = sum(branch_dims)
        hidden = max(len(branch_dims) * 4, total // reduction)
        self.fc = nn.Sequential(
            nn.Linear(total, hidden), nn.SiLU(),
            nn.Linear(hidden, len(branch_dims)), nn.Sigmoid(),
        )

    def forward(self, branch_outputs: list) -> list:
        try:
            if not branch_outputs:
                raise ValueError("BranchGate received empty list")
            ctx = torch.cat(branch_outputs, dim=-1)
            w   = self.fc(ctx)
            return [b * w[:, i:i+1] for i, b in enumerate(branch_outputs)]
        except Exception as exc:
            _log.warning("[BranchGate] %s — returning unweighted branches. (%s)",
                         type(exc).__name__, exc)
            return branch_outputs   # pass through unchanged on failure


# ---------------------------------------------------------------------------
#  DeepFusionNet v6 — twelve-branch + upgraded cross-attention
# ---------------------------------------------------------------------------

class DeepFusionNet(nn.Module):
    """
    Twelve-branch AI-image detector — v6.

    Branches:
      1.  EfficientNet-B3 vision backbone         → 1536-dim
      2.  SRMBranch (5 forensic filters)          → 64-dim
      3.  FrequencyBranch (multi-scale FFT×3)     → 128-dim
      4.  ColorForensicsBranch (YCbCr chroma)     → 64-dim
      5.  PRNUBranchV2 (32-dim PRNU + LayerNorm)  → 96-dim
      6.  SpatialCNNBranch (spatial texture CNN)  → 128-dim
      7.  HallucinationBranch (MobileNetV3-Small) → 128-dim
      8.  PRNUSpatialBranch (64×64 PRNU map CNN)  → 128-dim
      9.  GANDiffusionFingerprintBranch           → 128-dim  ← NEW
      10. CMOSCCDSensorBranch                     → 96-dim   ← NEW
      11. ColorChannelInconsistencyBranch         → 96-dim   ← NEW
      12. OpticalFlowIrregularityBranch           → 96-dim   ← NEW

    Fusion:
      CrossAttentionFusion(d_model=192, n_heads=4, n_layers=2) → 768-dim
      → Linear(768→384) + SiLU + Dropout(0.4)
      → Linear(384→64)  + SiLU + Dropout(0.3)
      → Linear(64→1)

    Auxiliary heads (training only — all branches get one):
      prnu_aux, halluc_aux, prnu_spatial_aux,
      gan_diff_aux, cmos_aux, color_incon_aux, flow_aux

    forward(img, prnu_feats, prnu_map) returns:
      eval mode  : logit  (B, 1)
      train mode : (logit, prnu_aux, halluc_aux, prnu_spatial_aux,
                    gan_diff_aux, cmos_aux, color_incon_aux, flow_aux)

    Old v5 checkpoints load with strict=False — new branches init fresh.
    """

    _FREEZE_RATIO = 0.60

    def __init__(self, dropout: float = 0.4, prnu_in_features: int = 64,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.use_ckpt = gradient_checkpointing

        # ── Vision backbone ──────────────────────────────────────────────
        net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        feature_children = list(net.features.children())
        freeze_until     = int(len(feature_children) * self._FREEZE_RATIO)
        for i, block in enumerate(feature_children):
            if i < freeze_until:
                for p in block.parameters():
                    p.requires_grad = False
        self.backbone    = net.features
        self.pool        = nn.AdaptiveAvgPool2d((1, 1))
        self._last_block = feature_children[-1]

        # ── Branches (v5 — unchanged) ─────────────────────────────────────
        self.srm_branch          = SRMBranch(out_features=64)
        self.freq_branch         = FrequencyBranch(out_features=128)
        self.color_branch        = ColorForensicsBranch(out_features=64)
        self.prnu_branch_v2      = PRNUBranchV2(in_features=prnu_in_features, out_features=96)
        self.spatial_branch      = SpatialCNNBranch(out_features=128)
        self.halluc_branch       = HallucinationBranch(out_features=128)
        self.prnu_spatial_branch = PRNUSpatialBranch(out_features=128)

        # ── Branches (v6 — NEW) ───────────────────────────────────────────
        self.gan_diff_branch     = GANDiffusionFingerprintBranch(out_features=128)
        self.cmos_branch         = CMOSCCDSensorBranch(out_features=96)
        self.color_incon_branch  = ColorChannelInconsistencyBranch(out_features=96)
        self.flow_branch         = OpticalFlowIrregularityBranch(out_features=96)

        # ── Cross-attention fusion ────────────────────────────────────────
        branch_dims = [
            1536,   # backbone
            64,     # SRM
            128,    # Frequency
            64,     # ColorForensics
            96,     # PRNU v2
            128,    # SpatialCNN
            128,    # Hallucination
            128,    # PRNUSpatial
            128,    # GAN/Diffusion fingerprint  ← NEW
            96,     # CMOS/CCD sensor noise      ← NEW
            96,     # Color Channel Inconsistency← NEW
            96,     # Optical Flow Irregularity  ← NEW
        ]   # 12 branches → total tokens = 12
        self.branch_gate = BranchGate(branch_dims)   # v6.1: SE-style branch weighting
        self.fusion = CrossAttentionFusion(
            branch_dims, out_dim=768, d_model=192, n_heads=4, n_layers=3
        )

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(768, 384), nn.SiLU(), nn.Dropout(p=dropout),
            nn.Linear(384, 64),  nn.SiLU(), nn.Dropout(p=dropout * 0.75),
            nn.Linear(64, 1),
        )

        # ── Auxiliary heads (training only) ──────────────────────────────
        def _aux(in_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))

        self.prnu_aux_head          = _aux(96)
        self.halluc_aux_head        = _aux(128)
        self.prnu_spatial_aux_head  = _aux(128)
        self.gan_diff_aux_head      = _aux(128)   # NEW
        self.cmos_aux_head          = _aux(96)    # NEW
        self.color_incon_aux_head   = _aux(96)    # NEW
        self.flow_aux_head          = _aux(96)    # NEW

        self._init_weights()

    def forward(self, img: torch.Tensor, prnu_feats: torch.Tensor,
                prnu_map: torch.Tensor = None):
        """
        Args:
            img        : (B, 3, H, W) ImageNet-normalised
            prnu_feats : (B, 64) scalar PRNU feature vector
            prnu_map   : (B, 3, 128, 128) spatial PRNU noise map, or None
        """
        B = img.size(0)
        _d, _t = img.device, img.dtype

        def _z(dim): return torch.zeros(B, dim, device=_d, dtype=_t)

        # ── Backbone ─────────────────────────────────────────────────────
        try:
            if self.use_ckpt and self.training:
                cnn_out = grad_checkpoint(
                    lambda x: self.pool(self.backbone(x)).view(x.size(0), -1),
                    img, use_reentrant=False,
                )
            else:
                cnn_out = self.pool(self.backbone(img)).view(B, -1)
            if torch.isnan(cnn_out).any() or torch.isinf(cnn_out).any():
                raise ValueError("NaN/Inf in backbone output")
        except Exception as exc:
            _log.warning("[DeepFusionNet] backbone failed (%s) — zeroing", exc)
            cnn_out = _z(1536)

        # ── Branches — each wrapped independently ────────────────────────
        srm_out          = _safe_branch(self.srm_branch,          img,       out_dim=64,   B=B, device=_d, dtype=_t)
        freq_out         = _safe_branch(self.freq_branch,         img,       out_dim=128,  B=B, device=_d, dtype=_t)
        color_out        = _safe_branch(self.color_branch,        img,       out_dim=64,   B=B, device=_d, dtype=_t)
        prnu_out         = _safe_branch(self.prnu_branch_v2,      prnu_feats,out_dim=96,   B=B, device=_d, dtype=_t)
        spatial_out      = _safe_branch(self.spatial_branch,      img,       out_dim=128,  B=B, device=_d, dtype=_t)
        halluc_out       = _safe_branch(self.halluc_branch,       img,       out_dim=128,  B=B, device=_d, dtype=_t)

        if prnu_map is None:
            prnu_map = torch.zeros(B, 3, 128, 128, device=_d, dtype=_t)
        prnu_spatial_out = _safe_branch(self.prnu_spatial_branch, prnu_map,  out_dim=128,  B=B, device=_d, dtype=_t)

        gan_diff_out     = _safe_branch(self.gan_diff_branch,     img,       out_dim=128,  B=B, device=_d, dtype=_t)
        cmos_out         = _safe_branch(self.cmos_branch,         img,       out_dim=96,   B=B, device=_d, dtype=_t)
        color_incon_out  = _safe_branch(self.color_incon_branch,  img,       out_dim=96,   B=B, device=_d, dtype=_t)
        flow_out         = _safe_branch(self.flow_branch,         img,       out_dim=96,   B=B, device=_d, dtype=_t)

        # ── Gate + Fusion + Classifier ────────────────────────────────────
        try:
            branch_list = [
                cnn_out, srm_out, freq_out, color_out,
                prnu_out, spatial_out, halluc_out, prnu_spatial_out,
                gan_diff_out, cmos_out, color_incon_out, flow_out,
            ]
            branch_list = self.branch_gate(branch_list)
            fused = self.fusion(branch_list)
            logit = self.classifier(fused)
            if torch.isnan(logit).any() or torch.isinf(logit).any():
                raise ValueError("NaN/Inf in classifier output")
        except Exception as exc:
            _log.warning("[DeepFusionNet] fusion/classifier failed (%s) — returning zeros", exc)
            logit = _z(1)

        if self.training:
            def _aux_safe(head, feat, name):
                try:
                    out = head(feat)
                    return out if not (torch.isnan(out).any() or torch.isinf(out).any()) else _z(1)
                except Exception as e:
                    _log.warning("[DeepFusionNet] aux head %s failed (%s)", name, e)
                    return _z(1)

            return (
                logit,
                _aux_safe(self.prnu_aux_head,         prnu_out,          "prnu"),
                _aux_safe(self.halluc_aux_head,        halluc_out,        "halluc"),
                _aux_safe(self.prnu_spatial_aux_head,  prnu_spatial_out,  "prnu_spatial"),
                _aux_safe(self.gan_diff_aux_head,      gan_diff_out,      "gan_diff"),
                _aux_safe(self.cmos_aux_head,          cmos_out,          "cmos"),
                _aux_safe(self.color_incon_aux_head,   color_incon_out,   "color_incon"),
                _aux_safe(self.flow_aux_head,          flow_out,          "flow"),
            )

        return logit

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self._last_block

    def param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(
            "  DeepFusionNet v6.1  (B3 + SRM×15 + FFT×3 + YCbCr + PRNU-64 + SpatialCNN + "
            "MobileNetV3 + PRNUSpatialMap + GAN/Diff + CMOS/CCD + "
            "ColorIncon + OpticalFlow + BranchGate + CrossAttn-192×3 + INT8-ready)"
        )
        print(f"  Total parameters    : {total:,}")
        print(f"  Trainable           : {trainable:,}  ({trainable/total*100:.1f}%)")
        print(f"  Frozen (backbone)   : {frozen:,}  ({frozen/total*100:.1f}%)")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv1d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


# ---------------------------------------------------------------------------
#  Backward-compat aliases
# ---------------------------------------------------------------------------

EfficientFusionNet = DeepFusionNet
PRNUFusionNet      = DeepFusionNet


def check_checkpoint_compat(model: nn.Module, ckpt_path_or_state) -> dict:
    """
    Compare a checkpoint against the current model BEFORE loading it.

    Reports which weights will be:
      ✓ preserved      — checkpoint key exists + shapes match  → training kept
      ~ new (random)   — key in model but not in checkpoint    → new branch, OK
      ✗ shape mismatch — key exists in both but shapes differ  → TRAINING LOST
      ✗ dropped keys   — key in checkpoint but not in model    → TRAINING LOST

    Call this before any load_state_dict(strict=False) to detect accidental
    renames or reshapes that would silently destroy past training.

    Parameters
    ----------
    model              : the nn.Module about to receive the checkpoint
    ckpt_path_or_state : path string  OR  already-loaded state_dict

    Returns
    -------
    dict with keys: preserved, new_random, lost, shape_mismatch
    """
    import os as _os

    # ── Load state dict ───────────────────────────────────────────────────
    if isinstance(ckpt_path_or_state, (_os.PathLike, str)):
        path = str(ckpt_path_or_state)
        if not _os.path.exists(path):
            print("  [compat] No checkpoint found — starting fresh.")
            return {'preserved': [], 'new_random': [], 'lost': [], 'shape_mismatch': []}
        try:
            raw = torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            raw = torch.load(path, map_location='cpu')
        if isinstance(raw, dict):
            ckpt_sd = raw.get('model_state_dict', raw.get('model_state', raw))
        else:
            ckpt_sd = raw
    else:
        ckpt_sd = ckpt_path_or_state  # already a state dict

    model_sd   = model.state_dict()
    ckpt_keys  = set(ckpt_sd.keys())
    model_keys = set(model_sd.keys())

    preserved      = []
    new_random     = []
    lost           = []
    shape_mismatch = []

    for key in sorted(model_keys):
        if key not in ckpt_keys:
            new_random.append(key)
        elif tuple(ckpt_sd[key].shape) != tuple(model_sd[key].shape):
            shape_mismatch.append((
                key,
                tuple(ckpt_sd[key].shape),
                tuple(model_sd[key].shape),
            ))
        else:
            preserved.append(key)

    for key in sorted(ckpt_keys):
        if key not in model_keys:
            lost.append(key)

    # ── Report ────────────────────────────────────────────────────────────
    has_problem = bool(lost or shape_mismatch)
    sep = "═" * 56
    print(f"\n  {sep}")
    print(f"  Checkpoint Compatibility Report")
    print(f"  {sep}")
    print(f"  ✓ Preserved      : {len(preserved):4d} layers  (trained weights load correctly)")
    if new_random:
        print(f"  ~ New (random)   : {len(new_random):4d} layers  (new branches — OK, expected)")
    if shape_mismatch:
        print(f"  ✗ Shape mismatch : {len(shape_mismatch):4d} layers  ← PAST TRAINING LOST FOR THESE")
        for key, cs, ms in shape_mismatch[:5]:
            print(f"      {key}")
            print(f"        saved {cs}  →  model {ms}")
        if len(shape_mismatch) > 5:
            print(f"      … and {len(shape_mismatch) - 5} more")
    if lost:
        print(f"  ✗ Dropped keys   : {len(lost):4d} layers  ← PAST TRAINING LOST FOR THESE")
        for key in lost[:5]:
            print(f"      {key}")
        if len(lost) > 5:
            print(f"      … and {len(lost) - 5} more")
    if has_problem:
        print(f"\n  ⚠  WARNING: architecture changed in a destructive way.")
        print(f"     Rename or reshape a layer → its trained weights are gone.")
        print(f"     Safe changes: ADD new branches (new_random above is fine).")
    else:
        print(f"  ✓ No training lost — safe to load.")
    print(f"  {sep}\n")

    return {
        'preserved':      preserved,
        'new_random':     new_random,
        'lost':           lost,
        'shape_mismatch': shape_mismatch,
    }


def quantize_model_for_inference(model: 'DeepFusionNet') -> nn.Module:
    """
    Apply post-training dynamic INT8 quantization to all nn.Linear layers.

    - Reduces model size ~2–3× on disk
    - Speeds up CPU inference ~1.5–2.5×
    - Zero retraining required
    - GPU inference: no speedup (INT8 linear runs on CPU only)

    Usage:
        detector.model = quantize_model_for_inference(detector.model)
    """
    import torch.ao.quantization as tq
    model.eval()
    return tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


# ---------------------------------------------------------------------------
#  Video sub-modules (v4 — unchanged)
# ---------------------------------------------------------------------------

class PRNUBranch(nn.Module):
    """Legacy MLP for 16-dim PRNU vector (used in VideoTemporalFusionNet)."""
    def __init__(self, in_features: int = 16, out_features: int = 64):
        super().__init__()
        self._out = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128), nn.SiLU(), nn.Dropout(p=0.2),
            nn.Linear(128, out_features), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.mlp(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in PRNUBranch output")
            return out
        except Exception as exc:
            _log.warning("[PRNUBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = x.shape[0] if x.dim() >= 1 else 1
            return torch.zeros(B, self._out, device=x.device, dtype=x.dtype)


class PRNUDeepBranch(nn.Module):
    """MLP for 64-dim per-frame PRNU features → out_features-dim."""
    def __init__(self, in_features: int = 64, out_features: int = 64):
        super().__init__()
        self._out = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),         nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, out_features), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.mlp(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in PRNUDeepBranch output")
            return out
        except Exception as exc:
            _log.warning("[PRNUDeepBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = x.shape[0] if x.dim() >= 1 else 1
            return torch.zeros(B, self._out, device=x.device, dtype=x.dtype)


class PRNUTemporalBranch(nn.Module):
    """MLP for 64-dim inter-frame PRNU drift vector → out_features-dim."""
    def __init__(self, in_features: int = 64, out_features: int = 32):
        super().__init__()
        self._out = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, out_features), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.mlp(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in PRNUTemporalBranch output")
            return out
        except Exception as exc:
            _log.warning("[PRNUTemporalBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = x.shape[0] if x.dim() >= 1 else 1
            return torch.zeros(B, self._out, device=x.device, dtype=x.dtype)


class TemporalFlowBranch(nn.Module):
    def __init__(self, in_channels: int = 6, out_features: int = 64):
        super().__init__()
        self._out = out_features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, out_features), nn.SiLU())

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        try:
            if flow.dim() != 4:
                raise ValueError(f"TemporalFlowBranch expects (B,C,H,W), got {flow.shape}")
            # Auto-adapt channel count: pad with zeros or truncate to match conv weight.
            expected_c = self.conv[0].in_channels
            if flow.shape[1] != expected_c:
                if flow.shape[1] < expected_c:
                    pad = torch.zeros(flow.shape[0], expected_c - flow.shape[1],
                                      flow.shape[2], flow.shape[3],
                                      device=flow.device, dtype=flow.dtype)
                    flow = torch.cat([flow, pad], dim=1)
                else:
                    flow = flow[:, :expected_c]
            out = self.head(self.conv(flow))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in TemporalFlowBranch output")
            return out
        except Exception as exc:
            _log.warning("[TemporalFlowBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = flow.shape[0] if flow.dim() >= 1 else 1
            return torch.zeros(B, self._out, device=flow.device, dtype=flow.dtype)


class MotionTemporalBranch(nn.Module):
    """
    Learns temporal motion signatures from real optical flow features.

    Processes the 48-dim feature vectors produced by MotionAnalyzer
    (src/motion_analyzer.py) for each consecutive frame pair:

      [0-3]   flow magnitude stats     — motion speed distribution
      [4]     motion coverage          — fraction of pixels moving
      [5]     flow coherence           — direction uniformity [0,1]
      [6]     direction entropy        — how chaotic is motion direction
      [7]     spatial smoothness       — low=AI-over-smooth, high=natural shake
      [8]     hf_spatial_ratio         — GAN grid artifacts in flow FFT
      [9]     vibration_score          — camera micro-vibration (AI=near-zero)
      [10]    corner_vs_center         — camera shake signature
      [11]    temporal_accel           — frame-to-frame velocity change
      [12-19] 8-bin direction histogram
      [20-21] horizontal / vertical symmetry
      [22]    GAN periodic artifact score
      [23]    zero-motion fraction
      [24-27] LK corner tracking stats  (std, mean, track_ratio, coherence)
      [28-31] flow gradient sharpness   (gx, gy, lap_fx, lap_fy)
      [32-35] affine decomposition      (tx, ty, curl, div)
      [36-39] multi-scale consistency   (incons, incons_var, cf_ratio, edge_ratio)
      [40-43] motion entropy/complexity (entropy, opposing_frac, dom_freq, smoothness)
      [44-47] frame-flow coupling       (edge_abs, flat_abs, edge_std, bg_frac)

    AI video signatures learned:
      - Near-zero vibration_score (no camera shake)
      - Sudden velocity jumps (high temporal_accel)
      - Elevated GAN grid score (periodic FFT peaks)
      - Over-smooth spatial flow (low spatial_smoothness)

    Input:  (B, T, 48) — T consecutive frame-pair feature vectors
            Accepts (B, 48) and auto-unsqueezes to (B, 1, 48)
    Output: (B, out_features)

    Architecture: LayerNorm → Bidirectional GRU(2 layers) → mean-pool → MLP
    VRAM @ batch=4, T=8: < 50 MB  (lightweight by design for RTX 3050 Laptop)
    """

    MOTION_DIM = 48

    def __init__(self, motion_dim: int = 48, out_features: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(motion_dim)
        # Bidirectional: 2 directions × 64 hidden = 128-dim output per step
        self.gru = nn.GRU(
            input_size  = motion_dim,
            hidden_size = 64,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(128, out_features), nn.GELU(), nn.Dropout(0.2),
        )

    def forward(self, motion_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_feats: (B, T, 48) or (B, 48) if single-pair
        """
        try:
            if motion_feats.dim() == 2:
                motion_feats = motion_feats.unsqueeze(1)   # (B, 1, 48)
            if motion_feats.dim() != 3:
                raise ValueError(f"MotionTemporalBranch expects (B,T,{self.MOTION_DIM}), got {motion_feats.shape}")
            if motion_feats.shape[-1] != self.MOTION_DIM:
                # Pad or truncate to expected dim
                _log.warning("[MotionTemporalBranch] got last-dim %d, expected %d — padding/truncating",
                             motion_feats.shape[-1], self.MOTION_DIM)
                if motion_feats.shape[-1] < self.MOTION_DIM:
                    pad = torch.zeros(*motion_feats.shape[:-1], self.MOTION_DIM - motion_feats.shape[-1],
                                      device=motion_feats.device, dtype=motion_feats.dtype)
                    motion_feats = torch.cat([motion_feats, pad], dim=-1)
                else:
                    motion_feats = motion_feats[..., :self.MOTION_DIM]
            x      = self.norm(motion_feats)               # (B, T, 48)
            out, _ = self.gru(x)                           # (B, T, 128)
            pooled = out.mean(dim=1)                       # (B, 128) — temporal mean-pool
            result = self.head(pooled)                     # (B, out_features)
            if torch.isnan(result).any() or torch.isinf(result).any():
                raise ValueError("NaN/Inf in MotionTemporalBranch output")
            return result
        except Exception as exc:
            _log.warning("[MotionTemporalBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = motion_feats.shape[0] if motion_feats.dim() >= 1 else 1
            out_dim = self.head[0].out_features if hasattr(self.head[0], 'out_features') else 128
            return torch.zeros(B, out_dim, device=motion_feats.device, dtype=motion_feats.dtype)


class AudioBranch(nn.Module):
    def __init__(self, out_features: int = 64):
        super().__init__()
        self._out = out_features
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, out_features), nn.SiLU())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        try:
            if mel.dim() != 4:
                raise ValueError(f"AudioBranch expects (B,1,F,T) mel-spectrogram, got {mel.shape}")
            out = self.head(self.conv(mel))
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in AudioBranch output")
            return out
        except Exception as exc:
            _log.warning("[AudioBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = mel.shape[0] if mel.dim() >= 1 else 1
            return torch.zeros(B, self._out, device=mel.device, dtype=mel.dtype)


class VideoTemporalFusionNet(nn.Module):
    """
    Video AI detector — EfficientNet-B3 backbone, v5.

    Branches:
      1. EfficientNet-B3 vision backbone                     → 1536-dim
      2. TemporalFlowBranch (6-ch pre-computed flow map)     → 64-dim
      3. PRNUDeepBranch (per-frame PRNU 64-dim)              → 64-dim
      4. PRNUTemporalBranch (inter-frame PRNU Δ 64-dim)     → 32-dim
      5. MotionTemporalBranch (real optical flow, GRU)       → 128-dim  ← NEW v5
      6. AudioBranch (optional mel-spectrogram)              → 64-dim

    MotionTemporalBranch inputs:
      motion_feats: (B, T, 48) sequence from motion_analyzer.extract_sequence_features()
        or None — zeros used as fallback (model gracefully degrades without it)

    Real motion features detected by MotionTemporalBranch:
      - Camera micro-vibration (absent in AI video)
      - Temporal acceleration / velocity jitter
      - GAN grid artifacts in flow FFT spectrum
      - Corner-vs-centre motion (rigid camera shake signature)
      - Direction entropy and flow coherence

    VRAM estimate @ batch=4, 512×512, T=8 pairs, AMP FP16: ~2.8 GB
    """
    _CNN_DIM    = 1536
    _FLOW_DIM   = 64
    _PRNU_DIM   = 64
    _PRNU_TEMP_DIM = 32
    _MOTION_DIM = 128   # ← NEW
    _AUDIO_DIM  = 64
    _FREEZE_RATIO = 0.60

    def __init__(self, dropout: float = 0.4, use_audio: bool = False):
        super().__init__()
        self.use_audio = use_audio

        net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        children = list(net.features.children())
        freeze_until = int(len(children) * self._FREEZE_RATIO)
        for i, b in enumerate(children):
            if i < freeze_until:
                for p in b.parameters(): p.requires_grad = False
        self.backbone    = net.features
        self.pool        = nn.AdaptiveAvgPool2d((1, 1))
        self._last_block = children[-1]

        self.temporal_branch      = TemporalFlowBranch(in_channels=6, out_features=self._FLOW_DIM)
        self.prnu_branch          = PRNUDeepBranch(in_features=64, out_features=self._PRNU_DIM)
        self.prnu_temporal_branch = PRNUTemporalBranch(in_features=64, out_features=self._PRNU_TEMP_DIM)
        self.motion_branch        = MotionTemporalBranch(motion_dim=48, out_features=self._MOTION_DIM)
        if use_audio:
            self.audio_branch = AudioBranch(out_features=self._AUDIO_DIM)

        branch_dims = [self._CNN_DIM, self._FLOW_DIM, self._PRNU_DIM,
                       self._PRNU_TEMP_DIM, self._MOTION_DIM]
        if use_audio:
            branch_dims.append(self._AUDIO_DIM)
        self.fusion = CrossAttentionFusion(branch_dims, out_dim=320)

        self.classifier = nn.Sequential(
            nn.Linear(320, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        # Aux head for motion branch (training supervision)
        self.motion_aux_head = nn.Sequential(
            nn.Linear(self._MOTION_DIM, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        self._init_head()

    def forward(self, img, flow, prnu_feats, prnu_delta,
                motion_feats=None, audio=None):
        """
        Args:
            img          : (B, 3, H, W)   ImageNet-normalised
            flow         : (B, 6, H, W)   pre-computed 6-channel flow map
            prnu_feats   : (B, 64)         per-frame PRNU features
            prnu_delta   : (B, 64)         inter-frame PRNU delta
            motion_feats : (B, T, 48) or (B, 48) — from MotionAnalyzer, or None
            audio        : (B, 1, F, T_mel) mel-spectrogram, or None

        Returns (train mode):  (logit, motion_aux_logit)
        Returns (eval  mode):  logit  (B, 1)
        """
        B  = img.size(0)
        _d = img.device
        _t = img.dtype

        def _z(dim): return torch.zeros(B, dim, device=_d, dtype=_t)

        # ── Backbone ─────────────────────────────────────────────────────
        try:
            cnn = self.pool(self.backbone(img)).view(B, -1)   # (B, 1536)
            if torch.isnan(cnn).any() or torch.isinf(cnn).any():
                raise ValueError("NaN/Inf in backbone output")
        except Exception as exc:
            _log.warning("[VideoTemporalFusionNet] backbone failed (%s) — zeroing", exc)
            cnn = _z(self._CNN_DIM)

        # ── Per-branch calls (each individually guarded via branch try-except) ─
        flw = _safe_branch(self.temporal_branch,      flow,        out_dim=self._FLOW_DIM,     B=B, device=_d, dtype=_t)
        prn = _safe_branch(self.prnu_branch,          prnu_feats,  out_dim=self._PRNU_DIM,     B=B, device=_d, dtype=_t)
        prd = _safe_branch(self.prnu_temporal_branch, prnu_delta,  out_dim=self._PRNU_TEMP_DIM,B=B, device=_d, dtype=_t)

        # Motion branch — fall back to zeros if not provided
        if motion_feats is not None:
            mot = _safe_branch(self.motion_branch, motion_feats, out_dim=self._MOTION_DIM, B=B, device=_d, dtype=_t)
        else:
            mot = _z(self._MOTION_DIM)

        # Audio branch — fall back to zeros if not provided or fails
        branches = [cnn, flw, prn, prd, mot]
        if self.use_audio and audio is not None:
            aud = _safe_branch(self.audio_branch, audio, out_dim=self._AUDIO_DIM, B=B, device=_d, dtype=_t)
            branches.append(aud)

        # ── Fusion + Classifier ───────────────────────────────────────────
        try:
            logit = self.classifier(self.fusion(branches))   # (B, 1)
            if torch.isnan(logit).any() or torch.isinf(logit).any():
                raise ValueError("NaN/Inf in classifier output")
        except Exception as exc:
            _log.warning("[VideoTemporalFusionNet] fusion/classifier failed (%s) — returning zeros", exc)
            logit = _z(1)

        if self.training:
            try:
                mot_aux = self.motion_aux_head(mot)
                if torch.isnan(mot_aux).any() or torch.isinf(mot_aux).any():
                    mot_aux = _z(1)
            except Exception as exc:
                _log.warning("[VideoTemporalFusionNet] motion_aux_head failed (%s)", exc)
                mot_aux = _z(1)
            return logit, mot_aux          # (B,1), (B,1)
        return logit

    def predict_proba(self, logits): return torch.sigmoid(logits)

    @property
    def gradcam_target_layer(self): return self._last_block

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  VideoTemporalFusionNet v5  "
              f"(B3 + flow + PRNU + PRNU-Δ + MotionGRU + CrossAttention, "
              f"Audio: {self.use_audio})")
        print(f"  Total: {total:,}  Trainable: {trainable:,}")


# ---------------------------------------------------------------------------
#  FormatForensicsBranch — 128-dim format feature → 96-dim embedding
# ---------------------------------------------------------------------------

class FormatForensicsBranch(nn.Module):
    """
    Process the 128-dim format feature vector produced by FormatAnalyzer.

    LayerNorm first: handles zero-vector fallback gracefully and equalises
    the mixed-scale inputs (one-hot flags mixed with normalised scalars).

    Input:  (B, 128)
    Output: (B, 96)
    """

    def __init__(self, in_features: int = 128, out_features: int = 96):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(256, 192),         nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(192, out_features), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 2:
                raise ValueError(f"FormatForensicsBranch expects (B, in_features), got {x.shape}")
            out = self.mlp(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("NaN/Inf in FormatForensicsBranch output")
            return out
        except Exception as exc:
            _log.warning("[FormatForensicsBranch] %s — returning zeros. (%s)", type(exc).__name__, exc)
            B = x.shape[0] if x.dim() >= 1 else 1
            return torch.zeros(B, 96, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
#  UnifiedFusionNet v1 — single model for both image and video detection
# ---------------------------------------------------------------------------

class UnifiedFusionNet(nn.Module):
    """
    Unified image + video AI-detector — v1.

    Shares one EfficientNet-B3 backbone, one BranchGate, one CrossAttentionFusion.
    Image branches are always computed; video branches are zeroed in image mode.

    Image mode  (default, mode="image"):
        forward(img, prnu_feats, prnu_map, format_feats=None)
        Train → (logit, *9_aux_logits)   Eval → logit

    Video mode  (mode="video"):
        forward(img, prnu_feats, prnu_map, flow, prnu_delta, motion_feats,
                format_feats=None, mode="video")
        Train → (logit, *9_aux_logits)   Eval → logit
        (index 8 = motion_aux, index 9 = format_aux)

    Branch layout (17 total, sum of dims = 3072):
      Image:  backbone-1536, SRM-64, Freq-128, Color-64, PRNU-96,
              Spatial-128, Halluc-128, PRNUSpatial-128, GAN-128,
              CMOS-96, ColorIncon-96, FlowIrreg-96, Format-96
      Video:  TemporalFlow-64, PRNUDeep-64, PRNUTemporal-32, MotionGRU-128

    Args:
        dropout                : dropout rate for classifier (default 0.4)
        prnu_in_features       : PRNU feature dim (default 64)
        gradient_checkpointing : save VRAM by checkpointing backbone (default False)
    """

    _FREEZE_RATIO    = 0.60
    _IMG_BRANCH_DIMS = [1536, 64, 128, 64, 96, 128, 128, 128, 128, 96, 96, 96, 96]
    _VID_BRANCH_DIMS = [64, 64, 32, 128]
    _ALL_DIMS        = _IMG_BRANCH_DIMS + _VID_BRANCH_DIMS   # 17 branches, sum=3072

    def __init__(self, dropout: float = 0.4, prnu_in_features: int = 64,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.use_ckpt = gradient_checkpointing

        # ── Shared backbone (EfficientNet-B3, bottom 60% frozen) ─────────
        net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        feature_children = list(net.features.children())
        freeze_until = int(len(feature_children) * self._FREEZE_RATIO)
        for i, block in enumerate(feature_children):
            if i < freeze_until:
                for p in block.parameters():
                    p.requires_grad_(False)
        self.backbone    = net.features
        self.pool        = nn.AdaptiveAvgPool2d((1, 1))
        self._last_block = feature_children[-1]

        # ── Image branches (identical attribute names to DeepFusionNet) ───
        self.srm_branch          = SRMBranch(out_features=64)
        self.freq_branch         = FrequencyBranch(out_features=128)
        self.color_branch        = ColorForensicsBranch(out_features=64)
        self.prnu_branch_v2      = PRNUBranchV2(in_features=prnu_in_features, out_features=96)
        self.spatial_branch      = SpatialCNNBranch(out_features=128)
        self.halluc_branch       = HallucinationBranch(out_features=128)
        self.prnu_spatial_branch = PRNUSpatialBranch(out_features=128)
        self.gan_diff_branch     = GANDiffusionFingerprintBranch(out_features=128)
        self.cmos_branch         = CMOSCCDSensorBranch(out_features=96)
        self.color_incon_branch  = ColorChannelInconsistencyBranch(out_features=96)
        self.flow_branch         = OpticalFlowIrregularityBranch(out_features=96)
        self.format_branch       = FormatForensicsBranch(in_features=128, out_features=96)

        # ── Video-specific branches ───────────────────────────────────────
        self.temporal_branch      = TemporalFlowBranch(in_channels=6, out_features=64)
        self.prnu_deep_branch     = PRNUDeepBranch(in_features=64, out_features=64)
        self.prnu_temporal_branch = PRNUTemporalBranch(in_features=64, out_features=32)
        self.motion_branch        = MotionTemporalBranch(motion_dim=48, out_features=128)

        # ── Unified gate + fusion (16 branches) ──────────────────────────
        self.branch_gate = BranchGate(self._ALL_DIMS)
        self.fusion = CrossAttentionFusion(
            self._ALL_DIMS, out_dim=768, d_model=192, n_heads=4, n_layers=5
        )
        # n_layers 3→5: adds 2 more transformer layers (new keys, safe with strict=False).
        # Existing layers.0/1/2 load from checkpoint; layers.3/4 init randomly.

        # ── Classifier ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(768, 384), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(384, 64),  nn.SiLU(), nn.Dropout(dropout * 0.75),
            nn.Linear(64, 1),
        )

        # ── Aux heads (training only) ─────────────────────────────────────
        def _aux(d: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1))

        self.prnu_aux_head         = _aux(96)
        self.halluc_aux_head       = _aux(128)
        self.prnu_spatial_aux_head = _aux(128)
        self.gan_diff_aux_head     = _aux(128)
        self.cmos_aux_head         = _aux(96)
        self.color_incon_aux_head  = _aux(96)
        self.flow_aux_head         = _aux(96)
        self.motion_aux_head       = _aux(128)   # meaningful in video mode
        self.format_aux_head       = _aux(96)

        self._init_weights()

    # ------------------------------------------------------------------

    def forward(self, img: torch.Tensor, prnu_feats: torch.Tensor,
                prnu_map: torch.Tensor = None,
                flow: torch.Tensor = None,
                prnu_delta: torch.Tensor = None,
                motion_feats: torch.Tensor = None,
                format_feats: torch.Tensor = None,
                mode: str = "image"):
        B  = img.size(0)
        _d = img.device
        _t = img.dtype

        def _z(dim): return torch.zeros(B, dim, device=_d, dtype=_t)

        # ── Backbone ─────────────────────────────────────────────────────
        try:
            if self.use_ckpt and self.training:
                cnn_out = grad_checkpoint(
                    lambda x: self.pool(self.backbone(x)).view(x.size(0), -1),
                    img, use_reentrant=False,
                )
            else:
                cnn_out = self.pool(self.backbone(img)).view(B, -1)      # (B,1536)
            if torch.isnan(cnn_out).any() or torch.isinf(cnn_out).any():
                raise ValueError("NaN/Inf in backbone output")
        except Exception as exc:
            _log.warning("[UnifiedFusionNet] backbone failed (%s) — zeroing", exc)
            cnn_out = _z(1536)

        # ── Image branches (each independently guarded via _safe_branch) ─
        srm_out         = _safe_branch(self.srm_branch,          img,        out_dim=64,  B=B, device=_d, dtype=_t)
        freq_out        = _safe_branch(self.freq_branch,         img,        out_dim=128, B=B, device=_d, dtype=_t)
        color_out       = _safe_branch(self.color_branch,        img,        out_dim=64,  B=B, device=_d, dtype=_t)
        prnu_out        = _safe_branch(self.prnu_branch_v2,      prnu_feats, out_dim=96,  B=B, device=_d, dtype=_t)
        spatial_out     = _safe_branch(self.spatial_branch,      img,        out_dim=128, B=B, device=_d, dtype=_t)
        halluc_out      = _safe_branch(self.halluc_branch,       img,        out_dim=128, B=B, device=_d, dtype=_t)

        if prnu_map is None:
            prnu_map = _z(3 * 128 * 128).view(B, 3, 128, 128)
        prnu_spatial_out = _safe_branch(self.prnu_spatial_branch, prnu_map,  out_dim=128, B=B, device=_d, dtype=_t)
        gan_diff_out     = _safe_branch(self.gan_diff_branch,    img,        out_dim=128, B=B, device=_d, dtype=_t)
        cmos_out         = _safe_branch(self.cmos_branch,        img,        out_dim=96,  B=B, device=_d, dtype=_t)
        color_incon_out  = _safe_branch(self.color_incon_branch, img,        out_dim=96,  B=B, device=_d, dtype=_t)
        flow_irr_out     = _safe_branch(self.flow_branch,        img,        out_dim=96,  B=B, device=_d, dtype=_t)
        _fmt_in          = format_feats if format_feats is not None else _z(128)
        format_out       = _safe_branch(self.format_branch,     _fmt_in,    out_dim=96,  B=B, device=_d, dtype=_t)

        # ── Video branches (zeros in image mode) ─────────────────────────
        if mode == "video" and flow is not None:
            temporal_out  = _safe_branch(self.temporal_branch,      flow,       out_dim=64,  B=B, device=_d, dtype=_t)
            prnu_deep_out = _safe_branch(self.prnu_deep_branch,     prnu_feats, out_dim=64,  B=B, device=_d, dtype=_t)
            _delta = prnu_delta if prnu_delta is not None else torch.zeros_like(prnu_feats)
            prnu_temp_out = _safe_branch(self.prnu_temporal_branch, _delta,     out_dim=32,  B=B, device=_d, dtype=_t)
            _mf = motion_feats if motion_feats is not None else torch.zeros(B, 1, 48, device=_d, dtype=_t)
            motion_out    = _safe_branch(self.motion_branch,        _mf,        out_dim=128, B=B, device=_d, dtype=_t)
        else:
            temporal_out  = _z(64)
            prnu_deep_out = _z(64)
            prnu_temp_out = _z(32)
            motion_out    = _z(128)

        # ── Gate + Fusion + Classifier ────────────────────────────────────
        try:
            branch_list = [
                cnn_out, srm_out, freq_out, color_out,
                prnu_out, spatial_out, halluc_out, prnu_spatial_out,
                gan_diff_out, cmos_out, color_incon_out, flow_irr_out,
                format_out,
                temporal_out, prnu_deep_out, prnu_temp_out, motion_out,
            ]
            branch_list = self.branch_gate(branch_list)
            fused = self.fusion(branch_list)
            logit = self.classifier(fused)                               # (B,1)
            if torch.isnan(logit).any() or torch.isinf(logit).any():
                raise ValueError("NaN/Inf in classifier output")
        except Exception as exc:
            _log.warning("[UnifiedFusionNet] fusion/classifier failed (%s) — returning zeros", exc)
            logit = _z(1)

        if self.training:
            def _aux_safe(head, feat, name):
                try:
                    out = head(feat)
                    return out if not (torch.isnan(out).any() or torch.isinf(out).any()) else _z(1)
                except Exception as e:
                    _log.warning("[UnifiedFusionNet] aux head %s failed (%s)", name, e)
                    return _z(1)

            return (
                logit,
                _aux_safe(self.prnu_aux_head,         prnu_out,         "prnu"),
                _aux_safe(self.halluc_aux_head,        halluc_out,       "halluc"),
                _aux_safe(self.prnu_spatial_aux_head,  prnu_spatial_out, "prnu_spatial"),
                _aux_safe(self.gan_diff_aux_head,      gan_diff_out,     "gan_diff"),
                _aux_safe(self.cmos_aux_head,          cmos_out,         "cmos"),
                _aux_safe(self.color_incon_aux_head,   color_incon_out,  "color_incon"),
                _aux_safe(self.flow_aux_head,          flow_irr_out,     "flow"),
                _aux_safe(self.motion_aux_head,        motion_out,       "motion"),
                _aux_safe(self.format_aux_head,        format_out,       "format"),
            )
        return logit

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self._last_block

    def param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(
            "  UnifiedFusionNet v1  (B3 + SRM×15 + FFT×3 + YCbCr + PRNU-64 + SpatialCNN +\n"
            "    MobileNetV3 + PRNUSpatialMap + GAN/Diff + CMOS + ColorIncon + FlowIrreg +\n"
            "    Format + TemporalFlow + PRNUDeep + PRNUTemporal + MotionGRU +\n"
            "    BranchGate-17 + CrossAttn-192×5×17 + INT8-ready)"
        )
        print(f"  Total parameters    : {total:,}")
        print(f"  Trainable           : {trainable:,}  ({trainable/total*100:.1f}%)")
        print(f"  Frozen (backbone)   : {frozen:,}  ({frozen/total*100:.1f}%)")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv1d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @classmethod
    def migrate_from_checkpoints(cls, image_ckpt_path: str,
                                  video_ckpt_path: str = None,
                                  device: str = 'cpu') -> 'UnifiedFusionNet':
        """
        Build a UnifiedFusionNet pre-loaded from existing separate checkpoints.

        DeepFusionNet keys map directly (same attribute names for all image branches).
        VideoTemporalFusionNet video-specific keys are remapped:
          prnu_branch.*  → prnu_deep_branch.*   (renamed in UnifiedFusionNet)
          all others     → unchanged             (temporal_branch, motion_branch, etc.)
        """
        net = cls()
        if image_ckpt_path and os.path.exists(image_ckpt_path):
            try:
                sd = torch.load(image_ckpt_path, map_location=device, weights_only=True)
            except Exception:
                sd = torch.load(image_ckpt_path, map_location=device)
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            elif 'model_state' in sd:
                sd = sd['model_state']
            net.load_state_dict(sd, strict=False)
            print(f"  Loaded image weights from {image_ckpt_path}")
        if video_ckpt_path and os.path.exists(video_ckpt_path):
            try:
                sd = torch.load(video_ckpt_path, map_location=device, weights_only=True)
            except Exception:
                sd = torch.load(video_ckpt_path, map_location=device)
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            elif 'model_state' in sd:
                sd = sd['model_state']
            remapped = {}
            for k, v in sd.items():
                if k.startswith('prnu_branch.'):
                    remapped['prnu_deep_branch.' + k[len('prnu_branch.'):]] = v
                else:
                    remapped[k] = v
            net.load_state_dict(remapped, strict=False)
            print(f"  Loaded video weights from {video_ckpt_path}")
        return net


# Alias
UnifiedDetector = UnifiedFusionNet


# ---------------------------------------------------------------------------
#  Legacy shim
# ---------------------------------------------------------------------------

class PyTorchCNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features   = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.avgpool    = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )
    def forward(self, x): return self.classifier(self.avgpool(self.features(x)))


# ---------------------------------------------------------------------------
#  Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running DeepFusionNet v6.1 sanity check...")
    model = DeepFusionNet(prnu_in_features=64, gradient_checkpointing=True)
    model.param_summary()

    img  = torch.randn(2, 3, 512, 512)
    prnu = torch.randn(2, 64)
    pmap = torch.randn(2, 3, 128, 128)

    model.train()
    with torch.enable_grad():
        out = model(img, prnu, pmap)
    print(f"\n  Train output: {[x.shape for x in out]}")
    # 1 logit + 7 aux heads = 8 tensors
    assert len(out) == 8, f"Expected 8 outputs in train mode, got {len(out)}"

    model.eval()
    with torch.no_grad():
        out = model(img, prnu, pmap)
    print(f"  Eval output : {out.shape}")
    assert out.shape == (2, 1)

    # Also test backward-compat (no prnu_map)
    with torch.no_grad():
        out2 = model(img, prnu)
    assert out2.shape == (2, 1)
    print("  Backward-compat (no prnu_map): PASSED")

    # INT8 quantization
    import model_prnu as m
    qnet = m.quantize_model_for_inference(model)
    print(f"  Quantized OK: {type(qnet).__name__}")
    print("DeepFusionNet v6.1 sanity check PASSED ✓")
