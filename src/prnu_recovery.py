"""
prnu_recovery.py — Neural PRNU Recovery Network

DnCNN-inspired UNet that recovers PRNU noise destroyed by JPEG compression,
social media sharing (Telegram Q~80, WhatsApp Q~75), ZIP, etc.

Architecture: PRNURecoveryNet
  Input: (B, 3, H, W) float32 [0, 1]
  Encoder:
    Stage 1: Conv2d(3→64, 3×3) + BN + ReLU + MaxPool → H/2
    Stage 2: Conv2d(64→128, 3×3) + BN + ReLU + MaxPool → H/4
    Stage 3: Conv2d(128→256, 3×3) + BN + ReLU + MaxPool → H/8
  Bottleneck: 2× ResBlock(256→256)
  Decoder (bilinear upsample ×2 + concat skip):
    Stage 3: (256+256)→128 + BN + ReLU
    Stage 2: (128+128)→64 + BN + ReLU
    Stage 1: (64+64)→32 + BN + ReLU
  Output: Conv2d(32→3, 1×1) → residual damage map
  Final: input − predicted_residual (recovered image)

Key: Residual learning — the network predicts the compression-induced damage
to subtract, not the clean image directly.  Near-zero output head init
(std=0.01) makes the untrained network approximate the identity function.
"""

import io
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
RECOVERY_PATH = os.path.join(MODELS_DIR, 'prnu_recovery.pth')

TILE_SIZE    = 256
TILE_OVERLAP = 32


# ---------------------------------------------------------------------------
#  Building block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block used in the bottleneck."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.block(x))


# ---------------------------------------------------------------------------
#  PRNURecoveryNet
# ---------------------------------------------------------------------------

class PRNURecoveryNet(nn.Module):
    """
    UNet-style residual damage predictor for PRNU signal recovery.

    The network learns to predict the compression-induced distortion
    (residual damage) and subtracts it from the degraded input, recovering
    an approximation of the original (uncompressed) image.

    Input:  (B, 3, H, W) float32 [0, 1]
    Output: (B, 3, H, W) float32 [0, 1] — recovered image
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
        )

        # Decoder stage 3: concat(upsample(b), e3) = (256+256) → 128
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder stage 2: concat(upsample(d3), e2) = (128+128) → 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Decoder stage 1: concat(upsample(d2), e1) = (64+64) → 32
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Output head: predict compression damage (near-zero init → identity)
        self.output_conv = nn.Conv2d(32, 3, 1, bias=True)
        nn.init.normal_(self.output_conv.weight, std=0.01)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                        # (B, 64,  H,   W)
        e2 = self.enc2(self.pool1(e1))            # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))            # (B, 256, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))       # (B, 256, H/8, W/8)

        # Decoder with skip connections
        d3 = F.interpolate(b,  size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))   # (B, 128, H/4, W/4)

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))   # (B, 64,  H/2, W/2)

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))   # (B, 32,  H,   W)

        damage    = self.output_conv(d1)              # (B, 3, H, W)
        recovered = x - damage                        # residual subtraction
        return torch.clamp(recovered, 0.0, 1.0)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def build_prnu_recovery_net(device=None) -> PRNURecoveryNet:
    """
    Build a PRNURecoveryNet and load weights if available.

    If `models/prnu_recovery.pth` exists the weights are loaded.
    Otherwise the untrained network (≈ identity) is returned.

    Args:
        device : torch.device, str, or None (defaults to CPU)

    Returns:
        PRNURecoveryNet in eval mode, on the requested device
    """
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    net = PRNURecoveryNet().to(device)

    if os.path.exists(RECOVERY_PATH):
        try:
            state = torch.load(RECOVERY_PATH, map_location=device, weights_only=True)
            net.load_state_dict(state)
            print(f"[prnu_recovery] Loaded weights from {RECOVERY_PATH}")
        except Exception as e:
            print(f"[prnu_recovery] Could not load {RECOVERY_PATH}: {e} — using untrained net")
    else:
        print("[prnu_recovery] No weights found — using untrained net (identity approximation)")

    net.eval()
    return net


def recover_prnu_signal(
    img_array: np.ndarray,
    net: PRNURecoveryNet,
    device,
) -> np.ndarray:
    """
    Run PRNU recovery on a full-resolution image using tiled inference.

    Processes the image in TILE_SIZE×TILE_SIZE tiles with TILE_OVERLAP px
    overlap and linear blending at seams to avoid tile-boundary artefacts.

    Args:
        img_array : (H, W, 3) float64 [0, 1]
        net       : PRNURecoveryNet in eval mode
        device    : torch.device or str

    Returns:
        np.ndarray  (H, W, 3) float64 [0, 1] — recovered image
    """
    if isinstance(device, str):
        device = torch.device(device)

    h, w      = img_array.shape[:2]
    result    = np.zeros((h, w, 3), dtype=np.float64)
    weight_map = np.zeros((h, w, 1), dtype=np.float64)
    stride    = TILE_SIZE - TILE_OVERLAP

    with torch.no_grad():
        for r0 in range(0, h, stride):
            for c0 in range(0, w, stride):
                r1 = min(r0 + TILE_SIZE, h)
                c1 = min(c0 + TILE_SIZE, w)
                tile = img_array[r0:r1, c0:c1]   # (th, tw, 3)
                th, tw = tile.shape[:2]

                # Pad to TILE_SIZE if needed
                if th < TILE_SIZE or tw < TILE_SIZE:
                    padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float64)
                    padded[:th, :tw] = tile
                    tile_in = padded
                else:
                    tile_in = tile

                tensor = (
                    torch.from_numpy(tile_in.transpose(2, 0, 1))
                    .float().unsqueeze(0).to(device)
                )
                rec = (
                    net(tensor)
                    .squeeze(0).permute(1, 2, 0)
                    .cpu().numpy()
                )                                # (TILE_SIZE, TILE_SIZE, 3)
                rec = rec[:th, :tw]              # crop back to tile size

                blend = _make_blend_weights(th, tw)   # (th, tw, 1)
                result[r0:r1, c0:c1]      += rec * blend
                weight_map[r0:r1, c0:c1]  += blend

    weight_map = np.maximum(weight_map, 1e-8)
    result /= weight_map
    return np.clip(result, 0.0, 1.0)


def train_recovery_net_one_step(
    net: PRNURecoveryNet,
    optimizer,
    scaler,
    clean_batch: torch.Tensor,
    quality_range: tuple = (60, 90),
) -> float:
    """
    One gradient step for the recovery network.

    Generates JPEG-compressed pairs from `clean_batch` on-the-fly (real DCT
    artefacts via PIL encode/decode), then trains the net to reverse the damage.

    Loss: L1 + 0.1 × (1 − SSIM)

    Args:
        net          : PRNURecoveryNet
        optimizer    : optimizer (zero_grad is called inside)
        scaler       : GradScaler for AMP
        clean_batch  : (B, 3, H, W) float32 [0, 1] on the correct device
        quality_range: (min_q, max_q) JPEG quality range

    Returns:
        float: loss value
    """
    device  = clean_batch.device
    use_amp = (device.type == 'cuda')

    # Build compressed batch on CPU (real PIL JPEG encode/decode)
    compressed_list = []
    for i in range(clean_batch.size(0)):
        img_np = (clean_batch[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil    = Image.fromarray(img_np)
        q      = int(np.random.randint(quality_range[0], quality_range[1] + 1))
        buf    = io.BytesIO()
        pil.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        comp = np.array(Image.open(buf).convert('RGB'), dtype=np.float32) / 255.0
        compressed_list.append(comp.transpose(2, 0, 1))

    compressed = torch.from_numpy(np.stack(compressed_list)).to(device)

    net.train()
    with torch.amp.autocast(device.type, enabled=use_amp):
        recovered = net(compressed)
        l1_loss   = F.l1_loss(recovered, clean_batch)
        ssim_loss = _ssim_loss(recovered, clean_batch)
        loss      = l1_loss + 0.1 * ssim_loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return float(loss.item())


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _make_blend_weights(h: int, w: int) -> np.ndarray:
    """
    Linear-fade blend weights for tiled overlap blending. Shape: (h, w, 1).
    Fades from 0 at tile edges to 1 at the centre over TILE_OVERLAP pixels.
    """
    wy   = np.ones(h, dtype=np.float64)
    wx   = np.ones(w, dtype=np.float64)
    fade = min(TILE_OVERLAP, h // 2, w // 2)
    if fade > 0:
        ramp       = np.linspace(0.0, 1.0, fade, endpoint=False)
        wy[:fade]  = ramp
        wy[-fade:] = ramp[::-1]
        wx[:fade]  = ramp
        wx[-fade:] = ramp[::-1]
    blend = np.outer(wy, wx)[:, :, np.newaxis]
    return np.maximum(blend, 1e-6)


def _ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Simplified SSIM loss — returns 1 − SSIM (0 = perfect match).
    Uses a uniform window for speed.
    """
    C1  = 0.01 ** 2
    C2  = 0.03 ** 2
    pad = window_size // 2
    k   = torch.ones(1, 1, window_size, window_size,
                     device=x.device) / (window_size ** 2)

    def blur(t: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = t.shape
        return F.conv2d(t.reshape(b * c, 1, h_, w_), k, padding=pad).reshape(b, c, h_, w_)

    mu_x  = blur(x)
    mu_y  = blur(y)
    sig_x  = blur(x * x) - mu_x ** 2
    sig_y  = blur(y * y) - mu_y ** 2
    sig_xy = blur(x * y) - mu_x * mu_y

    ssim_map = (
        (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    ) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2) + 1e-8
    )
    return 1.0 - ssim_map.mean()
