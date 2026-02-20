"""
model_prnu.py — Dual-Branch PRNUFusionNet

Architecture:
  Branch 1 (CNN):   Standard conv stack processing the RGB image tensor.
                    Last conv layer is registered as `self.last_conv` for
                    Grad-CAM backward-hook targeting.
  Branch 2 (PRNU):  Lightweight MLP processing the 8-dim PRNU feature vector.
  Fusion:           Both branches are concatenated and passed through a
                    shared classifier head.

Input signatures:
  forward(img, prnu_feats)
    img         — (B, 3, H, W)  float32, ImageNet-normalised
    prnu_feats  — (B, 8)        float32, all values in [0, 1]

Output:
  logits — (B, 1) unbounded float (apply sigmoid for probability)

Grad-CAM target:
  model.last_conv  — the final Conv2d layer of the CNN branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → (optional) MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNBranch(nn.Module):
    """
    4-block CNN:  3 → 32 → 64 → 128 → 256 channels.
    Input: (B, 3, 512, 512)
    Output after Global Average Pooling: (B, 256)

    Attribute `last_conv` exposes the final Conv2d for Grad-CAM.
    """
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3,   32,  pool=True)   # → (B,  32, 256, 256)
        self.block2 = ConvBlock(32,  64,  pool=True)   # → (B,  64, 128, 128)
        self.block3 = ConvBlock(64,  128, pool=True)   # → (B, 128,  64,  64)

        # Last conv block (no pool) — used by Grad-CAM
        self.last_conv = ConvBlock(128, 256, pool=False)  # → (B, 256, 64, 64)

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),     # → (B, 256, 32, 32)
            nn.AdaptiveAvgPool2d((1, 1)),              # → (B, 256,  1,  1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.last_conv(x)         # the Grad-CAM target layer
        x = self.pool(x)
        return x.view(x.size(0), -1)  # (B, 256)


class PRNUBranch(nn.Module):
    """
    3-layer MLP for the 8-dim PRNU feature vector.
    Input:  (B, 8)
    Output: (B, 32)
    """
    def __init__(self, in_features: int = 8, out_features: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------------------------
#  Fusion model
# ---------------------------------------------------------------------------

class PRNUFusionNet(nn.Module):
    """
    Dual-branch detector that trains the CNN and PRNU branches jointly.

    The model learns to weight the contribution of PRNU forensic signals
    during training (unlike the previous hard-coded 70/30 post-processing
    mix), making detection more accurate and robust.
    """

    #: Feature dimensions
    _CNN_DIM  = 256
    _PRNU_DIM = 32
    _FUSED_DIM = _CNN_DIM + _PRNU_DIM   # 288

    def __init__(self):
        super().__init__()
        self.cnn_branch  = CNNBranch()
        self.prnu_branch = PRNUBranch()

        self.classifier = nn.Sequential(
            nn.Linear(self._FUSED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            # No Sigmoid here — use BCEWithLogitsLoss during training
        )

        self._initialize_weights()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, img: torch.Tensor, prnu_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img        : (B, 3, H, W)  — normalised image tensor
            prnu_feats : (B, 8)        — normalized PRNU feature vector

        Returns:
            logits     : (B, 1)        — apply sigmoid for probability
        """
        cnn_out  = self.cnn_branch(img)          # (B, 256)
        prnu_out = self.prnu_branch(prnu_feats)  # (B, 32)
        fused    = torch.cat([cnn_out, prnu_out], dim=1)  # (B, 288)
        return self.classifier(fused)            # (B, 1)

    # ------------------------------------------------------------------
    #  Convenience: probability from raw image bytes (inference only)
    # ------------------------------------------------------------------

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert model logits to probability in [0, 1]."""
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    #  Grad-CAM target layer accessor
    # ------------------------------------------------------------------

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Returns the last conv block of the CNN branch for Grad-CAM."""
        return self.cnn_branch.last_conv

    # ------------------------------------------------------------------
    #  Weight initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
#  Legacy compatibility shim
# ---------------------------------------------------------------------------

class PyTorchCNN(nn.Module):
    """
    Original single-branch CNN kept for backward compatibility.
    Used by detect.py as a fallback when the new model file is absent.
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


if __name__ == '__main__':
    # Quick sanity check
    model = PRNUFusionNet()
    model.eval()
    img   = torch.randn(2, 3, 512, 512)
    prnu  = torch.rand(2, 8)
    out   = model(img, prnu)
    proba = model.predict_proba(out)
    print(f"Output shape : {out.shape}")    # expect (2, 1)
    print(f"Proba shape  : {proba.shape}")  # expect (2, 1)
    print(f"Proba values : {proba.detach().numpy().flatten()}")
    print("PRNUFusionNet sanity check PASSED ✓")
