import os
import io
import base64
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

# PRNU helpers
from prnu import analyze_prnu
from prnu_features import extract_prnu_features

# --- Configuration ---
IMG_WIDTH  = 512
IMG_HEIGHT = 512

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(SCRIPT_DIR, '..', 'models')

# New dual-branch model path (preferred)
NEW_MODEL_PATH = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion.pth')
# Original single-branch fallback
OLD_MODEL_PATH = os.path.join(MODELS_DIR, 'ai_detector_model_pytorch.pth')


# ═══════════════════════════════════════════════════════════════════════════
#  Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════

def compute_gradcam(
    model: nn.Module,
    img_tensor: torch.Tensor,
    prnu_tensor,
    target_layer: nn.Module,
    orig_width: int,
    orig_height: int,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for the given inputs.

    Registers forward + backward hooks on `target_layer`, runs one forward
    pass and one backward pass on the positive class logit, then computes
    the channel-weighted activation map.

    Args:
        model        : PRNUFusionNet (or any model with spatial feature maps)
        img_tensor   : (1, 3, H, W) — normalised image tensor, requires_grad
        prnu_tensor  : (1, 8) or None — PRNU features (None for legacy model)
        target_layer : the Conv layer to hook
        orig_width   : width to resize the heatmap to
        orig_height  : height to resize the heatmap to

    Returns:
        np.ndarray: heatmap of shape (orig_height, orig_width), float32 in [0, 1]
    """
    activations = {}
    gradients   = {}

    def fwd_hook(module, inp, out):
        activations['feat'] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients['grad'] = grad_out[0]

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        img_tensor = img_tensor.requires_grad_(True)

        # Forward
        if prnu_tensor is not None:
            logits = model(img_tensor, prnu_tensor)
        else:
            logits = model(img_tensor)

        # Backward on positive class (AI)
        score = logits[0, 0]
        score.backward()

        acts = activations['feat']          # (1, C, h, w)
        grads = gradients['grad']           # (1, C, h, w)

        # Global average pool the gradients → channel weights
        weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Normalise to [0, 1]
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to original image dimensions
        cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((orig_width, orig_height), Image.BILINEAR)
        return np.array(cam_pil, dtype=np.float32) / 255.0

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def heatmap_to_base64_png(
    heatmap: np.ndarray,
    orig_image_array: np.ndarray,
    alpha: float = 0.55,
) -> str:
    """
    Overlay the heatmap (JET colormap) on the original image and return
    a base64-encoded PNG string.

    Args:
        heatmap           : (H, W) float32 in [0, 1]
        orig_image_array  : (H, W, 3) uint8 RGB original image
        alpha             : heatmap blend factor (0 = original only, 1 = heatmap only)

    Returns:
        str: base64-encoded PNG
    """
    # Jet colormap: map [0,1] → RGB
    h_uint8 = (heatmap * 255).astype(np.uint8)
    jet = _apply_jet_colormap(h_uint8)  # (H, W, 3) uint8

    # Blend with original image
    orig = orig_image_array.astype(np.float32)
    blended = (1.0 - alpha) * orig + alpha * jet.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(blended)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _apply_jet_colormap(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Pure-numpy JET colormap.
    Input:  (H, W) uint8 in [0, 255]
    Output: (H, W, 3) uint8 RGB
    """
    v = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  Internal model definitions (kept local for backward compat)
# ═══════════════════════════════════════════════════════════════════════════

class _PyTorchCNN(nn.Module):
    """Legacy single-branch fallback model."""
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )
        # Expose last conv for Grad-CAM
        self.last_conv  = self.features[-3]   # the third Conv2d

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(self.avgpool(feat))


# ═══════════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════════

class Detector:
    """
    Main inference class.

    Automatically loads the new PRNUFusionNet if available; falls back to
    the legacy PyTorchCNN otherwise. Grad-CAM is supported for both.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fusion = False

        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # --- Try to load the new PRNUFusionNet first ---
        if os.path.exists(NEW_MODEL_PATH):
            try:
                from model_prnu import PRNUFusionNet
                self.model = PRNUFusionNet().to(self.device)
                state = torch.load(NEW_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.eval()
                self.use_fusion = True
                print(f"✓ Loaded PRNUFusionNet from {NEW_MODEL_PATH}")
            except Exception as e:
                print(f"⚠ Could not load PRNUFusionNet ({e}), falling back to legacy model.")
                self.use_fusion = False

        # --- Fallback to legacy model ---
        if not self.use_fusion:
            if not os.path.exists(OLD_MODEL_PATH):
                raise FileNotFoundError(
                    f"No model file found.\n"
                    f"  Tried: {NEW_MODEL_PATH}\n"
                    f"  Tried: {OLD_MODEL_PATH}\n"
                    f"Please train the model first."
                )
            self.model = _PyTorchCNN(num_classes=1).to(self.device)
            state = torch.load(OLD_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"✓ Loaded legacy PyTorchCNN from {OLD_MODEL_PATH}")

        # --- Identify Grad-CAM target layer ---
        if self.use_fusion:
            self.gradcam_layer = self.model.gradcam_target_layer
        else:
            self.gradcam_layer = self.model.last_conv

    # ------------------------------------------------------------------

    def predict(self, image_data: bytes, compute_heatmap: bool = True) -> dict:
        """
        Analyse raw image bytes.

        Args:
            image_data      : raw file bytes (JPEG, PNG, HEIC, …)
            compute_heatmap : if True, run Grad-CAM and include heatmap_base64

        Returns:
            dict with keys:
                ai_probability   — float [0, 1]
                conclusion       — "AI-Generated" | "REAL"
                prnu_analysis    — dict of raw PRNU metrics
                heatmap_base64   — base64 PNG string (if compute_heatmap)
                heatmap_width    — int
                heatmap_height   — int
                model_type       — "PRNUFusionNet" | "PyTorchCNN"
        """
        # --- Load image ---
        try:
            img = self._load_image(image_data)
        except Exception as e:
            return {"error": f"Could not load image: {e}"}

        orig_w, orig_h = img.size
        orig_array = np.array(img.resize((min(orig_w, 1024), min(orig_h, 1024)),
                                         Image.BILINEAR))

        # --- Preprocess ---
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # --- PRNU features (always computed, even for legacy model) ---
        _prnu_size = 128
        prnu_raw = np.array(img.resize((_prnu_size, _prnu_size), Image.BILINEAR))
        prnu_feats_np = extract_prnu_features(prnu_raw.astype(np.float64) / 255.0)
        prnu_tensor = torch.from_numpy(prnu_feats_np).unsqueeze(0).float().to(self.device)

        # --- Inference ---
        with torch.no_grad():
            if self.use_fusion:
                logits = self.model(img_tensor, prnu_tensor)
                ai_prob = float(torch.sigmoid(logits).item())
            else:
                # Legacy model outputs sigmoid directly
                ai_prob_cnn = float(self.model(img_tensor).item())
                # Post-process PRNU blend (old behaviour preserved)
                prnu_result_full = analyze_prnu(image_data)
                prnu_ai_score = 1.0 - prnu_result_full.get("prnu_likelihood_real", 0.5)
                ai_prob = 0.7 * ai_prob_cnn + 0.3 * prnu_ai_score

        # --- PRNU detailed analysis (for the response dict) ---
        try:
            prnu_analysis = analyze_prnu(image_data)
        except Exception:
            prnu_analysis = {}

        result = {
            "ai_probability": round(ai_prob, 4),
            "conclusion":     "AI-Generated" if ai_prob > 0.5 else "REAL",
            "prnu_analysis":  prnu_analysis,
            "model_type":     "PRNUFusionNet" if self.use_fusion else "PyTorchCNN",
        }

        # --- Grad-CAM heatmap ---
        if compute_heatmap:
            try:
                heatmap = compute_gradcam(
                    model        = self.model,
                    img_tensor   = img_tensor.clone(),
                    prnu_tensor  = prnu_tensor if self.use_fusion else None,
                    target_layer = self.gradcam_layer,
                    orig_width   = orig_array.shape[1],
                    orig_height  = orig_array.shape[0],
                )
                heatmap_b64 = heatmap_to_base64_png(heatmap, orig_array)
                result["heatmap_base64"] = heatmap_b64
                result["heatmap_width"]  = orig_array.shape[1]
                result["heatmap_height"] = orig_array.shape[0]
            except Exception as e:
                print(f"⚠ Grad-CAM failed: {e}")
                result["heatmap_base64"] = None
                result["heatmap_width"]  = orig_w
                result["heatmap_height"] = orig_h

        return result

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_data: bytes) -> Image.Image:
        """Load image bytes, with HEIC support."""
        try:
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            pass
        # HEIC fallback
        try:
            import pyheif
            heif_file = pyheif.read(image_data)
            return Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data,
                "raw", heif_file.mode, heif_file.stride,
            ).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Could not decode image: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect.py <path_to_image> [--no-heatmap]")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' does not exist.")
        return

    no_heatmap = '--no-heatmap' in sys.argv

    try:
        detector = Detector()
        with open(image_path, "rb") as f:
            image_data = f.read()

        print(f"\nAnalysing: {os.path.basename(image_path)} ...")
        result = detector.predict(image_data, compute_heatmap=not no_heatmap)

        print("\n--- Detection Result ---")
        print(f"  Model          : {result.get('model_type', 'unknown')}")
        print(f"  AI Probability : {result['ai_probability'] * 100:.2f}%")
        print(f"  Conclusion     : {result['conclusion']}")
        if result.get('heatmap_base64'):
            hm_len = len(result['heatmap_base64'])
            print(f"  Heatmap        : {hm_len} bytes (base64 PNG)")
        print(f"  PRNU Analysis  : {result.get('prnu_analysis', {})}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
