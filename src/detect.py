import os
import io
import base64
import sys
import random
import numpy as np
from PIL import Image
_BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR  # Pillow 10+ compat

import torch
import torch.nn as nn
from torchvision import transforms

# Model definitions
from model_prnu import DeepFusionNet, EfficientFusionNet, PRNUFusionNet, UnifiedFusionNet, check_checkpoint_compat
from prnu import analyze_prnu, extract_noise_residual
from prnu_features import extract_prnu_features, extract_prnu_features_fullres, extract_prnu_map
from image_loader import UniversalImageLoader
from prnu_recovery import build_prnu_recovery_net
from prnu_cuda import PRNUExtractorGPU
from format_analyzer import FormatAnalyzer

# --- Configuration ---
IMG_WIDTH  = 512
IMG_HEIGHT = 512

# Platform compression profiles — used to annotate results and scale PRNU
# reliability when the caller provides a platform hint.
# prnu_reliability: how much PRNU signal survives compression (1.0 = full, 0.0 = destroyed)
PLATFORM_PROFILES = {
    # ── Social / streaming platforms ────────────────────────────────────────
    'youtube':   {'tier': 'high',    'codec': 'H.264/AV1',      'max_px': 4096,
                  'quality_range': (85, 95),  'prnu_reliability': 1.00,
                  'description': 'High bitrate, minimal compression'},
    'vimeo':     {'tier': 'high',    'codec': 'H.264/VP9',      'max_px': 4096,
                  'quality_range': (90, 98),  'prnu_reliability': 1.00,
                  'description': 'Near-lossless, excellent quality'},
    'tiktok':    {'tier': 'medium',  'codec': 'H.265',          'max_px': 1080,
                  'quality_range': (70, 82),  'prnu_reliability': 0.80,
                  'description': 'Decent quality, efficient H.265'},
    'instagram': {'tier': 'medium',  'codec': 'H.264',          'max_px': 1080,
                  'quality_range': (60, 75),  'prnu_reliability': 0.70,
                  'description': 'Noticeably aggressive re-encode'},
    'facebook':  {'tier': 'heavy',   'codec': 'H.264',          'max_px': 960,
                  'quality_range': (55, 72),  'prnu_reliability': 0.55,
                  'description': 'Heavy compression, double-encoded'},
    'twitter':   {'tier': 'heavy',   'codec': 'H.264',          'max_px': 900,
                  'quality_range': (50, 68),  'prnu_reliability': 0.50,
                  'description': 'Among the worst quality platforms'},
    'snapchat':  {'tier': 'extreme', 'codec': 'H.264',          'max_px': 720,
                  'quality_range': (40, 60),  'prnu_reliability': 0.35,
                  'description': 'Lowest quality, speed over fidelity'},
    # ── Messaging ────────────────────────────────────────────────────────────
    'telegram':  {'tier': 'medium',  'codec': 'MTProto/JPEG',   'max_px': 1280,
                  'quality_range': (78, 85),  'prnu_reliability': 0.82,
                  'description': 'Telegram photo send (MTProto protocol)'},
    # ── Container / codec formats ────────────────────────────────────────────
    'heic':      {'tier': 'high',    'codec': 'HEVC/HEIF',      'max_px': 4032,
                  'quality_range': (85, 92),  'prnu_reliability': 0.92,
                  'description': 'Apple HEIC/HEIF — efficient near-lossless format'},
    'h264':      {'tier': 'medium',  'codec': 'H.264/AVC',      'max_px': 1920,
                  'quality_range': (72, 88),  'prnu_reliability': 0.72,
                  'description': 'Generic H.264/AVC encoded video frame'},
    'h265':      {'tier': 'medium',  'codec': 'H.265/HEVC',     'max_px': 3840,
                  'quality_range': (78, 90),  'prnu_reliability': 0.80,
                  'description': 'Generic H.265/HEVC encoded video frame'},
    'av1':       {'tier': 'high',    'codec': 'AV1',            'max_px': 7680,
                  'quality_range': (85, 98),  'prnu_reliability': 0.90,
                  'description': 'AV1 codec — high efficiency, minimal compression loss'},
    'vp9':       {'tier': 'high',    'codec': 'VP9',            'max_px': 4096,
                  'quality_range': (80, 95),  'prnu_reliability': 0.88,
                  'description': 'VP9 codec — open format, good quality retention'},
    'vp8':       {'tier': 'medium',  'codec': 'VP8',            'max_px': 1920,
                  'quality_range': (65, 80),  'prnu_reliability': 0.68,
                  'description': 'VP8 codec — older format, moderate compression'},
    'webm':      {'tier': 'high',    'codec': 'VP9/AV1',        'max_px': 4096,
                  'quality_range': (82, 96),  'prnu_reliability': 0.88,
                  'description': 'WebM container (VP8/VP9/AV1)'},
    'mp4':       {'tier': 'medium',  'codec': 'H.264/H.265/AV1','max_px': 3840,
                  'quality_range': (72, 90),  'prnu_reliability': 0.78,
                  'description': 'Generic MP4 container (H.264/H.265/AV1)'},
}

# Aliases accepted in the ?platform= query param
_PLATFORM_ALIASES = {
    'x':          'twitter',
    'twitter/x':  'twitter',
    'heif':       'heic',
    'apple':      'heic',
    'mtproto':    'telegram',
    'avc':        'h264',
    'h264/h265':  'h265',
    'hevc':       'h265',
    'mkv':        'webm',
    'matroska':   'webm',
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# Model paths — try unified first, then v6/v5, then legacy
UNIFIED_MODEL_PATH = os.path.join(MODELS_DIR, 'ai_detector_unified_v1.pth')
V6_MODEL_PATH  = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion_v6.pth')
V5_MODEL_PATH  = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion_v5.pth')
V4_MODEL_PATH  = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion.pth')
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

    Args:
        model        : DeepFusionNet (or any model with spatial feature maps)
        img_tensor   : (1, 3, H, W) — normalised image tensor, requires_grad
        prnu_tensor  : (1, dim) or None — PRNU features (None for legacy model)
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

        if prnu_tensor is not None:
            logits = model(img_tensor, prnu_tensor)
        else:
            logits = model(img_tensor)

        # Handle tuple output (shouldn't happen in eval mode, but guard anyway)
        if isinstance(logits, tuple):
            logits = logits[0]

        score = logits[0, 0]
        score.backward()

        acts  = activations['feat']     # (1, C, h, w)
        grads = gradients['grad']       # (1, C, h, w)

        weights = grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * acts).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam     = torch.relu(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((orig_width, orig_height), _BILINEAR)
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
    """
    h_uint8 = (heatmap * 255).astype(np.uint8)
    jet     = _apply_jet_colormap(h_uint8)
    orig    = orig_image_array.astype(np.float32)
    blended = np.clip((1.0 - alpha) * orig + alpha * jet.astype(np.float32), 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(blended)
    buf     = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _apply_jet_colormap(gray_uint8: np.ndarray) -> np.ndarray:
    """Pure-numpy JET colormap. Input: (H,W) uint8. Output: (H,W,3) uint8 RGB."""
    v = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  Internal model definitions (kept for backward compat)
# ═══════════════════════════════════════════════════════════════════════════

class _PyTorchCNN(nn.Module):
    """Legacy single-branch fallback model."""
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


# ═══════════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════════

class Detector:
    """
    Main inference class.

    Load order:
      0. ai_detector_unified_v1.pth    → UnifiedFusionNet v1 (16-branch, 64-dim PRNU)
      1. ai_detector_prnu_fusion_v6.pth → DeepFusionNet v6 (12-branch, 64-dim PRNU)
      2. ai_detector_prnu_fusion_v5.pth → DeepFusionNet v5 (8-branch,  64-dim PRNU)
      3. ai_detector_prnu_fusion.pth    → DeepFusionNet v5 (if new arch keys)
                                        → falls through if old v4 keys
      4. ai_detector_model_pytorch.pth  → legacy PyTorchCNN fallback
    """

    def __init__(self):
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fusion  = False
        self.prnu_dim    = 64
        self._model_type = 'unknown'
        self._image_loader = UniversalImageLoader()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # --- Load PRNU recovery net ---
        try:
            self.recovery_net = build_prnu_recovery_net(device=self.device)
        except Exception as e:
            print(f"  PRNU recovery net unavailable: {e}")
            self.recovery_net = None

        # --- Format analyzer (with restoration enabled for inference) ---
        try:
            self.format_analyzer = FormatAnalyzer(
                recovery_net=self.recovery_net,
                enable_restoration=True,
            )
        except Exception as e:
            print(f"  FormatAnalyzer unavailable: {e}")
            self.format_analyzer = None

        # --- GPU PRNU extractor (matches training path for consistency) ---
        self.prnu_gpu = None
        if self.device.type == 'cuda':
            try:
                self.prnu_gpu = PRNUExtractorGPU(self.device, map_size=128)
                self.prnu_gpu.eval()
                print("  PRNUExtractorGPU: active (GPU extraction consistent with training)")
            except Exception as _e:
                print(f"  PRNUExtractorGPU: unavailable ({_e}) — using CPU extraction")

        # --- Step 0: try UnifiedFusionNet v1 ---
        if os.path.exists(UNIFIED_MODEL_PATH):
            try:
                model = UnifiedFusionNet(prnu_in_features=64).to(self.device)
                sd = torch.load(UNIFIED_MODEL_PATH, map_location=self.device,
                                weights_only=True)
                check_checkpoint_compat(model, sd.get('model_state_dict', sd))
                model.load_state_dict(sd.get('model_state_dict', sd), strict=False)
                model.eval()
                self.model       = model
                self.prnu_dim    = 64
                self._model_type = 'UnifiedFusionNet v1'
                self.use_fusion  = True
                print(f"  Loaded UnifiedFusionNet v1 from {UNIFIED_MODEL_PATH}")
                # Grad-CAM target layer
                self.gradcam_layer = self.model.gradcam_target_layer
                return
            except Exception as e:
                print(f"  UnifiedFusionNet load failed: {e} — falling back")

        # --- Try DeepFusionNet v6/v5 ---
        for model_path in (V6_MODEL_PATH, V5_MODEL_PATH, V4_MODEL_PATH):
            if not os.path.exists(model_path):
                continue
            loaded, prnu_dim, model_type = self._try_load_fusion(model_path)
            if loaded is not None:
                self.model       = loaded
                self.prnu_dim    = prnu_dim
                self._model_type = model_type
                self.use_fusion  = True
                print(f"  Loaded {model_type} from {model_path}")
                break

        # --- Fallback to legacy model ---
        if not self.use_fusion:
            if not os.path.exists(OLD_MODEL_PATH):
                raise FileNotFoundError(
                    f"No model file found.\n"
                    f"  Tried: {V5_MODEL_PATH}\n"
                    f"  Tried: {V4_MODEL_PATH}\n"
                    f"  Tried: {OLD_MODEL_PATH}\n"
                    f"Please train the model first."
                )
            self.model = _PyTorchCNN(num_classes=1).to(self.device)
            state = torch.load(OLD_MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self._model_type = 'PyTorchCNN (legacy)'
            print(f"  Loaded legacy PyTorchCNN from {OLD_MODEL_PATH}")

        # --- Grad-CAM target layer ---
        if self.use_fusion:
            self.gradcam_layer = self.model.gradcam_target_layer
        else:
            self.gradcam_layer = self.model.features[-3]

    # ------------------------------------------------------------------

    def predict(self, image_data: bytes, compute_heatmap: bool = True,
                platform: str = None) -> dict:
        """
        Analyse raw image bytes.

        Args:
            image_data      : raw file bytes (JPEG, PNG, HEIC, RAW, …)
            compute_heatmap : if True, run Grad-CAM and include heatmap_base64
            platform        : optional source platform hint — one of
                              'youtube', 'vimeo', 'tiktok', 'instagram',
                              'facebook', 'twitter', 'snapchat'
                              Adds platform_compression info to the result and
                              scales PRNU reliability score accordingly.

        Returns:
            dict with keys:
                ai_probability, conclusion, prnu_analysis, model_type,
                platform_compression (if platform given),
                heatmap_base64 (optional), heatmap_width, heatmap_height
        """
        try:
            img = self._image_loader.load(image_data)
        except Exception as e:
            return {"error": f"Could not load image: {e}"}

        orig_w, orig_h = img.size
        orig_array = np.array(
            img.resize((min(orig_w, 1024), min(orig_h, 1024)), _BILINEAR)
        )

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # --- PRNU features + spatial map ---
        # Use GPU extractor when available — matches training pipeline exactly.
        # Fall back to CPU (wavelet) path if GPU extractor is absent/fails.
        prnu_tensor     = None
        prnu_map_tensor = None

        if self.prnu_gpu is not None:
            try:
                _mean  = torch.tensor([0.485, 0.456, 0.406],
                                      device=self.device).view(1, 3, 1, 1)
                _std   = torch.tensor([0.229, 0.224, 0.225],
                                      device=self.device).view(1, 3, 1, 1)
                imgs_01 = (img_tensor * _std + _mean).clamp(0, 1)
                with torch.no_grad():
                    _feats, _map = self.prnu_gpu.extract_both(imgs_01)   # (1,64), (1,3,128,128)
                prnu_tensor     = _feats.float()
                prnu_map_tensor = _map.float()
            except Exception as _pe:
                print(f"  GPU PRNU extraction failed ({_pe}) — falling back to CPU")

        if prnu_tensor is None:
            # CPU wavelet fallback
            prnu_feats_np = extract_prnu_features_fullres(
                img,
                recovery_net=self.recovery_net,
                device=self.device,
            )
            if len(prnu_feats_np) != self.prnu_dim:
                tmp = np.zeros(self.prnu_dim, dtype=np.float32)
                n   = min(len(prnu_feats_np), self.prnu_dim)
                tmp[:n] = prnu_feats_np[:n]
                prnu_feats_np = tmp
            prnu_tensor = torch.from_numpy(prnu_feats_np).unsqueeze(0).float().to(self.device)

        if prnu_map_tensor is None and self.use_fusion:
            try:
                noise_map = extract_prnu_map(img, output_size=128)     # (128,128,3) [-1,1]
                noise_map = noise_map.transpose(2, 0, 1)               # (3,128,128)
                prnu_map_tensor = torch.from_numpy(noise_map).unsqueeze(0).to(self.device)
            except Exception as e:
                print(f"  PRNU spatial map extraction failed: {e}")

        # --- Format forensics features ---
        format_tensor     = None
        format_forensics  = {"extraction_ok": False}
        if self.format_analyzer is not None:
            try:
                ff = self.format_analyzer.analyze(image_data)
                format_forensics = {
                    "format":        ff.fmt_name,
                    "media_type":    ff.media_type,
                    "extraction_ok": ff.extraction_ok,
                }
                format_tensor = torch.from_numpy(ff.feature_vector).unsqueeze(0).float().to(self.device)
            except Exception as _fe:
                print(f"  Format analysis failed: {_fe}")

        # --- Inference ---
        with torch.no_grad():
            if self.use_fusion:
                # Only UnifiedFusionNet accepts format_feats; DeepFusionNet does not
                if isinstance(self.model, UnifiedFusionNet) and format_tensor is not None:
                    out = self.model(img_tensor, prnu_tensor, prnu_map_tensor,
                                     format_feats=format_tensor)
                else:
                    out = self.model(img_tensor, prnu_tensor, prnu_map_tensor)
                logits = out[0] if isinstance(out, tuple) else out
                ai_prob = float(torch.sigmoid(logits).item())
            else:
                ai_prob_cnn  = float(self.model(img_tensor).item())
                prnu_result  = analyze_prnu(image_data)
                prnu_ai_score = 1.0 - prnu_result.get("prnu_likelihood_real", 0.5)
                ai_prob = 0.7 * ai_prob_cnn + 0.3 * prnu_ai_score

        # --- PRNU detailed analysis ---
        try:
            prnu_analysis = analyze_prnu(image_data)
        except Exception:
            prnu_analysis = {}

        # --- Platform compression annotation ---
        platform_info = None
        if platform:
            key     = platform.lower()
            key     = _PLATFORM_ALIASES.get(key, key)
            profile = PLATFORM_PROFILES.get(key)
            if profile:
                platform_info = {
                    "platform":         key,
                    "tier":             profile['tier'],
                    "codec":            profile['codec'],
                    "quality_range":    profile['quality_range'],
                    "max_resolution":   profile['max_px'],
                    "prnu_reliability": profile['prnu_reliability'],
                    "description":      profile['description'],
                    "note": (
                        "PRNU signal heavily degraded by platform compression — "
                        "visual branch results are more reliable."
                        if profile['prnu_reliability'] < 0.6 else
                        "Moderate compression — PRNU signal partially preserved."
                        if profile['prnu_reliability'] < 0.85 else
                        "Light compression — PRNU signal well preserved."
                    ),
                }

        result = {
            "ai_probability":      round(ai_prob, 4),
            "conclusion":          "AI-Generated" if ai_prob > 0.5 else "REAL",
            "prnu_analysis":       prnu_analysis,
            "model_type":          self._model_type,
            "platform_compression": platform_info,
            "format_forensics":    format_forensics,
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
                result["heatmap_base64"] = heatmap_to_base64_png(heatmap, orig_array)
                result["heatmap_width"]  = orig_array.shape[1]
                result["heatmap_height"] = orig_array.shape[0]
            except Exception as e:
                print(f"  Grad-CAM failed: {e}")
                result["heatmap_base64"] = None
                result["heatmap_width"]  = orig_w
                result["heatmap_height"] = orig_h

        return result

    # ------------------------------------------------------------------

    def predict_video(
        self,
        video_path: str,
        n_screen_frames: int = 8,
        n_full_frames: int = 24,
        screen_threshold: float = 0.35,
        platform: str = None,
    ) -> dict:
        """
        Cascade video AI detection.

        Stage 1 — Quick PRNU screen on n_screen_frames random frames.
                   Uses fast 8-dim PRNU features only.
                   If the AI score is below screen_threshold → return early as REAL.

        Stage 2 — Full frame-by-frame analysis.
                   Runs the image model on n_full_frames + measures PRNU
                   temporal consistency across frames.

        Args:
            video_path       : path to local video file
            n_screen_frames  : frames sampled in Stage 1
            n_full_frames    : frames analysed in Stage 2
            screen_threshold : Stage 1 AI score above which Stage 2 runs (0–1)
            platform         : optional platform hint (same as predict())

        Returns dict with keys:
            ai_probability, conclusion, method, frames_analyzed,
            stage1_score, frame_model_probability,
            prnu_temporal_consistency, frame_scores, model_type,
            platform_compression (if platform given)
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return {"error": "Video has no readable frames."}

        # ── Stage 1: Quick PRNU screen ────────────────────────────────
        n_s1 = min(n_screen_frames, total_frames)
        screen_indices = sorted(random.sample(range(total_frames), n_s1))
        stage1_scores = []

        for fi in screen_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            try:
                small = cv2.resize(frame, (256, 256))
                arr   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                feats = extract_prnu_features(arr)          # 8-dim fast
                # feats[0] noise_strength  — real cameras: high
                # feats[1] noise_uniformity — AI: high (flat noise)
                # feats[3] freq_energy_ratio — real: high HF content
                noise_strength  = float(np.clip(feats[0] * 10.0, 0.0, 1.0))
                noise_uniformity = float(feats[1])
                freq_ratio      = float(feats[3])
                # Low noise strength, uniform distribution, missing HF = AI-like
                score = (
                    noise_uniformity          * 0.50
                    + (1.0 - noise_strength)  * 0.30
                    + (1.0 - freq_ratio)      * 0.20
                )
                stage1_scores.append(float(np.clip(score, 0.0, 1.0)))
            except Exception:
                pass

        stage1_ai_score = float(np.mean(stage1_scores)) if stage1_scores else 0.5

        if stage1_ai_score < screen_threshold:
            cap.release()
            return {
                "ai_probability":  round(stage1_ai_score, 4),
                "conclusion":      "REAL",
                "method":          "cascade_stage1_prnu_screen",
                "frames_analyzed": len(stage1_scores),
                "stage1_score":    round(stage1_ai_score, 4),
                "model_type":      self._model_type,
            }

        # ── Stage 2: Full frame-by-frame analysis ─────────────────────
        n_s2 = min(n_full_frames, total_frames)
        full_indices = sorted(random.sample(range(total_frames), n_s2))
        frame_scores  = []
        prnu_vectors  = []

        for fi in full_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            try:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Run image model on this frame
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG', quality=95)
                frame_result = self.predict(
                    buf.getvalue(), compute_heatmap=False, platform=platform
                )
                if "ai_probability" in frame_result:
                    frame_scores.append(frame_result["ai_probability"])

                # Extract 64-dim PRNU for temporal consistency check
                try:
                    pv = extract_prnu_features_fullres(
                        pil_img, tile_size=256,
                        recovery_net=None, device=self.device,
                    )
                    prnu_vectors.append(pv)
                except Exception:
                    pass
            except Exception:
                pass

        cap.release()

        if not frame_scores:
            return {"error": "No frames could be analysed in Stage 2."}

        # PRNU temporal consistency — real cameras have a stable sensor
        # fingerprint across all frames; AI video lacks this.
        prnu_consistency = 0.5          # neutral default
        if len(prnu_vectors) >= 2:
            vecs    = np.stack(prnu_vectors)                     # (N, 64)
            norms_a = np.linalg.norm(vecs[:-1], axis=1, keepdims=True) + 1e-8
            norms_b = np.linalg.norm(vecs[1:],  axis=1, keepdims=True) + 1e-8
            cos_sim = np.sum(
                (vecs[:-1] / norms_a) * (vecs[1:] / norms_b), axis=1
            )
            prnu_consistency = float(np.clip(np.mean(cos_sim), 0.0, 1.0))

        frame_ai_prob  = float(np.mean(frame_scores))
        prnu_ai_score  = 1.0 - prnu_consistency   # low consistency = more AI-like
        combined       = 0.70 * frame_ai_prob + 0.30 * prnu_ai_score

        # Platform annotation
        platform_info = None
        if platform:
            key     = _PLATFORM_ALIASES.get(platform.lower(), platform.lower())
            profile = PLATFORM_PROFILES.get(key)
            if profile:
                platform_info = {
                    "platform":         key,
                    "tier":             profile['tier'],
                    "prnu_reliability": profile['prnu_reliability'],
                    "description":      profile['description'],
                }

        return {
            "ai_probability":            round(combined, 4),
            "conclusion":                "AI-Generated" if combined > 0.5 else "REAL",
            "method":                    "cascade_stage2_full",
            "frames_analyzed":           len(frame_scores),
            "stage1_score":              round(stage1_ai_score, 4),
            "frame_model_probability":   round(frame_ai_prob, 4),
            "prnu_temporal_consistency": round(prnu_consistency, 4),
            "frame_scores":              [round(s, 4) for s in frame_scores],
            "model_type":                self._model_type,
            "platform_compression":      platform_info,
        }

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _try_load_fusion(self, path: str):
        """
        Attempt to load a fusion model from `path`.

        Peeks at the state dict to determine version:
          - v6 has 'gan_diff_branch.spec_conv.0.weight'
          - v5 has 'prnu_branch_v2.mlp.0.weight'  (no gan_diff_branch)
          - v4 has 'prnu_branch.mlp.0.weight' with shape (128, 16)

        Returns (model, prnu_dim, model_type) or (None, None, None).
        """
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            try:
                state = torch.load(path, map_location=self.device)
            except Exception as e:
                print(f"  Could not load state dict from {path}: {e}")
                return None, None, None

        # ── Detect architecture version ──────────────────────────────────
        if 'gan_diff_branch.spec_conv.0.weight' in state:
            # v6 architecture: 12 branches
            try:
                model = DeepFusionNet(prnu_in_features=64).to(self.device)
                missing, unexpected = model.load_state_dict(state, strict=False)
                if unexpected:
                    print(f"  v6 load: {len(unexpected)} unexpected keys (ignored)")
                model.eval()
                return model, 64, 'DeepFusionNet v6'
            except Exception as e:
                print(f"  Failed to load v6 model from {path}: {e}")
                return None, None, None

        elif 'prnu_branch_v2.mlp.0.weight' in state:
            # v5 architecture: 8 branches — load into v6 with strict=False
            # (new v6 branches will be randomly initialised)
            try:
                model = DeepFusionNet(prnu_in_features=64).to(self.device)
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    print(f"  v5→v6 upgrade: {len(missing)} new-branch weights "
                          "initialised randomly")
                model.eval()
                return model, 64, 'DeepFusionNet v5→v6 (new branches randomly init)'
            except Exception as e:
                print(f"  Failed to upgrade v5 model from {path}: {e}")
                return None, None, None

        elif 'prnu_branch.mlp.0.weight' in state:
            # Old v4 architecture — incompatible
            print(f"  {path}: detected v4 checkpoint (16-dim PRNU) — "
                  "incompatible with v6 architecture, skipping")
            return None, None, None

        else:
            # Unknown — try v6 anyway
            try:
                model = DeepFusionNet(prnu_in_features=64).to(self.device)
                model.load_state_dict(state, strict=False)
                model.eval()
                return model, 64, 'DeepFusionNet v6 (unknown checkpoint)'
            except Exception as e:
                print(f"  Failed to load unknown-version model from {path}: {e}")
                return None, None, None


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
            print(f"  Heatmap        : {len(result['heatmap_base64'])} bytes (base64 PNG)")
        print(f"  PRNU Analysis  : {result.get('prnu_analysis', {})}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
