"""
train_streaming.py — Streaming Download-Train-Delete Trainer (v6)
=================================================================

Trains DeepFusionNet v6 on real camera images (Open Images / RAISE / Dresden)
without permanently storing them.  Each iteration:

  1. Download `batch_real` images from a URL list into a temp dir
  2. Apply CompressionAugment (platform-specific: YouTube/TikTok/Instagram/
     Facebook/Twitter/Vimeo/Snapchat or no compression)
  3. Mix with `batch_ai` local AI images → StreamBatchDataset
  4. Train DeepFusionNet v6 for `inner_epochs` epochs on this batch
     (loss = BCE + aux losses for all 7 branches)
  5. Evaluate on 10% held-out validation split per batch
  6. Delete the downloaded temp dir
  7. Save checkpoint + progress → safe to interrupt and resume
  8. Every 5 batches: train PRNU recovery net one step
  9. Every 50 batches: save prnu_recovery.pth

Storage at any point: only one batch (~50–200 MB) on disk.

Supported real-image sources
─────────────────────────────
• Open Images v7 validation (default) — ~40k original Flickr JPEGs
• Open Images v7 train               — ~1.7M original Flickr JPEGs
• Custom URL file                    — one URL per line (RAISE, Dresden, etc.)
"""

import argparse
import gc
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFilter
_BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR  # Pillow 10+ compat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Project imports
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR  = os.path.join(SCRIPT_DIR, '..', 'models')

sys.path.insert(0, SCRIPT_DIR)
from model_prnu import UnifiedFusionNet, DeepFusionNet, EfficientFusionNet, check_checkpoint_compat
from prnu_features import extract_prnu_features_fullres, extract_prnu_map
from prnu_cuda import PRNUExtractorGPU
from prnu_recovery import (
    build_prnu_recovery_net, PRNURecoveryNet, train_recovery_net_one_step,
)
from live_plot import LivePlot
from format_analyzer import FormatAnalyzer, get_shared_analyzer

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

PRNU_FEATURE_DIM  = 64      # 32 in v5; 64 in v6
IMG_SIZE          = 512     # T4 (14.6 GB VRAM) can handle 512 with batch=8 comfortably
BATCH_REAL        = 32
BATCH_AI          = 32
INNER_EPOCHS      = 1
GRAD_ACCUM_STEPS  = 8
LOADER_BATCH_SIZE = 32      # 4× more GPU work per step — T4 VRAM easily handles this
PRNU_TILE_SIZE    = 256     # reduced from 1024 — ~16x faster PRNU extraction

FORMAT_FEATURE_DIM   = 128    # FormatForensicsBranch input dim

LAMBDA_PRNU          = 0.35   # PRNU scalar features aux loss weight
LAMBDA_HALLUC        = 0.15   # Hallucination aux loss weight
LAMBDA_PRNU_SPATIAL  = 0.20   # PRNUSpatialBranch aux loss weight (PRNU map pattern)
LAMBDA_GAN_DIFF      = 0.25   # GAN/Diffusion fingerprint aux loss weight
LAMBDA_CMOS          = 0.20   # CMOS/CCD sensor noise aux loss weight
LAMBDA_COLOR_INCON   = 0.20   # Color channel inconsistency aux loss weight
LAMBDA_FLOW          = 0.15   # Optical flow irregularity aux loss weight
LAMBDA_MOTION        = 0.15   # MotionGRU temporal branch aux loss weight
LAMBDA_FORMAT        = 0.20   # FormatForensicsBranch aux loss weight

# Open Images metadata CSVs — primary (2018_04 with rotation data) + v7 fallback
OI_VAL_CSV_URL   = ("https://storage.googleapis.com/openimages/2018_04/"
                    "validation/validation-images-with-rotation.csv")
OI_TRAIN_CSV_URL = ("https://storage.googleapis.com/openimages/2018_04/"
                    "train/train-images-boxable-with-rotation.csv")

# Fallback URLs (Open Images v7 — no rotation data but more stable)
OI_VAL_CSV_URL_V7   = ("https://storage.googleapis.com/openimages/v7/"
                        "oidv7-image-ids-boxable-validation.csv")
OI_TRAIN_CSV_URL_V7 = ("https://storage.googleapis.com/openimages/v7/"
                        "oidv7-image-ids-boxable.csv")

PROGRESS_FILE      = os.path.join(MODELS_DIR, 'stream_progress.json')
FUSION_MODEL_PATH  = os.path.join(MODELS_DIR, 'ai_detector_unified_v1.pth')   # unified v1 (primary)
FUSION_MODEL_LEGACY = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion_v5.pth')  # fallback
CKPT_PATH          = os.path.join(MODELS_DIR, 'checkpoint_latest.pth')
RECOVERY_PATH      = os.path.join(MODELS_DIR, 'prnu_recovery.pth')
CACHE_DIR          = os.path.join(DATA_DIR, 'stream_cache')

AI_SOURCE_DIRS = [
    os.path.join(DATA_DIR, 'train', 'FAKE'),
    os.path.join(DATA_DIR, 'ai'),
    os.path.join(DATA_DIR, 'generated_ai', 'images'),
    os.path.join(DATA_DIR, 'edited_ai', 'images'),
    os.path.join(DATA_DIR, 'multi_cache', 'kaggle', 'cifake', 'train', 'FAKE'),
    os.path.join(DATA_DIR, 'multi_cache', 'kaggle', 'fake-faces',
                 'real_vs_fake', 'real-vs-fake', 'train', 'fake'),
    os.path.join(DATA_DIR, 'pipeline_frames', 'ai'),
]

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


# ═══════════════════════════════════════════════════════════════════════════
#  Resource Safety Guard
# ═══════════════════════════════════════════════════════════════════════════

class ResourceGuard:
    MIN_DISK_MB  = 500
    MIN_RAM_MB   = 400
    MAX_RAM_PCT  = 92
    MAX_GPU_PCT  = 95

    @staticmethod
    def _disk_free_mb(path='.'):
        try:
            return shutil.disk_usage(os.path.abspath(path)).free / 1024**2
        except Exception:
            return float('inf')

    @staticmethod
    def _ram_free_mb():
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return float('inf')

    @staticmethod
    def _ram_pct():
        try:
            info = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    p = line.split()
                    info[p[0].rstrip(':')] = int(p[1])
            total = info.get('MemTotal', 1)
            avail = info.get('MemAvailable', total)
            return (total - avail) / total * 100
        except Exception:
            return 0

    @staticmethod
    def _gpu_pct():
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return (alloc / total) * 100 if total else 0
        except Exception:
            pass
        return 0

    @classmethod
    def check(cls):
        if cls._disk_free_mb(MODELS_DIR) < cls.MIN_DISK_MB:
            return False, f"Disk critically low: {cls._disk_free_mb(MODELS_DIR):.0f} MB"
        if cls._ram_free_mb() < cls.MIN_RAM_MB:
            return False, f"RAM critically low: {cls._ram_free_mb():.0f} MB"
        if cls._ram_pct() > cls.MAX_RAM_PCT:
            return False, f"RAM usage too high: {cls._ram_pct():.0f}%"
        if cls._gpu_pct() > cls.MAX_GPU_PCT:
            return False, f"GPU VRAM too high: {cls._gpu_pct():.0f}%"
        return True, ''

    @classmethod
    def check_or_abort(cls, model, context=''):
        safe, reason = cls.check()
        if not safe:
            print(f"\n  EMERGENCY STOP: {reason}  [{context}]")
            print("       Saving checkpoint before exit ...")
            _save_checkpoint(model, None, 0, 0.0, 0.0)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  Streaming training stopped safely.  Re-run to resume.")
            sys.exit(1)
        return True


# ═══════════════════════════════════════════════════════════════════════════
#  Compression Augmentation
# ═══════════════════════════════════════════════════════════════════════════

class CompressionAugment:
    """
    Simulate platform-specific compression applied BEFORE PRNU extraction,
    so the model learns to detect AI content regardless of where it was shared.

    Platform profiles (weighted by estimated share of real-world training content):
      YouTube    16% — H.264/AV1, high bitrate, minimal loss         Q 85-95
      TikTok     15% — H.265, 1080p max, moderate compression        Q 70-82
      Instagram  15% — H.264, 1080p max, aggressive re-encode        Q 60-75
      Facebook   11% — H.264, 960px max, heavy + double-compress     Q 55-72
      Twitter/X   7% — H.264, 900px max, very heavy compression      Q 50-68
      Telegram    9% — MTProto/JPEG, 1280px max                      Q 78-85
      HEIC/HEIF   6% — Apple HEVC container, near-lossless           Q 85-92
      H.264       5% — generic H.264/AVC video frame                 Q 72-88
      H.265       4% — generic H.265/HEVC video frame                Q 78-90
      Vimeo       4% — near-lossless, minimal compression            Q 90-98
      Snapchat    4% — 720p max, lowest quality, speed-optimised     Q 40-60
      None        4% — no compression (preserve original PRNU)
    """

    _PLATFORMS = ['youtube', 'tiktok', 'instagram', 'facebook', 'twitter',
                  'telegram', 'heic', 'h264', 'h265', 'vimeo', 'snapchat', None]
    _WEIGHTS   = [16, 15, 15, 11, 7, 9, 6, 5, 4, 4, 4, 4]   # must sum to 100

    def __call__(self, img: Image.Image) -> Image.Image:
        platform = random.choices(self._PLATFORMS, weights=self._WEIGHTS, k=1)[0]
        if platform is None:
            return img
        return getattr(self, f'_{platform}')(img)

    # ── Per-platform methods ──────────────────────────────────────────────

    def _youtube(self, img: Image.Image) -> Image.Image:
        """H.264 / AV1, high bitrate — minimal loss, gentle deblocking filter."""
        img = self._jpeg(img, quality=random.randint(85, 95))
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.4)))

    def _vimeo(self, img: Image.Image) -> Image.Image:
        """Near-lossless — very light JPEG or WebP at high quality."""
        if random.random() < 0.5:
            return self._jpeg(img, quality=random.randint(90, 98))
        return self._webp(img, quality=random.randint(88, 95))

    def _tiktok(self, img: Image.Image) -> Image.Image:
        """H.265, 1080p max — moderate compression, mobile-optimised."""
        img = self._resize_max(img, 1080)
        img = self._jpeg(img, quality=random.randint(70, 82))
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    def _instagram(self, img: Image.Image) -> Image.Image:
        """H.264, 1080px max — noticeably aggressive re-encode."""
        img = self._resize_max(img, 1080)
        img = self._jpeg(img, quality=random.randint(60, 75))
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))

    def _facebook(self, img: Image.Image) -> Image.Image:
        """H.264, 960px max — heavy + double-compressed (upload + CDN re-encode)."""
        img  = self._resize_max(img, 960)
        img  = self._jpeg(img, quality=random.randint(70, 85))   # upload
        img  = self._jpeg(img, quality=random.randint(55, 72))   # CDN re-encode
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.0)))

    def _twitter(self, img: Image.Image) -> Image.Image:
        """H.264, 900px max — among the worst quality; double-compress 40% of the time."""
        img = self._resize_max(img, 900)
        if random.random() < 0.4:
            img = self._jpeg(img, quality=random.randint(65, 80))
            img = self._jpeg(img, quality=random.randint(50, 68))
        else:
            img = self._jpeg(img, quality=random.randint(50, 68))
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.1)))

    def _snapchat(self, img: Image.Image) -> Image.Image:
        """720p max — lowest quality, heavily optimised for speed over fidelity."""
        img = self._resize_max(img, 720)
        img = self._jpeg(img, quality=random.randint(40, 60))
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.5)))

    def _telegram(self, img: Image.Image) -> Image.Image:
        """MTProto — Telegram photo send: JPEG Q78-85, 1280px max.
        Forwarded/re-shared messages occasionally double-compress.
        No deblocking blur — server-side JPEG only, no video codec pipeline."""
        img = self._resize_max(img, 1280)
        if random.random() < 0.25:
            # Forwarded message: original upload re-encoded on delivery
            img = self._jpeg(img, quality=random.randint(82, 88))
            img = self._jpeg(img, quality=random.randint(78, 85))
        else:
            img = self._jpeg(img, quality=random.randint(78, 85))
        return img

    def _heic(self, img: Image.Image) -> Image.Image:
        """HEIC/HEIF — Apple iOS camera format, HEVC-based near-lossless container.
        Simulated with high-quality WebP (closest artifact structure to HEVC intra)."""
        return self._webp(img, quality=random.randint(85, 92))

    def _h264(self, img: Image.Image) -> Image.Image:
        """Generic H.264/AVC video frame — DCT block codec with deblocking filter."""
        img = self._jpeg(img, quality=random.randint(72, 88))
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    def _h265(self, img: Image.Image) -> Image.Image:
        """Generic H.265/HEVC video frame — more efficient than H.264, less blocking."""
        img = self._jpeg(img, quality=random.randint(78, 90))
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    # ── Shared helpers ────────────────────────────────────────────────────

    @staticmethod
    def _jpeg(img: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    @staticmethod
    def _webp(img: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    @staticmethod
    def _resize_max(img: Image.Image, max_px: int) -> Image.Image:
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), _BILINEAR)
        return img


# ═══════════════════════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════════════════════

def _make_real_transform():
    """
    Minimal augmentation for real camera images.

    Real PRNU fingerprints are authentic and consistent — we must not corrupt
    them with compression or heavy colour shifts.  Only mild spatial transforms
    are applied so the model learns real content as it truly is.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _make_ai_transform():
    """
    Aggressive visual augmentation for AI-generated images.

    AI content has no authentic PRNU fingerprint and appears in countless
    format/platform/generator combinations.  Heavy augmentation forces the
    model to rely on deep forensic cues rather than superficial style.

    NOTE: PRNU feature extraction happens BEFORE this transform inside
    StreamBatchDataset.__getitem__, so noise signals are NOT affected.

    Augmentations that scramble visual identity:
      • Strong blur (kernel 3–9 px, 40% chance) — destroys sharp edges
      • Heavy colour jitter — scrambles colour palette
      • Random grayscale (20%) — forces model to work without colour
      • Gaussian noise injection (30%) — adds synthetic noise on top
    """
    return transforms.Compose([
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Strong colour scrambling — model cannot cheat on palette/style
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        # Sometimes convert to grayscale so colour is not a shortcut
        transforms.RandomGrayscale(p=0.20),
        transforms.RandomRotation(20),
        # Blur with 40% probability — destroys visual sharpness, keeps noise
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0))], p=0.40
        ),
        transforms.ToTensor(),
        # Additive Gaussian noise (30% chance) — synthetic noise added AFTER
        # PRNU extraction so it does not corrupt the stored PRNU features,
        # but the CNN backbone must not rely on absence-of-noise as a cue
        transforms.RandomApply(
            [transforms.Lambda(
                lambda t: t + torch.randn_like(t) * random.uniform(0.01, 0.04)
            )],
            p=0.30,
        ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _make_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

_COMPRESSION_AUG = CompressionAugment()


class StreamBatchDataset(Dataset):
    """
    Ephemeral dataset for one streaming batch.

    real_paths  — list of paths, label 0 (real camera images)
    ai_paths    — list of paths, label 1 (AI-generated images)

    Per-item pipeline (asymmetric by label):

    REAL (label=0):
      1. Load PIL image
      2. Extract 64-dim PRNU features — NO compression, clean signal
      3. Extract 128×128 PRNU spatial map
      4. Apply real_transform (mild crop + flip only)

    AI (label=1):
      1. Load PIL image
      2. Apply CompressionAugment (JPEG/WebP/WhatsApp simulation)
      3. Extract 64-dim PRNU features (with optional recovery_net)
      4. Extract 128×128 PRNU spatial map
      5. Apply ai_transform (aggressive augmentation)

    Returns 4-item tuple: (img_t, prnu_64_t, prnu_map_t, label_t)
    GPU feature extraction (PRNUExtractorGPU) is handled in the training
    loop to avoid device issues in DataLoader workers.
    """

    def __init__(self, real_paths, ai_paths, real_transform, ai_transform,
                 recovery_net=None, device=None, prnu_extractor=None):
        self.paths           = [(p, 0) for p in real_paths] + [(p, 1) for p in ai_paths]
        self.real_transform  = real_transform
        self.ai_transform    = ai_transform
        self.recovery_net    = recovery_net
        self.device          = device
        self.prnu_extractor  = prnu_extractor  # PRNUExtractorGPU or None
        # Format analyzer — enable_restoration=False for training speed
        self._fmt_analyzer   = get_shared_analyzer(enable_restoration=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')

        if label == 1:
            # AI sample: apply compression augmentation before PRNU extraction
            try:
                img = _COMPRESSION_AUG(img)
            except Exception:
                pass

        # If GPU extractor is available, defer PRNU to the training loop
        # (DataLoader workers can't share GPU memory; return zeros as placeholder)
        if self.prnu_extractor is not None:
            prnu       = np.zeros(PRNU_FEATURE_DIM, dtype=np.float32)
            prnu_map_t = torch.zeros(3, 128, 128, dtype=torch.float32)
        else:
            # CPU path — used when no GPU extractor is wired
            try:
                prnu = extract_prnu_features_fullres(
                    img,
                    tile_size=PRNU_TILE_SIZE,
                    recovery_net=self.recovery_net if label == 1 else None,
                    device=self.device,
                )
                if not np.isfinite(prnu).all():
                    prnu = np.zeros(PRNU_FEATURE_DIM, dtype=np.float32)
            except Exception:
                prnu = np.zeros(PRNU_FEATURE_DIM, dtype=np.float32)

            try:
                prnu_map_np = extract_prnu_map(img, output_size=128)
                if not np.isfinite(prnu_map_np).all():
                    prnu_map_np = np.zeros((128, 128, 3), dtype=np.float32)
            except Exception:
                prnu_map_np = np.zeros((128, 128, 3), dtype=np.float32)

            prnu_map_t = torch.from_numpy(prnu_map_np.transpose(2, 0, 1)).float()

        # Format forensics features (128-dim) — fast extraction, no restoration
        try:
            ff = self._fmt_analyzer.analyze(path)
            fmt_feats = ff.feature_vector
            if not np.isfinite(fmt_feats).all():
                fmt_feats = np.zeros(FORMAT_FEATURE_DIM, dtype=np.float32)
        except Exception:
            fmt_feats = np.zeros(FORMAT_FEATURE_DIM, dtype=np.float32)

        # Visual transform for CNN input — asymmetric by label
        transform = self.ai_transform if label == 1 else self.real_transform
        if transform:
            img = transform(img)

        return (
            img,
            torch.from_numpy(prnu).float(),
            prnu_map_t,
            torch.from_numpy(fmt_feats).float(),
            torch.tensor(label, dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  URL list helpers
# ═══════════════════════════════════════════════════════════════════════════

def _download_metadata_csv(source: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname      = f"oi_{source}_meta.csv"
    cache_path = os.path.join(CACHE_DIR, fname)

    if os.path.exists(cache_path):
        print(f"  Metadata CSV already cached: {cache_path}")
        return cache_path

    primary  = OI_TRAIN_CSV_URL   if source == 'train' else OI_VAL_CSV_URL
    fallback = OI_TRAIN_CSV_URL_V7 if source == 'train' else OI_VAL_CSV_URL_V7
    tmp_path = cache_path + '.tmp'
    for label, url in [('2018_04', primary), ('v7 fallback', fallback)]:
        print(f"  Downloading Open Images {source} metadata CSV ({label}) ...")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, cache_path)
            print(f"  Downloaded {os.path.getsize(cache_path)/1024**2:.1f} MB → {cache_path}")
            break
        except Exception as e:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass
            print(f"  [warn] {label} CSV failed ({e}), trying fallback ...")
    else:
        raise RuntimeError("Failed to download Open Images metadata CSV (both URLs failed)")
    return cache_path


def load_url_list(source: str, url_file=None) -> list:
    if url_file:
        if not os.path.exists(url_file):
            raise FileNotFoundError(
                f"URL file not found: '{url_file}'\n"
                f"  Create it with one URL per line, or omit --url_file to stream from Open Images.\n"
                f"  Example: python train_streaming.py --source val --total_batches 500"
            )
        print(f"  Loading URLs from custom file: {url_file}")
        with open(url_file, encoding='utf-8') as f:
            urls = [l.strip() for l in f
                    if l.strip() and not l.strip().startswith('#')
                    and 'REAL_VIDEO_ID' not in l and 'AI_VIDEO_ID' not in l]
        print(f"  → {len(urls):,} URLs loaded")
        if not urls:
            raise ValueError(
                f"No valid URLs found in '{url_file}' (all lines are comments or placeholders).\n"
                f"  Add real URLs to the file or use --source val to stream from Open Images."
            )
        return urls

    csv_path = _download_metadata_csv(source)
    print(f"  Parsing URLs from {csv_path} ...")
    urls = []
    try:
        import csv
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = (row.get('Thumbnail300KURL', '').strip()
                       or row.get('OriginalURL', '').strip())
                if url:
                    urls.append(url)
    except Exception as e:
        raise RuntimeError(f"Failed to parse OI CSV: {e}")
    print(f"  → {len(urls):,} URLs parsed")
    return urls


# ═══════════════════════════════════════════════════════════════════════════
#  Download helpers
# ═══════════════════════════════════════════════════════════════════════════

def _download_image(url: str, dest_path: str, timeout: int = 20):
    tmp = dest_path + '.tmp'
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0 (compatible; AIDetectorTrainer/1.0)'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        Image.open(__import__('io').BytesIO(data)).verify()
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, dest_path)
        return dest_path, True
    except Exception:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass
        return dest_path, False


def download_batch(urls: list, dest_dir: str, max_workers: int = 8,
                   cookies_from_browser: str | None = None) -> list:
    os.makedirs(dest_dir, exist_ok=True)

    # Local file paths — just copy them, no download needed
    local_files = [u for u in urls if os.path.isfile(u)]
    urls = [u for u in urls if not os.path.isfile(u)]

    # Split: video URLs (YouTube/Vimeo/etc.) vs direct image URLs
    video_urls = [u for u in urls if _is_video_url(u)]
    image_urls = [u for u in urls if not _is_video_url(u)]

    ok = []

    # Copy local files directly — no download needed
    for lp in local_files:
        ext = os.path.splitext(lp)[1].lower()
        if ext in _IMG_EXTS:
            dest = os.path.join(dest_dir, f'lf_{abs(hash(lp)):08x}{ext}')
            try:
                shutil.copy2(lp, dest)
                ok.append(dest)
            except Exception:
                pass

    # Download direct image URLs in parallel
    if image_urls:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, url in enumerate(image_urls):
                ext  = os.path.splitext(url.split('?')[0])[-1].lower()
                ext  = ext if ext in _IMG_EXTS else '.jpg'
                dest = os.path.join(dest_dir, f'img_{i:05d}{ext}')
                futures[pool.submit(_download_image, url, dest)] = dest
            for fut in as_completed(futures):
                path, success = fut.result()
                if success:
                    ok.append(path)

    # Download video URLs with yt-dlp and extract frames
    for i, url in enumerate(video_urls):
        frames = _download_video_frames(url, dest_dir,
                                        prefix=f'vid_{i:03d}',
                                        max_frames=32,
                                        cookies_from_browser=cookies_from_browser)
        ok.extend(frames)

    return ok


# ─── Video URL helpers ────────────────────────────────────────────────────────

_VIDEO_URL_PATTERNS = (
    'youtube.com/watch', 'youtu.be/', 'youtube.com/shorts/',
    'youtube.com/playlist', 'youtube.com/channel', 'youtube.com/@',
    'vimeo.com/', 'dailymotion.com/', 'twitch.tv/', 'rumble.com/',
    'tiktok.com/', 'instagram.com/reel', 'instagram.com/p/',
    'twitter.com/', 'x.com/', 'reddit.com/r/',
)

_VIDEO_URL_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv')


def _is_video_url(url: str) -> bool:
    """Return True if `url` points to a video platform or direct video file."""
    u = url.lower()
    if any(p in u for p in _VIDEO_URL_PATTERNS):
        return True
    ext = os.path.splitext(u.split('?')[0])[-1]
    return ext in _VIDEO_URL_EXTS


def _download_video_frames(url: str, dest_dir: str,
                            prefix: str = 'vid',
                            max_frames: int = 32,
                            fps_cap: float = 1.0,
                            cookies_from_browser: str | None = None) -> list:
    """
    Download a video from any yt-dlp-supported URL and extract up to
    `max_frames` evenly-spaced frames as JPEG images.

    Args:
        url        : any URL supported by yt-dlp (YouTube, Vimeo, TikTok, …)
        dest_dir   : directory to save extracted frames
        prefix     : filename prefix for saved frames
        max_frames : max frames to extract per video
        fps_cap    : extract at most this many frames per second of video

    Returns:
        list of saved frame paths
    """
    try:
        import yt_dlp
    except ImportError:
        print("  [info] yt-dlp not installed — installing automatically...")
        import subprocess
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', 'yt-dlp'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            import yt_dlp
        except ImportError:
            print("  [warn] yt-dlp install failed — skipping video URL:", url)
            return []

    os.makedirs(dest_dir, exist_ok=True)
    tmp_video = os.path.join(dest_dir, f'{prefix}_video.mp4')

    # ── 1. Download video with yt-dlp ────────────────────────────────────
    ydl_opts = {
        # Prefer H.264 (avc1) so OpenCV cv2 can decode without hardware AV1 support.
        # Falls back to any non-AV1 mp4, then any mp4, then best available.
        'format': (
            'bestvideo[vcodec^=avc1][ext=mp4][height<=720]+bestaudio[ext=m4a]'
            '/bestvideo[ext=mp4][height<=720][vcodec!^=av01]+bestaudio[ext=m4a]'
            '/best[ext=mp4][height<=720]'
            '/best'
        ),
        'outtmpl': tmp_video,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,        # single video only (use --url_file for playlists)
        'socket_timeout': 30,
        'retries': 3,
        'merge_output_format': 'mp4',
    }
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 60)   # seconds
    except Exception as e:
        print(f"  [warn] yt-dlp failed for {url}: {e}")
        if os.path.exists(tmp_video):
            try: os.remove(tmp_video)
            except Exception: pass
        return []

    # If yt-dlp appended an extension, find the actual file
    if not os.path.exists(tmp_video):
        candidates = [
            f for f in os.listdir(dest_dir)
            if f.startswith(prefix) and f.endswith(('.mp4', '.mkv', '.webm'))
        ]
        if candidates:
            tmp_video = os.path.join(dest_dir, candidates[0])
        else:
            print(f"  [warn] Video file not found after download: {url}")
            return []

    # ── 2. Extract frames evenly spaced across the video ─────────────────
    frames = []
    try:
        import cv2
        cap = cv2.VideoCapture(tmp_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Cannot read frame count")

        # Determine which frame indices to extract
        n_extract = min(max_frames, max(1, int(duration * fps_cap)))
        indices   = [int(total_frames * i / n_extract) for i in range(n_extract)]

        for fi, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            out_path = os.path.join(dest_dir, f'{prefix}_f{fi:04d}.jpg')
            img.save(out_path, 'JPEG', quality=92)
            frames.append(out_path)

        cap.release()
    except Exception as e:
        print(f"  [warn] Frame extraction failed for {url}: {e}")
    finally:
        try: os.remove(tmp_video)
        except Exception: pass

    print(f"  [video] {url[:60]}… → {len(frames)} frames")
    return frames


def load_ai_image_paths() -> list:
    paths = []
    for d in AI_SOURCE_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
                    paths.append(os.path.join(root, fname))
    print(f"  AI image pool: {len(paths):,} images from local directories")
    return paths


# ═══════════════════════════════════════════════════════════════════════════
#  Model helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_or_init_model(device):
    """Load existing UnifiedFusionNet v1 if available, else init fresh."""
    model = UnifiedFusionNet(
        prnu_in_features=PRNU_FEATURE_DIM,
        gradient_checkpointing=True,   # saves ~300-600 MB VRAM during training
    ).to(device)

    # Try each candidate in priority order
    candidates = [FUSION_MODEL_PATH, CKPT_PATH, FUSION_MODEL_LEGACY]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            try:
                raw = torch.load(path, map_location=device, weights_only=True)
            except Exception:
                raw = torch.load(path, map_location=device)
            # Support both raw state-dict and {'model_state_dict': ...} formats
            state = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
            check_checkpoint_compat(model, state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"  Loaded UnifiedFusionNet v1 from {path}  "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
            return model
        except Exception as e:
            print(f"  Could not load {path}: {e}")

    print("  Initialising fresh UnifiedFusionNet v1 (no existing weights found)")
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Progress / checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                state = json.load(f)
            print(f"  Resumed streaming progress: batch {state.get('batches_done', 0)}, "
                  f"url_offset {state.get('url_offset', 0)}")
            return state
        except Exception as e:
            print(f"  Could not load progress file ({e}), starting fresh")
    return {'url_offset': 0, 'batches_done': 0, 'total_real_seen': 0}


def save_progress(state: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp = PROGRESS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def _save_checkpoint(model, optimizer, batches_done, loss, acc):
    os.makedirs(MODELS_DIR, exist_ok=True)
    payload = {
        'batches_done':     batches_done,
        'model_state_dict': model.state_dict(),
        'loss': loss, 'acc': acc,
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, CKPT_PATH)


def _save_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, FUSION_MODEL_PATH)
    print(f"  Model saved → {FUSION_MODEL_PATH}")


def _save_recovery_net(net):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(net.state_dict(), RECOVERY_PATH)
    print(f"  Recovery net saved → {RECOVERY_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop for one batch
# ═══════════════════════════════════════════════════════════════════════════

def _vram_str(device) -> str:
    """Return a compact VRAM usage string, e.g. '1.2/4.0GB 30%'."""
    if device.type != 'cuda':
        return 'CPU'
    try:
        alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        pct   = alloc / total * 100
        return f'{alloc:.1f}/{total:.1f}GB {pct:.0f}%'
    except Exception:
        return 'N/A'


def train_one_batch(model, optimizer, scaler, loader, device,
                    accum_steps=GRAD_ACCUM_STEPS, desc='Train',
                    prnu_gpu=None) -> tuple:
    """
    Run one pass through `loader` with AMP + gradient accumulation.
    Shows a tqdm progress bar with live loss / accuracy / VRAM.
    Returns (avg_loss, accuracy).
    """
    try:
        from tqdm import tqdm
        bar = tqdm(loader, desc=desc, leave=False,
                   ncols=110, unit='step',
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]')
    except ImportError:
        bar = loader   # graceful fallback if tqdm not installed

    use_amp   = (device.type == 'cuda')
    # Label smoothing ε=0.05: real label=0→0.025, AI label=1→0.975
    # Prevents overconfidence, improves generalisation on unseen generators
    _LS_EPS   = 0.05
    _bce_raw  = nn.BCEWithLogitsLoss()
    def criterion(logits, targets):
        smooth = targets * (1 - _LS_EPS) + 0.5 * _LS_EPS
        return _bce_raw(logits, smooth)
    trainable = [p for p in model.parameters() if p.requires_grad]

    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0
    optimizer.zero_grad()

    for i, (imgs, prnu, prnu_map, fmt_feats, labels) in enumerate(bar):
        imgs      = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        prnu      = prnu.to(device, non_blocking=True)
        prnu_map  = prnu_map.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        fmt_feats = fmt_feats.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True).view(-1, 1)

        # GPU PRNU extraction — replaces placeholder zeros from DataLoader workers
        if prnu_gpu is not None:
            try:
                with torch.no_grad():
                    # imgs is ImageNet-normalised; de-normalise to [0,1] for PRNU
                    _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
                    _std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
                    imgs_01 = (imgs * _std + _mean).clamp(0, 1)
                    prnu, prnu_map = prnu_gpu.extract_both(imgs_01)
            except Exception as _pe:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "[train_streaming] GPU PRNU extraction failed at batch %d (%s: %s)"
                    " — using zero placeholders for this batch", i + 1, type(_pe).__name__, _pe)

        try:
            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(imgs, prnu, prnu_map, format_feats=fmt_feats)
                if isinstance(out, tuple):
                    # UnifiedFusionNet returns 10 outputs:
                    # (logit, prnu, halluc, prnu_spatial, gan_diff, cmos, color_incon, flow, motion, format)
                    (logits, prnu_aux, halluc_aux, prnu_spatial_aux,
                     gan_diff_aux, cmos_aux, color_incon_aux, flow_aux,
                     motion_aux, format_aux) = out[:10]
                    loss = (
                        criterion(logits,           labels)
                        + LAMBDA_PRNU         * criterion(prnu_aux,         labels)
                        + LAMBDA_HALLUC       * criterion(halluc_aux,       labels)
                        + LAMBDA_PRNU_SPATIAL * criterion(prnu_spatial_aux, labels)
                        + LAMBDA_GAN_DIFF     * criterion(gan_diff_aux,     labels)
                        + LAMBDA_CMOS         * criterion(cmos_aux,         labels)
                        + LAMBDA_COLOR_INCON  * criterion(color_incon_aux,  labels)
                        + LAMBDA_FLOW         * criterion(flow_aux,         labels)
                        + LAMBDA_MOTION       * criterion(motion_aux,       labels)
                        + LAMBDA_FORMAT       * criterion(format_aux,       labels)
                    ) / accum_steps
                else:
                    logits = out
                    loss   = criterion(logits, labels) / accum_steps
        except Exception as _fwd_exc:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "[train_streaming] forward failed at batch %d (%s: %s) — skipping",
                i + 1, type(_fwd_exc).__name__, _fwd_exc)
            optimizer.zero_grad()
            continue

        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * imgs.size(0)
        preds   = (torch.sigmoid(logits.detach()) > 0.5).float()
        total  += labels.size(0)
        correct += (preds == labels).sum().item()

        # Update progress bar every step
        if hasattr(bar, 'set_postfix'):
            bar.set_postfix(
                loss=f"{running_loss / max(total, 1):.4f}",
                acc=f"{correct / max(total, 1):.3f}",
                vram=_vram_str(device),
                refresh=False,
            )

        del imgs, prnu, prnu_map, fmt_feats, labels, out, logits, loss

    # Flush remaining gradients
    if len(loader) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def eval_one_batch(model, loader, device, prnu_gpu=None) -> tuple:
    """Evaluate on the validation loader. Returns (avg_loss, accuracy)."""
    try:
        from tqdm import tqdm
        bar = tqdm(loader, desc='Val  ', leave=False, ncols=110, unit='step',
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} {postfix}]')
    except ImportError:
        bar = loader

    use_amp   = (device.type == 'cuda')
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for imgs, prnu, prnu_map, fmt_feats, labels in bar:
            imgs      = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            prnu      = prnu.to(device, non_blocking=True)
            prnu_map  = prnu_map.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            fmt_feats = fmt_feats.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True).view(-1, 1)

            # GPU PRNU extraction for val batch
            if prnu_gpu is not None:
                try:
                    _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
                    _std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
                    imgs_01 = (imgs * _std + _mean).clamp(0, 1)
                    prnu, prnu_map = prnu_gpu.extract_both(imgs_01)
                except Exception as _pe:
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "[eval_streaming] GPU PRNU extraction failed (%s: %s)"
                        " — using zero placeholders", type(_pe).__name__, _pe)

            try:
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out    = model(imgs, prnu, prnu_map, format_feats=fmt_feats)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss   = criterion(logits, labels)
            except Exception as _val_exc:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "[eval_streaming] val forward failed (%s: %s) — skipping batch",
                    type(_val_exc).__name__, _val_exc)
                continue

            running_loss += loss.item() * imgs.size(0)
            preds   = (torch.sigmoid(logits) > 0.5).float()
            total  += labels.size(0)
            correct += (preds == labels).sum().item()

            if hasattr(bar, 'set_postfix'):
                bar.set_postfix(
                    loss=f"{running_loss / max(total, 1):.4f}",
                    acc=f"{correct / max(total, 1):.3f}",
                    refresh=False,
                )

    return running_loss / max(total, 1), correct / max(total, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Main streaming loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Streaming Download-Train-Delete Trainer for DeepFusionNet v5"
    )
    parser.add_argument('--source', choices=['val', 'train'], default='val')
    parser.add_argument('--url_file', type=str, default=None,
                        help='Text file with one URL per line. Supports direct image URLs, '
                             'YouTube links, playlists, Vimeo, TikTok, and any yt-dlp source. '
                             'Lines starting with # are comments.')
    parser.add_argument('--ai_url_file', type=str, default=None,
                        help='Same format as --url_file but treated as AI-generated content (label=1). '
                             'Use for YouTube channels of AI art, Midjourney showcases, etc.')
    parser.add_argument('--batch_real',   type=int, default=BATCH_REAL)
    parser.add_argument('--batch_ai',     type=int, default=BATCH_AI)
    parser.add_argument('--inner_epochs', type=int, default=INNER_EPOCHS)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--max_workers',  type=int, default=8)
    parser.add_argument('--num_workers',  type=int, default=0)
    parser.add_argument('--total_batches', type=int, default=500)
    parser.add_argument('--save_every',   type=int, default=10)
    parser.add_argument('--min_real',     type=int, default=8)
    parser.add_argument('--no_resume',    action='store_true')
    parser.add_argument('--plot',         action='store_true', default=True)
    parser.add_argument('--no_plot',      dest='plot', action='store_false')
    parser.add_argument('--no_recovery',  action='store_true',
                        help='Disable PRNU recovery net (faster, less accurate)')
    parser.add_argument('--cookies_from_browser', type=str, default=None,
                        help='Pass browser cookies to yt-dlp for authenticated downloads '
                             '(e.g. "chrome", "firefox", "chromium"). Required for YouTube.')
    parser.add_argument('--pipeline', action='store_true',
                        help='Use pipeline-generated video frames as training data')
    parser.add_argument('--pipeline_generate', action='store_true',
                        help='Run generate_training_pairs.py before extracting frames '
                             '(requires ANTHROPIC_API_KEY + ITXIO_API_KEY)')
    parser.add_argument('--pipeline_source', type=str, default=None,
                        help='Source for pipeline generation (e.g. data/real_urls.txt)')
    parser.add_argument('--pipeline_max_pairs', type=int, default=None,
                        help='Max video pairs to generate')
    parser.add_argument('--pipeline_max_frames', type=int, default=32,
                        help='Frames per pipeline video (default: 32)')
    args = parser.parse_args()

    # Colab/Jupyter guard: forked workers cause CUDA context errors in notebooks
    if args.num_workers > 0 and (
        'google.colab' in sys.modules or 'ipykernel' in sys.modules
    ):
        print("  [Colab/Jupyter detected] Overriding --num_workers to 0")
        args.num_workers = 0

    print("\n=== Streaming Trainer — DeepFusionNet v5 ===\n")

    # ── CUDA / device setup ───────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu    = torch.cuda.get_device_properties(0)
        vram   = gpu.total_memory / 1024 ** 3
        print(f"  Device  : CUDA — {gpu.name}")
        print(f"  VRAM    : {vram:.1f} GB")
        print(f"  cuDNN   : {torch.backends.cudnn.version()}")
        # Enable cuDNN auto-tuner for fixed input sizes
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("  Device  : CPU  (no CUDA GPU detected)")
        print("  WARNING : Training will be very slow without a GPU.")
        print("            Install CUDA + PyTorch CUDA build to use your GPU.")

    use_amp = (device.type == 'cuda')
    print(f"  AMP     : {use_amp}  (fp16 mixed precision)")

    # ── Load URLs ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading URL list ...")
    urls = load_url_list(args.source, args.url_file)
    if not urls:
        print("ERROR: URL list is empty. Cannot train.")
        sys.exit(1)

    # ── Pre-download ALL external source files at startup ────────────────
    # When --url_file is given, download every URL (image + video) once into
    # data/real_url_frames/ at startup.  Training then runs exactly ONE pass
    # over the downloaded files and stops — no looping, no per-batch downloads.
    external_source = False
    if args.url_file:
        real_cache_dir = os.path.join(DATA_DIR, 'real_url_frames')
        os.makedirs(real_cache_dir, exist_ok=True)
        existing = sorted([
            os.path.join(real_cache_dir, f)
            for f in os.listdir(real_cache_dir)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        ])
        if existing:
            print(f"\n[1b/5] {len(existing)} frames already cached in {real_cache_dir}")
            urls = existing
        else:
            print(f"\n[1b/5] Downloading all {len(urls)} URL(s) → {real_cache_dir} ...")
            urls = download_batch(
                urls, real_cache_dir,
                max_workers=args.max_workers,
                cookies_from_browser=args.cookies_from_browser,
            )
            print(f"  → {len(urls)} frames cached")
        if not urls:
            print("ERROR: No frames downloaded from --url_file. Exiting.")
            sys.exit(1)
        # One-pass: train exactly once over all downloaded frames then stop
        one_pass_batches = max(1, -(-len(urls) // args.batch_real))  # ceil
        if args.total_batches != one_pass_batches:
            print(f"  --url_file mode: total_batches = {one_pass_batches} "
                  f"(one pass over {len(urls)} frames, batch_real={args.batch_real})")
            args.total_batches = one_pass_batches
        external_source = True
        random.shuffle(urls)  # shuffle once upfront, never wrap

    # ── Pipeline mode: generate + extract frames ─────────────────────────
    pipeline_ai_paths = []
    if args.pipeline:
        print("\n[1c/5] Pipeline mode ...")

        # Optionally run generation first
        if args.pipeline_generate:
            if not args.pipeline_source:
                print("ERROR: --pipeline_generate requires --pipeline_source")
                sys.exit(1)
            print(f"  Running generate_training_pairs.py (source={args.pipeline_source}) ...")
            from training_data_pipeline import VideoTrainingDataPipeline
            pipeline = VideoTrainingDataPipeline(
                real_source=args.pipeline_source,
                output_dir=os.path.join(DATA_DIR, 'video_pairs'),
                dry_run=False,
            )
            pipeline.run(max_pairs=args.pipeline_max_pairs, skip_existing=True)

        # Extract frames from pipeline videos
        from extract_pipeline_frames import extract_pipeline_frames
        real_frames, ai_frames = extract_pipeline_frames(
            max_frames_per_video=args.pipeline_max_frames,
            force=False,
        )

        if real_frames:
            urls = real_frames
            external_source = True
            one_pass_batches = max(1, -(-len(urls) // args.batch_real))
            if args.total_batches != one_pass_batches:
                print(f"  --pipeline mode: total_batches = {one_pass_batches} "
                      f"(one pass over {len(urls)} frames, batch_real={args.batch_real})")
                args.total_batches = one_pass_batches
            random.shuffle(urls)
            print(f"  Pipeline real frames: {len(real_frames)}")

        if ai_frames:
            pipeline_ai_paths = ai_frames
            print(f"  Pipeline AI frames: {len(ai_frames)}")

        if not real_frames and not ai_frames:
            print("  WARNING: No pipeline frames found. Falling back to default sources.")

    # ── Load AI image pool ────────────────────────────────────────────────
    print("\n[2/5] Scanning local AI image pool ...")
    ai_paths = load_ai_image_paths()
    if pipeline_ai_paths:
        ai_paths.extend(pipeline_ai_paths)
        print(f"  + {len(pipeline_ai_paths)} pipeline AI frames → total: {len(ai_paths)}")

    # Pre-download AI frames from --ai_url_file (YouTube/video/image URLs)
    if args.ai_url_file:
        print(f"  Loading AI URLs from: {args.ai_url_file}")
        with open(args.ai_url_file, encoding='utf-8') as f:
            ai_urls = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"  → {len(ai_urls)} AI URLs. Downloading frames...")
        ai_dl_dir = os.path.join(DATA_DIR, 'ai_url_frames')
        os.makedirs(ai_dl_dir, exist_ok=True)
        downloaded = download_batch(ai_urls, ai_dl_dir, max_workers=args.max_workers,
                                    cookies_from_browser=args.cookies_from_browser)
        ai_paths.extend(downloaded)
        print(f"  → {len(downloaded)} AI frames downloaded → total AI pool: {len(ai_paths)}")

    # ── Load / init model ─────────────────────────────────────────────────
    print("\n[3/5] Loading main model ...")
    model = load_or_init_model(device)
    model.param_summary()

    # channels_last: conv-heavy EfficientNet backbone runs 10-20% faster on CUDA
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("  channels_last memory format: enabled")

    # torch.compile: 20-40% faster forward/backward with no quality loss
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            print("  torch.compile: enabled (reduce-overhead mode)")
        except Exception as _ce:
            print(f"  torch.compile: skipped ({_ce})")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Fused AdamW: single CUDA kernel for param update — ~5-10% faster on GPU
    _fused_ok = device.type == 'cuda'
    try:
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4,
                                fused=_fused_ok)
        if _fused_ok:
            print("  Fused AdamW: enabled")
    except TypeError:
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ── GPU PRNU extractor (CuPy-backed, replaces slow CPU path) ──────────
    prnu_gpu = None
    if device.type == 'cuda':
        try:
            prnu_gpu = PRNUExtractorGPU(device, map_size=128)
            prnu_gpu.eval()
            print("  PRNUExtractorGPU: enabled (GPU PRNU extraction active)")
        except Exception as _e:
            print(f"  PRNUExtractorGPU: failed to init ({_e}) — falling back to CPU")
            prnu_gpu = None
    else:
        print("  PRNUExtractorGPU: skipped (CPU device)")

    # ── Load / init PRNU recovery net ─────────────────────────────────────
    print("\n[4/5] Loading PRNU recovery net ...")
    if args.no_recovery:
        recovery_net      = None
        recovery_optimizer = None
        recovery_scaler   = None
        print("  PRNU recovery net disabled (--no_recovery)")
    else:
        recovery_net = build_prnu_recovery_net(device=device)
        recovery_net.train()
        rec_params         = [p for p in recovery_net.parameters() if p.requires_grad]
        recovery_optimizer = optim.Adam(rec_params, lr=5e-5)
        recovery_scaler    = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ── Load progress ─────────────────────────────────────────────────────
    print("\n[5/5] Loading progress ...")
    progress = {} if args.no_resume else load_progress()
    url_offset      = progress.get('url_offset', 0)
    batches_done    = progress.get('batches_done', 0)
    total_real_seen = progress.get('total_real_seen', 0)

    real_transform = _make_real_transform()
    ai_transform   = _make_ai_transform()
    val_transform  = _make_val_transform()

    random.shuffle(urls)
    url_offset = 0

    print(f"\n  Starting from batch {batches_done + 1} / {args.total_batches}")
    print(f"  PRNU feature dim  : {PRNU_FEATURE_DIM}")
    print(f"  Loss weights      : main=1.0, prnu={LAMBDA_PRNU}, halluc={LAMBDA_HALLUC}, "
          f"prnu_spatial={LAMBDA_PRNU_SPATIAL}, gan_diff={LAMBDA_GAN_DIFF}, "
          f"cmos={LAMBDA_CMOS}, color_incon={LAMBDA_COLOR_INCON}, flow={LAMBDA_FLOW}")
    print(f"  Recovery net      : {'enabled' if recovery_net is not None else 'disabled'}")

    live_plot = LivePlot(
        title=f'DeepFusionNet v5  |  Open Images {args.source}',
        xlabel='Batch',
    ) if args.plot else None

    # LR reduction on plateau tracking
    val_loss_history: list = []
    consecutive_bad  = 0

    # ══════════════════════════════════════════════════════════════════════
    #  Prefetch helper
    # ══════════════════════════════════════════════════════════════════════
    def _prefetch(batch_num, offset):
        # ── External source: files already local, return slice directly ──
        if external_source:
            end         = min(offset + args.batch_real, len(urls))
            batch_files = urls[offset:end]
            print(f"  [prefetch] batch {batch_num}: {len(batch_files)} local frames "
                  f"(offset {offset}–{end})")
            return batch_files, None, end   # temp_dir=None — nothing to delete

        # ── Streaming source: download a fresh batch ──────────────────────
        end        = offset + args.batch_real
        batch_urls = urls[offset:end]
        if not batch_urls:
            offset     = 0
            batch_urls = urls[:args.batch_real]
            end        = args.batch_real
        tdir = os.path.join(CACHE_DIR, f'batch_{batch_num:06d}')
        t0   = time.time()
        rp   = download_batch(batch_urls, tdir, max_workers=args.max_workers,
                              cookies_from_browser=args.cookies_from_browser)
        dt   = time.time() - t0
        print(f"  [prefetch] batch {batch_num}: {len(rp)}/{len(batch_urls)} images in {dt:.1f}s")
        return rp, tdir, end

    # ══════════════════════════════════════════════════════════════════════
    #  Streaming loop
    # ══════════════════════════════════════════════════════════════════════
    avg_loss, acc = 0.0, 0.0

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='prefetch') as pool:
        prefetch_future = pool.submit(_prefetch, batches_done + 1, url_offset)

        while batches_done < args.total_batches:
            batch_num = batches_done + 1
            print(f"\n--- Batch {batch_num}/{args.total_batches} ---")

            real_paths, temp_dir, end = prefetch_future.result()
            url_offset = end

            if len(real_paths) < args.min_real:
                print(f"  Only {len(real_paths)} images (min {args.min_real}). Skipping.")
                if temp_dir and os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                # External source: no wrap — advance offset and let the loop end naturally
                next_skip_offset = url_offset if (not external_source and url_offset < len(urls)) else url_offset
                if not external_source and url_offset >= len(urls):
                    next_skip_offset = 0
                prefetch_future = pool.submit(_prefetch, batch_num, next_skip_offset)
                continue

            if batch_num < args.total_batches:
                # External source: never wrap past end of data
                if external_source:
                    next_offset = url_offset
                else:
                    next_offset = url_offset if url_offset < len(urls) else 0
                prefetch_future = pool.submit(_prefetch, batch_num + 1, next_offset)

            # ── Save monitor images ────────────────────────────────────
            monitor_real_dir = os.path.join(DATA_DIR, 'monitor_real')
            os.makedirs(monitor_real_dir, exist_ok=True)
            for i, src in enumerate(real_paths[:6]):
                try:
                    dst = os.path.join(monitor_real_dir,
                                       f'real_{i:02d}{os.path.splitext(src)[1]}')
                    shutil.copy2(src, dst)
                except Exception:
                    pass

            # ── Sample AI images ───────────────────────────────────────
            if ai_paths:
                n_ai       = min(args.batch_ai, len(ai_paths))
                sampled_ai = random.sample(ai_paths, n_ai)
            else:
                sampled_ai = []
                print("  No AI images — real-only batch")

            # ── Train / val split (90% / 10% of real) ─────────────────
            n_val       = max(1, len(real_paths) // 10)
            val_real    = real_paths[:n_val]
            train_real  = real_paths[n_val:]
            n_val_ai    = max(1, len(sampled_ai) // 10) if sampled_ai else 0
            val_ai      = sampled_ai[:n_val_ai]
            train_ai    = sampled_ai[n_val_ai:]

            train_dataset = StreamBatchDataset(
                train_real, train_ai,
                real_transform=real_transform,
                ai_transform=ai_transform,
                recovery_net=recovery_net, device=device,
                prnu_extractor=prnu_gpu,
            )
            val_dataset = StreamBatchDataset(
                val_real, val_ai,
                real_transform=val_transform,
                ai_transform=val_transform,
                recovery_net=recovery_net, device=device,
                prnu_extractor=prnu_gpu,
            )
            train_loader = DataLoader(
                train_dataset, batch_size=LOADER_BATCH_SIZE,
                shuffle=True, num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=LOADER_BATCH_SIZE,
                shuffle=False, num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
            )
            print(f"  Train: {len(train_real)} real + {len(train_ai)} AI = {len(train_dataset)} samples")
            print(f"  Val  : {len(val_real)} real + {len(val_ai)} AI = {len(val_dataset)} samples")

            # ── Train ──────────────────────────────────────────────────
            try:
                for ep in range(args.inner_epochs):
                    ep_str = f"ep {ep+1}/{args.inner_epochs}" if args.inner_epochs > 1 else ""
                    desc   = f"Train {ep_str}".strip()
                    avg_loss, acc = train_one_batch(
                        model, optimizer, scaler, train_loader, device,
                        desc=desc, prnu_gpu=prnu_gpu,
                    )
                    print(f"  {'Train' + (' ' + ep_str if ep_str else ''):12s}"
                          f"loss={avg_loss:.4f}  acc={acc*100:.1f}%"
                          f"  vram={_vram_str(device)}")

                # ── Validate ───────────────────────────────────────────
                if len(val_dataset) > 0:
                    val_loss, val_acc = eval_one_batch(model, val_loader, device, prnu_gpu=prnu_gpu)
                    print(f"  {'Val':12s}loss={val_loss:.4f}  acc={val_acc*100:.1f}%")

                    val_loss_history.append(val_loss)
                    if len(val_loss_history) > 1 and val_loss > val_loss_history[-2]:
                        consecutive_bad += 1
                    else:
                        consecutive_bad = 0

                    if consecutive_bad >= 3:
                        for pg in optimizer.param_groups:
                            pg['lr'] *= 0.5
                        new_lr = optimizer.param_groups[0]['lr']
                        print(f"  LR reduced to {new_lr:.2e} (3 consecutive val_loss increases)")
                        consecutive_bad = 0

            finally:
                if temp_dir and os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"  Deleted {temp_dir}")

            # ── Joint recovery net training (every 5 batches) ─────────
            if (recovery_net is not None and recovery_optimizer is not None
                    and batch_num % 5 == 0 and len(train_real) > 0):
                try:
                    # Load a few clean real images as training targets
                    clean_imgs = []
                    for p in train_real[:min(4, len(train_real))]:
                        try:
                            img = Image.open(p).convert('RGB').resize(
                                (IMG_SIZE, IMG_SIZE), _BILINEAR
                            )
                            arr = np.array(img, dtype=np.float32) / 255.0
                            clean_imgs.append(arr.transpose(2, 0, 1))
                        except Exception:
                            pass
                    if clean_imgs:
                        clean_t = torch.from_numpy(np.stack(clean_imgs)).to(device)
                        rec_loss = train_recovery_net_one_step(
                            recovery_net, recovery_optimizer, recovery_scaler,
                            clean_t, quality_range=(60, 90)
                        )
                        print(f"  Recovery net step: loss={rec_loss:.5f}")
                except Exception as e:
                    print(f"  Recovery net training error: {e}")

            # ── Save recovery net every 50 batches ────────────────────
            if recovery_net is not None and batch_num % 50 == 0:
                _save_recovery_net(recovery_net)

            # ── Update counters ────────────────────────────────────────
            batches_done    += 1
            total_real_seen += len(real_paths)

            save_progress({
                'url_offset':      url_offset,
                'batches_done':    batches_done,
                'total_real_seen': total_real_seen,
            })

            _save_checkpoint(model, optimizer, batches_done, avg_loss, acc)

            # Free GPU memory between batches — critical for long Colab runs
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            if batches_done % args.save_every == 0:
                _save_model(model)

            if live_plot:
                live_plot.update(batches_done, avg_loss, acc)

            print(f"  Progress: {batches_done}/{args.total_batches} batches | "
                  f"{total_real_seen:,} real seen total")

            ResourceGuard.check_or_abort(model, f"after batch {batches_done}")

            del train_dataset, val_dataset, train_loader, val_loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    #  Final save
    # ══════════════════════════════════════════════════════════════════════
    if external_source:
        print(f"\n{'='*60}")
        print(f"  Training complete — one full pass over external source data.")
        print(f"  {batches_done} batches | {total_real_seen:,} real frames seen")
        print(f"{'='*60}")

    if live_plot:
        live_plot.close()

    print("\n=== Streaming training complete ===")
    print(f"  Total batches   : {batches_done}")
    print(f"  Total real seen : {total_real_seen:,}")
    _save_model(model)
    _save_checkpoint(model, optimizer, batches_done, 0.0, 0.0)
    if recovery_net is not None:
        _save_recovery_net(recovery_net)

    try:
        if os.path.isdir(CACHE_DIR) and not os.listdir(CACHE_DIR):
            os.rmdir(CACHE_DIR)
    except Exception:
        pass


if __name__ == '__main__':
    main()
