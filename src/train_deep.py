"""
train_deep.py — Unified Deep AI Detector Training Pipeline
============================================================

Two training sections in one script:

  ════════════════════════════════════════════════
    SECTION A — IMAGE TRAINING
  ════════════════════════════════════════════════
  • Model : EfficientFusionNet  (EfficientNet-B0 + PRNU dual-branch)
  • PRNU  : 16-dim full-resolution feature vector (extract_prnu_features_fullres)
            Tiles image into 512px patches, captures spatial PRNU variance
  • Data  : data/real, data/ai, data/train/REAL, data/train/FAKE,
            custom --image_dir, previously-generated AI images
  • Train : AdamW + OneCycleLR + AMP + grad accumulation
  • Save  : ai_detector_image_deep.pth | .ts (TorchScript) | _int8.pt (quantised)

  ════════════════════════════════════════════════
    SECTION B — VIDEO TRAINING  (4K 120 FPS)
  ════════════════════════════════════════════════
  • Model : VideoTemporalFusionNet  (vision + optical-flow + PRNU triple-branch)
  • Input : native-resolution video frames (up to 3840×2160 @ 120 fps)
  • PRNU  : full-resolution 16-dim per 512px tile (deepest possible analysis)
  • Flow  : Farneback optical flow between consecutive frames → magnitude map
  • Tiles : each 4K frame → up to 32 × 512×512 non-overlapping patches
            Every patch gets its own PRNU vector + flow slice + EfficientNet pass
  • Data  : streaming IterableDataset — handles >30 GB without RAM blowup
            Real frames: --video_dir  |  AI variants: auto-generated edits
  • Train : batch=2, grad_accum=16, AMP, AdamW, cosine LR decay
  • Ckpt  : saved every --checkpoint_interval minutes (default 30)
  • Save  : ai_detector_video_deep.pth | .ts (TorchScript)

Usage:
    python src/train_deep.py --mode image
    python src/train_deep.py --mode video --video_dir /path/to/real_videos
    python src/train_deep.py --mode both  --video_dir /path/to/real_videos
    python src/train_deep.py --mode video --resume models/ai_detector_video_deep.pth
    python src/train_deep.py --help
"""

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import gc
import io
import json
import os
import random
import shutil
import sys
import time

# ── Third-party (lightweight — loaded before PyTorch) ────────────────────────
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ── PyTorch (imported after memory check) ────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms

# ── Project imports ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model_prnu import UnifiedFusionNet
from live_plot import LivePlot
from training_dashboard import TrainingDashboard
from prnu_features import (
    extract_prnu_features,
    extract_prnu_features_fullres,
    extract_prnu_map,
    PRNU_FAST_DIM,
    PRNU_FULLRES_DIM,
)

# ── C++ Hardware Acceleration ────────────────────────────────────────────────
try:
    import fast_video_processor
    HAS_CPP_FLOW = True
except ImportError:
    HAS_CPP_FLOW = False

# ─────────────────────────────────────────────────────────────────────────────
#  Paths & Global Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR       = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR     = os.path.join(SCRIPT_DIR, '..', 'models')
TEMP_FRAMES    = os.path.join(DATA_DIR, 'temp_frames')
TRAINED_FILE   = os.path.join(MODELS_DIR, 'trained_data.json')

UNIFIED_MODEL_PATH  = os.path.join(MODELS_DIR, 'ai_detector_unified_v1.pth')
IMAGE_MODEL_PATH    = os.path.join(MODELS_DIR, 'ai_detector_image_deep.pth')   # kept for compat
IMAGE_MODEL_TS      = os.path.join(MODELS_DIR, 'ai_detector_image_deep_script.ts')
IMAGE_MODEL_QUANT   = os.path.join(MODELS_DIR, 'ai_detector_image_deep_int8.pt')
VIDEO_MODEL_PATH    = os.path.join(MODELS_DIR, 'ai_detector_video_deep.pth')   # kept for compat
VIDEO_MODEL_TS      = os.path.join(MODELS_DIR, 'ai_detector_video_deep_script.ts')
VIDEO_CKPT_PATH     = os.path.join(MODELS_DIR, 'ai_detector_video_deep_ckpt.pth')

# Image training defaults — tuned for T4 (14.6 GB VRAM)
IMG_SIZE          = 512
IMG_BATCH         = 8       # T4 handles 8×512 with gradient_checkpointing=True
IMG_GRAD_ACCUM    = 4       # effective batch = 32 (same as before, fewer accum steps needed)
IMG_EPOCHS        = 10
IMG_LR            = 1e-3

# Video training defaults
VID_TILE_SIZE     = 512     # patch size cut from each 4K frame
VID_BATCH         = 2       # 2 tiles at a time (GPU-safe for 4 GB VRAM)
VID_GRAD_ACCUM    = 16      # effective = 32
VID_EPOCHS        = 120
VID_LR            = 5e-4
VID_FPS_SAMPLE    = 4       # frames to extract per real second (Higher for more detail)
VID_EDITS_PER_FRAME = 3     # AI-edited variants per real frame (Higher for better separation)
VID_FLOW_SIZE     = 256     # optical-flow computed at this resolution (faster)
CKPT_INTERVAL_MIN = 30      # checkpoint save interval in minutes

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mts', '.m2ts'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if device.type == 'cuda':
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"  VRAM  : {total_vram:.0f} MB")

# ─────────────────────────────────────────────────────────────────────────────
#  External Downloading & Streaming
# ─────────────────────────────────────────────────────────────────────────────

def _download_youtube_video(url, output_dir=TEMP_FRAMES, section_minutes=None):
    """
    Downloads a YouTube video using yt-dlp to output_dir.

    section_minutes : if set, only download that many minutes from the start
                      (uses --download-sections, requires ffmpeg).
    Returns the absolute path to the downloaded video file.
    """
    import yt_dlp
    os.makedirs(output_dir, exist_ok=True)
    out_tmpl = os.path.join(output_dir, '%(id)s.%(ext)s')

    ydl_opts = {
        # Prefer H.264 (avc1) so OpenCV cv2 can decode without hardware AV1 support.
        # Falls back to any non-AV1 mp4, then any mp4<=720p, then best available.
        'format': (
            'bestvideo[vcodec^=avc1][ext=mp4][height<=720]+bestaudio[ext=m4a]'
            '/bestvideo[ext=mp4][height<=720][vcodec!^=av01]+bestaudio[ext=m4a]'
            '/best[ext=mp4][height<=720]'
            '/best'
        ),
        'merge_output_format': 'mp4',
        'outtmpl': out_tmpl,
        'quiet': False,
        'no_warnings': True,
    }

    if section_minutes is not None:
        # Cap at first N minutes — avoids downloading 50+ GB 12-hour videos
        h = int(section_minutes) // 60
        m = int(section_minutes) % 60
        end = f'{h:02d}:{m:02d}:00'
        ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(
            None, [{'start_time': 0, 'end_time': section_minutes * 60}]
        )
        ydl_opts['force_keyframes_at_cuts'] = True
        print(f"\n  Downloading first {section_minutes} min of {url} ...")
    else:
        print(f"\n  Downloading {url} ...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not os.path.exists(filename):
            filename = os.path.splitext(filename)[0] + '.mp4'

    size_mb = os.path.getsize(filename) / 1024**2 if os.path.exists(filename) else 0
    print(f"  Downloaded: {filename}  ({size_mb:.0f} MB)")
    return filename

def _youtube_search_urls(query: str, n: int = 5) -> list:
    """
    Returns up to n YouTube video URLs matching the search query,
    without downloading anything.

    Uses yt-dlp's ytsearch: prefix with extract_flat so no media is fetched.
    Returns a list of full https://www.youtube.com/watch?v=... URLs.
    """
    import yt_dlp
    urls = []
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }
    search_url = f"ytsearch{n}:{query}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            results = ydl.extract_info(search_url, download=False)
            for entry in (results or {}).get('entries') or []:
                if not entry:
                    continue
                vid_id = entry.get('id') or ''
                if vid_id:
                    urls.append(f"https://www.youtube.com/watch?v={vid_id}")
        except Exception as e:
            print(f"  [warn] Search error: {e}")
    return urls


def _download_external_image(url, output_dir=TEMP_FRAMES):
    """
    Downloads a single image or a compressed archive (ZIP/TAR) from a URL.
    Returns the path to the downloaded image, or output_dir if it was an archive.
    """
    import requests
    import zipfile
    import tarfile
    os.makedirs(output_dir, exist_ok=True)
    try:
        print(f"  📥  Downloading {url} ...")
        r = requests.get(url, stream=True, timeout=20)
        r.raise_for_status()
        
        # Determine extension from URL or Content-Type
        ext = os.path.splitext(url)[1].lower()
        if not ext:
            ctype = r.headers.get('Content-Type', '')
            if 'image/jpeg' in ctype: ext = '.jpg'
            elif 'image/png' in ctype: ext = '.png'
            elif 'application/zip' in ctype: ext = '.zip'
            elif 'application/x-tar' in ctype: ext = '.tar'
            elif 'application/gzip' in ctype: ext = '.tar.gz'
        
        filename = os.path.join(output_dir, f"ext_download_{int(time.time())}{ext}")
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Handle archives
        if ext in {'.zip', '.tar', '.gz', '.tgz', '.tar.gz'}:
            print(f"  📦  Extracting archive: {filename} ...")
            if ext == '.zip':
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
            else:
                with tarfile.open(filename, 'r:*') as tar_ref:
                    tar_ref.extractall(output_dir)
            
            # Return the output_dir as the path to scan for images
            return output_dir
            
        return filename
    except Exception as e:
        print(f"  ⚠️  Failed to download {url}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Resource Guard (prevents system freeze)
# ─────────────────────────────────────────────────────────────────────────────

class ResourceGuard:
    MIN_DISK_MB  = 500
    MIN_RAM_MB   = 400
    MAX_RAM_PCT  = 92
    MAX_GPU_PCT  = 95

    @staticmethod
    def _disk_free_mb():
        try:
            return shutil.disk_usage(os.path.abspath(MODELS_DIR)).free / 1024**2
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
                return alloc / total * 100 if total > 0 else 0
        except Exception:
            pass
        return 0

    @classmethod
    def check(cls):
        """Return (is_safe: bool, reason: str)."""
        if cls._disk_free_mb() < cls.MIN_DISK_MB:
            return False, f"Disk critically low ({cls._disk_free_mb():.0f} MB)"
        if cls._ram_free_mb() < cls.MIN_RAM_MB:
            return False, f"RAM critically low ({cls._ram_free_mb():.0f} MB free)"
        if cls._ram_pct() > cls.MAX_RAM_PCT:
            return False, f"RAM {cls._ram_pct():.0f}% used"
        if cls._gpu_pct() > cls.MAX_GPU_PCT:
            return False, f"GPU VRAM {cls._gpu_pct():.0f}% used"
        return True, ''

    @classmethod
    def check_or_abort(cls, model, context='', save_path=None):
        safe, reason = cls.check()
        if not safe:
            print(f"\n  🛑  EMERGENCY STOP: {reason}  [{context}]")
            if model is not None and save_path:
                try:
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    torch.save(model.state_dict(), save_path + '_EMERGENCY.pth')
                    print(f"  💾  Emergency checkpoint: {save_path}_EMERGENCY.pth")
                except Exception as e:
                    print(f"  ⚠️  Could not save: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  ❌  Training stopped.  Free resources then re-run with --resume\n")
            sys.exit(1)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  AI-edit augmentation (reused from train_video_realtime.py approach)
# ─────────────────────────────────────────────────────────────────────────────

def _blur_sharpen(img):
    blurred = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 4.0)))
    return blurred.filter(ImageFilter.UnsharpMask(
        radius=random.randint(2, 6), percent=random.randint(120, 300), threshold=0))

def _color_hallucination(img):
    img = ImageEnhance.Color(img).enhance(random.uniform(1.2, 2.2))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.25))
    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + random.uniform(-20, 20), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def _noise_compress(img):
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, random.uniform(4, 18), arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=random.randint(20, 55))
    buf.seek(0)
    result = Image.open(buf).convert('RGB')
    result.load()
    return result

def _rescale(img):
    w, h = img.size
    f = random.uniform(0.25, 0.55)
    return img.resize((max(1, int(w*f)), max(1, int(h*f))), Image.BICUBIC)\
               .resize((w, h), Image.BICUBIC)

def _edge_smooth(img):
    arr = np.array(img)
    d = random.choice([5, 7, 9, 11])
    return Image.fromarray(cv2.bilateralFilter(arr, d,
                                               random.uniform(40, 100),
                                               random.uniform(40, 100)))

def _freq_manip(img):
    arr = np.array(img, dtype=np.float32)
    rows, cols = arr.shape[:2]
    crow, ccol = rows//2, cols//2
    r = random.randint(rows//8, rows//4)
    Y, X   = np.ogrid[:rows, :cols]
    dist   = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    ring   = (dist >= r) & (dist <= r + random.randint(rows//10, rows//5))
    mask   = np.ones((rows, cols), dtype=np.float32)
    mask[ring] *= random.uniform(0.1, 0.4)
    for c in range(3):
        fshift = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
        fshift *= mask
        arr[:, :, c] = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

_EDIT_POOL = [_blur_sharpen, _color_hallucination, _noise_compress,
              _rescale, _edge_smooth, _freq_manip]

def apply_ai_edits(frame_pil, num_variants=VID_EDITS_PER_FRAME):
    variants = []
    for _ in range(num_variants):
        edited = frame_pil.copy()
        for fn in random.sample(_EDIT_POOL, min(random.randint(1, 3), len(_EDIT_POOL))):
            try:
                edited = fn(edited)
            except Exception:
                pass
        variants.append(edited)
    return variants


class VideoCompressionAugment:
    """
    Simulates platform-specific video frame compression on a PIL tile.
    Applied only to AI tiles — real tiles are left pristine.
    """
    _PLATFORMS = [
        # (name, weight, q_min, q_max)
        ('youtube',   0.18, 80, 95),
        ('tiktok',    0.20, 55, 75),
        ('instagram', 0.18, 60, 80),
        ('facebook',  0.12, 55, 75),
        ('twitter',   0.10, 50, 72),
        ('whatsapp',  0.12, 45, 65),
        ('telegram',  0.10, 65, 85),
    ]

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        _, weight, q_min, q_max = random.choices(
            self._PLATFORMS, weights=[p[1] for p in self._PLATFORMS]
        )[0]
        quality = random.randint(q_min, q_max)

        # Optional resize-down + up (simulates bitrate reduction)
        if random.random() < 0.35:
            w, h = pil_img.size
            scale = random.uniform(0.6, 0.85)
            pil_img = pil_img.resize(
                (max(32, int(w * scale)), max(32, int(h * scale))), Image.BILINEAR
            ).resize((w, h), Image.BILINEAR)

        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


_VIDEO_COMPRESSION_AUG = VideoCompressionAugment()


# ═══════════════════════════════════════════════════════════════════════════════
#
#  ████████╗  ███████╗  ██████╗  ████████╗  ██╗  ██████╗  ███╗   ██╗       █████╗
#  ██╔════╝  ██╔════╝  ██╔════╝  ╚══██╔══╝  ██║ ██╔═══██╗ ████╗  ██║     ██╔══██╗
#  ███████╗  █████╗    ██║          ██║     ██║ ██║   ██║ ██╔██╗ ██║     ███████║
#  ╚════██║  ██╔══╝    ██║          ██║     ██║ ██║   ██║ ██║╚██╗██║     ██╔══██║
#  ███████║  ███████╗  ╚██████╗     ██║     ██║ ╚██████╔╝ ██║ ╚████║     ██║  ██║
#  ╚══════╝  ╚══════╝   ╚═════╝     ╚═╝     ╚═╝  ╚═════╝  ╚═╝  ╚═══╝     ╚═╝  ╚═╝
#
#  IMAGE TRAINING — EfficientFusionNet + 16-dim full-res PRNU
#
# ═══════════════════════════════════════════════════════════════════════════════


class ImageDataset(Dataset):
    """
    Dataset for image training.
    Returns (img_tensor, prnu_64_tensor, prnu_map_tensor, label_tensor).
    PRNU scalar features are 64-dim (v6 fullres); prnu_map is (3,128,128).
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        # ── Load image ──
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  [warn] Cannot open {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        # ── Full-res PRNU scalar features (64-dim v6) ──
        try:
            prnu_input = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR) \
                         if max(image.size) > IMG_SIZE * 2 else image
            arr = np.array(prnu_input, dtype=np.float64) / 255.0
            prnu_feats = extract_prnu_features_fullres(arr, tile_size=128)
            if not np.isfinite(prnu_feats).all():
                prnu_feats = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)
        except Exception as e:
            print(f"  [warn] PRNU failed for {img_path}: {e}")
            prnu_feats = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)

        # ── PRNU spatial map (3×128×128) ──
        try:
            prnu_map_np = extract_prnu_map(image, output_size=128)   # (128,128,3)
            if not np.isfinite(prnu_map_np).all():
                prnu_map_np = np.zeros((128, 128, 3), dtype=np.float32)
        except Exception:
            prnu_map_np = np.zeros((128, 128, 3), dtype=np.float32)
        prnu_map_t = torch.from_numpy(prnu_map_np.transpose(2, 0, 1)).float()

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.from_numpy(prnu_feats).float(),
            prnu_map_t,
            torch.tensor(label, dtype=torch.float32),
        )


def _collect_image_paths(image_data_dir=None):
    """Collect (paths, labels) for image training from standard directories."""
    previously_trained = set()
    if os.path.exists(TRAINED_FILE):
        try:
            with open(TRAINED_FILE) as f:
                previously_trained = set(json.load(f))
        except Exception:
            pass

    paths, labels, new_paths = [], [], []

    def _scan(directory, label, recursive=True):
        if not os.path.exists(directory):
            return
        walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]
        for root, _, files in walker:
            for fn in files:
                fp = os.path.join(root, fn)
                if fp in previously_trained:
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.bmp'}:
                    paths.append(fp)
                    labels.append(label)
                    new_paths.append(fp)

    if image_data_dir:
        print(f"  → Custom image directory: {image_data_dir}")
        _scan(image_data_dir, label=1)   # assume AI / custom
    else:
        print("  → Scanning standard data directories…")
        for cat in ['real', 'personal']:
            _scan(os.path.join(DATA_DIR, cat), label=0)
        _scan(os.path.join(DATA_DIR, 'ai'), label=1)
        _scan(os.path.join(DATA_DIR, 'generated_ai', 'images'), label=1)
        _scan(os.path.join(DATA_DIR, 'edited_ai', 'images'), label=1)
        _scan(os.path.join(DATA_DIR, 'train', 'REAL'), label=0)
        _scan(os.path.join(DATA_DIR, 'train', 'FAKE'), label=1)

    print(f"  → Found {len(paths)} new images "
          f"({sum(1 for l in labels if l==0)} real, {sum(1 for l in labels if l==1)} AI)")
    return paths, labels, previously_trained, new_paths


def _save_image_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, UNIFIED_MODEL_PATH)
    print(f"  💾  State-dict  → {UNIFIED_MODEL_PATH}")


def train_image_section(image_data_dir=None, epochs=IMG_EPOCHS,
                        lr=IMG_LR, batch_size=IMG_BATCH,
                        grad_accum=IMG_GRAD_ACCUM):
    """
    ════════════════════════════════════════════════
      SECTION A — IMAGE TRAINING
    ════════════════════════════════════════════════
    """
    print("\n")
    print("=" * 66)
    print("    SECTION A ─ IMAGE TRAINING")
    print("    Model : UnifiedFusionNet v1  (16-branch, 64-dim PRNU, image mode)")
    print("=" * 66)

    # ── Data ──
    paths, labels, prev_trained, new_paths = _collect_image_paths(image_data_dir)
    if not paths:
        print("  No new images found. Skipping image training.")
        return

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        paths, labels, test_size=0.2, random_state=42
    )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageDataset(X_tr, y_tr, transform=train_tf)
    val_ds   = ImageDataset(X_val, y_val, transform=val_tf)
    num_workers = 0   # 0 = main process only (Colab-safe; avoids fork issues)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    print(f"\n  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ──
    model = UnifiedFusionNet(prnu_in_features=PRNU_FULLRES_DIM,
                             gradient_checkpointing=True).to(device)
    model.param_summary()

    criterion = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    steps_per_epoch = max(len(train_dl) // grad_accum, 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs, pct_start=0.3,
    )

    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"\n  Training for {epochs} epochs…\n")

    for epoch in range(epochs):
        ResourceGuard.check_or_abort(model, f"Epoch {epoch+1}", IMAGE_MODEL_PATH)

        model.train()
        run_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for i, (imgs, prnu_feats, prnu_maps, targets) in enumerate(train_dl):
            imgs       = imgs.to(device)
            prnu_feats = prnu_feats.to(device)
            prnu_maps  = prnu_maps.to(device)
            targets    = targets.to(device).view(-1, 1)

            try:
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out = model(imgs, prnu_feats, prnu_maps)
                    if isinstance(out, tuple):
                        (logits, prnu_aux, halluc_aux, prnu_sp_aux,
                         gan_aux, cmos_aux, col_aux, flow_aux, *rest) = out
                        loss = (
                            criterion(logits,      targets)
                            + 0.35 * criterion(prnu_aux,   targets)
                            + 0.15 * criterion(halluc_aux, targets)
                            + 0.20 * criterion(prnu_sp_aux,targets)
                            + 0.25 * criterion(gan_aux,    targets)
                            + 0.20 * criterion(cmos_aux,   targets)
                            + 0.20 * criterion(col_aux,    targets)
                            + 0.15 * criterion(flow_aux,   targets)
                        )
                        if rest:   # motion aux head in video mode
                            loss = loss + 0.15 * criterion(rest[0], targets)
                        loss = loss / grad_accum
                    else:
                        logits = out
                        loss   = criterion(logits, targets) / grad_accum
            except Exception as _fwd_exc:
                print(f"  ⚠️  WARNING: image train forward failed at batch {i+1} "
                      f"({type(_fwd_exc).__name__}: {_fwd_exc}). Skipping.")
                optimizer.zero_grad()
                continue

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            run_loss += loss.item() * grad_accum * imgs.size(0)
            preds = (torch.sigmoid(logits.detach()) > 0.5).float()
            total += targets.size(0)
            correct += (preds == targets).sum().item()

            del imgs, prnu_feats, prnu_maps, targets, out, logits, loss

            if (i + 1) % 50 == 0:
                ResourceGuard.check_or_abort(model, f"E{epoch+1} B{i+1}", IMAGE_MODEL_PATH)
                print(f"    Batch {i+1}/{len(train_dl)} | "
                      f"Loss {run_loss/max(total,1):.4f} | "
                      f"Acc  {correct/max(total,1):.4f}")

        # Flush remaining grads if not already flushed by the last batch
        if (len(train_dl) % grad_accum) != 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        ep_loss = run_loss / max(len(train_ds), 1)
        ep_acc  = correct  / max(total, 1)
        lr_now  = optimizer.param_groups[0]['lr']
        print(f"\n  Epoch {epoch+1}/{epochs}  loss={ep_loss:.4f}  "
              f"acc={ep_acc:.4f}  lr={lr_now:.2e}")

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, prnu_feats, prnu_maps, targets in val_dl:
                imgs       = imgs.to(device)
                prnu_feats = prnu_feats.to(device)
                prnu_maps  = prnu_maps.to(device)
                targets    = targets.to(device).view(-1, 1)
                try:
                    out    = model(imgs, prnu_feats, prnu_maps)
                    logits = out[0] if isinstance(out, tuple) else out
                    v_loss    += criterion(logits, targets).item() * imgs.size(0)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    v_total   += targets.size(0)
                    v_correct += (preds == targets).sum().item()
                except Exception as _val_exc:
                    print(f"  ⚠️  WARNING: image val forward failed "
                          f"({type(_val_exc).__name__}: {_val_exc}). Skipping batch.")
                    continue

        print(f"  Val   loss={v_loss/max(len(val_ds),1):.4f}  "
              f"acc={v_correct/max(v_total,1):.4f}")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n  ✅  Image training complete.")
    _save_image_model(model)

    # Update trained-paths log
    all_trained = prev_trained | set(new_paths)
    with open(TRAINED_FILE, 'w') as f:
        json.dump(list(all_trained), f, indent=2)
    print(f"  📝  Trained-paths log updated ({len(all_trained)} total).")


# ═══════════════════════════════════════════════════════════════════════════════
#
#  ██╗   ██╗██╗██████╗ ███████╗ ██████╗     ████████╗██████╗  █████╗ ██╗███╗  ██╗
#  ██║   ██║██║██╔══██╗██╔════╝██╔═══██╗       ██╔══╝██╔══██╗██╔══██╗██║████╗ ██║
#  ██║   ██║██║██║  ██║█████╗  ██║   ██║       ██║   ██████╔╝███████║██║██╔██╗██║
#  ╚██╗ ██╔╝██║██║  ██║██╔══╝  ██║   ██║       ██║   ██╔══██╗██╔══██║██║██║╚████║
#   ╚████╔╝ ██║██████╔╝███████╗╚██████╔╝       ██║   ██║  ██║██║  ██║██║██║ ╚███║
#    ╚═══╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚══╝
#
#  VIDEO TRAINING — VideoTemporalFusionNet  (4K · 120 FPS · deep PRNU)
#
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_physics_flow(prev_bgr, curr_bgr, flow_size=VID_FLOW_SIZE):
    """
    Compute Farneback optical flow between two BGR frames.
    Extracts 4-channel Physics Flow: (dx, dy, magnitude, micro-jitter).
    Returns a float32 map of shape (flow_size, flow_size, 4),
    scaled to approx [-1, 1] for dx/dy and [0,1] for mag/micro.
    
    Uses C++ PyBind11 acceleration if available, otherwise falls back to Python.
    """
    if HAS_CPP_FLOW:
        # C++ extension handles resizing, grayscale, flow, and normalization completely natively
        return fast_video_processor.compute_physics_flow_cpp(prev_bgr, curr_bgr, flow_size)
        
    # Python Fallback
    h = w = flow_size
    prev_sm = cv2.resize(prev_bgr, (w, h))
    curr_sm = cv2.resize(curr_bgr, (w, h))

    prev_gray = cv2.cvtColor(prev_sm, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_sm, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx**2 + dy**2)   # (H, W)
    
    # Normalise mag
    p99 = np.percentile(mag, 99) if mag.size > 0 else 0.0
    p99 += 1e-6
    mag_norm = np.clip(mag / p99, 0.0, 1.0)
    
    # Normalise dx, dy (roughly to [-1, 1])
    dx_norm = np.clip(dx / p99, -1.0, 1.0)
    dy_norm = np.clip(dy / p99, -1.0, 1.0)

    # Micro-movement (Jitter via Laplacian of raw magnitude)
    micro = cv2.Laplacian(mag, cv2.CV_32F)
    micro = np.abs(micro)
    micro_p99 = np.percentile(micro, 99) if micro.size > 0 else 0.0
    micro_p99 += 1e-6
    micro_norm = np.clip(micro / micro_p99, 0.0, 1.0)

    # Stack into (H, W, 4)
    if mag.size == 0:
        return np.zeros((flow_size, flow_size, 4), dtype=np.float32)
        
    physics_flow = np.stack([dx_norm, dy_norm, mag_norm, micro_norm], axis=-1).astype(np.float32)
    return physics_flow


def _tile_frame(frame_rgb_pil, tile_size=VID_TILE_SIZE):
    """
    Split a PIL RGB frame into non-overlapping square tiles.
    Returns list of (tile_pil, row_idx, col_idx).
    Tiles at the boundary are padded to tile_size x tile_size.
    """
    w, h = frame_rgb_pil.size
    tiles = []
    for r in range(0, h, tile_size):
        for c in range(0, w, tile_size):
            box  = (c, r, min(c + tile_size, w), min(r + tile_size, h))
            tile = frame_rgb_pil.crop(box)
            # Pad to tile_size if needed
            if tile.size != (tile_size, tile_size):
                padded = Image.new('RGB', (tile_size, tile_size))
                padded.paste(tile, (0, 0))
                tile = padded
            tiles.append((tile, r // tile_size, c // tile_size))
    return tiles


def _tile_flow_map(physics_flow, tile_size=VID_TILE_SIZE, frame_w=None, frame_h=None):
    """
    Upsample the 4-channel physics flow map to match the original frame size,
    then tile into tile_size patches.
    Returns list of (4, tile_size, tile_size) tensors in same order as _tile_frame.
    """
    if frame_h is None or frame_w is None:
        return None

    # Upsample flow (H, W, 4) to (frame_h, frame_w, 4)
    flow_full = cv2.resize(physics_flow, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    flow_tiles = []
    for r in range(0, frame_h, tile_size):
        for c in range(0, frame_w, tile_size):
            patch = flow_full[r:r+tile_size, c:c+tile_size, :]
            # Pad if border
            if patch.shape[:2] != (tile_size, tile_size):
                n_ch   = flow_full.shape[2]
                padded = np.zeros((tile_size, tile_size, n_ch), dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded
            # Transpose to (4, tile_size, tile_size)
            patch = patch.transpose(2, 0, 1)
            flow_tiles.append(patch)
    return flow_tiles


class VideoDeepDataset(IterableDataset):
    """
    Streaming iterable dataset for video deep training.

    Scans all videos in video_dir, extracts frames sequentially (no full
    pre-load), tiles each 4K frame, and yields:
      (tile_img_tensor, flow_tile_tensor, prnu_tensor_64dim, label)

    For each real frame, VID_EDITS_PER_FRAME AI-edited variants are generated
    and also yielded with label=1.

    Designed for datasets >30 GB — zero disk pre-caching.
    """

    _IMG_TF = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _AUG_TF = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, video_dir, fps_sample=VID_FPS_SAMPLE,
                 tile_size=VID_TILE_SIZE, edits_per_frame=VID_EDITS_PER_FRAME,
                 flow_size=VID_FLOW_SIZE, use_audio=False, ai_video_dir=None):
        self.video_paths    = self._find_videos(video_dir) if video_dir else []
        self.ai_video_paths = self._find_videos(ai_video_dir) if ai_video_dir else []
        self.fps_sample     = fps_sample
        self.tile_size      = tile_size
        self.edits_per_frame = edits_per_frame
        self.flow_size      = flow_size
        self.use_audio      = use_audio

        if not self.video_paths and not self.ai_video_paths:
            raise FileNotFoundError(
                f"No video files found in: {video_dir}\n"
                f"Supported: {VIDEO_EXTS}"
            )
        print(f"  VideoDeepDataset: {len(self.video_paths)} real video(s) in {video_dir}")
        if self.ai_video_paths:
            print(f"  VideoDeepDataset: {len(self.ai_video_paths)} AI video(s) (label=1)")

    @staticmethod
    def _find_videos(directory):
        found = []
        for root, _, files in os.walk(directory):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                    found.append(os.path.join(root, fn))
        random.shuffle(found)  # randomise order per epoch
        return found

    def _process_video(self, video_path, label=0.0):
        """
        Generator: yields (tile_img, flow_tile, prnu_feats, audio_t, label)

        label=0.0 : real video — yield real tiles + apply_ai_edits synthetic tiles
        label=1.0 : known AI video (e.g. from meta_collector) — yield tiles directly,
                    skip apply_ai_edits (no need to synthesise what we already have)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [warn] Cannot open: {video_path}")
            return

        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interval = max(1, int(cam_fps / self.fps_sample))

        # ── Extract Audio if requested ──
        audio_y = None
        sr      = 16000
        if self.use_audio:
            import librosa
            import subprocess
            temp_wav = os.path.join(TEMP_FRAMES, f"temp_audio_{int(time.time())}_{random.randint(100,999)}.wav")
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                    '-ar', str(sr), '-ac', '1', temp_wav
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(temp_wav):
                    audio_y, _ = librosa.load(temp_wav, sr=sr)
                    os.remove(temp_wav)
            except Exception as e:
                print(f"  [warn] Audio extraction failed for {video_path}: {e}")
                if os.path.exists(temp_wav): os.remove(temp_wav)

        frame_idx       = 0
        prev_bgr        = None
        prev_frame_prnu = None   # 16-dim PRNU of previous sampled frame
        prev_flow_mag   = None   # (flow_size, flow_size) magnitude channel of prev flow
        vname           = os.path.basename(video_path)

        print(f"\n  📹  {vname}  [{w}×{h} @ {cam_fps:.0f}fps]  "
              f"sampling every {interval} frames ({self.fps_sample} fps)")

        def _get_audio_chunk(f_idx, fps):
            if audio_y is None:
                return torch.zeros(1, 64, 64, dtype=torch.float32)
            timestamp = f_idx / fps
            # Grab exactly 1.0 second of audio around the frame
            start_s   = max(0.0, timestamp - 0.5)
            end_s     = start_s + 1.0
            start_i   = int(start_s * sr)
            end_i     = int(end_s * sr)
            chunk     = audio_y[start_i:end_i]
            if len(chunk) < sr:
                # pad if too short
                chunk = np.pad(chunk, (0, max(0, sr - len(chunk))))
            
            # Mel Spectrogram (64 mels)
            melspec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64, hop_length=256)
            melspec = librosa.power_to_db(melspec, ref=np.max) # Shape roughly (64, 63)
            # Resize/Pad to exactly (64, 64) for the CNN
            padded = np.zeros((64, 64), dtype=np.float32)
            w_mel = min(melspec.shape[1], 64)
            padded[:, :w_mel] = melspec[:, :w_mel]
            
            # Normalize approx to [0, 1]
            padded = (padded - padded.min()) / (padded.max() - padded.min() + 1e-6)
            return torch.from_numpy(padded).unsqueeze(0).float() # (1, 64, 64)

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(rgb)    # full native resolution

                    # ── Optical flow physics (6-channel) ──────────────────
                    if prev_bgr is not None:
                        physics_flow_4 = _compute_physics_flow(
                            prev_bgr, bgr, self.flow_size
                        )
                    else:
                        physics_flow_4 = np.zeros(
                            (self.flow_size, self.flow_size, 4), dtype=np.float32
                        )

                    mag_ch = physics_flow_4[:, :, 2]  # (H, W) magnitude channel

                    # Channel 5: flow boundary map (Sobel on magnitude)
                    sx = cv2.Sobel(mag_ch, cv2.CV_32F, 1, 0, ksize=3)
                    sy = cv2.Sobel(mag_ch, cv2.CV_32F, 0, 1, ksize=3)
                    boundary = np.sqrt(sx ** 2 + sy ** 2)
                    b_p99 = float(np.percentile(boundary, 99)) + 1e-6
                    boundary_norm = np.clip(boundary / b_p99, 0.0, 1.0)

                    # Channel 6: inter-frame flow delta (temporal motion consistency)
                    if prev_flow_mag is not None:
                        flow_delta = np.abs(mag_ch - prev_flow_mag)
                        fd_p99 = float(np.percentile(flow_delta, 99)) + 1e-6
                        flow_delta_norm = np.clip(flow_delta / fd_p99, 0.0, 1.0)
                    else:
                        flow_delta_norm = np.zeros_like(mag_ch)

                    # Stack to (flow_size, flow_size, 6)
                    physics_flow = np.concatenate([
                        physics_flow_4,
                        boundary_norm[:, :, np.newaxis],
                        flow_delta_norm[:, :, np.newaxis],
                    ], axis=-1).astype(np.float32)

                    # ── Frame-level PRNU for temporal drift signal ─────────
                    # Downsample to 256×256 for speed, then extract 16-dim PRNU
                    small_rgb = cv2.resize(rgb, (256, 256))
                    small_arr = small_rgb.astype(np.float64) / 255.0
                    try:
                        frame_prnu = extract_prnu_features_fullres(
                            small_arr, tile_size=128
                        )
                    except Exception:
                        frame_prnu = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)

                    # PRNU delta: high in AI frames (inconsistent sensor noise)
                    if prev_frame_prnu is not None:
                        prnu_delta_frame = np.abs(
                            frame_prnu - prev_frame_prnu
                        ).astype(np.float32)
                    else:
                        prnu_delta_frame = np.zeros(
                            PRNU_FULLRES_DIM, dtype=np.float32
                        )

                    # ── Tile the frame ──
                    tiles         = _tile_frame(frame_pil, self.tile_size)
                    flow_tiles    = _tile_flow_map(
                        physics_flow, self.tile_size, frame_w=w, frame_h=h
                    ) or [physics_flow.transpose(2, 0, 1)] * len(tiles)
                    
                    # ── Audio Chunk ──
                    if self.use_audio:
                        audio_t = _get_audio_chunk(frame_idx, cam_fps)
                    else:
                        # Yield a dummy tensor instead of None to keep DataLoader collation happy
                        audio_t = torch.zeros(1, 64, 64, dtype=torch.float32)

                    prnu_delta_t = torch.from_numpy(prnu_delta_frame)  # (16,)

                    # ── Yield tiles with the ground-truth label ──
                    # For AI videos (label=1.0) we yield directly without synthesis.
                    # For real videos (label=0.0) we yield real tiles first.
                    tile_tf = self._AUG_TF if label == 1.0 else self._IMG_TF
                    for (tile_pil, _, _), flow_tile in zip(tiles, flow_tiles):
                        if label == 1.0:
                            tile_pil = _VIDEO_COMPRESSION_AUG(tile_pil)
                        try:
                            arr = np.array(tile_pil, dtype=np.float64) / 255.0
                            prnu = extract_prnu_features_fullres(
                                arr, tile_size=128
                            )
                        except Exception:
                            prnu = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)

                        img_t  = tile_tf(tile_pil)
                        flow_t = torch.from_numpy(flow_tile)  # (6,H,W)
                        prnu_t = torch.from_numpy(prnu)

                        yield img_t, flow_t, prnu_t, prnu_delta_t, audio_t, torch.tensor(label)

                    # ── Yield AI-edited tiles (real videos only) ──
                    if label != 0.0:
                        prev_bgr        = bgr.copy()
                        prev_frame_prnu = frame_prnu.copy()
                        prev_flow_mag   = mag_ch.copy()
                        print(".", end="", flush=True)
                        frame_idx += 1
                        continue

                    ai_frames = apply_ai_edits(frame_pil, self.edits_per_frame)
                    for ai_frame in ai_frames:
                        ai_tiles = _tile_frame(ai_frame, self.tile_size)

                        # Optionally create synthetic fake audio by rolling/pitch shifting
                        if self.use_audio and audio_t is not None:
                            fake_audio_t = torch.roll(audio_t, shifts=random.randint(10, 30), dims=2)
                        else:
                            fake_audio_t = torch.zeros(1, 64, 64, dtype=torch.float32)

                        for (tile_pil, _, _), flow_tile in zip(ai_tiles, flow_tiles):
                            tile_pil = _VIDEO_COMPRESSION_AUG(tile_pil)
                            try:
                                arr = np.array(tile_pil, dtype=np.float64) / 255.0
                                prnu = extract_prnu_features_fullres(
                                    arr, tile_size=128
                                )
                            except Exception:
                                prnu = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)

                            # AI PRNU delta vs previous real frame's PRNU —
                            # AI frames have inconsistent sensor signatures
                            ai_prnu_delta = np.abs(
                                prnu - frame_prnu
                            ).astype(np.float32)

                            img_t          = self._AUG_TF(tile_pil)
                            flow_t         = torch.from_numpy(flow_tile)   # (6,H,W)
                            prnu_t         = torch.from_numpy(prnu)
                            ai_prnu_delta_t = torch.from_numpy(ai_prnu_delta)

                            yield (img_t, flow_t, prnu_t, ai_prnu_delta_t,
                                   fake_audio_t, torch.tensor(1.0))        # 1=AI

                    prev_bgr        = bgr.copy()
                    prev_frame_prnu = frame_prnu.copy()
                    prev_flow_mag   = mag_ch.copy()
                    print(".", end="", flush=True)

                frame_idx += 1

        except KeyboardInterrupt:
            print("  ✋  Video processing interrupted.")
        finally:
            cap.release()

    def __iter__(self):
        for vpath in self.video_paths:
            yield from self._process_video(vpath, label=0.0)
        for vpath in self.ai_video_paths:
            yield from self._process_video(vpath, label=1.0)


def _save_video_model(model, tile_size=VID_TILE_SIZE):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, UNIFIED_MODEL_PATH)
    print(f"  💾  State-dict  → {UNIFIED_MODEL_PATH}")


def _save_checkpoint(model, optimizer, epoch, step, path=VIDEO_CKPT_PATH):
    """Save a resumable training checkpoint."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'epoch':          epoch,
        'step':           step,
        'model_state':    model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
    print(f"  💾  Checkpoint  → {path}  (epoch {epoch}, step {step})")


def _load_checkpoint(model, optimizer, path):
    """Load checkpoint if it exists. Returns (start_epoch, start_step)."""
    if not os.path.exists(path):
        return 0, 0
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device)
    try:
        missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
        # If too many keys mismatch the checkpoint is from a different architecture
        # — discard it and start fresh rather than training on broken weights.
        total_keys = len(list(model.state_dict().keys()))
        if len(missing) > total_keys * 0.3:
            print(f"  [warn] Checkpoint architecture mismatch ({len(missing)} missing keys). "
                  f"Starting from scratch.")
            return 0, 0
    except Exception as e:
        print(f"  [warn] Could not load checkpoint: {e}. Starting from scratch.")
        return 0, 0
    try:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    except Exception:
        pass
    epoch = ckpt.get('epoch', 0)
    step  = ckpt.get('step', 0)
    print(f"  Resumed from checkpoint (epoch {epoch}, step {step})")
    return epoch, step


def train_video_section(
    video_dir,
    epochs=VID_EPOCHS,
    lr=VID_LR,
    batch_size=VID_BATCH,
    grad_accum=VID_GRAD_ACCUM,
    tile_size=VID_TILE_SIZE,
    fps_sample=VID_FPS_SAMPLE,
    checkpoint_interval_min=CKPT_INTERVAL_MIN,
    resume_path=None,
    use_audio=False,
    ai_video_dir=None,
):
    """
    ════════════════════════════════════════════════
      SECTION B — VIDEO TRAINING  (4K / 120 FPS)
    ════════════════════════════════════════════════
    """
    print("\n")
    print("=" * 66)
    print("    SECTION B ─ VIDEO TRAINING  (4K · 120 FPS · Deep PRNU)")
    print(f"    Model : UnifiedFusionNet v1  (16-branch, 64-dim PRNU, video mode)")
    print(f"    Dir   : {video_dir}")
    print(f"    Tile  : {tile_size}×{tile_size} px  |  "
          f"FPS sample: {fps_sample}  |  AI variants: {VID_EDITS_PER_FRAME}")
    print("=" * 66)

    # ── Dataset ──
    dataset = VideoDeepDataset(
        video_dir       = video_dir,
        fps_sample      = fps_sample,
        tile_size       = tile_size,
        edits_per_frame = VID_EDITS_PER_FRAME,
        flow_size       = VID_FLOW_SIZE,
        use_audio       = use_audio,
        ai_video_dir    = ai_video_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = 0,          # IterableDataset: no multi-worker
        pin_memory  = (device.type == 'cuda'),
    )

    # ── Model ──
    model = UnifiedFusionNet(prnu_in_features=PRNU_FULLRES_DIM,
                             gradient_checkpointing=True).to(device)
    model.param_summary()

    criterion = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    # Cosine LR decay (no need for OneCycleLR for a streaming dataset
    # where steps_per_epoch is unknown upfront)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ── Optional resume ──
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        start_epoch, _ = _load_checkpoint(model, optimizer, resume_path)

    print(f"\n  Training for {epochs} epochs  "
          f"(eff. batch = {batch_size}×{grad_accum} = {batch_size*grad_accum})\n")

    ckpt_interval_sec = checkpoint_interval_min * 60
    live_plot = LivePlot(title='Video Training — UnifiedFusionNet v1', xlabel='Epoch')
    dashboard = TrainingDashboard(title='AI Detector — Video Training (UnifiedFusionNet v1)')

    for epoch in range(start_epoch, epochs):
        ResourceGuard.check_or_abort(
            model, f"Video Epoch {epoch+1}", VIDEO_MODEL_PATH
        )

        model.train()
        epoch_start     = time.time()
        last_ckpt_time  = time.time()
        run_loss, correct, total = 0.0, 0, 0
        step = 0
        optimizer.zero_grad()

        print(f"\n  ─── Epoch {epoch+1}/{epochs} ───")

        for imgs, flows, prnu_feats, prnu_deltas, audio_ts, targets in loader:
            imgs        = imgs.to(device, non_blocking=True)
            flows       = flows.to(device, non_blocking=True)
            prnu_feats  = prnu_feats.to(device, non_blocking=True)
            prnu_deltas = prnu_deltas.to(device, non_blocking=True)
            if use_audio and audio_ts[0] is not None:
                audio_ts = audio_ts.to(device, non_blocking=True)
            else:
                audio_ts = None
            targets = targets.to(device).view(-1, 1)

            try:
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out = model(imgs, prnu_feats, None, flows, prnu_deltas, mode="video")
                    if isinstance(out, tuple):
                        logits = out[0]
                        loss = criterion(logits, targets)
                        for aux in out[1:]:
                            loss = loss + 0.2 * criterion(aux, targets)
                        loss = loss / grad_accum
                    else:
                        logits = out
                        loss = criterion(logits, targets) / grad_accum
            except Exception as _fwd_exc:
                print(f"  ⚠️  WARNING: video train forward failed at step {step+1} "
                      f"({type(_fwd_exc).__name__}: {_fwd_exc}). Skipping.")
                optimizer.zero_grad()
                del imgs, flows, prnu_feats, prnu_deltas, audio_ts, targets
                continue

            if torch.isnan(loss):
                optimizer.zero_grad()
                del imgs, flows, prnu_feats, prnu_deltas, audio_ts, targets, out, logits, loss
                continue

            scaler.scale(loss).backward()
            step += 1

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            run_loss += loss.item() * grad_accum * imgs.size(0)
            preds = (torch.sigmoid(logits.detach()) > 0.5).float()
            total   += targets.size(0)
            correct += (preds == targets).sum().item()

            # Dashboard — capture sample every 10 optimizer steps
            if step % (grad_accum * 10) == 0:
                try:
                    _confs  = torch.sigmoid(logits.detach()[:, 0]).cpu().numpy()
                    _labels = targets[:, 0].cpu().numpy()
                    _img    = imgs[0].detach().cpu()
                    _prnu   = prnu_feats[0].detach().cpu().numpy()
                    dashboard.update_frame(_img, _prnu, _confs, _labels)
                    dashboard.update_metrics(
                        step,
                        run_loss / max(total, 1),
                        correct  / max(total, 1),
                    )
                except Exception:
                    pass

            del imgs, flows, prnu_feats, prnu_deltas, audio_ts, targets, out, logits, loss

            # Progress every 10 optimizer steps
            if step % (grad_accum * 10) == 0 and total > 0:
                elapsed = time.time() - epoch_start
                print(f"    step {step:>6d} | "
                      f"loss {run_loss/max(total,1):.4f} | "
                      f"acc  {correct/max(total,1):.4f} | "
                      f"{elapsed/60:.1f} min elapsed")

                ResourceGuard.check_or_abort(
                    model, f"Vid E{epoch+1} step {step}", VIDEO_MODEL_PATH
                )
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Periodic checkpoint
            if (time.time() - last_ckpt_time) >= ckpt_interval_sec:
                _save_checkpoint(model, optimizer, epoch, step)
                last_ckpt_time = time.time()

        # Flush remaining gradients
        if step % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        elapsed = time.time() - epoch_start
        ep_loss = run_loss / max(total, 1)
        ep_acc  = correct  / max(total, 1)
        print(f"\n  ✓  Epoch {epoch+1}/{epochs}  "
              f"loss={ep_loss:.4f}  "
              f"acc={ep_acc:.4f}  "
              f"({elapsed/60:.1f} min | {total} tiles)")

        live_plot.update(epoch + 1, ep_loss, ep_acc)
        dashboard.update_metrics(step, ep_loss, ep_acc)

        scheduler.step()

        # Save epoch checkpoint
        _save_checkpoint(model, optimizer, epoch + 1, 0)
        _save_video_model(model, tile_size)

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    live_plot.close()
    dashboard.close()
    print("\n  ✅  Video training complete.")
    _save_video_model(model, tile_size)


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def interactive_main():
    """
    Interactive training setup — asks for directories, links, etc.
    """
    while True:
        print("\n" + "=" * 66)
        print("    Deep AI Detector Training Pipeline")
        print("=" * 66 + "\n")

        print("Options:")
        print("  1) Train with Videos")
        print("  2) Train with Images")
        print("  Q) Quit")
        choice_type = input("Choice [1/2/Q]: ").strip().lower()

        if choice_type == 'q':
            break

        if choice_type == '1':
            while True:
                print("\nVideo Training Mode:")
                print("  1) Local Directory (Scan your disk)")
                print("  2) External Link (YouTube)")
                print("  B) Back")
                choice_src = input("Choice [1/2/B]: ").strip().lower()
                
                if choice_src == 'b':
                    break
                
                choice_audio = input("\nTrain with Audio? (y/n) [default: y]: ").strip().lower()
                use_audio = False if choice_audio == 'n' else True

                print(f"\nProfessional Training Configuration:")
                
                # Epoch selection
                try:
                    epochs_input = input(f"  Enter target epochs [default: {VID_EPOCHS}]: ").strip()
                    epochs = int(epochs_input) if epochs_input else VID_EPOCHS
                except ValueError:
                    epochs = VID_EPOCHS
                    
                # Sampling rate selection
                try:
                    fps_input = input(f"  Enter sampling rate (FPS) [default: {VID_FPS_SAMPLE}]: ").strip()
                    fps_sample = float(fps_input) if fps_input else VID_FPS_SAMPLE
                except ValueError:
                    fps_sample = VID_FPS_SAMPLE

                # AI variants selection
                try:
                    vars_input = input(f"  Enter AI variants per frame [default: {VID_EDITS_PER_FRAME}]: ").strip()
                    edits_per_frame = int(vars_input) if vars_input else VID_EDITS_PER_FRAME
                except ValueError:
                    edits_per_frame = VID_EDITS_PER_FRAME

                video_dir = None
                temp_video_path = None
                
                if choice_src == '2':
                    print("\nYouTube mode:")
                    print("  1) Enter specific video URLs")
                    print("  2) Search by query (e.g. 'nature 4k footage', 'dashcam raw')")
                    yt_mode = input("Choice [1/2]: ").strip()

                    youtube_urls = []

                    if yt_mode == '2':
                        print("\nEnter search queries, one per line.")
                        print("  Optionally append a count: 'nature footage 4k 8'  (default: 5 per query)")
                        print("  Blank line to start downloading.")
                        raw_queries = []
                        while True:
                            line = input(f"  Query {len(raw_queries)+1}: ").strip()
                            if not line:
                                break
                            raw_queries.append(line)
                        if not raw_queries:
                            print("  No queries entered.")
                            continue
                        for q in raw_queries:
                            parts = q.rsplit(' ', 1)
                            if len(parts) == 2 and parts[1].isdigit():
                                query, n_per_query = parts[0], int(parts[1])
                            else:
                                query, n_per_query = q, 5
                            print(f"\n  Searching YouTube: '{query}'  (up to {n_per_query} videos)...")
                            found = _youtube_search_urls(query, n=n_per_query)
                            print(f"  Found {len(found)} video(s)")
                            youtube_urls.extend(found)
                    else:
                        print("\nEnter YouTube URLs (one per line, blank line when done):")
                        while True:
                            line = input(f"  URL {len(youtube_urls)+1} (or Enter to start): ").strip()
                            if not line:
                                break
                            youtube_urls.append(line)

                    if not youtube_urls:
                        print("  No videos to process.")
                        continue

                    # Section limit — important for long videos (e.g. 12-hour 4K)
                    try:
                        sec_input = input(
                            "\n  Download limit per video in minutes "
                            "(e.g. 20 for first 20 min, 0 = full video) "
                            "[default: 20]: "
                        ).strip()
                        section_minutes = int(sec_input) if sec_input else 20
                        section_minutes = None if section_minutes == 0 else section_minutes
                    except ValueError:
                        section_minutes = 20

                    # Collect AI video dir before starting the batch
                    default_ai_dir = os.path.join(DATA_DIR, 'ai_videos')
                    ai_video_input = input(
                        f"\nEnter AI video clips directory (from meta_collector) "
                        f"[default: {default_ai_dir}, leave blank to skip]: "
                    ).strip()
                    if ai_video_input:
                        ai_video_dir = ai_video_input
                    elif os.path.isdir(default_ai_dir) and any(
                        os.path.splitext(f)[1].lower() in VIDEO_EXTS
                        for f in os.listdir(default_ai_dir)
                    ):
                        print(f"  Found AI clips in {default_ai_dir} — using them.")
                        ai_video_dir = default_ai_dir
                    else:
                        ai_video_dir = None

                    print(f"\n  {len(youtube_urls)} URL(s) queued.")

                    for url_idx, yt_url in enumerate(youtube_urls, 1):
                        print(f"\n{'='*60}")
                        print(f"  VIDEO {url_idx}/{len(youtube_urls)}: {yt_url}")
                        print(f"{'='*60}")
                        temp_video_path = None
                        try:
                            temp_video_path = _download_youtube_video(
                                yt_url, section_minutes=section_minutes
                            )
                        except Exception as exc:
                            print(f"  [ERROR] Download failed: {exc}. Skipping.")
                            continue
                        try:
                            train_video_section(
                                video_dir               = TEMP_FRAMES,
                                epochs                  = epochs,
                                batch_size              = VID_BATCH,
                                lr                      = VID_LR,
                                tile_size               = VID_TILE_SIZE,
                                fps_sample              = fps_sample,
                                checkpoint_interval_min = CKPT_INTERVAL_MIN,
                                resume_path             = VIDEO_CKPT_PATH,
                                use_audio               = use_audio,
                                ai_video_dir            = ai_video_dir,
                            )
                        finally:
                            if temp_video_path and os.path.exists(temp_video_path):
                                os.remove(temp_video_path)
                                print(f"  Removed temp file: {temp_video_path}")

                    print(f"\n  Batch complete — {len(youtube_urls)} video(s) trained.")
                    break
                else:
                    default_vid_dir = os.path.join(DATA_DIR, 'real_videos')
                    video_dir = input(f"\nEnter local video directory [default: {default_vid_dir}]: ").strip()
                    if not video_dir:
                        video_dir = default_vid_dir

                # Optional: AI video clips collected by meta_collector
                default_ai_dir = os.path.join(DATA_DIR, 'ai_videos')
                ai_video_input = input(
                    f"\nEnter AI video clips directory (from meta_collector) "
                    f"[default: {default_ai_dir}, leave blank to skip]: "
                ).strip()
                if ai_video_input:
                    ai_video_dir = ai_video_input
                elif os.path.isdir(default_ai_dir) and any(
                    os.path.splitext(f)[1].lower() in VIDEO_EXTS
                    for f in os.listdir(default_ai_dir)
                ):
                    print(f"  Found AI clips in {default_ai_dir} — using them.")
                    ai_video_dir = default_ai_dir
                else:
                    ai_video_dir = None

                # Train Video Section
                train_video_section(
                    video_dir               = video_dir,
                    epochs                  = epochs,
                    batch_size              = VID_BATCH,
                    lr                      = VID_LR,
                    tile_size               = VID_TILE_SIZE,
                    fps_sample              = fps_sample,
                    checkpoint_interval_min = CKPT_INTERVAL_MIN,
                    resume_path             = VIDEO_CKPT_PATH,
                    use_audio               = use_audio,
                    ai_video_dir            = ai_video_dir,
                )

                # Cleanup if we downloaded a temporary video
                if temp_video_path and os.path.exists(temp_video_path):
                    print(f"  🧹  Cleaning up downloaded video: {temp_video_path}")
                    try:
                        os.remove(temp_video_path)
                    except Exception as e:
                        print(f"  ⚠️  Could not delete temp video: {e}")
                break

        elif choice_type == '2':
            while True:
                print("\nImage Training Mode:")
                print("  1) Local Directory (Scan your disk)")
                print("  2) External Link (Direct image or Archive URL)")
                print("  B) Back")
                choice_src = input("Choice [1/2/B]: ").strip().lower()
                
                if choice_src == 'b':
                    break

                image_dir = None
                temp_img_path = None
                
                if choice_src == '2':
                    url = input("\nEnter direct Image or Archive URL: ").strip()
                    if url:
                        temp_img_path = _download_external_image(url)
                        # If an archive was extracted, downloader returns TEMP_FRAMES
                        image_dir = temp_img_path if temp_img_path else None
                    else:
                        print("No URL provided.")
                        continue
                else:
                    default_img_dir = os.path.join(DATA_DIR, 'real')
                    image_dir = input(f"\nEnter custom local image directory [press enter to scan standard dirs]: ").strip()
                    if not image_dir:
                        image_dir = None

                # Train Image Section
                train_image_section(
                    image_data_dir = image_dir,
                    epochs         = IMG_EPOCHS,
                    batch_size     = IMG_BATCH,
                    lr             = IMG_LR,
                )

                # Cleanup downloaded file (only if it's a file, not the whole temp dir)
                if temp_img_path and os.path.exists(temp_img_path) and temp_img_path != TEMP_FRAMES:
                    print(f"  🧹  Cleaning up downloaded file: {temp_img_path}")
                    try:
                        os.remove(temp_img_path)
                    except Exception as e:
                        print(f"  ⚠️  Could not delete temp file: {e}")
                break
        else:
            print("Invalid choice.")

    print("\n" + "=" * 66)
    print("    ✅  Training pipeline finished.")
    print("=" * 66 + "\n")

    # Final sweep of TEMP_FRAMES folder to avoid bloat
    if os.path.exists(TEMP_FRAMES):
        for f in os.listdir(TEMP_FRAMES):
            p = os.path.join(TEMP_FRAMES, f)
            if os.path.isfile(p):
                 try: os.remove(p)
                 except: pass

def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--youtube', nargs='+', metavar='URL',
                        help='One or more YouTube URLs to train on directly')
    parser.add_argument('--minutes', type=float, default=20,
                        help='Minutes to download per video (0 = full, default 20)')
    parser.add_argument('--epochs',  type=int,   default=VID_EPOCHS)
    parser.add_argument('--fps',     type=float, default=VID_FPS_SAMPLE)
    parser.add_argument('--variants',type=int,   default=VID_EDITS_PER_FRAME)
    args, _ = parser.parse_known_args()

    if args.youtube:
        section_minutes = None if args.minutes == 0 else args.minutes
        ai_video_dir    = os.path.join(DATA_DIR, 'ai_videos')
        ai_video_dir    = ai_video_dir if os.path.isdir(ai_video_dir) else None

        for idx, yt_url in enumerate(args.youtube, 1):
            print(f"\n{'='*60}")
            print(f"  VIDEO {idx}/{len(args.youtube)}: {yt_url}")
            print(f"{'='*60}")
            temp_path = None
            try:
                temp_path = _download_youtube_video(
                    yt_url, section_minutes=section_minutes
                )
                train_video_section(
                    video_dir               = TEMP_FRAMES,
                    epochs                  = args.epochs,
                    batch_size              = VID_BATCH,
                    lr                      = VID_LR,
                    tile_size               = VID_TILE_SIZE,
                    fps_sample              = args.fps,
                    checkpoint_interval_min = CKPT_INTERVAL_MIN,
                    resume_path             = VIDEO_CKPT_PATH,
                    use_audio               = False,
                    ai_video_dir            = ai_video_dir,
                )
            except Exception as exc:
                print(f"  [ERROR] {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"  Deleted: {temp_path}")
        print("\nDone.")
    else:
        interactive_main()

if __name__ == '__main__':
    main()
