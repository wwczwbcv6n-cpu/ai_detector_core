"""
train_multi_source.py — Multi-Source Streaming Trainer
=======================================================

Trains DeepFusionNet (EfficientNet-B3 + SRM + FFT + YCbCr + PRNU-16)
by streaming images from multiple public sources.  Download → Train → Delete.
Only one batch of images is on disk at any time.

Supported sources
─────────────────
  Open Images v7   Google's ~1.7M real photographs (GCS public bucket).
                   No credentials required.

  Kaggle           AI-generated / real image datasets via kaggle CLI.
                   Requires: pip install kaggle + ~/.kaggle/kaggle.json

  GCS bucket       Any public or private GCS path via gsutil.
                   Requires: gcloud SDK  OR  pip install google-cloud-storage

  Local dirs       Falls back to images already on disk.

Kaggle presets  (--kaggle <name>)
──────────────────────────────────
  cifake       birdy654/cifake-real-and-ai-generated-synthetic-images
               60K real + 60K AI CIFAR-10 style images
  ai-vs-human  philosopher0/ai-vs-human-generated-dataset
               AI-generated vs human photographs
  fake-faces   xhlulu/140k-real-and-fake-faces
               140K real + StyleGAN-generated face images
  deepfakes    manjilkarki/deepfake-and-real-images
               Deepfake detection dataset

Usage
─────
  python train_multi_source.py
      Default: Open Images val (real) + local AI images

  python train_multi_source.py --kaggle cifake
      Download CIFAKE from Kaggle, train on it

  python train_multi_source.py --kaggle cifake ai-vs-human --oi-source train
      Multiple Kaggle datasets + Open Images train split (~1.7M)

  python train_multi_source.py --gcs gs://my-bucket/ai-images --gcs-label 1
      Use your own GCS bucket for AI images

  python train_multi_source.py --total-batches 200 --batch-size 64
  python train_multi_source.py --no-resume

Setup
─────
  Kaggle credentials:
    pip install kaggle
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json

  GCS (option A — gcloud SDK):
    https://cloud.google.com/sdk/docs/install

  GCS (option B — Python SDK):
    pip install google-cloud-storage
    gcloud auth application-default login
"""

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
CACHE_DIR  = os.path.join(DATA_DIR, 'multi_cache')

sys.path.insert(0, SCRIPT_DIR)
from model_prnu import DeepFusionNet
from prnu_features import extract_prnu_features_fullres

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE         = 512
DEFAULT_BATCH    = 64     # total per streaming batch (split 50/50 real/AI)
INNER_EPOCHS     = 1      # gradient passes per batch
GRAD_ACCUM_STEPS = 4
LOADER_BATCH     = 8
PRNU_DIM         = 16

FUSION_MODEL = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion.pth')
CKPT_PATH    = os.path.join(MODELS_DIR, 'checkpoint_latest.pth')
PROGRESS_F   = os.path.join(MODELS_DIR, 'multi_progress.json')

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

# ── Open Images CSV URLs (Google GCS public) ───────────────────────────────────
# v7 paths (current as of 2024)
OI_CSV = {
    'val':   'https://storage.googleapis.com/openimages/v7/oidv7-image-ids-boxable-validation.csv',
    'train': 'https://storage.googleapis.com/openimages/v7/oidv7-image-ids-boxable.csv',
}
# Fallback v6 paths
OI_CSV_V6 = {
    'val':   'https://storage.googleapis.com/openimages/v6/validation-images-boxable.csv',
    'train': 'https://storage.googleapis.com/openimages/v6/train-images-boxable.csv',
}

# ── Kaggle dataset presets ────────────────────────────────────────────────────
KAGGLE_PRESETS = {
    'cifake': {
        'dataset': 'birdy654/cifake-real-and-ai-generated-synthetic-images',
        'desc':    '60K real + 60K AI CIFAR-10 images',
        'real':    ['test/REAL', 'train/REAL'],
        'fake':    ['test/FAKE', 'train/FAKE'],
    },
    'ai-vs-human': {
        'dataset': 'philosopher0/ai-vs-human-generated-dataset',
        'desc':    'AI-generated vs human photographs',
        'real':    ['real'],
        'fake':    ['ai'],
    },
    'fake-faces': {
        'dataset': 'xhlulu/140k-real-and-fake-faces',
        'desc':    '140K real + StyleGAN-generated faces',
        'real':    ['real_vs_fake/real-vs-fake/train/real'],
        'fake':    ['real_vs_fake/real-vs-fake/train/fake'],
    },
    'deepfakes': {
        'dataset': 'manjilkarki/deepfake-and-real-images',
        'desc':    'Deepfake detection images',
        'real':    ['Dataset/Real'],
        'fake':    ['Dataset/Fake'],
    },
}

# ── Local fallback directories ─────────────────────────────────────────────────
LOCAL_AI_DIRS = [
    os.path.join(DATA_DIR, 'test',  'FAKE'),
    os.path.join(DATA_DIR, 'ai'),
    os.path.join(DATA_DIR, 'generated_ai', 'images'),
    os.path.join(DATA_DIR, 'edited_ai',    'images'),
]
LOCAL_REAL_DIRS = [
    os.path.join(DATA_DIR, 'test',     'REAL'),
    os.path.join(DATA_DIR, 'real'),
    os.path.join(DATA_DIR, 'personal'),
]


# ═════════════════════════════════════════════════════════════════════════════
#  Resource Safety Guard
# ═════════════════════════════════════════════════════════════════════════════

class ResourceGuard:
    MIN_DISK_MB = 800
    MIN_RAM_MB  = 400
    MAX_RAM_PCT = 92
    MAX_GPU_PCT = 95

    @staticmethod
    def _disk_mb(path='.'):
        try:    return shutil.disk_usage(os.path.abspath(path)).free / 1024 ** 2
        except: return float('inf')

    @staticmethod
    def _ram_free_mb():
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) / 1024
        except: pass
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
        except: return 0

    @staticmethod
    def _gpu_pct():
        try:
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated()
                t = torch.cuda.get_device_properties(0).total_memory
                return a / t * 100 if t else 0
        except: pass
        return 0

    @classmethod
    def check(cls):
        if cls._disk_mb(MODELS_DIR) < cls.MIN_DISK_MB:
            return False, f'Disk low: {cls._disk_mb(MODELS_DIR):.0f} MB free'
        if cls._ram_free_mb() < cls.MIN_RAM_MB:
            return False, f'RAM low: {cls._ram_free_mb():.0f} MB free'
        if cls._ram_pct() > cls.MAX_RAM_PCT:
            return False, f'RAM usage {cls._ram_pct():.0f}%'
        if cls._gpu_pct() > cls.MAX_GPU_PCT:
            return False, f'GPU VRAM {cls._gpu_pct():.0f}%'
        return True, ''

    @classmethod
    def check_or_abort(cls, model, ctx=''):
        ok, reason = cls.check()
        if not ok:
            print(f'\n  EMERGENCY STOP: {reason}  [{ctx}]')
            print('  Saving model …')
            _save_model(model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('  Re-run to resume.')
            sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════════════

def _make_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class StreamBatchDataset(Dataset):
    def __init__(self, real_paths: List[str], fake_paths: List[str], transform):
        self.items     = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 'black')

        try:
            prnu = extract_prnu_features_fullres(img)
            if not np.isfinite(prnu).all():
                prnu = np.zeros(PRNU_DIM, dtype=np.float32)
        except Exception:
            prnu = np.zeros(PRNU_DIM, dtype=np.float32)

        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(prnu).float(), torch.tensor(label, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  Source: Open Images (Google GCS — no credentials needed)
# ═════════════════════════════════════════════════════════════════════════════

def _load_oi_urls(split: str) -> List[str]:
    """
    Download (and cache) the Open Images metadata CSV. Returns image URLs.
    Tries v7 URL first, then v6 fallback.  Returns [] gracefully on failure.
    """
    import csv
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'oi_{split}.csv')

    if not os.path.exists(cache):
        for label, url in [('v7', OI_CSV[split]), ('v6', OI_CSV_V6[split])]:
            print(f'  Trying Open Images {label} CSV: {url}')
            tmp = cache + '.tmp'
            try:
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, cache)
                print(f'  Cached ({label}): {cache}')
                break
            except Exception as e:
                print(f'  {label} failed: {e}')
                try: os.remove(tmp)
                except Exception: pass
        else:
            print('  Open Images CSV unavailable — skipping OI source.')
            return []

    with open(cache, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # OI CSVs use 'OriginalURL' or 'original_url' depending on version
        urls = []
        for row in reader:
            url = (row.get('OriginalURL') or row.get('original_url') or '').strip()
            if url:
                urls.append(url)

    if not urls:
        print(f'  Open Images CSV parsed but found no URLs — may be ID-only format.')
        print(f'  Skipping OI source (use --no-oi to suppress this message).')
        return []

    print(f'  Open Images {split}: {len(urls):,} URLs')
    return urls


def _dl_one(url: str, dest: str, timeout: int = 20) -> bool:
    """Download a single URL to dest. Returns True on success."""
    tmp = dest + '.tmp'
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0 (AIDetectorTrainer/2.0)'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        Image.open(__import__('io').BytesIO(data)).verify()   # confirm readable
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, dest)
        return True
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        return False


def _download_urls(urls: List[str], dest_dir: str, workers: int = 8) -> List[str]:
    """Parallel-download a list of URLs. Returns successful local paths."""
    os.makedirs(dest_dir, exist_ok=True)
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, url in enumerate(urls):
            ext  = Path(url.split('?')[0]).suffix.lower()
            ext  = ext if ext in _IMG_EXTS else '.jpg'
            dest = os.path.join(dest_dir, f'{i:05d}{ext}')
            futures[pool.submit(_dl_one, url, dest)] = dest
    return [dest for fut, dest in futures.items() if fut.result()]


# ═════════════════════════════════════════════════════════════════════════════
#  Source: Kaggle
# ═════════════════════════════════════════════════════════════════════════════

def _check_kaggle() -> bool:
    """Return True if kaggle CLI is installed and credentials exist."""
    creds = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(creds):
        print('  Kaggle: credentials not found at ~/.kaggle/kaggle.json')
        print('  Setup:')
        print('    1. Sign in at kaggle.com → Account → API → Create New Token')
        print('    2. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/')
        print('    3. chmod 600 ~/.kaggle/kaggle.json')
        return False
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('  Kaggle: CLI not found.  Install: pip install kaggle')
        return False


def _kaggle_download(preset: dict, dest_dir: str) -> Optional[str]:
    """
    Download and unzip a Kaggle dataset into dest_dir.
    Returns dest_dir on success, None on failure.
    Skips download if dest_dir already exists.
    """
    if os.path.isdir(dest_dir) and any(os.scandir(dest_dir)):
        print(f'  Kaggle: using cached download at {dest_dir}')
        return dest_dir

    dataset = preset['dataset']
    print(f'  Kaggle: downloading {dataset}')
    print(f'         ({preset["desc"]})')
    print(f'  Destination: {dest_dir}')
    os.makedirs(dest_dir, exist_ok=True)

    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset, '--unzip', '-p', dest_dir],
            check=True,
        )
        print(f'  Kaggle: download complete')
        return dest_dir
    except subprocess.CalledProcessError as e:
        print(f'  Kaggle: download failed — {e}')
        shutil.rmtree(dest_dir, ignore_errors=True)
        return None


def _scan_subdirs(base: str, subdirs: List[str]) -> List[str]:
    """
    Scan base/subdir for image files.  Case-insensitive subdir lookup.
    Returns a flat list of absolute paths.
    """
    paths = []
    for sd in subdirs:
        d = os.path.join(base, sd)
        if not os.path.isdir(d):
            # Try case-insensitive match
            parent = os.path.dirname(d)
            target = os.path.basename(d).lower()
            if os.path.isdir(parent):
                for entry in os.listdir(parent):
                    if entry.lower() == target:
                        d = os.path.join(parent, entry)
                        break
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if Path(f).suffix.lower() in _IMG_EXTS:
                        paths.append(os.path.join(root, f))
    return paths


# ═════════════════════════════════════════════════════════════════════════════
#  Source: Google Cloud Storage (gsutil)
# ═════════════════════════════════════════════════════════════════════════════

def _check_gsutil() -> bool:
    try:
        subprocess.run(['gsutil', 'version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('  gsutil not found.  Install: https://cloud.google.com/sdk/docs/install')
        print('                     or:  pip install google-cloud-storage')
        return False


def _gcs_list_images(gcs_path: str) -> List[str]:
    """List image files in a GCS path. Returns gs:// URIs."""
    print(f'  GCS: listing files in {gcs_path} …')
    try:
        result = subprocess.run(
            ['gsutil', 'ls', '-r', gcs_path.rstrip('/') + '/**'],
            capture_output=True, text=True, check=True,
        )
        uris = [
            line.strip() for line in result.stdout.splitlines()
            if Path(line.strip()).suffix.lower() in _IMG_EXTS
        ]
        print(f'  GCS: {len(uris):,} images found')
        return uris
    except subprocess.CalledProcessError as e:
        print(f'  GCS ls failed: {e.stderr.strip()}')
        return []


def _gcs_download_batch(gcs_uris: List[str], dest_dir: str) -> List[str]:
    """
    Download a batch of GCS URIs to dest_dir using gsutil -m cp.
    Returns local paths of successfully downloaded files.
    """
    if not gcs_uris:
        return []
    os.makedirs(dest_dir, exist_ok=True)
    try:
        subprocess.run(
            ['gsutil', '-m', 'cp'] + gcs_uris + [dest_dir + '/'],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f'  GCS cp error: {e.stderr.strip() if e.stderr else e}')

    return [
        os.path.join(dest_dir, os.path.basename(uri))
        for uri in gcs_uris
        if os.path.exists(os.path.join(dest_dir, os.path.basename(uri)))
    ]


# ═════════════════════════════════════════════════════════════════════════════
#  Source: Local directories
# ═════════════════════════════════════════════════════════════════════════════

def _scan_local(dirs: List[str]) -> List[str]:
    """
    Scan a list of directories for images.
    Deduplicates symlinked paths using realpath.
    """
    seen  = set()
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        real_d = os.path.realpath(d)
        if real_d in seen:
            continue
        seen.add(real_d)
        for root, _, files in os.walk(d):
            for f in files:
                if Path(f).suffix.lower() in _IMG_EXTS:
                    paths.append(os.path.join(root, f))
    return paths


# ═════════════════════════════════════════════════════════════════════════════
#  Model helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_model(device) -> DeepFusionNet:
    """Load existing DeepFusionNet weights if available, else init fresh."""
    model = DeepFusionNet().to(device)
    for path in [FUSION_MODEL, CKPT_PATH]:
        if not os.path.exists(path):
            continue
        try:
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
            print(f'  Loaded weights: {path}')
            return model
        except Exception as e:
            print(f'  Could not load {path}: {e}')
    print('  Initialising fresh DeepFusionNet (no saved weights found)')
    return model


def _save_model(model: DeepFusionNet):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), FUSION_MODEL)
    print(f'  Model saved → {FUSION_MODEL}')


def _save_ckpt(model: DeepFusionNet, optimizer, done: int, loss: float, acc: float):
    os.makedirs(MODELS_DIR, exist_ok=True)
    payload = {
        'batches_done':     done,
        'loss':             loss,
        'acc':              acc,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, CKPT_PATH)


# ═════════════════════════════════════════════════════════════════════════════
#  Progress (atomic JSON)
# ═════════════════════════════════════════════════════════════════════════════

def _load_progress() -> dict:
    if os.path.exists(PROGRESS_F):
        try:
            with open(PROGRESS_F) as f:
                p = json.load(f)
            print(f'  Resuming from batch {p.get("batches_done", 0)}')
            return p
        except Exception as e:
            print(f'  Progress file unreadable ({e}), starting fresh')
    return {}


def _save_progress(state: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp = PROGRESS_F + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, PROGRESS_F)


# ═════════════════════════════════════════════════════════════════════════════
#  Training loop (one streaming batch)
# ═════════════════════════════════════════════════════════════════════════════

def _train_one_batch(
    model: DeepFusionNet,
    optimizer,
    scaler,
    loader: DataLoader,
    device,
) -> Tuple[float, float]:
    """
    Run one pass through the DataLoader with AMP + gradient accumulation.
    Returns (avg_loss, accuracy).
    """
    use_amp   = device.type == 'cuda'
    criterion = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]

    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for i, (imgs, prnu, labels) in enumerate(loader):
        imgs   = imgs.to(device)
        prnu   = prnu.to(device)
        labels = labels.to(device).view(-1, 1)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(imgs, prnu)
            loss   = criterion(logits, labels) / GRAD_ACCUM_STEPS

        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_sum += loss.item() * GRAD_ACCUM_STEPS * imgs.size(0)
        preds  = (torch.sigmoid(logits.detach()) > 0.5).float()
        total  += labels.size(0)
        correct += (preds == labels).sum().item()
        del imgs, prnu, labels, logits, loss

    # Flush remaining gradient accumulation
    if len(loader) % GRAD_ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss_sum / max(total, 1), correct / max(total, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Source Streaming Trainer — DeepFusionNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--kaggle', nargs='+', metavar='PRESET',
        choices=list(KAGGLE_PRESETS.keys()),
        help='Kaggle preset(s): ' + ', '.join(KAGGLE_PRESETS.keys()),
    )
    parser.add_argument(
        '--gcs', metavar='GCS_PATH',
        help='GCS path for additional images (e.g. gs://bucket/ai-images)',
    )
    parser.add_argument(
        '--gcs-label', type=int, choices=[0, 1], default=1,
        help='Label for GCS images: 0=real, 1=AI (default: 1)',
    )
    parser.add_argument(
        '--oi-source', choices=['val', 'train'], default='val',
        help="Open Images split for real images (default: val ~40K; train ~1.7M)",
    )
    parser.add_argument(
        '--no-oi', action='store_true',
        help='Disable Open Images (use only Kaggle/local for real images)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=DEFAULT_BATCH,
        help=f'Total images per batch, split 50/50 real/AI (default: {DEFAULT_BATCH})',
    )
    parser.add_argument(
        '--total-batches', type=int, default=500,
        help='Number of streaming batches to run (default: 500)',
    )
    parser.add_argument(
        '--save-every', type=int, default=10,
        help='Save model every N batches (default: 10)',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)',
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Parallel download threads for Open Images (default: 8)',
    )
    parser.add_argument(
        '--num-workers', type=int, default=0,
        help='DataLoader worker processes (default: 0 — safe for all platforms)',
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help='Ignore existing progress and start from batch 1',
    )
    args = parser.parse_args()

    print('\n=== Multi-Source Streaming Trainer — DeepFusionNet ===\n')
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'  Device : {device}  |  AMP : {use_amp}')

    batch_half = args.batch_size // 2
    transform  = _make_transform()

    # ── Model & optimiser ─────────────────────────────────────────────────
    print('\n[1] Initialising model …')
    model     = _load_model(device)
    model.param_summary()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume progress ───────────────────────────────────────────────────
    print('\n[2] Loading progress …')
    progress     = {} if args.no_resume else _load_progress()
    batches_done = progress.get('batches_done', 0)
    oi_offset    = progress.get('oi_offset', 0)
    kr_offset    = progress.get('kaggle_real_offset', 0)
    kf_offset    = progress.get('kaggle_fake_offset', 0)
    gcs_offset   = progress.get('gcs_offset', 0)

    # ── Open Images URLs (real, no credentials needed) ────────────────────
    oi_urls: List[str] = []
    if not args.no_oi:
        print(f'\n[3] Loading Open Images {args.oi_source} URLs …')
        oi_urls = _load_oi_urls(args.oi_source)

    # ── Kaggle datasets ───────────────────────────────────────────────────
    kaggle_real: List[str] = []
    kaggle_fake: List[str] = []

    if args.kaggle:
        print('\n[4] Setting up Kaggle sources …')
        if not _check_kaggle():
            print('  Skipping Kaggle (setup required).')
            args.kaggle = []
        else:
            for name in args.kaggle:
                preset = KAGGLE_PRESETS[name]
                dest   = os.path.join(CACHE_DIR, 'kaggle', name)
                result = _kaggle_download(preset, dest)
                if result is None:
                    continue
                real_p = _scan_subdirs(dest, preset['real'])
                fake_p = _scan_subdirs(dest, preset['fake'])
                print(f'  {name}: {len(real_p):,} real + {len(fake_p):,} fake')
                kaggle_real.extend(real_p)
                kaggle_fake.extend(fake_p)

            # Deterministic shuffle (seed=42) so offsets work on resume
            rng = random.Random(42)
            rng.shuffle(kaggle_real)
            rng.shuffle(kaggle_fake)
            print(f'  Kaggle total: {len(kaggle_real):,} real + {len(kaggle_fake):,} fake')

    # ── GCS source ────────────────────────────────────────────────────────
    gcs_uris: List[str] = []
    if args.gcs:
        print('\n[5] Listing GCS files …')
        if _check_gsutil():
            gcs_uris = _gcs_list_images(args.gcs)
        else:
            print('  Skipping GCS source.')

    # ── Local fallback ────────────────────────────────────────────────────
    print('\n[6] Scanning local directories …')
    local_real = _scan_local(LOCAL_REAL_DIRS)
    local_ai   = _scan_local(LOCAL_AI_DIRS)
    print(f'  Local real : {len(local_real):,}')
    print(f'  Local AI   : {len(local_ai):,}')

    print(f'\n  Starting at batch {batches_done + 1} / {args.total_batches}')
    print(f'  Batch size : {args.batch_size} ({batch_half} real + {batch_half} AI)')
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  Streaming loop
    # ══════════════════════════════════════════════════════════════════════
    loss, acc = 0.0, 0.0

    while batches_done < args.total_batches:
        bn       = batches_done + 1
        temp_dir = os.path.join(CACHE_DIR, f'batch_{bn:06d}')
        print(f'─── Batch {bn}/{args.total_batches} ───────────────────────────────────')

        # ── Collect REAL images (priority: Kaggle → GCS → OI → local) ────

        real_paths: List[str] = []

        # 1. Kaggle real
        if kaggle_real and len(real_paths) < batch_half:
            need   = batch_half - len(real_paths)
            end    = min(kr_offset + need, len(kaggle_real))
            chunk  = kaggle_real[kr_offset:end]
            real_paths += chunk
            kr_offset = end % len(kaggle_real)

        # 2. GCS real (if gcs-label == 0)
        if gcs_uris and args.gcs_label == 0 and len(real_paths) < batch_half:
            need       = batch_half - len(real_paths)
            end        = min(gcs_offset + need, len(gcs_uris))
            batch_uris = gcs_uris[gcs_offset:end]
            gcs_offset = end % len(gcs_uris)
            if batch_uris:
                dl = _gcs_download_batch(batch_uris, os.path.join(temp_dir, 'gcs_real'))
                real_paths += dl
                print(f'  GCS real: +{len(dl)}')

        # 3. Open Images (fill remaining)
        if oi_urls and not args.no_oi and len(real_paths) < batch_half:
            need       = batch_half - len(real_paths)
            end        = min(oi_offset + need, len(oi_urls))
            batch_urls = oi_urls[oi_offset:end]
            if not batch_urls:           # wrap around
                oi_offset  = 0
                end        = need
                batch_urls = oi_urls[:need]
            t0 = time.time()
            dl = _download_urls(batch_urls, os.path.join(temp_dir, 'oi'), args.workers)
            real_paths += dl
            oi_offset   = end % len(oi_urls)
            print(f'  Open Images: +{len(dl)} in {time.time()-t0:.1f}s')

        # 4. Local real fallback
        if local_real and len(real_paths) < batch_half:
            need = batch_half - len(real_paths)
            real_paths += random.sample(local_real, min(need, len(local_real)))

        # ── Collect AI images (priority: Kaggle → GCS → local) ───────────

        fake_paths: List[str] = []

        # 1. Kaggle fake
        if kaggle_fake and len(fake_paths) < batch_half:
            need   = batch_half - len(fake_paths)
            end    = min(kf_offset + need, len(kaggle_fake))
            chunk  = kaggle_fake[kf_offset:end]
            fake_paths += chunk
            kf_offset = end % len(kaggle_fake)

        # 2. GCS AI (if gcs-label == 1)
        if gcs_uris and args.gcs_label == 1 and len(fake_paths) < batch_half:
            need       = batch_half - len(fake_paths)
            end        = min(gcs_offset + need, len(gcs_uris))
            batch_uris = gcs_uris[gcs_offset:end]
            gcs_offset = end % len(gcs_uris)
            if batch_uris:
                dl = _gcs_download_batch(batch_uris, os.path.join(temp_dir, 'gcs_ai'))
                fake_paths += dl
                print(f'  GCS AI: +{len(dl)}')

        # 3. Local AI fallback
        if local_ai and len(fake_paths) < batch_half:
            need = batch_half - len(fake_paths)
            fake_paths += random.sample(local_ai, min(need, len(local_ai)))

        # ── Skip if nothing to train on ───────────────────────────────────
        if not real_paths and not fake_paths:
            print('  No images available — skipping batch.')
            batches_done += 1
            continue

        print(f'  Batch: {len(real_paths)} real + {len(fake_paths)} AI '
              f'= {len(real_paths) + len(fake_paths)} images')

        # ── Build dataset + loader ─────────────────────────────────────────
        ds = StreamBatchDataset(real_paths, fake_paths, transform)
        dl = DataLoader(
            ds,
            batch_size  = LOADER_BATCH,
            shuffle     = True,
            num_workers = args.num_workers,
            pin_memory  = False,
        )

        # ── Train ─────────────────────────────────────────────────────────
        for ep in range(INNER_EPOCHS):
            loss, acc = _train_one_batch(model, optimizer, scaler, dl, device)
            ep_s = f'ep {ep+1}/{INNER_EPOCHS} ' if INNER_EPOCHS > 1 else ''
            print(f'  {ep_s}loss={loss:.4f}  acc={acc:.4f}')

        # ── Delete downloaded images ───────────────────────────────────────
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

        # ── Save progress + checkpoint ─────────────────────────────────────
        batches_done += 1
        _save_progress({
            'batches_done':         batches_done,
            'oi_offset':            oi_offset,
            'kaggle_real_offset':   kr_offset,
            'kaggle_fake_offset':   kf_offset,
            'gcs_offset':           gcs_offset,
            'kaggle_presets':       sorted(args.kaggle or []),
        })
        _save_ckpt(model, optimizer, batches_done, loss, acc)

        if batches_done % args.save_every == 0:
            _save_model(model)
            print(f'  Progress: {batches_done}/{args.total_batches} batches done')

        # ── Resource guard + memory cleanup ───────────────────────────────
        ResourceGuard.check_or_abort(model, f'batch {batches_done}')
        del ds, dl
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Final save ────────────────────────────────────────────────────────
    print(f'\n=== Done: {batches_done} batches ===')
    _save_model(model)
    _save_ckpt(model, optimizer, batches_done, loss, acc)
    print(f'  Model      : {FUSION_MODEL}')
    print(f'  Checkpoint : {CKPT_PATH}')


if __name__ == '__main__':
    main()
