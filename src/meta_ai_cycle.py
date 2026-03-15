"""
meta_ai_cycle.py — Endless download → PRNU-recover → train → delete cycle
===========================================================================

Downloads AI videos from meta.ai/vibes in small sub-batches, extracts
frames one-at-a-time directly from the video (never a full frame list in
RAM), applies PRNURecoveryNet to undo H.264 compression damage, trains
DeepFusionNet v5, then deletes every file before moving on.

Runs forever until Ctrl+C.  Safe for multi-day runs.

Memory strategy
  • Sub-batch of --sub_batch videos downloaded at once  (default 10)
  • Each video is processed frame-by-frame (never loaded fully)
  • Video deleted immediately after its last frame is trained
  • gc.collect() + cuda.empty_cache() after every video
  • RAM / VRAM / disk checked before every download — pauses if high
  • Emergency checkpoint saved if memory critical

Usage:
    python src/meta_ai_cycle.py
    python src/meta_ai_cycle.py --cycle_size 100 --sub_batch 10
    python src/meta_ai_cycle.py --frames_per_video 30 --lr 2e-5
"""

import argparse
import csv
import datetime
import gc
import io
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

META_VIBES_URL  = "https://www.meta.ai/vibes"
STATUS_FILE     = os.path.join(REPO_ROOT, 'models', 'cycle_status.json')
VIDEO_EXTS      = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

# Open Images v7 validation — ~41k real camera photos (Flickr originals)
OI_VAL_CSV_URL  = ("https://storage.googleapis.com/openimages/2018_04/"
                   "validation/validation-images-with-rotation.csv")
OI_CSV_CACHE    = os.path.join(REPO_ROOT, 'data', 'oi_val_meta.csv')

# ── Training hyper-params ─────────────────────────────────────────────────────
FRAMES_PER_VIDEO  = 25
ACCUM_STEPS       = 4
LR                = 3e-5
LAMBDA_PRNU       = 0.35
LAMBDA_HALLUC     = 0.15
LAMBDA_PRNU_SPATIAL = 0.20

IMG_SIZE = 512

# ── Memory thresholds ─────────────────────────────────────────────────────────
RAM_PAUSE_PCT    = 80    # pause download if RAM > 80%
RAM_CRITICAL_PCT = 90    # emergency checkpoint + GC if RAM > 90%
VRAM_PAUSE_PCT   = 85    # pause if VRAM > 85%
DISK_MIN_MB      = 800   # refuse to download if disk < 800 MB free
RAM_MIN_MB       = 600   # refuse to download if RAM free < 600 MB

# ── Transforms ────────────────────────────────────────────────────────────────
_TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.15),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5))], p=0.35
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  Memory monitor (no psutil — pure /proc/meminfo + torch)
# ═══════════════════════════════════════════════════════════════════════════════

def ram_info() -> dict:
    """Read RAM stats from /proc/meminfo. Returns dict with MB and percent."""
    info = {}
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':')
                info[k.strip()] = int(v.split()[0])   # kB
        total = info.get('MemTotal', 1)
        avail = info.get('MemAvailable', total)
        return {
            'total_mb': total / 1024,
            'avail_mb': avail / 1024,
            'pct':      (total - avail) / total * 100,
        }
    except Exception:
        return {'total_mb': 0, 'avail_mb': 9999, 'pct': 0}


def vram_info(device) -> dict:
    """VRAM stats. Returns dict with MB and percent."""
    if device.type != 'cuda':
        return {'alloc_mb': 0, 'total_mb': 0, 'pct': 0}
    try:
        alloc = torch.cuda.memory_allocated(device) / 1024 ** 2
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        return {'alloc_mb': alloc, 'total_mb': total, 'pct': alloc / total * 100}
    except Exception:
        return {'alloc_mb': 0, 'total_mb': 0, 'pct': 0}


def disk_free_mb(path=REPO_ROOT) -> float:
    try:
        return shutil.disk_usage(path).free / 1024 ** 2
    except Exception:
        return 9999.0


def mem_status(device) -> str:
    """One-line memory summary for display."""
    r = ram_info()
    v = vram_info(device)
    disk = disk_free_mb()
    parts = [f"RAM {r['pct']:.0f}% ({r['avail_mb']:.0f}MB free)"]
    if v['total_mb'] > 0:
        parts.append(f"VRAM {v['pct']:.0f}% ({v['alloc_mb']:.0f}/{v['total_mb']:.0f}MB)")
    parts.append(f"Disk {disk:.0f}MB free")
    return "  ".join(parts)


def free_memory(device):
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def wait_for_memory(device, context: str = ""):
    """
    Block until RAM and VRAM are below pause thresholds.
    Frees caches and waits in 5-second intervals.
    """
    while True:
        r = ram_info()
        v = vram_info(device)

        ram_ok  = r['pct'] < RAM_PAUSE_PCT and r['avail_mb'] > RAM_MIN_MB
        vram_ok = v['pct'] < VRAM_PAUSE_PCT or device.type != 'cuda'
        disk_ok = disk_free_mb() > DISK_MIN_MB

        if ram_ok and vram_ok and disk_ok:
            return

        reasons = []
        if not ram_ok:
            reasons.append(f"RAM {r['pct']:.0f}%")
        if not vram_ok:
            reasons.append(f"VRAM {v['pct']:.0f}%")
        if not disk_ok:
            reasons.append(f"Disk {disk_free_mb():.0f}MB")

        print(f"\n  [mem] Waiting — {', '.join(reasons)} ({context})")
        free_memory(device)
        time.sleep(5)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(device):
    from model_prnu import DeepFusionNet
    print("[init] Loading model ...")

    # Try v5 checkpoint directly first
    v5_path = os.path.join(REPO_ROOT, 'models', 'ai_detector_prnu_fusion_v5.pth')
    if os.path.exists(v5_path):
        try:
            model = DeepFusionNet().to(device)
            try:
                _sd = torch.load(v5_path, map_location=device, weights_only=True)
            except Exception:
                _sd = torch.load(v5_path, map_location=device)
            model.load_state_dict(_sd)
            print(f"  Loaded v5 checkpoint from {v5_path}")
            return model
        except Exception as e:
            print(f"  v5 load failed ({e}), initialising fresh.")

    # Try cycle checkpoint
    cycle_path = os.path.join(REPO_ROOT, 'models', 'cycle_checkpoint.pth')
    if os.path.exists(cycle_path):
        try:
            model = DeepFusionNet().to(device)
            try:
                _sd = torch.load(cycle_path, map_location=device, weights_only=True)
            except Exception:
                _sd = torch.load(cycle_path, map_location=device)
            model.load_state_dict(_sd)
            print(f"  Resumed from cycle checkpoint.")
            return model
        except Exception as e:
            print(f"  Cycle checkpoint load failed ({e}), starting fresh.")

    model = DeepFusionNet().to(device)
    print("  Initialised fresh DeepFusionNet v5.")
    return model


def save_model(model, path=None):
    models_dir = os.path.join(REPO_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    if path is None:
        path = os.path.join(models_dir, 'ai_detector_prnu_fusion_v5.pth')
    torch.save(model.state_dict(), path)
    # Also save a cycle-specific checkpoint for resuming
    cycle_path = os.path.join(models_dir, 'cycle_checkpoint.pth')
    torch.save(model.state_dict(), cycle_path)
    print(f"  Checkpoint → {path}")


def load_recovery_net(device):
    try:
        from prnu_recovery import build_prnu_recovery_net
        net = build_prnu_recovery_net(device=device)
        net.eval()
        print("[init] PRNU recovery net loaded.")
        return net
    except Exception as e:
        print(f"[init] PRNU recovery net unavailable: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Real image pool — Open Images v7 validation
# ═══════════════════════════════════════════════════════════════════════════════

_oi_urls: list[str] = []
_oi_idx: int = 0


def _load_oi_urls() -> list[str]:
    global _oi_urls
    if _oi_urls:
        return _oi_urls

    os.makedirs(os.path.dirname(OI_CSV_CACHE), exist_ok=True)
    if not os.path.exists(OI_CSV_CACHE):
        print("  Downloading Open Images metadata CSV (~10 MB) — one-time setup ...")
        tmp = OI_CSV_CACHE + '.tmp'
        urllib.request.urlretrieve(OI_VAL_CSV_URL, tmp)
        os.replace(tmp, OI_CSV_CACHE)
        print("  Done.")

    with open(OI_CSV_CACHE, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            url = (row.get('Thumbnail300KURL', '').strip()
                   or row.get('OriginalURL', '').strip())
            if url:
                _oi_urls.append(url)

    random.shuffle(_oi_urls)
    print(f"  Open Images: {len(_oi_urls):,} real image URLs ready")
    return _oi_urls


def _next_oi_urls(count: int) -> list[str]:
    global _oi_idx
    urls = _load_oi_urls()
    if not urls:
        return []
    out = []
    for _ in range(count):
        out.append(urls[_oi_idx % len(urls)])
        _oi_idx += 1
    return out


def _dl_one_image(url: str, dest: str) -> tuple[str, bool]:
    tmp = dest + '.tmp'
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0 (compatible; AIDetector/1.0)'}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        Image.open(io.BytesIO(data)).verify()   # reject corrupt files
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, dest)
        return dest, True
    except Exception:
        for p in (tmp, dest):
            try: os.remove(p)
            except Exception: pass
        return dest, False


# ── Civitai AI image pool ──────────────────────────────────────────────────────
CIVITAI_API = "https://civitai.com/api/v1/images?limit={n}&sort=Newest&nsfw=false&page={page}"
_civitai_page: int = 1


def download_civitai_images(dest_dir: str, count: int) -> list[str]:
    """
    Download `count` AI-generated images from Civitai public API.
    All Civitai images are generated by Stable Diffusion / Flux / DALL-E etc.
    No login required.
    """
    global _civitai_page
    import json as _json
    os.makedirs(dest_dir, exist_ok=True)

    paths: list[str] = []
    per_page = min(count * 2, 100)   # Civitai max = 200, we use 100 to be safe

    while len(paths) < count:
        url = CIVITAI_API.format(n=per_page, page=_civitai_page)
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = _json.loads(resp.read())
            items = data.get('items', [])
            if not items:
                _civitai_page = 1   # wrap around
                break
        except Exception as e:
            print(f"  [civitai] API error: {e}")
            break

        _civitai_page += 1
        # Respect Civitai rate limit: 10 req/min → 6 s between pages
        time.sleep(6)
        img_urls = [
            it['url'] for it in items
            if it.get('type') == 'image' and not it.get('nsfw', True)
               and it.get('nsfwLevel', 'None') == 'None'
        ]

        futures = {}
        with ThreadPoolExecutor(max_workers=10) as pool:
            for i, img_url in enumerate(img_urls):
                dest = os.path.join(dest_dir, f'ai_{_civitai_page}_{i:04d}.jpg')
                futures[pool.submit(_dl_one_image, img_url, dest)] = dest
            for fut in as_completed(futures):
                p, ok = fut.result()
                if ok:
                    paths.append(p)
                if len(paths) >= count:
                    break

    return paths[:count]


# ── Real image pool (Open Images) ─────────────────────────────────────────────

def download_real_images(dest_dir: str, count: int) -> list[str]:
    """Download `count` real camera images from Open Images into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    # Fetch 2× URLs to compensate for download failures (~20–30% fail rate)
    urls = _next_oi_urls(count * 2)
    futures = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for i, url in enumerate(urls):
            ext  = os.path.splitext(url.split('?')[0])[-1].lower()
            ext  = ext if ext in {'.jpg', '.jpeg', '.png', '.webp'} else '.jpg'
            dest = os.path.join(dest_dir, f'real_{i:05d}{ext}')
            futures[pool.submit(_dl_one_image, url, dest)] = dest
        paths = [r[0] for f in as_completed(futures)
                 if (r := f.result())[1]]
    return paths[:count]


def train_real_images(image_paths: list[str], model, optimizer, scaler,
                      device, recovery_net, criterion,
                      target_label: float = 0.0) -> tuple[float, int]:
    """
    Train on still images with a given label (0 = real, 1 = AI).
    Mirrors train_video() but for still images.
    Deletes each image after training.
    """
    if not image_paths:
        return 0.0, 0

    model.train()
    label        = torch.full((1, 1), target_label, device=device)
    use_amp      = (device.type == 'cuda')
    running_loss = 0.0
    trained      = 0
    optimizer.zero_grad()

    bar = tqdm(image_paths, desc='  real images', leave=False,
               ncols=110, unit='img',
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} {postfix}]')

    for step, img_path in enumerate(bar):
        try:
            pil = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        rgb = np.array(pil)

        prnu_feat, prnu_map = _recover_and_extract(rgb, recovery_net, device)
        prnu_feat = prnu_feat.to(device, non_blocking=True)
        prnu_map  = prnu_map.to(device, non_blocking=True)

        img_t = _TRAIN_TF(pil).unsqueeze(0).to(device, non_blocking=True)
        del pil, rgb

        with torch.amp.autocast(device.type, enabled=use_amp):
            out = model(img_t, prnu_feat, prnu_map)
            if isinstance(out, tuple):
                logit, prnu_aux, halluc_aux, prnu_spatial_aux = out
                loss = (
                    criterion(logit,            label)
                    + LAMBDA_PRNU         * criterion(prnu_aux,         label)
                    + LAMBDA_HALLUC       * criterion(halluc_aux,       label)
                    + LAMBDA_PRNU_SPATIAL * criterion(prnu_spatial_aux, label)
                ) / ACCUM_STEPS
            else:
                loss = criterion(out, label) / ACCUM_STEPS

        if not torch.isnan(loss):
            scaler.scale(loss).backward()
            running_loss += loss.item() * ACCUM_STEPS
            trained      += 1

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        del img_t, prnu_feat, prnu_map

        if hasattr(bar, 'set_postfix'):
            vram = ''
            if device.type == 'cuda':
                a = torch.cuda.memory_allocated(device) / 1024**3
                t = torch.cuda.get_device_properties(device).total_memory / 1024**3
                vram = f'{a:.1f}/{t:.1f}GB'
            bar.set_postfix(loss=f'{running_loss/max(trained,1):.4f}',
                            vram=vram, refresh=False)

        try: os.remove(img_path)
        except Exception: pass

    # Flush remaining gradients
    try:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    except Exception:
        pass

    return running_loss / max(trained, 1), trained


# ═══════════════════════════════════════════════════════════════════════════════
#  Download
# ═══════════════════════════════════════════════════════════════════════════════

def _scrape_video_urls(count: int) -> list[str]:
    """
    Use Playwright to visit meta.ai/vibes, auto-scroll, and intercept the
    actual video CDN URLs from network requests.
    Returns up to `count` unique .mp4 / .webm URL strings.
    """
    import re
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [warn] playwright not available — cannot scrape video URLs")
        return []

    video_urls: list[str] = []
    seen: set[str] = set()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                'Mozilla/5.0 (X11; Linux x86_64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/121.0.0.0 Safari/537.36'
            )
        )
        page = ctx.new_page()

        # Capture every response that looks like a video
        def _on_response(response):
            url = response.url
            ct  = response.headers.get('content-type', '')
            if len(video_urls) >= count:
                return
            if url in seen:
                return
            if ('video' in ct or re.search(r'\.(mp4|webm|m4v)(\?|$)', url, re.I)):
                seen.add(url)
                video_urls.append(url)

        page.on('response', _on_response)

        try:
            page.goto(META_VIBES_URL, timeout=30_000,
                      wait_until='domcontentloaded')
        except Exception as e:
            print(f"  [warn] page.goto failed: {e}")
            browser.close()
            return video_urls

        # Scroll to trigger lazy-loaded videos
        scrolled = 0
        while len(video_urls) < count and scrolled < 60:
            page.evaluate('window.scrollBy(0, window.innerHeight)')
            page.wait_for_timeout(800)
            scrolled += 1

        browser.close()

    return video_urls[:count]


def download_videos(output_dir: str, count: int) -> list[str]:
    """
    Download `count` AI videos from meta.ai/vibes into output_dir.
    Strategy:
      1. Scrape real video CDN URLs via Playwright
      2. Download each URL directly with requests (strips all metadata via ffmpeg)
    Returns list of downloaded .mp4 file paths.
    """
    import hashlib
    import requests as req

    os.makedirs(output_dir, exist_ok=True)

    print(f"  Scraping {count} video URLs from meta.ai/vibes ...")
    urls = _scrape_video_urls(count)
    if not urls:
        print("  [warn] no video URLs found — page may require login or layout changed")
        return []

    print(f"  Found {len(urls)} URL(s) — downloading ...")
    files = []
    for i, url in enumerate(urls, 1):
        vid_id  = hashlib.md5(url.encode()).hexdigest()[:12]
        raw_tmp = os.path.join(output_dir, f'{vid_id}_raw.mp4')
        out_mp4 = os.path.join(output_dir, f'{vid_id}.mp4')

        try:
            # Stream-download the raw bytes
            with req.get(url, stream=True, timeout=60,
                         headers={'Referer': META_VIBES_URL}) as r:
                r.raise_for_status()
                with open(raw_tmp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

            # Strip metadata via ffmpeg (-map_metadata -1 -c copy)
            subprocess.run([
                'ffmpeg', '-y', '-i', raw_tmp,
                '-map_metadata', '-1', '-fflags', '+bitexact',
                '-c', 'copy', out_mp4,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            os.remove(raw_tmp)
            files.append(out_mp4)
            print(f"  [{i}/{len(urls)}] saved {os.path.basename(out_mp4)}")
        except Exception as e:
            print(f"  [warn] failed to download {url[:60]}...: {e}")
            for p in (raw_tmp, out_mp4):
                if os.path.exists(p):
                    os.remove(p)

    print(f"  Got {len(files)} video(s)")
    return files


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-video training
# ═══════════════════════════════════════════════════════════════════════════════

def _recover_and_extract(frame_rgb: np.ndarray, recovery_net, device):
    """
    Apply PRNURecoveryNet to undo H.264 compression damage, then extract
    PRNU features and PRNU map from the recovered frame.

    Returns (prnu_feat_tensor, prnu_map_tensor) or zeros on failure.
    """
    from prnu_features import extract_prnu_features_fullres, extract_prnu_map, PRNU_FULLRES_DIM

    # Resize to 256×256 for PRNU (fast but full-res enough)
    small = cv2.resize(frame_rgb, (256, 256))
    arr   = small.astype(np.float32) / 255.0   # [0,1] float32

    # Apply recovery net to undo H.264 blocking artifacts
    if recovery_net is not None:
        try:
            from prnu_recovery import recover_prnu_signal
            arr = recover_prnu_signal(arr, recovery_net, device).astype(np.float32)
        except Exception:
            pass

    # 32-dim scalar PRNU features
    try:
        feats = extract_prnu_features_fullres(arr.astype(np.float64),
                                              tile_size=128)
    except Exception:
        feats = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)

    # 64×64 spatial PRNU map
    try:
        prnu_map = extract_prnu_map(arr.astype(np.float64),
                                    output_size=64)   # (64,64,3)
    except Exception:
        prnu_map = np.zeros((64, 64, 3), dtype=np.float32)

    feat_t = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)
    map_t  = torch.from_numpy(
        prnu_map.astype(np.float32).transpose(2, 0, 1)
    ).unsqueeze(0)

    return feat_t, map_t


def train_video(video_path: str, model, optimizer, scaler, device,
                recovery_net, criterion, frames_per_video: int,
                video_label: str = '') -> tuple[float, int]:
    """
    Train on one video file, frame by frame (no full frame list in RAM).
    Deletes the video file when done.
    Returns (avg_loss, frames_trained).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        try:
            os.remove(video_path)
        except Exception:
            pass
        return 0.0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    # Sample from the middle 80% of the video (skip static intros/outros)
    start_f = int(total_frames * 0.10)
    end_f   = int(total_frames * 0.90)
    span    = max(1, end_f - start_f)
    indices = sorted(set(
        (start_f + np.linspace(0, span - 1,
                               min(frames_per_video, span),
                               dtype=int)).tolist()
    ))

    model.train()
    label        = torch.ones(1, 1, device=device)
    use_amp      = (device.type == 'cuda')
    running_loss = 0.0
    trained      = 0
    optimizer.zero_grad()

    bar = tqdm(indices, desc=f"  {video_label}", leave=False,
               ncols=110, unit='fr',
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} {postfix}]')

    for step, fi in enumerate(bar):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, bgr = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── PRNU: recover H.264 damage → extract features ──────────────
        prnu_feat, prnu_map = _recover_and_extract(rgb, recovery_net, device)
        prnu_feat = prnu_feat.to(device, non_blocking=True)
        prnu_map  = prnu_map.to(device, non_blocking=True)

        # ── Visual transform ─────────────────────────────────────────────
        pil   = Image.fromarray(rgb)
        img_t = _TRAIN_TF(pil).unsqueeze(0).to(device, non_blocking=True)
        del pil, rgb, bgr

        # ── Forward / backward ───────────────────────────────────────────
        with torch.amp.autocast(device.type, enabled=use_amp):
            out = model(img_t, prnu_feat, prnu_map)
            if isinstance(out, tuple):
                logit, prnu_aux, halluc_aux, prnu_spatial_aux = out
                loss = (
                    criterion(logit,            label)
                    + LAMBDA_PRNU         * criterion(prnu_aux,         label)
                    + LAMBDA_HALLUC       * criterion(halluc_aux,       label)
                    + LAMBDA_PRNU_SPATIAL * criterion(prnu_spatial_aux, label)
                ) / ACCUM_STEPS
            else:
                logit = out
                loss  = criterion(logit, label) / ACCUM_STEPS

        if not torch.isnan(loss):
            scaler.scale(loss).backward()
            running_loss += loss.item() * ACCUM_STEPS
            trained      += 1

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Free GPU tensors immediately
        del img_t, prnu_feat, prnu_map, out, logit, loss

        if hasattr(bar, 'set_postfix'):
            vram = ''
            if device.type == 'cuda':
                a = torch.cuda.memory_allocated(device) / 1024 ** 3
                t = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
                vram = f'{a:.1f}/{t:.1f}GB'
            bar.set_postfix(
                loss=f'{running_loss/max(trained,1):.4f}',
                vram=vram,
                refresh=False,
            )

    # Flush any remaining gradients
    try:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    except Exception:
        pass

    cap.release()

    # Delete video immediately — no files left behind
    try:
        os.remove(video_path)
    except Exception as e:
        print(f"  [warn] Could not delete {video_path}: {e}")

    return running_loss / max(trained, 1), trained


# ═══════════════════════════════════════════════════════════════════════════════
#  Live status writer
# ═══════════════════════════════════════════════════════════════════════════════

_status_losses: list[float] = []

def _write_status(total_videos, cycle_num, last_loss, total_frames,
                  elapsed_min, device, current_video='', cycle_size=100):
    global _status_losses
    _status_losses.append(last_loss)

    vram_used = vram_total = 0.0
    gpu_util  = 0
    if device.type == 'cuda':
        try:
            vram_used  = torch.cuda.memory_allocated(device) / 1024**3
            vram_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        except Exception:
            pass
        try:
            import subprocess as _sp
            out = _sp.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'], text=True
            )
            gpu_util = int(out.strip().split('\n')[0])
        except Exception:
            pass

    r = ram_info()
    status = {
        'cycle':          cycle_num,
        'total_videos':   total_videos,
        'cycle_size':     cycle_size,
        'last_loss':      round(last_loss, 4),
        'losses':         [round(l, 4) for l in _status_losses[-200:]],
        'total_frames':   total_frames,
        'elapsed_min':    round(elapsed_min, 1),
        'current_video':  current_video,
        'vram_used_gb':   round(vram_used, 2),
        'vram_total_gb':  round(vram_total, 1),
        'gpu_util_pct':   gpu_util,
        'ram_pct':        round(r['pct'], 1),
        'updated_at':     datetime.datetime.now().isoformat(timespec='seconds'),
    }
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        tmp = STATUS_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(status, f)
        os.replace(tmp, STATUS_FILE)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Main cycle loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Endless download-recover-train-delete cycle on meta.ai/vibes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cycle_size',       type=int,   default=100,
                        help='Videos per cycle before checkpoint')
    parser.add_argument('--sub_batch',        type=int,   default=10,
                        help='Videos downloaded at once (controls disk usage)')
    parser.add_argument('--frames_per_video', type=int,   default=FRAMES_PER_VIDEO)
    parser.add_argument('--lr',               type=float, default=LR)
    parser.add_argument('--no_recovery',      action='store_true',
                        help='Disable PRNU recovery net')
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu    = torch.cuda.get_device_properties(0)
        print(f"\n  GPU    : {gpu.name}")
        print(f"  VRAM   : {gpu.total_memory/1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("\n  Device : CPU (no GPU — training will be slow)")

    # ── Model + optimiser ─────────────────────────────────────────────────
    model     = load_model(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    criterion = nn.BCEWithLogitsLoss()

    # ── PRNU recovery net ─────────────────────────────────────────────────
    recovery_net = None if args.no_recovery else load_recovery_net(device)

    # ── Live plot ─────────────────────────────────────────────────────────
    try:
        from live_plot import LivePlot
        plot = LivePlot(title='Meta AI Cycle Trainer', xlabel='Video #')
    except Exception:
        plot = None

    # ── Graceful Ctrl+C ───────────────────────────────────────────────────
    stop_requested = [False]

    def _handle_sigint(sig, frame):
        print('\n\n  Ctrl+C received — finishing current video then saving ...')
        stop_requested[0] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Stats ─────────────────────────────────────────────────────────────
    cycle_num      = 0
    total_videos   = 0
    total_frames   = 0
    session_start  = time.time()

    print(f"\n  Cycle size  : {args.cycle_size} videos")
    print(f"  Sub-batch   : {args.sub_batch} videos downloaded at once")
    print(f"  Frames/vid  : {args.frames_per_video}")
    print(f"  PRNU recov  : {'enabled' if recovery_net else 'disabled'}")
    print(f"  LR          : {args.lr}")
    print(f"  Class bal.  : 1:1  (AI frames ↔ real Open Images photos)")
    print(f"\n  Press Ctrl+C to stop gracefully.\n")

    # Pre-load Open Images URL list so first batch doesn't stall
    _load_oi_urls()

    # ══════════════════════════════════════════════════════════════════════
    #  Outer cycle loop
    # ══════════════════════════════════════════════════════════════════════
    while not stop_requested[0]:
        cycle_num    += 1
        cycle_videos  = 0
        cycle_loss    = 0.0
        cycle_start   = time.time()

        print(f"\n{'='*60}")
        print(f"  Cycle {cycle_num}  |  Total videos so far: {total_videos}")
        print(f"  {mem_status(device)}")
        print(f"{'='*60}")

        videos_remaining = args.cycle_size

        # ── Sub-batch download loop ────────────────────────────────────
        while videos_remaining > 0 and not stop_requested[0]:
            batch_count = min(args.sub_batch, videos_remaining)

            # Wait for memory before downloading
            wait_for_memory(device, context=f"cycle {cycle_num}")

            # Create a fresh temp dir for this sub-batch
            _data_dir = os.path.join(REPO_ROOT, 'data')
            os.makedirs(_data_dir, exist_ok=True)
            tmp_dir = tempfile.mkdtemp(prefix='meta_ai_cycle_', dir=_data_dir)
            try:
                video_files = download_videos(tmp_dir, count=batch_count)

                # ── Fallback: Civitai AI images when meta.ai fails ────
                civitai_ai_paths: list[str] = []
                if not video_files:
                    print("  meta.ai unavailable — falling back to Civitai AI images...")
                    n_ai_imgs = batch_count * args.frames_per_video
                    civitai_dir = os.path.join(tmp_dir, 'civitai')
                    civitai_ai_paths = download_civitai_images(civitai_dir, n_ai_imgs)
                    print(f"  Got {len(civitai_ai_paths)} Civitai AI images")
                    if not civitai_ai_paths:
                        print("  Civitai also failed — waiting 30s before retry")
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        time.sleep(30)
                        continue

                # ── Process each AI video ─────────────────────────────
                ai_frames_this_batch = 0
                for vpath in video_files:
                    if stop_requested[0]:
                        break

                    # Emergency memory check before each video
                    r = ram_info()
                    if r['pct'] > RAM_CRITICAL_PCT:
                        print(f"\n  [mem] CRITICAL RAM {r['pct']:.0f}% — "
                              f"saving checkpoint + waiting ...")
                        save_model(model)
                        free_memory(device)
                        time.sleep(10)

                    vname   = os.path.basename(vpath)
                    vlabel  = f"C{cycle_num} V{total_videos+1:04d} {vname[:20]}"
                    avg_loss, n_frames = train_video(
                        vpath, model, optimizer, scaler, device,
                        recovery_net, criterion,
                        args.frames_per_video, video_label=vlabel,
                    )

                    total_videos  += 1
                    cycle_videos  += 1
                    total_frames  += n_frames
                    cycle_loss    += avg_loss
                    videos_remaining -= 1
                    ai_frames_this_batch += n_frames

                    elapsed = (time.time() - session_start) / 60
                    print(
                        f"  [AI {total_videos:05d}] loss={avg_loss:.4f} "
                        f"frames={n_frames}  "
                        f"elapsed={elapsed:.0f}min  "
                        f"{mem_status(device)}"
                    )

                    if plot:
                        plot.update(total_videos, avg_loss,
                                    1.0 if avg_loss < 0.693 else 0.5)

                    # Write live status for monitor
                    _write_status(total_videos, cycle_num, avg_loss,
                                  total_frames, elapsed, device,
                                  vname, args.cycle_size)

                    # Free GPU memory after every video
                    free_memory(device)

                # ── Civitai AI still images (fallback when no videos) ──
                if civitai_ai_paths and not stop_requested[0]:
                    civ_loss, civ_n = train_real_images(
                        civitai_ai_paths, model, optimizer, scaler,
                        device, recovery_net, criterion,
                        target_label=1.0,
                    )
                    ai_frames_this_batch += civ_n
                    total_frames  += civ_n
                    # count each batch_count civitai images as one "video"
                    n_pseudo_vids = max(1, civ_n // args.frames_per_video)
                    total_videos     += n_pseudo_vids
                    cycle_videos     += n_pseudo_vids
                    videos_remaining -= n_pseudo_vids
                    elapsed = (time.time() - session_start) / 60
                    cycle_loss += civ_loss
                    print(
                        f"  [AI-CIV]  loss={civ_loss:.4f} "
                        f"images={civ_n}  "
                        f"elapsed={elapsed:.0f}min  "
                        f"{mem_status(device)}"
                    )
                    _write_status(total_videos, cycle_num, civ_loss,
                                  total_frames, elapsed, device,
                                  'civitai', args.cycle_size)
                    free_memory(device)

                # ── Real images — 1:1 class balance ───────────────────
                if ai_frames_this_batch > 0 and not stop_requested[0]:
                    n_real = ai_frames_this_batch  # match AI frame count
                    print(f"\n  Downloading {n_real} real images (Open Images)...")
                    real_dir   = os.path.join(tmp_dir, 'real')
                    real_paths = download_real_images(real_dir, n_real)
                    print(f"  Got {len(real_paths)} real images — training...")
                    real_loss, real_n = train_real_images(
                        real_paths, model, optimizer, scaler,
                        device, recovery_net, criterion,
                    )
                    total_frames += real_n
                    elapsed = (time.time() - session_start) / 60
                    print(
                        f"  [REAL]    loss={real_loss:.4f} "
                        f"images={real_n}  "
                        f"elapsed={elapsed:.0f}min  "
                        f"{mem_status(device)}"
                    )
                    free_memory(device)

            finally:
                # Always clean up temp dir — videos already deleted individually
                # but this catches any partial downloads or metadata files
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── End of cycle: checkpoint + summary ────────────────────────
        cycle_elapsed = (time.time() - cycle_start) / 60
        avg_cycle_loss = cycle_loss / max(cycle_videos, 1)

        print(f"\n  Cycle {cycle_num} complete:")
        print(f"    Videos this cycle : {cycle_videos}")
        print(f"    Avg loss          : {avg_cycle_loss:.4f}")
        print(f"    Cycle time        : {cycle_elapsed:.1f} min")
        print(f"    Total videos      : {total_videos}")
        print(f"    Total frames      : {total_frames:,}")

        save_model(model)
        free_memory(device)

        print(f"  {mem_status(device)}")

    # ══════════════════════════════════════════════════════════════════════
    #  Shutdown
    # ══════════════════════════════════════════════════════════════════════
    elapsed_h = (time.time() - session_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Session finished.")
    print(f"  Total videos   : {total_videos}")
    print(f"  Total frames   : {total_frames:,}")
    print(f"  Total time     : {elapsed_h:.2f} h")
    print(f"{'='*60}\n")

    save_model(model)
    if plot:
        plot.close()


if __name__ == '__main__':
    main()
