"""
meta_ai_realtime.py — Real-time training from meta.ai/vibes
============================================================

Opens meta.ai/vibes in a browser, auto-scrolls through AI-generated
videos one by one, captures frames live as each plays, trains the model
immediately, then moves to the next — no files saved to disk.

Setup (one-time):
    pip install playwright
    playwright install chromium

Usage:
    python src/meta_ai_realtime.py
    python src/meta_ai_realtime.py --videos 100        # train on 100 videos
    python src/meta_ai_realtime.py --fps 3             # capture 3 frames/sec
    python src/meta_ai_realtime.py --headless          # no visible browser
"""

import argparse
import asyncio
import io
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

META_VIBES_URL = "https://www.meta.ai/vibes"

# ─── Training hyper-params ───────────────────────────────────────────────────
FRAMES_PER_VIDEO = 20          # frames to capture per video
CAPTURE_FPS      = 2.0         # screenshots per second while video plays
MIN_FRAMES       = 5           # skip video if fewer frames captured
LR               = 3e-5        # small LR — we're fine-tuning, not training from scratch
ACCUM_STEPS      = 4           # gradient accumulation

IMG_SIZE = 512

LAMBDA_PRNU          = 0.35
LAMBDA_HALLUC        = 0.15
LAMBDA_PRNU_SPATIAL  = 0.20

# ─── Image transform ─────────────────────────────────────────────────────────
_TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.15),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5))], p=0.35
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(device):
    from model_prnu import DeepFusionNet
    from detect import Detector

    print("\n[model] Loading DeepFusionNet v5 ...")
    det = Detector()                       # uses existing load-order logic
    if hasattr(det, 'model') and det.model is not None:
        model = det.model.to(device)
        print("  Loaded from existing checkpoint.")
    else:
        model = DeepFusionNet().to(device)
        print("  No checkpoint found — starting from scratch.")

    return model


# ─── PRNU feature extraction ─────────────────────────────────────────────────

def _prnu_from_pil(pil_img, recovery_net=None, device=None):
    from prnu_features import extract_prnu_features_fullres, extract_prnu_map
    arr = np.array(pil_img.resize((256, 256)), dtype=np.float64) / 255.0
    try:
        feats = extract_prnu_features_fullres(arr, recovery_net=recovery_net,
                                               device=device)
    except Exception:
        from prnu_features import PRNU_FULLRES_DIM
        feats = np.zeros(PRNU_FULLRES_DIM, dtype=np.float32)
    try:
        prnu_map = extract_prnu_map(arr, output_size=64)   # (64,64,3)
    except Exception:
        prnu_map = np.zeros((64, 64, 3), dtype=np.float32)

    feat_t = torch.from_numpy(feats).float()
    map_t  = torch.from_numpy(prnu_map.transpose(2, 0, 1)).float()
    return feat_t, map_t


# ─── Train on one video's frames ─────────────────────────────────────────────

def train_on_frames(model, optimizer, scaler, frames: list[Image.Image],
                    device, recovery_net=None, video_idx: int = 0):
    """
    Fine-tune the model on `frames` (PIL Images) captured from one AI video.
    All frames are labelled 1.0 (AI-generated).
    Returns (avg_loss, accuracy).
    """
    if len(frames) < MIN_FRAMES:
        return None, None

    criterion = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    use_amp   = (device.type == 'cuda')

    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    correct      = 0
    label        = torch.ones(1, 1, device=device)   # AI = 1.0

    bar = tqdm(frames, desc=f"  Vid {video_idx:04d}", leave=False,
               ncols=100, unit="frame")

    for step, pil_img in enumerate(bar):
        img_t               = _TRAIN_TF(pil_img).unsqueeze(0).to(device)
        prnu_feat, prnu_map = _prnu_from_pil(pil_img, recovery_net, device)
        prnu_feat = prnu_feat.unsqueeze(0).to(device)
        prnu_map  = prnu_map.unsqueeze(0).to(device)

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

        if torch.isnan(loss):
            continue

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * ACCUM_STEPS
        pred   = (torch.sigmoid(logit.detach()) > 0.5).float()
        correct += int(pred.item() == 1.0)

        # VRAM stats for bar
        vram = ""
        if device.type == 'cuda':
            a = torch.cuda.memory_allocated(device) / 1024 ** 3
            t = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            vram = f"{a:.1f}/{t:.1f}GB"
        bar.set_postfix(loss=f"{running_loss/max(step+1,1):.4f}",
                        acc=f"{correct/max(step+1,1):.2f}",
                        vram=vram, refresh=False)

    # Flush remaining grads
    if len(frames) % ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    n = len(frames)
    return running_loss / max(n, 1), correct / max(n, 1)


# ─── Browser automation ───────────────────────────────────────────────────────

async def _wait_for_video(page, timeout_ms: int = 8000):
    """Wait until a <video> element is playing on the page."""
    try:
        await page.wait_for_selector('video', timeout=timeout_ms)
        # Give the video time to buffer and start
        await page.wait_for_timeout(1500)
    except Exception:
        pass


_JS_EXTRACT_FRAME = """
() => {
    const video = document.querySelector('video');
    if (!video || video.readyState < 2 || video.videoWidth === 0) return null;
    const canvas = document.createElement('canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    // PNG = lossless export of the H.264-decoded frame
    // No browser UI, no re-compression — raw video pixels only
    return canvas.toDataURL('image/png').split(',')[1];
}
"""

_JS_VIDEO_TIME = "() => { const v = document.querySelector('video'); return v ? v.currentTime : 0; }"


async def _extract_frame(page) -> Image.Image | None:
    """
    Pull the current video frame via canvas API.

    Returns actual H.264-decoded pixels — NOT a browser screenshot.
    This is the real compressed video data the model needs to learn from.
    PRNU extraction on these frames picks up AI generation artifacts and
    the absence of camera sensor noise (the PRNURecoveryNet handles the
    H.264 blocking artifacts first).
    """
    import base64
    try:
        b64 = await page.evaluate(_JS_EXTRACT_FRAME)
        if not b64:
            return None
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    except Exception:
        return None


async def _capture_frames(page, n_frames: int, fps: float) -> list[Image.Image]:
    """
    Capture n_frames from the playing video at the given fps rate.
    Uses canvas extraction — returns actual decoded video pixels,
    not browser screenshots.
    Deduplicates frames that haven't advanced (same video timestamp).
    """
    interval   = 1.0 / fps
    frames     = []
    last_time  = -1.0

    while len(frames) < n_frames:
        t0 = time.time()

        # Skip duplicate frames (video paused / buffering)
        vtime = await page.evaluate(_JS_VIDEO_TIME)
        if vtime == last_time:
            await asyncio.sleep(0.1)
            continue
        last_time = vtime

        frame = await _extract_frame(page)
        if frame is not None:
            frames.append(frame)

        elapsed = time.time() - t0
        wait    = max(0.0, interval - elapsed)
        if wait > 0:
            await asyncio.sleep(wait)

    return frames


async def _scroll_to_next(page):
    """Scroll down to load the next video in the feed."""
    await page.keyboard.press('ArrowDown')
    await page.wait_for_timeout(800)
    # If ArrowDown didn't work (feed is not focused), try scrolling by viewport height
    await page.evaluate("window.scrollBy(0, window.innerHeight)")
    await page.wait_for_timeout(1200)


async def _save_model(model):
    """Save checkpoint to models/ai_detector_prnu_fusion_v5.pth."""
    models_dir = os.path.join(REPO_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, 'ai_detector_prnu_fusion_v5.pth')
    torch.save(model.state_dict(), path)
    print(f"  Checkpoint saved → {path}")


# ─── Main loop ────────────────────────────────────────────────────────────────

async def run(args):
    from playwright.async_api import async_playwright

    # ── Device ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu    = torch.cuda.get_device_properties(0)
        print(f"\n  GPU    : {gpu.name}")
        print(f"  VRAM   : {gpu.total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("\n  Device : CPU (no CUDA GPU — training will be slow)")

    use_amp = (device.type == 'cuda')

    # ── Model ────────────────────────────────────────────────────────────
    model    = load_model(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ── PRNU recovery net ────────────────────────────────────────────────
    recovery_net = None
    if not args.no_recovery:
        try:
            from prnu_recovery import build_prnu_recovery_net
            recovery_net = build_prnu_recovery_net(device=device)
            print("  PRNU recovery net loaded.")
        except Exception as e:
            print(f"  PRNU recovery net unavailable: {e}")

    # ── Live plot ────────────────────────────────────────────────────────
    try:
        from live_plot import LivePlot
        plot = LivePlot(title='Real-time Meta AI Training', xlabel='Video #')
    except Exception:
        plot = None

    # ── Browser ──────────────────────────────────────────────────────────
    print(f"\n  Opening meta.ai/vibes ...")
    print(f"  Training on {args.videos} videos  |  {args.fps} fps  |  {FRAMES_PER_VIDEO} frames/video\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=args.headless,
            args=['--autoplay-policy=no-user-gesture-required'],
        )
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent=(
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
        )
        page = await context.new_page()
        await page.goto(META_VIBES_URL, wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)    # let the feed load

        videos_trained = 0
        total_frames   = 0
        session_losses = []

        print(f"  {'Video':<8} {'Frames':<8} {'Loss':<10} {'Acc':<8} {'VRAM'}")
        print(f"  {'-'*50}")

        while videos_trained < args.videos:

            # ── Wait for video to appear ──────────────────────────────
            await _wait_for_video(page)

            # ── Capture frames ────────────────────────────────────────
            frames = await _capture_frames(page, FRAMES_PER_VIDEO, args.fps)

            if len(frames) < MIN_FRAMES:
                print(f"  [{videos_trained+1:04d}] Too few frames ({len(frames)}) — skipping")
                await _scroll_to_next(page)
                continue

            # ── Train ─────────────────────────────────────────────────
            loss, acc = train_on_frames(
                model, optimizer, scaler, frames, device,
                recovery_net=recovery_net,
                video_idx=videos_trained + 1,
            )

            if loss is None:
                await _scroll_to_next(page)
                continue

            videos_trained += 1
            total_frames   += len(frames)
            session_losses.append(loss)

            # VRAM display
            vram_str = ""
            if device.type == 'cuda':
                a = torch.cuda.memory_allocated(device) / 1024 ** 3
                t = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
                vram_str = f"{a:.1f}/{t:.1f}GB"

            print(f"  [{videos_trained:04d}]   "
                  f"{len(frames):<8d} {loss:<10.4f} {acc*100:<8.1f}% {vram_str}")

            if plot:
                plot.update(videos_trained, loss, acc)

            # ── Save checkpoint every 10 videos ──────────────────────
            if videos_trained % 10 == 0:
                await _save_model(model)

            # ── Scroll to next video ──────────────────────────────────
            await _scroll_to_next(page)

        # ── Final save ────────────────────────────────────────────────
        await _save_model(model)
        if plot:
            plot.close()

        avg_session_loss = sum(session_losses) / max(len(session_losses), 1)
        print(f"\n  Done.  Videos: {videos_trained}  Frames: {total_frames}  "
              f"Avg loss: {avg_session_loss:.4f}")

        await browser.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Real-time training from meta.ai/vibes (no downloads).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--videos',      type=int,   default=50,
                   help='Number of videos to train on')
    p.add_argument('--fps',         type=float, default=CAPTURE_FPS,
                   help='Frame capture rate (screenshots per second)')
    p.add_argument('--lr',          type=float, default=LR,
                   help='Learning rate')
    p.add_argument('--headless',    action='store_true',
                   help='Run browser without a visible window')
    p.add_argument('--no_recovery', action='store_true',
                   help='Disable PRNU recovery net')
    args = p.parse_args()

    asyncio.run(run(args))


if __name__ == '__main__':
    main()
