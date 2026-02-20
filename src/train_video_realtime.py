"""
Real-Time Video Training with Auto AI-Editing + Memory Management
==================================================================

Takes real videos or live webcam feed, extracts frames, auto-generates
AI-edited counterparts, and trains the detector — infinitely.

KEY FEATURES:
  • Lazy imports    — checks memory BEFORE loading PyTorch
  • Memory control  — monitors RAM/VRAM, auto-cleans when too high
  • Infinite loop   — webcam training runs forever (Ctrl+C to stop)
  • Data cleanup    — caps saved data at a configurable size limit
  • GPU-optimised   — FP16, grad accumulation, tiny batch size

Optimised for RTX 3050 (4 GB VRAM).

Usage:
    python src/train_video_realtime.py --webcam
    python src/train_video_realtime.py --webcam --duration 120
    python src/train_video_realtime.py --video_dir data/real_videos
    python src/train_video_realtime.py --help
"""

import argparse
import gc
import io
import os
import random
import shutil
import sys
import time

# ── Lightweight imports only (no torch yet) ──
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
TEMP_FRAMES_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'temp_frames')

DEFAULT_RESOLUTION = 512
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM_STEPS = 8
DEFAULT_EPOCHS = 3
DEFAULT_FPS_SAMPLE = 1
DEFAULT_EDITS_PER_FRAME = 2
DEFAULT_LR = 0.0005
DEFAULT_WEBCAM_DURATION = 60        # per cycle (infinite mode keeps looping)
DEFAULT_MAX_DATA_MB = 1024          # max temp data in MB before cleanup

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}


# ═══════════════════════════════════════════════════════════════════════════
#  Memory Controller
# ═══════════════════════════════════════════════════════════════════════════

class MemoryController:
    """
    Monitors and manages system memory + GPU VRAM.
    
    Usage:
        mc = MemoryController(max_ram_percent=85, max_data_mb=1024)
        mc.check_startup()          # abort early if RAM is too low
        mc.cleanup_if_needed()      # free caches, delete old data
        mc.report()                 # print current memory status
    """

    def __init__(self, max_ram_percent=85, max_data_mb=DEFAULT_MAX_DATA_MB):
        self.max_ram_percent = max_ram_percent
        self.max_data_mb = max_data_mb
        self.min_disk_mb = 500    # emergency stop if disk < 500 MB
        self.min_ram_mb = 400     # emergency stop if RAM < 400 MB
        self.max_ram_critical = 93  # emergency stop above this %

    @staticmethod
    def get_ram_info():
        """Get RAM usage without psutil (reads /proc/meminfo on Linux)."""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                key = parts[0].rstrip(':')
                val_kb = int(parts[1])
                info[key] = val_kb
            total_mb = info.get('MemTotal', 0) / 1024
            available_mb = info.get('MemAvailable', 0) / 1024
            used_mb = total_mb - available_mb
            percent = (used_mb / total_mb * 100) if total_mb > 0 else 0
            return {
                'total_mb': total_mb,
                'available_mb': available_mb,
                'used_mb': used_mb,
                'percent': percent,
            }
        except Exception:
            return {'total_mb': 0, 'available_mb': 0, 'used_mb': 0, 'percent': 0}

    @staticmethod
    def get_gpu_info():
        """Get GPU memory info (requires torch to be imported)."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                free = total - allocated
                return {
                    'total_mb': total,
                    'allocated_mb': allocated,
                    'reserved_mb': reserved,
                    'free_mb': free,
                }
        except Exception:
            pass
        return None

    def check_startup(self):
        """Check if there's enough RAM to even start. Returns True if OK."""
        ram = self.get_ram_info()
        print(f"\n  💾 RAM: {ram['used_mb']:.0f} / {ram['total_mb']:.0f} MB "
              f"({ram['percent']:.0f}% used, {ram['available_mb']:.0f} MB free)")

        if ram['available_mb'] < 500:
            print(f"\n  ⚠️  WARNING: Only {ram['available_mb']:.0f} MB RAM available.")
            print(f"  PyTorch needs ~1-2 GB just to import.")
            print(f"  Close other apps or free memory, then retry.\n")
            return False

        if ram['percent'] > 90:
            print(f"\n  ⚠️  WARNING: RAM is {ram['percent']:.0f}% full.")
            print(f"  Training may be very slow due to swapping.\n")

        return True

    def free_memory(self):
        """Aggressively free memory: Python GC + GPU cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def cleanup_temp_data(self):
        """Delete temp frame data if it exceeds max_data_mb."""
        if not os.path.exists(TEMP_FRAMES_DIR):
            return 0

        total_size = 0
        files = []
        for root, _, filenames in os.walk(TEMP_FRAMES_DIR):
            for f in filenames:
                fp = os.path.join(root, f)
                sz = os.path.getsize(fp)
                total_size += sz
                files.append((fp, sz, os.path.getmtime(fp)))

        total_mb = total_size / 1024**2
        if total_mb <= self.max_data_mb:
            return 0

        # Delete oldest files first until we're under half the limit
        target_mb = self.max_data_mb / 2
        files.sort(key=lambda x: x[2])  # sort by modification time (oldest first)

        deleted_mb = 0
        deleted_count = 0
        for fp, sz, _ in files:
            if total_mb - deleted_mb <= target_mb:
                break
            try:
                os.remove(fp)
                deleted_mb += sz / 1024**2
                deleted_count += 1
            except Exception:
                pass

        if deleted_count > 0:
            print(f"  🗑️  Cleaned {deleted_mb:.0f} MB ({deleted_count} files) from temp data")

        return deleted_mb

    def cleanup_if_needed(self):
        """Run all cleanup if memory is high."""
        ram = self.get_ram_info()
        if ram['percent'] > self.max_ram_percent:
            print(f"  ⚠️  RAM at {ram['percent']:.0f}%, cleaning up...")
            self.free_memory()
            self.cleanup_temp_data()
            ram_after = self.get_ram_info()
            print(f"  ✓ RAM now at {ram_after['percent']:.0f}% "
                  f"(freed {ram['used_mb'] - ram_after['used_mb']:.0f} MB)")

    @staticmethod
    def get_disk_free_mb(path='.'):
        """Get free disk space in MB."""
        try:
            usage = shutil.disk_usage(os.path.abspath(path))
            return usage.free / 1024**2
        except Exception:
            return float('inf')

    def is_critical(self):
        """
        Check if any resource is dangerously low.
        Returns (is_critical: bool, reason: str).
        If critical, training MUST stop to prevent system freeze.
        """
        # Disk space
        disk_mb = self.get_disk_free_mb(MODELS_DIR)
        if disk_mb < self.min_disk_mb:
            return True, f"Disk space: {disk_mb:.0f} MB free (min {self.min_disk_mb} MB)"

        # RAM
        ram = self.get_ram_info()
        if ram['available_mb'] < self.min_ram_mb:
            return True, f"RAM: only {ram['available_mb']:.0f} MB free (min {self.min_ram_mb} MB)"
        if ram['percent'] > self.max_ram_critical:
            return True, f"RAM: {ram['percent']:.0f}% used (max {self.max_ram_critical}%)"

        # GPU VRAM
        gpu = self.get_gpu_info()
        if gpu and gpu['total_mb'] > 0:
            gpu_pct = (gpu['allocated_mb'] / gpu['total_mb']) * 100
            if gpu_pct > 95:
                return True, f"GPU VRAM: {gpu_pct:.0f}% used"

        return False, ''

    def report(self):
        """Print current memory status."""
        ram = self.get_ram_info()
        print(f"  💾 RAM: {ram['used_mb']:.0f}/{ram['total_mb']:.0f} MB "
              f"({ram['percent']:.0f}%) | Free: {ram['available_mb']:.0f} MB")

        gpu = self.get_gpu_info()
        if gpu:
            print(f"  🎮 GPU: {gpu['allocated_mb']:.0f}/{gpu['total_mb']:.0f} MB "
                  f"| Free: {gpu['free_mb']:.0f} MB")


# ═══════════════════════════════════════════════════════════════════════════
#  AI Edit Pipeline  (in-memory, no disk I/O)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_heavy_blur_sharpen(img):
    """Simulate AI upscaling artefacts."""
    blurred = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.0)))
    return blurred.filter(ImageFilter.UnsharpMask(
        radius=random.randint(3, 8), percent=random.randint(150, 400), threshold=0
    ))


def _apply_color_hallucination(img):
    """Simulate AI colour hallucination."""
    img = ImageEnhance.Color(img).enhance(random.uniform(1.3, 2.5))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))
    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + random.uniform(-25, 25), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _apply_noise_compression(img):
    """Simulate social-media re-encoding."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(5, 20), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=random.randint(15, 50))
    buf.seek(0)
    result = Image.open(buf).convert('RGB')
    result.load()          # Force pixel data into memory
    buf.close()            # Release the BytesIO buffer
    del arr, noise
    return result


def _apply_rescale_artefacts(img):
    """Simulate AI resolution artefacts: downscale then upscale."""
    w, h = img.size
    factor = random.uniform(0.2, 0.5)
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def _apply_edge_smoothing(img):
    """Simulate AI's unnaturally smooth textures."""
    arr = np.array(img)
    d = random.choice([7, 9, 11, 15])
    smoothed = cv2.bilateralFilter(arr, d, random.uniform(50, 120), random.uniform(50, 120))
    return Image.fromarray(smoothed)


def _apply_frequency_manipulation(img):
    """Simulate GAN spectral fingerprint."""
    arr = np.array(img, dtype=np.float32)
    rows, cols = arr.shape[:2]
    crow, ccol = rows // 2, cols // 2
    inner_r = random.randint(rows // 8, rows // 4)
    outer_r = inner_r + random.randint(rows // 10, rows // 5)

    # Pre-compute distance mask once (shared across channels)
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
    ring = (dist >= inner_r) & (dist <= outer_r)
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[ring] *= random.uniform(0.1, 0.5)
    del Y, X, dist, ring

    for c in range(3):
        f = np.fft.fft2(arr[:, :, c])
        np.fft.fftshift(f, axes=None)    # in-place via assignment below
        fshift = np.fft.fftshift(f)
        fshift *= mask
        arr[:, :, c] = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
        del f, fshift

    del mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


_EDIT_POOL = [
    _apply_heavy_blur_sharpen,
    _apply_color_hallucination,
    _apply_noise_compression,
    _apply_rescale_artefacts,
    _apply_edge_smoothing,
    _apply_frequency_manipulation,
]


def apply_ai_edits(frame_pil, num_variants=DEFAULT_EDITS_PER_FRAME):
    """Apply random AI-style edits to create 'AI-made' variants."""
    variants = []
    for _ in range(num_variants):
        edited = frame_pil.copy()
        num_edits = random.randint(1, 3)
        chosen = random.sample(_EDIT_POOL, min(num_edits, len(_EDIT_POOL)))
        for fn in chosen:
            try:
                edited = fn(edited)
            except Exception:
                pass
        variants.append(edited)
    return variants


# ═══════════════════════════════════════════════════════════════════════════
#  Lazy PyTorch Loader
# ═══════════════════════════════════════════════════════════════════════════

_torch_loaded = False


def _lazy_import_torch():
    """Import torch only when needed, after memory check passes."""
    global _torch_loaded
    if _torch_loaded:
        return

    print("  ⏳ Loading PyTorch (this may take a minute)...")
    start = time.time()

    global torch, nn, optim, IterableDataset, DataLoader, transforms
    import torch as _torch
    import torch.nn as _nn
    import torch.optim as _optim
    from torch.utils.data import IterableDataset as _ID, DataLoader as _DL
    from torchvision import transforms as _T

    torch = _torch
    nn = _nn
    optim = _optim
    IterableDataset = _ID
    DataLoader = _DL
    transforms = _T

    elapsed = time.time() - start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ PyTorch loaded in {elapsed:.1f}s (device: {device})")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  🎮 GPU: {gpu_name} ({gpu_mem:.0f} MB)")

    _torch_loaded = True


def _get_device():
    _lazy_import_torch()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════
#  Frame Capture (works for both video files and webcam)
# ═══════════════════════════════════════════════════════════════════════════

def capture_frames(source, resolution, fps_sample, edits_per_frame, duration=None):
    """
    Generator: yields (pil_real, [pil_ai_1, pil_ai_2, ...]) from a video
    source (file path or camera ID).

    For webcam, shows a live preview window.
    Yields PIL images — no torch dependency here.
    """
    is_webcam = isinstance(source, int)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open {'webcam' if is_webcam else source}")
        return

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps <= 0:
        cam_fps = 30.0

    frame_interval = max(1, int(cam_fps / fps_sample))
    frame_idx = 0
    extracted = 0
    start_time = time.time()

    if is_webcam:
        dur_str = f"{duration}s" if duration else "infinite"
        print(f"\n  📷 Webcam capture started (device {source})")
        print(f"     Duration: {dur_str}  |  Sampling: {fps_sample} fps")
        print(f"     Press 'q' in preview or Ctrl+C to stop\n")

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                if is_webcam:
                    print("[WARN] Webcam read failed.")
                break

            # Duration check
            if duration and (time.time() - start_time) >= duration:
                if is_webcam:
                    print(f"  ⏱  Duration reached ({duration}s)")
                break

            # Webcam preview
            if is_webcam and frame_idx % 3 == 0:
                try:
                    elapsed = time.time() - start_time
                    preview = bgr_frame.copy()
                    cv2.putText(preview,
                                f"Training | {elapsed:.0f}s | frames: {extracted}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Webcam Training (q=stop)', preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("  ✋ Stopped by user")
                        break
                except Exception:
                    pass

            # Sample frame
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb).resize(
                    (resolution, resolution), Image.LANCZOS
                )
                ai_variants = apply_ai_edits(pil_frame, edits_per_frame)
                yield pil_frame, ai_variants
                extracted += 1

            frame_idx += 1

    except KeyboardInterrupt:
        print("  ✋ Stopped by Ctrl+C")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    src_name = f"webcam" if is_webcam else os.path.basename(str(source))
    print(f"  → {extracted} frames from {src_name} ({time.time() - start_time:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════

def _build_model():
    """Build the PyTorchCNN model. Requires torch to be loaded."""
    _lazy_import_torch()

    class PyTorchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.classifier(self.avgpool(self.features(x)))

    return PyTorchCNN()


# ═══════════════════════════════════════════════════════════════════════════
#  Training Loop (with memory control)
# ═══════════════════════════════════════════════════════════════════════════

def train_from_source(
    source,                          # file path, list of paths, or int (camera)
    webcam=False,
    webcam_duration=DEFAULT_WEBCAM_DURATION,
    infinite=False,
    resume_model=None,
    resolution=DEFAULT_RESOLUTION,
    batch_size=DEFAULT_BATCH_SIZE,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    epochs=DEFAULT_EPOCHS,
    fps_sample=DEFAULT_FPS_SAMPLE,
    edits_per_frame=DEFAULT_EDITS_PER_FRAME,
    lr=DEFAULT_LR,
    max_data_mb=DEFAULT_MAX_DATA_MB,
    camera_id=0,
):
    mc = MemoryController(max_ram_percent=85, max_data_mb=max_data_mb)

    # ── Memory check BEFORE importing torch ──
    if not mc.check_startup():
        print("  Aborting — not enough memory to start safely.")
        print("  Tip: close browsers, IDEs, or other heavy apps, then retry.")
        sys.exit(1)

    # ── Now load torch (lazy) ──
    _lazy_import_torch()
    device = _get_device()

    # ── Print config ──
    mode = "Webcam (infinite)" if (webcam and infinite) else \
           "Webcam" if webcam else "Video files"
    print(f"\n{'='*60}")
    print(f"  Real-Time Video Training  [{mode}]")
    print(f"{'='*60}")
    if webcam:
        print(f"  Webcam duration : {webcam_duration}s per cycle")
        print(f"  Infinite loop   : {'Yes' if infinite else 'No'}")
    else:
        sources = source if isinstance(source, list) else [source]
        print(f"  Videos          : {len(sources)}")
    print(f"  Resolution      : {resolution}×{resolution}")
    print(f"  Batch           : {batch_size} (effective {batch_size * grad_accum_steps})")
    print(f"  Epochs          : {epochs}")
    print(f"  Learning rate   : {lr}")
    print(f"  Max data on disk: {max_data_mb} MB")
    print(f"  Mixed precision : {'Yes' if device.type == 'cuda' else 'No'}")
    print(f"{'='*60}")
    mc.report()
    print()

    # ── Transforms ──
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Model ──
    model = _build_model().to(device)
    if resume_model and os.path.exists(resume_model):
        print(f"  Loading model: {resume_model}")
        model.load_state_dict(torch.load(resume_model, map_location=device))
        print("  ✓ Fine-tuning mode\n")
    else:
        print("  Starting fresh training\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Main loop ──
    cycle = 0
    total_trained = 0

    try:
        while True:
            cycle += 1
            if webcam and infinite:
                print(f"\n{'─'*60}")
                print(f"  🔄 Cycle {cycle} — capturing {webcam_duration}s from webcam")
                print(f"{'─'*60}")

            for epoch in range(epochs):
                epoch_label = f"Cycle {cycle} Epoch {epoch+1}" if infinite else f"Epoch {epoch+1}/{epochs}"
                print(f"\n── {epoch_label} ──")
                epoch_start = time.time()

                # ── Safety check: is it safe to continue? ──
                critical, reason = mc.is_critical()
                if critical:
                    print(f"\n  🛑  EMERGENCY STOP: {reason}")
                    print(f"       Saving model before exit...")
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    epath = os.path.join(MODELS_DIR, 'ai_detector_model_EMERGENCY.pth')
                    torch.save(model.state_dict(), epath)
                    print(f"  💾  Emergency model saved: {epath}")
                    print(f"  ❌  Stopped safely to prevent system freeze.")
                    print(f"      Re-run with --resume_model {epath}\n")
                    mc.free_memory()
                    sys.exit(1)

                # ── Memory cleanup before each epoch ──
                mc.cleanup_if_needed()
                mc.free_memory()

                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                micro_step = 0
                batch_inputs = []
                batch_labels = []

                optimizer.zero_grad()

                # ── Pick the right source ──
                if webcam:
                    frame_gen = capture_frames(camera_id, resolution, fps_sample,
                                               edits_per_frame, duration=webcam_duration)
                elif isinstance(source, list):
                    def _multi_gen():
                        for s in source:
                            yield from capture_frames(s, resolution, fps_sample,
                                                      edits_per_frame)
                    frame_gen = _multi_gen()
                else:
                    frame_gen = capture_frames(source, resolution, fps_sample,
                                               edits_per_frame)

                # ── Process frames in mini-batches ──
                max_buffer = batch_size * 4  # Cap to prevent unbounded RAM growth
                for pil_real, ai_variants in frame_gen:
                    # Skip frame if buffer is already full
                    if len(batch_inputs) >= max_buffer:
                        del pil_real, ai_variants
                        continue

                    # Add real frame
                    batch_inputs.append(img_transform(pil_real))
                    batch_labels.append(0.0)

                    # Add AI variants
                    for v in ai_variants:
                        batch_inputs.append(img_transform(v))
                        batch_labels.append(1.0)
                    del pil_real, ai_variants

                    # Train when batch is full
                    while len(batch_inputs) >= batch_size:
                        inputs_t = torch.stack(batch_inputs[:batch_size]).to(device, non_blocking=True)
                        labels_t = torch.tensor(batch_labels[:batch_size],
                                                dtype=torch.float32).to(device).view(-1, 1)

                        batch_inputs = batch_inputs[batch_size:]
                        batch_labels = batch_labels[batch_size:]

                        # Forward
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            outputs = model(inputs_t)
                            loss = criterion(outputs, labels_t) / grad_accum_steps

                        # Backward
                        scaler.scale(loss).backward()
                        micro_step += 1

                        if micro_step % grad_accum_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                        # Metrics
                        running_loss += loss.item() * grad_accum_steps * inputs_t.size(0)
                        predicted = (torch.sigmoid(outputs.detach()) > 0.5).float()
                        total += labels_t.size(0)
                        correct += (predicted == labels_t).sum().item()

                        total_trained += inputs_t.size(0)

                        # Free GPU tensors immediately
                        del inputs_t, labels_t, outputs, loss

                        # Progress
                        if total > 0 and total % (batch_size * 25) == 0:
                            print(f"    samples {total:>5d} | "
                                  f"loss {running_loss / total:.4f} | "
                                  f"acc {correct / total:.4f}")

                    # Periodic memory check
                    if total > 0 and total % (batch_size * 100) == 0:
                        mc.cleanup_if_needed()
                        # Also check for critical state
                        critical, reason = mc.is_critical()
                        if critical:
                            print(f"\n  🛑  EMERGENCY STOP (mid-training): {reason}")
                            os.makedirs(MODELS_DIR, exist_ok=True)
                            epath = os.path.join(MODELS_DIR, 'ai_detector_model_EMERGENCY.pth')
                            torch.save(model.state_dict(), epath)
                            print(f"  💾  Emergency model saved: {epath}")
                            print(f"  ❌  Stopped safely. Re-run with --resume_model {epath}\n")
                            mc.free_memory()
                            sys.exit(1)

                # Flush remaining gradients
                if micro_step % grad_accum_steps != 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Process remaining partial batch
                if batch_inputs:
                    inputs_t = torch.stack(batch_inputs).to(device, non_blocking=True)
                    labels_t = torch.tensor(batch_labels,
                                            dtype=torch.float32).to(device).view(-1, 1)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        out = model(inputs_t)
                        loss = criterion(out, labels_t)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    predicted = (torch.sigmoid(out.detach()) > 0.5).float()
                    total += labels_t.size(0)
                    correct += (predicted == labels_t).sum().item()
                    running_loss += loss.item() * inputs_t.size(0)
                    total_trained += inputs_t.size(0)

                    del inputs_t, labels_t, out, loss
                    batch_inputs.clear()
                    batch_labels.clear()

                # ── Epoch summary ──
                elapsed = time.time() - epoch_start
                epoch_loss = running_loss / max(total, 1)
                epoch_acc = correct / max(total, 1)
                print(f"  ✓ {epoch_label} | loss {epoch_loss:.4f} | "
                      f"acc {epoch_acc:.4f} | {total} samples | {elapsed:.1f}s")

                # Free GPU memory between epochs
                mc.free_memory()
                mc.report()

            # ── Save model after each cycle ──
            os.makedirs(MODELS_DIR, exist_ok=True)

            pth_path = os.path.join(MODELS_DIR, 'ai_detector_model_pytorch.pth')
            torch.save(model.state_dict(), pth_path)
            print(f"\n  💾 Model saved: {pth_path}")

            try:
                ts_path = os.path.join(MODELS_DIR, 'ai_detector_model_pytorch_script.ts')
                dummy = torch.randn(1, 3, resolution, resolution).to(device)
                traced = torch.jit.trace(model, dummy)
                traced.save(ts_path)
                print(f"  💾 TorchScript saved: {ts_path}")
                del dummy, traced
            except Exception as e:
                print(f"  ⚠️  TorchScript save failed: {e}")

            mc.free_memory()
            mc.cleanup_temp_data()

            print(f"\n  📊 Total trained so far: {total_trained} samples across {cycle} cycle(s)")

            # ── Loop or exit ──
            if not (webcam and infinite):
                break

            print(f"\n  🔁 Starting next cycle in 3 seconds... (Ctrl+C to stop)")
            time.sleep(3)

    except KeyboardInterrupt:
        print(f"\n\n  ✋ Training stopped by user after {total_trained} total samples")
        # Save on interrupt
        os.makedirs(MODELS_DIR, exist_ok=True)
        pth_path = os.path.join(MODELS_DIR, 'ai_detector_model_pytorch.pth')
        torch.save(model.state_dict(), pth_path)
        print(f"  💾 Final model saved: {pth_path}")

    mc.free_memory()
    print(f"\n{'='*60}")
    print(f"  Training complete!  ({total_trained} total samples)")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _collect_videos(path):
    """Collect video file paths from a file or directory."""
    videos = []
    if os.path.isfile(path):
        videos.append(os.path.abspath(path))
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                    videos.append(os.path.join(root, f))
    else:
        print(f"[ERROR] Path does not exist: {path}")
    return videos


def main():
    parser = argparse.ArgumentParser(
        description='Train AI detector with real-time video + auto AI-editing + memory management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --webcam                                          # 60s webcam, 3 epochs
  %(prog)s --webcam --infinite                               # webcam forever (Ctrl+C to stop)
  %(prog)s --webcam --duration 120 --infinite                # 120s per cycle, forever
  %(prog)s --webcam --resume_model models/ai_detector_model_pytorch.pth
  %(prog)s --video_dir data/real_videos
  %(prog)s --video_path clip.mp4 --epochs 5
        """,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--video_path', type=str, help='Path to a single video file')
    src.add_argument('--video_dir', type=str, help='Directory containing video files')
    src.add_argument('--webcam', action='store_true', help='Train from live webcam')

    parser.add_argument('--infinite', action='store_true',
                        help='Run webcam training forever (Ctrl+C to stop)')
    parser.add_argument('--resume_model', type=str, default=None,
                        help='Path to existing .pth model to fine-tune')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--grad_accum', type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--fps_sample', type=float, default=DEFAULT_FPS_SAMPLE)
    parser.add_argument('--edits_per_frame', type=int, default=DEFAULT_EDITS_PER_FRAME)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--duration', type=int, default=DEFAULT_WEBCAM_DURATION,
                        help='Webcam capture duration per cycle in seconds')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Webcam device ID')
    parser.add_argument('--max_data_mb', type=int, default=DEFAULT_MAX_DATA_MB,
                        help='Max temp data on disk in MB before auto-cleanup')

    args = parser.parse_args()

    if args.webcam:
        print(f"\n📷 Webcam training mode"
              f"{' (INFINITE)' if args.infinite else ''}")
        train_from_source(
            source=args.camera_id,
            webcam=True,
            webcam_duration=args.duration,
            infinite=args.infinite,
            resume_model=args.resume_model,
            resolution=args.resolution,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            epochs=args.epochs,
            fps_sample=args.fps_sample,
            edits_per_frame=args.edits_per_frame,
            lr=args.lr,
            max_data_mb=args.max_data_mb,
            camera_id=args.camera_id,
        )
        return

    # Video file mode
    src_path = args.video_path or args.video_dir
    video_paths = _collect_videos(src_path)

    if not video_paths:
        print("[ERROR] No video files found.")
        return

    print(f"\nFound {len(video_paths)} video(s):")
    for v in video_paths:
        print(f"  • {v}")

    train_from_source(
        source=video_paths,
        resume_model=args.resume_model,
        resolution=args.resolution,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        epochs=args.epochs,
        fps_sample=args.fps_sample,
        edits_per_frame=args.edits_per_frame,
        lr=args.lr,
        max_data_mb=args.max_data_mb,
    )


if __name__ == '__main__':
    main()
