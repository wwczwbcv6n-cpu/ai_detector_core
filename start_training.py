"""Auto-starts training once ffmpeg finishes."""
import os, sys, time, subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

OUTPUT = 'data/temp_frames/sTsjt8J4OmA_1080p.mp4'

print('[watcher] Waiting for ffmpeg to finish...')
while True:
    alive = bool(subprocess.run(
        ['pgrep', '-x', 'ffmpeg'], capture_output=True).stdout.strip())
    if not alive:
        break
    size = os.path.getsize(OUTPUT) / 1024**3 if os.path.exists(OUTPUT) else 0
    print(f'[watcher] {time.strftime("%H:%M:%S")} — {size:.1f} GB written...',
          flush=True)
    time.sleep(60)

print(f'[watcher] ffmpeg done! File: {os.path.getsize(OUTPUT)/1024**3:.1f} GB')
print('[watcher] Starting training...', flush=True)

from train_deep import (
    train_video_section, VID_BATCH, VID_LR, VID_TILE_SIZE,
    VID_FPS_SAMPLE, VID_EPOCHS, CKPT_INTERVAL_MIN, VIDEO_CKPT_PATH,
)
train_video_section(
    video_dir               = 'data/temp_frames/',
    epochs                  = VID_EPOCHS,
    batch_size              = VID_BATCH,
    lr                      = VID_LR,
    tile_size               = VID_TILE_SIZE,
    fps_sample              = VID_FPS_SAMPLE,
    checkpoint_interval_min = CKPT_INTERVAL_MIN,
    resume_path             = VIDEO_CKPT_PATH,
    use_audio               = False,
)
