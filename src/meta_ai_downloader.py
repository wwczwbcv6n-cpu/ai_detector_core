"""
meta_ai_downloader.py — Download Meta AI videos and train the detector
=======================================================================

Downloads AI-generated videos from meta.ai/vibes (100% AI content — no
labelling or badge detection needed) then trains VideoTemporalFusionNet
and/or DeepFusionNet on them.

Usage:
    python src/meta_ai_downloader.py                      # download 50 + train both
    python src/meta_ai_downloader.py --count 200          # download 200 + train both
    python src/meta_ai_downloader.py --download_only      # download only
    python src/meta_ai_downloader.py --train_only         # skip download, train with existing clips
    python src/meta_ai_downloader.py --model image        # train image model only (extract frames)
    python src/meta_ai_downloader.py --model video        # train video model only
    python src/meta_ai_downloader.py --count 100 --model both --real_dir data/real_videos
"""

import argparse
import os
import subprocess
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)

AI_VIDEO_DIR  = os.path.join(REPO_ROOT, "data", "ai_videos")
AI_FRAME_DIR  = os.path.join(REPO_ROOT, "data", "ai")
REAL_VIDEO_DIR = os.path.join(REPO_ROOT, "data", "real_videos")

META_VIBES_URL = "https://www.meta.ai/vibes"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


# ─── Download ────────────────────────────────────────────────────────────────

def download_meta_ai_videos(
    output_dir: str,
    count: int = 50,
    url: str = META_VIBES_URL,
) -> list[str]:
    """
    Download up to `count` AI-generated videos from meta.ai/vibes using yt-dlp.

    Returns list of downloaded file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[meta_ai_downloader] Downloading up to {count} videos")
    print(f"  Source : {url}")
    print(f"  Output : {output_dir}\n")

    # yt-dlp options — video only, no metadata files at all
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        url,
        "--playlist-end", str(count),
        "-o", out_template,
        # Video only — no audio track needed for visual AI detection
        "--format", "bestvideo[ext=mp4]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        # Strip ALL metadata, descriptions, thumbnails, subtitles
        "--no-write-info-json",
        "--no-write-description",
        "--no-write-thumbnail",
        "--no-write-comments",
        "--no-write-annotations",
        "--no-write-playlist-metafiles",
        "--no-embed-metadata",
        "--no-embed-thumbnail",
        "--no-embed-subs",
        "--no-subs",
        "--no-overwrites",
        "--ignore-errors",
        "--no-warnings",
        "--quiet",
        "--progress",
        # Strip internal MP4 metadata atoms (title, artist, etc.) via ffmpeg
        "--postprocessor-args", "ffmpeg:-map_metadata -1 -fflags +bitexact",
    ]

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode not in (0, 1):  # 1 = some items skipped, not a fatal error
        print(f"\n[warn] yt-dlp exited with code {result.returncode}")
        print("  If download failed, try installing latest yt-dlp:")
        print("    pip install -U yt-dlp")

    # Collect downloaded files
    files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]
    print(f"\n  Downloaded / available: {len(files)} video(s) in {output_dir}")
    return files


# ─── Frame extraction (for image model) ─────────────────────────────────────

def extract_frames_from_videos(
    video_dir: str,
    frames_dir: str,
    frames_per_video: int = 30,
) -> int:
    """
    Extract evenly-spaced frames from every video in video_dir.

    Saves lossless PNG files (no re-compression) so PRNU signals are
    preserved. H.264 blocking artifacts from the download are handled by
    PRNURecoveryNet during training — do NOT apply JPEG here.
    """
    import cv2
    import numpy as np
    from PIL import Image

    os.makedirs(frames_dir, exist_ok=True)

    videos = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]

    if not videos:
        print(f"  [warn] No videos found in {video_dir}")
        return 0

    total_saved = 0
    for vi, vpath in enumerate(videos):
        cap   = cv2.VideoCapture(vpath)
        n_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_fr < 1:
            cap.release()
            continue

        # Skip first and last 10% of the video — often static intro/outro frames
        start_fr = int(n_fr * 0.10)
        end_fr   = int(n_fr * 0.90)
        span     = max(1, end_fr - start_fr)

        indices = set(
            (start_fr + np.linspace(0, span - 1, min(frames_per_video, span), dtype=int)).tolist()
        )
        base = os.path.splitext(os.path.basename(vpath))[0]

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_fr)

        for fi in range(start_fr, end_fr):
            ret, bgr = cap.read()
            if not ret:
                break
            if fi in indices:
                rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                fname = os.path.join(frames_dir, f"{base}_f{fi:05d}.png")
                # PNG = lossless: preserves all noise signal for PRNU extraction
                Image.fromarray(rgb).save(fname, "PNG", compress_level=1)
                total_saved += 1

        cap.release()
        print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vpath)} → {len(indices)} frames")

    print(f"\n  Frames saved: {total_saved}  →  {frames_dir}")
    return total_saved


# ─── Training ────────────────────────────────────────────────────────────────

def train_image_model(ai_frames_dir: str):
    """Train DeepFusionNet v5 (image model) using extracted AI frames."""
    print("\n" + "=" * 60)
    print("  Training IMAGE model (DeepFusionNet v5) with Meta AI frames")
    print("=" * 60)

    # Import here (after optional download) to keep startup fast
    sys.path.insert(0, SCRIPT_DIR)
    from train_streaming import main as stream_main

    # Patch sys.argv for train_streaming CLI
    # It uses --source local + reads from data/ai and data/real
    sys.argv = [
        "train_streaming.py",
        "--total_batches", "500",
    ]
    stream_main()


def train_video_model(ai_video_dir: str, real_video_dir: str):
    """Train VideoTemporalFusionNet using AI video clips."""
    print("\n" + "=" * 60)
    print("  Training VIDEO model (VideoTemporalFusionNet) with Meta AI clips")
    print("=" * 60)

    if not os.path.isdir(real_video_dir):
        print(f"\n  [warn] Real video dir not found: {real_video_dir}")
        print("  The video model needs both real and AI videos.")
        print(f"  Put real videos in: {real_video_dir}")
        print("  Skipping video model training.\n")
        return

    sys.path.insert(0, SCRIPT_DIR)
    import torch
    from train_deep import train_video_section

    train_video_section(
        video_dir    = real_video_dir,
        ai_video_dir = ai_video_dir,
        epochs       = 10,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Download Meta AI videos + train the detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--url", default=META_VIBES_URL,
        help="Page to download from (default: meta.ai/vibes)",
    )
    p.add_argument(
        "--count", type=int, default=50,
        help="Number of videos to download",
    )
    p.add_argument(
        "--video_dir", default=AI_VIDEO_DIR,
        help="Where to save downloaded AI video clips",
    )
    p.add_argument(
        "--frames_dir", default=AI_FRAME_DIR,
        help="Where to save extracted frames (for image model)",
    )
    p.add_argument(
        "--real_dir", default=REAL_VIDEO_DIR,
        help="Directory of real (non-AI) videos (needed for video model training)",
    )
    p.add_argument(
        "--frames_per_video", type=int, default=30,
        help="Frames to extract per video clip for image model training",
    )
    p.add_argument(
        "--model", choices=["image", "video", "both"], default="both",
        help="Which model to train after downloading",
    )
    p.add_argument(
        "--download_only", action="store_true",
        help="Download videos but skip training",
    )
    p.add_argument(
        "--train_only", action="store_true",
        help="Skip download — train on videos already in --video_dir",
    )

    args = p.parse_args()

    video_dir  = args.video_dir  if os.path.isabs(args.video_dir)  else os.path.join(REPO_ROOT, args.video_dir)
    frames_dir = args.frames_dir if os.path.isabs(args.frames_dir) else os.path.join(REPO_ROOT, args.frames_dir)
    real_dir   = args.real_dir   if os.path.isabs(args.real_dir)   else os.path.join(REPO_ROOT, args.real_dir)

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not args.train_only:
        download_meta_ai_videos(video_dir, count=args.count, url=args.url)
    else:
        print(f"  Skipping download — using existing videos in {video_dir}")

    if args.download_only:
        print("\n  --download_only set, stopping before training.")
        return

    # ── Step 2: Extract frames for image model ────────────────────────────
    if args.model in ("image", "both"):
        print(f"\n  Extracting frames for image model training ...")
        extract_frames_from_videos(video_dir, frames_dir, args.frames_per_video)

    # ── Step 3: Train ─────────────────────────────────────────────────────
    if args.model in ("image", "both"):
        train_image_model(frames_dir)

    if args.model in ("video", "both"):
        train_video_model(video_dir, real_dir)

    print("\n  Done.")


if __name__ == "__main__":
    main()
