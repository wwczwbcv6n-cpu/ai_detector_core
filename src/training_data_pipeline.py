"""
training_data_pipeline.py — End-to-end pipeline for generating matched real/AI video pairs.

For each real video:
  1. Copy to  data/real_videos/{pair_id}/real.mp4
  2. Analyze with VideoPromptAnalyzer → prompt
  3. Submit to VideoGenOrchestrator → all configured generators
  4. Poll until all jobs finish (or 10-min timeout per job)
  5. Save AI videos to data/ai_videos/{pair_id}/{generator}.mp4
  6. Update data/video_pairs/progress.json

Real video sources supported:
  - Local directory containing .mp4 / .mov / .mkv files
  - Plain text file listing YouTube/Vimeo URLs (one per line)
  - Single YouTube/Vimeo URL string
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

_PROGRESS_FILE = _ROOT / "data" / "video_pairs" / "progress.json"
_REAL_BASE     = _ROOT / "data" / "real_videos"
_AI_BASE       = _ROOT / "data" / "ai_videos"
_PAIRS_BASE    = _ROOT / "data" / "video_pairs"


# ── Progress helpers ──────────────────────────────────────────────────────────

def _load_progress() -> dict:
    if _PROGRESS_FILE.exists():
        try:
            return json.loads(_PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_progress(progress: dict) -> None:
    _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


# ── Source discovery ──────────────────────────────────────────────────────────

def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _collect_sources(source: str) -> list[str]:
    """
    Returns a list of local video paths or URLs.
    source can be:
      - A local directory path
      - A .txt file of URLs / local paths (one per line)
      - A single URL
      - A single local video file path
    """
    p = Path(source)

    if p.is_dir():
        return [
            str(f) for f in sorted(p.rglob("*"))
            if f.suffix.lower() in VIDEO_EXTS
        ]

    if p.is_file() and p.suffix.lower() == ".txt":
        lines = [l.strip() for l in p.read_text().splitlines()]
        return [l for l in lines if l and not l.startswith("#")]

    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        return [str(p)]

    if _is_url(source):
        return [source]

    raise ValueError(f"Cannot determine source type for: {source!r}")


def _pair_id_from_source(source: str, index: int) -> str:
    """Generate a deterministic pair ID from a source path or URL."""
    if _is_url(source):
        # extract last path component or video ID from URL
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", source.rstrip("/").split("/")[-1])[:32]
        return f"pair_{index:04d}_{slug}"
    stem = Path(source).stem
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:40]
    return f"pair_{index:04d}_{clean}"


# ── YouTube download (reuses train_deep.py logic) ─────────────────────────────

def _download_yt(url: str, dest_dir: Path) -> Optional[Path]:
    """Download a YouTube/Vimeo video to dest_dir.  Returns the output path."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed: pip install yt-dlp")

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": (
            "bestvideo[vcodec^=avc1][ext=mp4][height<=720]+bestaudio[ext=m4a]"
            "/bestvideo[ext=mp4][height<=720][vcodec!^=av01]+bestaudio[ext=m4a]"
            "/best[ext=mp4][height<=720]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith(".mp4"):
            filename = filename.rsplit(".", 1)[0] + ".mp4"
    path = Path(filename)
    return path if path.exists() else None


# ── Main pipeline class ───────────────────────────────────────────────────────

class VideoTrainingDataPipeline:
    """End-to-end pipeline: real video → prompt → AI variants → training pairs."""

    def __init__(
        self,
        real_source: str,
        output_dir: str = "data/video_pairs",
        generators: Optional[list[str]] = None,
        duration_s: int = 5,
        dry_run: bool = False,
    ):
        self.real_source = real_source
        self.output_dir  = Path(output_dir)
        self.generators  = generators    # None = all configured
        self.duration_s  = duration_s
        self.dry_run     = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _REAL_BASE.mkdir(parents=True, exist_ok=True)
        _AI_BASE.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        max_pairs: Optional[int] = None,
        skip_existing: bool = True,
    ) -> dict:
        """
        Process all sources.  Returns final progress dict.
        """
        from video_prompt_analyzer import VideoPromptAnalyzer
        from video_generators.orchestrator import VideoGenOrchestrator

        sources = _collect_sources(self.real_source)
        if max_pairs:
            sources = sources[:max_pairs]

        if not sources:
            print("No source videos found.")
            return {}

        print(f"Processing {len(sources)} source video(s) ...")

        analyzer    = VideoPromptAnalyzer()
        orchestrator = VideoGenOrchestrator(only=self.generators)
        progress    = _load_progress()

        for idx, src in enumerate(sources):
            pair_id = _pair_id_from_source(src, idx)
            print(f"\n[{idx+1}/{len(sources)}] pair_id={pair_id}")
            print(f"  source: {src}")

            if skip_existing and pair_id in progress:
                existing = progress[pair_id]
                # skip if all generators in progress are done
                gens = existing.get("generators", {})
                if gens and all(g.get("status") == "done" for g in gens.values()):
                    print("  Skipping — already complete.")
                    continue

            # ── Step 1: obtain local real video ──────────────────────────────
            real_dir  = _REAL_BASE / pair_id
            real_path = real_dir / "real.mp4"
            real_dir.mkdir(parents=True, exist_ok=True)

            if not real_path.exists():
                if _is_url(src):
                    print("  Downloading real video ...")
                    if self.dry_run:
                        print("  [dry_run] Would download from:", src)
                        continue
                    tmp = _download_yt(src, dest_dir=real_dir)
                    if not tmp:
                        print("  Download failed — skipping.")
                        continue
                    if tmp != real_path:
                        shutil.move(str(tmp), str(real_path))
                else:
                    shutil.copy2(src, real_path)

            # ── Step 2: analyze with Claude Vision ───────────────────────────
            print("  Analyzing with Claude Vision ...")
            if self.dry_run:
                print("  [dry_run] Would call Claude Vision API")
                continue

            try:
                vp = analyzer.analyze(
                    str(real_path),
                    pair_id=pair_id,
                    output_dir=str(self.output_dir),
                )
            except Exception as exc:
                print(f"  Prompt analysis failed: {exc}")
                continue

            prompt_text = vp.generation_prompt
            print(f"  Prompt ({len(prompt_text)} chars): {prompt_text[:80]}...")

            # ── Update progress ───────────────────────────────────────────────
            progress[pair_id] = {
                "real": str(real_path),
                "prompt": prompt_text,
                "generators": {},
            }
            _save_progress(progress)

            # ── Step 3: generate AI variants ─────────────────────────────────
            ai_dir = _AI_BASE / pair_id
            ai_dir.mkdir(parents=True, exist_ok=True)

            try:
                jobs = orchestrator.generate(
                    prompt=prompt_text,
                    pair_id=pair_id,
                    output_dir=ai_dir,
                    duration_s=self.duration_s,
                    use_all=True,
                )
            except Exception as exc:
                print(f"  Generation failed: {exc}")
                continue

            # ── Step 4: update progress with results ─────────────────────────
            for job in jobs:
                progress[pair_id]["generators"][job.generator_name] = {
                    "status":  job.status,
                    "job_id":  job.job_id,
                    "path":    job.output_path,
                    "error":   job.error,
                }
            _save_progress(progress)

            done  = sum(1 for j in jobs if j.status == "done")
            total = len(jobs)
            print(f"  Generated {done}/{total} AI video(s) for {pair_id}")

        print(f"\nPipeline complete.  Progress saved to {_PROGRESS_FILE}")
        return progress
