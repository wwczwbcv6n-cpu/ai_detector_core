"""
video_prompt_analyzer.py — Analyze a real video and generate an AI generation prompt.

Uses Claude Vision API to extract scene description, motion, lighting, and other
attributes from key frames, then assembles a structured prompt for AI video generators.
"""

import base64
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FRAMES = 8
FRAME_SIZE = (768, 432)   # resize frames before sending to Vision API
ANTHROPIC_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a video analysis assistant helping to create prompts for AI video generation.
Analyze the provided video frames and extract detailed visual and motion attributes.
Respond ONLY with valid JSON — no markdown fences, no extra text.
"""

_USER_PROMPT = """\
Analyze these {n} key frames from a video clip and return a JSON object with exactly
these fields:

{{
  "scene": "2-3 sentence description of the setting and environment",
  "subjects": "description of people, animals, or main objects and their actions",
  "camera_movement": "one of: static | pan_left | pan_right | tilt_up | tilt_down | zoom_in | zoom_out | tracking | handheld | aerial | rotation | dolly",
  "lighting": "description of lighting quality, direction, and color temperature",
  "color_palette": "dominant colors and overall color mood",
  "visual_style": "one of: cinematic | documentary | vlog | news | nature | sports | animation | timelapse | music_video",
  "motion_intensity": "one of: static | slow | medium | fast | very_fast",
  "mood": "emotional tone and atmosphere",
  "duration_estimate_s": <integer seconds, estimate from context>,
  "generation_prompt": "A single, rich, concrete text-to-video generation prompt (150-250 words) that captures ALL the above attributes and would reproduce this video's visual style as closely as possible."
}}
"""


@dataclass
class VideoPrompt:
    generation_prompt: str
    scene: str = ""
    subjects: str = ""
    camera_movement: str = "static"
    lighting: str = ""
    color_palette: str = ""
    visual_style: str = "cinematic"
    motion_intensity: str = "medium"
    mood: str = ""
    duration_estimate_s: int = 5
    metadata: dict = field(default_factory=dict)


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_key_frames(video_path: str, n: int = MAX_FRAMES) -> list[np.ndarray]:
    """Extract n evenly-spaced frames (first, last, and distributed middle)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # fallback: read up to n frames sequentially
        frames = []
        while len(frames) < n:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        return frames

    indices = _spread_indices(total, n)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def _spread_indices(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total))
    step = (total - 1) / (n - 1)
    return [round(i * step) for i in range(n)]


def _frame_to_b64(frame: np.ndarray) -> str:
    """Resize and encode a BGR frame as base64 JPEG."""
    resized = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    from PIL import Image
    import io
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Claude Vision call ────────────────────────────────────────────────────────

def _call_claude_vision(frames: list[np.ndarray]) -> dict:
    """Send frames to Claude Vision and return parsed JSON dict."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env

    content: list[dict] = []
    for i, frame in enumerate(frames):
        b64 = _frame_to_b64(frame)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
    content.append({
        "type": "text",
        "text": _USER_PROMPT.format(n=len(frames)),
    })

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    # strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── Public API ────────────────────────────────────────────────────────────────

class VideoPromptAnalyzer:
    """Analyze a real video clip and produce a structured generation prompt."""

    def __init__(self, n_frames: int = MAX_FRAMES):
        self.n_frames = n_frames

    def analyze(
        self,
        video_path: str,
        pair_id: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> VideoPrompt:
        """
        Analyze video_path and return a VideoPrompt.

        If pair_id and output_dir are given, saves:
          {output_dir}/{pair_id}/prompt.txt
          {output_dir}/{pair_id}/metadata.json
        """
        frames = extract_key_frames(video_path, self.n_frames)
        if not frames:
            raise RuntimeError(f"No frames extracted from {video_path}")

        data = _call_claude_vision(frames)

        vp = VideoPrompt(
            generation_prompt=data.get("generation_prompt", ""),
            scene=data.get("scene", ""),
            subjects=data.get("subjects", ""),
            camera_movement=data.get("camera_movement", "static"),
            lighting=data.get("lighting", ""),
            color_palette=data.get("color_palette", ""),
            visual_style=data.get("visual_style", "cinematic"),
            motion_intensity=data.get("motion_intensity", "medium"),
            mood=data.get("mood", ""),
            duration_estimate_s=int(data.get("duration_estimate_s", 5)),
            metadata={"source_video": video_path, "n_frames_analyzed": len(frames)},
        )

        if pair_id and output_dir:
            self._save(vp, pair_id, output_dir)

        return vp

    @staticmethod
    def _save(vp: VideoPrompt, pair_id: str, output_dir: str) -> None:
        pair_dir = Path(output_dir) / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / "prompt.txt").write_text(vp.generation_prompt)
        meta = asdict(vp)
        meta.pop("generation_prompt", None)
        (pair_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
