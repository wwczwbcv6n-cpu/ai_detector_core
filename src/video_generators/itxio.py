"""
itxio.py — LTX Video generator via the ltx.video API (Lightricks).

API docs: https://docs.ltx.video
Base URL:  https://api.ltx.video/v1
Auth:      Authorization: Bearer <ITXIO_API_KEY>

The /text-to-video endpoint is **synchronous** — it blocks until the video is
ready and returns the MP4 binary directly (Content-Type: video/mp4).
The x-request-id response header is used as the job ID for tracking.

All endpoint paths, auth details, and body schema live in
config/video_generators.json so no code changes are needed to reconfigure.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from .base import (
    BaseVideoGenerator,
    GenerationJob,
    STATUS_SUBMITTED,
    STATUS_DONE,
    STATUS_FAILED,
)

_DEFAULTS: dict[str, Any] = {
    "base_url":        "https://api.ltx.video/v1",
    "submit_endpoint": "/text-to-video",
    "auth_header":     "Authorization",
    "auth_prefix":     "Bearer ",
    "daily_limit":     None,
    "timeout_s":       300,
    "submit_body": {
        "prompt":     "{prompt}",
        "model":      "ltx-2-3-pro",
        "duration":   "{duration_s}",
        "resolution": "1280x720",
    },
}


class ItxIOGenerator(BaseVideoGenerator):
    """
    LTX Video (ltx.video) synchronous text-to-video generator.
    One POST call returns the MP4 binary directly — no async polling needed.
    """

    name = "itxio"

    def __init__(self, cfg: dict | None = None):
        merged: dict[str, Any] = {**_DEFAULTS, **(cfg or {})}

        self._base_url         = merged["base_url"].rstrip("/")
        self._submit_ep        = merged["submit_endpoint"]
        self._auth_header      = merged["auth_header"]
        self._auth_prefix      = merged["auth_prefix"]
        self._submit_body_tmpl = merged["submit_body"]
        self._timeout_s        = int(merged.get("timeout_s", 300))
        self.daily_limit       = merged.get("daily_limit")

        api_key_env = merged.get("api_key_env", "ITXIO_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")
        if not self._api_key:
            raise EnvironmentError(
                f"Missing API key — set env var {api_key_env}"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        value = f"{self._auth_prefix}{self._api_key}".strip()
        return {self._auth_header: value, "Accept": "video/mp4, application/json"}

    def _build_body(self, prompt: str, duration_s: int) -> dict:
        body: dict[str, Any] = {}
        for k, v in self._submit_body_tmpl.items():
            if isinstance(v, str):
                v = v.replace("{prompt}", prompt).replace(
                    "{duration_s}", str(duration_s)
                )
                if v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            body[k] = v
        return body

    def _url(self, endpoint: str) -> str:
        return self._base_url + endpoint

    # ── BaseVideoGenerator interface ──────────────────────────────────────────

    def submit(self, prompt: str, duration_s: int, pair_id: str) -> GenerationJob:
        """
        For the synchronous ltx.video API this method is a no-op placeholder.
        Use generate_and_download() for the actual blocking call.
        Returns a pending job so the orchestrator can call poll() immediately.
        """
        job_id = f"ltx_{pair_id}_{uuid.uuid4().hex[:8]}"
        return GenerationJob(
            job_id=job_id,
            generator_name=self.name,
            prompt=prompt,
            pair_id=pair_id,
            status=STATUS_SUBMITTED,
        )

    def poll(self, job: GenerationJob) -> GenerationJob:
        # Synchronous API — nothing to poll.  Caller should use generate_and_download().
        return job

    def download(self, job: GenerationJob, dest_path: Path) -> Path:
        if not job.output_path:
            raise ValueError(f"No output_path on job {job.job_id}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # If output_path is already a local file written by generate_and_download, just move it
        src = Path(job.output_path)
        if src.exists() and str(src) != str(dest_path):
            import shutil
            shutil.move(str(src), str(dest_path))
        return dest_path

    # ── Preferred entry point ─────────────────────────────────────────────────

    def generate_and_download(
        self,
        prompt: str,
        duration_s: int,
        pair_id: str,
        dest_path: Path,
    ) -> GenerationJob:
        """
        Single blocking call: POST → receive MP4 binary → write to dest_path.
        Returns a completed GenerationJob.
        """
        job = self.submit(prompt, duration_s, pair_id)

        body = self._build_body(prompt, duration_s)
        try:
            resp = requests.post(
                self._url(self._submit_ep),
                json=body,
                headers=self._headers(),
                timeout=self._timeout_s,
                stream=True,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            job.status = STATUS_FAILED
            job.error = str(exc)
            return job

        # Use x-request-id if the server returns one
        server_id = resp.headers.get("x-request-id")
        if server_id:
            job.job_id = server_id

        content_type = resp.headers.get("Content-Type", "")
        if "video" in content_type or "octet-stream" in content_type:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            job.status = STATUS_DONE
            job.output_path = str(dest_path)
            job.completed_at = time.time()
        else:
            # Unexpected response — save body for debugging
            job.status = STATUS_FAILED
            try:
                data = resp.json()
                job.error = str(data)
            except Exception:
                job.error = f"Unexpected Content-Type: {content_type}"

        return job
