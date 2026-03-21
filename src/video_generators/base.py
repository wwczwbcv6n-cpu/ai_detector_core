"""
base.py — Abstract base class and shared data types for video generators.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Job status constants ──────────────────────────────────────────────────────
STATUS_PENDING     = "pending"
STATUS_SUBMITTED   = "submitted"
STATUS_PROCESSING  = "processing"
STATUS_DONE        = "done"
STATUS_FAILED      = "failed"


@dataclass
class GenerationJob:
    job_id: str
    generator_name: str
    prompt: str
    pair_id: str
    status: str = STATUS_PENDING          # pending | submitted | processing | done | failed
    output_path: Optional[str] = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (STATUS_DONE, STATUS_FAILED)


class BaseVideoGenerator(ABC):
    """Abstract base for all video generation backends."""

    name: str = "base"
    daily_limit: Optional[int] = None   # None = unlimited

    # ── Must implement ────────────────────────────────────────────────────────

    @abstractmethod
    def submit(self, prompt: str, duration_s: int, pair_id: str) -> GenerationJob:
        """Submit a generation request.  Returns a GenerationJob with status=submitted."""

    @abstractmethod
    def poll(self, job: GenerationJob) -> GenerationJob:
        """Check and update the job status.  Returns updated GenerationJob."""

    @abstractmethod
    def download(self, job: GenerationJob, dest_path: Path) -> Path:
        """Download the finished video to dest_path.  Returns dest_path."""

    # ── Optional / helpers ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Returns True if the generator can accept new jobs right now."""
        return True

    def wait_for_completion(
        self,
        job: GenerationJob,
        timeout_s: int = 600,
        poll_interval_s: int = 10,
    ) -> GenerationJob:
        """
        Block until job reaches a terminal state or timeout.
        Uses exponential backoff capped at poll_interval_s.
        """
        deadline = time.time() + timeout_s
        wait = min(10, poll_interval_s)

        while not job.is_terminal:
            if time.time() > deadline:
                job.status = STATUS_FAILED
                job.error = f"Timeout after {timeout_s}s"
                break
            time.sleep(wait)
            wait = min(wait * 1.5, poll_interval_s)
            try:
                job = self.poll(job)
            except Exception as exc:
                job.status = STATUS_FAILED
                job.error = str(exc)
                break

        if job.status == STATUS_DONE:
            job.completed_at = time.time()
        return job
