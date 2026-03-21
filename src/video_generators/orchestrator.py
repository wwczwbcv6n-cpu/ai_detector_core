"""
orchestrator.py — Manages multiple video generators, daily limits, and concurrent polling.

Config: config/video_generators.json
Usage limits: data/video_pairs/generator_limits.json (auto-resets at midnight UTC)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import BaseVideoGenerator, GenerationJob, STATUS_DONE, STATUS_FAILED

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]   # project root
_CONFIG_FILE   = _ROOT / "config" / "video_generators.json"
_LIMITS_FILE   = _ROOT / "data" / "video_pairs" / "generator_limits.json"

# ── Registry: name → class (import lazily to avoid hard deps) ─────────────────
_REGISTRY: dict[str, type[BaseVideoGenerator]] = {}


def _register(name: str, cls: type[BaseVideoGenerator]) -> None:
    _REGISTRY[name] = cls


def _load_registry() -> None:
    from .itxio import ItxIOGenerator
    _register("itxio", ItxIOGenerator)
    # Add more generators here as they become available:
    # from .kling import KlingGenerator
    # _register("kling", KlingGenerator)


# ── Daily limit tracker ───────────────────────────────────────────────────────

class _LimitTracker:
    def __init__(self, path: Path = _LIMITS_FILE):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2))

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def used(self, generator_name: str) -> int:
        day = self._today()
        rec = self._data.get(generator_name, {})
        if rec.get("date") != day:
            return 0
        return rec.get("count", 0)

    def increment(self, generator_name: str) -> None:
        day = self._today()
        rec = self._data.get(generator_name, {})
        if rec.get("date") != day:
            rec = {"date": day, "count": 0}
        rec["count"] = rec.get("count", 0) + 1
        self._data[generator_name] = rec
        self._save()

    def can_use(self, gen: BaseVideoGenerator) -> bool:
        if gen.daily_limit is None:
            return True
        return self.used(gen.name) < gen.daily_limit


# ── Orchestrator ──────────────────────────────────────────────────────────────

class VideoGenOrchestrator:
    """
    Manages all configured generators, submits to all available ones,
    and polls concurrently until completion.
    """

    def __init__(
        self,
        config_path: Path = _CONFIG_FILE,
        only: Optional[list[str]] = None,
    ):
        _load_registry()
        self._limits = _LimitTracker()
        self._generators: dict[str, BaseVideoGenerator] = {}
        self._load_generators(config_path, only)

    def _load_generators(
        self, config_path: Path, only: Optional[list[str]]
    ) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"Generator config not found: {config_path}")

        cfg = json.loads(config_path.read_text())
        for name, gen_cfg in cfg.items():
            if only and name not in only:
                continue
            cls = _REGISTRY.get(name)
            if cls is None:
                print(f"  [orchestrator] Unknown generator '{name}' — skipping")
                continue
            try:
                self._generators[name] = cls(gen_cfg)
                print(f"  [orchestrator] Loaded generator: {name}")
            except EnvironmentError as e:
                print(f"  [orchestrator] {name} unavailable: {e}")
            except Exception as e:
                print(f"  [orchestrator] Failed to load {name}: {e}")

    def available_generators(self) -> list[BaseVideoGenerator]:
        return [
            g for g in self._generators.values()
            if self._limits.can_use(g)
        ]

    def generate(
        self,
        prompt: str,
        pair_id: str,
        output_dir: Path,
        duration_s: int = 5,
        use_all: bool = True,
    ) -> list[GenerationJob]:
        """
        Submit the prompt to all available generators (or the first if use_all=False).
        Polls concurrently and downloads finished videos.

        Returns list of completed GenerationJob objects.
        """
        generators = self.available_generators()
        if not generators:
            raise RuntimeError("No video generators available (check API keys + daily limits)")

        if not use_all:
            generators = generators[:1]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run all generators concurrently.
        # Prefer generate_and_download() for synchronous APIs (e.g. itxio),
        # fall back to submit → wait_for_completion → download for async APIs.

        def _run_generator(gen: BaseVideoGenerator) -> GenerationJob:
            dest = output_dir / f"{gen.name}.mp4"
            if hasattr(gen, "generate_and_download"):
                print(f"  [{gen.name}] Generating (synchronous) for pair {pair_id} ...")
                return gen.generate_and_download(prompt, duration_s, pair_id, dest)  # type: ignore[attr-defined]
            job = gen.submit(prompt, duration_s, pair_id)
            print(f"  [{gen.name}] Submitted job {job.job_id} for pair {pair_id}")
            job = gen.wait_for_completion(job)
            if job.status == STATUS_DONE:
                try:
                    gen.download(job, dest)
                    job.output_path = str(dest)
                except Exception as exc:
                    job.status = STATUS_FAILED
                    job.error = f"Download failed: {exc}"
            if job.status == STATUS_DONE:
                print(f"  [{gen.name}] Done → {job.output_path}")
            else:
                print(f"  [{gen.name}] Failed: {job.error}")
            return job

        completed: list[GenerationJob] = []
        with ThreadPoolExecutor(max_workers=len(generators)) as pool:
            futures = {pool.submit(_run_generator, gen): gen for gen in generators}
            for fut in as_completed(futures):
                gen = futures[fut]
                try:
                    job = fut.result()
                    self._limits.increment(gen.name)
                    completed.append(job)
                except Exception as exc:
                    dummy = GenerationJob(
                        job_id="error",
                        generator_name=gen.name,
                        prompt=prompt,
                        pair_id=pair_id,
                        status=STATUS_FAILED,
                        error=str(exc),
                    )
                    completed.append(dummy)

        return completed

    def list_generators(self) -> list[str]:
        return list(self._generators.keys())
