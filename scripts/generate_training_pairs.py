#!/usr/bin/env python3
"""
generate_training_pairs.py — CLI for the AI Video Training Data Pipeline.

Usage:
    python scripts/generate_training_pairs.py --source data/real_videos/
    python scripts/generate_training_pairs.py --source data/real_urls.txt --max_pairs 10
    python scripts/generate_training_pairs.py --source https://youtube.com/watch?v=xxx
    python scripts/generate_training_pairs.py --source data/real_videos/ --dry_run
    python scripts/generate_training_pairs.py --source data/real_videos/ \\
        --generators itxio --max_pairs 5 --duration 5

Environment variables required:
    ANTHROPIC_API_KEY   — for Claude Vision prompt analysis
    ITXIO_API_KEY       — for LTX Video generation (itx.io)
"""

import argparse
import os
import sys
from pathlib import Path

# ── Make src/ importable ──────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT       = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Load .env if python-dotenv is available ───────────────────────────────────
_env_file = _ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Fallback: parse .env manually (KEY=VALUE lines)
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matched real/AI video training pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Real video source: local directory, .txt URL file, "
            "single video path, or YouTube/Vimeo URL"
        ),
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of video pairs to generate (default: all)",
    )
    parser.add_argument(
        "--generators",
        default=None,
        help=(
            "Comma-separated list of generators to use "
            "(default: all configured). Example: itxio"
        ),
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        metavar="SECONDS",
        help="Target duration of each generated clip in seconds (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/video_pairs",
        help="Directory for pair metadata (default: data/video_pairs)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        default=False,
        help="Re-process pairs that are already marked as done",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help=(
            "Discover sources and show what would be done "
            "without calling any external APIs"
        ),
    )
    return parser.parse_args()


def _check_env(dry_run: bool) -> None:
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("ITXIO_API_KEY"):
        missing.append("ITXIO_API_KEY")
    if missing and not dry_run:
        print(
            f"ERROR: Missing environment variable(s): {', '.join(missing)}\n"
            "Set them before running this script."
        )
        sys.exit(1)
    elif missing and dry_run:
        print(f"[dry_run] NOTE: Missing env vars: {', '.join(missing)}")


def main() -> None:
    args = parse_args()

    _check_env(args.dry_run)

    generators = (
        [g.strip() for g in args.generators.split(",") if g.strip()]
        if args.generators
        else None
    )

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no external APIs will be called")
        print(f"  source     : {args.source}")
        print(f"  max_pairs  : {args.max_pairs or 'all'}")
        print(f"  generators : {generators or 'all configured'}")
        print(f"  duration   : {args.duration}s")
        print(f"  output_dir : {args.output_dir}")
        print("=" * 60)

    from training_data_pipeline import VideoTrainingDataPipeline

    pipeline = VideoTrainingDataPipeline(
        real_source=args.source,
        output_dir=args.output_dir,
        generators=generators,
        duration_s=args.duration,
        dry_run=args.dry_run,
    )

    progress = pipeline.run(
        max_pairs=args.max_pairs,
        skip_existing=not args.no_skip,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = len(progress)
    done    = sum(
        1 for p in progress.values()
        if all(g.get("status") == "done" for g in p.get("generators", {}).values())
        and p.get("generators")
    )
    print(f"\nSummary: {done}/{total} pairs fully generated.")


if __name__ == "__main__":
    main()
