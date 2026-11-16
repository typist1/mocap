"""CLI entry point for the 3D motion capture pipeline.

Usage
-----
    mocap video.mp4
    mocap video.mp4 --output-dir ./out --formats bvh npz json
    mocap video.mp4 --config configs/default.yaml --device cuda
    mocap --version
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Logging bootstrap (before importing heavy modules so errors are visible)
# ---------------------------------------------------------------------------

def _bootstrap_logging(level: str) -> None:
    """Configure root logger with a simple format before setup_logging() is called."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mocap",
        description="3D human motion capture from monocular video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mocap input.mp4\n"
            "  mocap input.mp4 --output-dir results/ --device cuda\n"
            "  mocap input.mp4 --formats bvh npz --cutoff 4.0 --fps 60\n"
        ),
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a YAML config file. Defaults to configs/default.yaml if present.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Directory where output files are saved (default: current directory).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Target animation FPS (default: 30.0, or value from config).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        metavar="HZ",
        help="Butterworth smoothing cutoff frequency in Hz (default: 6.0). Overrides config.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview video generation.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        metavar="FMT",
        help='Export formats (default: bvh npz json). Any subset of: bvh npz json.',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu", "mps"],
        help='Compute device (default: auto, selects cuda > mps > cpu).',
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print version and exit.",
    )

    return parser


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

def _load_settings(config_path: Optional[str]):
    """Import and load Settings, with a friendly error if the module is missing."""
    try:
        from src.config.settings import load_settings
    except ImportError as exc:
        logging.error("Cannot import settings module: %s", exc)
        sys.exit(1)

    try:
        return load_settings(config_path)
    except FileNotFoundError as exc:
        logging.error("Config file not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.error("Failed to load config: %s", exc)
        sys.exit(1)


def _apply_cli_overrides(settings, args: argparse.Namespace) -> None:
    """Overlay parsed CLI arguments on top of the loaded settings."""
    if args.fps is not None:
        settings.video.target_fps = args.fps
    if args.cutoff is not None:
        settings.filtering.butterworth_cutoff = args.cutoff
    if args.device is not None:
        settings.device = args.device
    if args.formats is not None:
        # Validate format names
        valid = {"bvh", "npz", "json"}
        bad = [f for f in args.formats if f.lower() not in valid]
        if bad:
            logging.warning("Unknown export format(s) ignored: %s", bad)
        settings.animation.export_formats = [
            f.lower() for f in args.formats if f.lower() in valid
        ]
    if args.output_dir != ".":
        settings.animation.output_dir = args.output_dir
    if args.log_level != "INFO":
        settings.log_level = args.log_level
    # --no-preview: not a settings field yet — stored on a synthesised attribute
    # so downstream code can check it if desired.
    settings._no_preview = args.no_preview  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Setup logging (project logger)
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    """Call the project-level setup_logging if available, else use basicConfig."""
    try:
        from src.utils.logging import setup_logging
        setup_logging(level)
    except ImportError:
        # Already bootstrapped with basicConfig — just adjust the level.
        logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def _print_summary(result) -> None:
    """Print a human-readable summary of ProcessingResult to stdout."""
    sep = "-" * 60
    print(sep)
    print("Motion Capture — Processing Summary")
    print(sep)
    print(f"  Source video    : {result.source_video}")
    print(f"  Frames          : {result.frame_count}")
    print(f"  Detection rate  : {result.detection_rate * 100:.1f}%")
    print(f"  Processing time : {result.processing_time_sec:.2f}s")
    if result.frame_count > 0 and result.processing_time_sec > 0:
        throughput = result.frame_count / result.processing_time_sec
        print(f"  Throughput      : {throughput:.1f} fps")
    if result.output_files:
        print("  Output files    :")
        for fmt, path in sorted(result.output_files.items()):
            print(f"    [{fmt:>5}] {path}")
    else:
        print("  Output files    : (none written)")
    if result.metrics is not None:
        print(f"  Metrics         : {result.metrics}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, run the pipeline, and return an exit code.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        0 on success, 1 on any error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Bootstrap logging early so we capture import errors
    _bootstrap_logging(args.log_level)

    # Full project logging setup
    _setup_logging(args.log_level)

    log = logging.getLogger(__name__)
    log.info("mocap %s — starting", __version__)

    # Load and override settings
    settings = _load_settings(args.config)
    _apply_cli_overrides(settings, args)

    # Resolve video path
    video_path = Path(args.video)

    # Import pipeline
    try:
        from src.pipeline import MocapPipeline
    except ImportError as exc:
        log.error("Cannot import pipeline: %s", exc)
        return 1

    # Instantiate and run
    pipeline = MocapPipeline(settings)
    output_dir = Path(settings.animation.output_dir)

    try:
        result = pipeline.process(video_path=video_path, output_dir=output_dir)
    except FileNotFoundError as exc:
        log.error("Input file not found: %s", exc)
        return 1
    except RuntimeError as exc:
        log.error("Pipeline error: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 1
    except Exception as exc:
        log.exception("Unexpected error during processing: %s", exc)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
