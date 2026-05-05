#!/usr/bin/env python3
"""
Check or update scorpion search regression baselines.

  --check   Run registered tracks and compare against committed baselines.
  --update  Regenerate committed baselines from current results.
  --full    Full mode: p01-p05, all configs, 60 s limit (default: light mode).
  --track   Run only the named track(s); default is all registered tracks.

Light mode (default): p01 only, all configs, 10 s limit; fast developer check.
Full mode (--full):   p01-p05, all configs, 60 s limit; intended as CI gate.

Set AUTOSCALE_BENCHMARKS to the autoscale-benchmarks root directory,
or pass --benchmarks PATH to override.
"""

import argparse
import os
import sys
from pathlib import Path

# Add tests dir to path so track modules can import regression_lib.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regression_heuristic import check_heuristics, update_heuristics  # noqa: E402
from regression_optimal import check_optimal, update_optimal  # noqa: E402
from regression_satisficing import check_satisficing, update_satisficing  # noqa: E402
from regression_anytime import check_anytime, update_anytime  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = Path(__file__).resolve().parent
BASELINE_DIR = TESTS_DIR / "regression-baselines"
DEFAULT_WORKERS = 4


def _get_benchmarks_path(cli_override):
    if cli_override:
        path = Path(cli_override)
    else:
        env_val = os.environ.get("AUTOSCALE_BENCHMARKS")
        if not env_val:
            sys.exit(
                "ERROR: AUTOSCALE_BENCHMARKS is not set.\n"
                "  Point it to the autoscale-benchmarks root, "
                "or use --benchmarks PATH."
            )
        path = Path(env_val)
    if not path.is_dir():
        sys.exit(f"ERROR: Benchmarks directory not found: {path}")
    return path.resolve()


# ---------------------------------------------------------------------------
# Track registry.
#
# Each entry is (name, check_fn, update_fn).
#   check_fn(benchmarks, baseline_dir, workers) -> list[str]  (failure msgs)
#   update_fn(benchmarks, baseline_dir, workers) -> None
#
# Tracks are added here as they are implemented in Phase 3:
# ---------------------------------------------------------------------------
TRACKS = [
    ("heuristics",  check_heuristics,  update_heuristics),
    ("optimal",     check_optimal,     update_optimal),
    ("satisficing", check_satisficing, update_satisficing),
    ("anytime",     check_anytime,     update_anytime),
]


def main():
    parser = argparse.ArgumentParser(
        description="Scorpion search regression checker"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Compare current results against committed baselines",
    )
    mode.add_argument(
        "--update",
        action="store_true",
        help="Regenerate committed baselines from current results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--benchmarks",
        metavar="PATH",
        help="Autoscale benchmarks root (overrides AUTOSCALE_BENCHMARKS)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full mode: p01-p05, 60 s (default: light mode p01-only, 10 s)",
    )
    track_names = [name for name, _, _ in TRACKS]
    parser.add_argument(
        "--track",
        nargs="+",
        metavar="TRACK",
        choices=track_names,
        help=f"Run only these track(s); choices: {track_names}",
    )
    args = parser.parse_args()

    benchmarks = _get_benchmarks_path(args.benchmarks)

    if args.check and not BASELINE_DIR.is_dir():
        sys.exit(
            f"ERROR: Baseline directory not found: {BASELINE_DIR}\n"
            "  Run  python misc/tests/generate_baseline.py  to create it."
        )
    if args.update:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    light = not args.full

    print(f"Repository root: {REPO_ROOT}")
    print(f"Benchmarks:      {benchmarks}")
    print(f"Baselines:       {BASELINE_DIR}")
    print(f"Workers:         {args.workers}")
    print(f"Mode:            {'CHECK' if args.check else 'UPDATE'} "
          f"({'light' if light else 'full'})")
    print()

    selected = (
        [(n, c, u) for n, c, u in TRACKS if n in args.track]
        if args.track else TRACKS
    )

    all_failures = []
    for track_name, check_fn, update_fn in selected:
        print(f"--- {track_name} ---")
        if args.check:
            failures = check_fn(benchmarks, BASELINE_DIR, args.workers, light=light)
            status = "PASS" if not failures else f"FAIL ({len(failures)} regression(s))"
            print(f"  {status}")
            for msg in failures:
                print(f"    * {msg}")
            all_failures.extend(failures)
        else:
            update_fn(benchmarks, BASELINE_DIR, args.workers, light=light)
            print("  updated")

    print("=" * 60)
    if not selected:
        print("No tracks registered; nothing to do.")
        return
    if args.check:
        if all_failures:
            print(f"FAILURE: {len(all_failures)} regression(s) detected.")
            print("  If changes are intentional, rebaseline with:")
            print("    python misc/tests/generate_baseline.py")
            sys.exit(1)
        else:
            print("SUCCESS: all tracks passed.")
    else:
        print("Baselines updated.")


if __name__ == "__main__":
    main()
