#!/usr/bin/env python3
"""
Regenerate committed scorpion search regression baselines (full mode only).

Tracks group into two benchmark sets, invoked sequentially:
    21.11-optimal-strips → heuristics, optimal
    21.11-agile-strips   → satisficing, anytime

Pass --track TRACK [TRACK ...] to regenerate only specific tracks; the
script automatically restricts each invocation to the relevant set.

Set AUTOSCALE_BENCHMARKS to the autoscale-benchmarks root directory,
or pass --benchmarks PATH to override.  All other arguments are
forwarded to test-regression.py unchanged.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TEST_REGRESSION = TESTS_DIR / "test-regression.py"

TRACK_SETS = {
    "heuristics":  "21.11-optimal-strips",
    "optimal":     "21.11-optimal-strips",
    "satisficing": "21.11-agile-strips",
    "anytime":     "21.11-agile-strips",
}


def _benchmarks_path(cli_override):
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


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate committed scorpion regression baselines",
    )
    parser.add_argument(
        "--track",
        nargs="+",
        choices=list(TRACK_SETS),
        metavar="TRACK",
        help=f"Restrict to these track(s); choices: {list(TRACK_SETS)}",
    )
    parser.add_argument(
        "--benchmarks",
        metavar="PATH",
        help="Autoscale benchmarks root (overrides AUTOSCALE_BENCHMARKS)",
    )
    args, forward = parser.parse_known_args()

    if any(a == "--domain-dir" or a.startswith("--domain-dir=")
           for a in forward):
        sys.exit(
            "ERROR: --domain-dir cannot be passed to generate_baseline.py; "
            "the script computes per-set domain-dirs itself.  Use "
            "test-regression.py directly for a custom --domain-dir."
        )
    forward = [a for a in forward if a not in ("--check", "--update")]

    benchmarks = _benchmarks_path(args.benchmarks)

    selected = args.track if args.track else list(TRACK_SETS)
    by_set = {}
    for track in selected:
        by_set.setdefault(TRACK_SETS[track], []).append(track)

    for subdir, tracks in by_set.items():
        domain_dir = benchmarks / subdir
        cmd = [sys.executable, str(TEST_REGRESSION),
               "--update", "--full",
               "--domain-dir", str(domain_dir),
               "--track", *tracks] + forward
        print(f"\n[generate_baseline] {subdir}: {', '.join(tracks)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
