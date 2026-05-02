#!/usr/bin/env python3
"""
Regenerate committed scorpion search regression baselines.

Runs both light and full baseline generation in sequence.  Pass --light-only
or --full-only to regenerate only one mode.  Pass --track TRACK [TRACK ...] to
regenerate only specific tracks.

Set AUTOSCALE_BENCHMARKS to the autoscale-benchmarks root directory,
or pass --benchmarks PATH to override.
"""

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TEST_REGRESSION = TESTS_DIR / "test-regression.py"


def _run_update(extra_flags: list[str], forwarded: list[str]) -> int:
    cmd = [sys.executable, str(TEST_REGRESSION), "--update"] + extra_flags + forwarded
    return subprocess.run(cmd).returncode


def main():
    # Strip mode flags the caller may have passed accidentally.
    stripped = [a for a in sys.argv[1:]
                if a not in ("--check", "--update", "--light-only", "--full-only")]

    light_only = "--light-only" in sys.argv[1:]
    full_only = "--full-only" in sys.argv[1:]

    if light_only and full_only:
        sys.exit("ERROR: --light-only and --full-only are mutually exclusive.")

    rc = 0
    if not full_only:
        print("=== Generating light baselines ===")
        rc = _run_update([], stripped)
    if rc == 0 and not light_only:
        print("=== Generating full baselines ===")
        rc = _run_update(["--full"], stripped)
    sys.exit(rc)


if __name__ == "__main__":
    main()
