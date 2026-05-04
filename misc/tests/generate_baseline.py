#!/usr/bin/env python3
"""
Regenerate committed scorpion search regression baselines (full mode only).

Pass --track TRACK [TRACK ...] to regenerate only specific tracks.

Set AUTOSCALE_BENCHMARKS to the autoscale-benchmarks root directory,
or pass --benchmarks PATH to override.
"""

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TEST_REGRESSION = TESTS_DIR / "test-regression.py"


def main():
    # Strip mode flags the caller may have passed accidentally.
    stripped = [a for a in sys.argv[1:]
                if a not in ("--check", "--update")]

    cmd = [sys.executable, str(TEST_REGRESSION), "--update", "--full"] + stripped
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
