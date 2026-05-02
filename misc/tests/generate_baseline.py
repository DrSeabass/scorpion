#!/usr/bin/env python3
"""
Regenerate committed scorpion search regression baselines.

This is the canonical tool for updating baselines after an intentional
algorithm or configuration change.  Equivalent to running
  python misc/tests/test-regression.py --update
but with a developer-facing interface.

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
    # Forward all arguments to test-regression.py --update.
    # Strip --check / --update if the caller accidentally included one.
    forwarded = [a for a in sys.argv[1:] if a not in ("--check", "--update")]
    cmd = [sys.executable, str(TEST_REGRESSION), "--update"] + forwarded
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
