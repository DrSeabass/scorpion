#!/usr/bin/env python3
"""
Check or update scorpion search regression baselines.

  --check        Run registered tracks and compare against committed baselines.
  --update       Regenerate committed baselines from current results.
  --full         instances=[1,2,3,4,5]; mutually exclusive with --instances.
  --instances    Run only these instances; each item is integer N (=p0N across
                 all domains) or "domain/problem.pddl" (exact instance).
  --track        Run only the named track(s); default is all registered tracks.
  --config-file  JSON dict {name: [cli_args, ...]} replacing each selected
                 track's CONFIGS; on --update, results are merged into the
                 existing baseline file.
  --json-output  Write a per-instance comparison dump to PATH (--check only).

Default --check (no flags): instances=[1], 10 s limit (fast developer check).
--check --full:              instances=[1,2,3,4,5], 60 s limit (CI gate).
--check --instances ...:     custom subset, 60 s limit.

Set AUTOSCALE_BENCHMARKS to the autoscale-benchmarks root directory,
or pass --benchmarks PATH to override.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

JSON_SCHEMA_VERSION = 1

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


def _scorpion_commit():
    """Return the HEAD commit short SHA, or None if unavailable."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        )
        return r.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _parse_instance_item(item):
    """Parse a CLI --instances element as int or 'domain/problem.pddl' string."""
    try:
        return int(item)
    except ValueError:
        pass
    if "/" in item and item.endswith(".pddl"):
        domain, problem = item.split("/", 1)
        if domain and problem and "/" not in domain:
            return item
    sys.exit(
        f"ERROR: --instances item {item!r} must be either an integer "
        f"or a 'domain/problem.pddl' string"
    )


def _load_config_file(path):
    if path is None:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: --config-file {path}: {e}")
    if not isinstance(data, dict) or not data:
        sys.exit(f"ERROR: --config-file {path}: top-level must be a non-empty object")
    for name, args in data.items():
        if not isinstance(name, str):
            sys.exit(f"ERROR: --config-file {path}: keys must be strings")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            sys.exit(
                f"ERROR: --config-file {path}: value for {name!r} must be a "
                f"list of strings (e.g. [\"--search\", \"astar(...)\"])"
            )
    return data


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
    instance_group = parser.add_mutually_exclusive_group()
    instance_group.add_argument(
        "--full",
        action="store_true",
        help="Run on instances=[1,2,3,4,5] (sugar; mutually exclusive with --instances)",
    )
    instance_group.add_argument(
        "--instances",
        nargs="+",
        metavar="ITEM",
        help="Run only on these instances; each ITEM is integer N "
             "(=p0N across all domains) or 'domain/problem.pddl' (exact)",
    )
    track_names = [name for name, _, _ in TRACKS]
    parser.add_argument(
        "--track",
        nargs="+",
        metavar="TRACK",
        choices=track_names,
        help=f"Run only these track(s); choices: {track_names}",
    )
    parser.add_argument(
        "--config-file",
        metavar="PATH",
        type=Path,
        help="Replace each selected track's CONFIGS with the JSON dict at PATH "
             "(shape: {name: [cli_arg, ...], ...}); on --update the results are "
             "merged into the existing baseline file rather than overwriting it",
    )
    parser.add_argument(
        "--json-output",
        metavar="PATH",
        type=Path,
        help="Write a per-instance comparison JSON dump to PATH (--check only). "
             "Use the committed regression-baselines/*.json files for --update output.",
    )
    args = parser.parse_args()

    benchmarks = _get_benchmarks_path(args.benchmarks)
    configs_override = _load_config_file(args.config_file)

    if args.instances:
        instances_arg = [_parse_instance_item(x) for x in args.instances]
    elif args.full:
        instances_arg = [1, 2, 3, 4, 5]
    else:
        instances_arg = None

    if args.update and instances_arg is None and configs_override is None:
        parser.error(
            "--update requires an explicit scope: --full (rebaseline "
            "everything), --instances ... (regenerate a subset), or "
            "--config-file ... (regenerate specific configs)"
        )

    if args.json_output and args.update:
        parser.error(
            "--json-output is only valid in --check mode; the updated "
            "regression-baselines/*.json files are the structured output "
            "for --update"
        )

    if args.check and not BASELINE_DIR.is_dir():
        sys.exit(
            f"ERROR: Baseline directory not found: {BASELINE_DIR}\n"
            "  Run  python misc/tests/generate_baseline.py  to create it."
        )
    if args.update:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    if instances_arg is None:
        scope = "instances=[1] (default)"
    elif args.full:
        scope = "instances=[1,2,3,4,5] (--full)"
    else:
        scope = f"instances={instances_arg}"

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    scorpion_commit = _scorpion_commit()

    print(f"Repository root: {REPO_ROOT}")
    print(f"Benchmarks:      {benchmarks}")
    print(f"Baselines:       {BASELINE_DIR}")
    print(f"Workers:         {args.workers}")
    print(f"Mode:            {'CHECK' if args.check else 'UPDATE'}  ({scope})")
    print()

    selected = (
        [(n, c, u) for n, c, u in TRACKS if n in args.track]
        if args.track else TRACKS
    )

    all_failures = []
    track_comparisons = {}
    any_error = False
    for track_name, check_fn, update_fn in selected:
        print(f"--- {track_name} ---")
        if args.check:
            comp = check_fn(benchmarks, BASELINE_DIR, args.workers,
                            configs=configs_override,
                            instances=instances_arg)
            track_comparisons[track_name] = comp
            outcome = comp.get("outcome", "error")
            msgs = comp.get("messages", [])
            if outcome == "pass":
                status = "PASS"
            elif outcome == "fail":
                status = f"FAIL ({len(msgs)} regression(s))"
            else:
                status = f"ERROR ({len(msgs)} message(s))"
                any_error = True
            print(f"  {status}")
            for msg in msgs:
                print(f"    * {msg}")
            if outcome != "pass":
                all_failures.extend(msgs)
        else:
            update_fn(benchmarks, BASELINE_DIR, args.workers,
                      configs=configs_override,
                      instances=instances_arg)
            print("  updated")

    if args.check and args.json_output:
        # `messages` is a stdout-only convenience field; strip it from the
        # JSON shape so consumers don't ingest derived data.
        json_tracks = {}
        for name, comp in track_comparisons.items():
            json_tracks[name] = {
                "outcome": comp["outcome"],
                "per_config": comp["per_config"],
                "runs": comp["runs"],
            }
        overall = (
            "fail"
            if any_error or any(c["outcome"] != "pass"
                                for c in track_comparisons.values())
            else "pass"
        )
        payload = {
            "schema_version": JSON_SCHEMA_VERSION,
            "mode": "check",
            "scorpion_commit": scorpion_commit,
            "started_at": started_at,
            "instances": (
                instances_arg if instances_arg is not None else [1]
            ),
            "tracks": json_tracks,
            "outcome": overall,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nJSON output: {args.json_output}")

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
