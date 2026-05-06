#!/usr/bin/env python3
"""
Check, update, or iterate (dev mode) scorpion search regression baselines.

  --check          Run registered tracks and compare against committed baselines.
  --update         Regenerate committed baselines from current results.
  --dev            Run + write a versioned iteration file under --baseline-dir,
                   for the LLM-driven algorithm-development loop.  Always exits 0;
                   the iteration file is the structured output.  Self-comparison
                   against the previous iteration is layered on by a follow-on
                   step.
  --domain-dir     (required) Path to a benchmark *set* (parent of per-domain
                   dirs, each holding p01.pddl, ...) or a single *domain* folder
                   (p01.pddl, ... directly).  Auto-detected.
  --baseline-dir   Read/write baselines under PATH instead of the default
                   misc/tests/regression-baselines/.  Required when --dev is
                   set; optional otherwise.  --check requires it to exist;
                   --update and --dev create it.
  --full           instances=[1,2,3,4,5]; mutually exclusive with --instances.
  --instances      Run only these instances, by integer id; each ITEM is an
                   integer N meaning p0N.pddl in every domain under --domain-dir.
  --track          Run only the named track(s); default is all registered tracks.
  --config-file    JSON dict {name: [cli_args, ...]} replacing each selected
                   track's CONFIGS; on --update, results are merged into the
                   existing baseline file.
  --json-output    Write a per-instance comparison dump to PATH (--check only).
  --skip-validate  Disable VAL plan validation (on by default for optimal,
                   satisficing, anytime; the heuristic track never validates).
  --validate-bin   Override VAL binary discovery; default search order is
                   --validate-bin > VAL env var > "validate" on PATH.

Default --check (no flags): instances=[1], 10 s limit (fast developer check).
--check --full:              instances=[1,2,3,4,5], 60 s limit (CI gate).
--check --instances ...:     custom subset, 60 s limit.

For an exact single-instance rerun: point --domain-dir at the domain folder
and pass --instances <id> (e.g. --domain-dir .../airport --instances 3).
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

JSON_SCHEMA_VERSION = 1

# Add tests dir to path so track modules can import regression_lib.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regression_heuristic import check_heuristics, dev_heuristics, update_heuristics  # noqa: E402
from regression_optimal import check_optimal, dev_optimal, update_optimal  # noqa: E402
from regression_satisficing import check_satisficing, dev_satisficing, update_satisficing  # noqa: E402
from regression_anytime import check_anytime, dev_anytime, update_anytime  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_DIR = TESTS_DIR / "regression-baselines"
DEFAULT_WORKERS = 4


def _find_validate_bin(cli_override):
    """Return the VAL binary path.  Search order: CLI > VAL env > PATH."""
    if cli_override:
        p = Path(cli_override)
        if not p.is_file():
            sys.exit(f"ERROR: --validate-bin {cli_override}: not a file")
        return p
    env_val = os.environ.get("VAL")
    if env_val:
        p = Path(env_val)
        if not p.is_file():
            sys.exit(f"ERROR: VAL env var points to {env_val}: not a file")
        return p
    found = shutil.which("validate")
    return Path(found) if found else None


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


def _parse_instance_id(item):
    """Parse a CLI --instances element as an integer id."""
    try:
        return int(item)
    except ValueError:
        sys.exit(f"ERROR: --instances item {item!r} must be an integer")


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


def _validate_domain_dir(path: Path) -> Path:
    if not path.is_dir():
        sys.exit(f"ERROR: --domain-dir directory not found: {path}")
    return path.resolve()


# ---------------------------------------------------------------------------
# Track registry.
#
# Each entry is (name, check_fn, update_fn, dev_fn).
#   check_fn(domain_dir, baseline_dir, workers, **kwargs) -> dict (comparison)
#   update_fn(domain_dir, baseline_dir, workers, **kwargs) -> list[str]  (errs)
#   dev_fn(domain_dir, baseline_dir, workers, **kwargs)    -> dict (iteration)
# ---------------------------------------------------------------------------
TRACKS = [
    ("heuristics",  check_heuristics,  update_heuristics,  dev_heuristics),
    ("optimal",     check_optimal,     update_optimal,     dev_optimal),
    ("satisficing", check_satisficing, update_satisficing, dev_satisficing),
    ("anytime",     check_anytime,     update_anytime,     dev_anytime),
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
    mode.add_argument(
        "--dev",
        action="store_true",
        help="Dev iteration: run + write a versioned <track>-NNNN.json under "
             "--baseline-dir (required).  Always exits 0; the iteration file "
             "is the structured output for the algorithm-development loop.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--domain-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path to a benchmark set (parent of per-domain dirs) or a "
             "single domain folder; auto-detected by checking for p01.pddl "
             "directly in PATH",
    )
    parser.add_argument(
        "--baseline-dir",
        metavar="PATH",
        type=Path,
        help="Read/write baselines under PATH instead of the default "
             "misc/tests/regression-baselines/.  Required when --dev is set.",
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
        metavar="ID",
        help="Run only on these instance ids; each ID is integer N meaning "
             "p0N.pddl in every domain under --domain-dir",
    )
    track_names = [name for name, *_ in TRACKS]
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
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Disable VAL plan validation (on by default for optimal, "
             "satisficing, and anytime tracks)",
    )
    parser.add_argument(
        "--validate-bin",
        metavar="PATH",
        help="Path to the VAL binary; overrides VAL env var and PATH lookup",
    )
    args = parser.parse_args()

    domain_dir = _validate_domain_dir(args.domain_dir)
    configs_override = _load_config_file(args.config_file)

    if args.dev and args.baseline_dir is None:
        parser.error("--dev requires --baseline-dir PATH")
    baseline_dir = (
        args.baseline_dir.resolve() if args.baseline_dir
        else DEFAULT_BASELINE_DIR
    )

    if args.instances:
        instances_arg = [_parse_instance_id(x) for x in args.instances]
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

    if args.json_output and not args.check:
        parser.error(
            "--json-output is only valid in --check mode; --update writes "
            "the regression-baselines/*.json files and --dev writes the "
            "<track>-NNNN.json iteration files"
        )

    validate = not args.skip_validate
    validate_bin = _find_validate_bin(args.validate_bin) if validate else None
    if validate and validate_bin is None:
        sys.exit(
            "ERROR: VAL plan validator not found.  Install VAL and put "
            "`validate` on PATH, set the VAL environment variable, pass "
            "--validate-bin PATH, or disable validation with --skip-validate."
        )

    if args.check and not baseline_dir.is_dir():
        sys.exit(
            f"ERROR: Baseline directory not found: {baseline_dir}\n"
            "  Run  python misc/tests/generate_baseline.py  to create it."
        )
    if args.update or args.dev:
        baseline_dir.mkdir(parents=True, exist_ok=True)

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

    validate_str = (
        f"on  ({validate_bin})" if validate else "off (--skip-validate)"
    )

    if args.check:
        mode_str = "CHECK"
    elif args.update:
        mode_str = "UPDATE"
    else:
        mode_str = "DEV"

    print(f"Repository root: {REPO_ROOT}")
    print(f"Domain dir:      {domain_dir}")
    print(f"Baselines:       {baseline_dir}")
    print(f"Workers:         {args.workers}")
    print(f"Mode:            {mode_str}  ({scope})")
    print(f"Validation:      {validate_str}")
    print()

    selected = (
        [(n, c, u, d) for n, c, u, d in TRACKS if n in args.track]
        if args.track else TRACKS
    )

    all_failures = []
    track_comparisons = {}
    update_errors = []
    dev_iterations = {}
    any_error = False
    for track_name, check_fn, update_fn, dev_fn in selected:
        print(f"--- {track_name} ---")
        if args.check:
            comp = check_fn(domain_dir, baseline_dir, args.workers,
                            configs=configs_override,
                            instances=instances_arg,
                            validate=validate,
                            validate_bin=validate_bin)
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
        elif args.update:
            errors = update_fn(domain_dir, baseline_dir, args.workers,
                               configs=configs_override,
                               instances=instances_arg,
                               validate=validate,
                               validate_bin=validate_bin) or []
            update_errors.extend(errors)
            if errors:
                print(f"  updated  ({len(errors)} validation error(s); see above)")
            else:
                print("  updated")
        else:
            payload = dev_fn(domain_dir, baseline_dir, args.workers,
                             configs=configs_override,
                             instances=instances_arg,
                             validate=validate,
                             validate_bin=validate_bin)
            dev_iterations[track_name] = payload
            n = payload["iteration_number"]
            run_count = len(payload.get("runs", {}))
            print(f"  iteration {n:04d} written  ({run_count} run(s))")

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
    elif args.update:
        if update_errors:
            print(f"FAILURE: {len(update_errors)} run(s) had invalid plans "
                  "and were not written to the baseline.")
            sys.exit(1)
        print("Baselines updated.")
    else:
        # Dev mode: the iteration file is the structured output.  Always
        # exit 0 — the driver reads the JSON and decides next steps.
        for track_name, payload in dev_iterations.items():
            n = payload["iteration_number"]
            print(f"  {track_name}: iteration {n:04d}")
        print("Dev iteration written.")


if __name__ == "__main__":
    main()
