"""
Shared utilities for the scorpion regression test harness.

Handles instance discovery, FD invocation, output parsing,
baseline I/O, and metric comparison.
"""

import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FAST_DOWNWARD = REPO_ROOT / "fast-downward.py"
MEM_LIMIT_MB = 2048
RUNTIME_REGRESSION_THRESHOLD = 2.0  # geo-mean ratio that triggers a failure
ITERATION_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# FD output parsing
# ---------------------------------------------------------------------------

_PATTERNS = [
    ("cost",        r"Plan cost: (.+)\n",             float),
    ("expansions",  r"Expanded (\d+) state\(s\)\.",   int),
    ("evaluations", r"Evaluated (\d+) state\(s\)\.",  int),
    ("generated",   r"Generated (\d+) state\(s\)\.",  int),
    ("search_time", r"Search time: (.+)s",             float),
    ("total_time",  r"Total time: (.+)s",              float),
]


def parse_fd_output(output: str) -> dict:
    """Return a metrics dict from combined FD stdout+stderr."""
    result = {}

    # Capture all plan costs (one per incumbent for anytime search; one for
    # single-solution search).  "cost" is the final/best plan; "incumbent_costs"
    # is the full improving sequence.
    cost_matches = re.findall(r"Plan cost: (.+)", output)
    if cost_matches:
        try:
            result["incumbent_costs"] = [float(v) for v in cost_matches]
            result["cost"] = result["incumbent_costs"][-1]
        except ValueError:
            pass

    for name, pattern, typ in _PATTERNS:
        if name == "cost":
            continue  # already handled above
        m = re.search(pattern, output)
        if m:
            try:
                result[name] = typ(m.group(1))
            except ValueError:
                pass

    result["coverage"] = 1 if "cost" in result else 0

    matches = re.findall(
        r"Initial heuristic value for .+?: ([-]?\d+|infinity)$",
        output, flags=re.M,
    )
    if len(matches) == 1:
        val = matches[0]
        result["initial_h_value"] = sys.maxsize if val == "infinity" else int(val)

    return result


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------

def find_domain_file(problem_path: Path) -> Path:
    """Return the domain PDDL file for a given problem file."""
    shared = problem_path.parent / "domain.pddl"
    if shared.exists():
        return shared
    per_instance = problem_path.parent / f"domain-{problem_path.stem}.pddl"
    if per_instance.exists():
        return per_instance
    raise FileNotFoundError(f"No domain file for {problem_path}")


def is_single_domain_layout(domain_dir: Path) -> bool:
    """True if *domain_dir* points at a single domain (p01.pddl directly
    under it); False if it's a benchmark set (parent of per-domain dirs).
    Same rule resolve_instances uses to decide its iteration shape."""
    return (domain_dir / "p01.pddl").exists()


def resolve_instances(domain_dir: Path, instance_ids: list[int]) -> list[dict]:
    """Resolve integer instance ids to discovered instance dicts.

    *domain_dir* may point at either:
      - a benchmark **set** — a directory containing per-domain
        subdirectories, each holding p01.pddl, p02.pddl, ...; or
      - a single **domain** — a directory containing p01.pddl, p02.pddl,
        ... files directly.

    The two layouts are distinguished by whether p01.pddl exists directly
    in *domain_dir*.  For each id *N* in *instance_ids*, this function
    locates p{N:02d}.pddl across all relevant domains, finds its companion
    domain file via `find_domain_file`, and returns one dict per instance.
    The result is sorted by (domain, problem) for stable iteration.

    Returns dicts with keys: domain, problem (filename), domain_file (path),
    problem_file (path).
    """
    for item in instance_ids:
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(
                f"Instance ids must be integers; got "
                f"{type(item).__name__}: {item!r}"
            )

    if is_single_domain_layout(domain_dir):
        domain_dirs = [domain_dir]
    else:
        domain_dirs = sorted(d for d in domain_dir.iterdir() if d.is_dir())

    seen = set()
    result = []
    for d in domain_dirs:
        domain = d.name
        for n in instance_ids:
            prob = d / f"p{n:02d}.pddl"
            if not prob.exists():
                continue
            key = (domain, prob.name)
            if key in seen:
                continue
            try:
                dom = find_domain_file(prob)
            except FileNotFoundError:
                continue
            seen.add(key)
            result.append({
                "domain": domain,
                "problem": prob.name,
                "domain_file": dom,
                "problem_file": prob,
            })

    result.sort(key=lambda d: (d["domain"], d["problem"]))
    return result


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def _find_final_plan(plan_path_prefix: Path):
    """Return the highest-numbered plan file at the given prefix, or the
    unsuffixed file if it exists, or None if no plan was emitted.

    For optimal/satisficing search FD writes the plan to <prefix>; for
    iterated/anytime search it writes <prefix>.1, <prefix>.2, ... — the
    final (best) plan is the highest-numbered.
    """
    if plan_path_prefix.exists():
        return plan_path_prefix
    parent = plan_path_prefix.parent
    if not parent.is_dir():
        return None
    needle = plan_path_prefix.name + "."
    candidates = []
    for entry in parent.iterdir():
        if entry.name.startswith(needle):
            suffix = entry.name[len(needle):]
            if suffix.isdigit():
                candidates.append((int(suffix), entry))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _validate_plan(validate_bin: Path, domain_file: Path, problem_file: Path,
                   plan_path: Path, timeout: int = 30) -> tuple[bool, str]:
    """Invoke VAL on a plan file.  Return (valid, message).

    VAL exits 0 for a valid plan, non-zero otherwise.
    """
    try:
        r = subprocess.run(
            [str(validate_bin), str(domain_file), str(problem_file), str(plan_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"VAL timed out after {timeout}s"
    except FileNotFoundError:
        return False, f"VAL binary not found: {validate_bin}"
    if r.returncode == 0:
        return True, ""
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return False, f"VAL exit={r.returncode}: {' / '.join(tail)}"


# ---------------------------------------------------------------------------
# Parallel experiment execution
# ---------------------------------------------------------------------------

def _run_one(args):
    """Worker: run one (config, instance) combination and return results."""
    (config_name, config_opts, domain_file, problem_file, time_limit,
     validate, validate_bin) = args

    workdir = Path(tempfile.mkdtemp(prefix="scorpion_reg_"))
    sas_path = workdir / "output.sas"
    plan_prefix = workdir / "plan"
    try:
        cmd = (
            [sys.executable, str(FAST_DOWNWARD),
             "--overall-time-limit", str(time_limit),
             "--overall-memory-limit", f"{MEM_LIMIT_MB}M",
             "--sas-file", str(sas_path),
             "--plan-file", str(plan_prefix),
             str(domain_file), str(problem_file)]
            + config_opts
        )
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=time_limit + 30,
            )
            wall = time.monotonic() - t0
            metrics = parse_fd_output(proc.stdout + proc.stderr)
            metrics["returncode"] = proc.returncode
            metrics["wall_time"] = wall
        except subprocess.TimeoutExpired:
            metrics = {"coverage": 0, "returncode": -1,
                       "wall_time": time.monotonic() - t0}

        # Validate emitted plan if requested.  plan_valid is None unless we
        # actually run VAL; coverage=0 runs and validate=False both leave
        # plan_valid unset.
        if validate and validate_bin and metrics.get("coverage", 0) == 1:
            plan_path = _find_final_plan(plan_prefix)
            if plan_path is None:
                metrics["plan_valid"] = False
                metrics["plan_validation_message"] = (
                    "coverage=1 but no plan file found at "
                    f"{plan_prefix}(.N) — FD output mismatch"
                )
            else:
                valid, msg = _validate_plan(
                    Path(validate_bin), domain_file, problem_file, plan_path
                )
                metrics["plan_valid"] = valid
                if msg:
                    metrics["plan_validation_message"] = msg
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return config_name, domain_file.parent.name, Path(problem_file).name, metrics


def run_experiment(
    instances: list[dict],
    configs: dict,
    time_limit: int,
    workers: int,
    *,
    validate: bool = False,
    validate_bin: Path = None,
) -> dict:
    """Run all configs × instances in parallel; return a results dict.

    Result keys are  "config|domain|problem"  strings.

    When validate=True (and validate_bin is set), each successful run's
    final plan is checked with VAL; the result is recorded as
    `plan_valid: True/False` in the run's metrics.  `plan_valid` is
    absent for runs that did not solve, or when validate=False.
    """
    tasks = [
        (cfg_name, cfg_opts,
         inst["domain_file"], inst["problem_file"], time_limit,
         validate, str(validate_bin) if validate_bin else None)
        for cfg_name, cfg_opts in configs.items()
        for inst in instances
    ]

    results = {}
    total = len(tasks)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            cfg_name, domain, problem, metrics = fut.result()
            run_id = f"{cfg_name}|{domain}|{problem}"
            results[run_id] = metrics
            done += 1
            cov = metrics.get("coverage", 0)
            pv = metrics.get("plan_valid")
            if not cov:
                status = "---"
            elif pv is False:
                status = "BAD"
            elif pv is True:
                status = "OK✓"
            else:
                status = "OK"
            print(f"  [{done:4d}/{total}] {status}  {run_id}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Baseline filtering
# ---------------------------------------------------------------------------

def filter_baseline(baseline: dict, configs: set,
                    instances: list[dict]) -> dict:
    """Return only entries whose config is in *configs* and whose
    `(domain, problem)` pair is in *instances*.

    Keys have the form "config|domain|instance.pddl".  *instances* is a
    list of resolved-instance dicts (as produced by `resolve_instances`)
    each with 'domain' and 'problem' keys.  Matching is exact on the
    pair, so callers that ran against a narrow `domain_dir` (e.g. a
    single domain folder) get back a baseline slice covering only the
    domains they actually ran — no false coverage_loss reports for the
    un-run ones.
    """
    pair_set = {(inst["domain"], inst["problem"]) for inst in instances}
    result = {}
    for key, value in baseline.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        config, domain, problem = parts
        if config in configs and (domain, problem) in pair_set:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def baseline_path(baseline_dir: Path, track_name: str) -> Path:
    return baseline_dir / f"{track_name}.json"


# ---------------------------------------------------------------------------
# Dev-mode iteration files
#
# In `--dev` mode the harness writes one versioned iteration file per
# track per invocation: `<baseline_dir>/<track>-NNNN.json`.  The 4-digit
# zero-padded suffix orders iterations; "latest" = highest NNNN.  These
# helpers handle naming, scanning, and writing.
# ---------------------------------------------------------------------------

ITERATION_SUFFIX_RE = re.compile(r"-(\d{4})\.json$")


def iteration_path(baseline_dir: Path, track_name: str, n: int) -> Path:
    """Return the path to iteration *n* of *track_name* in *baseline_dir*."""
    return baseline_dir / f"{track_name}-{n:04d}.json"


def latest_iteration_number(baseline_dir: Path, track_name: str) -> int:
    """Return the highest existing iteration number for *track_name*, or 0
    if no iteration file exists yet."""
    if not baseline_dir.is_dir():
        return 0
    prefix = f"{track_name}-"
    highest = 0
    for entry in baseline_dir.iterdir():
        if not entry.is_file() or not entry.name.startswith(prefix):
            continue
        m = ITERATION_SUFFIX_RE.search(entry.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest


def worktree_dirty_flag(repo_root: Path) -> bool:
    """Return True if the scorpion worktree at *repo_root* has any
    uncommitted changes.  Returns False if the directory is not a git
    checkout or git is unavailable — the recorded `scorpion_commit` is
    None in that case anyway, and the driver can detect the dirty/no-git
    situation itself.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root), capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(r.stdout.strip())


def load_previous_iteration(baseline_dir: Path,
                            track_name: str) -> tuple[int | None, dict]:
    """Return `(previous_iteration_number, previous_runs_dict)` for the
    latest iteration file in *baseline_dir*, or `(None, {})` if none
    exists.

    The driver compares the *runs* dict by config|domain|problem keys
    against the current run's results.  Other fields (per_config,
    metadata) from the previous iteration are not used for
    comparison and are not returned.
    """
    n = latest_iteration_number(baseline_dir, track_name)
    if n == 0:
        return None, {}
    path = iteration_path(baseline_dir, track_name, n)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, {}
    return n, data.get("runs", {})


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def compare_dev_iterations(current_runs: dict, previous_runs: dict,
                           configs, *,
                           time_key: str = "search_time") -> dict:
    """Compute per-config direction-aware aggregates comparing
    *current_runs* (this iteration) to *previous_runs* (the prior
    iteration).

    Returns a dict keyed by config name.  Each entry has:
      - `improved`: bool — true iff `expansions`, `generated`, and
        `<time_key>` all decreased on the geomean over instances solved
        in both runs.  On the first iteration (no prior data),
        unconditionally true ("extant > nothing").
      - `expansions_geomean_ratio`, `generated_geomean_ratio`,
        `time_geomean_ratio`, `cost_geomean_ratio`: float | null.
        Geomean of `current_value / previous_value` over instances
        solved (`coverage == 1`) by the same config in both runs.
        `null` when no shared solved instances exist for that metric.
      - `cost_changed`: bool — any shared solved instance has a
        different `cost` in the two runs.  Reported separately because
        cost direction depends on algorithm class.
      - `coverage`: int — number of instances this config solved in
        the current run.
      - `coverage_delta`: int — `current - previous` solved counts
        (= `coverage` on the first iteration, since previous = 0).
      - `compared_count`: int — instances counted in the geomean
        ratios for the time/expansion/generated metrics; 0 means we
        had no usable shared solves.

    *time_key* selects the runtime metric to use for `time_geomean_ratio`
    and the `improved` flag.  Heuristic / optimal / satisficing tracks
    pass `"search_time"`; the anytime track passes `"wall_time"` because
    iterated search does not emit a `Search time:` line.
    """
    is_first = not previous_runs
    per_config = {}

    for config in sorted(configs):
        cur_for_config = {
            k: v for k, v in current_runs.items()
            if k.split("|", 1)[0] == config
        }
        prev_for_config = {
            k: v for k, v in previous_runs.items()
            if k.split("|", 1)[0] == config
        }

        cur_solved = sum(
            1 for v in cur_for_config.values() if v.get("coverage") == 1
        )
        prev_solved = sum(
            1 for v in prev_for_config.values() if v.get("coverage") == 1
        )
        coverage_delta = cur_solved if is_first else cur_solved - prev_solved

        ratios = {
            "expansions": [],
            "generated": [],
            time_key: [],
            "cost": [],
        }
        cost_changed = False
        compared = 0
        for run_id, cur in cur_for_config.items():
            prev = prev_for_config.get(run_id, {})
            if cur.get("coverage") != 1 or prev.get("coverage") != 1:
                continue
            compared += 1
            for key in ratios:
                cv = cur.get(key)
                pv = prev.get(key)
                if (
                    cv is not None and pv is not None
                    and isinstance(cv, (int, float))
                    and isinstance(pv, (int, float))
                    and pv > 0 and cv > 0
                ):
                    ratios[key].append(cv / pv)
            if (
                cur.get("cost") is not None and prev.get("cost") is not None
                and cur.get("cost") != prev.get("cost")
            ):
                cost_changed = True

        e_ratio = _geomean(ratios["expansions"])
        g_ratio = _geomean(ratios["generated"])
        t_ratio = _geomean(ratios[time_key])
        c_ratio = _geomean(ratios["cost"])

        if is_first:
            improved = True
        else:
            improved = (
                e_ratio is not None and e_ratio < 1.0
                and g_ratio is not None and g_ratio < 1.0
                and t_ratio is not None and t_ratio < 1.0
            )

        per_config[config] = {
            "improved": improved,
            "expansions_geomean_ratio": e_ratio,
            "generated_geomean_ratio": g_ratio,
            "time_geomean_ratio": t_ratio,
            "cost_geomean_ratio": c_ratio,
            "cost_changed": cost_changed,
            "coverage": cur_solved,
            "coverage_delta": coverage_delta,
            "compared_count": compared,
        }

    return per_config


def build_iteration_payload(*, track_name: str, baseline_dir: Path,
                            domain_dir: Path, instances: list,
                            time_limit: int, configs: dict,
                            runs: dict,
                            previous_iteration_number: int | None = None,
                            per_config: dict | None = None) -> dict:
    """Assemble a `<track>-NNNN.json` payload.

    Picks the next iteration number from the existing files in
    *baseline_dir*, captures the scorpion commit + worktree-dirty flag,
    and returns a dict ready for `save_iteration`.

    *previous_iteration_number* and *per_config* are produced by
    `compare_dev_iterations`; pass `None` and `{}` for a first
    iteration (or when the caller chooses to skip comparison).
    """
    n = latest_iteration_number(baseline_dir, track_name) + 1
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    commit = None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        )
        commit = r.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return {
        "schema_version": ITERATION_SCHEMA_VERSION,
        "track": track_name,
        "iteration_number": n,
        "previous_iteration_number": previous_iteration_number,
        "scorpion_commit": commit,
        "worktree_dirty": worktree_dirty_flag(REPO_ROOT),
        "started_at": started_at,
        "domain_dir": str(domain_dir),
        "instances": list(instances),
        "time_limit": time_limit,
        "configs": dict(configs),
        "runs": runs,
        "per_config": dict(per_config) if per_config else {},
    }


def save_iteration(baseline_dir: Path, track_name: str, payload: dict) -> Path:
    """Write *payload* as the next iteration file for *track_name*.

    *payload* must already contain `iteration_number` matching the file
    suffix this function picks.  The caller computes that number (so it
    can be threaded into the payload before any per-iteration
    bookkeeping like comparison vs the previous iteration); this helper
    only handles I/O.

    Creates *baseline_dir* if missing.  Returns the written path.
    """
    baseline_dir.mkdir(parents=True, exist_ok=True)
    n = payload["iteration_number"]
    path = iteration_path(baseline_dir, track_name, n)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"  Saved iteration: {path}")
    return path


def drop_invalid_runs(results: dict) -> list[str]:
    """Remove runs whose plan_valid is False from *results* in place.

    Returns a list of human-readable error messages, one per dropped run,
    suitable for surfacing to the user.  The dropped entries must not be
    written to a baseline (per the design summary's plan-validation step).
    """
    errors = []
    for run_id in [r for r, m in results.items() if m.get("plan_valid") is False]:
        msg = (
            f"INVALID PLAN: {run_id} — not writing this entry to baseline"
            f" ({results[run_id].get('plan_validation_message', 'VAL rejected the plan')})"
        )
        print(f"  ERROR: {msg}", flush=True)
        errors.append(msg)
        del results[run_id]
    return errors


def baseline_missing_error(baseline_dir: Path, track_name: str) -> dict:
    """Return an `outcome="error"` comparison dict for a missing baseline file."""
    return {
        "outcome": "error",
        "per_config": {},
        "runs": {},
        "messages": [
            f"No baseline found at {baseline_dir}/{track_name}.json; "
            "run generate_baseline.py first"
        ],
    }


def load_baseline(baseline_dir: Path, track_name: str) -> dict:
    path = baseline_path(baseline_dir, track_name)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_baseline(baseline_dir: Path, track_name: str, results: dict,
                  *, merge: bool = False) -> None:
    """Write the baseline.  When merge=True, overlay results on the existing
    file instead of overwriting it; existing keys not in results are preserved."""
    path = baseline_path(baseline_dir, track_name)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    if merge and path.exists():
        with open(path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"  Saved baseline: {path}")


DEFAULT_LIGHT_INSTANCES = [1]
DEFAULT_FULL_INSTANCES = [1, 2, 3, 4, 5]


def is_full_default_instances(instances: list) -> bool:
    """Return True iff *instances* is a permutation of [1, 2, 3, 4, 5].

    Used by `update_*` to decide whether to overwrite the baseline (rebaseline
    default) or to merge new entries into it (a partial subset override).
    """
    return (
        len(instances) == len(DEFAULT_FULL_INSTANCES)
        and all(isinstance(i, int) and not isinstance(i, bool) for i in instances)
        and set(instances) == set(DEFAULT_FULL_INSTANCES)
    )


def resolve_configs(default: dict, configs=None, extra_configs=None) -> dict:
    """Return the effective configs dict given optional overrides.

    Both None  → copy of default.
    configs=X  → X (full replacement).
    extra_configs=Y → default with Y merged in (Y wins on key collisions).
    Both set   → ValueError.
    """
    if configs is not None and extra_configs is not None:
        raise ValueError("configs= and extra_configs= are mutually exclusive")
    if configs is not None:
        return dict(configs)
    if extra_configs is not None:
        return {**default, **extra_configs}
    return dict(default)


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------

def compare_results(
    current: dict,
    baseline: dict,
    exact_keys: list[str],
    runtime_key: str = "search_time",
    time_limit: int = None,
    prefix_keys: list[str] = None,
) -> dict:
    """Return a structured comparison dict.

    Shape:
        {
          "outcome": "pass" | "fail",
          "per_config": {<config>: {
              "runtime_geomean_ratio": <float | None>,
              "runtime_regression":    <bool>,
              "runtime_compared_count": <int>,
          }},
          "runs": {<run_id>: {
              "outcome": "pass" | "fail",
              "failure_kind": "exact_match" | "coverage_loss" | None,
              "metrics": {<metric>: {
                  "baseline": <v>, "current": <v>, "match": <bool>
              }},
          }},
          "messages": [<human-readable failure description>, ...],
        }

    `runs` contains one entry per baseline run_id (full dump, not just
    failures).  `messages` is a CLI-stdout convenience extension over
    the documented schema; JSON consumers should iterate `runs` and
    `per_config` instead.

    Coverage loss is only a hard failure when the baseline solved
    comfortably within the time limit (< 50% of time_limit).  Instances
    that solved close to the wall-clock limit are inherently
    timing-sensitive and are skipped for coverage comparison.

    Keys listed in prefix_keys are list-valued and compared only on the
    shared prefix (up to min length).  A shorter sequence is not a
    failure provided the shared values match.

    Runtime geo-mean is computed per config; any config exceeding the
    threshold is flagged in `per_config[name].runtime_regression`.
    """
    runs = {}
    messages = []
    prefix_keys = set(prefix_keys or [])
    coverage_stable_threshold = (0.5 * time_limit) if time_limit else None

    for run_id, base in baseline.items():
        curr = current.get(run_id, {})
        run_entry = {
            "outcome": "pass",
            "failure_kind": None,
            "plan_valid": curr.get("plan_valid"),
            "metrics": {},
        }

        base_cov = base.get("coverage", 0)
        curr_cov = curr.get("coverage", 0)

        if base_cov == 1 and curr_cov == 0:
            base_time = base.get("wall_time") or base.get(runtime_key, 0)
            is_borderline = (
                coverage_stable_threshold is not None
                and base_time > coverage_stable_threshold
            )
            if not is_borderline:
                run_entry["outcome"] = "fail"
                run_entry["failure_kind"] = "coverage_loss"
                if time_limit:
                    messages.append(
                        f"COVERAGE LOSS: {run_id} "
                        f"(baseline {base_time:.1f}s, limit {time_limit}s)"
                    )
                else:
                    messages.append(f"COVERAGE LOSS: {run_id}")
            runs[run_id] = run_entry
            continue

        if base_cov == 1 and curr_cov == 1:
            for key in exact_keys:
                bv = base.get(key)
                cv = curr.get(key)
                if bv is None or cv is None:
                    continue
                if key in prefix_keys:
                    n = min(len(bv), len(cv))
                    match = bv[:n] == cv[:n]
                else:
                    match = bv == cv
                run_entry["metrics"][key] = {
                    "baseline": bv, "current": cv, "match": match
                }
                if not match:
                    if run_entry["outcome"] == "pass":
                        run_entry["outcome"] = "fail"
                        run_entry["failure_kind"] = "exact_match"
                    messages.append(
                        f"MISMATCH {key}: {run_id} "
                        f"baseline={bv} current={cv}"
                    )

        # Invalid plan trumps exact_match (more severe).
        if curr.get("plan_valid") is False:
            run_entry["outcome"] = "fail"
            run_entry["failure_kind"] = "invalid_plan"
            val_msg = curr.get("plan_validation_message", "")
            messages.append(
                f"INVALID PLAN: {run_id}"
                + (f" — {val_msg}" if val_msg else "")
            )

        runs[run_id] = run_entry

    # Per-config runtime geo-mean.
    config_ratios = defaultdict(list)
    for run_id, base in baseline.items():
        config = run_id.split("|", 1)[0]
        curr = current.get(run_id, {})
        bt = base.get(runtime_key)
        ct = curr.get(runtime_key)
        if bt and ct and bt > 0 and ct > 0:
            config_ratios[config].append(ct / bt)

    per_config = {}
    all_configs = {run_id.split("|", 1)[0] for run_id in baseline}
    for config in sorted(all_configs):
        ratios = config_ratios.get(config, [])
        if ratios:
            geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            regression = geomean >= RUNTIME_REGRESSION_THRESHOLD
        else:
            geomean = None
            regression = False
        per_config[config] = {
            "runtime_geomean_ratio": geomean,
            "runtime_regression": regression,
            "runtime_compared_count": len(ratios),
        }
        if regression:
            messages.append(
                f"RUNTIME: {config} geo-mean ratio {geomean:.2f}x "
                f">= {RUNTIME_REGRESSION_THRESHOLD}x threshold "
                f"(over {len(ratios)} instances)"
            )

    any_fail = (
        any(r["outcome"] == "fail" for r in runs.values())
        or any(c["runtime_regression"] for c in per_config.values())
    )

    return {
        "outcome": "fail" if any_fail else "pass",
        "per_config": per_config,
        "runs": runs,
        "messages": messages,
    }
