"""
Shared utilities for the scorpion regression test harness.

Handles instance discovery, FD invocation, output parsing,
baseline I/O, and metric comparison.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FAST_DOWNWARD = REPO_ROOT / "fast-downward.py"
MEM_LIMIT_MB = 2048
RUNTIME_REGRESSION_THRESHOLD = 2.0  # geo-mean ratio that triggers a failure


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


def discover_instances(benchmark_subdir: Path, max_instance: int = 5) -> list[dict]:
    """Return a sorted list of instance dicts for p01..p<max_instance>.

    Each dict has keys: domain, problem (filename), domain_file (path),
    problem_file (path).
    """
    instances = []
    for domain_dir in sorted(benchmark_subdir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        for i in range(1, max_instance + 1):
            prob = domain_dir / f"p{i:02d}.pddl"
            if not prob.exists():
                continue
            try:
                dom = find_domain_file(prob)
            except FileNotFoundError:
                continue
            instances.append({
                "domain": domain,
                "problem": prob.name,
                "domain_file": dom,
                "problem_file": prob,
            })
    return instances


# ---------------------------------------------------------------------------
# Parallel experiment execution
# ---------------------------------------------------------------------------

def _run_one(args):
    """Worker: run one (config, instance) combination and return results."""
    config_name, config_opts, domain_file, problem_file, time_limit = args

    sas_fd, sas_path = tempfile.mkstemp(suffix=".sas")
    os.close(sas_fd)
    try:
        cmd = (
            [sys.executable, str(FAST_DOWNWARD),
             "--overall-time-limit", str(time_limit),
             "--overall-memory-limit", f"{MEM_LIMIT_MB}M",
             "--sas-file", sas_path,
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
    finally:
        Path(sas_path).unlink(missing_ok=True)

    return config_name, domain_file.parent.name, Path(problem_file).name, metrics


def run_experiment(
    instances: list[dict],
    configs: dict,
    time_limit: int,
    workers: int,
) -> dict:
    """Run all configs × instances in parallel; return a results dict.

    Result keys are  "config|domain|problem"  strings.
    """
    tasks = [
        (cfg_name, cfg_opts,
         inst["domain_file"], inst["problem_file"], time_limit)
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
            status = "OK" if cov else "---"
            print(f"  [{done:4d}/{total}] {status}  {run_id}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Baseline filtering
# ---------------------------------------------------------------------------

def filter_baseline(baseline: dict, configs: set, max_instance: int) -> dict:
    """Return only entries whose config is in *configs* and instance number ≤ max_instance.

    Keys have the form  "config|domain|instance.pddl"  where instance is e.g. "p01.pddl".
    """
    result = {}
    for key, value in baseline.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        config, _domain, instance = parts
        stem = Path(instance).stem          # "p01", "p05", ...
        if not (stem.startswith("p") and stem[1:].isdigit()):
            continue
        if config in configs and int(stem[1:]) <= max_instance:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def baseline_path(baseline_dir: Path, track_name: str) -> Path:
    return baseline_dir / f"{track_name}.json"


def load_baseline(baseline_dir: Path, track_name: str) -> dict:
    path = baseline_path(baseline_dir, track_name)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_baseline(baseline_dir: Path, track_name: str, results: dict) -> None:
    path = baseline_path(baseline_dir, track_name)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"  Saved baseline: {path}")


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
) -> list[str]:
    """Return a list of failure description strings (empty = pass).

    Coverage loss is only a hard failure when the baseline solved
    comfortably within the time limit (< 50% of time_limit).  Instances
    that solved close to the wall-clock limit are inherently timing-sensitive
    and are skipped for coverage comparison.

    Keys listed in prefix_keys are list-valued and compared only on the shared
    prefix (up to min length).  A shorter sequence is not a failure provided
    the shared values match — the sequence length varies with wall-clock luck,
    but the order and values are deterministic.
    """
    failures = []
    prefix_keys = set(prefix_keys or [])
    coverage_stable_threshold = (0.5 * time_limit) if time_limit else None

    for run_id, base in baseline.items():
        curr = current.get(run_id)
        if curr is None:
            failures.append(f"MISSING run: {run_id}")
            continue

        base_cov = base.get("coverage", 0)
        curr_cov = curr.get("coverage", 0)
        if base_cov == 1 and curr_cov == 0:
            # Use wall_time for the borderline check: search_time alone
            # understates elapsed time for configs with expensive precomputation
            # (e.g., merge-and-shrink).  Fall back to runtime_key if absent.
            base_time = base.get("wall_time") or base.get(runtime_key, 0)
            if coverage_stable_threshold and base_time > coverage_stable_threshold:
                # Borderline instance: solved close to the wall-clock limit.
                # Skip — coverage is timing-sensitive here.
                continue
            failures.append(
                f"COVERAGE LOSS: {run_id} "
                f"(baseline {base_time:.1f}s, limit {time_limit}s)"
            )
            continue

        if base_cov == 1 and curr_cov == 1:
            for key in exact_keys:
                bv = base.get(key)
                cv = curr.get(key)
                if bv is None or cv is None:
                    continue
                if key in prefix_keys:
                    n = min(len(bv), len(cv))
                    if bv[:n] != cv[:n]:
                        failures.append(
                            f"MISMATCH {key}: {run_id} "
                            f"baseline={bv} current={cv}"
                        )
                elif bv != cv:
                    failures.append(
                        f"MISMATCH {key}: {run_id} "
                        f"baseline={bv} current={cv}"
                    )

    # Runtime geo-mean check (only over commonly solved instances).
    ratios = []
    for run_id, base in baseline.items():
        curr = current.get(run_id, {})
        bt = base.get(runtime_key)
        ct = curr.get(runtime_key)
        if bt and ct and bt > 0 and ct > 0:
            ratios.append(ct / bt)
    if ratios:
        geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        if geomean >= RUNTIME_REGRESSION_THRESHOLD:
            failures.append(
                f"RUNTIME: geo-mean ratio {geomean:.2f}x "
                f">= {RUNTIME_REGRESSION_THRESHOLD}x threshold "
                f"(over {len(ratios)} instances)"
            )

    return failures
