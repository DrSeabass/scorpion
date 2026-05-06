"""
Heuristic track for the scorpion regression test harness.

Runs A* with five representative heuristics on the optimal-strips p01-p05
instance set, checking initial h-values and node counts for exact-match
regressions and search time for a geo-mean slowdown regression.
"""

from pathlib import Path

from regression_lib import (
    DEFAULT_FULL_INSTANCES,
    DEFAULT_LIGHT_INSTANCES,
    baseline_missing_error,
    build_iteration_payload,
    compare_dev_iterations,
    compare_results,
    filter_baseline,
    is_full_default_instances,
    is_single_domain_layout,
    load_baseline,
    load_previous_iteration,
    resolve_configs,
    resolve_instances,
    run_experiment,
    save_baseline,
    save_iteration,
)

TRACK_NAME = "heuristics"
TIME_LIMIT = 60       # seconds per instance (full mode)
LIGHT_TIME_LIMIT = 10 # seconds per instance (light mode)

# A* with each heuristic; one run per heuristic per instance.
CONFIGS = {
    "astar_lmcut":  ["--search", "astar(lmcut())"],
    "astar_hmax":   ["--search", "astar(hmax())"],
    "astar_ff":     ["--search", "astar(ff())"],
    "astar_add":    ["--search", "astar(add())"],
    "astar_cegar":  ["--search", "astar(cegar([goals()]))"],
}

# Metrics that must match exactly between baseline and current run.
EXACT_KEYS = ["initial_h_value", "expansions", "evaluations", "generated", "cost"]


def _run(domain_dir: Path, workers: int, time_limit: int,
         configs: dict, instances: list) -> tuple[list[dict], dict]:
    discovered = resolve_instances(domain_dir, instances)
    print(f"  Instances: {len(discovered)} | configs: {len(configs)} | "
          f"time limit: {time_limit}s | workers: {workers}")
    # Heuristic track does not validate plans: its primary metric is
    # h-values; the plan is incidental.
    results = run_experiment(discovered, configs, time_limit, workers,
                             validate=False, validate_bin=None)
    return discovered, results


def check_heuristics(domain_dir: Path, baseline_dir: Path, workers: int,
                     *, configs: dict = None,
                     extra_configs: dict = None,
                     instances: list = None,
                     validate: bool = True,
                     validate_bin: Path = None) -> dict:
    print("Running heuristic experiments...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_LIGHT_INSTANCES)
    )
    time_limit = (
        LIGHT_TIME_LIMIT if resolved_instances == DEFAULT_LIGHT_INSTANCES else TIME_LIMIT
    )
    discovered, current = _run(domain_dir, workers, time_limit,
                               resolved_configs, resolved_instances)
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return baseline_missing_error(baseline_dir, TRACK_NAME)
    baseline = filter_baseline(baseline, set(resolved_configs), discovered)
    return compare_results(current, baseline, EXACT_KEYS, time_limit=time_limit)


def update_heuristics(domain_dir: Path, baseline_dir: Path, workers: int,
                      *, configs: dict = None,
                      extra_configs: dict = None,
                      instances: list = None,
                      validate: bool = True,
                      validate_bin: Path = None) -> list[str]:
    print("Running heuristic experiments (baseline generation)...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_FULL_INSTANCES)
    )
    _discovered, results = _run(domain_dir, workers, TIME_LIMIT,
                                resolved_configs, resolved_instances)
    has_override = (
        configs is not None or extra_configs is not None
        or not is_full_default_instances(resolved_instances)
        or is_single_domain_layout(domain_dir)
    )
    save_baseline(baseline_dir, TRACK_NAME, results, merge=has_override)
    return []


def dev_heuristics(domain_dir: Path, baseline_dir: Path, workers: int,
                   *, configs: dict = None,
                   extra_configs: dict = None,
                   instances: list = None,
                   validate: bool = True,
                   validate_bin: Path = None) -> dict:
    """Run one dev iteration: experiments + compare against the previous
    iteration in *baseline_dir* + write `heuristics-NNNN.json`.

    Returns the iteration payload.  On the first iteration there is no
    prior to compare against, so `improved` is unconditionally true for
    every config and the geomean ratios are null.
    """
    print("Running heuristic experiments (dev iteration)...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_LIGHT_INSTANCES)
    )
    time_limit = (
        LIGHT_TIME_LIMIT if resolved_instances == DEFAULT_LIGHT_INSTANCES else TIME_LIMIT
    )
    prev_n, prev_runs = load_previous_iteration(baseline_dir, TRACK_NAME)
    _discovered, results = _run(domain_dir, workers, time_limit,
                                resolved_configs, resolved_instances)
    per_config = compare_dev_iterations(
        results, prev_runs, set(resolved_configs), time_key="search_time"
    )
    payload = build_iteration_payload(
        track_name=TRACK_NAME,
        baseline_dir=baseline_dir,
        domain_dir=domain_dir,
        instances=resolved_instances,
        time_limit=time_limit,
        configs=resolved_configs,
        runs=results,
        previous_iteration_number=prev_n,
        per_config=per_config,
    )
    save_iteration(baseline_dir, TRACK_NAME, payload)
    return payload
