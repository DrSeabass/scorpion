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
    compare_results,
    filter_baseline,
    is_full_default_instances,
    load_baseline,
    resolve_configs,
    resolve_instances,
    run_experiment,
    save_baseline,
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


def _run(benchmarks: Path, workers: int, time_limit: int,
         configs: dict, instances: list) -> dict:
    optimal_strips = benchmarks / "21.11-optimal-strips"
    discovered = resolve_instances(optimal_strips, instances)
    print(f"  Instances: {len(discovered)} | configs: {len(configs)} | "
          f"time limit: {time_limit}s | workers: {workers}")
    return run_experiment(discovered, configs, time_limit, workers)


def check_heuristics(benchmarks: Path, baseline_dir: Path, workers: int,
                     *, configs: dict = None,
                     extra_configs: dict = None,
                     instances: list = None) -> dict:
    print("Running heuristic experiments...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_LIGHT_INSTANCES)
    )
    time_limit = (
        LIGHT_TIME_LIMIT if resolved_instances == DEFAULT_LIGHT_INSTANCES else TIME_LIMIT
    )
    current = _run(benchmarks, workers, time_limit, resolved_configs, resolved_instances)
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return baseline_missing_error(baseline_dir, TRACK_NAME)
    baseline = filter_baseline(baseline, set(resolved_configs), resolved_instances)
    return compare_results(current, baseline, EXACT_KEYS, time_limit=time_limit)


def update_heuristics(benchmarks: Path, baseline_dir: Path, workers: int,
                      *, configs: dict = None,
                      extra_configs: dict = None,
                      instances: list = None) -> None:
    print("Running heuristic experiments (baseline generation)...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_FULL_INSTANCES)
    )
    results = _run(benchmarks, workers, TIME_LIMIT, resolved_configs, resolved_instances)
    has_override = (
        configs is not None or extra_configs is not None
        or not is_full_default_instances(resolved_instances)
    )
    save_baseline(baseline_dir, TRACK_NAME, results, merge=has_override)
