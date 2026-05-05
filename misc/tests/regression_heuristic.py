"""
Heuristic track for the scorpion regression test harness.

Runs A* with five representative heuristics on the optimal-strips p01-p05
instance set, checking initial h-values and node counts for exact-match
regressions and search time for a geo-mean slowdown regression.
"""

from pathlib import Path

from regression_lib import (
    compare_results,
    discover_instances,
    filter_baseline,
    load_baseline,
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


def _run(benchmarks: Path, workers: int, light: bool) -> dict:
    optimal_strips = benchmarks / "21.11-optimal-strips"
    time_limit = LIGHT_TIME_LIMIT if light else TIME_LIMIT
    max_inst = 1 if light else 5
    instances = discover_instances(optimal_strips, max_instance=max_inst)
    print(f"  Instances: {len(instances)} | configs: {len(CONFIGS)} | "
          f"time limit: {time_limit}s | workers: {workers}")
    return run_experiment(instances, CONFIGS, time_limit, workers)


def check_heuristics(benchmarks: Path, baseline_dir: Path, workers: int,
                     *, light: bool = True) -> list[str]:
    print("Running heuristic experiments...")
    current = _run(benchmarks, workers, light)
    time_limit = LIGHT_TIME_LIMIT if light else TIME_LIMIT
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return [f"No baseline found at {baseline_dir}/{TRACK_NAME}.json; "
                "run generate_baseline.py first"]
    if light:
        baseline = filter_baseline(baseline, set(CONFIGS), 1)
    return compare_results(current, baseline, EXACT_KEYS, time_limit=time_limit)


def update_heuristics(benchmarks: Path, baseline_dir: Path, workers: int,
                      *, light: bool = True) -> None:
    if light:
        raise ValueError("update_heuristics must not be called in light mode")
    print("Running heuristic experiments (baseline generation)...")
    results = _run(benchmarks, workers, light=False)
    save_baseline(baseline_dir, TRACK_NAME, results)
