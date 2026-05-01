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
    load_baseline,
    run_experiment,
    save_baseline,
)

TRACK_NAME = "heuristics"
TIME_LIMIT = 60  # seconds per instance

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


def _run(benchmarks: Path, workers: int) -> dict:
    optimal_strips = benchmarks / "21.11-optimal-strips"
    instances = discover_instances(optimal_strips)
    print(f"  Instances: {len(instances)} | configs: {len(CONFIGS)} | "
          f"time limit: {TIME_LIMIT}s | workers: {workers}")
    return run_experiment(instances, CONFIGS, TIME_LIMIT, workers)


def check_heuristics(benchmarks: Path, baseline_dir: Path, workers: int) -> list[str]:
    print("Running heuristic experiments...")
    current = _run(benchmarks, workers)
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return [f"No baseline found at {baseline_dir}/{TRACK_NAME}.json; "
                "run generate_baseline.py first"]
    return compare_results(current, baseline, EXACT_KEYS, time_limit=TIME_LIMIT)


def update_heuristics(benchmarks: Path, baseline_dir: Path, workers: int) -> None:
    print("Running heuristic experiments (baseline generation)...")
    results = _run(benchmarks, workers)
    save_baseline(baseline_dir, TRACK_NAME, results)
