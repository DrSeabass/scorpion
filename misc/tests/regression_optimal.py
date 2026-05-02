"""
Optimal track for the scorpion regression test harness.

Runs six representative optimal configurations on the optimal-strips p01-p05
instance set, checking plan cost and node counts for exact-match regressions
and search time for a geo-mean slowdown regression.

Config selection rationale:
  astar_blind         — blind-search baseline; confirms basic graph exploration
  astar_lmcut         — best classical admissible heuristic
  bjolp               — landmark cost partitioning with lazy evaluation
  astar_merge_and_shrink_rl_fh — M&S (state-count bounded; deterministic)
  blind-sss-simple    — blind + stubborn-sets pruning (orthogonal to heuristic quality)
  astar_cegar         — CEGAR abstractions (key scorpion-specific algorithm)

Excluded: ipdb/scp_single_order (hillclimbing uses wall-clock budget → ordering
risk), LP configs (require CPLEX), near-duplicate M&S variants.
"""

from pathlib import Path

from regression_lib import (
    compare_results,
    discover_instances,
    load_baseline,
    run_experiment,
    save_baseline,
)

TRACK_NAME = "optimal"
TIME_LIMIT = 60  # seconds per instance

CONFIGS = {
    "astar_blind": [
        "--search", "astar(blind())"],
    "astar_lmcut": [
        "--search", "astar(lmcut())"],
    "bjolp": [
        "--evaluator",
        "lmc=landmark_cost_partitioning(lm_merged([lm_rhw(),lm_hm(m=1)]))",
        "--search", "astar(lmc,lazy_evaluator=lmc)"],
    "astar_merge_and_shrink_rl_fh": [
        "--search",
        "astar(merge_and_shrink("
        "merge_strategy=merge_precomputed("
        "merge_tree=linear(variable_order=reverse_level)),"
        "shrink_strategy=shrink_fh(),"
        "label_reduction=exact(before_shrinking=false,"
        "before_merging=true),max_states=50000,verbosity=silent))"],
    "blind-sss-simple": [
        "--search", "astar(blind(), pruning=stubborn_sets_simple())"],
    "astar_cegar": [
        "--search", "astar(cegar([landmarks(), goals()]))"],
}

# Metrics that must match exactly between baseline and current run.
EXACT_KEYS = ["cost", "expansions", "evaluations", "generated"]


def _run(benchmarks: Path, workers: int) -> dict:
    optimal_strips = benchmarks / "21.11-optimal-strips"
    instances = discover_instances(optimal_strips)
    print(f"  Instances: {len(instances)} | configs: {len(CONFIGS)} | "
          f"time limit: {TIME_LIMIT}s | workers: {workers}")
    return run_experiment(instances, CONFIGS, TIME_LIMIT, workers)


def check_optimal(benchmarks: Path, baseline_dir: Path, workers: int) -> list[str]:
    print("Running optimal search experiments...")
    current = _run(benchmarks, workers)
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return [f"No baseline found at {baseline_dir}/{TRACK_NAME}.json; "
                "run generate_baseline.py first"]
    return compare_results(current, baseline, EXACT_KEYS, time_limit=TIME_LIMIT)


def update_optimal(benchmarks: Path, baseline_dir: Path, workers: int) -> None:
    print("Running optimal search experiments (baseline generation)...")
    results = _run(benchmarks, workers)
    save_baseline(baseline_dir, TRACK_NAME, results)
