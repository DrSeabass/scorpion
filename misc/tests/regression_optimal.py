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
TIME_LIMIT = 60       # seconds per instance (full mode)
LIGHT_TIME_LIMIT = 10 # seconds per instance (light mode)

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

# Light: one classical admissible + one scorpion-specific config, p01 only, 10 s.
LIGHT_CONFIGS = {
    "astar_lmcut":  CONFIGS["astar_lmcut"],
    "astar_cegar":  CONFIGS["astar_cegar"],
}

# Metrics that must match exactly between baseline and current run.
EXACT_KEYS = ["cost", "expansions", "evaluations", "generated"]


def _run(benchmarks: Path, workers: int, light: bool) -> dict:
    optimal_strips = benchmarks / "21.11-optimal-strips"
    configs = LIGHT_CONFIGS if light else CONFIGS
    time_limit = LIGHT_TIME_LIMIT if light else TIME_LIMIT
    max_inst = 1 if light else 5
    instances = discover_instances(optimal_strips, max_instance=max_inst)
    print(f"  Instances: {len(instances)} | configs: {len(configs)} | "
          f"time limit: {time_limit}s | workers: {workers}")
    return run_experiment(instances, configs, time_limit, workers)


def check_optimal(benchmarks: Path, baseline_dir: Path, workers: int,
                  *, light: bool = True) -> list[str]:
    print("Running optimal search experiments...")
    current = _run(benchmarks, workers, light)
    track = f"{TRACK_NAME}-light" if light else TRACK_NAME
    time_limit = LIGHT_TIME_LIMIT if light else TIME_LIMIT
    baseline = load_baseline(baseline_dir, track)
    if not baseline:
        return [f"No baseline found at {baseline_dir}/{track}.json; "
                "run generate_baseline.py first"]
    return compare_results(current, baseline, EXACT_KEYS, time_limit=time_limit)


def update_optimal(benchmarks: Path, baseline_dir: Path, workers: int,
                   *, light: bool = True) -> None:
    print("Running optimal search experiments (baseline generation)...")
    results = _run(benchmarks, workers, light)
    track = f"{TRACK_NAME}-light" if light else TRACK_NAME
    save_baseline(baseline_dir, track, results)
