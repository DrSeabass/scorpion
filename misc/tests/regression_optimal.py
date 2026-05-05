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

# Metrics that must match exactly between baseline and current run.
EXACT_KEYS = ["cost", "expansions", "evaluations", "generated"]


def _run(benchmarks: Path, workers: int, time_limit: int,
         configs: dict, instances: list) -> dict:
    optimal_strips = benchmarks / "21.11-optimal-strips"
    discovered = resolve_instances(optimal_strips, instances)
    print(f"  Instances: {len(discovered)} | configs: {len(configs)} | "
          f"time limit: {time_limit}s | workers: {workers}")
    return run_experiment(discovered, configs, time_limit, workers)


def check_optimal(benchmarks: Path, baseline_dir: Path, workers: int,
                  *, configs: dict = None,
                  extra_configs: dict = None,
                  instances: list = None) -> dict:
    print("Running optimal search experiments...")
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


def update_optimal(benchmarks: Path, baseline_dir: Path, workers: int,
                   *, configs: dict = None,
                   extra_configs: dict = None,
                   instances: list = None) -> None:
    print("Running optimal search experiments (baseline generation)...")
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
