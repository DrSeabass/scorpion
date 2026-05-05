"""
Satisficing track for the scorpion regression test harness.

Runs four representative satisficing configurations on the agile-strips p01-p05
instance set, checking plan cost and node counts for exact-match regressions
and search time for a geo-mean slowdown regression.

Config selection rationale:
  eager_greedy_ff   — standard eager greedy with FF; most widely used satisficing config
  lazy_greedy_ff    — lazy evaluation strategy with FF; different from eager_greedy_ff
  eager_greedy_add  — eager greedy with additive heuristic; orthogonal to FF-based search
  lama-first        — landmark-based LAMA; tests the landmark factory end-to-end

Excluded: lama-first-typed (type_based open list + randomize_successors adds noise risk),
eager/lazy_greedy_cea and _cg (covered by broader family; FF/add sufficient for diversity).
All selected configs are deterministic under scorpion's fixed RNG (seed 2011).

Uses 21.11-agile-strips because those instances are designed for satisficing planners
and produce substantially higher coverage than optimal-strips at the same time limit.
"""

from pathlib import Path

from regression_lib import (
    compare_results,
    discover_instances,
    filter_baseline,
    load_baseline,
    resolve_configs,
    run_experiment,
    save_baseline,
)

TRACK_NAME = "satisficing"
TIME_LIMIT = 60       # seconds per instance (full mode)
LIGHT_TIME_LIMIT = 10 # seconds per instance (light mode)

CONFIGS = {
    "eager_greedy_ff": [
        "--search",
        "let(h,ff(),eager_greedy([h],preferred=[h]))"],
    "lazy_greedy_ff": [
        "--search",
        "let(h,ff(),lazy_greedy([h],preferred=[h]))"],
    "eager_greedy_add": [
        "--search",
        "let(h,add(),eager_greedy([h],preferred=[h]))"],
    "lama-first": [
        "--search",
        "let(hlm,landmark_sum(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false),"
        "let(hff,ff(transform=adapt_costs(one)),"
        "lazy_greedy([hff,hlm],preferred=[hff,hlm],"
        "cost_type=one,reopen_closed=false)))"],
}

# Satisficing cost is not bounded from below, but configs are deterministic
# (fixed RNG seed 2011), so cost and node counts must reproduce exactly.
EXACT_KEYS = ["cost", "expansions", "evaluations", "generated"]


def _run(benchmarks: Path, workers: int, light: bool, configs: dict) -> dict:
    agile_strips = benchmarks / "21.11-agile-strips"
    time_limit = LIGHT_TIME_LIMIT if light else TIME_LIMIT
    max_inst = 1 if light else 5
    instances = discover_instances(agile_strips, max_instance=max_inst)
    print(f"  Instances: {len(instances)} | configs: {len(configs)} | "
          f"time limit: {time_limit}s | workers: {workers}")
    return run_experiment(instances, configs, time_limit, workers)


def check_satisficing(benchmarks: Path, baseline_dir: Path, workers: int,
                      *, light: bool = True,
                      configs: dict = None,
                      extra_configs: dict = None) -> list[str]:
    print("Running satisficing search experiments...")
    resolved = resolve_configs(CONFIGS, configs, extra_configs)
    current = _run(benchmarks, workers, light, resolved)
    time_limit = LIGHT_TIME_LIMIT if light else TIME_LIMIT
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return [f"No baseline found at {baseline_dir}/{TRACK_NAME}.json; "
                "run generate_baseline.py first"]
    if light:
        baseline = filter_baseline(baseline, set(resolved), 1)
    return compare_results(current, baseline, EXACT_KEYS, time_limit=time_limit)


def update_satisficing(benchmarks: Path, baseline_dir: Path, workers: int,
                       *, light: bool = True,
                       configs: dict = None,
                       extra_configs: dict = None) -> None:
    if light:
        raise ValueError("update_satisficing must not be called in light mode")
    print("Running satisficing search experiments (baseline generation)...")
    resolved = resolve_configs(CONFIGS, configs, extra_configs)
    results = _run(benchmarks, workers, False, resolved)
    has_override = configs is not None or extra_configs is not None
    save_baseline(baseline_dir, TRACK_NAME, results, merge=has_override)
