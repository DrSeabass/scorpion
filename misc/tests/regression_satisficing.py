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
    load_baseline,
    run_experiment,
    save_baseline,
)

TRACK_NAME = "satisficing"
TIME_LIMIT = 60  # seconds per instance

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


def _run(benchmarks: Path, workers: int) -> dict:
    agile_strips = benchmarks / "21.11-agile-strips"
    instances = discover_instances(agile_strips)
    print(f"  Instances: {len(instances)} | configs: {len(CONFIGS)} | "
          f"time limit: {TIME_LIMIT}s | workers: {workers}")
    return run_experiment(instances, CONFIGS, TIME_LIMIT, workers)


def check_satisficing(benchmarks: Path, baseline_dir: Path, workers: int) -> list[str]:
    print("Running satisficing search experiments...")
    current = _run(benchmarks, workers)
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return [f"No baseline found at {baseline_dir}/{TRACK_NAME}.json; "
                "run generate_baseline.py first"]
    return compare_results(current, baseline, EXACT_KEYS, time_limit=TIME_LIMIT)


def update_satisficing(benchmarks: Path, baseline_dir: Path, workers: int) -> None:
    print("Running satisficing search experiments (baseline generation)...")
    results = _run(benchmarks, workers)
    save_baseline(baseline_dir, TRACK_NAME, results)
