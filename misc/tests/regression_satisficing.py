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
    DEFAULT_FULL_INSTANCES,
    DEFAULT_LIGHT_INSTANCES,
    baseline_missing_error,
    compare_results,
    drop_invalid_runs,
    filter_baseline,
    is_full_default_instances,
    load_baseline,
    resolve_configs,
    resolve_instances,
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


def _run(domain_dir: Path, workers: int, time_limit: int,
         configs: dict, instances: list,
         validate: bool, validate_bin) -> tuple[list[dict], dict]:
    discovered = resolve_instances(domain_dir, instances)
    print(f"  Instances: {len(discovered)} | configs: {len(configs)} | "
          f"time limit: {time_limit}s | workers: {workers} | "
          f"validate: {'on' if validate else 'off'}")
    results = run_experiment(discovered, configs, time_limit, workers,
                             validate=validate, validate_bin=validate_bin)
    return discovered, results


def check_satisficing(domain_dir: Path, baseline_dir: Path, workers: int,
                      *, configs: dict = None,
                      extra_configs: dict = None,
                      instances: list = None,
                      validate: bool = True,
                      validate_bin: Path = None) -> dict:
    print("Running satisficing search experiments...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_LIGHT_INSTANCES)
    )
    time_limit = (
        LIGHT_TIME_LIMIT if resolved_instances == DEFAULT_LIGHT_INSTANCES else TIME_LIMIT
    )
    discovered, current = _run(domain_dir, workers, time_limit, resolved_configs,
                               resolved_instances, validate, validate_bin)
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return baseline_missing_error(baseline_dir, TRACK_NAME)
    baseline = filter_baseline(baseline, set(resolved_configs), discovered)
    return compare_results(current, baseline, EXACT_KEYS, time_limit=time_limit)


def update_satisficing(domain_dir: Path, baseline_dir: Path, workers: int,
                       *, configs: dict = None,
                       extra_configs: dict = None,
                       instances: list = None,
                       validate: bool = True,
                       validate_bin: Path = None) -> list[str]:
    print("Running satisficing search experiments (baseline generation)...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_FULL_INSTANCES)
    )
    _discovered, results = _run(domain_dir, workers, TIME_LIMIT, resolved_configs,
                                resolved_instances, validate, validate_bin)
    errors = drop_invalid_runs(results)
    has_override = (
        configs is not None or extra_configs is not None
        or not is_full_default_instances(resolved_instances)
    )
    save_baseline(baseline_dir, TRACK_NAME, results, merge=has_override)
    return errors
