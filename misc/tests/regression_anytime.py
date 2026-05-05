"""
Anytime track for the scorpion regression test harness.

Runs two representative anytime configurations on the agile-strips p01-p05
instance set, checking the full incumbent cost sequence for exact-match
regressions and search time for a geo-mean slowdown regression.

Config selection rationale:
  iterated_wa_ff  — iterated weighted A* with FF; decreasing weights 10→5→3→2→1;
                    standard baseline for anytime search quality.
  lama            — LAMA-style iterated with landmark heuristic + FF, unit-cost
                    adaptation; tests landmark factory + cost-bounding iteration.

Exact match on "incumbent_costs" (the full improving cost sequence) is the
primary regression signal: any change to the search algorithm, heuristic, or
tie-breaking will shift the sequence.  Node counts are not compared because
iterated search reports them per phase and aggregation is non-trivial.

Uses 21.11-agile-strips (same as satisficing track) and a 60 s time limit.
"""

from pathlib import Path

from regression_lib import (
    DEFAULT_FULL_INSTANCES,
    DEFAULT_LIGHT_INSTANCES,
    compare_results,
    filter_baseline,
    is_full_default_instances,
    load_baseline,
    resolve_configs,
    resolve_instances,
    run_experiment,
    save_baseline,
)

TRACK_NAME = "anytime"
TIME_LIMIT = 60  # seconds per instance (both modes — anytime planners must run to budget)

CONFIGS = {
    "iterated_wa_ff": [
        "--search",
        "let(h,ff(),iterated(["
        "lazy_wastar([h],w=10,preferred=[h]),"
        "lazy_wastar([h],w=5,preferred=[h]),"
        "lazy_wastar([h],w=3,preferred=[h]),"
        "lazy_wastar([h],w=2,preferred=[h]),"
        "lazy_wastar([h],w=1,preferred=[h])"
        "],repeat_last=true,continue_on_fail=true))"],
    "lama": [
        "--search",
        "let(hlm,landmark_sum(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false),"
        "let(hff,ff(transform=adapt_costs(one)),"
        "iterated(["
        "lazy_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one,reopen_closed=false),"
        "lazy_wastar([hff,hlm],preferred=[hff,hlm],w=5,cost_type=one,reopen_closed=false),"
        "lazy_wastar([hff,hlm],preferred=[hff,hlm],w=3,cost_type=one,reopen_closed=false),"
        "lazy_wastar([hff,hlm],preferred=[hff,hlm],w=2,cost_type=one,reopen_closed=false),"
        "lazy_wastar([hff,hlm],preferred=[hff,hlm],w=1,cost_type=one,reopen_closed=false)"
        "],repeat_last=true,continue_on_fail=true)))"],
}

# The full incumbent cost sequence must reproduce exactly.
EXACT_KEYS = ["incumbent_costs"]


def _run(benchmarks: Path, workers: int,
         configs: dict, instances: list) -> dict:
    agile_strips = benchmarks / "21.11-agile-strips"
    discovered = resolve_instances(agile_strips, instances)
    print(f"  Instances: {len(discovered)} | configs: {len(configs)} | "
          f"time limit: {TIME_LIMIT}s | workers: {workers}")
    return run_experiment(discovered, configs, TIME_LIMIT, workers)


def check_anytime(benchmarks: Path, baseline_dir: Path, workers: int,
                  *, configs: dict = None,
                  extra_configs: dict = None,
                  instances: list = None) -> list[str]:
    print("Running anytime search experiments...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_LIGHT_INSTANCES)
    )
    current = _run(benchmarks, workers, resolved_configs, resolved_instances)
    baseline = load_baseline(baseline_dir, TRACK_NAME)
    if not baseline:
        return [f"No baseline found at {baseline_dir}/{TRACK_NAME}.json; "
                "run generate_baseline.py first"]
    baseline = filter_baseline(baseline, set(resolved_configs), resolved_instances)
    # wall_time: iterated search does not emit "Search time:".
    # time_limit=None: disables the coverage-stability threshold — anytime
    # wall_time is always ~60s so the threshold would skip every coverage loss.
    return compare_results(current, baseline, EXACT_KEYS,
                           runtime_key="wall_time", time_limit=None,
                           prefix_keys=EXACT_KEYS)


def update_anytime(benchmarks: Path, baseline_dir: Path, workers: int,
                   *, configs: dict = None,
                   extra_configs: dict = None,
                   instances: list = None) -> None:
    print("Running anytime search experiments (baseline generation)...")
    resolved_configs = resolve_configs(CONFIGS, configs, extra_configs)
    resolved_instances = (
        list(instances) if instances is not None else list(DEFAULT_FULL_INSTANCES)
    )
    results = _run(benchmarks, workers, resolved_configs, resolved_instances)
    has_override = (
        configs is not None or extra_configs is not None
        or not is_full_default_instances(resolved_instances)
    )
    save_baseline(baseline_dir, TRACK_NAME, results, merge=has_override)
