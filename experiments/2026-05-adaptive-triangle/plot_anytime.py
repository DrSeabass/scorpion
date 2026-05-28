#!/usr/bin/env python3
"""Per-domain anytime quality plots from the B-adaptive eval data.

For each (domain, instance), best_known is the minimum solution cost found
by ANY algorithm in this dataset at any time within the search budget.
For each algorithm, the cost-in-hand at time t is a step function built from
the (cost_times:all, cost:all) streams: infinity before the first solution,
then the most recent reported cost. Normalized score = best_known / cost(t),
so unsolved-at-t = 0 and best_known = 1. Per-domain curves average the
normalized score across instances solved by at least one algorithm.

Outputs PNG per domain plus an all-domains aggregate to ./plots/.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).resolve().parent
EVAL_PROPS = DIR / "data" / "2026-05-26-B-adaptive-vs-static-vs-lama-eval" / "properties"
PLOTS_DIR = DIR / "plots"

BUDGET_SECONDS = 1800.0          # 30m from the experiment script
GRID_MIN = 0.05                  # earliest meaningful time on x-axis
GRID_POINTS = 240                # log-spaced samples along x

ALGORITHMS = [
    "adaptive-triangle-gonly",
    "adaptive-triangle-hmax",
    "adaptive-triangle-lmcut",
    "ratchet-triangle-gonly",
    "ratchet-triangle-hmax",
    "ratchet-triangle-lmcut",
]

# Distinct colors + linestyles so the two families are visually separable.
STYLE = {
    "adaptive-triangle-gonly": dict(color="#1f77b4", linestyle="-",  linewidth=1.6),
    "adaptive-triangle-hmax":  dict(color="#2ca02c", linestyle="-",  linewidth=1.6),
    "adaptive-triangle-lmcut": dict(color="#d62728", linestyle="-",  linewidth=1.6),
    "ratchet-triangle-gonly":  dict(color="#1f77b4", linestyle="--", linewidth=1.6),
    "ratchet-triangle-hmax":   dict(color="#2ca02c", linestyle="--", linewidth=1.6),
    "ratchet-triangle-lmcut":  dict(color="#d62728", linestyle="--", linewidth=1.6),
}


def load_eval(path: Path):
    with path.open() as f:
        return json.load(f)


def stream_cost_at(times: np.ndarray, costs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step function value of the (times, costs) stream sampled at grid.
    Returns +inf before the first reported solution time."""
    out = np.full_like(grid, np.inf, dtype=float)
    if times.size == 0:
        return out
    # For each grid point, find the index of the latest time <= t.
    # np.searchsorted with side='right' gives count of times <= t.
    idx = np.searchsorted(times, grid, side="right") - 1
    mask = idx >= 0
    out[mask] = costs[idx[mask]]
    return out


def build_dataset(eval_data):
    """Returns dict[domain][problem][algorithm] = (times_arr, costs_arr).
    Only entries with a non-empty solution stream are kept."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    for entry in eval_data.values():
        if entry.get("coverage") != 1:
            continue
        times = entry.get("cost_times:all") or []
        costs = entry.get("cost:all") or []
        if not times or not costs or len(times) != len(costs):
            continue
        dom = entry["domain"]
        prob = entry["problem"]
        algo = entry["algorithm"]
        data[dom][prob][algo] = (
            np.asarray(times, dtype=float),
            np.asarray(costs, dtype=float),
        )
    return data


def per_domain_curves(domain_data, grid):
    """For one domain, return dict[algorithm] -> mean normalized score curve.
    Mean is taken over instances solved by at least one algorithm. Algorithms
    that did not solve a given instance contribute 0 at every t."""
    score_sums = {a: np.zeros_like(grid) for a in ALGORITHMS}
    instance_count = 0

    for prob, algo_streams in domain_data.items():
        # best_known across all algos that solved this instance
        final_costs = [c[-1] for (_, c) in algo_streams.values()]
        if not final_costs:
            continue
        best_known = min(final_costs)
        instance_count += 1

        for algo in ALGORITHMS:
            stream = algo_streams.get(algo)
            if stream is None:
                continue  # contributes 0 to this instance's sum
            times, costs = stream
            cost_at_t = stream_cost_at(times, costs, grid)
            solved = ~np.isinf(cost_at_t)
            if best_known == 0:
                # Degenerate trivially-solved instance: any solution matches optimum.
                score = np.where(solved, 1.0, 0.0)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    score = np.where(solved, best_known / cost_at_t, 0.0)
            score_sums[algo] += score

    if instance_count == 0:
        return None, 0
    return {a: score_sums[a] / instance_count for a in ALGORITHMS}, instance_count


def plot_curves(curves, grid, title, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for algo in ALGORITHMS:
        y = curves[algo]
        ax.plot(grid, y, label=algo, **STYLE[algo])
    ax.set_xscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean normalized quality   (best_known / cost(t))")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    print(f"loading {EVAL_PROPS}")
    eval_data = load_eval(EVAL_PROPS)
    data = build_dataset(eval_data)
    print(f"{len(data)} domains with at least one solved instance")

    PLOTS_DIR.mkdir(exist_ok=True)
    grid = np.logspace(math.log10(GRID_MIN), math.log10(BUDGET_SECONDS), GRID_POINTS)

    # Per-domain plots
    domain_curves = {}
    domain_counts = {}
    skipped = []
    for dom in sorted(data):
        curves, n = per_domain_curves(data[dom], grid)
        if curves is None:
            skipped.append(dom)
            continue
        domain_curves[dom] = curves
        domain_counts[dom] = n
        out = PLOTS_DIR / f"{dom}.png"
        plot_curves(curves, grid, f"{dom}   (n={n} instances)", out)
        print(f"  {dom}: n={n}  -> {out.name}")

    if skipped:
        print(f"skipped (no solved instances): {skipped}")

    # Overall aggregate: equal-weight per domain (so big domains don't dominate)
    if domain_curves:
        overall = {a: np.zeros_like(grid) for a in ALGORITHMS}
        for curves in domain_curves.values():
            for a in ALGORITHMS:
                overall[a] += curves[a]
        for a in ALGORITHMS:
            overall[a] /= len(domain_curves)
        out = PLOTS_DIR / "_all-domains-mean.png"
        plot_curves(
            overall, grid,
            f"all domains   (equal-weight mean across {len(domain_curves)} domains)",
            out,
        )
        print(f"  aggregate: {out.name}")


if __name__ == "__main__":
    main()
