#!/usr/bin/env python3
"""One-off special plot (not part of the default shard2_sweep.py generation):
for a given domain/level, lmcount results -- ratchet_triangle, adaptive_triangle,
adaptive_rectangle, and static rectangle aspect=1 -- on one linear-scale
control-parameter axis, capped at 30. Static triangle (slope=48) and
rectangle_a500 are excluded: both dwarf the rest on a linear axis, hiding the
dynamics that are actually of interest.

Requires the static "triangle" (slope=48) and "rectangle_a1"/"rectangle_a500"
runs for the given domain/instance under results-lmcount -- generated once
per level via, e.g.:
  shard2_sweep.py --algos triangle,rectangle_a1,rectangle_a500 \
      --domains elevators --levels easy,hard \
      --heuristic 'landmark_sum(lm_reasonable_orders_hps(lm_rhw()), \
      transform=adapt_costs(one))' --hlabel lmcount --out results-lmcount

Run with the same matplotlib-capable interpreter as shard2_sweep.py:
  .venv/bin/python elevators_easy_combined_control.py [domain] [easy|hard]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shard2_sweep as S

# domain -> {level: problem file}, filled in as each domain gets this treatment.
DOMAIN_LEVEL_PROBS = {
    "elevators": {"easy": "p01.pddl", "hard": "p12.pddl"},
    "ged": {"easy": "p01.pddl", "hard": "p28.pddl"},
    "tidybot": {"easy": "p01.pddl", "hard": "p23.pddl"},
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results-lmcount")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", nargs="?", default="elevators")
    ap.add_argument("level", nargs="?", default="easy")
    ap.add_argument("--logx", action="store_true",
                    help="log-scale the expansions (x) axis")
    ap.add_argument("--yscale", choices=("linear", "log"), default="linear",
                    help="control-parameter (y) axis scale (default: linear, "
                         "capped at 30)")
    ap.add_argument("--baselines", action="store_true",
                    help="plot only the static triangle (slope=48) and "
                         "rectangle (aspect=500) baselines instead of the "
                         "four dynamic configs")
    ap.add_argument("--all", action="store_true",
                    help="plot all six configs: the four dynamic ones plus "
                         "both static baselines")
    args = ap.parse_args()
    DOM = args.domain
    LEVEL = args.level
    if DOM not in DOMAIN_LEVEL_PROBS:
        print(f"unknown domain {DOM!r}, expected one of "
              f"{list(DOMAIN_LEVEL_PROBS)}", file=sys.stderr)
        return 1
    if LEVEL not in DOMAIN_LEVEL_PROBS[DOM]:
        print(f"unknown level {LEVEL!r}, expected one of "
              f"{list(DOMAIN_LEVEL_PROBS[DOM])}", file=sys.stderr)
        return 1
    PROB = DOMAIN_LEVEL_PROBS[DOM][LEVEL]

    S.OUT = OUT
    S.HLABEL = "lmcount"

    if args.all:
        algo_names = ("triangle", "ratchet_triangle", "adaptive_triangle",
                      "adaptive_rectangle", "rectangle_a1", "rectangle_a500")
    elif args.baselines:
        algo_names = ("triangle", "rectangle_a500")
    else:
        # aspect=500 and static slope=48 both dwarf the rest on a linear
        # axis, so they're excluded here -- the remaining dynamics are what's
        # actually of interest.
        algo_names = ("ratchet_triangle", "adaptive_triangle",
                      "adaptive_rectangle", "rectangle_a1")
    algos_by_name = {a["name"]: a for a in S.ALGOS}
    tasks = [(DOM, LEVEL, PROB, algos_by_name[name]) for name in algo_names]

    series = S._control_series(tasks, set(algo_names))
    if not series:
        print("no data found -- run the searches first", file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType), not the default Type 3 -- AAAI/IEEE reject Type 3
    # fonts in submitted PDFs.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    xrange = S._common_xrange(series)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for s in series:
        ax.plot(s["expansions"], s["y"], lw=0.9, color=s["color"],
                label=S.display_name(s["name"]))

    if args.yscale == "log":
        ax.set_yscale("log")
        ax.set_ylabel("control parameter (log scale)")
    else:
        ax.set_ylabel("control parameter")
    prob_stem = os.path.splitext(PROB)[0]
    ax.set_title(f"Control Parameters, {DOM} {prob_stem}", fontsize=12)
    ax.set_xlabel("expansions")
    if args.logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    if xrange is not None:
        xlo, xhi = xrange
        if args.logx:
            # log scale can't display x <= 0, which is where every series
            # starts (expansion 0) -- clip the left edge to the first
            # positive expansion value instead.
            xlo = max(xlo, 1)
        ax.set_xlim(xlo, xhi)
        # matplotlib's y-autoscale looks at the *whole* line, including the
        # part clipped off by set_xlim above -- a late-run outlier past the
        # shared window (e.g. ratchet_triangle's slope spiking to 1e9 right
        # before its own run ends, well past the common window) would then
        # blow up the y-range and squash everything actually visible. Rescale
        # to only the values inside the visible window instead. Use the
        # unclipped left edge here (not the log-scale-adjusted one above) so
        # this still reflects the true shared window.
        xlo, xhi = xrange
        visible_ys = [y for s in series
                      for x, y in zip(s["expansions"], s["y"])
                      if xlo <= x <= xhi]
        if visible_ys:
            ymin = min(visible_ys)
            ymax = max(visible_ys)
            if args.yscale == "log":
                ax.set_ylim(ymin / 1.5, ymax * 1.5)
            elif args.baselines or args.all:
                # the two static baselines (slope=48, aspect=500) both sit
                # well above the 30 cap used for the dynamic-config plot, so
                # just pad the natural data range instead of capping it.
                pad = (ymax - ymin) * 0.05 or 1.0
                ax.set_ylim(0, ymax + pad)
            else:
                pad = (ymax - ymin) * 0.05 or 1.0
                ax.set_ylim(ymin - pad, 30)
    fig.tight_layout()

    plotdir = os.path.join(OUT, "plots")
    os.makedirs(plotdir, exist_ok=True)
    outpng = os.path.join(
        plotdir, f"{DOM}_{LEVEL}_special_combined_control.pdf")
    fig.savefig(outpng)
    plt.close(fig)
    print(f"wrote {outpng}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
