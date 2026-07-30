#!/usr/bin/env python3
"""Sweep the depth-instrumented anytime searches over shard2.

For every domain in the aaai27 shard2 suite it runs three algorithms
(adaptive_triangle, ratchet_triangle, adaptive_rectangle) on two instances --
an "easy" (smallest translator task size) and a "hard" (largest) -- with lmcut
for guidance and, where supported, for g+h pruning. Each run emits the
control/mindepth/maxdepth trace plus incumbent-improvement markers, and the
script renders a 2x2 depth figure (control, mindepth, maxdepth, overlaid) per
run.

Run it with a matplotlib-capable interpreter, e.g. the lab venv:
  /home/jorth68/research/scorpion/experiments/2026-07-13-rectangle-aspect-pilot/.venv/bin/python \
      shard2_sweep.py --time 180 --jobs 6

Useful flags:
  --time N        search time limit (s) per run           (default 180)
  --jobs N        parallel searches                       (default 4)
  --domains a,b   restrict to these domains               (default all shard2)
  --levels easy,hard   which instances                    (default both)
  --mem 8G        driver overall memory limit             (default 8G)
  --out DIR       output root                             (default scratchpad/shard2_sweep)
  --smoke         quick self-test: pathways/easy only, 20s, then plot
"""
import argparse
import collections
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- fixed locations -------------------------------------------------------
FD = "/home/jorth68/research/scorpion/fast-downward.py"
FD_PYTHON = "python3"  # interpreter for the FD driver (system python3)
BM = "/home/jorth68/research/autoscale-benchmarks/21.11-agile-strips"
# combined-eval is a superset of shard2-eval (same translator_task_size data
# for the original 11 domains -- verified) that also covers agricola and
# nomystery, needed to extend this sweep beyond shard2.
PROPS = ("/home/jorth68/research/scorpion/experiments/aaai27/data/"
         "anytime-param-config-combined-eval/properties")
# Default output lives next to this script so results persist with it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- algorithms: lmcut guidance + (where available) g+h pruning ------------
# adaptive_rectangle prunes on g+h via its eval already, so it needs no
# separate pruning_heuristic; the triangles take pruning_heuristic=lmcut().
# {h} = ranking/pruning heuristic (set via --heuristic), {t} = time limit.
# The triangles take pruning_heuristic={h} for g+h pruning; adaptive_rectangle
# prunes on g+h via its eval already. NB: g+h pruning is only sound when {h} is
# admissible (e.g. lmcut); with an inadmissible {h} (e.g. lm-count) the anytime
# incumbent may not converge to optimal, but the parameter/depth traces remain
# valid to observe.
ALGOS = [
    dict(name="adaptive_triangle", control="budget", clabel="budget",
         csv="adaptive_triangle_budget.csv",
         sol="adaptive_triangle_solutions.csv",
         search=("adaptive_triangle(eval={h}, pruning_heuristic={h}, "
                 "anytime=true, log_budget=true, max_time={t})")),
    dict(name="ratchet_triangle", control="slope", clabel="slope",
         csv="ratchet_triangle_slope.csv",
         sol="ratchet_triangle_solutions.csv",
         search=("ratchet_triangle(eval={h}, pruning_heuristic={h}, "
                 "anytime=true, log_slope=true, max_time={t})")),
    dict(name="adaptive_rectangle", control="aspect", clabel="aspect ratio",
         csv="adaptive_rectangle_aspect.csv",
         sol="adaptive_rectangle_solutions.csv",
         search=("adaptive_rectangle(eval={h}, anytime=true, "
                 "log_aspect=true, max_time={t})")),
    # Static (non-adaptive) triangle baseline, slope fixed at 48 to match the
    # AAAI-2026-triangle iteration-zero slope and the STATIC_BASELINES
    # reference line below. control="slope" is constant here by design (this
    # is the fixed-parameter baseline) -- mindepth/maxdepth still trace real
    # open-list depth-range dynamics.
    dict(name="triangle", control="slope", clabel="slope",
         csv="triangle_slope.csv",
         sol="triangle_solutions.csv",
         search=("triangle(eval={h}, slope=48, pruning_heuristic={h}, "
                 "anytime=true, log_slope=true, max_time={t})")),
    # Static (non-adaptive) rectangle baselines, aspect fixed at the two
    # agreed-on reference values (matches RECTANGLE_BASELINES below). control=
    # "aspect" is constant here by design (this is the fixed-parameter
    # baseline) -- mindepth/maxdepth still trace real open-list depth-range
    # dynamics. Both configs share the same hardcoded per-run CSV names
    # (rectangle_search.cc writes rectangle_aspect.csv / rectangle_solutions
    # .csv regardless of aspect); task_outdir keys on algo name, so the a=1 and
    # a=500 runs still land in separate directories.
    dict(name="rectangle_a1", control="aspect", clabel="aspect ratio",
         csv="rectangle_aspect.csv",
         sol="rectangle_solutions.csv",
         search=("rectangle(eval={h}, aspect=1, anytime=true, "
                 "log_aspect=true, max_time={t})")),
    dict(name="rectangle_a500", control="aspect", clabel="aspect ratio",
         csv="rectangle_aspect.csv",
         sol="rectangle_solutions.csv",
         search=("rectangle(eval={h}, aspect=500, anytime=true, "
                 "log_aspect=true, max_time={t})")),
]

# Heuristic expression + short label; overridden by --heuristic / --hlabel.
HEURISTIC = "lmcut()"
HLABEL = "lmcut"

MAX_POINTS = 120000  # plot downsample cap

# Fixed per-algorithm colors for the multi-algorithm overlay plots (one color
# per algorithm, held constant across mindepth/maxdepth/spread so the same
# algorithm reads as the same color everywhere).
ALGO_COLORS = {
    "adaptive_triangle": "#2b6cb0",
    "ratchet_triangle": "#c05621",
    "adaptive_rectangle": "#2f855a",
    "triangle": "#d69e2e",
    "rectangle_a1": "#1a365d",
    "rectangle_a500": "#9b2c2c",
}

# Legend/label display names: adaptive_triangle's per-step budget mechanism is
# "Triangle-Step"; the ratchet mechanism (fires at completed sweeps) is
# "Triangle-Sweep" on the triangle base and "Rectangle-Sweep" on the rectangle
# base. Algorithms not listed here display under their raw ALGOS name.
DISPLAY_NAMES = {
    "adaptive_triangle": "Triangle-Step",
    "ratchet_triangle": "Triangle-Sweep",
    "adaptive_rectangle": "Rectangle-Sweep",
}


def display_name(name):
    return DISPLAY_NAMES.get(name, name)


# ---- instance selection ----------------------------------------------------
# Overrides the size-based "hard" pick (the largest instance in the suite,
# which the 15-minute Arrhenius paper run shows going unsolved by everyone --
# see anytime-param-config-shard2-eval/properties). Picked as: the largest
# instance per domain where adaptive-triangle, ratchet-triangle, and
# adaptive-rectangle each reached >=3 incumbents (cost:all length) in that
# 15-minute run (>=2 where nothing bigger cleared 3), so this shorter local
# sweep has a real shot at multiple solutions per algorithm instead of zero.
HARD_OVERRIDES = {
    # p30 (the literal-largest, translator_task_size 3.3M) repeatedly hit
    # FD's own --overall-memory-limit deep into search (at both 5G and
    # observed the same pattern that would recur at 8G) and, worse, lost
    # adaptive_triangle's entire CSV trace since it only flushes on a clean
    # finish. p20 (2.97M) is the next largest local instance -- picked over
    # p29 (3.12M) to actually reduce load rather than nudge it.
    "agricola": "p20.pddl",
    "blocksworld": "p06.pddl",
    "elevators": "p12.pddl",
    "mprime": "p02.pddl",
    "pathways": "p05.pddl",
    "pipesworld-notankage": "p29.pddl",
    "satellite": "p01.pddl",
    "termes": "p03.pddl",
    "thoughtful": "p08.pddl",
    "tidybot": "p23.pddl",
    "tpp": "p09.pddl",
    "zenotravel": "p01.pddl",
}


def pick_instances(props_path):
    """domain -> {'easy': prob, 'hard': prob} by min-nonzero / max task size,
    with HARD_OVERRIDES substituted in for 'hard' where set."""
    with open(props_path) as f:
        data = json.load(f)
    sizes = collections.defaultdict(dict)
    for r in data.values():
        if r.get("algorithm") != "adaptive-triangle":
            continue
        sizes[r["domain"]][r["problem"]] = r.get("translator_task_size") or 0
    out = {}
    for dom, probs in sizes.items():
        items = sorted((s, p) for p, s in probs.items() if s > 0)
        if not items:
            continue
        hard = HARD_OVERRIDES.get(dom, items[-1][1])
        out[dom] = {"easy": items[0][1], "hard": hard}
    return out


def resolve_domain_file(ddir, prob):
    """Single domain.pddl, else per-instance domain-<prob>.pddl (e.g. pathways)."""
    cand = os.path.join(ddir, "domain.pddl")
    if os.path.exists(cand):
        return cand
    stem = os.path.splitext(prob)[0]
    cand = os.path.join(ddir, f"domain-{stem}.pddl")
    if os.path.exists(cand):
        return cand
    return None


# ---- running ---------------------------------------------------------------
# "Driver aborting after search" covers FD's own internal
# --overall-memory-limit abort (e.g. "Failed to allocate memory" after one or
# more incumbents) -- that's a completed run, not a hung one, so it must count
# as terminal or run_already_complete() would retry it forever.
TERMINAL_MARKERS = ("Time limit reached", "Solution found",
                    "without finding a solution", "is empty",
                    "Driver aborting after search")


def run_already_complete(outdir, algo):
    """A run counts as done if its control CSV has data and run.log shows the
    search terminated -- lets the sweep be re-run to fill only what's missing."""
    csv_path = os.path.join(outdir, algo["csv"])
    log_path = os.path.join(outdir, "run.log")
    if not os.path.exists(csv_path) or not os.path.exists(log_path):
        return False
    with open(csv_path) as f:
        rows = sum(1 for _ in f) - 1
    if rows <= 0:
        return False
    with open(log_path, errors="ignore") as f:
        text = f.read()
    return any(m in text for m in TERMINAL_MARKERS)


def run_one(task, time_limit, mem):
    dom, level, prob, algo = task
    ddir = os.path.join(BM, dom)
    domfile = resolve_domain_file(ddir, prob)
    probfile = os.path.join(ddir, prob)
    outdir = task_outdir(task)
    os.makedirs(outdir, exist_ok=True)
    if run_already_complete(outdir, algo):
        return task, "skip-done"
    if not domfile or not os.path.exists(probfile):
        return task, "missing-pddl"

    search = algo["search"].format(t=time_limit, h=HEURISTIC)
    cmd = [FD_PYTHON, FD, "--overall-memory-limit", mem,
           "--plan-file", "sas_plan", domfile, probfile, "--search", search]
    # translation of the largest tasks can be slow; allow generous head-room
    # over the search limit before we give up on the subprocess entirely.
    hard_timeout = time_limit + 1200
    with open(os.path.join(outdir, "run.log"), "w") as log:
        # fast-downward.py spawns translate/search as its own children; run it
        # in a new process group so a timeout can kill the whole tree instead
        # of just the driver wrapper, which would otherwise orphan a
        # still-running (and possibly memory-hungry) translate subprocess.
        proc = subprocess.Popen(cmd, cwd=outdir, stdout=log,
                                 stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            proc.wait(timeout=hard_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return task, "subprocess-timeout"
    return task, "ok"


def task_outdir(task):
    dom, level, prob, algo = task
    return os.path.join(OUT, dom, f"{level}_{os.path.splitext(prob)[0]}",
                        algo["name"])


# ---- plotting --------------------------------------------------------------
def _load(path, cols):
    """Read a control CSV, dropping any trailing row truncated by an
    out-of-memory kill mid-flush (missing/empty fields).

    adaptive_triangle's budget is floored to 1 at the top of every step()
    regardless of how negative it fell during the previous step's uninformed
    stretch (adaptive_triangle_search.cc), so a logged value <= 0 is display
    noise from mid-step debit bookkeeping, not a lasting state -- show it as
    1 (still-affordable-next-layer) rather than the raw debit.
    """
    out = {c: [] for c in cols}
    with open(path) as f:
        for row in csv.DictReader(f):
            if any(not row.get(c) for c in cols):
                continue
            for c in cols:
                v = float(row[c])
                if c == "budget" and v <= 0:
                    v = 1.0
                out[c].append(v)
    return out


def _load_sols(path):
    xs = []
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                xs.append(int(row["log_element"]))
    return xs


def plot_one(task):
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType), not the default Type 3 -- AAAI/IEEE reject Type 3
    # fonts in submitted PDFs.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    dom, level, prob, algo = task
    outdir = task_outdir(task)
    path = os.path.join(outdir, algo["csv"])
    if not os.path.exists(path):
        return task, "no-csv"
    data = _load(path, ["expansions", algo["control"], "mindepth", "maxdepth"])
    n = len(data[algo["control"]])
    if n == 0:
        return task, "empty-csv"
    xs = data["expansions"]
    sol_idx = _load_sols(os.path.join(outdir, algo["sol"]))
    sols = [xs[i] for i in sol_idx if i < n]

    def add_sols(ax):
        for k, sx in enumerate(sols):
            ax.axvline(sx, color="red", lw=1.0, alpha=0.7,
                       label="solution found" if k == 0 else None)

    depth_spread = [maxd - mind
                    for maxd, mind in zip(data["maxdepth"], data["mindepth"])]

    plotdir = os.path.join(OUT, "plots")
    os.makedirs(plotdir, exist_ok=True)
    indiv_dir = os.path.join(plotdir, "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    def save_individual(col, series, title, legend=False):
        """series is a list of (ys, label, color); saved standalone next to
        the combined page so single panels can be dropped straight into the
        paper without cropping the multi-panel figure."""
        fig_i, a = plt.subplots(figsize=(7, 5))
        for ys, lab, color in series:
            a.plot(xs, ys, lw=0.9, color=color, label=lab)
        add_sols(a)
        a.set_title(f"{title} - {dom} {level} {algo['name']}")
        a.set_xlabel("expansions")
        a.set_ylabel("magnitude")
        a.grid(True, alpha=0.3)
        if legend:
            a.legend(loc="best", fontsize=8)
        fig_i.tight_layout()
        outp = os.path.join(indiv_dir, f"{dom}_{level}_{algo['name']}_{col}.pdf")
        fig_i.savefig(outp)
        plt.close(fig_i)

    fig, ax = plt.subplots(2, 3, figsize=(18, 7))
    panels = [
        (ax[0][0], algo["control"], data[algo["control"]], algo["clabel"],
         "#2b6cb0"),
        (ax[0][1], "mindepth", data["mindepth"],
         "mindepth (shallowest open list)", "#2f855a"),
        (ax[0][2], "maxdepth", data["maxdepth"],
         "maxdepth (deepest open list)", "#c05621"),
        (ax[1][0], "depth_spread", depth_spread,
         "maxdepth - mindepth (open-list depth spread)", "#805ad5"),
    ]
    for a, col, ys, lab, color in panels:
        a.plot(xs, ys, lw=0.8, color=color)
        add_sols(a)
        a.set_title(lab)
        a.set_xlabel("expansions")
        a.set_ylabel("magnitude")
        a.grid(True, alpha=0.3)
        save_individual(col, [(ys, lab, color)], lab)
    ov = ax[1][1]
    ov.plot(xs, data[algo["control"]], lw=0.8, color="#2b6cb0",
            label=algo["clabel"])
    ov.plot(xs, data["mindepth"], lw=0.8, color="#2f855a", label="mindepth")
    ov.plot(xs, data["maxdepth"], lw=0.8, color="#c05621", label="maxdepth")
    add_sols(ov)
    ov.set_title("overlaid")
    ov.set_xlabel("expansions")
    ov.set_ylabel("magnitude")
    ov.grid(True, alpha=0.3)
    ov.legend(loc="best", fontsize=8)
    ax[1][2].axis("off")
    save_individual("overlaid", [
        (data[algo["control"]], algo["clabel"], "#2b6cb0"),
        (data["mindepth"], "mindepth", "#2f855a"),
        (data["maxdepth"], "maxdepth", "#c05621"),
    ], "overlaid", legend=True)

    fig.suptitle(f"{algo['name']} - {dom} {level} ({prob}, {HLABEL})   "
                 f"[{n} log elements, {len(sol_idx)} solutions]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpng = os.path.join(plotdir,
                          f"{dom}_{level}_{algo['name']}.pdf")
    fig.savefig(outpng)
    plt.close(fig)
    return task, os.path.relpath(outpng, OUT)


def _load_algo_series(task):
    """Per-algorithm mindepth/maxdepth/spread keyed on `expansions` (a
    cumulative node count shared across algorithms) rather than row index --
    algorithms log at wildly different rates (e.g. adaptive_triangle logs ~70
    rows for the same run where ratchet_triangle logs ~4M), so overlaying by
    row index would misalign them; expansions is the common progress axis."""
    dom, level, prob, algo = task
    outdir = task_outdir(task)
    path = os.path.join(outdir, algo["csv"])
    if not os.path.exists(path):
        return None
    data = _load(path, ["expansions", "mindepth", "maxdepth"])
    n = len(data["expansions"])
    if n == 0:
        return None
    spread = [maxd - mind for maxd, mind in zip(data["maxdepth"], data["mindepth"])]
    sol_idx = _load_sols(os.path.join(outdir, algo["sol"]))
    sol_x = [data["expansions"][i] for i in sol_idx if i < n]
    return dict(name=algo["name"], color=ALGO_COLORS.get(algo["name"], "#333333"),
                expansions=data["expansions"], mindepth=data["mindepth"],
                maxdepth=data["maxdepth"], spread=spread, sol_x=sol_x)


def _common_xrange(series, xkey="expansions", pad_frac=0.02):
    """(left, right) view bounds so every algorithm has data across the
    whole visible x-range: right is clipped to the shallowest algorithm's
    max value -- algorithms log at wildly different rates (see
    _load_algo_series), so without this the view would run past where the
    fastest-finishing algorithm's line simply stops. Left gets a small pad
    off the shared data minimum rather than matplotlib's default 5%
    autoscale margin, which is sized against the *untrimmed* range and
    leaves a misleadingly large gap before the first point once the right
    edge is clipped."""
    maxes = [max(s[xkey]) for s in series if s[xkey]]
    mins = [min(s[xkey]) for s in series if s[xkey]]
    if not maxes or not mins:
        return None
    xmax = min(maxes)
    xmin = min(mins)
    return xmin - (xmax - xmin) * pad_frac, xmax


def plot_multi(dom, level, prob, group_tasks):
    """Three cross-algorithm overlay plots per (domain, level): depth spread
    (maxdepth-mindepth) for all algorithms on one axes, mindepth+maxdepth for
    all algorithms on one axes (color = algorithm, linestyle = min/max), and
    maxdepth alone for all algorithms on one axes."""
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType), not the default Type 3 -- AAAI/IEEE reject Type 3
    # fonts in submitted PDFs.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    series = [s for s in (_load_algo_series(t) for t in group_tasks)
              if s is not None]
    if not series:
        return "no-data", "no-data"
    xrange = _common_xrange(series)
    plotdir = os.path.join(OUT, "plots")
    os.makedirs(plotdir, exist_ok=True)

    def add_sols(ax, s):
        for sx in s["sol_x"]:
            ax.axvline(sx, color=s["color"], lw=0.8, alpha=0.3, linestyle=":")

    # Plot A: depth spread per algorithm.
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in series:
        ax.plot(s["expansions"], s["spread"], lw=0.8, color=s["color"],
                label=display_name(s["name"]))
        add_sols(ax, s)
    ax.set_title(f"maxdepth - mindepth by algorithm - {dom} {level} "
                 f"({prob}, {HLABEL})")
    ax.set_xlabel("expansions")
    ax.set_ylabel("maxdepth - mindepth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if xrange is not None:
        ax.set_xlim(*xrange)
    fig.tight_layout()
    spread_png = os.path.join(plotdir, f"{dom}_{level}_multi_depth_spread.pdf")
    fig.savefig(spread_png)
    plt.close(fig)

    # Plot B: mindepth + maxdepth per algorithm, color=algorithm,
    # linestyle=min(dashed)/max(solid).
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in series:
        ax.plot(s["expansions"], s["maxdepth"], lw=0.9, color=s["color"],
                linestyle="-", label=f"{display_name(s['name'])} maxdepth")
        ax.plot(s["expansions"], s["mindepth"], lw=0.9, color=s["color"],
                linestyle="--", label=f"{display_name(s['name'])} mindepth")
        add_sols(ax, s)
    ax.set_title(f"mindepth & maxdepth by algorithm - {dom} {level} "
                 f"({prob}, {HLABEL})")
    ax.set_xlabel("expansions")
    ax.set_ylabel("magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if xrange is not None:
        ax.set_xlim(*xrange)
    fig.tight_layout()
    minmax_png = os.path.join(plotdir,
                              f"{dom}_{level}_multi_mindepth_maxdepth.pdf")
    fig.savefig(minmax_png)
    plt.close(fig)

    # Plot C: maxdepth alone per algorithm.
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in series:
        ax.plot(s["expansions"], s["maxdepth"], lw=0.8, color=s["color"],
                label=display_name(s["name"]))
        add_sols(ax, s)
    ax.set_title(f"maxdepth by algorithm - {dom} {level} ({prob}, {HLABEL})")
    ax.set_xlabel("expansions")
    ax.set_ylabel("maxdepth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if xrange is not None:
        ax.set_xlim(*xrange)
    fig.tight_layout()
    maxdepth_png = os.path.join(plotdir, f"{dom}_{level}_multi_maxdepth.pdf")
    fig.savefig(maxdepth_png)
    plt.close(fig)

    return (os.path.relpath(spread_png, OUT), os.path.relpath(minmax_png, OUT),
             os.path.relpath(maxdepth_png, OUT))


# Flat reference lines for static (non-adaptive) baseline configurations,
# split by family: triangle-s48 (the AAAI-2026-triangle iteration-zero slope)
# for the triangle plot, and the two rectangle aspect baselines from the
# agreed 9-config matrix (rectangle-a1, rectangle-a500) for the rectangle
# plot. adaptive_triangle's budget is a different quantity from slope with no
# static equivalent, plotted alongside ratchet_triangle's slope for magnitude
# comparison only.
TRIANGLE_BASELINES = [(48, "static triangle (slope=48)", ":")]
RECTANGLE_BASELINES = [
    (1, "static rectangle (aspect=1)", "--"),
    (500, "static rectangle (aspect=500)", "-."),
]


def _control_series(group_tasks, algo_names):
    series = []
    for t in group_tasks:
        _, _, _, algo = t
        if algo["name"] not in algo_names:
            continue
        outdir = task_outdir(t)
        path = os.path.join(outdir, algo["csv"])
        if not os.path.exists(path):
            continue
        data = _load(path, ["expansions", algo["control"]])
        if not data["expansions"]:
            continue
        ys = [max(v, 1e-2) for v in data[algo["control"]]]
        series.append(dict(name=algo["name"], clabel=algo["clabel"],
                            color=ALGO_COLORS.get(algo["name"], "#333333"),
                            expansions=data["expansions"], y=ys))
    return series


def _plot_control_family(dom, level, prob, series, baselines, family, title):
    """One log-scale control-parameter plot for a single algorithm family
    (triangle or rectangle), with that family's static reference lines. Log
    scale because ratchet_triangle's slope alone spans ~9 orders of
    magnitude within a single run -- a linear axis would flatten everything
    else to invisibility."""
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType), not the default Type 3 -- AAAI/IEEE reject Type 3
    # fonts in submitted PDFs.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    if not series:
        return "no-data"

    xrange = _common_xrange(series)
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in series:
        ax.plot(s["expansions"], s["y"], lw=0.9, color=s["color"],
                label=f"{display_name(s['name'])} ({s['clabel']})")
    for val, lab, ls in baselines:
        ax.axhline(val, color="#4a5568", lw=1.2, linestyle=ls, label=lab)

    ax.set_yscale("log")
    ax.set_title(f"{title} - {dom} {level} ({prob}, {HLABEL})")
    ax.set_xlabel("expansions")
    ax.set_ylabel("control parameter (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)
    if xrange is not None:
        ax.set_xlim(*xrange)
    fig.tight_layout()
    plotdir = os.path.join(OUT, "plots")
    os.makedirs(plotdir, exist_ok=True)
    outpng = os.path.join(plotdir,
                          f"{dom}_{level}_multi_control_{family}.pdf")
    fig.savefig(outpng)
    plt.close(fig)
    return os.path.relpath(outpng, OUT)


def plot_multi_control(dom, level, prob, group_tasks):
    """Two control-parameter plots split by family: triangle
    (adaptive_triangle's budget + ratchet_triangle's slope + static
    triangle's own slope=48, vs. the slope=48 reference line) and rectangle
    (adaptive_rectangle's aspect, vs. static aspect=1/500)."""
    tri_series = _control_series(
        group_tasks, {"adaptive_triangle", "ratchet_triangle", "triangle"})
    rect_series = _control_series(
        group_tasks,
        {"adaptive_rectangle", "rectangle_a1", "rectangle_a500"})
    tri_msg = _plot_control_family(dom, level, prob, tri_series,
                                    TRIANGLE_BASELINES, "triangle",
                                    "triangle control parameter")
    rect_msg = _plot_control_family(dom, level, prob, rect_series,
                                     RECTANGLE_BASELINES, "rectangle",
                                     "rectangle control parameter")
    return tri_msg, rect_msg


# ---- driver ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=int, default=180)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--domains", default="")
    ap.add_argument("--levels", default="easy,hard")
    ap.add_argument("--algos", default="",
                    help="comma list of algorithm names to run/plot, e.g. "
                         "adaptive_triangle (default: all three)")
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--heuristic", default="lmcut()",
                    help="ranking/pruning heuristic expression (FD syntax)")
    ap.add_argument("--hlabel", default="",
                    help="short label for plot titles/paths (default: guessed "
                         "from --heuristic)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "results"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip the search phase; just (re)render plots from "
                         "whatever data is already on disk")
    args = ap.parse_args()

    global OUT, HEURISTIC, HLABEL
    OUT = args.out
    HEURISTIC = args.heuristic
    HLABEL = args.hlabel or args.heuristic.split("(")[0] or "h"

    if args.smoke:
        args.domains, args.levels, args.time = "pathways", "easy", 20

    insts = pick_instances(PROPS)
    domains = ([d.strip() for d in args.domains.split(",") if d.strip()]
               or sorted(insts))
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    algo_names = [a.strip() for a in args.algos.split(",") if a.strip()]
    algos = ([a for a in ALGOS if a["name"] in algo_names] if algo_names
             else ALGOS)
    if algo_names:
        unknown = set(algo_names) - {a["name"] for a in ALGOS}
        if unknown:
            print(f"unknown --algos: {sorted(unknown)}", file=sys.stderr)

    tasks = []
    for dom in domains:
        if dom not in insts:
            print(f"skip unknown domain: {dom}", file=sys.stderr)
            continue
        for level in levels:
            prob = insts[dom][level]
            for algo in algos:
                tasks.append((dom, level, prob, algo))

    print(f"{len(tasks)} runs over {len(domains)} domain(s), "
          f"{args.time}s each, {args.jobs} parallel. out={OUT}")
    if os.path.isdir(OUT):
        print(f"(reusing existing {OUT})")

    # Phase 1: run searches in parallel (subprocess releases the GIL).
    if not args.plot_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, t, args.time, args.mem): t for t in tasks}
            for fut in as_completed(futs):
                t, status = fut.result()
                dom, level, prob, algo = t
                print(f"  [run] {dom}/{level}/{algo['name']}: {status}")
    else:
        print("plot-only: skipping search phase")

    # Phase 2: plot serially (matplotlib is not thread-safe).
    print("plotting...")
    groups = {}
    for t in tasks:
        _, msg = plot_one(t)
        dom, level, prob, algo = t
        print(f"  [plot] {dom}/{level}/{algo['name']}: {msg}")
        groups.setdefault((dom, level, prob), []).append(t)

    print("plotting multi-algorithm overlays...")
    for (dom, level, prob), group_tasks in groups.items():
        spread_msg, minmax_msg, maxdepth_msg = plot_multi(
            dom, level, prob, group_tasks)
        print(f"  [plot-multi] {dom}/{level}: spread={spread_msg} "
              f"minmax={minmax_msg} maxdepth={maxdepth_msg}")
        tri_msg, rect_msg = plot_multi_control(dom, level, prob, group_tasks)
        print(f"  [plot-multi] {dom}/{level}: control-triangle={tri_msg} "
              f"control-rectangle={rect_msg}")

    print("done.")


if __name__ == "__main__":
    OUT = None
    main()
