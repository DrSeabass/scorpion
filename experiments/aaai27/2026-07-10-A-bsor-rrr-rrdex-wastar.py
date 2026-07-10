#!/usr/bin/env python3
"""
Bounded-Suboptimal Rectangle Search vs RR-DXES and weighted A* on agile-strips.

Runs a pared roster of bounded-suboptimal searches on the Autoscale agile-strips
benchmark, in a first-solution (coverage) setting, over the suboptimality-weight
schedule. The roster is split into two independently-run SETS (AAAI27_SET), each
targeting its OWN data/eval directory so they can be merged after the fact --
the lab-idiomatic way to add algorithms without re-running the others:

  "min" -- minimum-viable set: rectangle search at the PRIMARY aspect (500) plus
  the RR-DXES baseline; writes data/<script>-min[-eval] --
    bsor-a{P}-w{W}:   bsor(eval=ff(), dist=[lmcut()], w=W, aspect=P, rr=false)
    rrr-a{P}-w{W}:    bsor(eval=ff(), dist=[lmcut()], w=W, aspect=P, rr=true)
    rrdex-w{W}:       rrdex(h=lmcut(), hhat=ff(), dhat=[lmcount], w=W, debias=true)

  "remainder" -- rectangle search at the SECONDARY aspect (1) plus weighted-A*;
  writes data/<script>-remainder[-eval] --
    bsor-a{S}-w{W}:   bsor(eval=ff(), dist=[lmcut()], w=W, aspect=S, rr=false)
    rrr-a{S}-w{W}:    bsor(eval=ff(), dist=[lmcut()], w=W, aspect=S, rr=true)
    wastar-w{W}:      eager_wastar([ff()], w=W)  # integer weights only

  "combined" -- owns no runs; merges the -min-eval and -remainder-eval dirs into
  data/<script>-combined-eval (via merge fetchers) and produces the combined
  report + bounded-suboptimal plot. Run after both sets have been fetched.

Workflow:
    AAAI27_SET=min       ... build start parse fetch <name>-min-abs plot
    AAAI27_SET=remainder ... build start parse fetch <name>-remainder-abs plot
    AAAI27_SET=combined  ... fetch-min fetch-remainder <name>-combined-abs plot

with W over WEIGHTS (default the 10-point schedule {1, 1.05, 1.1, 1.25, 1.5,
1.75, 2, 2.5, 3, 5}). In config names the weight's decimal point is written as
'_' (e.g. w=1.05 -> ...-w1_05).

BSOR ranking: eval = ff() (so f = g + ff orders the open list and the w-bound),
dist = lmcut() (the distance-to-go proxy ordering each rectangle depth bucket).
RRR is the rr=true (round-robin rectangle) variant. The RR-DXES baseline
(Fickert, Gu & Ruml, AAAI 2022) uses admissible h = lmcut() (so f = g + h drives
the w-bound, returned solution within w times optimal), inadmissible hhat = ff()
for its f^ ordering, and dhat = landmark_sum() under unit costs (an lmcount
action-count proxy); hhat/dhat are debiased by the mean single-step path error
(debias=true; Thayer, Dionne & Ruml 2011). The weighted-A* baseline is
integer-weight only, so it runs at the integer weights in the schedule.

(Earlier revisions also swept a single-heuristic RRR and EES/RR-D best-first
baselines; those were pared out. RR-D remains available as the `rrd` plugin.)

Modeled on ../2026-05-triangle-vs-lama-arrhenius/.  Same cluster-aware shape and
Slurm environment (NAISS Arrhenius); see project.ArrheniusEnvironment for the
cluster specifics (partition ``cpu``, account suffix ``-cpu``, ~1 GiB/core
proportional memory, no named QOS).  lab runs under the Python 3.9 venv on the
cluster; the submitting python (its absolute path, carried via PATH export)
becomes the python used inside the job scripts.

Environment variables (defaults shown; Arrhenius overrides marked):
  AAAI27_BUDGET                  default 5m local / 5m Arrhenius
  AAAI27_MEMORY                  default 8G (solver soft limit)
  AAAI27_PROCESSES               local-only, default 2
  AAAI27_INSTANCES_PER_DOMAIN    default 1 local / 0=all Arrhenius
  AAAI27_INSTANCE_STEP           default 3 local / 1 Arrhenius
  AAAI27_DOMAINS                 comma-separated; default all-discovered
  AAAI27_WEIGHTS                 comma-separated floats; default
                                         1,1.05,1.1,1.25,1.5,1.75,2,2.5,3,5
  AAAI27_SET                     min|remainder|combined; default min
  AAAI27_ASPECT_PRIMARY          rect aspect in min tier; default 500
  AAAI27_ASPECT_SECONDARY        extra rect aspect (full only); default 1
  AAAI27_H_EVAL                  BSOR/RRR + wA* eval; default ff()
  AAAI27_D_EVAL                  BSOR d (rect) evaluator; default lmcut()
  AAAI27_RRDEX_H_EVAL              rrdex admissible h; default lmcut()
  AAAI27_RRDEX_HHAT_EVAL           rrdex inadmissible hhat; default ff()
  AAAI27_RRDEX_DHAT_EVAL           rrdex distance-to-go dhat; default
                                         landmark_sum(...,adapt_costs(one)) (lmcount)
  AAAI27_WALL_TIME_FLOOR         Arrhenius reservation floor, default 10:00:00
  AAAI27_BENCHMARK_TARGET        default autoscale-agile-21.11-strips
  DOWNWARD_REPO                          default <repo root>
  DOWNWARD_BUILD                         default release
  ARRHENIUS_ACCOUNT                      default naiss2026-4-694-cpu
  ARRHENIUS_PARTITION                    default cpu
  ARRHENIUS_QOS                          default unset (no --qos emitted)
  ARRHENIUS_MAX_TASKS                    Slurm MaxArraySize; default 1000
  ARRHENIUS_VAL_BIN                      VAL bin dir prepended to PATH for
                                         --validate; default
                                         /home/thayer/search/shared/libs/VAL/bin

Default local scope is intentionally small (handful of instances, 5-minute
budget, 2 parallel processes) so a first-cut run completes quickly.  Increase
via env vars when you have headroom.  Config counts (10-weight schedule): the
min set is bsor + rrr (primary aspect) + rrdex = 3 * 10 = 30 configs; the
remainder set is bsor + rrr (secondary aspect) = 20 plus wA* at the 4 integer
weights = 24 configs; combined merges the two eval dirs (54 rows, no new runs).
"""

import math
import os
import platform
from pathlib import Path

import custom_parser
import project

from downward import suites
from downward.experiment import FastDownwardAlgorithm
from lab import tools
from lab.experiment import Experiment, Run
from lab.reports import Attribute, arithmetic_mean

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
DIR = Path(__file__).resolve().parent
SCORPION_REPO = DIR.parent.parent  # experiments/<name>/script.py -> repo root
NODE = platform.node()
IS_ARRHENIUS = (
    project.ArrheniusEnvironment.is_present()
    or os.environ.get("ARRHENIUS_FORCE") == "1"
)

LOCAL_REPO = os.path.expanduser(os.environ.get("DOWNWARD_REPO", str(SCORPION_REPO)))
LOCAL_BUILD = os.environ.get("DOWNWARD_BUILD", "release")
LOCAL_DRIVER = os.path.join(LOCAL_REPO, "fast-downward.py")

HEURISTIC_SEARCH_NAISS_ID = "naiss2026-4-694"

# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------
LOCAL_BENCHMARK_DIRS = {
    "autoscale-agile-21.11-strips":
        "~/research/autoscale-benchmarks/21.11-agile-strips",
}
ARRHENIUS_BENCHMARK_DIRS = {
    "autoscale-agile-21.11-strips":
        "/home/thayer/search/shared/instances/autoscale-benchmarks/21.11-agile-strips",
}
BENCHMARK_DIRS = ARRHENIUS_BENCHMARK_DIRS if IS_ARRHENIUS else LOCAL_BENCHMARK_DIRS

BENCHMARK_TARGET_DEFAULT = "autoscale-agile-21.11-strips"
BENCHMARK_TARGET = os.environ.get(
    "AAAI27_BENCHMARK_TARGET", BENCHMARK_TARGET_DEFAULT
)
if BENCHMARK_TARGET not in BENCHMARK_DIRS:
    raise ValueError(
        f"Unknown AAAI27_BENCHMARK_TARGET={BENCHMARK_TARGET!r}; "
        f"valid: {sorted(BENCHMARK_DIRS)}"
    )
BENCHMARKS_DIR = os.path.expanduser(
    os.environ.get("DOWNWARD_BENCHMARKS", BENCHMARK_DIRS[BENCHMARK_TARGET])
)

# ----------------------------------------------------------------------------
# Scope (default tight for local sanity; Arrhenius goes wider)
# ----------------------------------------------------------------------------
if IS_ARRHENIUS:
    BUDGET = os.environ.get("AAAI27_BUDGET", "5m")
    MEMORY = os.environ.get("AAAI27_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("AAAI27_INSTANCES_PER_DOMAIN", "0")
    )
    INSTANCE_STEP = int(os.environ.get("AAAI27_INSTANCE_STEP", "1"))
else:
    BUDGET = os.environ.get("AAAI27_BUDGET", "5m")
    MEMORY = os.environ.get("AAAI27_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("AAAI27_INSTANCES_PER_DOMAIN", "1")
    )
    INSTANCE_STEP = int(os.environ.get("AAAI27_INSTANCE_STEP", "3"))

LOCAL_PROCESSES = int(os.environ.get("AAAI27_PROCESSES", "2"))


def _float_list_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} parsed to an empty list")
    return values


def _weight_value(w):
    """Search-string literal for a weight: 1.0 -> '1', 1.05 -> '1.05'."""
    return "%g" % w


def _weight_tag(w):
    """Filesystem/report-safe token for a weight: 1.0 -> '1', 1.05 -> '1_05'."""
    return _weight_value(w).replace(".", "_")


# Suboptimality-bound schedule (co-authors' proposed 10-point ladder). bsor and
# rrdex take w as a double, so fractional weights pass through verbatim; the
# eager_wastar baseline is integer-only (see the wastar config below).
WEIGHTS = _float_list_env(
    "AAAI27_WEIGHTS", [1.0, 1.05, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0]
)
# The roster is split into two independently-run sets, each targeting its OWN
# data/eval directory, to be merged after the fact (AAAI27_SET):
#   "min"       -- minimum-viable: bsor + rrr at the primary aspect + rrdex
#   "remainder" -- bsor + rrr at the secondary aspect + weighted-A*
#   "combined"  -- no runs; merges the min and remainder eval dirs into one
#                  combined report + plot (run after both sets have been fetched)
# Typical workflow: run "min", run "remainder", then run "combined".
SET = os.environ.get("AAAI27_SET", "min").lower()
if SET not in ("min", "remainder", "combined"):
    raise ValueError(
        f"AAAI27_SET must be 'min', 'remainder', or 'combined', got {SET!r}"
    )
ASPECT_PRIMARY = int(os.environ.get("AAAI27_ASPECT_PRIMARY", "500"))
ASPECT_SECONDARY = int(os.environ.get("AAAI27_ASPECT_SECONDARY", "1"))
H_EVAL = os.environ.get("AAAI27_H_EVAL", "ff()")
D_EVAL = os.environ.get("AAAI27_D_EVAL", "lmcut()")
# Estimator triple for the rrdex baseline: an admissible h for the w-bound
# (f = g + h), an inadmissible cost estimate hhat for the f^ = g + hhat ordering,
# and a distance-to-go estimate dhat. Defaults follow the intended mapping:
# lmcut() (admissible), ff() (inadmissible), and lmcount (landmark_sum under unit
# costs) as the action-count proxy.
RRDEX_H_EVAL = os.environ.get("AAAI27_RRDEX_H_EVAL", "lmcut()")
RRDEX_HHAT_EVAL = os.environ.get("AAAI27_RRDEX_HHAT_EVAL", "ff()")
RRDEX_DHAT_EVAL = os.environ.get(
    "AAAI27_RRDEX_DHAT_EVAL",
    "landmark_sum(lm_rhw(), transform=adapt_costs(one), pref=false)",
)


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def _mem_to_gib(value):
    """Parse a memory string like '8G'/'512M' to whole GiB (rounded up)."""
    text = str(value).strip().upper()
    units = {"K": 1 / 1024 / 1024, "M": 1 / 1024, "G": 1, "T": 1024}
    if text and text[-1] in units:
        return math.ceil(float(text[:-1]) * units[text[-1]])
    return math.ceil(float(text) / (1024 ** 3))  # bare value = bytes


ARRHENIUS_RESERVE_GIB = _mem_to_gib(MEMORY) + 1
ARRHENIUS_ACCOUNT = os.environ.get(
    "ARRHENIUS_ACCOUNT", f"{HEURISTIC_SEARCH_NAISS_ID}-cpu"
)
ARRHENIUS_VAL_BIN = os.environ.get(
    "ARRHENIUS_VAL_BIN", "/home/thayer/search/shared/libs/VAL/bin"
)

if IS_ARRHENIUS:
    ENV = project.ArrheniusEnvironment(
        partition=os.environ.get("ARRHENIUS_PARTITION", "cpu"),
        qos=os.environ.get("ARRHENIUS_QOS") or "",
        memory_per_cpu="1G",
        cpus_per_task=ARRHENIUS_RESERVE_GIB,
        setup=f'export PATH="{ARRHENIUS_VAL_BIN}:$PATH"',
        extra_options=f"#SBATCH --account={ARRHENIUS_ACCOUNT}",
    )
    _max_tasks_override = os.environ.get("ARRHENIUS_MAX_TASKS")
    if _max_tasks_override:
        ENV.MAX_TASKS = int(_max_tasks_override)
else:
    ENV = project.LocalEnvironment(processes=LOCAL_PROCESSES)


# ----------------------------------------------------------------------------
# Suite
# ----------------------------------------------------------------------------
def list_domains(benchmarks_dir):
    if not os.path.isdir(benchmarks_dir):
        raise RuntimeError(f"Benchmarks dir not found: {benchmarks_dir}")
    return sorted(
        name for name in os.listdir(benchmarks_dir)
        if os.path.isdir(os.path.join(benchmarks_dir, name))
    )


SUITE = list_domains(BENCHMARKS_DIR)
_domains_filter = os.environ.get("AAAI27_DOMAINS")
if _domains_filter:
    requested = [d.strip() for d in _domains_filter.split(",") if d.strip()]
    missing = [d for d in requested if d not in SUITE]
    if missing:
        raise ValueError(
            f"AAAI27_DOMAINS includes unknown: {missing}; "
            f"available: {SUITE}"
        )
    SUITE = requested
print(f"[aaai27] {len(SUITE)} domains under {BENCHMARKS_DIR}")


def build_limited_suite(benchmarks_dir, domains, instances_per_domain, instance_step):
    if instance_step <= 0:
        raise ValueError("INSTANCE_STEP must be positive")
    tasks = []
    for domain in domains:
        domain_tasks = list(suites.Domain(benchmarks_dir, domain))
        if instances_per_domain <= 0:
            selected = domain_tasks[::instance_step]
        else:
            limited = domain_tasks[:instances_per_domain]
            selected = limited[::instance_step]
        tasks.extend(selected)
    print(
        f"[aaai27] {len(tasks)} tasks "
        f"({len(domains)} domains, ipd={instances_per_domain}, step={instance_step})"
    )
    return tasks


TASKS = build_limited_suite(BENCHMARKS_DIR, SUITE, INSTANCES_PER_DOMAIN, INSTANCE_STEP)


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------
TRANSLATE_OPTIONS = ["--translate-options"]

SEARCH_TEMPLATES = {}


def _add_rectangle(prefix, aspect, rr):
    """BSOR (rr=false) / RRR (rr=true) rectangle configs at a given aspect."""
    for weight in WEIGHTS:
        SEARCH_TEMPLATES[f"{prefix}-a{aspect}-w{_weight_tag(weight)}"] = (
            f"bsor(eval={H_EVAL}, dist=[{D_EVAL}], w={_weight_value(weight)}, "
            f"aspect={aspect}, rr={'true' if rr else 'false'})"
        )


def _add_rrdex():
    """RR-DXES baseline (Fickert, Gu & Ruml, AAAI 2022): dynamic expected-effort
    round robin. Admissible h, so the returned solution is within w * optimal;
    hhat/dhat debiased by path-based single-step error (Thayer, Dionne & Ruml
    2011)."""
    for weight in WEIGHTS:
        SEARCH_TEMPLATES[f"rrdex-w{_weight_tag(weight)}"] = (
            f"rrdex(h={RRDEX_H_EVAL}, hhat={RRDEX_HHAT_EVAL}, "
            f"dhat=[{RRDEX_DHAT_EVAL}], w={_weight_value(weight)}, debias=true)"
        )


def _add_wastar():
    """Weighted A* baseline. eager_wastar's weight is integer-only, so it runs
    only at the integer-valued weights in the schedule."""
    for weight in WEIGHTS:
        if not float(weight).is_integer():
            continue
        SEARCH_TEMPLATES[f"wastar-w{_weight_tag(weight)}"] = (
            f"eager_wastar([{H_EVAL}], w={int(weight)})"
        )


if SET == "min":
    # Minimum-viable set: rectangle search (BSOR and RRR) at the primary aspect,
    # plus the RR-DXES baseline.
    _add_rectangle("bsor", ASPECT_PRIMARY, rr=False)
    _add_rectangle("rrr", ASPECT_PRIMARY, rr=True)
    _add_rrdex()
elif SET == "remainder":
    # Remainder set: rectangle search at the secondary aspect, plus weighted-A*.
    _add_rectangle("bsor", ASPECT_SECONDARY, rr=False)
    _add_rectangle("rrr", ASPECT_SECONDARY, rr=True)
    _add_wastar()
# SET == "combined" has no configs of its own; it merges the two eval dirs.

DRIVER_OPTIONS_DICT = {
    "--validate": None,
    "--search-time-limit": BUDGET,
    "--overall-memory-limit": MEMORY,
}


def driver_options_to_list(opts):
    result = []
    for flag, value in opts.items():
        result.append(flag)
        if value is not None:
            result.append(value)
    return result


DRIVER_OPTIONS = driver_options_to_list(DRIVER_OPTIONS_DICT)

CONFIGS = []
for name, search in SEARCH_TEMPLATES.items():
    CONFIGS.append(
        (name, TRANSLATE_OPTIONS + ["--search-options", "--search", search])
    )
print(f"[aaai27] {len(CONFIGS)} configs")


# ----------------------------------------------------------------------------
# Slurm wall-clock reservation
# ----------------------------------------------------------------------------
PER_RUN_OVERHEAD_SECONDS = 180  # translate + validate + lab


def _duration_to_seconds(value):
    """Parse a duration ('30m', '1800s', '0.5h', '10:00:00', '1800') to seconds."""
    text = str(value).strip().lower()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        hours, minutes, secs = parts
        return hours * 3600 + minutes * 60 + secs
    units = {"s": 1, "m": 60, "h": 3600}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)  # bare value is seconds


def _seconds_to_hms(seconds):
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


if IS_ARRHENIUS and SET != "combined":
    NUM_RUNS = len(CONFIGS) * len(TASKS)
    RUNS_PER_TASK = math.ceil(NUM_RUNS / ENV.MAX_TASKS)
    EST_SECONDS = RUNS_PER_TASK * (
        _duration_to_seconds(BUDGET) + PER_RUN_OVERHEAD_SECONDS
    )
    FLOOR_SECONDS = _duration_to_seconds(
        os.environ.get("AAAI27_WALL_TIME_FLOOR", "10:00:00")
    )
    WALL_SECONDS = max(EST_SECONDS, FLOOR_SECONDS)
    ENV.time_limit_per_task = _seconds_to_hms(WALL_SECONDS)
    print(
        f"[aaai27] {NUM_RUNS} runs, {RUNS_PER_TASK} runs/array-task; "
        f"estimate {_seconds_to_hms(EST_SECONDS)}, "
        f"floor {_seconds_to_hms(FLOOR_SECONDS)} "
        f"-> requesting {ENV.time_limit_per_task} per task"
    )


# ----------------------------------------------------------------------------
# Run wrapper (bypasses CachedFastDownwardRevision: use the local repo's
# fast-downward.py as-is rather than cloning revisions)
# ----------------------------------------------------------------------------
class LocalFastDownwardRun(Run):
    def __init__(self, exp, algo, task, driver_path, build_name):
        super().__init__(exp)
        driver_options = algo.driver_options[:]
        if build_name:
            driver_options += ["--build", build_name]

        if task.domain_file is None:
            self.add_resource("task", task.problem_file, "task.sas", symlink=True)
            input_files = ["{task}"]
            driver_options = [opt for opt in driver_options if opt != "--validate"]
        else:
            self.add_resource("domain", task.domain_file, "domain.pddl", symlink=True)
            self.add_resource(
                "problem", task.problem_file, "problem.pddl", symlink=True
            )
            input_files = ["{domain}", "{problem}"]

        planner_command = [tools.get_python_executable()] + [driver_path] + driver_options
        if "--alias" in algo.component_options:
            planner_command += algo.component_options + input_files
        else:
            planner_command += input_files + algo.component_options

        self.add_command("planner", planner_command)
        self._set_properties(algo, driver_options, task, build_name)

    def _set_properties(self, algo, driver_options, task, build_name):
        self.set_property("algorithm", algo.name)
        self.set_property("repo", LOCAL_REPO)
        self.set_property("local_revision", "local")
        self.set_property("global_revision", "local")
        self.set_property("build_options", [build_name] if build_name else [])
        self.set_property("driver_options", driver_options)
        self.set_property("component_options", algo.component_options)
        for key, value in task.properties.items():
            self.set_property(key, value)
        self.set_property("experiment_name", self.experiment.name)
        self.set_property("id", [algo.name, task.domain, task.problem])


# ----------------------------------------------------------------------------
# Experiment construction
# ----------------------------------------------------------------------------
ATTRIBUTES = [
    "error",
    "run_dir",
    "search_start_time",
    "search_start_memory",
    "total_time",
    "search_time",
    "coverage",
    "cost",
    Attribute("expansions", function=arithmetic_mean),
    Attribute("generated", function=arithmetic_mean),
    Attribute("evaluations", function=arithmetic_mean),
    "reopened",
    "memory",
    project.EVALUATIONS_PER_TIME,
]


# Each set targets its own experiment/eval directory under data/, so the min and
# remainder runs never collide and can be merged after the fact. The "combined"
# set owns no runs -- it fetches (merges) the two sibling eval dirs.
DATA_DIR = DIR / "data"
STEM = Path(__file__).stem
MIN_EVAL = DATA_DIR / f"{STEM}-min-eval"
REMAINDER_EVAL = DATA_DIR / f"{STEM}-remainder-eval"

exp = Experiment(path=str(DATA_DIR / f"{STEM}-{SET}"), environment=ENV)

if SET == "combined":
    # No runs of our own: merge the min and remainder eval dirs into this
    # (combined) experiment's eval dir, then report and plot over the union.
    # Requires both sets to have been run + fetched first.
    exp.add_fetcher(str(MIN_EVAL), merge=True, name="fetch-min")
    exp.add_fetcher(str(REMAINDER_EVAL), merge=True, name="fetch-remainder")
else:
    for config_name, config in CONFIGS:
        algo = FastDownwardAlgorithm(config_name, None, DRIVER_OPTIONS, config)
        for task in TASKS:
            run = LocalFastDownwardRun(exp, algo, task, LOCAL_DRIVER, LOCAL_BUILD)
            exp.add_run(run)

    # BSOR and wA* each return a single final solution (BSOR reports improving
    # incumbents via its own log line, not repeated "Plan cost:" lines, and emits
    # no "Cumulative statistics:"), so the single-search parser applies and
    # captures the expansions/search_time/evaluations metrics we report on.
    exp.add_parser(project.FastDownwardExperiment.EXITCODE_PARSER)
    exp.add_parser(project.FastDownwardExperiment.TRANSLATOR_PARSER)
    exp.add_parser(project.FastDownwardExperiment.SINGLE_SEARCH_PARSER)
    exp.add_parser(custom_parser.get_parser())
    exp.add_parser(project.FastDownwardExperiment.PLANNER_PARSER)

    exp.add_step("build", exp.build)
    exp.add_step("start", exp.start_runs)
    exp.add_step("parse", exp.parse)
    exp.add_fetcher(name="fetch")

project.add_absolute_report(
    exp,
    attributes=ATTRIBUTES,
    filter=[project.add_evaluations_per_time],
)
# Bounded-suboptimal curves: coverage / effort / cost vs the weight schedule,
# one line per algorithm family. Run after `fetch` (or the merge fetchers).
project.add_bounded_suboptimal_plot_step(exp)
if SET != "combined":
    project.add_compress_exp_dir_step(exp)

exp.run_steps()
