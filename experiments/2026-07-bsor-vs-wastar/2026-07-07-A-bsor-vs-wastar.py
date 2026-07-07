#!/usr/bin/env python3
"""
Bounded-Suboptimal Rectangle Search vs weighted A* on agile-strips.

Runs a grid of bounded-suboptimal rectangle search (BSOR, Algorithms 4/5 of
Thomas et al., HSDIP 2026) configurations against weighted A* as the baseline,
on the Autoscale agile-strips benchmark, in a first-solution (coverage) setting.

BSOR ranking: h = ff() (so f = g + ff, orders the open list and the w-bound),
d = lmcut() (the distance-to-go proxy that orders each rectangle depth bucket,
passed via dist=[...]).  Both BSOR (rr=false) and Round-Robin Rectangle Search
(RRR, rr=true) are swept over aspect ratios and suboptimality weights.  The
weighted A* baseline uses ff() with the same integer weights.

  bsor-a{A}-w{W}:     bsor(eval=ff(), dist=[lmcut()], w=W, aspect=A, rr=false)
  rrr-a{A}-w{W}:      bsor(eval=ff(), dist=[lmcut()], w=W, aspect=A, rr=true)
  rrr-ff-a{A}-w{W}:   bsor(eval=ff(), w=W, aspect=A, rr=true)  # ff for h and d
  wastar-w{W}:        eager_wastar([ff()], w=W)

with A over ASPECTS (default {1, 500}) and W over WEIGHTS (default {1, 2, 3, 5}).

Modeled on ../2026-05-triangle-vs-lama-arrhenius/.  Same cluster-aware shape and
Slurm environment (NAISS Arrhenius); see project.ArrheniusEnvironment for the
cluster specifics (partition ``cpu``, account suffix ``-cpu``, ~1 GiB/core
proportional memory, no named QOS).  lab runs under the Python 3.9 venv on the
cluster; the submitting python (its absolute path, carried via PATH export)
becomes the python used inside the job scripts.

Environment variables (defaults shown; Arrhenius overrides marked):
  BSOR_VS_WASTAR_BUDGET                  default 5m local / 5m Arrhenius
  BSOR_VS_WASTAR_MEMORY                  default 8G (solver soft limit)
  BSOR_VS_WASTAR_PROCESSES               local-only, default 2
  BSOR_VS_WASTAR_INSTANCES_PER_DOMAIN    default 1 local / 0=all Arrhenius
  BSOR_VS_WASTAR_INSTANCE_STEP           default 3 local / 1 Arrhenius
  BSOR_VS_WASTAR_DOMAINS                 comma-separated; default all-discovered
  BSOR_VS_WASTAR_WEIGHTS                 comma-separated ints; default 1,2,3,5
  BSOR_VS_WASTAR_ASPECTS                 comma-separated ints; default 1,500
  BSOR_VS_WASTAR_H_EVAL                  BSOR h + wA* evaluator; default ff()
  BSOR_VS_WASTAR_D_EVAL                  BSOR d (rect) evaluator; default lmcut()
  BSOR_VS_WASTAR_WALL_TIME_FLOOR         Arrhenius reservation floor, default 10:00:00
  BSOR_VS_WASTAR_BENCHMARK_TARGET        default autoscale-agile-21.11-strips
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
via env vars when you have headroom.  Note the config grid is large (with the
defaults: 2 aspects * 4 weights * 2 rr modes = 16 ff/lmcut rectangle configs,
plus 2 aspects * 4 weights = 8 single-heuristic (ff-only) RRR configs, plus 4
wA* = 28), so widen the instance scope with the grid in mind.
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
    "BSOR_VS_WASTAR_BENCHMARK_TARGET", BENCHMARK_TARGET_DEFAULT
)
if BENCHMARK_TARGET not in BENCHMARK_DIRS:
    raise ValueError(
        f"Unknown BSOR_VS_WASTAR_BENCHMARK_TARGET={BENCHMARK_TARGET!r}; "
        f"valid: {sorted(BENCHMARK_DIRS)}"
    )
BENCHMARKS_DIR = os.path.expanduser(
    os.environ.get("DOWNWARD_BENCHMARKS", BENCHMARK_DIRS[BENCHMARK_TARGET])
)

# ----------------------------------------------------------------------------
# Scope (default tight for local sanity; Arrhenius goes wider)
# ----------------------------------------------------------------------------
if IS_ARRHENIUS:
    BUDGET = os.environ.get("BSOR_VS_WASTAR_BUDGET", "5m")
    MEMORY = os.environ.get("BSOR_VS_WASTAR_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("BSOR_VS_WASTAR_INSTANCES_PER_DOMAIN", "0")
    )
    INSTANCE_STEP = int(os.environ.get("BSOR_VS_WASTAR_INSTANCE_STEP", "1"))
else:
    BUDGET = os.environ.get("BSOR_VS_WASTAR_BUDGET", "5m")
    MEMORY = os.environ.get("BSOR_VS_WASTAR_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("BSOR_VS_WASTAR_INSTANCES_PER_DOMAIN", "1")
    )
    INSTANCE_STEP = int(os.environ.get("BSOR_VS_WASTAR_INSTANCE_STEP", "3"))

LOCAL_PROCESSES = int(os.environ.get("BSOR_VS_WASTAR_PROCESSES", "2"))


def _int_list_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} parsed to an empty list")
    return values


WEIGHTS = _int_list_env("BSOR_VS_WASTAR_WEIGHTS", [1, 2, 3, 5])
ASPECTS = _int_list_env("BSOR_VS_WASTAR_ASPECTS", [1, 500])
H_EVAL = os.environ.get("BSOR_VS_WASTAR_H_EVAL", "ff()")
D_EVAL = os.environ.get("BSOR_VS_WASTAR_D_EVAL", "lmcut()")


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
_domains_filter = os.environ.get("BSOR_VS_WASTAR_DOMAINS")
if _domains_filter:
    requested = [d.strip() for d in _domains_filter.split(",") if d.strip()]
    missing = [d for d in requested if d not in SUITE]
    if missing:
        raise ValueError(
            f"BSOR_VS_WASTAR_DOMAINS includes unknown: {missing}; "
            f"available: {SUITE}"
        )
    SUITE = requested
print(f"[bsor-vs-wastar] {len(SUITE)} domains under {BENCHMARKS_DIR}")


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
        f"[bsor-vs-wastar] {len(tasks)} tasks "
        f"({len(domains)} domains, ipd={instances_per_domain}, step={instance_step})"
    )
    return tasks


TASKS = build_limited_suite(BENCHMARKS_DIR, SUITE, INSTANCES_PER_DOMAIN, INSTANCE_STEP)


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------
TRANSLATE_OPTIONS = ["--translate-options"]

SEARCH_TEMPLATES = {}
# BSOR and RRR: rectangle family over aspect ratios and weights.
for aspect in ASPECTS:
    for weight in WEIGHTS:
        for rr in (False, True):
            prefix = "rrr" if rr else "bsor"
            name = f"{prefix}-a{aspect}-w{weight}"
            SEARCH_TEMPLATES[name] = (
                f"bsor(eval={H_EVAL}, dist=[{D_EVAL}], w={weight}, "
                f"aspect={aspect}, rr={'true' if rr else 'false'})"
            )
# Single-heuristic RRR (BSRRR): ff() alone for both h and d (dist omitted, so it
# defaults to eval), matching wA*'s single-evaluator setup for a cleaner
# expansions comparison.
for aspect in ASPECTS:
    for weight in WEIGHTS:
        SEARCH_TEMPLATES[f"rrr-ff-a{aspect}-w{weight}"] = (
            f"bsor(eval={H_EVAL}, w={weight}, aspect={aspect}, rr=true)"
        )
# Weighted A* baseline (integer weights; shares the ff() evaluator).
for weight in WEIGHTS:
    SEARCH_TEMPLATES[f"wastar-w{weight}"] = f"eager_wastar([{H_EVAL}], w={weight})"

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
print(f"[bsor-vs-wastar] {len(CONFIGS)} configs")


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


if IS_ARRHENIUS:
    NUM_RUNS = len(CONFIGS) * len(TASKS)
    RUNS_PER_TASK = math.ceil(NUM_RUNS / ENV.MAX_TASKS)
    EST_SECONDS = RUNS_PER_TASK * (
        _duration_to_seconds(BUDGET) + PER_RUN_OVERHEAD_SECONDS
    )
    FLOOR_SECONDS = _duration_to_seconds(
        os.environ.get("BSOR_VS_WASTAR_WALL_TIME_FLOOR", "10:00:00")
    )
    WALL_SECONDS = max(EST_SECONDS, FLOOR_SECONDS)
    ENV.time_limit_per_task = _seconds_to_hms(WALL_SECONDS)
    print(
        f"[bsor-vs-wastar] {NUM_RUNS} runs, {RUNS_PER_TASK} runs/array-task; "
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


exp = Experiment(environment=ENV)

for config_name, config in CONFIGS:
    algo = FastDownwardAlgorithm(config_name, None, DRIVER_OPTIONS, config)
    for task in TASKS:
        run = LocalFastDownwardRun(exp, algo, task, LOCAL_DRIVER, LOCAL_BUILD)
        exp.add_run(run)

# BSOR and wA* each return a single final solution (BSOR reports improving
# incumbents via its own log line, not repeated "Plan cost:" lines, and emits no
# "Cumulative statistics:"), so the single-search parser applies and captures the
# expansions/search_time/evaluations metrics the HSDIP paper reports on.
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
project.add_compress_exp_dir_step(exp)

exp.run_steps()
