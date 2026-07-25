#!/usr/bin/env python3
"""
Anytime Nonparametric A* (ana) baseline for the triangle-search
parameter-configuration paper.

Adds `ana` (van den Berg et al., AAAI 2011) alongside anytime-param-config.py's
triangle/rectangle/adaptive/ratchet/lama roster: a genuinely parameter-free
anytime search (no shape/slope/aspect knob at all), as opposed to the static
triangle/rectangle family or their online shape-adapting siblings.

Config (anytime):
    ana    ana(eval=lmcount, anytime=true, prune_with_h=false)

Heuristic and prune_with_h: ana's prune_with_h defaults to true, which is only
sound if eval is admissible (src/search/search_algorithms/plugin_ana.cc). This
roster ranks on the same lm-count heuristic as every other config in the
paper (landmark_sum over reasonable-orders RHW landmarks, costs adapted to
one) for a heuristic-controlled comparison -- but lm-count is NOT admissible,
so prune_with_h is set to false here. That trades away ana's pruning
efficiency and its guarantee of converging to the optimal solution; it does
not affect anytime-param-config.py's own configs, which don't use ana.

Same TSPC_SET shard/combined workflow as anytime-param-config.py, reusing the
identical DOMAIN_SHARDS partition from shard_domains.py so the two experiments'
domain groupings can never drift apart:

    TSPC_SET=shard1    ... --all   # run shard1; writes its own report + plot
    TSPC_SET=combined  ... --all   # regenerate over {shard1}
    TSPC_SET=shard2    ... --all   # run shard2
    TSPC_SET=combined  ... --all   # regenerate over {shard1, shard2}
    ...  shard3, shard4, re-running combined after each ...

This script's own "combined" eval dir (data/anytime-param-config-ana-
combined-eval) is what anytime-param-config.py's "combined" branch merges in
as a single supplemental fetcher -- run all four shards and this script's own
combined step before regenerating the master combined report.

Environment variables (defaults shown; Arrhenius overrides marked):
  TSPC_SET                     all|shard1..4|combined; default all
  TSPC_MERGE_SETS              combined-only; shards to merge, default=existing
  TSPC_BUDGET                  default 5m local / 15m Arrhenius
  TSPC_MEMORY                  default 8G (solver soft limit)
  TSPC_PROCESSES               local-only, default 2
  TSPC_INSTANCES_PER_DOMAIN    default 0=all (whole suite everywhere)
  TSPC_INSTANCE_STEP           default 1 (every instance everywhere)
  TSPC_WALL_TIME_FLOOR         Arrhenius reservation floor, default 10:00:00
  TSPC_BENCHMARK_TARGET        default autoscale-agile-21.11-strips
  DOWNWARD_REPO                default <repo root>
  DOWNWARD_BUILD               default release
  ARRHENIUS_ACCOUNT            default naiss2026-4-694-cpu
  ARRHENIUS_PARTITION          default cpu
  ARRHENIUS_QOS                default unset (no --qos emitted)
  ARRHENIUS_MAX_TASKS          Slurm MaxArraySize; default 1000
  ARRHENIUS_VAL_BIN            VAL bin dir prepended to PATH for --validate
"""

import math
import os
import platform
from pathlib import Path

import custom_parser
import project
from shard_domains import DOMAIN_SHARDS, SHARD_NAMES, check_partition

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
BENCHMARK_TARGET = os.environ.get("TSPC_BENCHMARK_TARGET", BENCHMARK_TARGET_DEFAULT)
if BENCHMARK_TARGET not in BENCHMARK_DIRS:
    raise ValueError(
        f"Unknown TSPC_BENCHMARK_TARGET={BENCHMARK_TARGET!r}; "
        f"valid: {sorted(BENCHMARK_DIRS)}"
    )
BENCHMARKS_DIR = os.path.expanduser(
    os.environ.get("DOWNWARD_BENCHMARKS", BENCHMARK_DIRS[BENCHMARK_TARGET])
)

# ----------------------------------------------------------------------------
# Scope (default tight for local sanity; Arrhenius goes wider)
# ----------------------------------------------------------------------------
if IS_ARRHENIUS:
    BUDGET = os.environ.get("TSPC_BUDGET", "15m")
    MEMORY = os.environ.get("TSPC_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(os.environ.get("TSPC_INSTANCES_PER_DOMAIN", "0"))
    INSTANCE_STEP = int(os.environ.get("TSPC_INSTANCE_STEP", "1"))
else:
    BUDGET = os.environ.get("TSPC_BUDGET", "5m")
    MEMORY = os.environ.get("TSPC_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(os.environ.get("TSPC_INSTANCES_PER_DOMAIN", "0"))
    INSTANCE_STEP = int(os.environ.get("TSPC_INSTANCE_STEP", "1"))

LOCAL_PROCESSES = int(os.environ.get("TSPC_PROCESSES", "2"))


def _str_list(env_name, default):
    raw = os.environ.get(env_name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


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
        setup=(
            f'export PATH="{ARRHENIUS_VAL_BIN}:$PATH"\n'
            f'export TSPC_SET="{os.environ.get("TSPC_SET", "all").lower()}"'
        ),
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


ALL_DOMAINS = list_domains(BENCHMARKS_DIR)

# ---- Run sets: identical shard/combined mechanism as anytime-param-config.py
SET = os.environ.get("TSPC_SET", "all").lower()
VALID_SETS = ["all", "combined"] + SHARD_NAMES
if SET not in VALID_SETS:
    raise ValueError(f"TSPC_SET={SET!r} must be one of {VALID_SETS}")
IS_COMBINED = SET == "combined"

_dupes, _unassigned, _unknown_shard = check_partition(ALL_DOMAINS)
if _dupes or _unassigned or _unknown_shard:
    _msg = (
        f"DOMAIN_SHARDS is not a clean partition of the {len(ALL_DOMAINS)} "
        f"discovered domains: dupes={_dupes}, unassigned={_unassigned}, "
        f"unknown={_unknown_shard}"
    )
    if SET in SHARD_NAMES or IS_COMBINED:
        raise ValueError(_msg)
    print(f"[param-config-ana] WARNING: {_msg}")

if SET == "all":
    SUITE = list(ALL_DOMAINS)
elif SET in DOMAIN_SHARDS:
    SUITE = list(DOMAIN_SHARDS[SET])
else:  # combined: no runs of our own
    SUITE = []

_domains_filter = os.environ.get("TSPC_DOMAINS")
if _domains_filter and not IS_COMBINED:
    SUITE = _str_list("TSPC_DOMAINS", "")

_unknown_suite = [d for d in SUITE if d not in ALL_DOMAINS]
if _unknown_suite:
    raise ValueError(
        f"set={SET!r} references unknown domains: {_unknown_suite}; "
        f"available: {ALL_DOMAINS}"
    )
print(f"[param-config-ana] set={SET}: {len(SUITE)} domains under {BENCHMARKS_DIR}")


def build_limited_suite(benchmarks_dir, domains, instances_per_domain, instance_step):
    if instance_step <= 0:
        raise ValueError("INSTANCE_STEP must be positive")
    tasks = []
    for domain in domains:
        domain_tasks = list(suites.Domain(benchmarks_dir, domain))
        if instances_per_domain <= 0:
            selected = domain_tasks[::instance_step]
        else:
            selected = domain_tasks[:instances_per_domain][::instance_step]
        tasks.extend(selected)
    print(
        f"[param-config-ana] {len(tasks)} tasks "
        f"({len(domains)} domains, ipd={instances_per_domain}, step={instance_step})"
    )
    return tasks


TASKS = [] if IS_COMBINED else build_limited_suite(
    BENCHMARKS_DIR, SUITE, INSTANCES_PER_DOMAIN, INSTANCE_STEP
)


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------
TRANSLATE_OPTIONS = ["--translate-options"]

# lm-count ranking heuristic: landmark_sum over reasonable-orders RHW
# landmarks, costs adapted to one -- matches anytime-param-config.py exactly.
LMCOUNT = (
    "landmark_sum(lm_reasonable_orders_hps(lm_rhw()), transform=adapt_costs(one))"
)

# ana (Anytime Nonparametric A*, van den Berg et al. AAAI 2011): no shape
# parameter at all. prune_with_h=false because lm-count is inadmissible (see
# module docstring); prune_with_h=true would be unsound with this heuristic.
SEARCH_TEMPLATES = {
    "ana": f"ana(eval={LMCOUNT}, anytime=true, prune_with_h=false)",
}

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
print(f"[param-config-ana] {len(CONFIGS)} configs: {[c[0] for c in CONFIGS]}")


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


if IS_ARRHENIUS and not IS_COMBINED:
    NUM_RUNS = len(CONFIGS) * len(TASKS)
    RUNS_PER_TASK = math.ceil(NUM_RUNS / ENV.MAX_TASKS)
    EST_SECONDS = RUNS_PER_TASK * (
        _duration_to_seconds(BUDGET) + PER_RUN_OVERHEAD_SECONDS
    )
    FLOOR_SECONDS = _duration_to_seconds(
        os.environ.get("TSPC_WALL_TIME_FLOOR", "10:00:00")
    )
    WALL_SECONDS = max(EST_SECONDS, FLOOR_SECONDS)
    ENV.time_limit_per_task = _seconds_to_hms(WALL_SECONDS)
    print(
        f"[param-config-ana] {NUM_RUNS} runs, {RUNS_PER_TASK} runs/array-task; "
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
    "cost:all",
    project.SCORE_ANYTIME,
    Attribute("expansions", function=arithmetic_mean),
    Attribute("generated", function=arithmetic_mean),
    "memory",
    project.EVALUATIONS_PER_TIME,
]

# Same per-SET experiment/eval dir layout as anytime-param-config.py, so the
# shard/combined workflow is identical.
DATA_DIR = DIR / "data"
STEM = Path(__file__).stem
exp = Experiment(path=str(DATA_DIR / f"{STEM}-{SET}"), environment=ENV)

if IS_COMBINED:
    def _shard_eval_dir(s):
        return DATA_DIR / f"{STEM}-{s}-eval"

    _existing = [s for s in SHARD_NAMES if _shard_eval_dir(s).is_dir()]
    MERGE_SETS = _str_list("TSPC_MERGE_SETS", ",".join(_existing))
    _bad_merge = [s for s in MERGE_SETS if s not in SHARD_NAMES]
    if _bad_merge:
        raise ValueError(
            f"TSPC_MERGE_SETS includes non-shard sets {_bad_merge}; "
            f"valid shards: {SHARD_NAMES}"
        )
    if not MERGE_SETS:
        raise ValueError(
            f"combined: no shard eval dirs found under {DATA_DIR}. Run a shard "
            f"first (e.g. TSPC_SET=shard1 ... --all) or set TSPC_MERGE_SETS."
        )
    _no_dir = [s for s in MERGE_SETS if not _shard_eval_dir(s).is_dir()]
    if _no_dir:
        raise ValueError(
            f"combined: requested shards {_no_dir} have no eval dir under "
            f"{DATA_DIR}; run them first."
        )
    print(f"[param-config-ana] combined merges: {MERGE_SETS}")
    for s in MERGE_SETS:
        exp.add_fetcher(str(_shard_eval_dir(s)), merge=True, name=f"fetch-{s}")
else:
    for config_name, config in CONFIGS:
        algo = FastDownwardAlgorithm(config_name, None, DRIVER_OPTIONS, config)
        for task in TASKS:
            run = LocalFastDownwardRun(exp, algo, task, LOCAL_DRIVER, LOCAL_BUILD)
            exp.add_run(run)

    exp.add_parser(project.FastDownwardExperiment.EXITCODE_PARSER)
    exp.add_parser(project.FastDownwardExperiment.TRANSLATOR_PARSER)
    exp.add_parser(project.FastDownwardExperiment.ANYTIME_SEARCH_PARSER)
    exp.add_parser(custom_parser.get_parser())
    exp.add_parser(project.FastDownwardExperiment.PLANNER_PARSER)

    exp.add_step("build", exp.build)
    exp.add_step("start", exp.start_runs)
    exp.add_step("parse", exp.parse)
    exp.add_fetcher(name="fetch")

_anytime_score = project.AnytimeQualityFilter(max_time=_duration_to_seconds(BUDGET))
project.add_absolute_report(
    exp,
    attributes=ATTRIBUTES,
    filter=[
        _anytime_score.collect,
        _anytime_score.add_score,
        project.add_evaluations_per_time,
    ],
)
project.add_anytime_profile_plot_step(exp, max_time=_duration_to_seconds(BUDGET))
if not IS_COMBINED:
    project.add_compress_exp_dir_step(exp)

exp.run_steps()
