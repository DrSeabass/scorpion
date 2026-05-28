#!/usr/bin/env python3
"""
Adaptive / ratchet triangle on agile-strips, lmcount edition.

Sibling of 2026-05-26-B-adaptive-vs-static-vs-lama.py.  Same six dynamic-slope
configurations across the same three pruning settings, but the search-guidance
heuristic is the LAMA-style landmark-count (lmcount) heuristic instead of FF:

    landmark_sum(lm_reasonable_orders_hps(lm_rhw()),
                 transform=adapt_costs(one))

This is the same lmcount expression used elsewhere in this branch (see
../2026-05-multi-heuristic/) as the landmark heuristic inside LAMA's pair.

  adaptive-triangle-gonly: adaptive_triangle(eval=lmcount, anytime=true)
  adaptive-triangle-hmax:  adaptive_triangle(eval=lmcount,
                                             pruning_heuristic=hmax(),
                                             anytime=true)
  adaptive-triangle-lmcut: adaptive_triangle(eval=lmcount,
                                             pruning_heuristic=lmcut(),
                                             anytime=true)
  ratchet-triangle-gonly:  ratchet_triangle(eval=lmcount, anytime=true)
  ratchet-triangle-hmax:   ratchet_triangle(eval=lmcount,
                                            pruning_heuristic=hmax(),
                                            anytime=true)
  ratchet-triangle-lmcut:  ratchet_triangle(eval=lmcount,
                                            pruning_heuristic=lmcut(),
                                            anytime=true)

The static triangle-{gonly,hmax,lmcut} and lama-anytime baselines are still
NOT rerun here -- the FF-guided static triangle / LAMA baselines come from
../2026-05-triangle-vs-lama/.  If you need an lmcount-guided static triangle
baseline for an apples-to-apples comparison against these dynamic variants,
add it as its own config or stitch it in during analysis.

Environment variables (defaults shown; Tetralith overrides marked):
  ADAPTIVE_TRIANGLE_BUDGET                  default 5m local / 30m Tetralith
  ADAPTIVE_TRIANGLE_MEMORY                  default 8G
  ADAPTIVE_TRIANGLE_PROCESSES               local-only, default 2
  ADAPTIVE_TRIANGLE_INSTANCES_PER_DOMAIN    default 1 local / 0=all Tetralith
  ADAPTIVE_TRIANGLE_INSTANCE_STEP           default 3 local / 1 Tetralith
  ADAPTIVE_TRIANGLE_DOMAINS                 comma-separated; default all-discovered
  ADAPTIVE_TRIANGLE_WALL_TIME_FLOOR         Tetralith reservation floor, default 10:00:00
  ADAPTIVE_TRIANGLE_BENCHMARK_TARGET        default autoscale-agile-21.11-strips
  DOWNWARD_REPO                             default <repo root>
  DOWNWARD_BUILD                            default release
  TETRALITH_ACCOUNT                         default naiss2026-4-694

Default local scope is intentionally small (handful of instances, 5-minute
budget, 2 parallel processes) so a first-cut run completes in well under an
hour and uses ~16 GB peak (2 * 8G memory limit).  Increase via env vars when
you have headroom.
"""

import math
import os
import platform
import re
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
IS_TETRALITH = bool(re.match(r"tetralith\d+\.nsc\.liu\.se|n\d+", NODE))

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
TETRALITH_BENCHMARK_DIRS = {
    "autoscale-agile-21.11-strips":
        "/proj/mrlab_search_strategies/instances/autoscale-benchmarks/21.11-agile-strips",
}
BENCHMARK_DIRS = TETRALITH_BENCHMARK_DIRS if IS_TETRALITH else LOCAL_BENCHMARK_DIRS

BENCHMARK_TARGET_DEFAULT = "autoscale-agile-21.11-strips"
BENCHMARK_TARGET = os.environ.get(
    "ADAPTIVE_TRIANGLE_BENCHMARK_TARGET", BENCHMARK_TARGET_DEFAULT
)
if BENCHMARK_TARGET not in BENCHMARK_DIRS:
    raise ValueError(
        f"Unknown ADAPTIVE_TRIANGLE_BENCHMARK_TARGET={BENCHMARK_TARGET!r}; "
        f"valid: {sorted(BENCHMARK_DIRS)}"
    )
BENCHMARKS_DIR = os.path.expanduser(
    os.environ.get("DOWNWARD_BENCHMARKS", BENCHMARK_DIRS[BENCHMARK_TARGET])
)

# ----------------------------------------------------------------------------
# Scope (default tight for local sanity; Tetralith goes wider)
# ----------------------------------------------------------------------------
if IS_TETRALITH:
    BUDGET = os.environ.get("ADAPTIVE_TRIANGLE_BUDGET", "30m")
    MEMORY = os.environ.get("ADAPTIVE_TRIANGLE_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("ADAPTIVE_TRIANGLE_INSTANCES_PER_DOMAIN", "0")
    )
    INSTANCE_STEP = int(os.environ.get("ADAPTIVE_TRIANGLE_INSTANCE_STEP", "1"))
else:
    BUDGET = os.environ.get("ADAPTIVE_TRIANGLE_BUDGET", "5m")
    MEMORY = os.environ.get("ADAPTIVE_TRIANGLE_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("ADAPTIVE_TRIANGLE_INSTANCES_PER_DOMAIN", "1")
    )
    INSTANCE_STEP = int(os.environ.get("ADAPTIVE_TRIANGLE_INSTANCE_STEP", "3"))

LOCAL_PROCESSES = int(os.environ.get("ADAPTIVE_TRIANGLE_PROCESSES", "2"))


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class ChunkedTetralithEnvironment(project.TetralithEnvironment):
    MAX_TASKS = 504


if IS_TETRALITH:
    ENV = ChunkedTetralithEnvironment(
        setup=project.TetralithEnvironment.DEFAULT_SETUP,
        memory_per_cpu="9G",
        cpus_per_task=1,
        extra_options=f"#SBATCH --account={os.environ.get('TETRALITH_ACCOUNT', HEURISTIC_SEARCH_NAISS_ID)}",
    )
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
_domains_filter = os.environ.get("ADAPTIVE_TRIANGLE_DOMAINS")
if _domains_filter:
    requested = [d.strip() for d in _domains_filter.split(",") if d.strip()]
    missing = [d for d in requested if d not in SUITE]
    if missing:
        raise ValueError(
            f"ADAPTIVE_TRIANGLE_DOMAINS includes unknown: {missing}; "
            f"available: {SUITE}"
        )
    SUITE = requested
print(f"[adaptive-triangle-lmcount] {len(SUITE)} domains under {BENCHMARKS_DIR}")


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
        f"[adaptive-triangle-lmcount] {len(tasks)} tasks "
        f"({len(domains)} domains, ipd={instances_per_domain}, step={instance_step})"
    )
    return tasks


TASKS = build_limited_suite(BENCHMARKS_DIR, SUITE, INSTANCES_PER_DOMAIN, INSTANCE_STEP)


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------
TRANSLATE_OPTIONS = ["--translate-options"]

# LAMA-style landmark-count (lmcount) heuristic: same factory / cost transform
# used in ../2026-05-multi-heuristic/ for the LM side of the LAMA pair.
LMCOUNT = (
    "landmark_sum(lm_reasonable_orders_hps(lm_rhw()), "
    "transform=adapt_costs(one))"
)

SEARCH_TEMPLATES = {
    # As in the FF-guided sibling (2026-05-26-B-...), the static
    # triangle-{gonly,hmax,lmcut} and lama-anytime baselines are intentionally
    # NOT run here.  This experiment runs only the novel adaptive/ratchet
    # variants, with lmcount as the search-guidance heuristic.
    "adaptive-triangle-gonly": (
        f"adaptive_triangle(eval={LMCOUNT}, anytime=true)"
    ),
    "adaptive-triangle-hmax": (
        f"adaptive_triangle(eval={LMCOUNT}, pruning_heuristic=hmax(), "
        f"anytime=true)"
    ),
    "adaptive-triangle-lmcut": (
        f"adaptive_triangle(eval={LMCOUNT}, pruning_heuristic=lmcut(), "
        f"anytime=true)"
    ),
    "ratchet-triangle-gonly": (
        f"ratchet_triangle(eval={LMCOUNT}, anytime=true)"
    ),
    "ratchet-triangle-hmax": (
        f"ratchet_triangle(eval={LMCOUNT}, pruning_heuristic=hmax(), "
        f"anytime=true)"
    ),
    "ratchet-triangle-lmcut": (
        f"ratchet_triangle(eval={LMCOUNT}, pruning_heuristic=lmcut(), "
        f"anytime=true)"
    ),
}

# lama-anytime is run by the sibling ../2026-05-triangle-vs-lama/ experiment;
# omitted here to avoid rerunning a shared configuration (see note above).
ALIAS_CONFIGS = {}

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
for name, alias_opts in ALIAS_CONFIGS.items():
    CONFIGS.append((name, alias_opts))


# ----------------------------------------------------------------------------
# Slurm wall-clock reservation
# ----------------------------------------------------------------------------
# Lab packs ceil(num_runs / MAX_TASKS) runs into each array task and runs them
# back-to-back, so a task needs the FD search limit plus translate/validate
# overhead, times that many runs.  We compute that estimate (per-run padding,
# so it stays safe as runs/task grows) but reserve at least WALL_TIME_FLOOR.
# --time is a hard SIGKILL and losing a run to it is far worse than slightly
# over-reserving; the 10h floor still sits well under the 24h
# TetralithEnvironment default, so backfill stays happy.  Bump the floor via
# ADAPTIVE_TRIANGLE_WALL_TIME_FLOOR if a future scope pushes the estimate higher.
PER_RUN_OVERHEAD_SECONDS = 180  # translate + validate + lab; grid tops out ~2m


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


if IS_TETRALITH:
    NUM_RUNS = len(CONFIGS) * len(TASKS)
    RUNS_PER_TASK = math.ceil(NUM_RUNS / ENV.MAX_TASKS)
    EST_SECONDS = RUNS_PER_TASK * (
        _duration_to_seconds(BUDGET) + PER_RUN_OVERHEAD_SECONDS
    )
    FLOOR_SECONDS = _duration_to_seconds(
        os.environ.get("ADAPTIVE_TRIANGLE_WALL_TIME_FLOOR", "10:00:00")
    )
    WALL_SECONDS = max(EST_SECONDS, FLOOR_SECONDS)
    ENV.time_limit_per_task = _seconds_to_hms(WALL_SECONDS)
    print(
        f"[adaptive-triangle-lmcount] {NUM_RUNS} runs, {RUNS_PER_TASK} runs/array-task; "
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
    Attribute("expansions", function=arithmetic_mean),
    Attribute("generated", function=arithmetic_mean),
    "memory",
    project.EVALUATIONS_PER_TIME,
]


exp = Experiment(environment=ENV)

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

project.add_absolute_report(
    exp,
    attributes=ATTRIBUTES,
    filter=[project.add_evaluations_per_time],
)
project.add_compress_exp_dir_step(exp)

exp.run_steps()
