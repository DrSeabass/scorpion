#!/usr/bin/env python3
"""
Anytime triangle vs LAMA-2011 on agile-strips.

Compares four configurations on the agile-strips benchmark suite, with an
anytime-style time budget.  Triangle's f-pruning is fed an admissible
heuristic; LAMA is included via the standard alias.

  triangle-gonly:  triangle(eval=ff(), slope=48, anytime=true)
                   -- only the pre-generation g-prune
  triangle-hmax:   triangle(eval=ff(), pruning_heuristic=hmax(),
                            slope=48, anytime=true)
                   -- admissible h_max for f-pruning; cheap
  triangle-lmcut:  triangle(eval=ff(), pruning_heuristic=lmcut(),
                            slope=48, anytime=true)
                   -- admissible lmcut for f-pruning; tight but expensive
  lama-anytime:    seq-sat-lama-2011 alias

Modeled on ../../beam-smoke-classic.py for the cluster-aware shape.

Environment variables (defaults shown; Tetralith overrides marked):
  TRIANGLE_VS_LAMA_BUDGET                  default 5m local / 30m Tetralith
  TRIANGLE_VS_LAMA_MEMORY                  default 8G
  TRIANGLE_VS_LAMA_PROCESSES               local-only, default 2
  TRIANGLE_VS_LAMA_INSTANCES_PER_DOMAIN    default 1 local / 5 Tetralith
  TRIANGLE_VS_LAMA_INSTANCE_STEP           default 3 local / 1 Tetralith
  TRIANGLE_VS_LAMA_DOMAINS                 comma-separated; default all-discovered
  TRIANGLE_VS_LAMA_SLOPE                   default 48
  TRIANGLE_VS_LAMA_BENCHMARK_TARGET        default autoscale-agile-21.11-strips
  DOWNWARD_REPO                            default <repo root>
  DOWNWARD_BUILD                           default release
  TETRALITH_ACCOUNT                        default naiss2026-4-694

Default local scope is intentionally small (handful of instances, 5-minute
budget, 2 parallel processes) so a first-cut run completes in well under an
hour and uses ~16 GB peak (2 * 8G memory limit).  Increase via env vars when
you have headroom.
"""

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
    "TRIANGLE_VS_LAMA_BENCHMARK_TARGET", BENCHMARK_TARGET_DEFAULT
)
if BENCHMARK_TARGET not in BENCHMARK_DIRS:
    raise ValueError(
        f"Unknown TRIANGLE_VS_LAMA_BENCHMARK_TARGET={BENCHMARK_TARGET!r}; "
        f"valid: {sorted(BENCHMARK_DIRS)}"
    )
BENCHMARKS_DIR = os.path.expanduser(
    os.environ.get("DOWNWARD_BENCHMARKS", BENCHMARK_DIRS[BENCHMARK_TARGET])
)

# ----------------------------------------------------------------------------
# Scope (default tight for local sanity; Tetralith goes wider)
# ----------------------------------------------------------------------------
if IS_TETRALITH:
    BUDGET = os.environ.get("TRIANGLE_VS_LAMA_BUDGET", "30m")
    MEMORY = os.environ.get("TRIANGLE_VS_LAMA_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("TRIANGLE_VS_LAMA_INSTANCES_PER_DOMAIN", "5")
    )
    INSTANCE_STEP = int(os.environ.get("TRIANGLE_VS_LAMA_INSTANCE_STEP", "1"))
else:
    BUDGET = os.environ.get("TRIANGLE_VS_LAMA_BUDGET", "5m")
    MEMORY = os.environ.get("TRIANGLE_VS_LAMA_MEMORY", "8G")
    INSTANCES_PER_DOMAIN = int(
        os.environ.get("TRIANGLE_VS_LAMA_INSTANCES_PER_DOMAIN", "1")
    )
    INSTANCE_STEP = int(os.environ.get("TRIANGLE_VS_LAMA_INSTANCE_STEP", "3"))

LOCAL_PROCESSES = int(os.environ.get("TRIANGLE_VS_LAMA_PROCESSES", "2"))
SLOPE = int(os.environ.get("TRIANGLE_VS_LAMA_SLOPE", "48"))


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
_domains_filter = os.environ.get("TRIANGLE_VS_LAMA_DOMAINS")
if _domains_filter:
    requested = [d.strip() for d in _domains_filter.split(",") if d.strip()]
    missing = [d for d in requested if d not in SUITE]
    if missing:
        raise ValueError(
            f"TRIANGLE_VS_LAMA_DOMAINS includes unknown: {missing}; "
            f"available: {SUITE}"
        )
    SUITE = requested
print(f"[triangle-vs-lama] {len(SUITE)} domains under {BENCHMARKS_DIR}")


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
        f"[triangle-vs-lama] {len(tasks)} tasks "
        f"({len(domains)} domains, ipd={instances_per_domain}, step={instance_step})"
    )
    return tasks


TASKS = build_limited_suite(BENCHMARKS_DIR, SUITE, INSTANCES_PER_DOMAIN, INSTANCE_STEP)


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------
TRANSLATE_OPTIONS = ["--translate-options"]

TRIANGLE_TEMPLATES = {
    "triangle-gonly": (
        f"triangle(eval=ff(), slope={SLOPE}, anytime=true)"
    ),
    "triangle-hmax": (
        f"triangle(eval=ff(), pruning_heuristic=hmax(), "
        f"slope={SLOPE}, anytime=true)"
    ),
    "triangle-lmcut": (
        f"triangle(eval=ff(), pruning_heuristic=lmcut(), "
        f"slope={SLOPE}, anytime=true)"
    ),
}

ALIAS_CONFIGS = {
    "lama-anytime": ["--alias", "seq-sat-lama-2011"],
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
for name, search in TRIANGLE_TEMPLATES.items():
    CONFIGS.append(
        (name, TRANSLATE_OPTIONS + ["--search-options", "--search", search])
    )
for name, alias_opts in ALIAS_CONFIGS.items():
    CONFIGS.append((name, alias_opts))


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
