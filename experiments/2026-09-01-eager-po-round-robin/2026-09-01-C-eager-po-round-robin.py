#!/usr/bin/env python3
"""Eager preferred-operator x scheduling ablation, slope 48, first solution."""

import math
import os
import platform
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO = DIR.parent.parent
sys.path.insert(
    0, str(DIR.parent / "2026-08-11-icaps27-lama-closest-competitors")
)

import custom_parser  # noqa: E402
import project  # noqa: E402
from downward import suites  # noqa: E402
from downward.experiment import FastDownwardAlgorithm  # noqa: E402
from lab import tools  # noqa: E402
from lab.experiment import Experiment, Run  # noqa: E402
from lab.reports import Attribute, arithmetic_mean  # noqa: E402

SLOPE = 48
forced = os.environ.get("EPORR_CLUSTER", "").lower()
IS_TETRALITH = forced == "tetralith" or (
    not forced
    and (project.TetralithEnvironment.is_present() or
         "tetralith" in platform.node().lower())
)
CLUSTER = "tetralith" if IS_TETRALITH else "local"
LOCAL_REPO = os.path.expanduser(os.environ.get("DOWNWARD_REPO", str(REPO)))
LOCAL_DRIVER = str(Path(LOCAL_REPO) / "fast-downward.py")
LOCAL_BUILD = os.environ.get("DOWNWARD_BUILD", "release")
BENCHMARKS = os.path.expanduser(os.environ.get(
    "DOWNWARD_BENCHMARKS",
    "/proj/mrlab_search_strategies/instances/autoscale-benchmarks/21.11-agile-strips"
    if IS_TETRALITH else "~/research/autoscale-benchmarks/21.11-agile-strips",
))
BUDGET = os.environ.get("EPORR_BUDGET", "15m" if IS_TETRALITH else "180s")
MEMORY = os.environ.get("EPORR_MEMORY", "6G" if IS_TETRALITH else "4G")
STEP = int(os.environ.get("EPORR_INSTANCE_STEP", "1" if IS_TETRALITH else "10"))
IPD = int(os.environ.get("EPORR_INSTANCES_PER_DOMAIN", "0"))


def mem_gib(value):
    text = value.upper()
    return math.ceil(float(text[:-1])) if text.endswith("G") else 7


def duration_seconds(value):
    units = {"s": 1, "m": 60, "h": 3600}
    return int(value[:-1]) * units[value[-1].lower()]


def hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


if IS_TETRALITH:
    ENV = project.TetralithEnvironment(
        memory_per_cpu=f"{mem_gib(MEMORY) + 1}G",
        cpus_per_task=1,
        extra_options="#SBATCH --account=" + os.environ.get(
            "TETRALITH_ACCOUNT", "naiss2026-4-694"
        ),
    )
else:
    ENV = project.LocalEnvironment(
        processes=int(os.environ.get("EPORR_PROCESSES", "2"))
    )

if not os.path.isdir(BENCHMARKS):
    raise RuntimeError(f"Benchmarks directory not found: {BENCHMARKS}")
domains = sorted(
    name for name in os.listdir(BENCHMARKS)
    if os.path.isdir(os.path.join(BENCHMARKS, name))
)
if os.environ.get("EPORR_DOMAINS"):
    domains = [x.strip() for x in os.environ["EPORR_DOMAINS"].split(",")]
index_text = os.environ.get("EPORR_INSTANCE_INDEX")
index = int(index_text) if index_text else None
TASKS = []
for domain in domains:
    domain_tasks = list(suites.Domain(BENCHMARKS, domain))
    if index:
        if len(domain_tasks) >= index:
            TASKS.append(domain_tasks[index - 1])
    elif IPD:
        TASKS.extend(domain_tasks[:IPD:STEP])
    else:
        TASKS.extend(domain_tasks[::STEP])

LM = "landmark_sum(lm_reasonable_orders_hps(lm_rhw()), pref=true)"


def pair(search):
    return f"let(hff, ff(), let(hlm, {LM}, {search}))"


SEARCHES = {
    "eager-sweep-no-po": pair(
        "round_robin_triangle(evals=[hff, hlm], slope=48, schedule=sweep)"
    ),
    "eager-depth-no-po": pair(
        "round_robin_triangle(evals=[hff, hlm], slope=48, schedule=depth)"
    ),
    "eager-sweep-po": pair(
        "round_robin_triangle(evals=[hff, hlm], preferred_evals=[hff, hlm], "
        "slope=48, schedule=sweep)"
    ),
    "eager-depth-po": pair(
        "round_robin_triangle(evals=[hff, hlm], preferred_evals=[hff, hlm], "
        "slope=48, schedule=depth)"
    ),
}
if os.environ.get("EPORR_CONFIGS"):
    selected = [x.strip() for x in os.environ["EPORR_CONFIGS"].split(",")]
    SEARCHES = {name: SEARCHES[name] for name in selected}

DRIVER_OPTIONS = [
    "--validate", "--search-time-limit", BUDGET,
    "--overall-memory-limit", MEMORY,
]


class FDRun(Run):
    def __init__(self, exp, algo, task):
        super().__init__(exp)
        opts = algo.driver_options[:] + ["--build", LOCAL_BUILD]
        self.add_resource("domain", task.domain_file, "domain.pddl", symlink=True)
        self.add_resource("problem", task.problem_file, "problem.pddl", symlink=True)
        self.add_command(
            "planner",
            [tools.get_python_executable(), LOCAL_DRIVER]
            + opts + ["{domain}", "{problem}"] + algo.component_options,
        )
        self.set_property("algorithm", algo.name)
        self.set_property("driver_options", opts)
        self.set_property("component_options", algo.component_options)
        self.set_property("build_options", [LOCAL_BUILD])
        self.set_property("local_revision", "local")
        self.set_property("global_revision", "local")
        self.set_property("experiment_name", Path(__file__).stem)
        self.set_property("repo", LOCAL_REPO)
        for key, value in task.properties.items():
            self.set_property(key, value)
        self.set_property("id", [algo.name, task.domain, task.problem])


stem = Path(__file__).stem + os.environ.get("EPORR_EXPERIMENT_SUFFIX", "")
exp = Experiment(path=str(DIR / "data" / stem), environment=ENV)
for name, search in SEARCHES.items():
    component = ["--translate-options", "--search-options", "--search", search]
    algo = FastDownwardAlgorithm(name, None, DRIVER_OPTIONS, component)
    for task in TASKS:
        exp.add_run(FDRun(exp, algo, task))

if IS_TETRALITH:
    runs_per_task = math.ceil(len(SEARCHES) * len(TASKS) / ENV.MAX_TASKS)
    seconds_per_run = duration_seconds(BUDGET) + 3 * 60
    ENV.time_limit_per_task = hms(max(10 * 3600, runs_per_task * seconds_per_run))

print(
    f"Environment: {CLUSTER}; tasks: {len(TASKS)}; "
    f"configurations: {', '.join(SEARCHES)}"
)

exp.add_parser(project.FastDownwardExperiment.EXITCODE_PARSER)
exp.add_parser(project.FastDownwardExperiment.TRANSLATOR_PARSER)
exp.add_parser(project.FastDownwardExperiment.ANYTIME_SEARCH_PARSER)
exp.add_parser(project.FastDownwardExperiment.PLANNER_PARSER)
exp.add_parser(custom_parser.get_parser())
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")
project.add_absolute_report(
    exp,
    attributes=[
        "error", "run_dir", "total_time", "search_time", "coverage", "cost",
        Attribute("expansions", function=arithmetic_mean),
        Attribute("evaluated", function=arithmetic_mean),
        Attribute("generated", function=arithmetic_mean),
        "memory", project.EVALUATIONS_PER_TIME,
    ],
    filter=[project.add_evaluations_per_time],
)
project.add_compress_exp_dir_step(exp)
exp.run_steps()
