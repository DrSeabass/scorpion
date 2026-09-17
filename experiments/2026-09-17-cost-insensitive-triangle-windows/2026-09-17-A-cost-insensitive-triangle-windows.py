#!/usr/bin/env python3
"""Cost-insensitive window triangle experiments.

Default: four configurations shortlisted from the September 15 cost-sensitive
results. RRTWC_PRESET=full restores all 48 configurations. Both heuristics use
adapt_costs(one), and every search uses cost_type=one. First solution only.
Tetralith: full Autoscale Agile STRIPS, 15m / 6G. Local: every tenth task,
180s / 4G. Overrides use RRTWC_; see README.md.
"""

import math
import os
import platform
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
SCORPION_REPO = DIR.parent.parent
SUPPORT_DIR = DIR.parent / "2026-08-11-icaps27-lama-closest-competitors"
sys.path.insert(0, str(SUPPORT_DIR))

import custom_parser  # noqa: E402
import project  # noqa: E402
from downward import suites  # noqa: E402
from downward.experiment import FastDownwardAlgorithm  # noqa: E402
from lab import tools  # noqa: E402
from lab.experiment import Experiment, Run  # noqa: E402
from lab.reports import Attribute, arithmetic_mean  # noqa: E402

SLOPE = int(os.environ.get("RRTWC_SLOPE", "1"))
if SLOPE < 1:
    raise ValueError("RRTWC_SLOPE must be positive")
WINDOWS = (1, 5, 10, 100, 1000)
RANDOM_SEED = int(os.environ.get("RRTWC_RANDOM_SEED", "42"))
if RANDOM_SEED < 0:
    raise ValueError("RRTWC_RANDOM_SEED must be nonnegative for reproducible runs")
SCHEDULES = tuple(name.strip() for name in
                  os.environ.get("RRTWC_SCHEDULES", "sweep,depth").split(",") if name.strip())
if not SCHEDULES or len(set(SCHEDULES)) != len(SCHEDULES) or any(
    name not in ("sweep", "depth") for name in SCHEDULES
):
    raise ValueError("RRTWC_SCHEDULES must contain sweep and/or depth without duplicates")


def detect_cluster():
    forced = os.environ.get("RRTWC_CLUSTER", "").strip().lower()
    if forced:
        if forced not in ("tetralith", "local"):
            raise ValueError("RRTWC_CLUSTER must be tetralith or local")
        return forced
    if (
        project.TetralithEnvironment.is_present()
        or os.environ.get("TETRALITH_FORCE") == "1"
        or "tetralith" in platform.node().lower()
    ):
        return "tetralith"
    return "local"


CLUSTER = detect_cluster()
IS_TETRALITH = CLUSTER == "tetralith"
LOCAL_REPO = os.path.expanduser(
    os.environ.get("DOWNWARD_REPO", str(SCORPION_REPO))
)
LOCAL_BUILD = os.environ.get("DOWNWARD_BUILD", "release")
LOCAL_DRIVER = os.path.join(LOCAL_REPO, "fast-downward.py")

DEFAULT_BENCHMARKS = {
    "tetralith": (
        "/proj/mrlab_search_strategies/instances/"
        "autoscale-benchmarks/21.11-agile-strips"
    ),
    "local": "~/research/autoscale-benchmarks/21.11-agile-strips",
}
BENCHMARKS_DIR = os.path.expanduser(
    os.environ.get("DOWNWARD_BENCHMARKS", DEFAULT_BENCHMARKS[CLUSTER])
)

if IS_TETRALITH:
    BUDGET = os.environ.get("RRTWC_BUDGET", "15m")
    MEMORY = os.environ.get("RRTWC_MEMORY", "6G")
    INSTANCE_STEP = int(os.environ.get("RRTWC_INSTANCE_STEP", "1"))
else:
    BUDGET = os.environ.get("RRTWC_BUDGET", "180s")
    MEMORY = os.environ.get("RRTWC_MEMORY", "4G")
    INSTANCE_STEP = int(os.environ.get("RRTWC_INSTANCE_STEP", "10"))
INSTANCES_PER_DOMAIN = int(os.environ.get("RRTWC_INSTANCES_PER_DOMAIN", "0"))
LOCAL_PROCESSES = int(os.environ.get("RRTWC_PROCESSES", "2"))


def mem_to_gib(value):
    text = str(value).strip().upper()
    units = {"K": 1 / 1024 / 1024, "M": 1 / 1024, "G": 1, "T": 1024}
    if text and text[-1] in units:
        return math.ceil(float(text[:-1]) * units[text[-1]])
    return math.ceil(float(text) / (1024 ** 3))


if IS_TETRALITH:
    ENV = project.TetralithEnvironment(
        memory_per_cpu=f"{mem_to_gib(MEMORY) + 1}G",
        cpus_per_task=1,
        extra_options=(
            "#SBATCH --account="
            + os.environ.get("TETRALITH_ACCOUNT", "naiss2025-5-561")
        ),
    )
    if os.environ.get("TETRALITH_MAX_TASKS"):
        ENV.MAX_TASKS = int(os.environ["TETRALITH_MAX_TASKS"])
else:
    ENV = project.LocalEnvironment(processes=LOCAL_PROCESSES)


def list_domains():
    if not os.path.isdir(BENCHMARKS_DIR):
        raise RuntimeError(f"Benchmarks directory not found: {BENCHMARKS_DIR}")
    return sorted(
        name
        for name in os.listdir(BENCHMARKS_DIR)
        if os.path.isdir(os.path.join(BENCHMARKS_DIR, name))
    )


ALL_DOMAINS = list_domains()
domain_filter = os.environ.get("RRTWC_DOMAINS")
SUITE = (
    [name.strip() for name in domain_filter.split(",") if name.strip()]
    if domain_filter
    else ALL_DOMAINS
)
unknown_domains = [domain for domain in SUITE if domain not in ALL_DOMAINS]
if unknown_domains:
    raise ValueError(f"Unknown domains: {unknown_domains}")


def build_suite():
    index_text = os.environ.get("RRTWC_INSTANCE_INDEX")
    instance_index = int(index_text) if index_text else None
    if instance_index is not None and instance_index < 1:
        raise ValueError("RRTWC_INSTANCE_INDEX must be >= 1")
    if INSTANCE_STEP <= 0:
        raise ValueError("RRTWC_INSTANCE_STEP must be positive")
    tasks = []
    for domain in SUITE:
        domain_tasks = list(suites.Domain(BENCHMARKS_DIR, domain))
        if instance_index is not None:
            if len(domain_tasks) >= instance_index:
                tasks.append(domain_tasks[instance_index - 1])
        elif INSTANCES_PER_DOMAIN <= 0:
            tasks.extend(domain_tasks[::INSTANCE_STEP])
        else:
            tasks.extend(domain_tasks[:INSTANCES_PER_DOMAIN:INSTANCE_STEP])
    return tasks


TASKS = build_suite()
print(
    f"[cost-insensitive-triangle-windows] cluster={CLUSTER}, "
    f"{len(SUITE)} domains, {len(TASKS)} tasks, slope={SLOPE}"
)

LANDMARK_SUM = (
    "landmark_sum(lm_reasonable_orders_hps(lm_rhw()), "
    "transform=adapt_costs(one))"
)


def ff_lm(inner):
    return (
        f"let(hff, ff(transform=adapt_costs(one)), "
        f"let(hlm, {LANDMARK_SUM}, {inner}))"
    )


# Use the same plugin for PO and non-PO runs within each evaluation timing.
SEARCH_VARIANTS = (
    ("eager", "round_robin_triangle"),
    ("lazy", "lazy_multi_triangle"),
)
START_MODES = [(f"random-seed{RANDOM_SEED}",
                f"random_start=true, random_seed={RANDOM_SEED}")] + [
    (f"window{window}", f"window={window}") for window in WINDOWS
]
SEARCH_TEMPLATES = {}
for timing, plugin in SEARCH_VARIANTS:
    for schedule in SCHEDULES:
        for preferred in (False, True):
            po_name = "po" if preferred else "no-po"
            preferred_options = "[hff, hlm]" if preferred else "[]"
            for mode, start_options in START_MODES:
                name = f"triangle-{timing}-{schedule}-{po_name}-ff-lm-s{SLOPE}-{mode}"
                SEARCH_TEMPLATES[name] = ff_lm(
                    f"{plugin}(evals=[hff, hlm], slope={SLOPE}, "
                    f"schedule={schedule}, preferred_evals={preferred_options}, "
                    f"{start_options}, cost_type=one)"
                )
assert len(SEARCH_TEMPLATES) == 24 * len(SCHEDULES)

# Preserve eager/lazy and sweep/depth contenders without the full factorial grid.
# Window 100 and 1000 have complementary coverage, despite similar totals.
SHORTLIST = (
    f"triangle-lazy-sweep-po-ff-lm-s{SLOPE}-window100",
    f"triangle-lazy-sweep-po-ff-lm-s{SLOPE}-window1000",
    f"triangle-eager-sweep-po-ff-lm-s{SLOPE}-window5",
    f"triangle-eager-depth-po-ff-lm-s{SLOPE}-window10",
)
PRESET = os.environ.get("RRTWC_PRESET", "shortlist").strip().lower()
if PRESET not in ("shortlist", "full"):
    raise ValueError("RRTWC_PRESET must be shortlist or full")

# Explicit names override the preset and may select any generated configuration.
config_filter = os.environ.get("RRTWC_CONFIGS")
if config_filter:
    requested = [name.strip() for name in config_filter.split(",") if name.strip()]
    unknown = [name for name in requested if name not in SEARCH_TEMPLATES]
    if unknown:
        raise ValueError(f"Unknown configurations: {unknown}")
    SEARCH_TEMPLATES = {name: SEARCH_TEMPLATES[name] for name in requested}
elif PRESET == "shortlist":
    SEARCH_TEMPLATES = {
        name: SEARCH_TEMPLATES[name] for name in SHORTLIST if name in SEARCH_TEMPLATES
    }

CONFIGS = [
    (name, ["--translate-options", "--search-options", "--search", search])
    for name, search in SEARCH_TEMPLATES.items()
]
print(f"[cost-insensitive-triangle-windows] configs: {[n for n, _ in CONFIGS]}")

DRIVER_OPTIONS = [
    "--validate",
    "--search-time-limit", BUDGET,
    "--overall-memory-limit", MEMORY,
]


def duration_to_seconds(value):
    text = str(value).strip().lower()
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    units = {"s": 1, "m": 60, "h": 3600}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def seconds_to_hms(seconds):
    hours, remainder = divmod(int(math.ceil(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


if IS_TETRALITH:
    num_runs = len(CONFIGS) * len(TASKS)
    runs_per_task = math.ceil(num_runs / ENV.MAX_TASKS)
    estimate = runs_per_task * (duration_to_seconds(BUDGET) + 180)
    floor = duration_to_seconds(
        os.environ.get("RRTWC_WALL_TIME_FLOOR", "10:00:00")
    )
    ENV.time_limit_per_task = seconds_to_hms(max(estimate, floor))
    print(
        f"[cost-insensitive-triangle-windows] {num_runs} runs, "
        f"{runs_per_task} runs/array task, wall time {ENV.time_limit_per_task}"
    )


class LocalFastDownwardRun(Run):
    def __init__(self, exp, algo, task):
        super().__init__(exp)
        driver_options = algo.driver_options[:] + ["--build", LOCAL_BUILD]
        self.add_resource("domain", task.domain_file, "domain.pddl", symlink=True)
        self.add_resource("problem", task.problem_file, "problem.pddl", symlink=True)
        command = (
            [tools.get_python_executable(), LOCAL_DRIVER]
            + driver_options
            + ["{domain}", "{problem}"]
            + algo.component_options
        )
        self.add_command("planner", command)
        self.set_property("algorithm", algo.name)
        self.set_property("cost_type", "one")
        self.set_property("heuristic_cost_type", "one")
        self.set_property("repo", LOCAL_REPO)
        self.set_property("local_revision", "local")
        self.set_property("global_revision", "local")
        self.set_property("build_options", [LOCAL_BUILD])
        self.set_property("driver_options", driver_options)
        self.set_property("component_options", algo.component_options)
        for key, value in task.properties.items():
            self.set_property(key, value)
        self.set_property("experiment_name", exp.name)
        self.set_property("id", [algo.name, task.domain, task.problem])


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
    Attribute("evaluated", function=arithmetic_mean),
    Attribute("generated", function=arithmetic_mean),
    "memory",
    project.EVALUATIONS_PER_TIME,
]

stem = Path(__file__).stem + os.environ.get("RRTWC_EXPERIMENT_SUFFIX", "")
exp = Experiment(path=str(DIR / "data" / stem), environment=ENV)
for config_name, config in CONFIGS:
    algorithm = FastDownwardAlgorithm(config_name, None, DRIVER_OPTIONS, config)
    for task in TASKS:
        exp.add_run(LocalFastDownwardRun(exp, algorithm, task))

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
    attributes=ATTRIBUTES,
    filter=[project.add_evaluations_per_time],
)
project.add_compress_exp_dir_step(exp)

exp.run_steps()
