#!/usr/bin/env python3
"""
The 4 boosted-triangle configs closest to LAMA in mechanism, vs. LAMA itself,
across the FULL autoscale agile-strips suite (all domains, all instances).

This isn't the progress-credit/depth-budget parameter-configuration paper
(see ../aaai27/anytime-param-config.py) -- it's the ICAPS-27 boosted/lazy
triangle line of work (see icaps-27-plan.md, icaps-27-lazy-eval-design.md,
icaps-27-lazy-eval-implementation-prompt.md), evaluated against LAMA at full
scale. "Closest to LAMA" is judged on mechanism fidelity, not performance:
LAMA is defined by (1) the ff + landmark_sum(pref=true) heuristic pair, (2)
lazy evaluation (heuristics computed only when a node is popped for
expansion, not at generation time), and (3) the alternation-queue selection
policy (global per-list priority counters, boost_preferred triggered by any
evaluator reporting a new global-best value). Every config below shares (1)
identically; they differ only in how much of (2)/(3) they replicate vs.
substitute:

  lazy-triangle-lama-mimick   lazy_triangle_lama_mimick(...)
      Zero substitutions: lazy evaluation AND the real alternation/boost
      mechanism. The closest mechanical match to LAMA in this roster.
  triangle-lama-mimick        triangle_lama_mimick(...)
      One substitution: same real alternation/boost mechanism, but eager
      evaluation (computed at generation time) -- the one axis where it
      departs from what LAMA actually does.
  lazy-adaptive-boosted-triangle   lazy_adaptive_boosted_triangle(...)
      A different substitution: lazy evaluation matches LAMA, but selection
      uses a self-referential per-list token budget (credit_boost) plus an
      adaptive depth-budget cascade -- neither has a LAMA analogue.
  adaptive-boosted-triangle   adaptive_boosted_triangle(...)
      Both substitutions stacked: eager evaluation AND the invented
      credit-boost/depth-budget mechanism. Included per explicit request,
      even though it's the least LAMA-like of the four -- the point of this
      experiment is to see whether mechanism fidelity actually predicts
      performance, not to assume it.
  lama                        seq-sat-lama-2011 alias (the real thing)

Heuristic note: earlier ad hoc pilot runs during this line of work mixed
landmark_sum conventions (the two triangle-lama-mimick configs picked up a
transform=adapt_costs(one) borrowed from the parameter-configuration paper's
own LMCOUNT constant; the two adaptive-boosted-triangle configs did not).
Real seq-sat-lama-2011 does not apply that transform to its own hlm in the
unit-cost case (see driver/aliases.py's _get_lama), so this script normalizes
all four boosted-triangle configs onto the same untransformed
landmark_sum(lm_reasonable_orders_hps(lm_rhw()), pref=true) -- matching real
LAMA's own heuristic exactly and keeping the 4-way comparison uncontaminated
by a heuristic-definition mismatch. Results from this script are therefore
not bit-for-bit identical to the earlier pilot's triangle-lama-mimick numbers.

Runs on Tetralith by default (per explicit request); also supports Arrhenius
or local execution for testing. See detect_cluster() below. Local runs default
to a lighter sanity-check scope than the real cluster runs -- every 10th
instance, 180s, 4G -- since a local run is for validating the pipeline, not
collecting real data; every knob below is still env-overridable regardless of
cluster.

Environment variables:
  LCC_CLUSTER              force tetralith|arrhenius|local instead of
                            autodetecting from the hostname
  LCC_BUDGET                search-time-limit; default 15m on Tetralith/
                            Arrhenius, 180s locally
  LCC_MEMORY                overall-memory-limit; default 6G on Tetralith/
                            Arrhenius, 4G locally
  LCC_PROCESSES             local-only, default 2
  LCC_INSTANCES_PER_DOMAIN  default 0 = all instances in every domain
  LCC_INSTANCE_STEP         default 1 (every instance) on Tetralith/
                            Arrhenius, 10 (every 10th instance) locally
  LCC_INSTANCE_INDEX        1-indexed; if set, overrides INSTANCES_PER_DOMAIN/
                            INSTANCE_STEP and takes exactly that one instance
                            per domain (domains with fewer instances are
                            skipped) -- e.g. LCC_INSTANCE_INDEX=7 for a
                            one-instance-per-domain smoke test
  LCC_DOMAINS               comma-separated; power-user override of the full
                            domain list (default: every domain discovered
                            under the benchmarks dir)
  LCC_CREDIT_BOOST          adaptive-boosted-triangle credit_boost; default 10
  LCC_BOOST_AMOUNT          triangle-lama-mimick boost_amount (LAMA's own
                            DEFAULT_LAZY_BOOST); default 1000
  LCC_SLOPE                 triangle-lama-mimick slope; default 48 (matches
                            the HSDIP-paper-determined value used throughout
                            this branch)
  LCC_WALL_TIME_FLOOR       Slurm reservation floor, default 10:00:00
  LCC_BENCHMARK_TARGET      default autoscale-agile-21.11-strips
  DOWNWARD_REPO             default <repo root>
  DOWNWARD_BUILD            default release
  TETRALITH_ACCOUNT         default naiss2026-4-694
  TETRALITH_MAX_TASKS       Slurm MaxArraySize; default 2000
  TETRALITH_FORCE           set to 1 to force Tetralith detection
  ARRHENIUS_ACCOUNT         default naiss2026-4-694-cpu
  ARRHENIUS_PARTITION       default cpu
  ARRHENIUS_QOS             default unset (no --qos emitted)
  ARRHENIUS_MAX_TASKS       Slurm MaxArraySize; default 1000
  ARRHENIUS_VAL_BIN         VAL bin dir prepended to PATH for --validate
  ARRHENIUS_FORCE           set to 1 to force Arrhenius detection

Full-suite scope by design (no sharding): every domain under the benchmarks
dir, 5 configs, and (on Tetralith/Arrhenius) every instance -- this is a
large run, that's the point of running it on a cluster. Locally, every 10th
instance is enough to confirm the pipeline works end to end.
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


def detect_cluster():
    """Return 'tetralith', 'arrhenius', or 'local' for the current host.

    LCC_CLUSTER forces the choice (e.g. for testing generated commands off
    the cluster). Tetralith is checked first since it's this script's
    primary target; Arrhenius remains available for flexibility.
    """
    forced = os.environ.get("LCC_CLUSTER", "").strip().lower()
    if forced:
        if forced not in ("tetralith", "arrhenius", "local"):
            raise ValueError(
                f"LCC_CLUSTER must be one of tetralith/arrhenius/local, "
                f"got {forced!r}"
            )
        return forced
    if (
        project.TetralithEnvironment.is_present()
        or os.environ.get("TETRALITH_FORCE") == "1"
    ):
        return "tetralith"
    if (
        project.ArrheniusEnvironment.is_present()
        or os.environ.get("ARRHENIUS_FORCE") == "1"
    ):
        return "arrhenius"
    return "local"


CLUSTER = detect_cluster()
IS_TETRALITH = CLUSTER == "tetralith"
IS_ARRHENIUS = CLUSTER == "arrhenius"
IS_REMOTE = IS_TETRALITH or IS_ARRHENIUS

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
ARRHENIUS_BENCHMARK_DIRS = {
    "autoscale-agile-21.11-strips":
        "/home/thayer/search/shared/instances/autoscale-benchmarks/21.11-agile-strips",
}
BENCHMARK_DIRS_BY_CLUSTER = {
    "tetralith": TETRALITH_BENCHMARK_DIRS,
    "arrhenius": ARRHENIUS_BENCHMARK_DIRS,
    "local": LOCAL_BENCHMARK_DIRS,
}
BENCHMARK_DIRS = BENCHMARK_DIRS_BY_CLUSTER[CLUSTER]

BENCHMARK_TARGET_DEFAULT = "autoscale-agile-21.11-strips"
BENCHMARK_TARGET = os.environ.get("LCC_BENCHMARK_TARGET", BENCHMARK_TARGET_DEFAULT)
if BENCHMARK_TARGET not in BENCHMARK_DIRS:
    raise ValueError(
        f"Unknown LCC_BENCHMARK_TARGET={BENCHMARK_TARGET!r}; "
        f"valid: {sorted(BENCHMARK_DIRS)}"
    )
BENCHMARKS_DIR = os.path.expanduser(
    os.environ.get("DOWNWARD_BENCHMARKS", BENCHMARK_DIRS[BENCHMARK_TARGET])
)

# ----------------------------------------------------------------------------
# Scope: full suite, every instance, 15m/6G on Tetralith/Arrhenius (per
# explicit request); a lighter local sanity-check scope by default --
# every 10th instance, 180s, 4G -- since a local run is for validating the
# pipeline, not collecting real data. Every knob is still env-overridable.
# ----------------------------------------------------------------------------
if IS_REMOTE:
    BUDGET = os.environ.get("LCC_BUDGET", "15m")
    MEMORY = os.environ.get("LCC_MEMORY", "6G")
    INSTANCES_PER_DOMAIN = int(os.environ.get("LCC_INSTANCES_PER_DOMAIN", "0"))
    INSTANCE_STEP = int(os.environ.get("LCC_INSTANCE_STEP", "1"))
else:
    BUDGET = os.environ.get("LCC_BUDGET", "180s")
    MEMORY = os.environ.get("LCC_MEMORY", "4G")
    INSTANCES_PER_DOMAIN = int(os.environ.get("LCC_INSTANCES_PER_DOMAIN", "0"))
    INSTANCE_STEP = int(os.environ.get("LCC_INSTANCE_STEP", "10"))
LOCAL_PROCESSES = int(os.environ.get("LCC_PROCESSES", "2"))

CREDIT_BOOST = int(os.environ.get("LCC_CREDIT_BOOST", "10"))
BOOST_AMOUNT = int(os.environ.get("LCC_BOOST_AMOUNT", "1000"))
SLOPE = int(os.environ.get("LCC_SLOPE", "48"))


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def _mem_to_gib(value):
    """Parse a memory string like '6G'/'512M' to whole GiB (rounded up)."""
    text = str(value).strip().upper()
    units = {"K": 1 / 1024 / 1024, "M": 1 / 1024, "G": 1, "T": 1024}
    if text and text[-1] in units:
        return math.ceil(float(text[:-1]) * units[text[-1]])
    return math.ceil(float(text) / (1024 ** 3))  # bare value = bytes


TETRALITH_RESERVE_GIB = _mem_to_gib(MEMORY) + 1  # +1 GiB headroom for translate/validate/lab
TETRALITH_ACCOUNT = os.environ.get("TETRALITH_ACCOUNT", HEURISTIC_SEARCH_NAISS_ID)

ARRHENIUS_RESERVE_GIB = _mem_to_gib(MEMORY) + 1
ARRHENIUS_ACCOUNT = os.environ.get(
    "ARRHENIUS_ACCOUNT", f"{HEURISTIC_SEARCH_NAISS_ID}-cpu"
)
ARRHENIUS_VAL_BIN = os.environ.get(
    "ARRHENIUS_VAL_BIN", "/home/thayer/search/shared/libs/VAL/bin"
)

if IS_TETRALITH:
    ENV = project.TetralithEnvironment(
        memory_per_cpu=f"{TETRALITH_RESERVE_GIB}G",
        cpus_per_task=1,
        extra_options=f"#SBATCH --account={TETRALITH_ACCOUNT}",
    )
    _max_tasks_override = os.environ.get("TETRALITH_MAX_TASKS")
    if _max_tasks_override:
        ENV.MAX_TASKS = int(_max_tasks_override)
elif IS_ARRHENIUS:
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
# Suite: every domain, every instance -- no sharding
# ----------------------------------------------------------------------------
def list_domains(benchmarks_dir):
    if not os.path.isdir(benchmarks_dir):
        raise RuntimeError(f"Benchmarks dir not found: {benchmarks_dir}")
    return sorted(
        name for name in os.listdir(benchmarks_dir)
        if os.path.isdir(os.path.join(benchmarks_dir, name))
    )


ALL_DOMAINS = list_domains(BENCHMARKS_DIR)

_domains_filter = os.environ.get("LCC_DOMAINS")
SUITE = [d.strip() for d in _domains_filter.split(",") if d.strip()] if _domains_filter else list(ALL_DOMAINS)

_unknown_suite = [d for d in SUITE if d not in ALL_DOMAINS]
if _unknown_suite:
    raise ValueError(
        f"references unknown domains: {_unknown_suite}; available: {ALL_DOMAINS}"
    )
print(f"[lama-closest-competitors] cluster={CLUSTER}: {len(SUITE)} domains under {BENCHMARKS_DIR}")


def build_limited_suite(benchmarks_dir, domains, instances_per_domain, instance_step, instance_index=None):
    """instance_index (1-indexed), if set, takes exactly that one instance
    per domain -- a smoke-test selection mode distinct from
    instances_per_domain/instance_step (which both select ranges/strides
    starting at instance 1, never "just the Nth instance"). Domains with
    fewer than instance_index instances are skipped, not padded.
    """
    if instance_index is not None:
        if instance_index < 1:
            raise ValueError("LCC_INSTANCE_INDEX must be >= 1 (1-indexed)")
        tasks = []
        skipped = []
        for domain in domains:
            domain_tasks = list(suites.Domain(benchmarks_dir, domain))
            if len(domain_tasks) >= instance_index:
                tasks.append(domain_tasks[instance_index - 1])
            else:
                skipped.append(domain)
        if skipped:
            print(
                f"[lama-closest-competitors] skipped {len(skipped)} domain(s) with "
                f"fewer than {instance_index} instances: {skipped}"
            )
        print(
            f"[lama-closest-competitors] {len(tasks)} tasks "
            f"(1 per domain, instance #{instance_index})"
        )
        return tasks

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
        f"[lama-closest-competitors] {len(tasks)} tasks "
        f"({len(domains)} domains, ipd={instances_per_domain}, step={instance_step})"
    )
    return tasks


_instance_index_raw = os.environ.get("LCC_INSTANCE_INDEX")
INSTANCE_INDEX = int(_instance_index_raw) if _instance_index_raw else None

TASKS = build_limited_suite(
    BENCHMARKS_DIR, SUITE, INSTANCES_PER_DOMAIN, INSTANCE_STEP, INSTANCE_INDEX
)


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------
TRANSLATE_OPTIONS = ["--translate-options"]

# seq-sat-lama-2011's own hlm, verified against driver/aliases.py's
# _get_lama(pref="true") -- no transform in the unit-cost branch. Shared via
# let() so 'evals' and 'preferred_evals' bind the identical evaluator
# instances (a second ff()/landmark_sum() call would construct separate
# instances, which every boosted-search constructor's pointer-identity check
# on preferred_evals correctly rejects).
LANDMARK_SUM_PREF_TRUE = (
    "landmark_sum(lm_reasonable_orders_hps(lm_rhw()), pref=true)"
)


def lama_pair_config(inner):
    return (
        f"let(hff, ff(), "
        f"let(hlm, {LANDMARK_SUM_PREF_TRUE}, "
        f"{inner}))"
    )


SEARCH_TEMPLATES = {
    "lazy-triangle-lama-mimick": lama_pair_config(
        f"lazy_triangle_lama_mimick(evals=[hff, hlm], preferred_evals=[hff, hlm], "
        f"boost_amount={BOOST_AMOUNT}, slope={SLOPE}, anytime=true)"
    ),
    "triangle-lama-mimick": lama_pair_config(
        f"triangle_lama_mimick(evals=[hff, hlm], preferred_evals=[hff, hlm], "
        f"boost_amount={BOOST_AMOUNT}, slope={SLOPE}, anytime=true)"
    ),
    "lazy-adaptive-boosted-triangle": lama_pair_config(
        f"lazy_adaptive_boosted_triangle(evals=[hff, hlm], preferred_evals=[hff, hlm], "
        f"credit_boost={CREDIT_BOOST}, anytime=true)"
    ),
    "adaptive-boosted-triangle": lama_pair_config(
        f"adaptive_boosted_triangle(evals=[hff, hlm], preferred_evals=[hff, hlm], "
        f"credit_boost={CREDIT_BOOST}, anytime=true)"
    ),
}

ALIAS_CONFIGS = {
    "lama": ["--alias", "seq-sat-lama-2011"],
}

ANYTIME_DISPLAY_NAMES = {
    "lazy-triangle-lama-mimick": "Lazy Triangle-LAMA-mimick",
    "triangle-lama-mimick": "Triangle-LAMA-mimick",
    "lazy-adaptive-boosted-triangle": "Lazy Adaptive-Boosted-Triangle",
    "adaptive-boosted-triangle": "Adaptive-Boosted-Triangle",
    "lama": "LAMA",
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
for name, alias_opts in ALIAS_CONFIGS.items():
    CONFIGS.append((name, alias_opts))

print(f"[lama-closest-competitors] {len(CONFIGS)} configs: {[c[0] for c in CONFIGS]}")


# ----------------------------------------------------------------------------
# Slurm wall-clock reservation
# ----------------------------------------------------------------------------
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


if IS_REMOTE:
    NUM_RUNS = len(CONFIGS) * len(TASKS)
    RUNS_PER_TASK = math.ceil(NUM_RUNS / ENV.MAX_TASKS)
    EST_SECONDS = RUNS_PER_TASK * (
        _duration_to_seconds(BUDGET) + PER_RUN_OVERHEAD_SECONDS
    )
    FLOOR_SECONDS = _duration_to_seconds(
        os.environ.get("LCC_WALL_TIME_FLOOR", "10:00:00")
    )
    WALL_SECONDS = max(EST_SECONDS, FLOOR_SECONDS)
    ENV.time_limit_per_task = _seconds_to_hms(WALL_SECONDS)
    print(
        f"[lama-closest-competitors] {NUM_RUNS} runs, {RUNS_PER_TASK} runs/array-task; "
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

DATA_DIR = DIR / "data"
STEM = Path(__file__).stem
exp = Experiment(path=str(DATA_DIR / STEM), environment=ENV)

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
project.add_anytime_profile_plot_step(
    exp, max_time=_duration_to_seconds(BUDGET),
    display_names=ANYTIME_DISPLAY_NAMES,
)
project.add_anytime_profile_plot_step(
    exp, name="anytime-pdf", max_time=_duration_to_seconds(BUDGET), pdf=True,
    display_names=ANYTIME_DISPLAY_NAMES,
)
project.add_compress_exp_dir_step(exp)

exp.run_steps()
