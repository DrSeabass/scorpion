import contextlib
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

from downward.experiment import FastDownwardExperiment
from downward.reports.absolute import AbsoluteReport
from downward.reports.compare import ComparativeReport
from downward.reports.scatter import ScatterPlotReport
from downward.reports.taskwise import TaskwiseReport
from lab import tools
from lab.environments import (
    BaselSlurmEnvironment,
    LocalEnvironment,
    SlurmEnvironment,
    TetralithEnvironment,
)
from lab.experiment import ARGPARSER
from lab.reports import Attribute, geometric_mean

# Silence import-unused messages. Experiment scripts may use these imports.
assert (
    BaselSlurmEnvironment
    and ComparativeReport
    and FastDownwardExperiment
    and LocalEnvironment
    and ScatterPlotReport
    and SlurmEnvironment
    and TaskwiseReport
    and TetralithEnvironment
)


class ArrheniusEnvironment(SlurmEnvironment):
    """Environment for the NAISS Arrhenius cluster (Slurm).

    Mirrors :class:`~lab.environments.TetralithEnvironment` but for Arrhenius.
    Arrhenius specifics (from the NAISS Arrhenius quickstart / job-management
    docs) that differ from Tetralith:

      * Partitions are ``cpu`` (default here), ``gpu`` and ``fat``; there is no
        ``tetralith`` partition.
      * The Slurm account must be *suffixed for the partition family*, e.g.
        ``naiss2026-4-694-cpu`` for CPU jobs (``-gpu`` for GPU jobs).  Pass it
        via ``extra_options`` (``#SBATCH --account=...``), exactly as the
        Tetralith scripts did with the bare id.
      * Shared ``cpu`` nodes have 256 logical cores / 256 GB RAM, i.e. ~1 GiB
        per logical core, and memory is billed *proportionally to cores*.  So
        reserve memory as ``cpus_per_task`` cores at ``memory_per_cpu="1G"``
        rather than one core with a large ``memory_per_cpu``.
      * No named QOS is documented, so we leave ``DEFAULT_QOS`` empty and strip
        the ``--qos`` line the shared Slurm template always emits (an empty or
        invalid QOS would make ``sbatch`` reject the job).  Set ``qos`` (or the
        ``ARRHENIUS_QOS`` env var in the experiment) explicitly if your
        allocation requires one.

    Values the docs say to confirm on the machine with ``sinfo`` /
    ``scontrol show config`` (all overridable): ``MAX_TASKS`` (Slurm
    ``MaxArraySize``) and the maximum wall-clock time.  The defaults below are
    deliberately conservative.
    """

    DEFAULT_PARTITION = "cpu"
    # Arrhenius has no documented named QOS; leave empty so no --qos line is
    # emitted (see _get_job_header, which drops it when qos is falsy).
    DEFAULT_QOS = ""
    # Conservative wall-clock default; override per-experiment as needed.
    DEFAULT_TIME_LIMIT_PER_TASK = "24:00:00"
    # ~1 GiB per logical core on the shared cpu partition. Reserve total memory
    # by scaling cpus_per_task (memory is billed proportionally to cores).
    DEFAULT_MEMORY_PER_CPU = "1G"
    # MaxArraySize is not documented for Arrhenius. 1000 is a safe lower bound;
    # verify with `scontrol show config | grep MaxArraySize` and raise it if you
    # can (Tetralith allows 2000). Lab packs ceil(runs / MAX_TASKS) runs/task.
    MAX_TASKS = 1000

    @classmethod
    def is_present(cls):
        """True on an Arrhenius login or compute node."""
        node = platform.node()
        if "arrhenius" in node:
            return True
        # Inside a job the hostname may be a bare compute-node name, so also
        # trust Slurm's own cluster identifier when it is set.
        return os.environ.get("SLURM_CLUSTER_NAME", "").lower() == "arrhenius"

    def _get_job_header(self, step, is_last):
        header = super()._get_job_header(step, is_last)
        if not self.qos:
            # The shared slurm-job-header template unconditionally writes a
            # `#SBATCH --qos=<qos>` line; drop it when no QOS is configured.
            header = "\n".join(
                line
                for line in header.splitlines()
                if not line.startswith("#SBATCH --qos=")
            )
        return header


DIR = Path(__file__).resolve().parent
SCRIPT = Path(sys.argv[0]).resolve()

# Cover both the Basel and Linköping clusters for simplicity.
REMOTE = BaselSlurmEnvironment.is_present() or TetralithEnvironment.is_present()


def parse_args():
    ARGPARSER.add_argument("--tex", action="store_true", help="produce LaTeX output")
    ARGPARSER.add_argument(
        "--relative", action="store_true", help="make relative scatter plots"
    )
    args, _ = ARGPARSER.parse_known_args()
    return args


ARGS = parse_args()
TEX = ARGS.tex
RELATIVE = ARGS.relative

EVALUATIONS_PER_TIME = Attribute(
    "evaluations_per_time", min_wins=False, function=geometric_mean, digits=1
)

# Per-task time-integrated anytime quality in [0, 1] (see AnytimeQualityFilter);
# summed over tasks in the report, IPC-style, so higher is better. Named in the
# score_* family so downward's reports format it consistently.
SCORE_ANYTIME = Attribute(
    "score_anytime", absolute=True, min_wins=False, function=sum, digits=2
)

UNSOLVABLE_TASKS = {
    "mystery:prob%02d.pddl" % index
    for index in [4, 5, 7, 8, 12, 16, 18, 21, 22, 23, 24]
}

# Generated by "./suites.py satisficing" in aibasel/downward-benchmarks repo.
# fmt: off
SUITE_SATISFICING = [
    "agricola-sat18-strips", "airport", "assembly", "barman-sat11-strips",
    "barman-sat14-strips", "blocks", "caldera-sat18-adl",
    "caldera-split-sat18-adl", "cavediving-14-adl", "childsnack-sat14-strips",
    "citycar-sat14-adl", "data-network-sat18-strips", "depot", "driverlog",
    "elevators-sat08-strips", "elevators-sat11-strips", "flashfill-sat18-adl",
    "floortile-sat11-strips", "floortile-sat14-strips", "freecell",
    "ged-sat14-strips", "grid", "gripper", "hiking-sat14-strips",
    "logistics00", "logistics98", "maintenance-sat14-adl", "miconic",
    "miconic-fulladl", "miconic-simpleadl", "movie", "mprime", "mystery",
    "nomystery-sat11-strips", "nurikabe-sat18-adl", "openstacks",
    "openstacks-sat08-adl", "openstacks-sat08-strips",
    "openstacks-sat11-strips", "openstacks-sat14-strips", "openstacks-strips",
    "optical-telegraphs", "organic-synthesis-sat18-strips",
    "organic-synthesis-split-sat18-strips", "parcprinter-08-strips",
    "parcprinter-sat11-strips", "parking-sat11-strips", "parking-sat14-strips",
    "pathways", "pegsol-08-strips", "pegsol-sat11-strips", "philosophers",
    "pipesworld-notankage", "pipesworld-tankage", "psr-large", "psr-middle",
    "psr-small", "rovers", "satellite", "scanalyzer-08-strips",
    "scanalyzer-sat11-strips", "schedule", "settlers-sat18-adl",
    "snake-sat18-strips", "sokoban-sat08-strips", "sokoban-sat11-strips",
    "spider-sat18-strips", "storage", "termes-sat18-strips",
    "tetris-sat14-strips", "thoughtful-sat14-strips", "tidybot-sat11-strips",
    "tpp", "transport-sat08-strips", "transport-sat11-strips",
    "transport-sat14-strips", "trucks", "trucks-strips",
    "visitall-sat11-strips", "visitall-sat14-strips",
    "woodworking-sat08-strips", "woodworking-sat11-strips", "zenotravel",
]

SUITE_OPTIMAL_STRIPS = [
    "agricola-opt18-strips", "airport", "barman-opt11-strips",
    "barman-opt14-strips", "blocks", "childsnack-opt14-strips",
    "data-network-opt18-strips", "depot", "driverlog", "elevators-opt08-strips",
    "elevators-opt11-strips", "floortile-opt11-strips", "floortile-opt14-strips",
    "freecell", "ged-opt14-strips", "grid", "gripper", "hiking-opt14-strips",
    "logistics00", "logistics98", "miconic", "movie", "mprime", "mystery",
    "nomystery-opt11-strips", "openstacks-opt08-strips", "openstacks-opt11-strips",
    "openstacks-opt14-strips", "openstacks-strips", "organic-synthesis-opt18-strips",
    "organic-synthesis-split-opt18-strips", "parcprinter-08-strips",
    "parcprinter-opt11-strips", "parking-opt11-strips", "parking-opt14-strips",
    "pathways", "pegsol-08-strips", "pegsol-opt11-strips",
    "petri-net-alignment-opt18-strips", "pipesworld-notankage", "pipesworld-tankage",
    "psr-small", "quantum-layout-opt23-strips","rovers", "satellite", "scanalyzer-08-strips",
    "scanalyzer-opt11-strips", "snake-opt18-strips", "sokoban-opt08-strips",
    "sokoban-opt11-strips", "spider-opt18-strips", "storage", "termes-opt18-strips",
    "tetris-opt14-strips", "tidybot-opt11-strips", "tidybot-opt14-strips", "tpp",
    "transport-opt08-strips", "transport-opt11-strips", "transport-opt14-strips",
    "trucks-strips", "visitall-opt11-strips", "visitall-opt14-strips",
    "woodworking-opt08-strips", "woodworking-opt11-strips", "zenotravel",
]

SUITE_AUTOSCALE = [
    "agricola", "airport", "barman", "blocksworld", "childsnack", "data-network",
    "depots", "driverlog", "elevators", "floortile", "freecell", "ged", "grid",
    "gripper", "hiking", "logistics", "miconic", "mprime", "nomystery", "openstacks",
    "organic-synthesis-split", "parcprinter", "parking", "pathways", "pegsol",
    "pipesworld-notankage", "pipesworld-tankage", "rovers", "satellite", "scanalyzer",
    "snake", "sokoban", "storage", "termes", "tetris", "thoughtful", "tidybot", "tpp",
    "transport", "visitall", "woodworking", "zenotravel",
]


DOMAIN_GROUPS = {
    "airport": ["airport"],
    "assembly": ["assembly"],
    "barman": [
        "barman", "barman-opt11-strips", "barman-opt14-strips",
        "barman-sat11-strips", "barman-sat14-strips"],
    "blocksworld": ["blocks", "blocksworld"],
    "cavediving": ["cavediving-14-adl"],
    "childsnack": ["childsnack-opt14-strips", "childsnack-sat14-strips"],
    "citycar": ["citycar-opt14-adl", "citycar-sat14-adl"],
    "depots": ["depot", "depots"],
    "driverlog": ["driverlog"],
    "elevators": [
        "elevators-opt08-strips", "elevators-opt11-strips",
        "elevators-sat08-strips", "elevators-sat11-strips"],
    "floortile": [
        "floortile-opt11-strips", "floortile-opt14-strips",
        "floortile-sat11-strips", "floortile-sat14-strips"],
    "folding": ["folding-opt23-adl", "folding-sat23-adl"],
    "freecell": ["freecell"],
    "ged": ["ged-opt14-strips", "ged-sat14-strips"],
    "grid": ["grid"],
    "gripper": ["gripper"],
    "hiking": ["hiking-opt14-strips", "hiking-sat14-strips"],
    "labyrinth": ["labyrinth-opt23-adl", "labyrinth-sat23-adl"],
    "logistics": ["logistics98", "logistics00"],
    "maintenance": ["maintenance-opt14-adl", "maintenance-sat14-adl"],
    "miconic": ["miconic", "miconic-strips"],
    "miconic-fulladl": ["miconic-fulladl"],
    "miconic-simpleadl": ["miconic-simpleadl"],
    "movie": ["movie"],
    "mprime": ["mprime"],
    "mystery": ["mystery"],
    "nomystery": ["nomystery-opt11-strips", "nomystery-sat11-strips"],
    "openstacks": [
        "openstacks", "openstacks-strips", "openstacks-opt08-strips",
        "openstacks-opt11-strips", "openstacks-opt14-strips",
        "openstacks-sat08-adl", "openstacks-sat08-strips",
        "openstacks-sat11-strips", "openstacks-sat14-strips",
        "openstacks-opt08-adl", "openstacks-sat08-adl"],
    "optical-telegraphs": ["optical-telegraphs"],
    "parcprinter": [
        "parcprinter-08-strips", "parcprinter-opt11-strips",
        "parcprinter-sat11-strips"],
    "parking": [
        "parking-opt11-strips", "parking-opt14-strips",
        "parking-sat11-strips", "parking-sat14-strips"],
    "pathways": ["pathways"],
    "pathways-noneg": ["pathways-noneg"],
    "pegsol": ["pegsol-08-strips", "pegsol-opt11-strips", "pegsol-sat11-strips"],
    "philosophers": ["philosophers"],
    "pipes-nt": ["pipesworld-notankage"],
    "pipes-t": ["pipesworld-tankage"],
    "psr": ["psr-middle", "psr-large", "psr-small"],
    "quantum-layout": ["quantum-layout-opt23-strips", "quantum-layout-sat23-strips"],
    "recharging-robots": ["recharging-robots-opt23-adl", "recharging-robots-sat23-adl"],
    "ricochet-robots": ["ricochet-robots-opt23-adl", "ricochet-robots-sat23-adl"],
    "rovers": ["rover", "rovers"],
    "rubiks-cube": ["rubiks-cube-opt23-adl", "rubiks-cube-sat23-adl"],
    "satellite": ["satellite"],
    "scanalyzer": [
        "scanalyzer-08-strips", "scanalyzer-opt11-strips", "scanalyzer-sat11-strips"],
    "schedule": ["schedule"],
    "slitherlink": ["slitherlink-opt23-adl", "slitherlink-sat23-adl"],
    "sokoban": [
        "sokoban-opt08-strips", "sokoban-opt11-strips",
        "sokoban-sat08-strips", "sokoban-sat11-strips"],
    "storage": ["storage"],
    "tetris": ["tetris-opt14-strips", "tetris-sat14-strips"],
    "thoughtful": ["thoughtful-sat14-strips"],
    "tidybot": [
        "tidybot-opt11-strips", "tidybot-opt14-strips",
        "tidybot-sat11-strips", "tidybot-sat14-strips"],
    "tpp": ["tpp"],
    "transport": [
        "transport-opt08-strips", "transport-opt11-strips", "transport-opt14-strips",
        "transport-sat08-strips", "transport-sat11-strips", "transport-sat14-strips"],
    "trucks": ["trucks", "trucks-strips"],
    "visitall": [
        "visitall-opt11-strips", "visitall-opt14-strips",
        "visitall-sat11-strips", "visitall-sat14-strips"],
    "woodworking": [
        "woodworking-opt08-strips", "woodworking-opt11-strips",
        "woodworking-sat08-strips", "woodworking-sat11-strips"],
    "zenotravel": ["zenotravel"],
    # IPC 2018:
    "agricola": ["agricola", "agricola-opt18-strips", "agricola-sat18-strips"],
    "caldera": ["caldera-opt18-adl", "caldera-sat18-adl"],
    "caldera-split": ["caldera-split-opt18-adl", "caldera-split-sat18-adl"],
    "data-network": [
        "data-network", "data-network-opt18-strips", "data-network-sat18-strips"],
    "flashfill": ["flashfill-sat18-adl"],
    "nurikabe": ["nurikabe-opt18-adl", "nurikabe-sat18-adl"],
    "organic-split": [
        "organic-synthesis-split", "organic-synthesis-split-opt18-strips",
        "organic-synthesis-split-sat18-strips"],
    "organic" : [
        "organic-synthesis", "organic-synthesis-opt18-strips",
        "organic-synthesis-sat18-strips"],
    "petri-net": [
        "petri-net-alignment", "petri-net-alignment-opt18-strips",
        "petri-net-alignment-sat18-strips"],
    "settlers": ["settlers-opt18-adl", "settlers-sat18-adl"],
    "snake": ["snake", "snake-opt18-strips", "snake-sat18-strips"],
    "spider": ["spider", "spider-opt18-strips", "spider-sat18-strips"],
    "termes": ["termes", "termes-opt18-strips", "termes-sat18-strips"],
}
# fmt: on


DOMAIN_RENAMINGS = {}
for group_name, domains in DOMAIN_GROUPS.items():
    for domain in domains:
        DOMAIN_RENAMINGS[domain] = group_name
for group_name in DOMAIN_GROUPS:
    DOMAIN_RENAMINGS[group_name] = group_name


def group_domains(run):
    old_domain = run["domain"]
    run["domain"] = DOMAIN_RENAMINGS[old_domain]
    run["problem"] = old_domain + "-" + run["problem"]
    run["id"][2] = run["problem"]
    return run


def get_repo_base() -> Path:
    """Get base directory of the repository, as an absolute path.

    Search upwards in the directory tree from the main script until a
    directory with a subdirectory named ".git" is found.

    Abort if the repo base cannot be found."""
    path = Path(SCRIPT)
    while path.parent != path:
        if (path / ".git").is_dir():
            return path
        path = path.parent
    sys.exit("repo base could not be found")


def remove_file(path: Path):
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def add_evaluations_per_time(run):
    evaluations = run.get("evaluations")
    time = run.get("search_time")
    if evaluations is not None and evaluations >= 100 and time:
        run["evaluations_per_time"] = evaluations / time
    return run


def _get_exp_dir_relative_to_repo():
    repo_name = get_repo_base().name
    script = Path(SCRIPT)
    script_dir = script.parent
    rel_script_dir = script_dir.relative_to(get_repo_base())
    expname = script.stem
    return repo_name / rel_script_dir / "data" / expname


def add_scp_step(exp, login, repos_dir, name="scp-eval-dir"):
    remote_exp = Path(repos_dir) / _get_exp_dir_relative_to_repo()
    exp.add_step(
        name,
        subprocess.call,
        [
            "rsync",
            "-Pavz",
            f"{login}:{remote_exp}-eval/",
            f"{exp.path}-eval/",
        ],
    )


def add_compress_exp_dir_step(exp):
    def compress_exp_dir():
        tar_file_path = Path(exp.path).parent / f"{exp.name}.tar.xz"
        exp_dir_path = Path(exp.path)

        with tarfile.open(tar_file_path, mode="w:xz", dereference=True) as tar:
            for file in exp_dir_path.rglob("*"):
                relpath = file.relative_to(exp_dir_path.parent)
                print(f"Adding {relpath}")
                tar.add(file, arcname=relpath)

        shutil.rmtree(exp_dir_path)

    exp.add_step("compress-exp-dir", compress_exp_dir)


def add_bounded_suboptimal_plot_step(
    exp,
    attributes=("coverage", "expansions", "search_time", "cost"),
    name="plot",
    outfile=None,
):
    """Add a step plotting each attribute against the suboptimality weight.

    Reads the fetched properties, recovers (family, weight) from each algorithm
    name -- split on the last '-w', reading the weight tag's '_' back as '.', so
    e.g. 'rrdex-db-w2_5' -> family 'rrdex-db', weight 2.5 -- and draws one line
    per family with a 95% confidence band (shaded region):
      - coverage is summed over tasks; its band is the Wilson score interval on
        the solved/attempted proportion, scaled back to counts (well-behaved at
        0/1 and small samples).
      - the other attributes are geometric-mean-aggregated over the runs solved
        at that weight (so effort curves stay comparable across weights); their
        band is the 95% interval computed in log space (mean +/- 1.96 * SE of
        log values) and exponentiated, matching the log y-axis.
    Writes one multi-panel PNG into the eval dir. Runs locally after `fetch`; a
    no-op (with a message) if matplotlib is unavailable or no runs carry a
    parseable weight.
    """

    def make_plot():
        import json
        import math
        from collections import defaultdict

        properties_path = Path(exp.eval_dir) / "properties"
        if not properties_path.exists():
            print(f"No properties file at {properties_path}; run fetch first.")
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping bounded-suboptimal plot.")
            return

        with open(properties_path) as f:
            props = json.load(f)

        NAN = float("nan")
        Z = 1.96  # 95% (normal approximation)

        # (family, weight) -> solved run dicts, and attempted-task counts.
        solved = defaultdict(lambda: defaultdict(list))
        attempted = defaultdict(lambda: defaultdict(int))
        for run in props.values():
            algo = run.get("algorithm", "")
            if "-w" not in algo:
                continue
            family, tag = algo.rsplit("-w", 1)
            try:
                weight = float(tag.replace("_", "."))
            except ValueError:
                continue
            attempted[family][weight] += 1
            if run.get("coverage"):
                solved[family][weight].append(run)

        if not attempted:
            print("No runs with parseable weights; nothing to plot.")
            return

        families = sorted(attempted)
        weights = sorted({w for fam in attempted.values() for w in fam})

        def gmean_ci(values):
            """(point, lo, hi) for the geometric mean; log-space 95% CI."""
            xs = [v for v in values if v and v > 0]
            n = len(xs)
            if n == 0:
                return NAN, NAN, NAN
            logs = [math.log(v) for v in xs]
            mean = sum(logs) / n
            point = math.exp(mean)
            if n < 2:
                return point, point, point
            var = sum((x - mean) ** 2 for x in logs) / (n - 1)
            se = math.sqrt(var) / math.sqrt(n)
            return point, math.exp(mean - Z * se), math.exp(mean + Z * se)

        def wilson_ci(k, n):
            """(count, lo, hi) with a Wilson score band scaled to counts."""
            if n == 0:
                return NAN, NAN, NAN
            p = k / n
            denom = 1 + Z * Z / n
            center = (p + Z * Z / (2 * n)) / denom
            half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
            return k, max(0.0, center - half) * n, min(1.0, center + half) * n

        def series(family, attribute):
            points, los, his = [], [], []
            for w in weights:
                runs = solved[family].get(w, [])
                if attribute == "coverage":
                    pt, lo, hi = wilson_ci(len(runs), attempted[family].get(w, 0))
                else:
                    pt, lo, hi = gmean_ci([r.get(attribute) for r in runs])
                points.append(pt)
                los.append(lo)
                his.append(hi)
            return points, los, his

        n = len(attributes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
        for ax, attribute in zip(axes[0], attributes):
            for family in families:
                points, los, his = series(family, attribute)
                # Plot over the full weight grid, leaving weights with no data as
                # NaN. matplotlib then draws a marker at every weight the family
                # has data for but connects only CONSECUTIVE weights, breaking the
                # line across any gap -- so we never draw a segment through a
                # weight we didn't measure (e.g. wA*'s non-integer weights, or a
                # weight where the family solved nothing).
                line, = ax.plot(weights, points, marker="o", label=family)
                # Coverage is a raw count, not an estimate; showing a band there
                # over-implies uncertainty, so shade the CI on the other panels only.
                if attribute != "coverage":
                    ax.fill_between(
                        weights, los, his, color=line.get_color(), alpha=0.15,
                        linewidth=0,
                    )
            ax.set_xlabel("suboptimality weight w")
            ax.set_title(
                attribute if attribute == "coverage" else f"gmean {attribute}"
            )
            ax.grid(True, alpha=0.3)
            if attribute != "coverage":
                ax.set_yscale("log")
        axes[0][0].legend(fontsize=7, loc="best")
        fig.suptitle(
            f"{exp.name}: bounded-suboptimal curves vs weight "
            f"(95% CI shaded, except coverage)"
        )
        fig.tight_layout()

        out = outfile or (
            Path(exp.eval_dir) / f"{exp.name}-bounded-suboptimal.png"
        )
        fig.savefig(out, dpi=110)
        print(f"Wrote {out}")

    exp.add_step(name, make_plot)


def _incumbent_cost(pairs, t):
    """Best (min) cost among solutions found at wall-clock time <= *t*.

    *pairs* is a time-sorted list of (time, cost). Returns None if the config
    has no solution by *t* (i.e. before its first plan)."""
    best = None
    for ti, ci in pairs:
        if ti > t:
            break
        best = ci if best is None else min(best, ci)
    return best


def _run_pairs(run):
    """Time-sorted [(t, cost)] trajectory for a single run.

    Pairs the standard ``cost:all`` list element-wise with ``cost_times:all``
    (the wall-clock timestamp on each "Plan cost:" line, from custom_parser).
    Empty if the run recorded no plans."""
    costs = run.get("cost:all") or []
    times = run.get("cost_times:all") or []
    return sorted((float(t), float(c)) for t, c in zip(times, costs))


def _anytime_grid(first_times, limit_times, observed_tmax, max_time, num_points):
    """Shared log-spaced sampling grid for anytime curves and scores.

    Bounding the grid identically for the plot and the per-run score keeps the
    two consistent: a run's score is exactly the area under its own quality
    curve, and the mean of the scores is the area under the plotted mean curve.

    *max_time* (upper bound) defaults to the largest ``limit_search_time``,
    falling back to *observed_tmax* (the latest observed solution time); the
    lower bound is the earliest observed solution time in *first_times*, floored
    at 1ms. Returns ``(grid, max_time)``."""
    if max_time is None:
        max_time = max(limit_times) if limit_times else observed_tmax
    if not max_time or max_time <= 0:
        max_time = observed_tmax or 1.0

    tmin = min((t for t in first_times if t > 0), default=max_time / 1000.0)
    tmin = max(tmin, 1e-3)
    if tmin >= max_time:
        tmin = max_time / 1000.0

    grid = [
        tmin * (max_time / tmin) ** (i / (num_points - 1))
        for i in range(num_points)
    ]
    return grid, max_time


class AnytimeQualityFilter:
    """Two-pass report filter adding a per-run time-integrated anytime score.

    The score is the scalar counterpart of :func:`render_anytime_profile`'s
    quality curve: for a task whose best cost across *all* configs is ``ref``, a
    run's quality at time t is ``ref / incumbent_cost(t)`` in (0, 1] (0 before
    its first plan), and the score is the mean of that quality over the shared
    log-spaced time grid -- i.e. the (normalised) area under the run's quality
    vs. log-time curve. Unsolved runs score 0. Because the grid is shared, the
    mean of the per-run scores equals the area under the plotted mean curve.

    Reference costs are cross-run, so this needs two passes; lab applies each
    filter in a list to every run before the next, so pass it as::

        f = AnytimeQualityFilter(max_time=<cutoff>)
        add_absolute_report(exp, filter=[f.collect, f.add_score])

    *max_time* should match the search cutoff (as passed to the plot) so the
    score and the figure share an x-axis; if None it is inferred from the runs'
    ``limit_search_time``."""

    def __init__(self, *, max_time=None, num_points=200, attribute="score_anytime"):
        self.max_time = max_time
        self.num_points = num_points
        self.attribute = attribute
        self._best_cost = {}
        self._limit_times = []
        self._first_times = []
        self._observed_tmax = 0.0
        self._grid = None

    @staticmethod
    def _task(run):
        return (run.get("domain"), run.get("problem"))

    def collect(self, run):
        """Pass 1: record each task's best cost and the global grid bounds."""
        lt = run.get("limit_search_time")
        if lt:
            self._limit_times.append(float(lt))
        pairs = _run_pairs(run)
        if pairs:
            self._first_times.append(pairs[0][0])
            self._observed_tmax = max(self._observed_tmax, pairs[-1][0])
            cmin = min(c for _, c in pairs)
            task = self._task(run)
            if task not in self._best_cost or cmin < self._best_cost[task]:
                self._best_cost[task] = cmin
        return True

    def add_score(self, run):
        """Pass 2: set ``run[attribute]`` to the integrated anytime quality."""
        if self._grid is None:
            self._grid, _ = _anytime_grid(
                self._first_times,
                self._limit_times,
                self._observed_tmax,
                self.max_time,
                self.num_points,
            )
        pairs = _run_pairs(run)
        ref = self._best_cost.get(self._task(run))
        if not pairs or ref is None:
            run[self.attribute] = 0.0
            return run
        total = 0.0
        for t in self._grid:
            inc = _incumbent_cost(pairs, t)
            if inc and inc > 0:
                total += ref / inc
        run[self.attribute] = total / len(self._grid)
        return run


def _wilson_interval(k, n, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion ``k`` of ``n``.

    Preferred over the normal approximation for proportions, which can
    produce out-of-[0, 1] bounds and is unreliable near p=0 or p=1 -- exactly
    where a coverage curve starts (nothing solved yet) and often ends
    (most/all instances solved). Returns ``(lo, hi)`` as fractions in [0, 1]."""
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = phat + z2 / (2 * n)
    half = z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)
    return max(0.0, (center - half) / denom), min(1.0, (center + half) / denom)


def _compute_anytime_curves(properties, *, max_time=None, num_points=200):
    """Reduce fetched *properties* to plottable anytime curves.

    Reads the (time, cost) trajectory each run recorded -- ``cost_times:all``
    (from custom_parser) paired element-wise with the standard ``cost:all`` --
    and returns, for each algorithm, two curves sampled on a shared log-spaced
    time grid:

      * IPC anytime quality. For each instance the reference cost is the best
        (min) cost *any* config ever found on it; a config's quality at time t
        is ref / (its incumbent cost by t), in (0, 1], and 0 before its first
        solution. Averaged over *all* instances, so the curve folds in coverage
        (unsolved-by-t counts as 0) -- the usual IPC-style anytime score.
      * Coverage: number of instances solved (any plan found) by t, out of
        ``n_inst`` total.
      * CI: half-width of the 95% confidence interval of the mean quality
        across instances at t (normal approximation on the per-instance
        quality values, which include 0 for unsolved instances), so the band
        is ``quality +/- ci``.
      * Coverage CI: 95% Wilson score interval on the coverage proportion at
        t, rescaled to instance counts (``cov_lo``, ``cov_hi``), treating
        solved-by-t as a Bernoulli indicator per instance.

    *max_time* bounds the grid; if None it is taken from the runs'
    ``limit_search_time`` (the search cutoff), falling back to the latest
    observed solution time.

    Returns ``(grid, curves, n_inst)`` where *curves* maps algorithm ->
    ``(quality, coverage, ci, cov_lo, cov_hi)`` lists aligned with *grid*, or
    ``None`` if there is no trajectory data to plot.
    """
    # algo -> instance -> time-sorted [(t, cost)]; plus per-instance best cost.
    trajectories = defaultdict(dict)
    best_cost = {}
    instances = set()
    limit_times = []
    first_times = []
    observed_tmax = 0.0

    for run in properties.values():
        algo = run.get("algorithm")
        domain = run.get("domain")
        problem = run.get("problem")
        if algo is None or domain is None or problem is None:
            continue
        inst = (domain, problem)
        instances.add(inst)
        lt = run.get("limit_search_time")
        if lt:
            limit_times.append(float(lt))
        pairs = _run_pairs(run)
        trajectories[algo][inst] = pairs
        if pairs:
            first_times.append(pairs[0][0])
            observed_tmax = max(observed_tmax, pairs[-1][0])
            cmin = min(c for _, c in pairs)
            if inst not in best_cost or cmin < best_cost[inst]:
                best_cost[inst] = cmin

    if not instances or not trajectories:
        print("No runs with (algorithm, domain, problem); nothing to plot.")
        return None

    grid, max_time = _anytime_grid(
        first_times, limit_times, observed_tmax, max_time, num_points
    )
    n_inst = len(instances)

    def quality_curve(algo):
        by_inst = trajectories[algo]
        qs, covs, cis, cov_los, cov_his = [], [], [], [], []
        for t in grid:
            per_inst = []
            solved = 0
            for inst in instances:
                inc = _incumbent_cost(by_inst.get(inst, []), t)
                if inc is None:
                    per_inst.append(0.0)
                    continue
                solved += 1
                ref = best_cost.get(inst)
                q = ref / inc if (ref is not None and inc > 0) else 0.0
                per_inst.append(q)
            qs.append(sum(per_inst) / n_inst)
            covs.append(solved)
            # 95% CI of the across-instance mean via the normal approximation
            # (n_inst is typically large); 0 when there's no spread to estimate.
            cis.append(
                1.96 * statistics.stdev(per_inst) / n_inst**0.5
                if n_inst > 1
                else 0.0
            )
            lo, hi = _wilson_interval(solved, n_inst)
            cov_los.append(lo * n_inst)
            cov_his.append(hi * n_inst)
        return qs, covs, cis, cov_los, cov_his

    curves = {algo: quality_curve(algo) for algo in sorted(trajectories)}
    return grid, curves, n_inst


# Panel definitions shared by the combined and per-panel renderers, keyed by a
# short id: (y-axis label, index into the (quality, coverage, ci, cov_lo,
# cov_hi) curve tuple). "coverage-ci" plots the same counts as "coverage" but
# as its own panel/figure with a Wilson-interval band -- kept separate rather
# than overlaid on "coverage" so the plain browsing curve stays uncluttered.
# "coverage-rate" is the same Wilson band again, rescaled to a [0, 1]
# proportion instead of instance counts -- the more natural axis for the
# interval, since Wilson is computed on the proportion in the first place.
_ANYTIME_PANELS = {
    "quality": ("mean anytime quality", 0),
    "coverage": ("coverage (instances solved)", 1),
    "coverage-ci": ("coverage (instances solved)\n95% CI", 1),
    "coverage-rate": ("coverage (fraction solved)\n95% CI", 1),
}

# Cycled per-algorithm styles so a config keeps the same colour *and* line
# style across both panels and stays distinguishable in greyscale print.
_ANYTIME_LINESTYLES = ["-", "--", "-.", ":"]


def _anytime_style(algorithms):
    """Map each algorithm to a stable ``(color, linestyle)`` for print."""
    return {
        algo: (f"C{i % 10}", _ANYTIME_LINESTYLES[i % len(_ANYTIME_LINESTYLES)])
        for i, algo in enumerate(algorithms)
    }


def _anytime_label(algo, display_names):
    """Legend label for *algo*, substituted via *display_names* if given
    (falls back to the raw algorithm/run-id name otherwise)."""
    if not display_names:
        return algo
    return display_names.get(algo, algo)


def render_anytime_profile(
    properties, outfile, *, max_time=None, num_points=200, display_names=None
):
    """Render all four anytime panels side by side to a single *outfile*.

    A quick-look combined figure (with a descriptive suptitle) intended for
    browsing during a run -- not for paper inclusion; use
    :func:`render_anytime_profile_pdfs` for that. Returns True if a plot was
    written, else False (matplotlib missing or no trajectory data).

    *display_names* optionally maps a raw algorithm/run-id name to the legend
    label it should show instead (algorithms not in the map show their raw
    name)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping anytime-profile plot.")
        return False

    computed = _compute_anytime_curves(
        properties, max_time=max_time, num_points=num_points
    )
    if computed is None:
        return False
    grid, curves, n_inst = computed
    style = _anytime_style(curves)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6), squeeze=False)
    ax_q, ax_c, ax_cc, ax_cr = axes[0]
    for algo, (qs, covs, cis, cov_lo, cov_hi) in curves.items():
        color, ls = style[algo]
        label = _anytime_label(algo, display_names)
        rate = [c / n_inst for c in covs]
        rate_lo = [v / n_inst for v in cov_lo]
        rate_hi = [v / n_inst for v in cov_hi]
        ax_q.plot(grid, qs, label=label, color=color, linestyle=ls)
        ax_q.fill_between(
            grid,
            [q - c for q, c in zip(qs, cis)],
            [q + c for q, c in zip(qs, cis)],
            color=color, alpha=0.15, linewidth=0,
        )
        ax_c.plot(grid, covs, label=label, color=color, linestyle=ls)
        ax_cc.plot(grid, covs, label=label, color=color, linestyle=ls)
        ax_cc.fill_between(grid, cov_lo, cov_hi, color=color, alpha=0.15, linewidth=0)
        ax_cr.plot(grid, rate, label=label, color=color, linestyle=ls)
        ax_cr.fill_between(grid, rate_lo, rate_hi, color=color, alpha=0.15, linewidth=0)
    for ax in (ax_q, ax_c, ax_cc, ax_cr):
        ax.set_xscale("log")
        ax.set_xlabel("elapsed search time (s)")
        ax.grid(True, alpha=0.3)
    ax_q.set_ylim(0, 1.02)
    ax_c.set_ylim(0, n_inst * 1.02)
    ax_cc.set_ylim(0, n_inst * 1.02)
    ax_cr.set_ylim(0, 1.02)
    ax_q.set_ylabel(_ANYTIME_PANELS["quality"][0])
    ax_q.set_title("anytime quality vs time")
    ax_c.set_ylabel(_ANYTIME_PANELS["coverage"][0])
    ax_c.set_title("coverage vs time")
    ax_cc.set_ylabel(_ANYTIME_PANELS["coverage"][0])
    ax_cc.set_title("coverage vs time, 95% CI (Wilson)")
    ax_cr.set_ylabel(_ANYTIME_PANELS["coverage-rate"][0])
    ax_cr.set_title("coverage rate vs time, 95% CI (Wilson)")
    ax_q.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"anytime profiles over {n_inst} instances "
        f"(quality ref = best cost found by any config; "
        f"shading = 95% CI of the mean / Wilson CI for coverage)"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    print(f"Wrote {outfile}")
    plt.close(fig)
    return True


def render_anytime_profile_pdfs(
    properties,
    stem,
    *,
    max_time=None,
    num_points=200,
    panels=("quality", "coverage", "coverage-ci", "coverage-rate"),
    figsize=(3.5, 2.7),
    display_names=None,
):
    """Render each anytime panel to its own paper-ready vector PDF.

    Writes ``<stem>-<panel>.pdf`` for each requested panel (default:
    ``quality``, ``coverage``, ``coverage-ci``, and ``coverage-rate`` -- the
    last two both a Wilson-interval band around coverage, in instance-count
    and proportion units respectively). Each figure is self-contained and
    sized for a two-column layout (~half text width): no suptitle or axes
    title (the LaTeX caption carries that), print-legible fonts, an embedded
    legend, and TrueType-embedded fonts so venues that reject Type-3 fonts
    accept it.

    *display_names* optionally maps a raw algorithm/run-id name to the legend
    label it should show instead (algorithms not in the map show their raw
    name).

    Returns the list of paths written (empty if matplotlib is missing or there
    is no trajectory data)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping anytime-profile PDFs.")
        return []

    computed = _compute_anytime_curves(
        properties, max_time=max_time, num_points=num_points
    )
    if computed is None:
        return []
    grid, curves, n_inst = computed
    style = _anytime_style(curves)
    stem = Path(stem)

    # Embed TrueType (type 42) rather than Type 3 fonts -- AAAI/IEEE reject the
    # latter -- and use print-legible sizes for a half-text-width figure.
    rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    written = []
    with plt.rc_context(rc):
        for panel in panels:
            if panel not in _ANYTIME_PANELS:
                print(f"Unknown anytime panel {panel!r}; skipping.")
                continue
            ylabel, idx = _ANYTIME_PANELS[panel]
            fig, ax = plt.subplots(figsize=figsize)
            for algo, series in curves.items():
                color, ls = style[algo]
                ys = series[idx]
                if panel == "coverage-rate":
                    ys = [y / n_inst for y in ys]
                ax.plot(
                    grid, ys, label=_anytime_label(algo, display_names),
                    color=color, linestyle=ls,
                )
                if panel == "quality":
                    cis = series[2]
                    ax.fill_between(
                        grid,
                        [y - c for y, c in zip(ys, cis)],
                        [y + c for y, c in zip(ys, cis)],
                        color=color, alpha=0.15, linewidth=0,
                    )
                elif panel == "coverage-ci":
                    cov_lo, cov_hi = series[3], series[4]
                    ax.fill_between(
                        grid, cov_lo, cov_hi, color=color, alpha=0.15, linewidth=0,
                    )
                elif panel == "coverage-rate":
                    cov_lo = [v / n_inst for v in series[3]]
                    cov_hi = [v / n_inst for v in series[4]]
                    ax.fill_between(
                        grid, cov_lo, cov_hi, color=color, alpha=0.15, linewidth=0,
                    )
            ax.set_xscale("log")
            ax.set_xlabel("elapsed search time (s)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.02 if panel in ("quality", "coverage-rate") else n_inst * 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", framealpha=0.9)
            out = stem.with_name(f"{stem.name}-{panel}.pdf")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")
            written.append(out)
    return written


def add_anytime_profile_plot_step(
    exp, name="anytime-plot", outfile=None, max_time=None, *, pdf=False,
    display_names=None,
):
    """Add a step drawing anytime profile curves from the fetched properties.

    Runs locally after ``fetch``; a no-op (with a message) if the properties
    file is missing or carries no (time, cost) trajectories.

    With ``pdf=False`` (default) draws the combined browsing PNG via
    :func:`render_anytime_profile`. With ``pdf=True`` writes one paper-ready
    vector PDF per panel via :func:`render_anytime_profile_pdfs`; here
    *outfile* is used as the filename stem (default
    ``<eval_dir>/<exp.name>-anytime-profile``), each panel appending
    ``-<panel>.pdf``.

    *display_names* optionally maps a raw algorithm/run-id name to the legend
    label it should show instead; passed straight through to the renderer."""

    def make_plot():
        import json

        properties_path = Path(exp.eval_dir) / "properties"
        if not properties_path.exists():
            print(f"No properties file at {properties_path}; run fetch first.")
            return
        with open(properties_path) as f:
            props = json.load(f)
        if pdf:
            stem = outfile or (
                Path(exp.eval_dir) / f"{exp.name}-anytime-profile"
            )
            render_anytime_profile_pdfs(
                props, stem, max_time=max_time, display_names=display_names
            )
        else:
            out = outfile or (
                Path(exp.eval_dir) / f"{exp.name}-anytime-profile.png"
            )
            render_anytime_profile(
                props, out, max_time=max_time, display_names=display_names
            )

    exp.add_step(name, make_plot)


def fetch_algorithm(exp, expname, algo, *, new_algo=None):
    """Fetch (and possibly rename) a single algorithm from *expname*."""
    new_algo = new_algo or algo

    def rename_and_filter(run):
        if run["algorithm"] == algo:
            run["algorithm"] = new_algo
            run["id"][0] = new_algo
            return run
        return False

    exp.add_fetcher(
        f"data/{expname}-eval",
        filter=rename_and_filter,
        name=f"fetch-{new_algo}-from-{expname}",
        merge=True,
    )


def fetch_algorithms(exp, expname, *, algos=None, name=None, filters=None):
    """
    Fetch multiple or all algorithms.
    """
    assert not expname.rstrip("/").endswith("-eval")
    algos = set(algos or [])
    filters = filters or []
    if algos:

        def algo_filter(run):
            return run["algorithm"] in algos

        filters.append(algo_filter)

    exp.add_fetcher(
        f"data/{expname}-eval",
        filter=filters,
        name=name or f"fetch-from-{expname}",
        merge=True,
    )


def add_absolute_report(exp, *, name=None, outfile=None, **kwargs):
    report = AbsoluteReport(**kwargs)
    if name and not outfile:
        outfile = f"{name}.{report.output_format}"
    elif outfile and not name:
        name = Path(outfile).name
    elif not name and not outfile:
        name = f"{exp.name}-abs"
        outfile = f"{name}.{report.output_format}"

    if not Path(outfile).is_absolute():
        outfile = Path(exp.eval_dir) / outfile

    exp.add_report(report, name=name, outfile=outfile)
    if not REMOTE:
        exp.add_step(f"open-{name}", subprocess.call, ["xdg-open", outfile])


def add_comparative_report(exp, algorithm_pairs, *, name=None, outfile=None, **kwargs):
    report = ComparativeReport(algorithm_pairs=algorithm_pairs, **kwargs)
    if name and not outfile:
        outfile = f"{name}.{report.output_format}"
    elif outfile and not name:
        name = Path(outfile).name
    elif not name and not outfile:
        name = f"{exp.name}-cmp"
        outfile = f"{name}.{report.output_format}"

    if not Path(outfile).is_absolute():
        outfile = Path(exp.eval_dir) / outfile

    exp.add_report(report, name=name, outfile=outfile)
    if not REMOTE:
        exp.add_step(f"open-{name}", subprocess.call, ["xdg-open", outfile])



def add_scatter_plot_reports(exp, algorithm_pairs, attributes, *, filter=None):
    suffix = "-relative" if RELATIVE else ""
    for algo1, algo2 in algorithm_pairs:
        for attribute in attributes:
            exp.add_report(
                ScatterPlotReport(
                    relative=RELATIVE,
                    get_category=None if TEX else lambda run1, run2: run1["domain"],
                    attributes=[attribute],
                    filter_algorithm=[algo1, algo2],
                    filter=[add_evaluations_per_time, group_domains]
                    + tools.make_list(filter),
                    format="tex" if TEX else "png",
                ),
                name=f"{exp.name}-{algo1}-{algo2}-{attribute}{suffix}",
            )


def check_initial_h_value(run):
    h = run.get("initial_h_value")
    task = f"{run['domain']}:{run['problem']}"
    if h == 9223372036854775807 and task not in UNSOLVABLE_TASKS:
        tools.add_unexplained_error(run, f"infinite initial h value: {h}")
    return True


def check_search_started(run):
    if "search_start_time" not in run:
        error = run.get("error")
        if error not in ["search-unsolvable-incomplete", "translate-out-of-memory"]:
            tools.add_unexplained_error(run, f"search not started due to {error}")
    return True


class OptimalityCheckFilter:
    """Check that all algorithms have the same cost for commonly solved tasks.

    >>> from downward.reports.absolute import AbsoluteReport
    >>> filter = OptimalityCheckFilter()
    >>> report = AbsoluteReport(filter=[filter.check_costs])

    """

    def __init__(self):
        self.tasks_to_costs = defaultdict(list)
        self.warned_tasks = set()

    def _get_task(self, run):
        return (run["domain"], run["problem"])

    def check_costs(self, run):
        cost = run.get("cost")
        if cost is not None:
            assert run["coverage"]
            task = self._get_task(run)
            self.tasks_to_costs[task].append(cost)
            if (
                task not in self.warned_tasks
                and len(set(self.tasks_to_costs[task])) > 1
            ):
                tools.add_unexplained_error(
                    run,
                    f"found multiple costs for task {task}: "
                    f"{self.tasks_to_costs[task]}",
                )
                self.warned_tasks.add(task)
        return True
