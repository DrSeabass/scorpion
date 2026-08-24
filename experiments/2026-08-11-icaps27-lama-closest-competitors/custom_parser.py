import logging
import re

from lab.parser import Parser
from lab import tools


class CommonParser(Parser):
    def add_repeated_pattern(
        self, name, regex, file="run.log", required=False, type=int
    ):
        def find_all_occurences(content, props):
            matches = re.findall(regex, content)
            if required and not matches:
                logging.error(f"Pattern {regex} not found in file {file}")
            props[name] = [type(m) for m in matches]

        self.add_function(find_all_occurences, file=file)

    def add_bottom_up_pattern(
        self, name, regex, file="run.log", required=False, type=int
    ):
        def search_from_bottom(content, props):
            reversed_content = "\n".join(reversed(content.splitlines()))
            match = re.search(regex, reversed_content)
            if required and not match:
                logging.error(f"Pattern {regex} not found in file {file}")
            if match:
                props[name] = type(match.group(1))

        self.add_function(search_from_bottom, file=file)


# Benign single-line messages that Scorpion/the translator prints to stderr but
# that do not indicate a failed run.  lab's fetcher flags *any* non-empty
# run.err as an "unexplained error", so we strip these lines during parsing
# (parser functions run in the run dir, before the fetcher reads run.err).
BENIGN_STDERR_LINE_PATTERNS = [
    # h^add overflows on tasks with very large action costs; the value is
    # clamped and the search continues normally.
    re.compile(r"^WARNING: overflow on h\^add! Costs clamped to \d+$"),
    # A duplicated atom in the initial state is a modelling quirk in some
    # benchmark domains (e.g. organic-synthesis); the translator ignores it.
    re.compile(r"^Warning: Atom .+ is specified twice in initial state specification$"),
]

# Markers that identify a VAL plan-validation failure.  Here the *planner*
# succeeded and only the external Validate binary returned non-zero, so this is
# not a search error and should not count as an unexplained error.
VAL_FAILURE_MARKERS = ("run_validate", "Validate", "returned non-zero exit status")


def clean_stderr(content, props):
    """Strip known-benign messages from run.err so they don't count as errors.

    Anything not recognised as benign is left in run.err untouched, so genuine
    errors are still reported. A compact record of what was suppressed is stored
    under ``suppressed_stderr`` for auditing.

    """
    if not content or not content.strip():
        return

    kept = []
    suppressed = []
    for line in content.splitlines():
        if any(p.match(line.strip()) for p in BENIGN_STDERR_LINE_PATTERNS):
            suppressed.append("benign-warning")
        else:
            kept.append(line)
    remainder = "\n".join(kept).strip()

    # If everything that's left is a VAL plan-validation traceback, that's
    # benign too: the plan was found, only external validation failed.
    if remainder and all(marker in remainder for marker in VAL_FAILURE_MARKERS):
        suppressed.append("validation-failed")
        remainder = ""

    if not suppressed:
        return

    props["suppressed_stderr"] = sorted(set(suppressed))

    # Rewrite run.err (cwd is the run dir) so the fetcher sees clean stderr.
    with open("run.err", "w") as f:
        f.write(remainder + "\n" if remainder else "")


def add_scores(content, props):
    """
    Convert some properties into scores in the range [0, 1].

    Best possible performance in a task is counted as 1, while worst
    performance is counted as 0.

    """
    try:
        max_time = props["limit_search_time"]
    except KeyError:
        print("search time limit missing -> can't compute time scores")
    else:
        props["search_start_time_score"] = tools.compute_log_score(
            props.get("coverage", 0), props.get("search_start_time"),
            lower_bound=1.0, upper_bound=max_time
        )

    try:
        max_memory_kb = props["limit_search_memory"] * 1024
    except KeyError:
        print("search memory limit missing -> can't compute memory score")
    else:
        props["search_start_memory_score"] = tools.compute_log_score(
            props.get("coverage", 0), props.get("search_start_memory"),
            lower_bound=2000, upper_bound=max_memory_kb
        )


def get_parser():
    parser = CommonParser()
    parser.add_pattern(
        "search_start_time",
        r"\[t=(.+)s, \d+ KB\] Initial heuristic value for .+: (?:\d+|infinity)\n",
        type=float,
    )

    # Wall-clock timestamp on every "Plan cost: N" announcement.  Paired
    # element-wise with the standard `cost:all` list to give (time, cost)
    # series for anytime trajectory plots.
    parser.add_repeated_pattern(
        "cost_times:all",
        r"\[t=([\d.]+)s,\s*\d+\s*KB\]\s*Plan cost:\s*\d+",
        type=float,
    )
    parser.add_pattern(
        "search_start_memory",
        r"\[t=.+s, (\d+) KB\] Initial heuristic value for .+: (?:\d+|infinity)\n",
        type=int,
    )

    parser.add_function(add_scores)

    # Strip known-benign messages from run.err so they don't count as
    # unexplained errors during fetching.
    parser.add_function(clean_stderr, file="run.err")

    parser.add_pattern("preprocessor_time", r"Preprocessor time: (.+)s\n", type=float)
    parser.add_pattern("preprocessor_memory", r"Preprocessor peak memory: (.+) KB\n", type=int)
    for name in ["task size", "variables", "facts", "operators", "mutex groups"]:
        parser.add_pattern(f"preprocessor_{name.replace(' ', '_')}", rf"Preprocessor {name}: (.+)\n", type=int)

    parser.add_pattern("preprocessor_merged_variable_groups", r"Merged (\d+) variable groups\n", type=int)
    parser.add_pattern("preprocessor_variables_merged", r"Total variables merged: (\d+)\n", type=int)
    parser.add_pattern("preprocessor_variables_eliminated_by_merging", r"Variables eliminated by merging: (\d+)\n", type=int)
    parser.add_pattern("preprocessor_variable_merging_time", r"Variable merging time: (.+)s\n", type=float)

    parser.add_pattern("preprocessor_mutex_computation_time", r"Mutex computation completed in (.+?)s \(\d+ iterations\)\n", type=float)
    parser.add_pattern("preprocessor_mutex_computation_iterations", r"Mutex computation completed in .+?s \((\d+) iterations\)\n", type=int)

    return parser
