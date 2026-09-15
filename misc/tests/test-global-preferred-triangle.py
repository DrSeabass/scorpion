"""Integration tests for global FIFO preferred sweeps (requires a release build).

Run with: python3 misc/tests/test-global-preferred-triangle.py
"""

from collections import defaultdict
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DOWNWARD = ROOT / "builds/release/bin/downward"
ALGORITHMS = ("round_robin_triangle", "lazy_multi_triangle")


def independent_goals(count):
    """Each operator sets one bit once; valid plans set every bit exactly once."""
    lines = ["begin_version", "3", "end_version", "begin_metric", "1",
             "end_metric", str(count)]
    for index in range(count):
        lines += ["begin_variable", f"var{index}", "-1", "2",
                  f"Atom unset{index}()", f"Atom set{index}()", "end_variable"]
    lines += ["0", "begin_state"] + ["0"] * count + ["end_state", "begin_goal", str(count)]
    lines += [f"{index} 1" for index in range(count)] + ["end_goal", str(count)]
    for index in range(count):
        lines += ["begin_operator", f"set{index}", "0", "1",
                  f"0 {index} 0 1", "1", "end_operator"]
    return "\n".join(lines + ["0", ""])


class GlobalPreferredTriangleTest(unittest.TestCase):
    def run_search(self, algorithm, options, count=7, evaluators="ff(),cea()"):
        first, second = evaluators.split(",")
        search = (f"let(h,{first},let(h2,{second},{algorithm}([h,h2],"
                  f"verbosity=debug,{options})))")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(DOWNWARD), "--search", search],
                input=independent_goals(count), text=True, capture_output=True,
                cwd=directory, timeout=15)
            plan_path = Path(directory) / "sas_plan"
            plan = plan_path.read_text() if plan_path.exists() else ""
        return result.returncode, result.stdout + result.stderr, plan

    def check_plan(self, plan, count):
        actions = re.findall(r"^\(set(\d+)\)$", plan, re.MULTILINE)
        self.assertEqual(sorted(map(int, actions)), list(range(count)))

    def test_global_fifo_and_stratified_guidance(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                code, output, plan = self.run_search(
                    algorithm, "preferred_evals=[h,h2],schedule=sweep,global_preferred=true")
                self.assertEqual(code, 0, output)
                self.check_plan(plan, 7)
                sweeps = defaultdict(list)
                for sweep, queue, depth in re.findall(
                        r"Triangle expansion: sweep=(\d+) queue=(\w+) depth=(\d+)", output):
                    sweep, depth = int(sweep), int(depth)
                    self.assertEqual(queue, "preferred" if sweep % 3 == 2 else "guidance")
                    sweeps[sweep].append(depth)
                # FIFO can serve the same depth repeatedly and later cross depths.
                # Stale copies of nodes expanded by guidance do not consume slots.
                self.assertEqual(sweeps[2], [1, 1])
                self.assertEqual(sweeps[5], [1, 1, 2, 2])
                for sweep, depths in sweeps.items():
                    if sweep % 3 != 2:
                        self.assertEqual(depths, sorted(set(depths)))

    def test_exhaustion_with_duplicate_and_stale_entries(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                code, output, plan = self.run_search(
                    algorithm, "preferred_evals=[h,h2],schedule=sweep,"
                    "global_preferred=true,bound=7")
                self.assertEqual(code, 12, output)
                self.assertIn("All open lists are empty", output)
                self.assertFalse(plan)

    def test_empty_preferred_queue(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                code, output, plan = self.run_search(
                    algorithm, "preferred_evals=[h,h2],schedule=sweep,global_preferred=true",
                    evaluators="blind(),blind()")
                self.assertEqual(code, 0, output)
                self.assertNotIn("queue=preferred", output)
                self.check_plan(plan, 7)

    def test_slopes_and_reopening_options(self):
        for algorithm in ALGORITHMS:
            for reopen in ("true", "false"):
                with self.subTest(algorithm=algorithm, reopen=reopen):
                    code, output, plan = self.run_search(
                        algorithm, "preferred_evals=[h,h2],schedule=sweep,"
                        f"global_preferred=true,slope=3,reopen_closed={reopen}")
                    self.assertEqual(code, 0, output)
                    self.check_plan(plan, 7)

    def test_depth_preferred_batches(self):
        for algorithm in ALGORITHMS:
            for schedule in ("sweep", "depth"):
                for mode in ("k=3", "expand_equal=true"):
                    with self.subTest(algorithm=algorithm, schedule=schedule, mode=mode):
                        code, output, plan = self.run_search(
                            algorithm, f"preferred_evals=[h,h2],schedule={schedule},{mode}")
                        self.assertEqual(code, 0, output)
                        self.check_plan(plan, 7)

    def test_invalid_options(self):
        for algorithm in ALGORITHMS:
            for options in ("schedule=depth,preferred_evals=[h,h2]",
                            "schedule=sweep,preferred_evals=[]"):
                with self.subTest(algorithm=algorithm, options=options):
                    code, output, _ = self.run_search(algorithm, options + ",global_preferred=true")
                    self.assertEqual(code, 33, output)
                    self.assertIn("global_preferred requires", output)

    def test_default_mode_unchanged(self):
        for algorithm in ALGORITHMS:
            for schedule in ("sweep", "depth"):
                with self.subTest(algorithm=algorithm, schedule=schedule):
                    options = f"preferred_evals=[h,h2],schedule={schedule}"
                    implicit = self.run_search(algorithm, options)
                    explicit = self.run_search(algorithm, options + ",global_preferred=false")
                    self.assertEqual(implicit[0], 0, implicit[1])
                    self.assertEqual(explicit[0], 0, explicit[1])
                    self.assertEqual(implicit[2], explicit[2])
                    for statistic in ("Expanded", "Evaluated", "Generated"):
                        pattern = rf"{statistic} \d+ state\(s\)"
                        self.assertEqual(re.findall(pattern, implicit[1]),
                                         re.findall(pattern, explicit[1]))


if __name__ == "__main__":
    unittest.main()
