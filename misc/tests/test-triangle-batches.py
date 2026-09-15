"""Run with python3 misc/tests/test-triangle-batches.py after a release build."""

from collections import defaultdict
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


DOWNWARD = Path(__file__).resolve().parents[2] / "builds/release/bin/downward"
ALGORITHMS = ("round_robin_triangle", "lazy_multi_triangle")


def graph_task(edges, goal):
    """A single SAS variable encodes a graph, including unreachable goals."""
    size = max([goal] + [v for src, dst, cost in edges for v in (src, dst)]) + 1
    lines = ["begin_version", "3", "end_version", "begin_metric", "1", "end_metric",
             "1", "begin_variable", "location", "-1", str(size)]
    lines += [f"Atom at{i}()" for i in range(size)]
    lines += ["end_variable", "0", "begin_state", "0", "end_state",
              "begin_goal", "1", f"0 {goal}", "end_goal", str(len(edges))]
    for src, dst, cost in edges:
        lines += ["begin_operator", f"move-{src}-{dst}", "0", "1",
                  f"0 0 {src} {dst}", str(cost), "end_operator"]
    return "\n".join(lines + ["0", ""])


# All seven reachable states must be expanded to prove the goal unreachable.
# The depth-one FIFO insertion order interleaves g=1, g=2, g=1.
EDGES = [(0, 1, 1), (0, 2, 2), (0, 3, 1),
         (1, 4, 1), (2, 5, 1), (3, 6, 1)]


class TriangleBatchTest(unittest.TestCase):
    def run_search(self, algorithm, options, edges=EDGES, goal=7, heuristic="blind()"):
        config = (f"let(h,{heuristic},let(h2,{heuristic},"
                  f"{algorithm}([h,h2],verbosity=debug,{options})))")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([str(DOWNWARD), "--search", config],
                                    input=graph_task(edges, goal), text=True,
                                    capture_output=True, cwd=directory, timeout=15)
        return result.returncode, result.stdout + result.stderr

    def batches(self, output):
        batches = defaultdict(list)
        for sweep, depth, queue, h, g in re.findall(
                r"Triangle expansion: sweep=(\d+) queue=guidance depth=(\d+) "
                r"list=(\d+)(?: h=(-?\d+) g=(\d+))?", output):
            batches[int(sweep), int(depth)].append((int(queue), h, g))
        return batches

    def test_fixed_k_and_queue_rotation(self):
        for algorithm in ALGORITHMS:
            for schedule in ("sweep", "depth"):
                for k in (1, 2, 5):
                    with self.subTest(algorithm=algorithm, schedule=schedule, k=k):
                        code, output = self.run_search(algorithm, f"schedule={schedule},k={k}")
                        self.assertEqual(code, 12, output)
                        self.assertIn("Expanded 7 state(s)", output)
                        batches = self.batches(output)
                        at_depth_one = [batch for (_, depth), batch in batches.items() if depth == 1]
                        self.assertEqual(len(at_depth_one[0]), min(k, 3))
                        self.assertEqual(sum(map(len, at_depth_one)), 3)
                        for batch in batches.values():
                            self.assertLessEqual(len(batch), k)
                            self.assertEqual(len({entry[0] for entry in batch}), 1)
                        if schedule == "depth" and k == 2:
                            self.assertNotEqual(at_depth_one[0][0][0], at_depth_one[1][0][0])

    def test_equal_h_and_g_group(self):
        for algorithm in ALGORITHMS:
            for schedule in ("sweep", "depth"):
                with self.subTest(algorithm=algorithm, schedule=schedule):
                    code, output = self.run_search(algorithm, f"schedule={schedule},expand_equal=true")
                    self.assertEqual(code, 12, output)
                    self.assertIn("Expanded 7 state(s)", output)
                    batches = self.batches(output)
                    depth_one = [batch for (_, depth), batch in batches.items() if depth == 1]
                    self.assertEqual([len(batch) for batch in depth_one], [2, 1])
                    self.assertEqual([batch[0][2] for batch in depth_one], ["1", "2"])
                    for batch in batches.values():
                        self.assertEqual(len(set(batch)), 1)

    def test_equal_mode_drains_large_ties_and_stale_duplicates(self):
        # Parallel edges to state 1 add stale copies to lazy queues. They must
        # neither terminate the group early nor count as expansions.
        edges = [(0, i, 1) for i in range(1, 9)] + [(0, 1, 1)] * 3
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                code, output = self.run_search(algorithm, "schedule=sweep,expand_equal=true",
                                               edges=edges, goal=9)
                self.assertEqual(code, 12, output)
                self.assertIn("Expanded 9 state(s)", output)
                depth_one = [batch for (_, depth), batch in self.batches(output).items() if depth == 1]
                self.assertEqual([len(batch) for batch in depth_one], [8])

    def test_fixed_k_ignores_stale_duplicates(self):
        edges = [(0, i, 1) for i in range(1, 9)] + [(0, 1, 1)] * 3
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                code, output = self.run_search(algorithm, "schedule=sweep,k=3",
                                               edges=edges, goal=9)
                self.assertEqual(code, 12, output)
                self.assertIn("Expanded 9 state(s)", output)
                depth_one = [batch for (_, depth), batch in self.batches(output).items() if depth == 1]
                self.assertEqual([len(batch) for batch in depth_one], [3, 3, 2])

    def test_single_expansion_default(self):
        for algorithm in ALGORITHMS:
            for schedule in ("sweep", "depth"):
                with self.subTest(algorithm=algorithm, schedule=schedule):
                    implicit = self.run_search(algorithm, f"schedule={schedule}")
                    explicit = self.run_search(
                        algorithm, f"schedule={schedule},k=1,expand_equal=false")
                    self.assertEqual(implicit[0], 12, implicit[1])
                    self.assertEqual(explicit[0], 12, explicit[1])
                    self.assertEqual(self.batches(implicit[1]), self.batches(explicit[1]))

    def test_equal_h_boundary(self):
        # The two branches have different distances to the goal. Eager h
        # differs at depth 1; lazy parent h differs at depth 2.
        edges = [(0, 1, 1), (0, 2, 1), (1, 3, 1), (3, 5, 1),
                 (2, 4, 1), (4, 6, 1), (6, 5, 1)]
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                code, output = self.run_search(
                    algorithm, "schedule=sweep,expand_equal=true", edges=edges, goal=5,
                    heuristic="ff()")
                self.assertEqual(code, 0, output)
                batches = self.batches(output)
                for batch in batches.values():
                    self.assertEqual(len(set(batch)), 1)
                depth = 1 if algorithm == "round_robin_triangle" else 2
                first_batch = next(batch for (_, d), batch in batches.items() if d == depth)
                self.assertEqual(len(first_batch), 1)

    def test_multi_triangle_sweep_matches_eager_batches(self):
        for mode in ("k=1", "k=2", "expand_equal=true"):
            with self.subTest(mode=mode):
                options = f"schedule=sweep,{mode}"
                old = self.run_search("multi_triangle", options)
                new = self.run_search("round_robin_triangle", options)
                self.assertEqual(old[0], 12, old[1])
                self.assertEqual(self.batches(old[1]), self.batches(new[1]))

    def test_pop_fixed_k_rotates_each_live_expansion(self):
        edges = [(0, i, 1) for i in range(1, 9)] + [(0, 1, 1)] * 3
        code, output = self.run_search("multi_triangle", "schedule=pop,k=3",
                                       edges=edges, goal=9)
        self.assertEqual(code, 12, output)
        self.assertIn("Expanded 9 state(s)", output)
        batches = self.batches(output)
        entries = [entry for batch in batches.values() for entry in batch]
        self.assertEqual([entry[0] for entry in entries], [i % 2 for i in range(9)])
        self.assertEqual([len(batch) for (_, d), batch in batches.items() if d == 1], [3, 3, 2])

    def test_pop_equal_rotates_once_per_live_group(self):
        for options in ("", ",guide_by_pruning=true,pruning_heuristic=blind()"):
            with self.subTest(options=options):
                code, output = self.run_search(
                    "multi_triangle", "schedule=pop,expand_equal=true" + options)
                self.assertEqual(code, 12, output)
                self.assertIn("Expanded 7 state(s)", output)
                batches = list(self.batches(output).values())
                count = 3 if options else 2
                self.assertEqual([batch[0][0] for batch in batches],
                                 [i % count for i in range(len(batches))])
                self.assertTrue(any(len(batch) == 2 for batch in batches))
                for batch in batches:
                    self.assertEqual(len(set(batch)), 1)

    def test_multi_triangle_anytime_improves_incumbent(self):
        # A costly shallow goal is discovered before its cheaper deeper path.
        edges = [(0, 1, 1), (1, 4, 8), (0, 2, 1), (2, 3, 1), (3, 4, 1)]
        for mode in ("k=2", "expand_equal=true"):
            with self.subTest(mode=mode):
                code, output = self.run_search(
                    "multi_triangle", "schedule=pop,anytime=true," + mode +
                    ",guide_by_pruning=true,pruning_heuristic=blind()", edges=edges, goal=4)
                self.assertEqual(code, 0, output)
                self.assertIn("improved incumbent with cost 9", output)
                self.assertIn("improved incumbent with cost 3", output)

    def test_multi_triangle_pop_default(self):
        implicit = self.run_search("multi_triangle", "schedule=pop")
        explicit = self.run_search("multi_triangle", "schedule=pop,k=1,expand_equal=false")
        self.assertEqual(implicit[0], 12, implicit[1])
        self.assertEqual(self.batches(implicit[1]), self.batches(explicit[1]))

    def test_window_start_and_exhaustion(self):
        variants = [("multi_triangle", "sweep"), ("multi_triangle", "pop"),
                    ("round_robin_triangle", "sweep"), ("round_robin_triangle", "depth"),
                    ("lazy_multi_triangle", "sweep"), ("lazy_multi_triangle", "depth")]
        for algorithm, schedule in variants:
            for window in (0, 1, 100):
                with self.subTest(algorithm=algorithm, schedule=schedule, window=window):
                    code, output = self.run_search(algorithm, f"schedule={schedule},window={window}")
                    self.assertEqual(code, 12, output)
                    self.assertIn("Expanded 7 state(s)", output)
                    starts = re.findall(r"Triangle sweep start: depth=(\d+) front=(\d+) offset=(\d+)", output)
                    self.assertTrue(starts)
                    for depth, front, offset in starts:
                        self.assertEqual(int(depth), max(int(offset), int(front) - window))
                    current_start = None
                    for line in output.splitlines():
                        start = re.search(r"Triangle sweep start: depth=(\d+)", line)
                        expansion = re.search(r"Triangle expansion:.* depth=(\d+)", line)
                        if start:
                            current_start = int(start[1])
                        if expansion:
                            self.assertGreaterEqual(int(expansion[1]), current_start)

    def test_random_start_is_seeded_and_within_frontier(self):
        edges = [(i, i + 1, 1) for i in range(10)] + [(0, 11, 1), (0, 12, 1)]
        for algorithm in (*ALGORITHMS, "multi_triangle"):
            with self.subTest(algorithm=algorithm):
                options = "schedule=sweep,random_start=true,random_seed=17"
                first = self.run_search(algorithm, options, edges=edges, goal=13)
                second = self.run_search(algorithm, options, edges=edges, goal=13)
                self.assertEqual(first[0], 12, first[1])
                self.assertIn("Expanded 13 state(s)", first[1])
                pattern = r"Triangle sweep start: depth=(\d+) front=(\d+) offset=(\d+)"
                starts = re.findall(pattern, first[1])
                self.assertEqual(starts, re.findall(pattern, second[1]))
                self.assertEqual(self.batches(first[1]), self.batches(second[1]))
                self.assertTrue(any(int(d) > int(o) for d, f, o in starts))
                for depth, front, offset in starts:
                    self.assertLessEqual(int(offset), int(depth))
                    self.assertLessEqual(int(depth), int(front))

    def test_custom_starts_with_batches(self):
        for algorithm in (*ALGORITHMS, "multi_triangle"):
            for start in ("window=0", "random_start=true,random_seed=7"):
                for batch in ("k=3", "expand_equal=true"):
                    with self.subTest(algorithm=algorithm, start=start, batch=batch):
                        code, output = self.run_search(algorithm, f"schedule=sweep,{start},{batch}")
                        self.assertEqual(code, 12, output)
                        self.assertIn("Expanded 7 state(s)", output)

    def test_wide_window_and_disabled_start_preserve_expansions(self):
        for algorithm in (*ALGORITHMS, "multi_triangle"):
            with self.subTest(algorithm=algorithm):
                baseline = self.run_search(algorithm, "schedule=sweep")
                for options in ("window=-1,random_start=false", "window=100"):
                    result = self.run_search(algorithm, "schedule=sweep," + options)
                    self.assertEqual(result[0], baseline[0], result[1])
                    self.assertEqual(self.batches(result[1]), self.batches(baseline[1]))

    def test_custom_starts_find_valid_plans(self):
        for algorithm in (*ALGORITHMS, "multi_triangle"):
            for options in ("window=0", "window=1", "random_start=true,random_seed=42"):
                with self.subTest(algorithm=algorithm, options=options):
                    code, output = self.run_search(algorithm, "schedule=sweep," + options, goal=6)
                    self.assertEqual(code, 0, output)
                    state = 0
                    actions = re.findall(r"^move-(\d+)-(\d+) \(\d+\)$", output, re.MULTILINE)
                    self.assertTrue(actions)
                    for source, target in actions:
                        self.assertEqual(int(source), state)
                        self.assertTrue(any(src == state and dst == int(target) for src, dst, _ in EDGES))
                        state = int(target)
                    self.assertEqual(state, 6)

    def test_initial_goal_with_custom_start(self):
        for algorithm in (*ALGORITHMS, "multi_triangle"):
            for options in ("window=0", "random_start=true"):
                with self.subTest(algorithm=algorithm, options=options):
                    code, output = self.run_search(algorithm, options, edges=[], goal=0)
                    self.assertEqual(code, 0, output)
                    self.assertIn("Plan length: 0", output)

    def test_invalid_options(self):
        for algorithm in (*ALGORITHMS, "multi_triangle"):
            for options in ("k=0", "k=-1", "k=2,expand_equal=true",
                            "window=-2", "random_start=true,window=1"):
                with self.subTest(algorithm=algorithm, options=options):
                    code, output = self.run_search(algorithm, options)
                    self.assertEqual(code, 33, output)


if __name__ == "__main__":
    unittest.main()
