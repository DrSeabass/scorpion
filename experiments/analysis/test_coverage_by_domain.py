import json
from pathlib import Path
import tempfile
import unittest

from coverage_by_domain import aggregate, combine, combined_columns, discover, has_required_runs, table


class CoverageTest(unittest.TestCase):
    def test_required_runs_rejects_partial_files(self):
        self.assertTrue(has_required_runs({("d", "a"): [12, 30, 0]}, 30))
        self.assertFalse(has_required_runs({("d", "a"): [1, 1, 0]}, 30))
        self.assertFalse(has_required_runs({
            ("d1", "a"): [12, 30, 0], ("d2", "b"): [12, 30, 0]}, 30))

    def test_combination_preserves_repeated_algorithms(self):
        cells = combine([
            (Path("first-eval/properties"), {("d", "a"): [12, 30, 0]}),
            (Path("second-eval/properties"), {("d", "a"): [13, 30, 0]}),
        ])
        self.assertEqual(cells["d", "a [original] [first-eval]"], [12, 30, 0])
        self.assertEqual(cells["d", "a [original] [second-eval]"], [13, 30, 0])

    def test_unit_cost_pairing_preserves_counts_and_unmatched_algorithms(self):
        results = [
            (Path("old-eval/properties"), {
                ("d", "a"): [12, 30, 0], ("d", "lama"): [29, 30, 0]}),
            (Path("rerun-eval/properties"), {("d", "a"): [13, 30, 0]}),
            (Path("2026-09-08-A-cost-insensitive-triangle-cost-one-eval/properties"), {
                ("d", "a"): [25, 30, 0], ("d", "b"): [30, 30, 0],
                ("d", "lama-first-phase"): [28, 30, 0]}),
        ]
        columns = combined_columns(results)
        self.assertEqual(columns, [
            "a [original] [old-eval]", "a [original] [rerun-eval]", "a [unit cost]",
            "b [unit cost]", "lama [original]", "lama-first-phase [unit cost]",
        ])
        self.assertEqual(table(combine(results), columns), [
            ["domain", *columns], ["d", 12, 13, 25, 30, 29, 28],
            ["TOTAL", 12, 13, 25, 30, 29, 28],
        ])

    def test_counts_missing_and_absent_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "properties"
            path.write_text(json.dumps({
                "a": dict(domain="d1", algorithm="a", problem="p1", coverage=1),
                "b": dict(domain="d1", algorithm="a", problem="p2", coverage=0),
                "c": dict(domain="d2", algorithm="a", problem="p1"),
                "d": dict(domain="d2", algorithm="b", problem="p1", coverage=0),
            }))
            cells = aggregate(path)
            self.assertEqual(cells["d1", "a"], [1, 2, 0])
            self.assertEqual(cells["d2", "a"], [0, 1, 1])
            self.assertEqual(table(cells), [
                ["domain", "a", "b"], ["d1", 1, "NA"],
                ["d2", "NA", 0], ["TOTAL", "NA", 0],
            ])

    def test_reject_duplicate_tasks_and_invalid_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "properties"
            run = dict(domain="d", algorithm="a", problem="p", coverage=1)
            path.write_text(json.dumps({"one": run, "two": run}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                aggregate(path)
            run["coverage"] = "1"
            path.write_text(json.dumps({"one": run}))
            with self.assertRaisesRegex(ValueError, "invalid coverage"):
                aggregate(path)

    def test_discovery_excludes_raw_runs_and_deduplicates_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "example-eval" / "properties"
            raw = root / "data" / "example" / "runs-00001-00100" / "00001" / "properties"
            for path in (evaluation, raw):
                path.parent.mkdir(parents=True)
                path.write_text("{}")
            self.assertEqual(discover([root, evaluation]), [evaluation.resolve()])


if __name__ == "__main__":
    unittest.main()
