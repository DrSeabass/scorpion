#!/usr/bin/env python3
"""Export per-domain coverage from Downward Lab evaluation properties (JSON)."""

import argparse
import csv
import json
from pathlib import Path
import sys


EXPERIMENTS = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = (
    "2026-08-31-round-robin-triangle",
    "2026-09-01-eager-po-round-robin",
    "2026-09-01-lazy-po-round-robin",
    "2026-09-01-round-robin-triangle-slopes",
    "2026-09-04-greedy-po-round-robin",
    "2026-08-11-icaps27-lama-closest-competitors",
    "2026-09-08-cost-insensitive-triangle",
)


def discover(inputs):
    """Discover evaluation files only, avoiding individual raw-run properties."""
    files = set()
    for path in inputs:
        if path.is_file():
            found = [path]
        elif path.is_dir():
            found = ([path / "properties"] if (path / "properties").is_file()
                     else list(path.glob("**/*-eval/properties")))
        else:
            raise ValueError(f"Input does not exist: {path}")
        if not found:
            print(f"Warning: no evaluation properties found in {path}", file=sys.stderr)
        files.update(p.resolve() for p in found)
    return sorted(files)


def aggregate(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a nonempty mapping of run IDs to properties")
    cells = {}
    seen = set()
    for run_id, run in data.items():
        if not isinstance(run, dict):
            raise ValueError(f"{path}: {run_id}: expected run properties")
        for key in ("algorithm", "domain", "problem"):
            if not isinstance(run.get(key), str) or not run[key]:
                raise ValueError(f"{path}: {run_id}: missing or invalid {key}")
        key = (run["domain"], run["algorithm"])
        identity = (*key, run["problem"])
        if identity in seen:
            raise ValueError(f"{path}: duplicate domain/algorithm/problem: {identity}")
        seen.add(identity)
        coverage = run.get("coverage")
        if coverage is not None and coverage not in (0, 1):
            raise ValueError(f"{path}: {run_id}: invalid coverage {coverage!r}")
        # Missing coverage is counted separately, never silently treated as failure.
        counts = cells.setdefault(key, [0, 0, 0])
        counts[0] += coverage or 0
        counts[1] += 1
        counts[2] += coverage is None
    return cells


def table(cells, columns=None):
    domains = sorted({domain for domain, _ in cells})
    totals = {}
    for (_, algorithm), counts in cells.items():
        totals[algorithm] = totals.get(algorithm, 0) + counts[0]
    algorithms = (columns if columns is not None else
                  sorted(totals, key=lambda algorithm: (-totals[algorithm], algorithm)))
    rows = [["domain", *algorithms]]
    for domain in domains:
        rows.append([domain, *[
            ("NA" if (domain, algorithm) not in cells
             or cells[domain, algorithm][2] else cells[domain, algorithm][0])
            for algorithm in algorithms
        ]])
    rows.append(["TOTAL", *[
        ("NA" if any(cells[d, a][2] for d, a in cells if a == algorithm)
         else totals[algorithm])
        for algorithm in algorithms
    ]])
    return rows


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def cost_variant(path):
    """Identify the dedicated unit-cost experiment, including suffixed reruns."""
    return ("unit cost" if path.parent.name.startswith(
        "2026-09-08-A-cost-insensitive-triangle-cost-one") else "original")


def combined_label(path, algorithm, sources):
    variant = cost_variant(path)
    label = f"{algorithm} [{variant}]"
    if sum(cost_variant(p) == variant for p in sources[algorithm]) > 1:
        label += f" [{path.parent.name}]"
    return label


def combination_sources(results):
    sources = {}
    for path, cells in results:
        for algorithm in {a for _, a in cells}:
            sources.setdefault(algorithm, set()).add(path)
    return sources


def combined_columns(results):
    """Group by algorithm name, original results first, preserving reruns."""
    sources = combination_sources(results)
    return [combined_label(path, algorithm, sources)
            for algorithm in sorted(sources)
            for path in sorted(sources[algorithm], key=lambda p: (cost_variant(p), str(p)))]


def combine(results):
    sources = combination_sources(results)
    combined = {}
    for path, cells in results:
        for (domain, algorithm), counts in cells.items():
            label = combined_label(path, algorithm, sources)
            if (domain, label) in combined:
                raise ValueError(f"Ambiguous combined column: {label}")
            combined[domain, label] = counts
    return combined


def has_required_runs(cells, required):
    domains = {d for d, _ in cells}
    algorithms = {a for _, a in cells}
    return all((d, a) in cells and cells[d, a][1] == required
               for d in domains for a in algorithms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="Properties files, evaluation directories, or experiment directories; "
                        "default: the seven round-robin/LAMA and unit-cost experiments.")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=EXPERIMENTS / "analysis" / "coverage",
                        help="Output directory (default: experiments/analysis/coverage).")
    parser.add_argument("--runs-per-domain", type=int, default=30,
                        help="Require this many runs in every domain/algorithm cell "
                        "or discard the entire file (default: 30).")
    args = parser.parse_args()
    if args.runs_per_domain < 1:
        parser.error("--runs-per-domain must be positive")
    try:
        files = discover(args.inputs or [EXPERIMENTS / n for n in DEFAULT_EXPERIMENTS])
        if not files:
            raise ValueError("No properties files found")
        results = [(path, aggregate(path)) for path in files]
        excluded = [path for path, cells in results
                    if not has_required_runs(cells, args.runs_per_domain)]
        for path in excluded:
            print(f"Excluded {path}: not {args.runs_per_domain} runs in every "
                  "domain/algorithm cell", file=sys.stderr)
        results = [(path, cells) for path, cells in results if path not in excluded]
        if not results:
            raise ValueError("No properties files meet the required run count")
        combined = combine(results)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = ["# Coverage by domain", "",
              "Cells show solved task counts. NA means absent runs or missing coverage. "
              "Totals sum the recorded runs for each algorithm. "
              "See coverage-long.csv for run counts and missing coverage; "
              "different files may contain different task sets.", ""]
    report.extend([f"Only files with {args.runs_per_domain} runs per domain and "
                   "algorithm are included. Columns are grouped alphabetically by algorithm, "
                   "with original results followed by their unit-cost versions. "
                   "Repeated results within a variant remain separate and are labeled by experiment.", "",
                   "Unit-cost versions come from the September 8 experiment: "
                   "search uses `cost_type=one`, and both FF and landmark-sum use "
                   "`transform=adapt_costs(one)`. Reported plan costs still use original action costs. "
                   "These runs stop at the first solution; older boosted triangle runs used anytime search. "
                   "The original columns retain their original experiment settings. "
                   "Algorithms without a matching variant are shown on their own; "
                   "`lama` and `lama-first-phase` are distinct configurations.", "",
                   "[Combined CSV table](coverage-by-domain.csv)", ""])
    rows = table(combined, combined_columns(results))
    with (args.output_dir / "coverage-by-domain.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    for index, row in enumerate(rows):
        report.append("| " + " | ".join(map(markdown_cell, row)) + " |")
        if index == 0:
            report.append("| --- |" + " ---: |" * (len(row) - 1))
    report.append("")
    if excluded:
        report.extend(["Excluded files:", ""])
        report.extend(f"- `{path}` (run count mismatch)" for path in excluded)
        report.append("")
    with (args.output_dir / "coverage-long.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["properties_file", "domain", "algorithm", "solved",
                         "runs", "missing_coverage"])
        for index, (path, cells) in enumerate(results, 1):
            name = f"{index:02d}-{path.parent.name}.csv"
            rows = table(cells)
            with (args.output_dir / name).open("w", newline="") as table_stream:
                csv.writer(table_stream).writerows(rows)
            for (domain, algorithm), counts in sorted(cells.items()):
                writer.writerow([str(path), domain, algorithm, *counts])
            report.extend([f"## {path.parent.name}", "", f"Source: `{path}`", "",
                           f"[CSV table]({name})", ""])
            for row_index, row in enumerate(rows):
                report.append("| " + " | ".join(map(markdown_cell, row)) + " |")
                if row_index == 0:
                    report.append("| --- |" + " ---: |" * (len(row) - 1))
            report.append("")
    (args.output_dir / "coverage.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {len(results)} coverage tables, coverage-by-domain.csv, "
          "coverage-long.csv, and coverage.md "
          f"to {args.output_dir}")


if __name__ == "__main__":
    main()
