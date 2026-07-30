#!/usr/bin/env python3
"""Export aaai27-bsor results to an Apache Arrow IPC file for sharing.

Reads a Lab ``-eval/properties`` file (the aggregated, parsed per-run JSON) and
writes one flat table in the Arrow IPC *file* format (random-access, with
footer -- the canonical shareable ``.arrow`` file). One row per
(config, domain, instance) run.

Each config name encodes three experiment axes, which we split into their own
columns:
    bsor-a500-w1_05  -> algorithm=bsor,  aspect=500,  suboptimality_bound=1.05
    rrr-a500-w2_5    -> algorithm=rrr,   aspect=500,  suboptimality_bound=2.5
    rrdex-w1_1       -> algorithm=rrdex, aspect=<null>, suboptimality_bound=1.1

The four requested per-run measurements are:
    generated, expansions, search_time (time to solution), cost (solution cost)
plus total_time (wall clock incl. translation) as a convenience.

A run has a solution iff coverage == 1. Runs that found no solution
(timeout / out-of-memory / translate failure) record none of these numbers, so
per the consumer's request every measurement column is float64 and unsolved
runs are filled with NaN (`solved` is False for those rows). Set
BSOR_UNSOLVED_SENTINEL=inf to use +inf instead of NaN.

Environment knobs:
    BSOR_EVAL_DIR          source ``-eval`` dir (default the min-eval run)
    BSOR_OUT               output path; the extension is adjusted per format
                           (default data/export/aaai27-bsor.arrow)
    BSOR_FORMAT            arrow | csv | both  (default arrow)
    BSOR_COMPRESSION       arrow buffer codec: zstd | lz4 | none (default zstd)
    BSOR_UNSOLVED_SENTINEL nan (default) | inf -- value for unsolved runs

Every Arrow file written is self-checked with Table.validate(full=True).

Usage:
    uv run --with pyarrow python export_arrow.py            # arrow, zstd
    BSOR_FORMAT=both uv run --with pyarrow python export_arrow.py
    BSOR_FORMAT=csv uv run --with pyarrow python export_arrow.py
"""

import datetime
import json
import math
import os
import re
from pathlib import Path

import pyarrow as pa

DIR = Path(__file__).resolve().parent
DEFAULT_EVAL = DIR / "data" / "2026-07-10-A-bsor-rrr-rrdex-wastar-min-eval"

EVAL_DIR = Path(os.environ.get("BSOR_EVAL_DIR", str(DEFAULT_EVAL)))
OUT_PATH = Path(os.environ.get("BSOR_OUT", str(DIR / "data" / "export" / "aaai27-bsor.arrow")))
SENTINEL = math.inf if os.environ.get("BSOR_UNSOLVED_SENTINEL") == "inf" else math.nan
FORMAT = os.environ.get("BSOR_FORMAT", "arrow").lower()
if FORMAT not in ("arrow", "csv", "both"):
    raise SystemExit(f"BSOR_FORMAT must be arrow|csv|both, got {FORMAT!r}")
_codec = os.environ.get("BSOR_COMPRESSION", "zstd").lower()
COMPRESSION = None if _codec in ("none", "", "off") else _codec

# config name -> (algorithm, aspect, suboptimality_bound)
_ASPECT_RE = re.compile(r"^a(\d+)$")
_WEIGHT_RE = re.compile(r"^w(.+)$")


def parse_config(name):
    parts = name.split("-")
    algorithm = parts[0]
    aspect = None
    weight = None
    for token in parts[1:]:
        m = _ASPECT_RE.match(token)
        if m:
            aspect = int(m.group(1))
            continue
        m = _WEIGHT_RE.match(token)
        if m:
            # weight tag writes the decimal point as '_' (1_05 -> 1.05)
            weight = float(m.group(1).replace("_", "."))
    if weight is None:
        raise ValueError(f"no weight token in config name {name!r}")
    return algorithm, aspect, weight


def main():
    props_path = EVAL_DIR / "properties"
    if not props_path.is_file():
        raise SystemExit(f"no properties file at {props_path}")
    print(f"[export] reading {props_path}")
    with open(props_path) as fh:
        data = json.load(fh)
    print(f"[export] {len(data)} runs")

    cols = {k: [] for k in (
        "algorithm", "aspect", "suboptimality_bound", "config",
        "domain", "instance", "solved",
        "generated", "expansions", "search_time", "total_time", "cost",
    )}

    def measure(run, key):
        # present iff the run found a solution (coverage == 1)
        val = run.get(key)
        return float(val) if val is not None else SENTINEL

    for run in data.values():
        config = run["algorithm"]
        algorithm, aspect, bound = parse_config(config)
        solved = run.get("coverage") == 1
        cols["algorithm"].append(algorithm)
        cols["aspect"].append(aspect)
        cols["suboptimality_bound"].append(bound)
        cols["config"].append(config)
        cols["domain"].append(run["domain"])
        cols["instance"].append(run["problem"])
        cols["solved"].append(bool(solved))
        cols["generated"].append(measure(run, "generated"))
        cols["expansions"].append(measure(run, "expansions"))
        cols["search_time"].append(measure(run, "search_time"))
        cols["total_time"].append(measure(run, "total_time"))
        cols["cost"].append(measure(run, "cost"))

    schema = pa.schema(
        [
            pa.field("algorithm", pa.string()),
            pa.field("aspect", pa.int64()),  # nullable; null for rrdex
            pa.field("suboptimality_bound", pa.float64()),
            pa.field("config", pa.string()),
            pa.field("domain", pa.string()),
            pa.field("instance", pa.string()),
            pa.field("solved", pa.bool_()),
            pa.field("generated", pa.float64()),
            pa.field("expansions", pa.float64()),
            pa.field("search_time", pa.float64()),
            pa.field("total_time", pa.float64()),
            pa.field("cost", pa.float64()),
        ],
        metadata={
            "source_experiment": EVAL_DIR.name,
            "generated_at": datetime.datetime.now().astimezone().isoformat(),
            "unsolved_sentinel": "inf" if SENTINEL == math.inf else "nan",
            "notes": (
                "One row per (config, domain, instance). suboptimality_bound is "
                "the weight w. aspect is null for algorithms without an aspect "
                "axis (rrdex). Measurement columns (generated, expansions, "
                "search_time, total_time, cost) hold the sentinel when solved is "
                "False. search_time is search-only time to solution; total_time "
                "includes translation."
            ),
        },
    )

    table = pa.table(cols, schema=schema)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if FORMAT in ("arrow", "both"):
        arrow_path = OUT_PATH.with_suffix(".arrow")
        # Arrow IPC file format (random-access, with footer). compression keeps
        # the share small; downstream Arrow readers decompress transparently.
        options = pa.ipc.IpcWriteOptions(compression=COMPRESSION)
        with pa.OSFile(str(arrow_path), "wb") as sink:
            with pa.ipc.new_file(sink, schema, options=options) as writer:
                writer.write_table(table)
        # Self-check: reopen and fully validate what we just wrote.
        reopened = pa.ipc.open_file(arrow_path).read_all()
        reopened.validate(full=True)
        assert reopened.num_rows == table.num_rows
        size = arrow_path.stat().st_size
        print(f"[export] wrote {table.num_rows} rows x {table.num_columns} cols "
              f"(compression={COMPRESSION or 'none'}) -> {arrow_path} "
              f"({size/1024:.1f} KiB) [validate(full=True) OK]")

    if FORMAT in ("csv", "both"):
        import pyarrow.csv as pacsv
        csv_path = OUT_PATH.with_suffix(".csv")
        # NaN/inf render as empty/inf in CSV; keep them literal so the unsolved
        # sentinel survives a round-trip through the text file.
        write_opts = pacsv.WriteOptions(quoting_style="needed")
        pacsv.write_csv(table, csv_path, write_options=write_opts)
        size = csv_path.stat().st_size
        print(f"[export] wrote CSV -> {csv_path} ({size/1024:.1f} KiB)")

    print(f"[export] algorithms: {sorted(set(cols['algorithm']))}")
    print(f"[export] bounds: {sorted(set(cols['suboptimality_bound']))}")
    print(f"[export] domains: {len(set(cols['domain']))}, "
          f"instances: {len(set(zip(cols['domain'], cols['instance'])))}, "
          f"solved rows: {sum(cols['solved'])}")


if __name__ == "__main__":
    main()
