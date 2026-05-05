# Search Regression Testing

The regression test suite checks that search metrics (plan cost, node counts,
heuristic values, anytime incumbent sequences) reproduce exactly after a code
change, and that runtime does not degrade by more than 2× geo-mean.

## Prerequisites

Set `AUTOSCALE_BENCHMARKS` to the root of the autoscale-benchmarks repository:

    export AUTOSCALE_BENCHMARKS=/path/to/autoscale-benchmarks

The directory must contain `21.11-optimal-strips/` and `21.11-agile-strips/`.

## Running the tests

### Light mode (default) — developer iteration

Runs p01 only, all configs, 10 s per instance (60 s for anytime).
Compares against the matching subset of the full baseline files.

    tox -e regression

or directly:

    python misc/tests/test-regression.py --check

### Full mode — pre-merge CI gate

Runs p01–p05, all configs, 60 s per instance.

    tox -e regression-full

or directly:

    python misc/tests/test-regression.py --check --full

### Running a single track

Pass `--track` to check or update only one track:

    python misc/tests/test-regression.py --check --track satisficing
    python misc/tests/test-regression.py --check --full --track optimal satisficing

### Running custom configs (`--config-file`)

`--config-file PATH` replaces each selected track's `CONFIGS` with a JSON
dict for the duration of the invocation.  The file shape is

    {
      "my_algo":       ["--search", "astar(my_h())"],
      "my_other_algo": ["--evaluator", "h=my_h()", "--search", "astar(h)"]
    }

Typical usage targets a single track:

    python misc/tests/test-regression.py --check --full \
        --track optimal --config-file my-configs.json

The same dict is applied to every selected track, so `--config-file` is
usually combined with `--track`.  In `--check` mode, the comparison runs
against whichever entries already exist in the baseline (a custom config
not present in the baseline simply has nothing to compare against).  In
`--update` mode, the new entries are *merged* into the existing baseline
file — entries for non-overridden configs are preserved, entries for
overridden configs are replaced.

## Rebaselining

After an intentional algorithm or configuration change, regenerate the baselines:

    python misc/tests/generate_baseline.py
    python misc/tests/generate_baseline.py --track satisficing
    python misc/tests/generate_baseline.py --workers 8

Then commit the updated `misc/tests/regression-baselines/*.json` files.

Light mode uses a filtered subset of the full baseline files; there are no
separate `*-light.json` files.  Rebaselining always runs full mode.

## Test tracks

| Track | Benchmark set | Configs | Exact-match keys |
|---|---|---|---|
| heuristics | optimal-strips | 5 A* configs | initial_h_value, expansions, evaluations, generated, cost |
| optimal | optimal-strips | 6 configs | cost, expansions, evaluations, generated |
| satisficing | agile-strips | 4 configs | cost, expansions, evaluations, generated |
| anytime | agile-strips | 2 configs | incumbent_costs (full improving sequence) |

Light mode runs the same configs as full mode, only on p01 instead of p01–p05.

**Light mode time limits**: 10 s per instance for heuristics, optimal, and
satisficing; 60 s for anytime (anytime planners must run to budget to produce
a comparable incumbent sequence).

## Adding a new track

1. Create `misc/tests/regression_<track>.py` following the pattern of
   `regression_optimal.py`.  Define `CONFIGS`, `TIME_LIMIT`, `EXACT_KEYS`,
   and implement `check_<track>` / `update_<track>` with `*, light: bool = True`
   kwargs.  `update_<track>` must raise `ValueError` when `light=True` —
   baselines are only generated in full mode.  `check_<track>` should load the
   full baseline and call `filter_baseline(baseline, set(CONFIGS), 1)` when in
   light mode.

2. Import and register in `misc/tests/test-regression.py`:

        from regression_<track> import check_<track>, update_<track>

        TRACKS = [
            ...
            ("<track>", check_<track>, update_<track>),
        ]

3. Generate baselines:

        python misc/tests/generate_baseline.py --track <track>

4. Verify:

        python misc/tests/test-regression.py --check --track <track>
        python misc/tests/test-regression.py --check --full --track <track>
