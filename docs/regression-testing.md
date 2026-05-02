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

Runs p01 only, 1–2 configs per track, 10 s per instance.  Target: under 5 min.

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

## Rebaselining

After an intentional algorithm or configuration change, regenerate the baselines:

    python misc/tests/generate_baseline.py           # both light and full
    python misc/tests/generate_baseline.py --light-only
    python misc/tests/generate_baseline.py --full-only
    python misc/tests/generate_baseline.py --track satisficing

Then commit the updated `misc/tests/regression-baselines/*.json` files.

`--workers N` sets the number of parallel workers (default: 4):

    python misc/tests/generate_baseline.py --workers 8

## Test tracks

| Track | Benchmark set | Configs (full) | Configs (light) | Exact-match keys |
|---|---|---|---|---|
| heuristics | optimal-strips | 5 A* configs | astar_lmcut, astar_cegar | initial_h_value, expansions, evaluations, generated, cost |
| optimal | optimal-strips | 6 configs | astar_lmcut, astar_cegar | cost, expansions, evaluations, generated |
| satisficing | agile-strips | 4 configs | eager_greedy_ff, lama-first | cost, expansions, evaluations, generated |
| anytime | agile-strips | 2 configs | iterated_wa_ff | incumbent_costs (full improving sequence) |

## Adding a new track

1. Create `misc/tests/regression_<track>.py` following the pattern of
   `regression_optimal.py`.  Define `CONFIGS`, `LIGHT_CONFIGS`,
   `TIME_LIMIT`, `LIGHT_TIME_LIMIT`, `EXACT_KEYS`, and implement
   `check_<track>` / `update_<track>` with `*, light: bool = True` kwargs.

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
