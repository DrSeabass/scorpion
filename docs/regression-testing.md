# Search Regression Testing

The regression test suite checks that search metrics (plan cost, node counts,
heuristic values, anytime incumbent sequences) reproduce exactly after a code
change, and that runtime does not degrade by more than 2× geo-mean.

## Prerequisites

Set `AUTOSCALE_BENCHMARKS` to the root of the autoscale-benchmarks repository:

    export AUTOSCALE_BENCHMARKS=/path/to/autoscale-benchmarks

The directory must contain `21.11-optimal-strips/` and `21.11-agile-strips/`.

The harness validates emitted plans with [VAL](https://github.com/KCL-Planning/VAL)
by default.  Install VAL and ensure the `validate` binary is on your `PATH`
(or set the `VAL` environment variable, or pass `--validate-bin PATH`).
Pass `--skip-validate` to disable validation.

## Running the tests

### Default — developer iteration

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

### Plan validation (`--skip-validate`, `--validate-bin`)

Each successful run on the optimal, satisficing, and anytime tracks has
its emitted plan checked with VAL by default.  An invalid plan is a
hard failure (`failure_kind: "invalid_plan"` in the JSON output) and
its run's metrics are not written to the baseline in `--update` mode.
The heuristic track never validates plans; its primary metric is
h-values and the plan is incidental.

For anytime search, only the *final* (best) plan is validated;
intermediate incumbents are already exact-matched via the
`incumbent_costs` sequence.

Disabling and overrides:

    # disable validation entirely
    python misc/tests/test-regression.py --check --skip-validate

    # point at a specific VAL binary
    python misc/tests/test-regression.py --check --validate-bin /opt/val/validate

    # equivalent via env var
    VAL=/opt/val/validate python misc/tests/test-regression.py --check

If validation is enabled (default) and no VAL binary is found, the
harness exits at startup with a clear error rather than silently
skipping validation.

### Structured JSON output (`--json-output`)

Pass `--json-output PATH` (with `--check`) to write a per-instance
comparison dump alongside the human-readable stdout output:

    python misc/tests/test-regression.py --check --json-output /tmp/r.json

Top-level shape:

    {
      "schema_version": 1,
      "mode": "check",
      "scorpion_commit": "abc1234",
      "started_at": "2026-05-05T14:00:00+00:00",
      "instances": [1],
      "outcome": "pass" | "fail",
      "tracks": {
        "<track>": {
          "outcome": "pass" | "fail" | "error",
          "per_config": {
            "<config>": {
              "runtime_geomean_ratio": 1.08,
              "runtime_regression":    false,
              "runtime_compared_count": 42
            }
          },
          "runs": {
            "<config>|<domain>|<instance>.pddl": {
              "outcome": "pass" | "fail",
              "failure_kind": "exact_match" | "coverage_loss" | null,
              "metrics": {
                "<metric>": {"baseline": <v>, "current": <v>, "match": <bool>}
              }
            }
          }
        }
      }
    }

Every baseline run gets a `runs` entry (full dump, not failures-only).
Per-config aggregates (geo-mean ratio, regression flag) sit alongside
`runs`.  `--json-output` is invalid in `--update` mode; use the
committed `regression-baselines/*.json` files as the structured output
for updates.

### Running on a custom instance set (`--instances`)

`--instances ITEM [ITEM ...]` overrides the default instance scope.  Each
item is either an integer N (= `p0N.pddl` across every domain that has
one) or a `domain/problem.pddl` string for an exact instance:

    # rerun a single instance that failed
    python misc/tests/test-regression.py --check \
        --instances airport/p03.pddl --track optimal

    # broader subset: p01–p03 across all domains
    python misc/tests/test-regression.py --check --instances 1 2 3

    # mix: p01 across all domains + one extra ad-hoc instance
    python misc/tests/test-regression.py --check \
        --instances 1 airport/p03.pddl

`--instances` and `--full` are mutually exclusive.  Any `--instances`
invocation uses the 60 s per-instance time budget (the 10 s budget is
reserved for the default no-flag path).  On `--update`, results are
*merged* into the existing baseline file rather than overwriting it.

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

`--update` requires an explicit scope flag — `--full`, `--instances`, or
`--config-file` — to confirm the intended overwrite/merge semantics:

| Flag combination                  | Behavior                                                                |
|-----------------------------------|-------------------------------------------------------------------------|
| `--update --full`                 | Rebaseline everything for the selected tracks; **overwrites** the file. |
| `--update --instances ...`        | Regenerate only those instances; **merges** into the existing file.     |
| `--update --config-file ...`      | Regenerate only those configs; **merges** into the existing file.       |
| `--update` alone                  | Error: pick one of the above.                                           |

`generate_baseline.py` always passes `--update --full`, so the standard
rebaseline path is unchanged.

The default light-mode check uses a filtered subset of the full baseline
files; there are no separate `*-light.json` files.

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
   and implement `check_<track>` / `update_<track>` with the standard
   keyword args:

        def check_<track>(benchmarks, baseline_dir, workers, *,
                          configs=None, extra_configs=None, instances=None):
            ...
        def update_<track>(benchmarks, baseline_dir, workers, *,
                           configs=None, extra_configs=None, instances=None):
            ...

   Resolve `configs` via `resolve_configs(CONFIGS, configs, extra_configs)`.
   Default `instances` to `DEFAULT_LIGHT_INSTANCES` in `check_*` and
   `DEFAULT_FULL_INSTANCES` in `update_*`.  Filter the baseline with
   `filter_baseline(baseline, set(resolved_configs), resolved_instances)`.
   `update_*` should pass `merge=True` to `save_baseline` whenever any
   override was supplied (configs, extra_configs, or non-default instances —
   use `is_full_default_instances` to test).

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
