# Search Regression Testing

The regression test suite checks that search metrics (plan cost, node counts,
heuristic values, anytime incumbent sequences) reproduce exactly after a code
change, and that runtime does not degrade by more than 2× geo-mean.

For iterating on an *in-development* algorithm against its own previous
iteration (rather than a committed baseline), see
[Algorithm Development Loop](algorithm-dev-loop.md), which uses the same
harness in `--dev` mode.

## Prerequisites

Set `AUTOSCALE_BENCHMARKS` to the root of the autoscale-benchmarks repository:

    export AUTOSCALE_BENCHMARKS=/path/to/autoscale-benchmarks

The directory must contain `21.11-optimal-strips/` and `21.11-agile-strips/`.
The `tox` envs and `generate_baseline.py` read this variable directly.
`test-regression.py` itself does not — see [`--domain-dir`](#--domain-dir)
below for how it locates instances.

The harness validates emitted plans with [VAL](https://github.com/KCL-Planning/VAL)
by default.  Install VAL and ensure the `validate` binary is on your `PATH`
(or set the `VAL` environment variable, or pass `--validate-bin PATH`).
Pass `--skip-validate` to disable validation.

## Running the tests

### Default — developer iteration

Runs p01 only, all configs, 10 s per instance (60 s for anytime).
Compares against the matching subset of the full baseline files.
Two sequential per-set invocations (heuristics + optimal against
`21.11-optimal-strips`, then satisficing + anytime against
`21.11-agile-strips`):

    tox -e regression

### Full mode — pre-merge CI gate

Runs p01–p05, all configs, 60 s per instance:

    tox -e regression-full

### Direct invocation

`test-regression.py` requires `--domain-dir PATH` and always runs
against a single benchmark set or single domain.  To replicate
`tox -e regression` by hand, run two commands:

    python misc/tests/test-regression.py --check \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-optimal-strips \
        --track heuristics optimal
    python misc/tests/test-regression.py --check \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-agile-strips \
        --track satisficing anytime

Add `--full` to get the pre-merge-CI scope.

### `--domain-dir`

`--domain-dir PATH` is required.  PATH may be either:

- A **benchmark set** — a directory whose children are per-domain dirs
  (`airport/`, `blocks/`, ...), each containing `p01.pddl`, `p02.pddl`,
  ...  Iterates every domain under PATH that has a matching problem
  file for each requested instance id.
- A **single domain** — a directory containing `p01.pddl` directly.
  Iterates just that one domain.

The two layouts are auto-detected by checking for `p01.pddl` directly
under PATH.  Single-domain mode is useful for reruns of one failing
instance:

    python misc/tests/test-regression.py --check \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-optimal-strips/airport \
        --instances 3 --track optimal

In single-domain mode, the comparison narrows the baseline to just
that domain's runs, so coverage_loss reports do not appear for
domains that were not run.  In `--update` mode, single-domain
invocations *always* merge into the existing baseline file rather
than overwriting it, regardless of whether the full instance set is
selected.

### Running a single track

Pass `--track` to check or update only one track:

    python misc/tests/test-regression.py --check \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-agile-strips \
        --track satisficing
    python misc/tests/test-regression.py --check --full \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-optimal-strips \
        --track heuristics optimal

#### Multi-track invocations and benchmark sets

A single `test-regression.py` call has one `--domain-dir`, so all
tracks listed via `--track` must target that benchmark set:

| Tracks                  | Benchmark set         |
|-------------------------|-----------------------|
| heuristics, optimal     | `21.11-optimal-strips` |
| satisficing, anytime    | `21.11-agile-strips`   |

To run all four tracks, issue two commands (or use `tox -e
regression[-full]`, which sequences them).  `generate_baseline.py`
applies the same per-set split for `--update` and forwards every
other flag unchanged.

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

Disabling and overrides (omit `--domain-dir` for brevity in the
snippets below — it is still required):

    # disable validation entirely
    python misc/tests/test-regression.py --check --skip-validate ...

    # point at a specific VAL binary
    python misc/tests/test-regression.py --check --validate-bin /opt/val/validate ...

    # equivalent via env var
    VAL=/opt/val/validate python misc/tests/test-regression.py --check ...

If validation is enabled (default) and no VAL binary is found, the
harness exits at startup with a clear error rather than silently
skipping validation.

### Structured JSON output (`--json-output`)

Pass `--json-output PATH` (with `--check`) to write a per-instance
comparison dump alongside the human-readable stdout output:

    python misc/tests/test-regression.py --check \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-optimal-strips \
        --track heuristics optimal --json-output /tmp/r.json

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
              "failure_kind": "exact_match" | "coverage_loss" | "invalid_plan" | null,
              "plan_valid":   true | false | null,
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
`runs`.  `plan_valid` is `null` for the heuristic track and for runs
that did not solve.  `--json-output` is invalid in `--update` mode;
use the committed `regression-baselines/*.json` files as the
structured output for updates.

### Running on a custom instance set (`--instances`)

`--instances ID [ID ...]` overrides the default instance scope.  Each
ID is an integer N meaning `p0N.pddl` in every domain under
`--domain-dir`:

    # subset across the whole benchmark set: p01–p03 in every domain
    python misc/tests/test-regression.py --check \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-optimal-strips \
        --track heuristics optimal --instances 1 2 3

    # rerun one failing instance: point --domain-dir at the domain
    # folder, pass --instances <id>
    python misc/tests/test-regression.py --check \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-optimal-strips/airport \
        --track optimal --instances 3

`--instances` and `--full` are mutually exclusive.  Any `--instances`
invocation uses the 60 s per-instance time budget (the 10 s budget is
reserved for the default no-flag path).  On `--update`, results are
*merged* into the existing baseline file rather than overwriting it.
Single-domain `--domain-dir` also forces merge in `--update` mode.

### Running custom configs (`--config-file`)

`--config-file PATH` replaces each selected track's `CONFIGS` with a JSON
dict for the duration of the invocation.  The file shape is

    {
      "my_algo":       ["--search", "astar(my_h())"],
      "my_other_algo": ["--evaluator", "h=my_h()", "--search", "astar(h)"]
    }

Typical usage targets a single track:

    python misc/tests/test-regression.py --check --full \
        --domain-dir $AUTOSCALE_BENCHMARKS/21.11-optimal-strips \
        --track optimal --config-file my-configs.json

The same dict is applied to every selected track, so `--config-file` is
usually combined with `--track`.  In `--check` mode, the comparison runs
against whichever entries already exist in the baseline (a custom config
not present in the baseline simply has nothing to compare against).  In
`--update` mode, the new entries are *merged* into the existing baseline
file — entries for non-overridden configs are preserved, entries for
overridden configs are replaced.

## Rebaselining

After an intentional algorithm or configuration change, regenerate
the baselines:

    python misc/tests/generate_baseline.py
    python misc/tests/generate_baseline.py --track satisficing
    python misc/tests/generate_baseline.py --workers 8

Then commit the updated `misc/tests/regression-baselines/*.json`
files.  `generate_baseline.py` reads `AUTOSCALE_BENCHMARKS` (or
`--benchmarks PATH`), then sub-process invokes `test-regression.py
--update --full` once per benchmark set with the appropriate
`--domain-dir` and per-set `--track` filter.  Stops on the first
non-zero exit; forwards every other flag (`--workers`,
`--config-file`, etc.) to each sub-process unchanged.

`--update` directly on `test-regression.py` requires an explicit
scope flag — `--full`, `--instances`, or `--config-file` — to
confirm the intended overwrite/merge semantics:

| Flag combination                  | Behavior                                                                |
|-----------------------------------|-------------------------------------------------------------------------|
| `--update --full` (benchmark set) | Rebaseline everything for the selected tracks; **overwrites** the file. |
| `--update --full` (single domain) | Regenerate that domain only; **merges** into the existing file.         |
| `--update --instances ...`        | Regenerate only those instances; **merges** into the existing file.     |
| `--update --config-file ...`      | Regenerate only those configs; **merges** into the existing file.       |
| `--update` alone                  | Error: pick one of the above.                                           |

The default light-mode check uses a filtered subset of the full
baseline files; there are no separate `*-light.json` files.

## Test tracks

| Track | Benchmark set | Configs | Exact-match keys | Validates plans? |
|---|---|---|---|---|
| heuristics  | optimal-strips | 5 A* configs | initial_h_value, expansions, evaluations, generated, cost | no |
| optimal     | optimal-strips | 6 configs | cost, expansions, evaluations, generated | yes |
| satisficing | agile-strips   | 4 configs | cost, expansions, evaluations, generated | yes |
| anytime     | agile-strips   | 2 configs | incumbent_costs (full improving sequence) | yes (final plan only) |

Light mode runs the same configs as full mode, only on p01 instead of p01–p05.

**Light mode time limits**: 10 s per instance for heuristics, optimal, and
satisficing; 60 s for anytime (anytime planners must run to budget to produce
a comparable incumbent sequence).

## Adding a new track

1. Create `misc/tests/regression_<track>.py` following the pattern of
   `regression_optimal.py`.  Define `CONFIGS`, `TIME_LIMIT`,
   `EXACT_KEYS`, and implement `check_<track>` / `update_<track>`
   with the standard keyword args:

        def check_<track>(domain_dir, baseline_dir, workers, *,
                          configs=None, extra_configs=None,
                          instances=None,
                          validate=True, validate_bin=None) -> dict:
            ...

        def update_<track>(domain_dir, baseline_dir, workers, *,
                           configs=None, extra_configs=None,
                           instances=None,
                           validate=True, validate_bin=None) -> list[str]:
            ...

   - Resolve `configs` via `resolve_configs(CONFIGS, configs, extra_configs)`.
   - Default `instances` to `DEFAULT_LIGHT_INSTANCES` in `check_*`
     and `DEFAULT_FULL_INSTANCES` in `update_*`.
   - Have `_run` call `resolve_instances(domain_dir, resolved_instances)`,
     run the experiment, and return `(discovered, results)`.  Tracks
     are domain-dir-agnostic: do **not** join a hardcoded benchmark
     subdirectory on `domain_dir`.
   - Filter the baseline with
     `filter_baseline(baseline, set(resolved_configs), discovered)`
     in `check_*` — the third argument is the resolved-instance dict
     list, not the raw integer ids.  This makes the comparison narrow
     correctly when `domain_dir` is a single domain.
   - Forward `validate` and `validate_bin` to `run_experiment` (or
     hardcode `validate=False` if your track's primary metric isn't
     plan-related, like the heuristic track does).
   - `update_*` should call `drop_invalid_runs(results)` before
     `save_baseline` and return the resulting error list.  Pass
     `merge=True` to `save_baseline` whenever any override was
     supplied (configs, extra_configs, non-default instances, **or**
     a single-domain `domain_dir`):

            has_override = (
                configs is not None or extra_configs is not None
                or not is_full_default_instances(resolved_instances)
                or is_single_domain_layout(domain_dir)
            )
            save_baseline(baseline_dir, TRACK_NAME, results, merge=has_override)

   - `check_*` returns the comparison dict from `compare_results` or
     `baseline_missing_error(baseline_dir, TRACK_NAME)` if the
     baseline file is missing.

2. Import and register in `misc/tests/test-regression.py`:

        from regression_<track> import check_<track>, update_<track>

        TRACKS = [
            ...
            ("<track>", check_<track>, update_<track>),
        ]

3. Add the track → benchmark-set mapping to
   `misc/tests/generate_baseline.py`'s `TRACK_SETS` dict so the
   regenerator routes `--update` invocations to the correct
   `--domain-dir`:

        TRACK_SETS = {
            ...
            "<track>": "21.11-<set>-strips",
        }

4. Update `misc/tox.ini` if the new track shares a benchmark set with
   an existing track in the same `tox -e regression[-full]` env;
   otherwise add a third sequenced command for the new set.

5. Generate baselines:

        python misc/tests/generate_baseline.py --track <track>

6. Verify:

        python misc/tests/test-regression.py --check \
            --domain-dir $AUTOSCALE_BENCHMARKS/21.11-<set>-strips \
            --track <track>
        python misc/tests/test-regression.py --check --full \
            --domain-dir $AUTOSCALE_BENCHMARKS/21.11-<set>-strips \
            --track <track>
