# Algorithm Development Loop

The regression suite's `--dev` mode is the run-and-record half of an
LLM-driven (or human-driven) iterate-on-an-algorithm loop.  It builds
on the same harness used for the regular `--check` and `--update`
modes — it does not replace it.  This document specifies the contract
a *driver* must implement to use `--dev`, and shows one concrete
worked example.

The driver lives outside scorpion: it can be a shell script, a
Python program, or an LLM-orchestration tool (pi, llmloop, MCP, etc.).
The harness produces structured data; the driver decides what to do
with it.

## Driver contract

Per iteration, the driver does:

1. **Build**.  Rebuild scorpion if source changed since the last iteration
   (`./build.py` from the repo root).

2. **Invoke harness** in `--dev` mode:

       python misc/tests/test-regression.py --dev \
           --baseline-dir <iteration-dir> \
           --track <track> \
           --domain-dir <benchmark-set-or-domain> \
           --config-file <dev-config.json> \
           [--full]                              # promotion gates only

   Each invocation reads the latest `<track>-NNNN.json` in
   `<iteration-dir>`, runs the experiments, and writes a new file with
   suffix `NNNN+1`.  `--dev` always exits 0; the iteration file is the
   structured output.

3. **Read the new iteration file**, `<iteration-dir>/<track>-NNNN.json`.
   The fields the driver consumes (full schema below):
   - **Per-config aggregates** — `improved`, `expansions_geomean_ratio`,
     `generated_geomean_ratio`, `time_geomean_ratio`,
     `cost_geomean_ratio`, `cost_changed`, `coverage`,
     `coverage_delta`, `compared_count`.
   - **Iteration metadata** — `iteration_number`,
     `previous_iteration_number`, `scorpion_commit`, `worktree_dirty`.
   - **Per-instance runs** (under `runs`) — `coverage`, `cost`,
     `incumbent_costs`, `expansions`, `evaluations`, `generated`,
     `search_time`, `wall_time`, `returncode`, `plan_valid`,
     `plan_validation_message`.

4. **Apply rules** (driver-side; not in the harness):
   - `correctness_ok = (no crashes) AND (all plans valid) AND (coverage_delta >= 0)`.
     Crashes are derived from `runs[*].returncode` (a value < 0 other
     than -1 indicates signal-induced kill; -1 is the harness's
     timeout marker).  Plan validity is `runs[*].plan_valid` —
     `False` is a hard fail; `None` means the run did not produce a
     plan or the track does not validate (heuristic track).
   - `improved` flags fed into a rolling convergence window
     (default size N = 3).
   - Optional class-specific cost cross-check against the committed
     `regression-baselines/<track>.json` — relevant when the
     algorithm under development claims optimality (cost must match
     baseline) or admissibility-equivalent behavior.

5. **Decide**: continue iterating (LLM proposes another edit), promote
   the candidate (rerun with `--full` for a tighter pre-PR sanity
   check), revert (`git checkout <previous_iteration's scorpion_commit>`),
   or stop (ready-to-PR achieved, or N non-improving iterations
   followed by a correctness failure → bail).

## Iteration file schema

    {
      "schema_version":         1,
      "track":                  "<track-name>",
      "iteration_number":       <int>,
      "previous_iteration_number": <int> | null,
      "scorpion_commit":        "<sha>" | null,
      "worktree_dirty":         <bool>,
      "started_at":             "<ISO-8601 UTC>",
      "domain_dir":             "<absolute path>",
      "instances":              [<integer ids>],
      "time_limit":             <int seconds>,
      "configs":                {"<name>": ["--search", "...", ...], ...},
      "runs": {
        "<config>|<domain>|<problem>.pddl": {
          "coverage": 0|1, "cost": <num>, "incumbent_costs": [...],
          "expansions": <int>, "evaluations": <int>, "generated": <int>,
          "search_time": <float>, "wall_time": <float>,
          "returncode": <int>, "plan_valid": true|false|null,
          "plan_validation_message": "<str>"          // present only on validation failure
        }
      },
      "per_config": {
        "<config>": {
          "improved":                 <bool>,
          "expansions_geomean_ratio": <float> | null,
          "generated_geomean_ratio":  <float> | null,
          "time_geomean_ratio":       <float> | null,
          "cost_geomean_ratio":       <float> | null,
          "cost_changed":             <bool>,
          "coverage":                 <int>,
          "coverage_delta":           <int>,
          "compared_count":           <int>
        }
      }
    }

`improved = true` iff the geomean ratios for `expansions`, `generated`,
and `time_geomean_ratio` are all non-null and < 1.0.  On the first
iteration (no prior file), `improved` is unconditionally true and the
ratios are null.  Cost is reported separately because direction depends
on algorithm class.

## Worked example: beam search on a single domain

Iterate on a beam-search implementation against airport (one domain,
one config).  Stops on convergence (`improved == false` for 3
consecutive iterations) or when `correctness_ok` is false.  Replace
the LLM step with whatever orchestration tool you use.

    #!/usr/bin/env bash
    set -euo pipefail

    REPO=$HOME/research/scorpion
    BENCH=$HOME/research/autoscale-benchmarks/21.11-optimal-strips/airport
    DEV_DIR=$REPO/misc/tests/regression-baselines/dev/beam_search
    CONFIG=/tmp/beam_search.json
    TRACK=optimal
    N_WINDOW=3

    mkdir -p "$DEV_DIR"
    cat >"$CONFIG" <<'EOF'
    {"beam_w8": ["--search", "beam_search(ff(), beam_width=8)"]}
    EOF

    not_improved=0
    while :; do
      (cd "$REPO" && ./build.py)               # 1. build
      python "$REPO/misc/tests/test-regression.py" --dev \
          --baseline-dir "$DEV_DIR" \
          --track "$TRACK" \
          --domain-dir "$BENCH" \
          --config-file "$CONFIG"              # 2. invoke

      latest=$(ls "$DEV_DIR/$TRACK-"*.json | sort | tail -1)
      improved=$(jq -r '.per_config.beam_w8.improved' "$latest")
      cov_delta=$(jq -r '.per_config.beam_w8.coverage_delta' "$latest")
      crashes=$(jq '[.runs | to_entries[] | select(.value.returncode < -1 and .value.returncode != null)] | length' "$latest")
      bad_plans=$(jq '[.runs | to_entries[] | select(.value.plan_valid == false)] | length' "$latest")

      if [ "$crashes" -ne 0 ] || [ "$bad_plans" -ne 0 ] || [ "$cov_delta" -lt 0 ]; then
        echo "correctness fail at $(basename "$latest"); revert and try again"
        prev_commit=$(jq -r '.scorpion_commit' "$latest")
        # llm_propose_revert "$prev_commit"; continue, or break to inspect.
        break
      fi

      if [ "$improved" = "true" ]; then
        not_improved=0
      else
        not_improved=$((not_improved + 1))
      fi

      if [ "$not_improved" -ge "$N_WINDOW" ]; then
        echo "converged after $N_WINDOW non-improving iterations"
        break
      fi

      llm_propose_edit "$REPO" "$latest"     # 3.-5. driver-side
    done                                      #     LLM proposes next edit

    # Promotion: rerun the candidate at full scope as a pre-PR sanity check.
    python "$REPO/misc/tests/test-regression.py" --dev --full \
        --baseline-dir "$DEV_DIR" \
        --track "$TRACK" \
        --domain-dir "$HOME/research/autoscale-benchmarks/21.11-optimal-strips" \
        --config-file "$CONFIG"

`llm_propose_edit` and `llm_propose_revert` stand in for whatever
orchestration tool drives the loop — they read the iteration file,
mutate the scorpion source tree, and return so the loop continues.

## Notes

- **Seeding iteration 0.**  No bootstrapping needed — on the first
  `--dev` invocation, `previous_iteration_number` is `null` and
  `improved` is `true` for every config (extant > nothing).  The
  loop's convergence counter starts at 0.

- **Reverting.**  Each iteration file embeds `scorpion_commit` and
  `worktree_dirty`.  To revert to the prior iteration's code, read
  the bad iteration's file, look up the previous iteration's
  `scorpion_commit`, and `git checkout <hash>`.  Iteration files on
  disk are not consulted in order — the harness picks "latest" by
  highest filename suffix — so leaving discarded iterations on disk
  is harmless.  The driver may delete them.

- **Promotion to `--full`.**  The same loop with `--full` and a
  benchmark-set `--domain-dir` runs p01–p05 across all 42 domains
  with a 60 s budget (~30 min).  Use this as the final pre-PR
  sanity check before editing the track's `CONFIGS` dict and
  regenerating the committed baseline.  Graduation to the
  regression suite is **human-gated** — the loop never
  auto-graduates.

- **Per-branch isolation.**  Use a different `--baseline-dir` per
  algorithm under development (e.g.
  `regression-baselines/dev/beam_search/`,
  `regression-baselines/dev/lazy_beam/`).  The `dev/` subdirectory
  is `.gitignore`'d.

- **Anytime track.**  `time_geomean_ratio` is computed against
  `wall_time` (not `search_time`, which iterated search omits).  The
  budget is the planner's input, so `wall_time` clusters around the
  time limit and `improved` rarely fires on the time component
  alone — algorithm-class judgment lives in `cost_geomean_ratio` and
  the raw `incumbent_costs` sequences.
