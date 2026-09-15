# Coverage by domain

Run with Python 3; no additional packages are needed:

```sh
python3 experiments/analysis/coverage_by_domain.py
```

By default, this reads evaluation properties from the seven round-robin, LAMA,
and combined cost-insensitive triangle
experiment directories listed in the script. Missing evaluation data produces a
warning. Files are excluded entirely unless every domain/algorithm combination
has exactly 30 runs. Override this with `--runs-per-domain N`.
To select other inputs or an output directory:

```sh
python3 experiments/analysis/coverage_by_domain.py \
  experiments/2026-08-31-round-robin-triangle \
  experiments/2026-09-01-eager-po-round-robin/data/2026-09-01-C-eager-po-round-robin-eval/properties \
  --output-dir /tmp/coverage
```

Directory discovery finds `*-eval/properties` files recursively (or `properties`
directly inside an input directory). Individual raw-run properties are not
evaluation files and cannot be used as explicit inputs. Each evaluation file is
kept separate, including corrected reruns, to avoid double-counting tasks or
combining different experimental conditions.

Outputs in `experiments/analysis/coverage/` by default:

- `coverage-by-domain.csv`: a combined domains × algorithms table with totals.
  Columns are grouped alphabetically by algorithm, with `[original]` followed by
  `[unit cost]` so the two versions sit next to each other. Multiple original
  results or unit-cost reruns remain separate, labeled by experiment. Unmatched
  algorithms appear on their own; `lama` and `lama-first-phase` are not paired.
- `coverage.md`: the combined table followed by individual tables and excluded files.
- One CSV matrix per properties file, with the same rows and columns.
  These individual tables sort by aggregate recorded coverage, highest first,
  with alphabetical ordering for ties.
- `coverage-long.csv`: source file, domain, algorithm, solved count, run count,
  and missing-coverage count, suitable for further analysis.

The `[unit cost]` columns are the September 8 versions of the earlier algorithms:
search uses `cost_type=one`, and both FF and landmark-sum use
`transform=adapt_costs(one)`. Reported plan costs still use original action costs.
All new runs stop at the first solution; the older boosted triangle runs used
anytime search. `[original]` retains the original experiment settings, including
any heuristic cost transforms. Thus the columns identify experiment variants,
not a guarantee that cost handling is their only difference.

Coverage is the sum of the recorded binary `coverage` values. Missing coverage
makes the corresponding matrix cell and algorithm total `NA`; the long CSV
still gives the known solved count and number of missing values. Absent
domain/algorithm combinations also display `NA`. Totals cover recorded runs;
use the long CSV run counts to check sample sizes before comparing algorithms.
Duplicate algorithm/domain/problem records and invalid coverage values are
rejected. Existing output files with matching names are overwritten on rerun.

Run the focused checks with:

```sh
python3 -m unittest discover -s experiments/analysis -p 'test_*.py'
```
