# Reproducibility protocol

## Design rules

- Every graph and energy instance has a deterministic seed shared by all
  algorithms in that comparison.
- Graph construction preserves the declared random-geometric/unit-disk model
  where applicable.
- Every scheduled set is checked for domination; connected-mode sets are also
  checked for sink-rooted connectivity.
- Failed nodes are removed from both candidate selection and the surviving
  service universe.
- Runtime is measured. `MS-RG-WT` is calibrated as a wall-time comparator; it
  is not described as receiving identical runtime on every instance.
- Paired comparisons use bootstrap confidence intervals, Wilcoxon signed-rank
  tests, and Holm correction.

## Released matrices and acceptance rules

| Study | Configuration | Result | Acceptance rule |
|---|---|---|---|
| Primary dynamic | `configs/submission_event.json` | `results/corrected/merged/submission_event_raw.csv` | 2,700 instances; 86,400 rows |
| Weighted static | `configs/modern_static.json` | `results/modern_static_raw.csv` | 3,150 instances; 18,900 rows |
| Unweighted static | `configs/modern_unweighted.json` | `results/modern_unweighted_raw.csv` | 1,890 instances; 9,450 rows |
| Radio | `configs/radio.json` | `results/corrected/merged/radio_raw.csv` | 450 instances; 9,000 rows |
| Failures | `configs/failures.json` | `results/corrected/merged/failures_raw.csv` | 450 instances; 12,150 rows |
| Ablation | `configs/ablation_submission.json` | `results/corrected/merged/ablation_submission_raw.csv` | 450 instances; 8,100 rows |
| Exact challenge | fixed in script | `results/exact_challenge.csv` | 12 instances; 48 rows |

For each principal matrix, validation additionally requires zero duplicate
comparison keys, invalid schedules, or missing values. Run:

```bash
python3 scripts/validate_submission.py
```

The released validation report is `results/submission_validation.json`.

## Full rerun

Use `scripts/run_pilot.py` with each configuration and a separate output
filename for a fresh run; do not overwrite the released corrected matrices.
Runs are resumable at the instance level. The number of workers may
be changed without changing deterministic instance seeds.

After generation, run `scripts/validate_submission.py`, then
`scripts/analyse_submission.py` with the corrected input paths shown in the root README. The
analysis writes paired tests, summaries, runtime-fairness results, and evidence
metadata to `results/corrected/analysis/`. `scripts/make_submission_figures.py` rebuilds the paper
figures from those matrices.

## Claim restrictions

- `-R` means paper-based reproduction or reconstruction, not author code.
- Results on the unweighted objective are not generalized to heterogeneous WSN
  lifetime.
- CARE-DS is not claimed to dominate when handover cost is zero.
- The radio model is not packet-level or hardware validation.
- The lower-tail result is exploratory and post hoc; it is not preregistered
  confirmatory evidence.
- Repository readiness and completed simulations do not imply acceptance by a
  journal.
