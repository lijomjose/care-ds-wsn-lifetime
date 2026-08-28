# Data dictionary

## Principal raw matrices

The primary dynamic, weighted static, unweighted static, radio, failure, and
ablation CSV files share this schema.

| Column | Meaning |
|---|---|
| `instance_id` | Deterministic human-readable graph-instance key |
| `instance_seed` | Seed used to generate the paired instance |
| `topology` | Graph family (`rgg`, `clustered`, `er`, `watts_strogatz`, or `barabasi_albert`) |
| `n` | Number of vertices before any failure event |
| `target_degree` | Requested mean-degree regime |
| `actual_degree` | Realized mean degree |
| `run` | Replicate index |
| `mode` | `dominating` or sink-rooted `connected` service |
| `switch_cost` | Handover/wake-up cost charged by the selected energy model |
| `failure_scenario` | `none`, `random`, or `targeted` low-reserve failure |
| `energy_model` | `unit` or first-order `radio` |
| `algorithm` | Evaluated scheduler or baseline label |
| `lifetime` | Number of feasible service rounds completed |
| `runtime_s` | Measured wall-clock runtime in seconds |
| `switches` | Count of active-set membership changes |
| `activations` | Aggregate node activations over the schedule |
| `mean_set_size` | Mean active-set cardinality |
| `valid` | Independent feasibility check for every returned schedule |

## Exact CSV files

`exact_stress.csv` and `exact_challenge.csv` add the exact optimum,
`ratio_to_optimum`, solver status, candidate-set count, heuristic and exact
runtimes, and whether the heuristic attained the optimum. The challenge file
also records its calibration-based selection policy and is not used for
population inference.

## Derived tables

- `submission_summary.csv` contains aggregate lifetime, runtime, and validity.
- `submission_paired_tests.csv` contains paired effects, bootstrap intervals,
  signed-rank statistics, and Holm-adjusted p-values.
- `submission_best_competitor.csv` compares CARE-DS with the retrospective best
  comparator on each paired instance.
- `runtime_fairness.csv` reports CARE-DS/MS-RG-WT runtime ratios.
- `lower_tail_summary.csv` and `lower_tail_deciles.csv` contain the exploratory
  within-stratum analysis ranked only by comparator lifetime.
- `upper_bound_gaps.csv` and `upper_bound_summary.csv` compare unit-cost
  lifetimes with the closed-neighborhood resource bound.
