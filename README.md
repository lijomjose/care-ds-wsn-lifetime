# CARE-DS: Dynamic Dominating-Set Scheduling for Heterogeneous WSNs

This repository is the public reproducibility artifact for:

> **CARE-DS: Critical-Neighborhood-Aware Residual-Energy Dominating-Set
> Scheduling for Heterogeneous Wireless Sensor Networks**
> 
> Lijo M. Jose and Deepu Benson

CARE-DS schedules a new dominating set—or a sink-rooted connected dominating
set—when residual energy, handover cost, or node failures make a fixed disjoint
family wasteful. The implementation evaluates CARE-DS against dynamic greedy,
time-matched randomized, classical static, and modern paper-based baseline
reproductions.

## Scope of the result

The evidence supports a conditional conclusion, not universal dominance:

- CARE-DS is **not** the best method when handovers are free.
- Its advantage appears when reconfiguration consumes energy, especially at
  handover costs 0.25 and 0.5, and when failures require reconstruction.
- The first-order radio study is an energy-model simulation, not a packet-level
  or hardware experiment.
- The lower-tail analysis is explicitly exploratory and post hoc.
- Baselines ending in `-R` are documented paper-based reproductions or
  reconstructions, not the original authors' executable code.

## What is included

| Path           | Contents                                                                   |
| -------------- | -------------------------------------------------------------------------- |
| `src/wsnlife/` | Graph generation, energy models, schedulers, baselines, and exact routines |
| `configs/`     | Fixed configurations and deterministic seeds for every reported study      |
| `results/`     | Final validated raw matrices and derived statistical tables                |
| `scripts/`     | Experiment, validation, analysis, and figure-generation entry points       |
| `tests/`       | Feasibility, energy-accounting, failure, and exactness checks              |
| `paper/`       | Pre-submission manuscript PDF and generated figures                        |
| `docs/`        | Reproduction protocol, data dictionary, baseline audit, and disclosure     |

The public release intentionally excludes superseded runs, interrupted logs,
reviewer-response notes, internal decision reports, legacy MATLAB experiments,
local working archives, and copies of third-party papers.

## Evidence matrix

| Study               | Instances | Rows   | Main purpose                                                                                             |
| ------------------- | ---------:| ------:| -------------------------------------------------------------------------------------------------------- |
| Primary dynamic     | 2,700     | 86,400 | Five topology families, 30 seeds, 50–500 nodes, three densities, four handover costs, two service models |
| Weighted static     | 3,150     | 18,900 | Modern static baselines up to 1,000 nodes                                                                |
| Unweighted static   | 1,890     | 9,450  | FSS-oriented validation under equal 0.5 s budgets                                                        |
| First-order radio   | 450       | 9,000  | Transmission, reception, aggregation, and wake-up energy                                                 |
| Controlled failures | 450       | 12,150 | No failure, random failure, and low-reserve-node failure                                                 |
| Ablation            | 450       | 8,100  | Reserve and switch-awareness components                                                                  |
| Exact challenge     | 12        | 48     | Fixed adversarial 22-node ordinary-domination cases                                                      |

All principal raw matrices use paired deterministic instances. Every scheduled
set is checked for domination and, in connected mode, sink-rooted connectivity.

## Quick start

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 scripts/self_check.py
python3 scripts/validate_submission.py
```

To run a small end-to-end experiment without overwriting the released data:

```bash
python3 scripts/run_pilot.py \
  --config configs/smoke.json \
  --output /tmp/care_ds_smoke.csv \
  --workers 2
```

To rebuild the released statistical summaries and figures from the included raw
matrices:

```bash
python3 scripts/analyse_submission.py \
  --unit results/submission_event_raw.csv \
  --modern-static results/modern_static_raw.csv \
  --modern-unweighted results/modern_unweighted_raw.csv \
  --radio results/radio_raw.csv \
  --failures results/failures_raw.csv \
  --ablation results/ablation_submission_raw.csv \
  --exact results/exact_challenge.csv

python3 scripts/analyse_lower_tail.py
MPLCONFIGDIR=/tmp/matplotlib-care python3 scripts/make_submission_figures.py
python3 scripts/validate_submission.py
```

Full simulation runs can take substantial CPU time. Output is instance-resumable:
rerunning a command computes only missing instance identifiers unless
`--no-resume` is supplied.

## Reproducibility and interpretation

Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before rerunning the
full matrix. It records the acceptance rules, the paired-inference protocol,
and restrictions on claims. The baseline provenance and deliberate
implementation limits are in
[`docs/BASELINE_REPRODUCTIONS.md`](docs/BASELINE_REPRODUCTIONS.md).

The authors conceived the problem, model, CARE-DS algorithm, comparator
protocol, experimental design, interpretation, and conclusions. The complete
AI-assistance disclosure is preserved in
[`docs/AI_ASSISTANCE_DISCLOSURE.md`](docs/AI_ASSISTANCE_DISCLOSURE.md).

## Citation

Until a DOI is assigned, cite this repository using [`CITATION.cff`](CITATION.cff)
and identify the release version or commit hash used.

## License

Software in `src/`, `scripts/`, and `tests/` is released under the MIT License.
The manuscript, figures, and result tables remain subject to the authors' and
publisher's scholarly-use terms; see [`NOTICE.md`](NOTICE.md).
