# CARE-DS: Dynamic Dominating-Set Scheduling for Heterogeneous WSNs

This repository accompanies the manuscript **“CARE-DS: Critical-Neighborhood-Aware Residual-Energy Dominating-Set Scheduling for Heterogeneous Wireless Sensor Networks”** by Lijo M. Jose and Deepu Benson.

A fixed family of disjoint dominating sets can leave usable battery energy stranded when one member of a set runs out before the others. CARE-DS instead constructs feasible sets from the current network state. It uses the remaining energy in critical closed neighborhoods to choose a set and a dwell rule to avoid repeated handover costs.

## What the evidence says

CARE-DS is most useful in the tested unit-cost settings with nonzero handover energy and when failures require reconstruction. It does not lead when switching is free. In the first-order radio study, it exceeds a direct dwell-matched shared-portfolio control by 4.85% in ordinary service and 0.88% in connected service, but it falls below a retrospective per-instance oracle that may select among four controls overall. The 14.44% ordinary-service gain in the reselected lowest-lifetime decile is exploratory, with a stratified-bootstrap 95% interval of 6.04%–19.25%; the connected-tail interval includes zero. These simulations are not packet-level or testbed evidence.

Comparators suffixed `-R` are documented reproductions or reconstructions, not executables supplied by their authors.

## Contents

- `src/wsnlife/`: graphs, energy models, schedulers, baseline methods, and exact routines.
- `configs/`: fixed study settings and deterministic seeds, including `radio_dwell_control.json` and `radio_ablation.json`.
- `results/`: raw matrices and derived summaries. The corrected final primary matrices are under `results/corrected/merged/`; the audit-motivated radio additions are under `results/revision/`.
- `scripts/`: simulation, analysis, validation, and figure generation.
- `tests/`: feasibility and energy-accounting regression checks.
- `docs/`: protocol, baseline provenance, and AI-assistance disclosure.

The audit-motivated radio additions contain 900 dwell-control and 1,800 ablation rows, each covering 450 graphs in both service modes. `MS-RG-WT-DW` gives the weighted multi-start control CARE-DS's feasible-set dwell rule. The full revised radio tail uses D-EAA-ER, DWG, MS-RG-WT, and MS-RG-WT-DW as the comparator set. The dwell control and radio ablation are exploratory analyses.

## Validate the released evidence

Python 3.10 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 scripts/self_check.py
python3 scripts/validate_submission.py
```

The validator checks nine study entries, including the two new radio matrices and the reselected lower-tail summary. It verifies row counts, instance counts, duplicate keys, missing values, and schedule validity.

To rebuild the audit-motivated analysis from the included raw matrices:

```bash
python3 scripts/analyse_radio_controls.py
python3 scripts/analyse_lower_tail.py
```

To rebuild the primary comparisons and manuscript figures from the corrected
matrices, run:

```bash
python3 scripts/analyse_submission.py \
  --unit results/corrected/merged/submission_event_raw.csv \
  --modern-static results/modern_static_raw.csv \
  --modern-unweighted results/modern_unweighted_raw.csv \
  --radio results/corrected/merged/radio_raw.csv \
  --failures results/corrected/merged/failures_raw.csv \
  --ablation results/corrected/merged/ablation_submission_raw.csv \
  --exact results/exact_challenge.csv \
  --output-dir results/corrected/analysis
MPLCONFIGDIR=/tmp/matplotlib-care python3 scripts/make_submission_figures.py
```

The older root-level primary matrices are retained for provenance. The paper
uses the corrected matrices. Sensitivity and repeated-seed matrices, complete
derived tables, and the final manuscript source are in the separate
supplementary research artifact.

To rerun those two experiments without overwriting the released results:

```bash
python3 scripts/run_pilot.py --config configs/radio_dwell_control.json --output /tmp/radio_dwell_control_raw.csv --workers 8
python3 scripts/run_pilot.py --config configs/radio_ablation.json --output /tmp/radio_ablation_raw.csv --workers 8
```

Full simulations may take substantial CPU time. Runs are deterministic at the instance level and support resumption. The paper's baseline limitations are documented in `docs/BASELINE_REPRODUCTIONS.md`; the authors' AI-use disclosure is in `docs/AI_ASSISTANCE_DISCLOSURE.md`.

## Citation and license

Until a DOI is assigned, use `CITATION.cff` and identify the release commit. Software in `src/`, `scripts/`, and `tests/` is MIT-licensed. See `NOTICE.md` for manuscript, figure, and results terms.
