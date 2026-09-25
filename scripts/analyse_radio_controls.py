#!/usr/bin/env python3
"""Audit-motivated dwell control and radio ablation analysis.

This script combines the validated primary radio matrix with a new MS-RG-WT
arm that receives CARE-DS's identical feasible-set dwell rule.  It also
compares the two declared CARE ablations in the same radio condition.  The
existing primary comparator rows are never recomputed or overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from analyse_lower_tail import Study, prepare


ROOT = Path(__file__).resolve().parents[1]
KEY = ["instance_id", "mode", "switch_cost", "failure_scenario", "algorithm"]
PUBLISHED = ["D-EAA-ER", "DWG", "MS-RG-WT"]
MATCHED = "MS-RG-WT-DW"


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 2**32


def validate_input(data: pd.DataFrame, expected_rows: int, algorithms: set[str], name: str) -> None:
    if len(data) != expected_rows:
        raise RuntimeError(f"{name}: expected {expected_rows} rows, found {len(data)}")
    if set(data.algorithm) != algorithms:
        raise RuntimeError(f"{name}: unexpected algorithms {sorted(set(data.algorithm))}")
    if data.duplicated(KEY).any() or data.isna().any().any() or not data.valid.astype(bool).all():
        raise RuntimeError(f"{name}: duplicate, missing, or invalid rows")


def rank_biserial(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum())


def mean_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def comparison(block: pd.DataFrame, other: np.ndarray, label: str, subset: str,
               samples: int) -> dict[str, object]:
    care = block["CARE-DS"].to_numpy(float)
    other = np.asarray(other, dtype=float)
    diff = care - other
    pvalue = 1.0 if np.all(diff == 0) else float(wilcoxon(diff).pvalue)
    mode = block["mode"].iloc[0]
    low, high = mean_ci(diff, stable_seed(label, mode, subset), samples)
    return {
        "mode": mode,
        "subset": subset,
        "comparator": label,
        "pairs": len(block),
        "care_mean": float(care.mean()),
        "comparator_mean": float(other.mean()),
        "aggregate_gain_pct": float(100 * (care.mean() / other.mean() - 1)),
        "mean_difference": float(diff.mean()),
        "mean_difference_ci95_low": low,
        "mean_difference_ci95_high": high,
        "wins": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "losses": int(np.sum(diff < 0)),
        "rank_biserial": rank_biserial(diff),
        "p_raw_exploratory": pvalue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radio", type=Path, default=ROOT / "results/corrected/merged/radio_raw.csv")
    parser.add_argument("--dwell", type=Path, default=ROOT / "results/revision/radio_dwell_control_raw.csv")
    parser.add_argument("--ablation", type=Path, default=ROOT / "results/revision/radio_ablation_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/revision/analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    args = parser.parse_args()

    radio = pd.read_csv(args.radio)
    dwell = pd.read_csv(args.dwell)
    ablation = pd.read_csv(args.ablation)
    validate_input(dwell, 900, {MATCHED}, "dwell control")
    validate_input(ablation, 1800, {"CARE-noreserve", "CARE-noswitch"}, "radio ablation")
    if len(radio) != 9000 or radio.duplicated(KEY).any() or not radio.valid.astype(bool).all():
        raise RuntimeError("primary radio matrix failed validation")

    extended = pd.concat([radio, dwell], ignore_index=True).sort_values(KEY)
    if len(extended) != 9900 or extended.duplicated(KEY).any():
        raise RuntimeError("extended radio matrix failed dimension/key validation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extended_path = ROOT / "results/revision/radio_with_dwell_control_raw.csv"
    extended.to_csv(extended_path, index=False)

    study = Study(
        name="radio",
        label="Radio, h/a_ref = 0.5, dwell-matched control included",
        path=extended_path,
        switch_costs=(1e-4,),
        competitors=tuple(PUBLISHED + [MATCHED]),
    )
    tail = prepare(study)
    control_rows: list[dict[str, object]] = []
    for mode, mode_block in tail.groupby("mode", sort=False):
        blocks = {
            "all": mode_block,
            "lowest_decile": mode_block[mode_block.tail_rank <= np.ceil(0.1 * mode_block.stratum_size)],
        }
        for subset, block in blocks.items():
            published_best = block[PUBLISHED].max(axis=1).to_numpy(float)
            strengthened_best = block[PUBLISHED + [MATCHED]].max(axis=1).to_numpy(float)
            control_rows.append(comparison(block, published_best, "published-form oracle", subset,
                                           args.bootstrap_samples))
            control_rows.append(comparison(block, block[MATCHED].to_numpy(float), MATCHED, subset,
                                           args.bootstrap_samples))
            control_rows.append(comparison(block, strengthened_best, "oracle including dwell control", subset,
                                           args.bootstrap_samples))
    controls = pd.DataFrame(control_rows)
    controls.to_csv(args.output_dir / "radio_dwell_control_summary.csv", index=False)

    care = radio[
        (radio.algorithm == "CARE-DS") & np.isclose(radio.switch_cost, 1e-4)
    ]
    radio_ablation = pd.concat([care, ablation], ignore_index=True)
    validate_input(
        radio_ablation, 2700,
        {"CARE-DS", "CARE-noreserve", "CARE-noswitch"}, "combined radio ablation",
    )
    ablation_pivot = radio_ablation.pivot_table(
        index=["instance_id", "mode", "switch_cost"], columns="algorithm",
        values="lifetime", aggfunc="first",
    ).reset_index()
    membership = tail[["instance_id", "mode", "switch_cost", "tail_rank", "stratum_size"]]
    ablation_pivot = ablation_pivot.merge(
        membership, on=["instance_id", "mode", "switch_cost"], validate="one_to_one"
    )
    ablation_rows: list[dict[str, object]] = []
    for mode, mode_block in ablation_pivot.groupby("mode", sort=False):
        blocks = {
            "all": mode_block,
            "lowest_decile": mode_block[mode_block.tail_rank <= np.ceil(0.1 * mode_block.stratum_size)],
        }
        for subset, block in blocks.items():
            for variant in ("CARE-noreserve", "CARE-noswitch"):
                ablation_rows.append(comparison(
                    block, block[variant].to_numpy(float), variant, subset,
                    args.bootstrap_samples,
                ))
    ablations = pd.DataFrame(ablation_rows)
    ablations.to_csv(args.output_dir / "radio_ablation_summary.csv", index=False)

    lines = [
        "AUDIT-MOTIVATED RADIO MECHANISM CONTROLS",
        "",
        "The strengthened oracle includes MS-RG-WT-DW, which differs from",
        "MS-RG-WT only by receiving CARE-DS's identical feasible-set dwell rule.",
        "Lowest-decile membership is reselected using that strengthened oracle.",
        "All p-values below are exploratory because these controls followed the audit.",
        "",
        "DWELL CONTROL",
        controls.to_string(index=False),
        "",
        "RADIO ABLATION",
        ablations.to_string(index=False),
    ]
    (args.output_dir / "radio_mechanism_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
