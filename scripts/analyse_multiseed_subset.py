#!/usr/bin/env python3
"""Summarize the repeated-algorithm-seed robustness experiment."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def _seed(*parts):
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 2**32


def _bootstrap_mean_ci(values, seed, samples=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, (0.025, 0.975)))


def _rank_biserial(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    if not len(diff):
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def _holm_adjust(pvalues):
    """Return Holm step-down adjusted p-values in the original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def dynamic_summary(data):
    dynamic = data[data.study == "dynamic"].copy()
    keys = ["instance_id", "topology", "n", "target_degree", "mode", "switch_cost", "algorithm"]
    per_instance = dynamic.groupby(keys, as_index=False).agg(
        repeat_mean=("lifetime", "mean"), repeat_sd=("lifetime", "std"),
        repeat_min=("lifetime", "min"), repeat_max=("lifetime", "max"),
        mean_runtime_s=("runtime_s", "mean"), repeats=("algorithm_repeat", "nunique"),
    )
    per_instance["repeat_cv_pct"] = 100 * per_instance.repeat_sd / per_instance.repeat_mean
    rows = []
    for (mode, switch_cost), block in per_instance.groupby(["mode", "switch_cost"]):
        pivot = block.pivot(index="instance_id", columns="algorithm", values="repeat_mean").dropna()
        diff = pivot["CARE-DS"] - pivot["MS-RG-WT"]
        lo, hi = _bootstrap_mean_ci(diff, _seed(mode, switch_cost))
        raw = dynamic[
            (dynamic["mode"] == mode) & (dynamic["switch_cost"] == switch_cost)
        ]
        raw_pivot = raw.pivot_table(
            index=["instance_id", "algorithm_repeat"], columns="algorithm", values="lifetime"
        ).dropna()
        pvalue = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)
        row = {
            "mode": mode, "switch_cost": switch_cost, "graph_instances": len(pivot),
            "algorithm_repeats": int(block.repeats.min()),
            "care_mean": float(pivot["CARE-DS"].mean()),
            "ms_rg_wt_mean": float(pivot["MS-RG-WT"].mean()),
            "relative_improvement_pct": float(100 * (pivot["CARE-DS"].mean() / pivot["MS-RG-WT"].mean() - 1)),
            "mean_difference": float(diff.mean()), "ci95_low": lo, "ci95_high": hi,
            "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()), "rank_biserial": _rank_biserial(diff),
            "wilcoxon_p": pvalue,
            "repeat_level_care_win_pct": float(100 * (raw_pivot["CARE-DS"] > raw_pivot["MS-RG-WT"]).mean()),
        }
        for algorithm in ("CARE-DS", "MS-RG-WT"):
            a = block[block.algorithm == algorithm]
            prefix = "care" if algorithm == "CARE-DS" else "ms_rg_wt"
            row[f"{prefix}_mean_within_instance_sd"] = float(a.repeat_sd.mean())
            row[f"{prefix}_mean_within_instance_cv_pct"] = float(a.repeat_cv_pct.mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["wilcoxon_p_holm"] = _holm_adjust(summary["wilcoxon_p"])
    return per_instance, summary


def static_summary(data):
    static = data[data.study == "static-unweighted"].copy()
    keys = ["instance_id", "topology", "n", "target_degree", "algorithm"]
    per_instance = static.groupby(keys, as_index=False).agg(
        repeat_mean=("lifetime", "mean"), repeat_sd=("lifetime", "std"),
        repeat_min=("lifetime", "min"), repeat_max=("lifetime", "max"),
        mean_runtime_s=("runtime_s", "mean"), repeats=("algorithm_repeat", "nunique"),
    )
    per_instance["repeat_cv_pct"] = 100 * per_instance.repeat_sd / per_instance.repeat_mean
    pivot = per_instance.pivot(index="instance_id", columns="algorithm", values="repeat_mean").dropna()
    best = pivot.max(axis=1)
    rows = []
    for algorithm in pivot.columns:
        a = per_instance[per_instance.algorithm == algorithm]
        rows.append({
            "algorithm": algorithm, "graph_instances": len(a),
            "algorithm_repeats": int(a.repeats.min()),
            "mean_objective": float(pivot[algorithm].mean()),
            "wins_or_ties": int((pivot[algorithm] == best).sum()),
            "mean_gap_to_best_pct": float((100 * (best - pivot[algorithm]) / best).mean()),
            "mean_within_instance_sd": float(a.repeat_sd.mean()),
            "mean_within_instance_cv_pct": float(a.repeat_cv_pct.mean()),
            "mean_runtime_s": float(a.mean_runtime_s.mean()),
        })
    return per_instance, pd.DataFrame(rows).sort_values("mean_objective", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if len(data) != 4280:
        raise RuntimeError(f"expected 4280 rows, found {len(data)}")
    key = ["study", "instance_id", "mode", "switch_cost", "algorithm_repeat", "algorithm"]
    if data.duplicated(key).any() or data.isna().any().any() or not data.valid.astype(bool).all():
        raise RuntimeError("multi-seed integrity check failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_instances, dynamic = dynamic_summary(data)
    static_instances, static = static_summary(data)
    dynamic_instances.to_csv(args.output_dir / "multiseed_dynamic_per_instance.csv", index=False)
    dynamic.to_csv(args.output_dir / "multiseed_dynamic_summary.csv", index=False)
    static_instances.to_csv(args.output_dir / "multiseed_static_per_instance.csv", index=False)
    static.to_csv(args.output_dir / "multiseed_static_summary.csv", index=False)
    report = [
        "REPEATED-ALGORITHM-SEED ROBUSTNESS", "", "DYNAMIC PAIRED SUMMARY",
        dynamic.to_string(index=False), "", "STATIC UNWEIGHTED SUMMARY",
        static.to_string(index=False),
    ]
    (args.output_dir / "multiseed_report.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
