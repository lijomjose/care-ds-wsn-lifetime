#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_ci(values, rng, samples=5000):
    values = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return np.quantile(means, [0.025, 0.975])


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    args = parser.parse_args()
    path = Path(args.csv)
    df = pd.read_csv(path)
    assert df["valid"].all(), "at least one schedule failed validation"
    group_cols = ["mode", "switch_cost", "topology", "n", "target_degree", "algorithm"]
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(mean_lifetime=("lifetime", "mean"), sd_lifetime=("lifetime", "std"),
             mean_runtime_s=("runtime_s", "mean"), mean_set_size=("mean_set_size", "mean"),
             mean_switches=("switches", "mean"))
    )
    rng = np.random.default_rng(20260818)
    cis = []
    for keys, block in df.groupby(group_cols):
        lo, hi = bootstrap_ci(block["lifetime"].to_numpy(), rng)
        cis.append((*keys, lo, hi))
    ci = pd.DataFrame(cis, columns=group_cols + ["ci95_low", "ci95_high"])
    summary = summary.merge(ci, on=group_cols)

    tests = []
    proposed = df[df.algorithm == "CARE-DS"]
    comparison_keys = ["mode", "switch_cost", "topology", "n", "target_degree"]
    for keys, care in proposed.groupby(comparison_keys):
        subset = df.copy()
        for col, value in zip(comparison_keys, keys):
            subset = subset[subset[col] == value]
        for competitor in sorted(set(subset.algorithm) - {"CARE-DS"}):
            other = subset[subset.algorithm == competitor]
            paired = care[["instance_id", "lifetime"]].merge(
                other[["instance_id", "lifetime"]], on="instance_id", suffixes=("_care", "_other")
            )
            if len(paired) < 2:
                continue
            diff = paired.lifetime_care - paired.lifetime_other
            if np.all(diff == 0):
                pvalue = 1.0
            else:
                pvalue = float(wilcoxon(diff, alternative="two-sided").pvalue)
            tests.append((*keys, competitor, len(paired), float(diff.mean()),
                          float((diff > 0).mean()), pvalue))
    tests = pd.DataFrame(tests, columns=comparison_keys + [
        "competitor", "pairs", "mean_difference", "win_rate", "p_raw"
    ])
    if len(tests):
        tests["p_holm"] = holm(tests.p_raw.to_numpy())
    result_dir = path.parent
    summary.to_csv(result_dir / "pilot_summary.csv", index=False)
    tests.to_csv(result_dir / "pilot_paired_tests.csv", index=False)
    global_tests = []
    for (mode, switch_cost), care in proposed.groupby(["mode", "switch_cost"]):
        subset = df[(df["mode"] == mode) & (df["switch_cost"] == switch_cost)]
        for competitor in sorted(set(subset.algorithm) - {"CARE-DS"}):
            other = subset[subset.algorithm == competitor]
            paired = care[["instance_id", "lifetime"]].merge(
                other[["instance_id", "lifetime"]], on="instance_id", suffixes=("_care", "_other")
            )
            if len(paired) < 2:
                continue
            diff = paired.lifetime_care - paired.lifetime_other
            pvalue = 1.0 if np.all(diff == 0) else float(
                wilcoxon(diff, alternative="two-sided").pvalue
            )
            global_tests.append({
                "mode": mode,
                "switch_cost": switch_cost,
                "competitor": competitor,
                "pairs": len(paired),
                "care_mean": paired.lifetime_care.mean(),
                "competitor_mean": paired.lifetime_other.mean(),
                "relative_improvement_pct": 100 * (
                    paired.lifetime_care.mean() / paired.lifetime_other.mean() - 1
                ),
                "mean_difference": diff.mean(),
                "wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "losses": int((diff < 0).sum()),
                "p_raw": pvalue,
            })
    global_tests = pd.DataFrame(global_tests)
    if len(global_tests):
        global_tests["p_holm"] = holm(global_tests.p_raw.to_numpy())
    global_tests.to_csv(result_dir / "pilot_global_paired_tests.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaired CARE-DS comparisons\n", tests.to_string(index=False))
    print("\nGlobal paired CARE-DS comparisons\n", global_tests.to_string(index=False))


if __name__ == "__main__":
    main()
