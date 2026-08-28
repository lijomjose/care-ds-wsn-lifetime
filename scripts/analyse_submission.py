#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def bootstrap_mean_ci(values, seed=20260822, samples=3000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def stable_seed(*parts):
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 2**32


def matched_rank_biserial(values):
    """Matched-pairs rank-biserial correlation for nonzero paired differences."""
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    return float(
        (ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum()
    )


def paired_rows(df, study):
    condition_cols = [
        c for c in ["mode", "switch_cost", "failure_scenario", "energy_model"]
        if c in df.columns
    ]
    identifier = ["instance_id"] + condition_cols
    care = df[df.algorithm == "CARE-DS"][identifier + ["lifetime", "runtime_s"]]
    rows = []
    for conditions, care_block in care.groupby(condition_cols, dropna=False):
        if not isinstance(conditions, tuple):
            conditions = (conditions,)
        subset = df.copy()
        for column, value in zip(condition_cols, conditions):
            subset = subset[subset[column] == value]
        for algorithm in sorted(set(subset.algorithm) - {"CARE-DS"}):
            other = subset[subset.algorithm == algorithm][identifier + ["lifetime", "runtime_s"]]
            paired = care_block.merge(other, on=identifier, suffixes=("_care", "_other"))
            if len(paired) < 5:
                continue
            diff = paired.lifetime_care - paired.lifetime_other
            pvalue = 1.0 if np.all(diff == 0) else float(wilcoxon(diff, alternative="two-sided").pvalue)
            low, high = bootstrap_mean_ci(diff, seed=stable_seed(study, algorithm, conditions))
            effect = matched_rank_biserial(diff.to_numpy())
            row = {
                "study": study, "competitor": algorithm, "pairs": len(paired),
                "care_mean": paired.lifetime_care.mean(),
                "competitor_mean": paired.lifetime_other.mean(),
                "relative_improvement_pct": 100 * (paired.lifetime_care.mean() / paired.lifetime_other.mean() - 1),
                "mean_instance_improvement_pct": float(np.mean(
                    100 * (paired.lifetime_care / paired.lifetime_other - 1)
                )),
                "mean_difference": diff.mean(), "ci95_low": low, "ci95_high": high,
                "wins": int(np.sum(diff > 0)), "ties": int(np.sum(diff == 0)),
                "losses": int(np.sum(diff < 0)), "rank_biserial": effect,
                "care_runtime_s": paired.runtime_s_care.mean(),
                "competitor_runtime_s": paired.runtime_s_other.mean(), "p_raw": pvalue,
            }
            row.update(dict(zip(condition_cols, conditions)))
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm"] = holm(result.p_raw.to_numpy())
    return result


def best_competitor_rows(df, study):
    condition_cols = [
        c for c in ["mode", "switch_cost", "failure_scenario", "energy_model"]
        if c in df.columns
    ]
    rows = []
    for conditions, block in df.groupby(condition_cols, dropna=False):
        if not isinstance(conditions, tuple):
            conditions = (conditions,)
        pivot = block.pivot_table(index="instance_id", columns="algorithm", values="lifetime", aggfunc="first")
        if "CARE-DS" not in pivot or len(pivot) < 5:
            continue
        competitors = [c for c in pivot.columns if c != "CARE-DS"]
        paired = pivot[["CARE-DS"] + competitors].dropna(subset=["CARE-DS"])
        best = paired[competitors].max(axis=1, skipna=True)
        valid = best.notna()
        diff = paired.loc[valid, "CARE-DS"] - best[valid]
        if len(diff) < 5:
            continue
        pvalue = 1.0 if np.all(diff == 0) else float(wilcoxon(diff, alternative="two-sided").pvalue)
        low, high = bootstrap_mean_ci(diff, seed=stable_seed(study, "best", conditions))
        row = {
            "study": study, "pairs": len(diff), "care_mean": paired.loc[valid, "CARE-DS"].mean(),
            "best_competitor_mean": best[valid].mean(),
            "relative_improvement_pct": 100 * (paired.loc[valid, "CARE-DS"].mean() / best[valid].mean() - 1),
            "mean_instance_improvement_pct": float(np.mean(
                100 * (paired.loc[valid, "CARE-DS"] / best[valid] - 1)
            )),
            "mean_difference": diff.mean(), "ci95_low": low, "ci95_high": high,
            "wins": int(np.sum(diff > 0)), "ties": int(np.sum(diff == 0)),
            "losses": int(np.sum(diff < 0)),
            "rank_biserial": matched_rank_biserial(diff.to_numpy()),
            "p_raw": pvalue,
        }
        row.update(dict(zip(condition_cols, conditions)))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm"] = holm(result.p_raw.to_numpy())
    return result


def summarize(df, study):
    columns = [
        c for c in ["mode", "switch_cost", "failure_scenario", "energy_model", "algorithm"]
        if c in df.columns
    ]
    result = df.groupby(columns, as_index=False).agg(
        instances=("lifetime", "size"), mean_lifetime=("lifetime", "mean"),
        median_lifetime=("lifetime", "median"), sd_lifetime=("lifetime", "std"),
        mean_runtime_s=("runtime_s", "mean"), mean_switches=("switches", "mean"),
        validity=("valid", "mean"),
    )
    result.insert(0, "study", study)
    return result


def modern_static_analysis(df):
    """Return aggregate quality, ranks, and paired non-inferiority evidence."""
    index = ["instance_id", "mode", "switch_cost", "failure_scenario", "energy_model"]
    index = [c for c in index if c in df.columns]
    pivot = df.pivot_table(index=index, columns="algorithm", values="lifetime", aggfunc="first")
    pivot = pivot.dropna(axis=0, how="any")
    algorithms = list(pivot.columns)
    if pivot.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    ranks = np.vstack([
        rankdata(-row, method="average") for row in pivot.to_numpy(dtype=float)
    ])
    aggregate = pd.DataFrame({
        "algorithm": algorithms,
        "instances": len(pivot),
        "mean_lifetime": pivot.mean(axis=0).to_numpy(),
        "median_lifetime": pivot.median(axis=0).to_numpy(),
        "mean_rank": ranks.mean(axis=0),
        "wins_or_ties": [int((pivot[a] == pivot.max(axis=1)).sum()) for a in algorithms],
        "mean_gap_to_best_pct": [
            float((100 * (pivot.max(axis=1) - pivot[a]) / pivot.max(axis=1)).mean())
            for a in algorithms
        ],
    }).sort_values(["mean_rank", "mean_gap_to_best_pct"])

    pair_rows = []
    for i, first in enumerate(algorithms):
        for second in algorithms[i + 1:]:
            diff = pivot[first] - pivot[second]
            if np.all(diff == 0):
                pvalue = 1.0
            else:
                pvalue = float(wilcoxon(diff, alternative="two-sided").pvalue)
            low, high = bootstrap_mean_ci(diff, seed=stable_seed("static", first, second))
            pair_rows.append({
                "first": first, "second": second, "pairs": len(diff),
                "first_mean": pivot[first].mean(), "second_mean": pivot[second].mean(),
                "mean_difference": diff.mean(), "ci95_low": low, "ci95_high": high,
                "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                "losses": int((diff < 0).sum()), "p_raw": pvalue,
            })
    pairs = pd.DataFrame(pair_rows)
    pairs["p_holm"] = holm(pairs.p_raw.to_numpy())
    friedman_p = 1.0
    if len(algorithms) >= 3:
        friedman_p = float(friedmanchisquare(*[pivot[a].to_numpy() for a in algorithms]).pvalue)
    evidence = {
        "complete_instances": int(len(pivot)),
        "algorithms": algorithms,
        "friedman_p": friedman_p,
        "best_mean_rank": str(aggregate.iloc[0].algorithm),
    }
    return aggregate, pairs, evidence


def detailed_scaling(df, study):
    columns = [c for c in ["topology", "n", "target_degree", "mode", "switch_cost",
                           "failure_scenario", "algorithm"] if c in df.columns]
    if not columns:
        return pd.DataFrame()
    result = df.groupby(columns, as_index=False).agg(
        instances=("lifetime", "size"), mean_lifetime=("lifetime", "mean"),
        median_lifetime=("lifetime", "median"), mean_runtime_s=("runtime_s", "mean"),
        mean_switches=("switches", "mean"), validity=("valid", "mean"),
    )
    result.insert(0, "study", study)
    return result


def runtime_fairness(df):
    required = {"CARE-DS", "MS-RG-WT"}
    if not required <= set(df.algorithm):
        return pd.DataFrame()
    index = ["instance_id", "mode", "switch_cost"]
    pivot = df[df.algorithm.isin(required)].pivot_table(
        index=index, columns="algorithm", values="runtime_s", aggfunc="first"
    ).dropna().reset_index()
    metadata = df[["instance_id", "n", "topology"]].drop_duplicates()
    pivot = pivot.merge(metadata, on="instance_id", validate="many_to_one")
    pivot["care_to_matched_runtime_ratio"] = pivot["CARE-DS"] / pivot["MS-RG-WT"]
    return pivot.groupby(["mode", "n"], as_index=False).agg(
        pairs=("care_to_matched_runtime_ratio", "size"),
        mean_ratio=("care_to_matched_runtime_ratio", "mean"),
        median_ratio=("care_to_matched_runtime_ratio", "median"),
        care_mean_s=("CARE-DS", "mean"),
        matched_mean_s=("MS-RG-WT", "mean"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=Path)
    parser.add_argument("--radio", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--modern-static", type=Path)
    parser.add_argument("--modern-unweighted", type=Path)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--exact", type=Path, default=ROOT / "results" / "exact_hard.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    inputs = {
        "unit": args.unit, "radio": args.radio, "failures": args.failures,
        "ablation": args.ablation,
    }
    summaries, pairs, bests, details = [], [], [], []
    for study, path in inputs.items():
        if path is None or not path.exists():
            continue
        data = pd.read_csv(path)
        if not data.valid.all():
            raise RuntimeError(f"invalid schedule in {path}")
        summaries.append(summarize(data, study))
        details.append(detailed_scaling(data, study))
        pairs.append(paired_rows(data, study))
        if study != "ablation":
            bests.append(best_competitor_rows(data, study))
    result_dir = args.output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    paired = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    best = pd.concat(bests, ignore_index=True) if bests else pd.DataFrame()
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary.to_csv(result_dir / "submission_summary.csv", index=False)
    paired.to_csv(result_dir / "submission_paired_tests.csv", index=False)
    best.to_csv(result_dir / "submission_best_competitor.csv", index=False)
    detail.to_csv(result_dir / "submission_detailed_summary.csv", index=False)
    unit_path = inputs.get("unit")
    if unit_path is not None and unit_path.exists():
        runtime_fairness(pd.read_csv(unit_path)).to_csv(
            result_dir / "runtime_fairness.csv", index=False
        )

    static_evidence = {}
    if args.modern_static is not None and args.modern_static.exists():
        static = pd.read_csv(args.modern_static)
        if not static.valid.all():
            raise RuntimeError(f"invalid schedule in {args.modern_static}")
        static_summary = detailed_scaling(static, "modern_static")
        static_summary.to_csv(result_dir / "modern_static_summary.csv", index=False)
        static_aggregate, static_pairs, static_evidence = modern_static_analysis(static)
        static_aggregate.to_csv(result_dir / "modern_static_ranks.csv", index=False)
        static_pairs.to_csv(result_dir / "modern_static_paired_tests.csv", index=False)
    unweighted_evidence = {}
    if args.modern_unweighted is not None and args.modern_unweighted.exists():
        unweighted = pd.read_csv(args.modern_unweighted)
        if not unweighted.valid.all():
            raise RuntimeError(f"invalid schedule in {args.modern_unweighted}")
        unweighted_summary = detailed_scaling(unweighted, "modern_unweighted")
        unweighted_summary.to_csv(result_dir / "modern_unweighted_summary.csv", index=False)
        unweighted_ranks, unweighted_pairs, unweighted_evidence = modern_static_analysis(unweighted)
        unweighted_ranks.to_csv(result_dir / "modern_unweighted_ranks.csv", index=False)
        unweighted_pairs.to_csv(result_dir / "modern_unweighted_paired_tests.csv", index=False)

    exact_summary = pd.DataFrame()
    if args.exact.exists():
        exact = pd.read_csv(args.exact)
        exact_summary = exact.groupby(["mode", "algorithm"], as_index=False).agg(
            instances=("optimal", "size"), exact_hits=("optimal", "sum"),
            mean_ratio=("ratio_to_optimum", "mean"), min_ratio=("ratio_to_optimum", "min"),
        )
        exact_summary.to_csv(result_dir / "exact_hard_summary.csv", index=False)

    evidence = {
        "studies_loaded": [k for k, p in inputs.items() if p is not None and p.exists()],
        "all_schedules_valid": bool((summary.validity == 1).all()) if len(summary) else False,
        "care_positive_vs_best_cells": int((best.mean_difference > 0).sum()) if len(best) else 0,
        "care_nonpositive_vs_best_cells": int((best.mean_difference <= 0).sum()) if len(best) else 0,
        "care_significant_positive_cells": int(((best.mean_difference > 0) & (best.p_holm < 0.05)).sum()) if len(best) else 0,
        "modern_static": static_evidence,
        "modern_unweighted": unweighted_evidence,
    }
    (result_dir / "submission_evidence.json").write_text(json.dumps(evidence, indent=2))
    report = ["SUBMISSION STUDY ANALYSIS", "", json.dumps(evidence, indent=2), ""]
    report += ["BEST-COMPETITOR TESTS", best.to_string(index=False), "", "EXACT SUMMARY", exact_summary.to_string(index=False)]
    if static_evidence:
        report += ["", "MODERN STATIC EVIDENCE", json.dumps(static_evidence, indent=2)]
    if unweighted_evidence:
        report += ["", "MODERN UNWEIGHTED EVIDENCE", json.dumps(unweighted_evidence, indent=2)]
    (result_dir / "submission_analysis.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
