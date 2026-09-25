#!/usr/bin/env python3
"""Fail-fast integrity checks for every reported result table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC = {
    "EAA-static", "GH-MWDDS+", "PBIG-R", "MC-CMSA-R", "FSS-2026-R",
    "EAAS-S4C-MAB-R",
}


def expected_rows(config):
    instances = (
        len(config["topologies"]) * len(config["node_sizes"])
        * len(config["target_degrees"]) * int(config["runs"])
    )
    scenarios = config.get("failure_scenarios", ["none"])
    rows_per_instance = 0
    for algorithm in config["algorithms"]:
        if algorithm in STATIC:
            rows_per_instance += len(scenarios)  # dominating, first switch cost only
        else:
            rows_per_instance += (
                len(config["modes"]) * len(config["switch_costs"]) * len(scenarios)
            )
    return instances, instances * rows_per_instance


def validate(name, config_path, result_path):
    config = json.loads(config_path.read_text())
    expected_instances, expected = expected_rows(config)
    data = pd.read_csv(result_path)
    key = ["instance_id", "mode", "switch_cost", "failure_scenario", "algorithm"]
    key = [column for column in key if column in data.columns]
    report = {
        "study": name,
        "result": str(result_path.relative_to(ROOT)),
        "rows": int(len(data)),
        "expected_rows": int(expected),
        "instances": int(data.instance_id.nunique()),
        "expected_instances": int(expected_instances),
        "duplicate_rows": int(data.duplicated(key).sum()),
        "invalid_rows": int((~data.valid.astype(bool)).sum()),
        "missing_values": int(data.isna().sum().sum()),
    }
    report["passed"] = bool(
        report["rows"] == expected
        and report["instances"] == expected_instances
        and report["duplicate_rows"] == 0
        and report["invalid_rows"] == 0
        and report["missing_values"] == 0
    )
    return report


def validate_lower_tail(summary_path, deciles_path):
    if not summary_path.exists() or not deciles_path.exists():
        return {
            "study": "lower_tail_analysis",
            "result": f"{summary_path.relative_to(ROOT)}; {deciles_path.relative_to(ROOT)}",
            "passed": False,
            "reason": "missing",
        }
    summary = pd.read_csv(summary_path)
    deciles = pd.read_csv(deciles_path)
    expected_pairs = {
        ("unit", "dominating", "all"): 5400,
        ("unit", "dominating", "lowest_quintile"): 1080,
        ("unit", "dominating", "lowest_decile"): 540,
        ("unit", "connected", "all"): 5400,
        ("unit", "connected", "lowest_quintile"): 1080,
        ("unit", "connected", "lowest_decile"): 540,
        ("radio", "dominating", "all"): 450,
        ("radio", "dominating", "lowest_quintile"): 90,
        ("radio", "dominating", "lowest_decile"): 45,
        ("radio", "connected", "all"): 450,
        ("radio", "connected", "lowest_quintile"): 90,
        ("radio", "connected", "lowest_decile"): 45,
    }
    observed_pairs = {
        (row.study, row.mode, row.subset): int(row.pairs)
        for row in summary.itertuples()
    }
    finite_summary = summary[
        ["care_mean", "best_comparator_mean", "aggregate_gain_pct"]
    ].notna().all().all()
    report = {
        "study": "lower_tail_analysis",
        "result": f"{summary_path.relative_to(ROOT)}; {deciles_path.relative_to(ROOT)}",
        "summary_rows": int(len(summary)),
        "decile_rows": int(len(deciles)),
        "duplicate_summary_rows": int(summary.duplicated(["study", "mode", "subset"]).sum()),
        "duplicate_decile_rows": int(deciles.duplicated(["study", "mode", "decile"]).sum()),
        "expected_pair_counts": bool(observed_pairs == expected_pairs),
        "finite_core_statistics": bool(finite_summary),
    }
    report["passed"] = bool(
        report["summary_rows"] == 12
        and report["decile_rows"] == 40
        and report["duplicate_summary_rows"] == 0
        and report["duplicate_decile_rows"] == 0
        and report["expected_pair_counts"]
        and report["finite_core_statistics"]
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    corrected = ROOT / "results" / "corrected" / "merged"
    def preferred(corrected_name, legacy_name):
        candidate = corrected / corrected_name
        return candidate if candidate.exists() else ROOT / "results" / legacy_name

    studies = [
        ("unit", ROOT / "configs/submission_event.json",
         preferred("submission_event_raw.csv", "submission_event_raw.csv")),
        ("modern_static", ROOT / "configs/modern_static.json", ROOT / "results/modern_static_raw.csv"),
        ("modern_unweighted", ROOT / "configs/modern_unweighted.json", ROOT / "results/modern_unweighted_raw.csv"),
        ("radio", ROOT / "configs/radio.json",
         preferred("radio_raw.csv", "radio_raw.csv")),
        ("failures", ROOT / "configs/failures.json",
         preferred("failures_raw.csv", "failures_raw.csv")),
        ("ablation", ROOT / "configs/ablation_submission.json",
         preferred("ablation_submission_raw.csv", "ablation_submission_raw.csv")),
        ("radio_dwell_control", ROOT / "configs/radio_dwell_control.json",
         ROOT / "results/revision/radio_dwell_control_raw.csv"),
        ("radio_ablation", ROOT / "configs/radio_ablation.json",
         ROOT / "results/revision/radio_ablation_raw.csv"),
    ]
    reports = []
    for name, config_path, result_path in studies:
        if not result_path.exists():
            reports.append({"study": name, "result": str(result_path), "passed": False,
                            "reason": "missing"})
            continue
        reports.append(validate(name, config_path, result_path))
    reports.append(validate_lower_tail(
        ROOT / "results/revision/analysis/lower_tail_summary.csv",
        ROOT / "results/revision/analysis/lower_tail_deciles.csv",
    ))
    output = ROOT / "results/submission_validation.json"
    output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    if not args.allow_incomplete and not all(item["passed"] for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
