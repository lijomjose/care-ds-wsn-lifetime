#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CORRECTED = ROOT / "results" / "corrected" / "merged"
DATA = CORRECTED if (CORRECTED / "submission_event_raw.csv").exists() else ROOT / "results"
COLORS = {
    "D-EAA": "#7f7f7f", "D-EAA-ER": "#7f7f7f", "DWG": "#4c78a8", "MS-RG-TM": "#f58518",
    "MS-RG-WT": "#e45756", "CARE-DS": "#2ca02c",
    "EAA-static": "#9c755f", "GH-MWDDS+": "#bab0ac", "PBIG-R": "#4c78a8",
    "MC-CMSA-R": "#f58518", "FSS-2026-R": "#e45756", "EAAS-S4C-MAB-R": "#9467bd",
    "CARE-noreserve": "#7f7f7f", "CARE-noswitch": "#4c78a8",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def unit_figures(path):
    data = pd.read_csv(path)
    algorithms = ["D-EAA-ER", "DWG", "MS-RG-WT", "CARE-DS"]
    view = data[data.algorithm.isin(algorithms)]
    means = view.groupby(["mode", "switch_cost", "algorithm"], as_index=False).lifetime.mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for ax, mode in zip(axes, ["dominating", "connected"]):
        block = means[means["mode"] == mode]
        for algorithm in algorithms:
            line = block[block.algorithm == algorithm].sort_values("switch_cost")
            ax.plot(line.switch_cost, line.lifetime, marker="o", lw=1.8,
                    color=COLORS[algorithm], label=algorithm)
        ax.set_title("Ordinary domination" if mode == "dominating" else "Connected domination")
        ax.set_xlabel("Handover cost")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Mean service lifetime (rounds)")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "submission_lifetime_handover")

    pivot = view.pivot_table(
        index=["instance_id", "mode", "switch_cost"], columns="algorithm", values="lifetime"
    ).reset_index()
    competitors = [x for x in algorithms if x != "CARE-DS"]
    pivot["best"] = pivot[competitors].max(axis=1)
    pivot["improvement_pct"] = 100 * (pivot["CARE-DS"] / pivot["best"] - 1)
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    for mode, marker in [("dominating", "o"), ("connected", "s")]:
        block = pivot[pivot["mode"] == mode]
        grouped = block.groupby("switch_cost").improvement_pct
        mean = grouped.mean()
        se = grouped.sem()
        ax.errorbar(mean.index, mean.values, yerr=1.96 * se.values, marker=marker,
                    capsize=3, lw=1.8, label=mode)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("Handover cost")
    ax.set_ylabel("CARE-DS improvement over\nbest dynamic comparator (%)")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    save(fig, "submission_best_improvement")

    runtime = view.pivot_table(
        index=["instance_id", "mode", "switch_cost", "n"],
        columns="algorithm", values="runtime_s"
    ).reset_index()
    runtime["ratio"] = runtime["CARE-DS"] / runtime["MS-RG-WT"]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for mode, marker in [("dominating", "o"), ("connected", "s")]:
        part = runtime[runtime["mode"] == mode].groupby("n").ratio
        mean, se = part.mean(), part.sem()
        ax.errorbar(mean.index, mean.values, yerr=1.96 * se.values,
                    marker=marker, capsize=3, lw=1.8, label=mode)
    ax.axhline(1.0, color="black", lw=.8)
    ax.set_xlabel("Number of vertices")
    ax.set_ylabel("CARE-DS / MS-RG-WT runtime")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    save(fig, "submission_runtime_fairness")


def supplementary_figures(radio_path, failures_path, exact_path):
    if radio_path.exists():
        data = pd.read_csv(radio_path)
        shown = ["EAAS-S4C-MAB-R", "D-EAA-ER", "DWG", "MS-RG-WT", "CARE-DS"]
        block = data[data.algorithm.isin(shown)]
        means = block.groupby(["mode", "switch_cost", "algorithm"], as_index=False).lifetime.mean()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, mode in zip(axes, ["dominating", "connected"]):
            part = means[means["mode"] == mode]
            mode_shown = (
                ["EAAS-S4C-MAB-R", "D-EAA-ER", "MS-RG-WT", "CARE-DS"]
                if mode == "dominating"
                else ["D-EAA-ER", "DWG", "MS-RG-WT", "CARE-DS"]
            )
            cost_order = sorted(data.switch_cost.unique())
            x = np.arange(len(cost_order))
            for j, algorithm in enumerate(mode_shown):
                vals = part[part.algorithm == algorithm].set_index("switch_cost").reindex(cost_order)
                ax.bar(x + (j - 1.5) * .19, vals.lifetime, width=.18,
                       color=COLORS[algorithm], label=algorithm)
            ax.set_xticks(x, [str(v) for v in cost_order])
            ax.set_title(mode.capitalize())
            ax.set_xlabel("Wake-up energy (J)")
        axes[0].set_ylabel("Mean radio-model lifetime")
        axes[1].legend(frameon=False, fontsize=8)
        save(fig, "submission_radio")

    if failures_path.exists():
        data = pd.read_csv(failures_path)
        shown = ["EAAS-S4C-MAB-R", "D-EAA-ER", "MS-RG-WT", "CARE-DS"]
        block = data[data.algorithm.isin(shown)]
        means = block.groupby(["mode", "failure_scenario", "algorithm"], as_index=False).lifetime.mean()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        order = ["none", "random", "targeted"]
        for ax, mode in zip(axes, ["dominating", "connected"]):
            part = means[means["mode"] == mode]
            x = np.arange(len(order))
            for j, algorithm in enumerate(shown):
                vals = part[part.algorithm == algorithm].set_index("failure_scenario").reindex(order)
                ax.bar(x + (j - 1.5) * .19, vals.lifetime, width=.18,
                       color=COLORS[algorithm], label=algorithm)
            ax.set_xticks(x, ["No failure", "Random", "Low-reserve"])
            ax.set_title(mode.capitalize())
        axes[0].set_ylabel("Mean service lifetime")
        axes[1].legend(frameon=False, fontsize=8)
        save(fig, "submission_failures")

    if exact_path.exists():
        data = pd.read_csv(exact_path)
        preferred = ["D-EAA-ER", "D-EAA", "DWG", "MS-RG-WT", "MS-RG-TM", "CARE-DS"]
        order = [algorithm for algorithm in preferred if algorithm in set(data.algorithm)]
        means = data.groupby(["mode", "algorithm"], as_index=False).ratio_to_optimum.mean()
        fig, ax = plt.subplots(figsize=(5.8, 3.0))
        x = np.arange(len(order))
        width = .34
        modes = [mode for mode in ["dominating", "connected"] if mode in set(data["mode"])]
        for j, mode in enumerate(modes):
            vals = means[means["mode"] == mode].set_index("algorithm").reindex(order)
            offset = 0 if len(modes) == 1 else (j - .5) * width
            ax.bar(x + offset, vals.ratio_to_optimum, width=width, label=mode)
        ax.set_xticks(x, order, rotation=15)
        floor = max(0.0, float(means.ratio_to_optimum.min()) - .05)
        ax.set_ylim(floor, 1.01)
        ax.set_ylabel("Mean ratio to exact optimum")
        if len(modes) > 1:
            ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.25)
        save(fig, "submission_exact")


def modern_and_ablation_figures(modern_path, unweighted_path, ablation_path):
    if modern_path.exists():
        data = pd.read_csv(modern_path)
        order = ["EAA-static", "GH-MWDDS+", "PBIG-R", "MC-CMSA-R", "FSS-2026-R", "EAAS-S4C-MAB-R"]
        means = data.groupby(["n", "algorithm"], as_index=False).lifetime.mean()
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        for algorithm in order:
            part = means[means.algorithm == algorithm].sort_values("n")
            ax.plot(part.n, part.lifetime, marker="o", ms=3.5, lw=1.5,
                    color=COLORS[algorithm], label=algorithm)
        ax.set_xlabel("Number of vertices")
        ax.set_ylabel("Mean static-family lifetime")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        save(fig, "submission_modern_static")

    if unweighted_path.exists():
        data = pd.read_csv(unweighted_path)
        order = ["EAA-static", "GH-MWDDS+", "PBIG-R", "MC-CMSA-R", "FSS-2026-R"]
        means = data.groupby(["n", "algorithm"], as_index=False).lifetime.mean()
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        for algorithm in order:
            part = means[means.algorithm == algorithm].sort_values("n")
            ax.plot(part.n, part.lifetime, marker="o", ms=3.5, lw=1.5,
                    color=COLORS[algorithm], label=algorithm)
        ax.set_xlabel("Number of vertices")
        ax.set_ylabel("Mean disjoint dominating sets")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        save(fig, "submission_modern_unweighted")

    if ablation_path.exists():
        data = pd.read_csv(ablation_path)
        order = ["CARE-noreserve", "CARE-noswitch", "CARE-DS"]
        means = data.groupby(["mode", "switch_cost", "algorithm"], as_index=False).lifetime.mean()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, mode in zip(axes, ["dominating", "connected"]):
            part = means[means["mode"] == mode]
            for algorithm in order:
                line = part[part.algorithm == algorithm].sort_values("switch_cost")
                ax.plot(line.switch_cost, line.lifetime, marker="o", lw=1.7,
                        color=COLORS[algorithm], label=algorithm)
            ax.set_title(mode.capitalize())
            ax.set_xlabel("Handover cost")
            ax.grid(alpha=.25)
        axes[0].set_ylabel("Mean service lifetime")
        axes[1].legend(frameon=False, fontsize=8)
        save(fig, "submission_ablation")


def main():
    unit = DATA / "submission_event_raw.csv"
    unit_figures(unit)
    supplementary_figures(
        DATA / "radio_raw.csv",
        DATA / "failures_raw.csv",
        (ROOT / "results" / "exact_challenge.csv")
        if (ROOT / "results" / "exact_challenge.csv").exists()
        else (ROOT / "results" / "exact_stress.csv"),
    )
    modern_and_ablation_figures(
        ROOT / "results" / "modern_static_raw.csv",
        ROOT / "results" / "modern_unweighted_raw.csv",
        DATA / "ablation_submission_raw.csv",
    )


if __name__ == "__main__":
    main()
