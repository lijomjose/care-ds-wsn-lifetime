#!/usr/bin/env python3
"""Exploratory lower-tail robustness analysis for CARE-DS.

The script uses only completed, validated per-instance result matrices.  Within
each fixed topology/size/density/mode/handover stratum, instances are ordered
by the lifetime of the best observed comparator.  CARE-DS outcomes never enter
the tail definition.  Ties are broken deterministically by instance ID.

Because this question was formulated after the primary matrices were complete,
the outputs are explicitly labelled exploratory rather than confirmatory.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIT = ROOT / "results" / "submission_event_raw.csv"
DEFAULT_RADIO = ROOT / "results" / "radio_raw.csv"
DEFAULT_OUTPUT = ROOT / "results"
DEFAULT_FIGURES = ROOT / "paper" / "figures"


@dataclass(frozen=True)
class Study:
    name: str
    label: str
    path: Path
    switch_costs: tuple[float, ...]
    competitors: tuple[str, ...] | None


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 2**32


def prepare(study: Study) -> pd.DataFrame:
    raw = pd.read_csv(study.path)
    if not raw["valid"].all():
        raise RuntimeError(f"Invalid schedule found in {study.path}")

    index = ["instance_id", "mode", "switch_cost"]
    metadata = raw.drop_duplicates(index)[
        index + ["topology", "n", "target_degree"]
    ]
    pivot = raw.pivot_table(
        index=index, columns="algorithm", values="lifetime", aggfunc="first"
    ).reset_index()
    if "CARE-DS" not in pivot:
        raise RuntimeError(f"CARE-DS is absent from {study.path}")

    if study.competitors is None:
        competitors = [c for c in pivot.columns if c not in index + ["CARE-DS"]]
    else:
        competitors = list(study.competitors)
    missing = set(competitors) - set(pivot.columns)
    if missing:
        raise RuntimeError(f"Missing comparators in {study.path}: {sorted(missing)}")

    pivot["best_comparator"] = pivot[competitors].max(axis=1, skipna=True)
    pivot = pivot.merge(metadata, on=index, validate="one_to_one")
    selected = np.zeros(len(pivot), dtype=bool)
    for cost in study.switch_costs:
        selected |= np.isclose(pivot["switch_cost"].to_numpy(float), cost)
    pivot = pivot.loc[selected].copy()
    pivot = pivot.dropna(subset=["CARE-DS", "best_comparator"])
    if pivot.empty:
        raise RuntimeError(f"No selected conditions in {study.path}")

    strata = ["topology", "n", "target_degree", "mode", "switch_cost"]
    pivot = pivot.sort_values(strata + ["best_comparator", "instance_id"])
    pivot["tail_rank"] = pivot.groupby(strata, sort=False).cumcount() + 1
    pivot["stratum_size"] = pivot.groupby(strata, sort=False)["instance_id"].transform("size")
    pivot["decile"] = np.ceil(10 * pivot["tail_rank"] / pivot["stratum_size"]).astype(int)
    pivot["difference"] = pivot["CARE-DS"] - pivot["best_comparator"]
    pivot["instance_gain_pct"] = 100 * pivot["difference"] / pivot["best_comparator"]
    pivot["study"] = study.name
    if (pivot["stratum_size"] < 10).any():
        raise RuntimeError(f"A selected stratum in {study.path} has fewer than ten instances")
    if set(pivot["decile"].unique()) != set(range(1, 11)):
        raise RuntimeError(f"Incomplete decile coverage in {study.path}")
    return pivot


def aggregate(block: pd.DataFrame) -> dict[str, float | int]:
    diff = block["difference"].to_numpy(float)
    care = block["CARE-DS"].to_numpy(float)
    best = block["best_comparator"].to_numpy(float)
    return {
        "pairs": len(block),
        "care_mean": float(care.mean()),
        "best_comparator_mean": float(best.mean()),
        "aggregate_gain_pct": float(100 * (care.mean() / best.mean() - 1)),
        "mean_instance_gain_pct": float(block["instance_gain_pct"].mean()),
        "median_instance_gain_pct": float(block["instance_gain_pct"].median()),
        "wins": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "losses": int(np.sum(diff < 0)),
    }


def stratified_tail_bootstrap(
    block: pd.DataFrame,
    fraction: float,
    samples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Return CIs for lower-tail gain and its contrast with the remaining rows.

    Each bootstrap replicate resamples paired graph instances within every
    fixed design stratum, reorders them using comparator lifetime only, and
    reselects the requested lower tail.
    """
    strata = ["topology", "n", "target_degree", "mode", "switch_cost"]
    arrays = []
    for _, part in block.groupby(strata, sort=False):
        arrays.append(
            (
                part["CARE-DS"].to_numpy(float),
                part["best_comparator"].to_numpy(float),
            )
        )
    rng = np.random.default_rng(seed)
    tail_gain = np.empty(samples)
    contrast = np.empty(samples)
    for sample in range(samples):
        low_care = low_best = rest_care = rest_best = 0.0
        for care, best in arrays:
            n = len(care)
            draw = rng.integers(0, n, size=n)
            sampled_best = best[draw]
            sampled_care = care[draw]
            # `care` and `best` inherit the observed best-lifetime/instance-ID
            # order. Lexicographic sorting by sampled best and original row
            # index therefore preserves the declared independent tie break.
            order = np.lexsort((draw, sampled_best))
            cutoff = int(np.ceil(fraction * n))
            low = order[:cutoff]
            rest = order[cutoff:]
            low_care += sampled_care[low].sum()
            low_best += sampled_best[low].sum()
            rest_care += sampled_care[rest].sum()
            rest_best += sampled_best[rest].sum()
        tail_gain[sample] = 100 * (low_care / low_best - 1)
        rest_gain = 100 * (rest_care / rest_best - 1)
        contrast[sample] = tail_gain[sample] - rest_gain
    tail_ci = np.quantile(tail_gain, [0.025, 0.975])
    contrast_ci = np.quantile(contrast, [0.025, 0.975])
    return float(tail_ci[0]), float(tail_ci[1]), float(contrast_ci[0]), float(contrast_ci[1])


def summarize(studies: list[tuple[Study, pd.DataFrame]], samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    for study, data in studies:
        for mode, block in data.groupby("mode", sort=False):
            for subset, fraction in [("all", 1.0), ("lowest_quintile", 0.2), ("lowest_decile", 0.1)]:
                if subset == "all":
                    selected = block
                    ci = (np.nan, np.nan, np.nan, np.nan)
                else:
                    cutoff = np.ceil(fraction * block["stratum_size"])
                    selected = block[block["tail_rank"] <= cutoff]
                    ci = stratified_tail_bootstrap(
                        block,
                        fraction=fraction,
                        samples=samples,
                        seed=stable_seed(study.name, mode, subset),
                    )
                row: dict[str, object] = {
                    "study": study.name,
                    "study_label": study.label,
                    "mode": mode,
                    "subset": subset,
                    "tail_fraction": fraction,
                }
                row.update(aggregate(selected))
                row.update(
                    {
                        "aggregate_gain_ci95_low": ci[0],
                        "aggregate_gain_ci95_high": ci[1],
                        "tail_minus_rest_ci95_low": ci[2],
                        "tail_minus_rest_ci95_high": ci[3],
                    }
                )
                summary_rows.append(row)

            for decile, selected in block.groupby("decile", sort=True):
                row = {
                    "study": study.name,
                    "study_label": study.label,
                    "mode": mode,
                    "decile": int(decile),
                }
                row.update(aggregate(selected))
                decile_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(decile_rows)


def make_figure(deciles: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    modes = [("dominating", "Ordinary", "o"), ("connected", "Connected", "s")]
    studies = [("unit", "Unit cost: $h/a=0.25,0.5$"), ("radio", "Radio: $h/a_{ref}=0.5$")]
    for ax, (study, title) in zip(axes, studies):
        part = deciles[deciles["study"] == study]
        for mode, label, marker in modes:
            line = part[part["mode"] == mode].sort_values("decile")
            ax.plot(
                line["decile"],
                line["aggregate_gain_pct"],
                marker=marker,
                linewidth=1.8,
                markersize=4,
                label=label,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Best-comparator lifetime decile")
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("CARE-DS aggregate gain (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(
        0.5,
        -0.01,
        "1 = lowest competitor lifetime within each fixed design stratum; 10 = highest",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "submission_lower_tail.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "submission_lower_tail.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, deciles: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "EXPLORATORY LOWER-TAIL ROBUSTNESS ANALYSIS",
        "",
        "Tail membership is determined solely by best-comparator lifetime within",
        "each fixed topology/size/density/mode/handover stratum. Ties are broken",
        "deterministically by instance ID. CARE-DS outcomes are not used for selection.",
        "The hypothesis was formulated after the primary matrices were complete.",
        "",
        "SUMMARY",
        summary.to_string(index=False),
        "",
        "DECILES",
        deciles.to_string(index=False),
    ]
    (output_dir / "lower_tail_report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=Path, default=DEFAULT_UNIT)
    parser.add_argument("--radio", type=Path, default=DEFAULT_RADIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    args = parser.parse_args()

    definitions = [
        Study(
            name="unit",
            label="Unit cost, h/a in {0.25, 0.5}",
            path=args.unit,
            switch_costs=(0.25, 0.5),
            competitors=("D-EAA-ER", "DWG", "MS-RG-WT"),
        ),
        Study(
            name="radio",
            label="Radio, h/a_ref = 0.5",
            path=args.radio,
            switch_costs=(1e-4,),
            competitors=None,
        ),
    ]
    studies = [(study, prepare(study)) for study in definitions]
    summary, deciles = summarize(studies, samples=args.bootstrap_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "lower_tail_summary.csv", index=False)
    deciles.to_csv(args.output_dir / "lower_tail_deciles.csv", index=False)
    write_report(summary, deciles, args.output_dir)
    make_figure(deciles, args.figure_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
