#!/usr/bin/env python3
"""Compute the closed-neighbourhood budget bound and empirical utilization."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsnlife.graph import closed_neighbourhoods, generate_instance


def bounds_for_instance(task):
    instance_id, topology, n, degree, seed, low, high, active_cost = task
    graph = generate_instance(int(n), float(degree), int(seed), topology, low, high)
    capacities = np.floor(graph.energies / active_cost)
    closed = closed_neighbourhoods(graph.adjacency)
    ordinary = min(float(capacities[list(neighbourhood)].sum()) for neighbourhood in closed)
    # The sink is free in connected mode. Neighbourhood constraints already
    # covered by the sink do not yield a finite energy bound and are skipped.
    constrained = [
        float(capacities[list(neighbourhood)].sum())
        for neighbourhood in closed if 0 not in neighbourhood
    ]
    connected = min(constrained) if constrained else np.inf
    return instance_id, ordinary, connected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results/submission_event_raw.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/submission_event.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data = pd.read_csv(args.input)
    instances = data[[
        "instance_id", "topology", "n", "target_degree", "instance_seed"
    ]].drop_duplicates()
    tasks = [
        (*row, config["energy_low"], config["energy_high"], config["active_cost"])
        for row in instances.itertuples(index=False, name=None)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        bounds = list(executor.map(bounds_for_instance, tasks, chunksize=8))
    bound_table = pd.DataFrame(bounds, columns=["instance_id", "bound_dominating", "bound_connected"])
    merged = data.merge(bound_table, on="instance_id", validate="many_to_one")
    merged["upper_bound"] = np.where(
        merged["mode"].eq("connected"), merged.bound_connected, merged.bound_dominating
    )
    finite = np.isfinite(merged.upper_bound) & (merged.upper_bound > 0)
    merged["bound_utilization"] = np.where(
        finite, merged.lifetime / merged.upper_bound, np.nan
    )
    merged["finite_bound"] = finite
    columns = [
        "instance_id", "topology", "n", "target_degree", "mode", "switch_cost",
        "algorithm", "lifetime", "upper_bound", "finite_bound",
        "bound_utilization",
    ]
    merged[columns].to_csv(ROOT / "results/upper_bound_gaps.csv", index=False)
    summary = merged.groupby(["mode", "switch_cost", "algorithm"], as_index=False).agg(
        instances=("bound_utilization", "size"),
        finite_bound_instances=("bound_utilization", "count"),
        mean_utilization=("bound_utilization", "mean"),
        median_utilization=("bound_utilization", "median"),
        mean_finite_upper_bound=(
            "upper_bound", lambda values: values[np.isfinite(values)].mean()
        ),
    )
    summary.to_csv(ROOT / "results/upper_bound_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
