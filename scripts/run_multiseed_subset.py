#!/usr/bin/env python3
"""Run the predeclared repeated-algorithm-seed robustness subset.

Graph instances are fixed independently of algorithm outcomes. Dynamic CARE-DS
and its shared-portfolio comparator receive paired stochastic streams. Modern
static reproductions are repeated on their native unweighted objective.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsnlife.experiment import derived_seed
from wsnlife.graph import generate_instance
from wsnlife.schedulers import run_algorithm


MASTER_SEED = 20260901
TOPOLOGIES = ("rgg", "clustered", "er", "watts_strogatz", "barabasi_albert")
DYNAMIC_SIZES = (100, 500)
STATIC_SIZES = (100, 500, 1000)
DEGREES = (15, 35)
GRAPH_REPEATS = range(2)
ALGORITHM_REPEATS = range(10)


def _dynamic_task(task):
    topology, n, degree, graph_repeat, mode, switch_cost, algorithm_repeat = task
    instance_seed = derived_seed(
        MASTER_SEED, "dynamic", topology, n, degree, graph_repeat
    )
    graph = generate_instance(n, degree, instance_seed, topology, 20, 100)
    algorithm_seed = derived_seed(
        instance_seed, mode, switch_cost, algorithm_repeat, "paired-search"
    )
    rows = []
    for algorithm in ("CARE-DS", "MS-RG-WT"):
        result = run_algorithm(
            algorithm, graph.adjacency, graph.energies, mode=mode,
            active_cost=1.0, switch_cost=switch_cost, random_starts=4,
            seed=algorithm_seed, max_rounds=5000, positions=graph.positions,
            energy_model="unit", refresh_interval=100, dwell_scale=100.0,
        )
        rows.append({
            "study": "dynamic", "instance_id": (
                f"{topology}-n{n}-d{degree}-g{graph_repeat:02d}"
            ), "instance_seed": instance_seed, "topology": topology, "n": n,
            "target_degree": degree, "graph_repeat": graph_repeat,
            "mode": mode, "switch_cost": switch_cost,
            "algorithm_repeat": algorithm_repeat, "algorithm_seed": algorithm_seed,
            "algorithm": algorithm, "lifetime": result.lifetime,
            "runtime_s": result.runtime_s, "valid": result.valid,
        })
    return rows


def _static_task(task):
    topology, n, degree, graph_repeat, algorithm_repeat = task
    instance_seed = derived_seed(
        MASTER_SEED, "static", topology, n, degree, graph_repeat
    )
    graph = generate_instance(n, degree, instance_seed, topology, 1, 1)
    rows = []
    for algorithm in ("PBIG-R", "MC-CMSA-R", "FSS-2026-R"):
        algorithm_seed = derived_seed(instance_seed, algorithm, algorithm_repeat)
        result = run_algorithm(
            algorithm, graph.adjacency, np.ones(n), active_cost=1.0,
            seed=algorithm_seed, static_time_budget_s=0.5,
            energy_model="unit", max_rounds=5000,
        )
        rows.append({
            "study": "static-unweighted", "instance_id": (
                f"{topology}-n{n}-d{degree}-g{graph_repeat:02d}"
            ), "instance_seed": instance_seed, "topology": topology, "n": n,
            "target_degree": degree, "graph_repeat": graph_repeat,
            "mode": "dominating", "switch_cost": 0.0,
            "algorithm_repeat": algorithm_repeat, "algorithm_seed": algorithm_seed,
            "algorithm": algorithm, "lifetime": result.lifetime,
            "runtime_s": result.runtime_s, "valid": result.valid,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "revision" / "multiseed_subset_raw.csv"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    dynamic_tasks = [
        (topology, n, degree, graph_repeat, mode, switch_cost, algorithm_repeat)
        for topology in TOPOLOGIES for n in DYNAMIC_SIZES for degree in DEGREES
        for graph_repeat in GRAPH_REPEATS for mode in ("dominating", "connected")
        for switch_cost in (0.25, 0.5) for algorithm_repeat in ALGORITHM_REPEATS
    ]
    static_tasks = [
        (topology, n, degree, graph_repeat, algorithm_repeat)
        for topology in ("er", "watts_strogatz", "barabasi_albert")
        for n in STATIC_SIZES for degree in DEGREES for graph_repeat in GRAPH_REPEATS
        for algorithm_repeat in ALGORITHM_REPEATS
    ]
    jobs = [("dynamic", task) for task in dynamic_tasks] + [
        ("static", task) for task in static_tasks
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_dynamic_task if study == "dynamic" else _static_task, task):
            (study, task) for study, task in jobs
        }
        for done, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if done % 50 == 0 or done == len(jobs):
                print(f"completed task {done}/{len(jobs)}", flush=True)
    frame = pd.DataFrame(rows).sort_values([
        "study", "instance_id", "mode", "switch_cost", "algorithm_repeat", "algorithm"
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
