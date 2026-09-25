from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .graph import generate_instance
from .schedulers import run_algorithm


def derived_seed(master_seed, *parts):
    payload = "|".join(map(str, (master_seed, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32)


STATIC_ALGORITHMS = {
    "EAA-static", "GH-MWDDS+", "PBIG-R", "MC-CMSA-R", "FSS-2026-R",
    "EAAS-S4C-MAB-R",
}


def failure_events(graph, scenario, seed, fraction=0.04, rounds=(25, 50)):
    if scenario == "none":
        return {}
    rng = np.random.default_rng(seed)
    available = set(range(graph.n)) - {0}
    count = max(1, int(round(graph.n * fraction)))
    events = {}
    for event_round in rounds:
        take = min(count, len(available))
        if not take:
            break
        if scenario == "random":
            selected = set(map(int, rng.choice(sorted(available), size=take, replace=False)))
        elif scenario == "targeted":
            # Attack the least-redundant coverage neighbourhoods rather than hubs.
            # These vertices have the smallest local energy reserve and are the
            # failure locations most relevant to the CARE design principle.
            def criticality(v):
                closed = {v, *graph.adjacency[v]}
                return (sum(graph.energies[u] for u in closed), len(closed), v)
            selected = set(sorted(available, key=criticality)[:take])
        else:
            raise ValueError(f"unknown failure scenario: {scenario}")
        events[int(event_round)] = selected
        available.difference_update(selected)
    return events


def _run_instance(task):
    config, topology, n, degree, run = task
    instance_seed = derived_seed(config["master_seed"], topology, n, degree, run)
    graph = generate_instance(
        n, degree, instance_seed, topology,
        config["energy_low"], config["energy_high"],
    )
    rows = []
    scenarios = config.get("failure_scenarios", ["none"])
    for mode, switch_cost, scenario, algorithm in itertools.product(
        config["modes"], config["switch_costs"], scenarios, config["algorithms"]
    ):
        if algorithm in STATIC_ALGORITHMS and mode != "dominating":
            continue
        if algorithm in STATIC_ALGORITHMS and switch_cost != config["switch_costs"][0]:
            continue
        failure_seed = derived_seed(instance_seed, scenario, "failures")
        events = failure_events(
            graph, scenario, failure_seed,
            config.get("failure_fraction", 0.04),
            tuple(config.get("failure_rounds", [25, 50])),
        )
        # Equal-search algorithms share their stochastic candidate stream.
        if algorithm in {
            "CARE-DS", "DWG-TM", "D-EAA-TM", "D-EAA-ER", "MS-RG-TM",
            "MS-RG-WT", "MS-RG-WT-DW", "CARE-noreserve", "CARE-noswitch",
        }:
            algorithm_seed = derived_seed(instance_seed, mode, switch_cost, scenario, "equal-portfolio")
        else:
            algorithm_seed = derived_seed(instance_seed, mode, switch_cost, scenario, algorithm)
        static_budget = config.get("static_time_budget_s", 0.25)
        static_budget = min(
            config.get("static_time_budget_cap_s", static_budget),
            static_budget + config.get("static_time_budget_per_node_s", 0.0) * n,
        )
        result = run_algorithm(
            algorithm,
            graph.adjacency,
            graph.energies,
            mode=mode,
            active_cost=config["active_cost"],
            switch_cost=switch_cost,
            random_starts=config["random_starts"],
            seed=algorithm_seed,
            max_rounds=config["max_rounds"],
            positions=graph.positions,
            energy_model=config.get("energy_model", "unit"),
            radio_parameters=config.get("radio_parameters"),
            failure_events=events,
            refresh_interval=config.get("refresh_interval", 1),
            static_time_budget_s=static_budget,
            dwell_scale=config.get("dwell_scale", 100.0),
            dwell_rounds_override=config.get("dwell_rounds_override"),
        )
        rows.append({
            "instance_id": f"{topology}-n{n}-d{degree}-r{run:03d}",
            "instance_seed": instance_seed,
            "topology": topology,
            "n": n,
            "target_degree": degree,
            "actual_degree": graph.average_degree,
            "run": run,
            "mode": mode,
            "switch_cost": switch_cost,
            "failure_scenario": scenario,
            "energy_model": config.get("energy_model", "unit"),
            "algorithm": algorithm,
            "lifetime": result.lifetime,
            "runtime_s": result.runtime_s,
            "switches": result.switches,
            "activations": result.activations,
            "mean_set_size": result.mean_set_size,
            "valid": result.valid,
        })
    return rows


def run_config(config_path, output_path, workers=None, resume=True):
    config = json.loads(Path(config_path).read_text())
    dimensions = list(itertools.product(
        config["topologies"], config["node_sizes"], config["target_degrees"], range(config["runs"])
    ))
    total_instances = (
        len(config["topologies"]) * len(config["node_sizes"])
        * len(config["target_degrees"]) * config["runs"]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if resume and output.exists() and output.stat().st_size:
        existing = pd.read_csv(output)
        completed = set(existing["instance_id"].unique())
    pending = []
    for topology, n, degree, run in dimensions:
        instance_id = f"{topology}-n{n}-d{degree}-r{run:03d}"
        if instance_id not in completed:
            pending.append((config, topology, n, degree, run))
    workers = workers or int(config.get("workers", max(1, min(8, (os.cpu_count() or 2) - 1))))
    first_write = not output.exists() or not output.stat().st_size
    if workers == 1:
        iterator = enumerate(map(_run_instance, pending), 1)
        for done, rows in iterator:
            pd.DataFrame(rows).to_csv(output, mode="a", header=first_write, index=False)
            first_write = False
            print(f"instance {len(completed)+done}/{total_instances}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_instance, task): task for task in pending}
            for done, future in enumerate(as_completed(futures), 1):
                rows = future.result()
                pd.DataFrame(rows).to_csv(output, mode="a", header=first_write, index=False)
                first_write = False
                task = futures[future]
                print(
                    f"instance {len(completed)+done}/{total_instances}: "
                    f"{task[1]}, n={task[2]}, d={task[3]}, run={task[4]}",
                    flush=True,
                )
    return pd.read_csv(output)
