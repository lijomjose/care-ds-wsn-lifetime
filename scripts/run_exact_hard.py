#!/usr/bin/env python3
"""Prespecified harder exact instances for the no-handover scheduling model."""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsnlife.exact import capacitated_schedule_optimum
from wsnlife.experiment import derived_seed
from wsnlife.graph import generate_instance
from wsnlife.schedulers import run_algorithm


def main():
    rows = []
    master = 20260821
    designs = [
        (16, "rgg", 4), (16, "er", 4), (16, "watts_strogatz", 4),
        (18, "rgg", 5), (18, "er", 5), (18, "watts_strogatz", 4),
        (20, "rgg", 5), (20, "er", 5), (20, "watts_strogatz", 4),
    ]
    for mode in ("dominating", "connected"):
        for design_no, (n, topology, degree) in enumerate(designs):
            seed = derived_seed(master, mode, n, topology, degree)
            graph = generate_instance(n, degree, seed, topology, 1, 8)
            start = time.perf_counter()
            optimum, sets, result = capacitated_schedule_optimum(
                graph.adjacency, graph.energies, mode=mode
            )
            exact_time = time.perf_counter() - start
            if optimum is None:
                print("exact unresolved", mode, n, topology, result.message, flush=True)
                continue
            for algorithm in ("D-EAA", "DWG", "MS-RG-TM", "CARE-DS"):
                heuristic = run_algorithm(
                    algorithm, graph.adjacency, graph.energies, mode=mode,
                    seed=derived_seed(seed, algorithm), random_starts=20,
                    refresh_interval=1, max_rounds=5000,
                )
                rows.append({
                    "mode": mode, "design": design_no, "topology": topology,
                    "n": n, "target_degree": degree, "seed": seed,
                    "algorithm": algorithm, "lifetime": heuristic.lifetime,
                    "optimum": optimum,
                    "ratio_to_optimum": heuristic.lifetime / optimum,
                    "optimal": heuristic.lifetime == optimum,
                    "candidate_sets": len(sets),
                    "heuristic_runtime_s": heuristic.runtime_s,
                    "exact_runtime_s": exact_time,
                    "milp_status": result.status,
                    "valid": heuristic.valid,
                })
            print(
                mode, n, topology, "opt", optimum, "minimal sets", len(sets),
                "exact_s", round(exact_time, 2), flush=True,
            )
    output = ROOT / "results" / "exact_hard.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
