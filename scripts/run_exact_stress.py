#!/usr/bin/env python3
"""Prespecified n=21--22 exact stress suite using the full-study heuristics."""
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsnlife.exact import capacitated_schedule_optimum
from wsnlife.experiment import derived_seed
from wsnlife.graph import generate_instance
from wsnlife.schedulers import run_algorithm


def main():
    rows = []
    master = 20260825
    # Fixed before evaluation: sparse, heterogeneous instances reduce the
    # prevalence of trivial optimum hits without outcome-based seed selection.
    designs = [
        (21, "rgg", 3), (21, "clustered", 3), (21, "er", 3),
        (21, "watts_strogatz", 4), (21, "barabasi_albert", 4),
        (22, "rgg", 4), (22, "clustered", 4), (22, "er", 4),
        (22, "watts_strogatz", 4), (22, "barabasi_albert", 4),
    ]
    algorithms = ("D-EAA-ER", "DWG", "MS-RG-WT", "CARE-DS")
    for mode in ("dominating", "connected"):
        for design_no, (n, topology, degree) in enumerate(designs):
            seed = derived_seed(master, "exact-stress", mode, design_no)
            graph = generate_instance(n, degree, seed, topology, 1, 6)
            start = time.perf_counter()
            optimum, sets, result = capacitated_schedule_optimum(
                graph.adjacency, graph.energies, mode=mode
            )
            exact_time = time.perf_counter() - start
            if optimum is None:
                print("exact unresolved", mode, design_no, result.message, flush=True)
                continue
            for algorithm in algorithms:
                heuristic = run_algorithm(
                    algorithm, graph.adjacency, graph.energies, mode=mode,
                    seed=derived_seed(seed, algorithm), random_starts=4,
                    refresh_interval=1, max_rounds=2000,
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
    output = ROOT / "results" / "exact_stress.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
