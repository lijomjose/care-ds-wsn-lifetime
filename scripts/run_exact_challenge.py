#!/usr/bin/env python3
"""Adversarial n=22 exact challenge cases for dynamic domination.

The fixed cases below were selected from a separate calibration pool by low
heuristic-to-closed-neighbourhood-bound utilization.  They are therefore a
transparent stress set, not an unbiased sample and not part of confirmatory
statistical tests.
"""
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


CASES = (
    ("barabasi_albert", 3, 2792999952),
    ("watts_strogatz", 6, 260906790),
    ("barabasi_albert", 3, 1378229411),
    ("watts_strogatz", 6, 3204751680),
    ("watts_strogatz", 2, 3946487045),
    ("watts_strogatz", 6, 1799537453),
    ("barabasi_albert", 5, 3844033684),
    ("watts_strogatz", 6, 3086486433),
    ("barabasi_albert", 6, 2778765328),
    ("er", 6, 2073815662),
    ("er", 6, 1170379085),
    ("barabasi_albert", 5, 2076869113),
)


def main():
    algorithms = ("D-EAA-ER", "DWG", "MS-RG-WT", "CARE-DS")
    rows = []
    for case_no, (topology, degree, seed) in enumerate(CASES):
        graph = generate_instance(22, degree, seed, topology, 1, 8)
        start = time.perf_counter()
        optimum, sets, result = capacitated_schedule_optimum(
            graph.adjacency, graph.energies, mode="dominating"
        )
        exact_time = time.perf_counter() - start
        if optimum is None:
            print("exact unresolved", case_no, result.message, flush=True)
            continue
        bound = min(
            sum(graph.energies[u] for u in ({v} | set(graph.adjacency[v])))
            for v in range(graph.n)
        )
        for algorithm in algorithms:
            heuristic = run_algorithm(
                algorithm, graph.adjacency, graph.energies,
                mode="dominating", seed=derived_seed(seed, algorithm),
                random_starts=4, refresh_interval=1, max_rounds=2000,
            )
            rows.append({
                "mode": "dominating", "design": case_no,
                "topology": topology, "n": graph.n,
                "target_degree": degree, "seed": seed,
                "algorithm": algorithm, "lifetime": heuristic.lifetime,
                "optimum": optimum,
                "ratio_to_optimum": heuristic.lifetime / optimum,
                "optimal": heuristic.lifetime == optimum,
                "upper_bound": bound, "candidate_sets": len(sets),
                "heuristic_runtime_s": heuristic.runtime_s,
                "exact_runtime_s": exact_time, "milp_status": result.status,
                "valid": heuristic.valid,
                "selection_policy": "low calibration heuristic/bound utilization",
            })
        print(
            case_no, topology, "degree", degree, "opt", optimum,
            "bound", int(bound), "minimal sets", len(sets),
            "exact_s", round(exact_time, 2), flush=True,
        )
    output = ROOT / "results" / "exact_challenge.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
