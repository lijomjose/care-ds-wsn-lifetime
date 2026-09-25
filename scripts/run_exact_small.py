#!/usr/bin/env python3
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
    master = 20260818
    for mode in ("dominating", "connected"):
        for n in (10, 12, 14):
            for run in range(5):
                seed = derived_seed(master, "exact", mode, n, run)
                graph = generate_instance(n, 5, seed, "rgg", 2, 6)
                start = time.perf_counter()
                optimum, sets, result = capacitated_schedule_optimum(
                    graph.adjacency, graph.energies, mode=mode
                )
                exact_time = time.perf_counter() - start
                if optimum is None:
                    raise RuntimeError(result.message)
                for algorithm in ("D-EAA", "DWG", "MS-RG", "CARE-DS"):
                    heuristic = run_algorithm(
                        algorithm, graph.adjacency, graph.energies, mode=mode,
                        seed=derived_seed(seed, algorithm), random_starts=12,
                    )
                    rows.append({
                        "mode": mode,
                        "n": n,
                        "run": run,
                        "seed": seed,
                        "algorithm": algorithm,
                        "lifetime": heuristic.lifetime,
                        "optimum": optimum,
                        "ratio_to_optimum": heuristic.lifetime / optimum,
                        "candidate_sets": len(sets),
                        "heuristic_runtime_s": heuristic.runtime_s,
                        "exact_runtime_s": exact_time,
                        "valid": heuristic.valid,
                    })
                print(mode, n, run, "opt", optimum, "sets", len(sets), flush=True)
    output = ROOT / "results" / "exact_small.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

