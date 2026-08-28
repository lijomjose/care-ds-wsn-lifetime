from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .graph import is_feasible_set


def enumerate_minimal_feasible_sets(adjacency, mode="dominating", sink=0, max_n=22):
    """Enumerate inclusion-minimal (connected) dominating sets for small graphs."""
    n = len(adjacency)
    if n > max_n:
        raise ValueError(f"exact enumeration is limited to n <= {max_n}")
    sets = []
    mandatory = {sink} if mode == "connected" else set()
    optional = [v for v in range(n) if v not in mandatory]
    for r in range(len(optional) + 1):
        for subset in itertools.combinations(optional, r):
            chosen = mandatory | set(subset)
            if not is_feasible_set(adjacency, chosen, mode, sink):
                continue
            if any(existing <= chosen for existing in sets):
                continue
            sets.append(frozenset(chosen))
    return sets


def capacitated_schedule_optimum(adjacency, budgets, mode="dominating", sink=0):
    """Exact no-handover integer schedule over all minimal feasible sets."""
    sets = enumerate_minimal_feasible_sets(adjacency, mode, sink)
    n, m = len(adjacency), len(sets)
    incidence = np.zeros((n, m), dtype=float)
    for j, ds in enumerate(sets):
        incidence[list(ds), j] = 1.0
    rhs = np.asarray(budgets, dtype=float).copy()
    if mode == "connected":
        incidence[sink, :] = 0.0
        rhs[sink] = 0.0
    result = milp(
        c=-np.ones(m),
        integrality=np.ones(m),
        bounds=Bounds(np.zeros(m), np.full(m, np.inf)),
        constraints=LinearConstraint(incidence, -np.inf, rhs),
        options={"time_limit": 120.0},
    )
    optimum = int(round(-result.fun)) if result.fun is not None else None
    return optimum, sets, result
