from __future__ import annotations

import numpy as np

from .graph import closed_neighbourhoods, is_feasible_set, neighbourhood_sums


def minimal_by_deletion(
    adjacency, allowed, priorities, mode="dominating", sink=0,
    required_vertices=None,
):
    """Produce an inclusion-minimal feasible set by low-priority-first deletion."""
    selected = set(allowed)
    required = (
        set(range(len(adjacency)))
        if required_vertices is None else set(required_vertices)
    )
    if not is_feasible_set(adjacency, selected, mode, sink, required):
        return frozenset()
    closed = closed_neighbourhoods(adjacency)
    domination_count = np.zeros(len(adjacency), dtype=np.int32)
    for v in selected:
        domination_count[list(closed[v])] += 1
    for v in sorted(selected, key=lambda x: (priorities[x], x)):
        if mode == "connected" and v == sink:
            continue
        affected = list(closed[v] & required)
        if any(domination_count[u] <= 1 for u in affected):
            continue
        trial = selected - {v}
        if mode != "connected" or is_feasible_set(
            adjacency, trial, mode, sink, required
        ):
            selected.remove(v)
            domination_count[affected] -= 1
    return frozenset(selected)


def constructive_greedy(
    adjacency,
    allowed,
    energy,
    mode="dominating",
    sink=0,
    energy_power=1.0,
    rng=None,
    rcl_alpha=0.0,
    required_vertices=None,
):
    """Residual-energy x white-degree greedy, with optional GRASP randomization."""
    n = len(adjacency)
    allowed = set(allowed)
    required = set(range(n)) if required_vertices is None else set(required_vertices)
    allowed.intersection_update(required)
    if not required:
        return frozenset()
    if mode == "connected" and sink not in allowed:
        return frozenset()
    closed = closed_neighbourhoods(adjacency)
    chosen = {sink} if mode == "connected" else set()
    chosen_mask = np.zeros(n, dtype=bool)
    if chosen:
        chosen_mask[list(chosen)] = True
    allowed_mask = np.zeros(n, dtype=bool)
    allowed_mask[list(allowed)] = True
    covered_mask = np.zeros(n, dtype=bool)
    frontier_mask = np.zeros(n, dtype=bool)
    for v in chosen:
        covered_mask[list(closed[v])] = True
        frontier_mask[list(adjacency[v])] = True
    energy_weight = np.maximum(np.asarray(energy, dtype=float), 1e-9) ** energy_power

    required_mask = np.zeros(n, dtype=bool)
    required_mask[list(required)] = True

    while not np.all(covered_mask[required_mask]):
        candidate_mask = allowed_mask & ~chosen_mask
        if mode == "connected" and chosen:
            candidate_mask &= frontier_mask
        candidates = np.flatnonzero(candidate_mask)
        if candidates.size == 0:
            return frozenset()
        gains = neighbourhood_sums(
            adjacency, ((~covered_mask) & required_mask).astype(float)
        )
        connector_gain = 0.05 if mode == "connected" else 0.0
        score_vector = (gains + connector_gain) * energy_weight
        candidate_scores = score_vector[candidates]
        hi, lo = float(np.max(candidate_scores)), float(np.min(candidate_scores))
        threshold = hi - rcl_alpha * (hi - lo)
        rcl = candidates[candidate_scores >= threshold - 1e-12]
        if rng is None or len(rcl) == 1:
            rcl_scores = score_vector[rcl]
            v = int(np.min(rcl[rcl_scores >= np.max(rcl_scores) - 1e-12]))
        else:
            v = int(rng.choice(rcl))
        chosen.add(v)
        chosen_mask[v] = True
        covered_mask[list(closed[v])] = True
        frontier_mask[list(adjacency[v])] = True

    # Remove low-energy redundant nodes first; preserve sink/connectivity if required.
    priorities = np.asarray(energy, dtype=float)
    return minimal_by_deletion(
        adjacency, chosen, priorities, mode, sink, required
    )


def gh_mwdds_plus(adjacency, energy):
    """Faithful implementation of GH-MWDDS+ (Balbal et al., 2021, Algorithm 1)."""
    n = len(adjacency)
    closed = closed_neighbourhoods(adjacency)
    remaining = set(range(n))
    done = set()
    family = []

    while not any(closed[v] <= done for v in range(n)):
        chosen = set()
        covered = set()
        while len(covered) < n:
            candidates = remaining - chosen
            if not candidates:
                return family
            def score(v):
                return float(energy[v]) * len(closed[v] - covered)
            v = min(candidates, key=lambda u: (-score(u), u))
            chosen.add(v)
            remaining.remove(v)
            covered.update(closed[v])

        # Optional Reduce step: return redundant vertices to Vrem.
        changed = True
        while changed:
            changed = False
            for v in sorted(chosen, key=lambda u: (energy[u], u)):
                trial = chosen - {v}
                if is_feasible_set(adjacency, trial):
                    chosen = trial
                    remaining.add(v)
                    changed = True
                    break
        family.append(frozenset(chosen))
        done.update(chosen)
    return family


def eaa_disjoint_family(adjacency, energy):
    """Faithful port of the uploaded paper_disjoint_EAA.m."""
    remaining = set(range(len(adjacency)))
    family = []
    while remaining and is_feasible_set(adjacency, remaining):
        # The MATLAB algorithm repeatedly deletes the first removable vertex in
        # ascending-energy order. Removability is monotone under deletions, so
        # the same decisions are obtained by the incremental one-pass routine.
        selected = minimal_by_deletion(adjacency, remaining, energy)
        if not selected:
            break
        family.append(selected)
        remaining.difference_update(selected)
    return family
