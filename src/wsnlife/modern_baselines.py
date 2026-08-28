from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .domination import gh_mwdds_plus, minimal_by_deletion
from .graph import closed_neighbourhoods, is_feasible_set


Family = tuple[frozenset[int], ...]


def _coverage(adjacency, vertices):
    covered = set(vertices)
    for v in vertices:
        covered.update(adjacency[v])
    return covered


def _family_key(adjacency, family, energy=None):
    used = set().union(*family) if family else set()
    leftover_coverage = len(_coverage(adjacency, set(range(len(adjacency))) - used))
    if energy is None:
        return (len(family), leftover_coverage, -sum(map(len, family)))
    lifetime = sum(float(np.min(np.asarray(energy)[list(ds)])) for ds in family)
    return (lifetime, len(family), leftover_coverage, -sum(map(len, family)))


def _generate_ds(adjacency, available, rng, energy=None, determinism=1.0, initial=None, score_mode=0):
    n = len(adjacency)
    closed = closed_neighbourhoods(adjacency)
    chosen = set(initial or ())
    covered = _coverage(adjacency, chosen)
    while len(covered) < n:
        candidates = set(available) - chosen
        if not candidates:
            return frozenset()
        scored = []
        for v in candidates:
            gain = len(closed[v] - covered)
            degree = len(adjacency[v]) + 1
            e = 1.0 if energy is None else max(float(energy[v]), 1e-12)
            if score_mode == 0:
                score = gain
            elif score_mode == 1:
                score = gain / degree
            elif score_mode == 2:
                score = gain * degree
            elif score_mode == 3:
                score = gain * e
            elif score_mode == 4:
                score = gain * math.sqrt(e)
            else:
                score = gain * e / degree
            scored.append((score, v))
        scored.sort(reverse=True)
        rcl_size = max(1, int(math.ceil((1.0 - determinism) * len(scored))))
        threshold = scored[min(rcl_size - 1, len(scored) - 1)][0]
        rcl = [v for score, v in scored if score >= threshold - 1e-12]
        chosen.add(int(rng.choice(rcl)))
        covered = _coverage(adjacency, chosen)
    priorities = np.ones(n) if energy is None else np.asarray(energy, dtype=float)
    # Random jitter reproduces the papers' randomized redundant-node deletion.
    jitter = rng.random(n) * 1e-8
    return minimal_by_deletion(adjacency, chosen, priorities + jitter)


def randomized_greedy_family(adjacency, rng, energy=None, determinism=1.0, fixed=None, score_mode=0):
    available = set(range(len(adjacency)))
    family = []
    cores = list(fixed or ())
    rng.shuffle(cores)
    for core in cores:
        core = set(core)
        pool = available | core
        if not is_feasible_set(adjacency, pool):
            continue
        ds = _generate_ds(adjacency, pool, rng, energy, determinism, core, score_mode)
        if ds and ds <= available:
            family.append(ds)
            available.difference_update(ds)
    while is_feasible_set(adjacency, available):
        ds = _generate_ds(adjacency, available, rng, energy, determinism, score_mode=score_mode)
        if not ds:
            break
        family.append(ds)
        available.difference_update(ds)
    return tuple(family)


def _canonical_family(family):
    return tuple(sorted((tuple(sorted(ds)) for ds in family)))


def _prune_and_complete(adjacency, family, rng):
    """Prune changed sets and greedily exploit the resulting leftovers."""
    priorities = np.ones(len(adjacency)) + rng.random(len(adjacency)) * 1e-8
    pruned = [
        set(minimal_by_deletion(adjacency, ds, priorities)) for ds in family
    ]
    pruned = [ds for ds in pruned if ds]
    used = set().union(*pruned) if pruned else set()
    available = set(range(len(adjacency))) - used
    extension = randomized_greedy_family(
        adjacency, rng, None, 1.0, score_mode=0
    ) if available == set(range(len(adjacency))) else ()
    if available != set(range(len(adjacency))):
        while is_feasible_set(adjacency, available):
            ds = _generate_ds(adjacency, available, rng, None, 1.0, score_mode=0)
            if not ds:
                break
            pruned.append(set(ds))
            available.difference_update(ds)
    else:
        pruned.extend(map(set, extension))
    return tuple(frozenset(ds) for ds in pruned)


def _swap_local_search(adjacency, family, rng, deadline=None, max_checks=300):
    """Time-bounded implementation of the paper's improving swap search."""
    incumbent = tuple(frozenset(ds) for ds in family)
    checks = 0
    improved = True
    while improved and (deadline is None or time.perf_counter() < deadline):
        improved = False
        family_list = [set(ds) for ds in incumbent]
        used = set().union(*family_list) if family_list else set()
        leftover = list(set(range(len(adjacency))) - used)
        rng.shuffle(leftover)
        order = list(range(len(family_list)))
        rng.shuffle(order)
        for i in order:
            members = list(family_list[i])
            rng.shuffle(members)
            for u in members:
                for v in leftover:
                    checks += 1
                    trial_ds = (family_list[i] - {u}) | {v}
                    if not is_feasible_set(adjacency, trial_ds):
                        if checks >= max_checks:
                            return incumbent
                        continue
                    trial_family = [set(ds) for ds in family_list]
                    trial_family[i] = trial_ds
                    candidate = _prune_and_complete(adjacency, trial_family, rng)
                    if _family_key(adjacency, candidate) > _family_key(adjacency, incumbent):
                        incumbent = candidate
                        improved = True
                        break
                    if checks >= max_checks:
                        return incumbent
                if improved:
                    break
            if improved:
                break
    return incumbent


def _extended_local_search(adjacency, family, rng, deadline, max_checks=700):
    """Swap plus the 1-in-2-out neighbourhood introduced by FSS 2026."""
    current = _swap_local_search(adjacency, family, rng, deadline, max_checks // 2)
    best = current
    seen = {_canonical_family(current)}
    checks = 0
    while time.perf_counter() < deadline and checks < max_checks:
        family_list = [set(ds) for ds in current]
        used = set().union(*family_list) if family_list else set()
        leftover = list(set(range(len(adjacency))) - used)
        rng.shuffle(leftover)
        order = [i for i, ds in enumerate(family_list) if len(ds) >= 2]
        rng.shuffle(order)
        moved = False
        for i in order:
            members = list(family_list[i])
            rng.shuffle(members)
            pairs = [(members[a], members[b])
                     for a in range(len(members))
                     for b in range(a + 1, len(members))]
            rng.shuffle(pairs)
            for u, v in pairs:
                for w in leftover:
                    checks += 1
                    trial_ds = (family_list[i] - {u, v}) | {w}
                    if not is_feasible_set(adjacency, trial_ds):
                        if checks >= max_checks or time.perf_counter() >= deadline:
                            break
                        continue
                    trial_family = [set(ds) for ds in family_list]
                    trial_family[i] = trial_ds
                    candidate = _prune_and_complete(adjacency, trial_family, rng)
                    key = _canonical_family(candidate)
                    if key not in seen:
                        seen.add(key)
                        current = _swap_local_search(
                            adjacency, candidate, rng, deadline,
                            max(40, max_checks - checks),
                        )
                        if _family_key(adjacency, current) > _family_key(adjacency, best):
                            best = current
                        moved = True
                        break
                if moved or checks >= max_checks or time.perf_counter() >= deadline:
                    break
            if moved or checks >= max_checks or time.perf_counter() >= deadline:
                break
        if not moved:
            break
    return best


def pbig_family(adjacency, energy, seed=0, time_budget_s=0.25, population_size=12):
    """Paper-based PBIG reproduction with adaptive destruction and determinism."""
    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + max(time_budget_s, 0.01)
    gh_seed = tuple(gh_mwdds_plus(adjacency, energy))
    population = [[gh_seed, 0.99, 0.22]]
    for _ in range(population_size - 1):
        if time.perf_counter() >= deadline:
            break
        det = float(rng.uniform(0.56, 0.99))
        fam = randomized_greedy_family(adjacency, rng, energy, det, score_mode=3)
        population.append([fam, det, float(rng.uniform(0.22, 0.44))])
    best = max((x[0] for x in population), key=lambda f: _family_key(adjacency, f, energy))
    no_improve = 0
    while time.perf_counter() < deadline:
        new = []
        for family, det, destruction in population:
            keep = [ds for ds in family if rng.random() >= destruction]
            fixed = tuple(keep)
            candidate = randomized_greedy_family(adjacency, rng, energy, det, fixed, 3)
            if _family_key(adjacency, candidate, energy) > _family_key(adjacency, family, energy):
                new.append([candidate, det, destruction])
            else:
                det = det - 0.1 if det - 0.1 >= 0.56 else 0.99
                destruction += (0.44 - 0.22) / 9.0
                if destruction > 0.44:
                    destruction = 0.22
                new.append([candidate, det, destruction])
            if time.perf_counter() >= deadline:
                break
        combined = population + new
        combined.sort(key=lambda x: _family_key(adjacency, x[0], energy), reverse=True)
        population = combined[:population_size]
        incumbent = population[0][0]
        if _family_key(adjacency, incumbent, energy) > _family_key(adjacency, best, energy):
            best, no_improve = incumbent, 0
        else:
            no_improve += 1
        if no_improve >= 20:
            population = [population[0]]
            while len(population) < population_size:
                if time.perf_counter() >= deadline:
                    break
                det = float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
                population.append([
                    randomized_greedy_family(adjacency, rng, energy, det, score_mode=3),
                    det, 0.22,
                ])
            no_improve = 0
    return tuple(best)


def fss_family(adjacency, seed=0, time_budget_s=0.25, archive_size=30):
    """Paper-based 2026 FSS reproduction with both published neighborhoods."""
    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + max(time_budget_s, 0.01)
    archive = []
    initial = min(30, archive_size)
    for _ in range(initial):
        fam = randomized_greedy_family(adjacency, rng, None, 1.0)
        archive.append(_extended_local_search(adjacency, fam, rng, deadline, 180))
        if time.perf_counter() >= deadline:
            break
    portion_index = 1
    stagnation = 0
    best = max(archive, key=lambda f: _family_key(adjacency, f))
    while time.perf_counter() < deadline:
        archive.sort(key=lambda f: _family_key(adjacency, f), reverse=True)
        elite = archive[: min(archive_size, len(archive))]
        base = elite[int(rng.integers(len(elite)))]
        used = [(i, v) for i, ds in enumerate(base) for v in ds]
        free_min = min(15, max(1, len(used) // 4))
        target = min(int(len(used) * (1.0 - 1.0 / (2**portion_index))), len(used) - free_min)
        sample = [elite[int(rng.integers(len(elite)))] for _ in range(min(8, len(elite)))]
        scores = []
        for i, v in used:
            score = 0.0
            for other in sample:
                containing = next((ds for ds in other if v in ds), frozenset())
                if containing:
                    score += len(set(base[i]) & set(containing)) / max(len(base[i]), 1)
            scores.append((score + float(rng.random()) * 1e-9, i, v))
        scores.sort(reverse=True)
        cores_by_index = {}
        for _, i, v in scores[: max(target, 0)]:
            cores_by_index.setdefault(i, set()).add(v)
        fixed = tuple(frozenset(x) for x in cores_by_index.values() if x)
        candidate = randomized_greedy_family(adjacency, rng, None, 1.0, fixed)
        candidate = _extended_local_search(adjacency, candidate, rng, deadline, 260)
        archive.append(candidate)
        archive.sort(key=lambda f: _family_key(adjacency, f), reverse=True)
        archive = archive[:archive_size]
        if _family_key(adjacency, archive[0]) > _family_key(adjacency, best):
            best, stagnation = archive[0], 0
        else:
            stagnation += 1
        if stagnation >= 10:
            portion_index = 1 if portion_index >= 4 else portion_index + 1
            stagnation = 0
    return tuple(best)


def mc_cmsa_family(adjacency, seed=0, time_budget_s=0.25):
    """Validated CMSA-style reconstruction using six constructors and reduced MILP.

    Candidate dominating sets are the reduced sub-instance components. This is
    intentionally named a reconstruction rather than author code.
    """
    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + max(time_budget_s, 0.01)
    pool = set()
    best = tuple()
    ages = {}
    while time.perf_counter() < deadline:
        for score_mode in range(6):
            if time.perf_counter() >= deadline:
                break
            fam = randomized_greedy_family(
                adjacency, rng, None, float(rng.uniform(0.75, 1.0)), score_mode=score_mode
            )
            if _family_key(adjacency, fam) > _family_key(adjacency, best):
                best = fam
            for ds in fam:
                pool.add(ds)
                ages[ds] = 0
        if time.perf_counter() >= deadline:
            break
        candidates = list(pool)
        if not candidates:
            break
        n, m = len(adjacency), len(candidates)
        incidence = np.zeros((n, m), dtype=float)
        for j, ds in enumerate(candidates):
            incidence[list(ds), j] = 1.0
        remaining = max(0.01, min(0.05, deadline - time.perf_counter()))
        result = milp(
            c=-np.ones(m), integrality=np.ones(m),
            bounds=Bounds(np.zeros(m), np.ones(m)),
            constraints=LinearConstraint(incidence, -np.inf, np.ones(n)),
            options={"time_limit": remaining},
        )
        if result.x is not None:
            chosen = tuple(candidates[j] for j, x in enumerate(result.x) if x > 0.5)
            if _family_key(adjacency, chosen) > _family_key(adjacency, best):
                best = chosen
            chosen_set = set(chosen)
            for ds in list(pool):
                ages[ds] = 0 if ds in chosen_set else ages.get(ds, 0) + 1
                if ages[ds] >= 4:
                    pool.remove(ds)
                    ages.pop(ds, None)
    return tuple(best)


def eaas_skyline_family(adjacency, energy, seed=0, candidates_per_round=12,
                        size_penalty=0.5):
    """EAAS formation from Salim et al. (2026), Algorithms 1--2.

    The unused stochastic arguments remain in the signature for backwards
    compatibility. EAAS itself is the disclosed energy-ordered progressive
    deletion followed by BNL skyline filtering and composite selection.
    """
    remaining = set(range(len(adjacency)))
    family = []
    closed = closed_neighbourhoods(adjacency)
    energy_array = np.asarray(energy, dtype=float)
    while is_feasible_set(adjacency, remaining):
        current = set(remaining)
        candidates = [frozenset(current)]
        # Exact incremental form of the paper's progressive deletion. Removing
        # v changes coverage only in N[v], so a full domination traversal for
        # every tentative deletion is unnecessary.
        coverage_count = np.zeros(len(adjacency), dtype=np.int32)
        for v in current:
            coverage_count[list(closed[v])] += 1
        for v in sorted(remaining, key=lambda u: (float(energy[u]), u)):
            affected = list(closed[v])
            if len(current) > 1 and np.all(coverage_count[affected] >= 2):
                current.remove(v)
                coverage_count[affected] -= 1
                candidates.append(frozenset(current))
        skyline = []
        for ds in candidates:
            life = float(np.min(energy_array[list(ds)]))
            size = len(ds)
            if any(
                other_life >= life and other_size <= size
                and (other_life > life or other_size < size)
                for _, other_life, other_size in skyline
            ):
                continue
            skyline = [
                (other, other_life, other_size)
                for other, other_life, other_size in skyline
                if not (
                    life >= other_life and size <= other_size
                    and (life > other_life or size < other_size)
                )
            ]
            skyline.append((ds, life, size))
        selected = max(
            skyline,
            key=lambda item: (
                item[1] * max(0.0, 1.0 - size_penalty / item[2]),
                item[1],
                -item[2],
            ),
        )[0]
        family.append(selected)
        remaining.difference_update(selected)
    return tuple(family)


@dataclass
class ModernBaselineAudit:
    algorithm: str
    source_status: str
    validation_basis: str


AUDIT = (
    ModernBaselineAudit("PBIG-R", "paper-based reproduction", "Algorithms 1--2 and tuned domains"),
    ModernBaselineAudit("MC-CMSA-R", "paper-based reconstruction", "six constructors, merge, reduced MILP, ageing"),
    ModernBaselineAudit("FSS-2026-R", "paper-based reproduction", "elite archive, fixed cores, swap local search"),
    ModernBaselineAudit("EAAS-S4C-MAB-R", "paper-based reconstruction", "skyline formation and UCB case scheduling"),
)
