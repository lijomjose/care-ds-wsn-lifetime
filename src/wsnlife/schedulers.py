from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .energy import RadioParameters, round_cost
from .domination import (
    constructive_greedy,
    eaa_disjoint_family,
    gh_mwdds_plus,
    minimal_by_deletion,
)
from .graph import closed_neighbourhoods, is_feasible_set, neighbourhood_sums
from .modern_baselines import (
    eaas_skyline_family,
    fss_family,
    mc_cmsa_family,
    pbig_family,
)


@dataclass
class ScheduleResult:
    algorithm: str
    lifetime: int
    runtime_s: float
    switches: int
    activations: int
    mean_set_size: float
    final_energy: np.ndarray
    valid: bool


def _cached_round_cost(
    selected, previous, adjacency, positions, active_cost, switch_cost, sink, mode,
    energy_model, radio_parameters, cache=None, routing_vertices=None,
    previous_working=None,
):
    if cache is None:
        return round_cost(
            selected,
            previous if previous_working is None else previous_working,
            adjacency, positions, active_cost, switch_cost, sink, mode,
            energy_model, radio_parameters, routing_vertices,
        )
    # Routing availability changes the ordinary-radio path, but neither the
    # unit model nor a connected induced-tree cost. Avoid invalidating those
    # caches when unrelated nodes cross zero energy.
    routing_key = (
        frozenset(routing_vertices)
        if energy_model == "radio" and mode != "connected" and routing_vertices is not None
        else None
    )
    key = (mode, sink, frozenset(selected), routing_key)
    if key not in cache:
        cache[key] = round_cost(
            selected, None, adjacency, positions, active_cost, 0.0, sink, mode,
            energy_model, radio_parameters, routing_vertices,
        )
    cost = cache[key].copy()
    prior = previous if previous_working is None else previous_working
    if prior is not None and switch_cost:
        working = set(np.flatnonzero(cost > 0.0))
        newly_active = working - set(prior)
        if mode == "connected":
            newly_active.discard(sink)
        cost[list(newly_active)] += switch_cost
    return cost


def _candidate_score(adjacency, selected, energy, cost, previous_working, active_cost, switch_cost,
                     variant="CARE-DS", base_neighbourhood_energy=None,
                     reserve_targets=None):
    remaining = energy - cost
    # Every future dominating set must use at least one vertex from N[v].
    if base_neighbourhood_energy is None:
        closed = closed_neighbourhoods(adjacency)
        targets = range(len(adjacency)) if reserve_targets is None else reserve_targets
        neighbourhood_reserve = min(
            sum(max(remaining[u], 0.0) for u in closed[v]) for v in targets
        )
    else:
        # The impact on target v is sum(cost[u] for u in N[v]).  The compiled
        # neighbourhood-sum helper is exactly the former nested update, but is
        # substantially faster on the 500-node submission cases.
        impact = neighbourhood_sums(adjacency, cost)
        neighbourhood_reserve = float(np.min(base_neighbourhood_energy - impact))
    # Under graph-valid radio routing, shortest-path relays outside the DS can
    # also transmit, receive, pay handover energy, and become the bottleneck.
    # Use the complete positive-cost support for every working-set tie breaker.
    # In the unit model this reduces to the selected non-sink vertices.
    working = set(np.flatnonzero(cost > 0.0))
    working_after = [remaining[v] for v in working]
    bottleneck = min(working_after, default=float("inf"))
    entries = len(working - set(previous_working or ()))
    # Reserve dominates, followed by the working-node bottleneck; entries and
    # working-set size provide deterministic tie breaks.
    if variant == "CARE-noreserve":
        return (bottleneck, -switch_cost * entries, -len(working))
    if variant == "CARE-noswitch":
        return (neighbourhood_reserve, bottleneck, -len(working))
    return (neighbourhood_reserve, bottleneck, -switch_cost * entries, -len(working))


def _portfolio(
    adjacency, allowed, energy, mode, sink, rng, random_starts,
    required_vertices=None,
):
    candidates = set()
    degree = np.array([len(x) for x in adjacency], dtype=float)
    candidates.add(minimal_by_deletion(
        adjacency, allowed, energy, mode, sink, required_vertices
    ))
    candidates.add(minimal_by_deletion(
        adjacency, allowed, energy / (degree + 1), mode, sink,
        required_vertices,
    ))
    for power in (0.5, 1.0, 1.5, 2.0):
        candidates.add(
            constructive_greedy(
                adjacency, allowed, energy, mode, sink, power, None, 0.0,
                required_vertices,
            )
        )
    for _ in range(random_starts):
        alpha = float(rng.uniform(0.1, 0.45))
        power = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
        candidates.add(
            constructive_greedy(
                adjacency, allowed, energy, mode, sink, power, rng, alpha,
                required_vertices,
            )
        )
    return [c for c in candidates if c]


def _select(
    algorithm, adjacency, energy, previous, mode, sink, rng, random_starts,
    active_cost, switch_cost, positions=None, energy_model="unit", radio_parameters=None,
    archived_candidates=None, cost_cache=None, required_vertices=None,
    previous_working=None,
):
    n = len(adjacency)
    required = (
        set(range(n)) if required_vertices is None else set(required_vertices)
    )
    # Archived candidates were structurally validated when constructed.  Avoid
    # repeating graph traversals on every round; energy feasibility is still
    # checked below and a depleted archive triggers an immediate reconstruction.
    allowed = None
    if archived_candidates is None:
        eligibility_cost = active_cost if energy_model == "unit" else 1e-12
        allowed = {v for v in required if energy[v] + 1e-12 >= eligibility_cost}
        if mode == "connected":
            allowed.add(sink)
        if not is_feasible_set(adjacency, allowed, mode, sink, required):
            return frozenset()

    if algorithm in {"D-EAA", "D-EAA-ER"}:
        candidates = (
            list(archived_candidates)
            if archived_candidates is not None
            else [minimal_by_deletion(
                adjacency, allowed, energy, mode, sink, required
            )]
        )
    elif algorithm == "DWG":
        candidates = [constructive_greedy(
            adjacency, allowed, energy, mode, sink, 1.0,
            required_vertices=required,
        )]
    elif algorithm == "MS-RG":
        candidates = [
            constructive_greedy(
                adjacency, allowed, energy, mode, sink,
                float(rng.choice([0.5, 1.0, 1.5, 2.0])), rng,
                float(rng.uniform(0.1, 0.45)), required,
            )
            for _ in range(random_starts)
        ]
    elif algorithm in {"MS-RG-TM", "MS-RG-WT"}:
        candidates = (
            list(archived_candidates)
            if archived_candidates is not None
            else [
                constructive_greedy(
                    adjacency, allowed, energy, mode, sink,
                    float(rng.choice([0.5, 1.0, 1.5, 2.0])), rng,
                    float(rng.uniform(0.1, 0.45)), required,
                )
                for _ in range(random_starts + 6)
            ]
        )
    elif algorithm in {"DWG-TM", "D-EAA-TM"}:
        candidates = (
            list(archived_candidates)
            if archived_candidates is not None
            else _portfolio(
                adjacency, allowed, energy, mode, sink, rng, random_starts,
                required,
            )
        )
    elif algorithm in {"CARE-DS", "CARE-noreserve", "CARE-noswitch"}:
        candidates = (
            list(archived_candidates)
            if archived_candidates is not None
            else _portfolio(
                adjacency, allowed, energy, mode, sink, rng, random_starts,
                required,
            )
        )
    else:
        raise ValueError(algorithm)

    feasible = []
    routing_vertices = {
        v for v in required if energy[v] > 1e-12
    } | ({sink} if mode == "connected" else set())
    for candidate in candidates:
        if not candidate or (
            archived_candidates is None
            and not is_feasible_set(adjacency, candidate, mode, sink, required)
        ):
            continue
        cost = _cached_round_cost(
            candidate, previous, adjacency, positions, active_cost, switch_cost, sink, mode,
            energy_model, radio_parameters, cost_cache, routing_vertices,
            previous_working,
        )
        if np.all(energy + 1e-12 >= cost):
            feasible.append((candidate, cost))
    if not feasible:
        return frozenset()
    if algorithm in {"CARE-DS", "CARE-noreserve", "CARE-noswitch"}:
        closed = closed_neighbourhoods(adjacency)
        reserve_targets = set(required)
        if mode == "connected":
            # A target already dominated by the externally powered sink does
            # not impose a finite sensor-energy restriction.
            reserve_targets = {v for v in reserve_targets if sink not in closed[v]}
        # A graph in which the sink covers every live target has no finite
        # critical-neighbourhood reserve.  The remaining score coordinates
        # still provide deterministic candidate selection.
        if not reserve_targets:
            reserve_targets = None
        base_neighbourhood_energy = neighbourhood_sums(
            adjacency, np.maximum(energy, 0.0)
        )
        inactive_targets = np.ones(len(adjacency), dtype=bool)
        inactive_targets[list(required)] = False
        base_neighbourhood_energy[inactive_targets] = np.inf
        if mode == "connected":
            for v in required:
                if sink in closed[v]:
                    base_neighbourhood_energy[v] = np.inf
        return max(
            feasible,
            key=lambda item: _candidate_score(
                adjacency, item[0], energy, item[1],
                previous if previous_working is None else previous_working,
                active_cost, switch_cost,
                algorithm, base_neighbourhood_energy, reserve_targets,
            ),
        )[0]
    if algorithm in {"MS-RG", "MS-RG-TM", "MS-RG-WT", "DWG-TM", "D-EAA-TM"}:
        return max(
            feasible,
            key=lambda item: (
                min(
                    (energy - item[1])[
                        [v for v in item[0] if not (mode == "connected" and v == sink)]
                    ],
                    default=float("inf"),
                ),
                -len(item[0]),
            ),
        )[0]
    return feasible[0][0]


def _family_for_algorithm(algorithm, adjacency, energy, seed, time_budget_s):
    if algorithm == "EAA-static":
        return eaa_disjoint_family(adjacency, energy)
    if algorithm == "GH-MWDDS+":
        return gh_mwdds_plus(adjacency, energy)
    if algorithm == "PBIG-R":
        return pbig_family(adjacency, energy, seed, time_budget_s)
    if algorithm == "MC-CMSA-R":
        return mc_cmsa_family(adjacency, seed, time_budget_s)
    if algorithm == "FSS-2026-R":
        return fss_family(adjacency, seed, time_budget_s)
    if algorithm == "EAAS-S4C-MAB-R":
        eaas = eaas_skyline_family(adjacency, energy, seed)
        eaa = tuple(eaa_disjoint_family(adjacency, energy))
        def weighted_value(family):
            return sum(float(np.min(energy[list(ds)])) for ds in family)
        return max((eaas, eaa), key=weighted_value)
    raise ValueError(algorithm)


def _family_schedule(
    name, family, adjacency, energy, positions, active_cost, switch_cost, energy_model,
    radio_parameters, failure_events, max_rounds, seed,
):
    """Schedule a fixed disjoint family; EAAS uses a UCB four-case selector."""
    rng = np.random.default_rng(seed)
    initial = energy.copy()
    previous = None
    previous_working = None
    lifetime = switches = activations = 0
    sizes = []
    failed = set()
    alive = set(range(len(adjacency)))
    family = tuple(ds for ds in family if ds)
    if not family:
        return ScheduleResult(name, 0, 0.0, 0, 0, 0.0, energy, True)
    pulls = np.zeros(4, dtype=int)
    rewards = np.array([1.00, 0.95, 0.70, 0.60], dtype=float)
    last_used = np.full(len(family), -10**9, dtype=int)
    consecutive = np.zeros(len(family), dtype=int)
    cooldown = np.zeros(len(family), dtype=int)
    activation_caps = np.array([1, 3, 1, 2], dtype=int)
    cooldown_lengths = np.array([2, 1, 3, 2], dtype=int)
    mab_beta = 0.05
    size_penalty = 0.5
    cost_cache = {}
    set_lives = np.array([min(initial[list(ds)]) for ds in family], dtype=float)
    median_size = float(np.median([len(ds) for ds in family]))
    median_life = float(np.median(set_lives))

    def case_of(index):
        small = len(family[index]) <= median_size
        long = set_lives[index] >= median_life
        return (0 if small else 2) + (0 if long else 1)

    for round_no in range(max_rounds):
        for v in failure_events.get(round_no, ()):
            failed.add(v)
            alive.discard(v)
            energy[v] = 0.0
        feasible = []
        for i, ds in enumerate(family):
            residual_ds = frozenset(set(ds) - failed)
            if not residual_ds or not is_feasible_set(
                adjacency, residual_ds, required_vertices=alive
            ):
                continue
            routing_vertices = {
                v for v in alive if energy[v] > 1e-12
            } | set(residual_ds)
            cost = _cached_round_cost(
                residual_ds, previous, adjacency, positions, active_cost, switch_cost, 0, "dominating",
                energy_model, radio_parameters, cost_cache, routing_vertices,
                previous_working,
            )
            if np.all(energy + 1e-12 >= cost):
                positive = cost > 0
                remaining_rounds = np.min(energy[positive] / cost[positive]) if np.any(positive) else 0
                feasible.append((i, residual_ds, cost, float(remaining_rounds)))
        if not feasible:
            break
        if name == "EAAS-S4C-MAB-R":
            median_size_now = float(np.median([len(x[1]) for x in feasible]))
            median_life_now = float(np.median([x[3] for x in feasible]))

            def current_case(ds, life):
                small = len(ds) < median_size_now
                strong = life >= median_life_now
                return (0 if small else 1) if strong else (2 if small else 3)

            total = max(1, int(np.sum(pulls)))
            weights = rewards + mab_beta * np.sqrt(
                np.log(round_no + 2.0) / (pulls + 1.0)
            )
            scored = []
            for i, ds, candidate_cost, remaining_rounds in feasible:
                case = current_case(ds, remaining_rounds)
                adjusted = remaining_rounds * max(
                    0.0, 1.0 - size_penalty / max(len(ds), 1)
                )
                score = float(weights[case]) * adjusted
                strict = (
                    cooldown[i] == 0
                    and consecutive[i] < activation_caps[case]
                    and adjusted > 0
                )
                scored.append((strict, score, adjusted, remaining_rounds,
                               len(ds), i, ds, candidate_cost, case))
            strict_pool = [x for x in scored if x[0]]
            relaxed_pool = [x for x in scored if x[2] > 0]
            if strict_pool:
                chosen = max(strict_pool, key=lambda x: (x[1], x[4], -x[5]))
            elif relaxed_pool:
                chosen = max(relaxed_pool, key=lambda x: (x[1], x[4], -x[5]))
            else:
                chosen = max(scored, key=lambda x: (x[3], x[4], -x[5]))
            _, _, _, _, _, i, selected, cost, case = chosen
            q_pre = float(sum(x[2] for x in scored))
        else:
            # Block scheduling minimizes handovers for a static family.
            continued = next((x for x in feasible if previous is not None and x[1] == previous), None)
            if continued is not None:
                _, selected, cost, _ = continued
            else:
                _, selected, cost, _ = max(feasible, key=lambda x: (x[3], -len(x[1])))
        if previous is not None:
            switches += len(set(selected) ^ set(previous))
        energy -= cost
        energy[np.abs(energy) < 1e-10] = 0.0
        current_working = frozenset(np.flatnonzero(cost > 0.0))
        if name == "EAAS-S4C-MAB-R":
            post_utilities = []
            for _, ds0 in enumerate(family):
                residual_ds = frozenset(set(ds0) - failed)
                if not residual_ds or not is_feasible_set(
                    adjacency, residual_ds, required_vertices=alive
                ):
                    continue
                next_cost = _cached_round_cost(
                    residual_ds, selected, adjacency, positions, active_cost,
                    switch_cost, 0, "dominating", energy_model,
                    radio_parameters, cost_cache,
                    {v for v in alive if energy[v] > 1e-12} | set(residual_ds),
                    current_working,
                )
                if np.all(energy + 1e-12 >= next_cost):
                    positive = next_cost > 0
                    life = (
                        float(np.min(energy[positive] / next_cost[positive]))
                        if np.any(positive) else 0.0
                    )
                    post_utilities.append(
                        life * max(0.0, 1.0 - size_penalty / len(residual_ds))
                    )
            reward = float(sum(post_utilities)) / max(q_pre, 1e-12)
            pulls[case] += 1
            rewards[case] += (reward - rewards[case]) / pulls[case]
            for index in range(len(family)):
                if index == i:
                    continue
                cooldown[index] = max(0, cooldown[index] - 1)
                consecutive[index] = 0
            cooldown[i] = cooldown_lengths[case]
            consecutive[i] += 1
            last_used[i] = round_no
        lifetime += 1
        activations += len(selected)
        sizes.append(len(selected))
        previous = selected
        previous_working = current_working
    return ScheduleResult(
        name, lifetime, 0.0, switches, activations,
        float(np.mean(sizes)) if sizes else 0.0, energy, True,
    )


def _static_result(name, family, energy, elapsed, active_cost=1.0):
    blocks = [int(np.floor(min(energy[list(ds)]) / active_cost)) for ds in family]
    lifetime = int(sum(blocks))
    activations = sum(len(ds) * rounds for ds, rounds in zip(family, blocks))
    total_rounds = max(lifetime, 1)
    return ScheduleResult(
        algorithm=name,
        lifetime=lifetime,
        runtime_s=elapsed,
        switches=max(len(family) - 1, 0),
        activations=activations,
        mean_set_size=activations / total_rounds,
        final_energy=energy.copy(),
        valid=all(family),
    )


def run_algorithm(
    algorithm,
    adjacency,
    initial_energy,
    mode="dominating",
    sink=0,
    active_cost=1.0,
    switch_cost=0.0,
    random_starts=10,
    seed=0,
    max_rounds=5000,
    positions=None,
    energy_model="unit",
    radio_parameters=None,
    failure_events=None,
    refresh_interval=1,
    static_time_budget_s=0.25,
    dwell_scale=100.0,
):
    start = time.perf_counter()
    energy = np.asarray(initial_energy, dtype=float).copy()
    failure_events = failure_events or {}
    if isinstance(radio_parameters, dict):
        radio_parameters = RadioParameters(**radio_parameters)

    static_algorithms = {
        "EAA-static", "GH-MWDDS+", "PBIG-R", "MC-CMSA-R", "FSS-2026-R",
        "EAAS-S4C-MAB-R",
    }
    if algorithm in static_algorithms:
        if mode != "dominating":
            return ScheduleResult(algorithm, -1, 0.0, 0, 0, 0.0, energy, True)
        family = _family_for_algorithm(
            algorithm, adjacency, energy, seed, static_time_budget_s,
        )
        construction_elapsed = time.perf_counter() - start
        # For a disjoint family under the unit model, with no handover or
        # exogenous failures, block order cannot affect lifetime. Evaluate the
        # exact sum of per-set bottleneck lifetimes instead of replaying rounds.
        if energy_model == "unit" and switch_cost == 0 and not failure_events:
            return _static_result(
                algorithm, family, energy, construction_elapsed, active_cost
            )
        scheduled = _family_schedule(
            algorithm, family, adjacency, energy, positions, active_cost, switch_cost,
            energy_model, radio_parameters, failure_events, max_rounds, seed,
        )
        scheduled.runtime_s = construction_elapsed + (time.perf_counter() - start - construction_elapsed)
        return scheduled

    rng = np.random.default_rng(seed)
    previous = None
    previous_working = None
    lifetime = switches = activations = 0
    sizes = []
    valid = True
    archive = None
    cost_cache = {}
    validity_cache = {}
    alive = set(range(len(adjacency)))
    last_change_round = -10**9
    portfolio_algorithms = {
        "CARE-DS", "CARE-noreserve", "CARE-noswitch", "DWG-TM", "D-EAA-TM",
        "MS-RG-TM", "MS-RG-WT", "D-EAA-ER",
    }
    for round_no in range(max_rounds):
        failures_now = failure_events.get(round_no, ())
        for v in failures_now:
            if not (mode == "connected" and v == sink):
                energy[v] = 0.0
                alive.discard(v)
        if failures_now:
            archive = None
            validity_cache.clear()
            cost_cache.clear()
        effective_refresh = (
            max(1, int(round(refresh_interval * 0.45)))
            if algorithm == "MS-RG-WT" else max(1, refresh_interval)
        )
        if algorithm in portfolio_algorithms and (
            archive is None or round_no % effective_refresh == 0
        ):
            eligibility_cost = active_cost if energy_model == "unit" else 1e-12
            allowed = {v for v in alive if energy[v] + 1e-12 >= eligibility_cost}
            if mode == "connected":
                allowed.add(sink)
            if not is_feasible_set(adjacency, allowed, mode, sink, alive):
                archive = []
            elif algorithm == "D-EAA-ER":
                archive = [minimal_by_deletion(
                    adjacency, allowed, energy, mode, sink, alive
                )]
            else:
                archive = _portfolio(
                    adjacency, allowed, energy, mode, sink, rng,
                    random_starts, alive,
                )
        selected = frozenset()
        # Cost-aware hysteresis: a positive handover cost creates a minimum
        # residence period, scaled once from the declared cost.  The schedule
        # may always leave early if the current set becomes infeasible.  This
        # rule is disabled in CARE-noswitch for a controlled ablation.
        if (
            algorithm in {"CARE-DS", "CARE-noreserve"}
            and previous is not None
            and switch_cost > 0
        ):
            if energy_model == "unit":
                reference_cost = max(float(active_cost), 1e-12)
            else:
                params = radio_parameters or RadioParameters()
                reference_cost = max(
                    params.packet_bits * params.electronics_j_per_bit, 1e-12
                )
            dwell_rounds = max(
                1,
                int(math.ceil(float(dwell_scale) * switch_cost / reference_cost)),
            )
            if round_no - last_change_round < dwell_rounds:
                routing_vertices = {
                    v for v in alive if energy[v] > 1e-12
                } | set(previous)
                continued_cost = _cached_round_cost(
                    previous, previous, adjacency, positions, active_cost,
                    switch_cost, sink, mode, energy_model, radio_parameters,
                    cost_cache, routing_vertices,
                    previous_working,
                )
                if np.all(energy + 1e-12 >= continued_cost):
                    selected = previous
        if not selected:
            selected = _select(
                algorithm, adjacency, energy, previous, mode, sink, rng,
                random_starts, active_cost, switch_cost, positions, energy_model,
                radio_parameters, archive, cost_cache, alive,
                previous_working,
            )
        if not selected and algorithm in portfolio_algorithms and refresh_interval > 1:
            eligibility_cost = active_cost if energy_model == "unit" else 1e-12
            allowed = {v for v in alive if energy[v] + 1e-12 >= eligibility_cost}
            if mode == "connected":
                allowed.add(sink)
            archive = (
                [minimal_by_deletion(
                    adjacency, allowed, energy, mode, sink, alive
                )]
                if algorithm == "D-EAA-ER"
                else _portfolio(
                    adjacency, allowed, energy, mode, sink, rng,
                    random_starts, alive,
                )
            )
            selected = _select(
                algorithm, adjacency, energy, previous, mode, sink, rng,
                random_starts, active_cost, switch_cost, positions, energy_model,
                radio_parameters, archive, cost_cache, alive,
                previous_working,
            )
        if not selected:
            break
        structurally_valid = validity_cache.get(selected)
        if structurally_valid is None:
            structurally_valid = is_feasible_set(
                adjacency, selected, mode, sink, alive
            )
            validity_cache[selected] = structurally_valid
        if not structurally_valid:
            valid = False
            break
        routing_vertices = {
            v for v in alive if energy[v] > 1e-12
        } | set(selected) | ({sink} if mode == "connected" else set())
        cost = _cached_round_cost(
            selected, previous, adjacency, positions, active_cost, switch_cost, sink, mode,
            energy_model, radio_parameters, cost_cache,
            routing_vertices,
            previous_working,
        )
        if np.any(cost > energy + 1e-9):
            valid = False
            break
        if previous is not None:
            switches += len(set(selected) ^ set(previous))
            if selected != previous:
                last_change_round = round_no
        else:
            last_change_round = round_no
        energy -= cost
        energy[np.abs(energy) < 1e-10] = 0.0
        lifetime += 1
        activations += len(selected)
        sizes.append(len(selected))
        previous = selected
        previous_working = frozenset(np.flatnonzero(cost > 0.0))
    return ScheduleResult(
        algorithm=algorithm,
        lifetime=lifetime,
        runtime_s=time.perf_counter() - start,
        switches=switches,
        activations=activations,
        mean_set_size=float(np.mean(sizes)) if sizes else 0.0,
        final_energy=energy,
        valid=valid,
    )
