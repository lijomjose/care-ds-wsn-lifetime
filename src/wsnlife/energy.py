from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadioParameters:
    """First-order radio parameters in SI units.

    Defaults follow the common LEACH-style model: 50 nJ/bit electronics,
    100 pJ/bit/m^2 free-space amplifier, and 5 nJ/bit aggregation.
    """

    packet_bits: int = 4000
    electronics_j_per_bit: float = 50e-9
    amplifier_j_per_bit_m2: float = 100e-12
    aggregation_j_per_bit: float = 5e-9
    field_side_m: float = 100.0
    sink_xy_m: tuple[float, float] = (50.0, 110.0)


def _distance_m(positions, u, v, params):
    p = np.asarray(positions, dtype=float) * params.field_side_m
    return float(np.linalg.norm(p[u] - p[v]))


def _tx(distance_m, params):
    k = params.packet_bits
    return k * (params.electronics_j_per_bit + params.amplifier_j_per_bit_m2 * distance_m**2)


def _rx(params):
    return params.packet_bits * params.electronics_j_per_bit


def _da(params):
    return params.packet_bits * params.aggregation_j_per_bit


def _shortest_path(adjacency, source, target, allowed):
    """Return a deterministic minimum-hop path restricted to ``allowed`` nodes."""
    if source == target:
        return [source]
    parents = {source: None}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbour in sorted(adjacency[current] & allowed):
            if neighbour in parents:
                continue
            parents[neighbour] = current
            if neighbour == target:
                path = [target]
                while path[-1] != source:
                    path.append(parents[path[-1]])
                return list(reversed(path))
            queue.append(neighbour)
    return None


def _nearest_neighbour_chain(selected, positions, params):
    """Order selected terminals for a deterministic PEGASIS-style chain."""
    selected = list(selected)
    p = np.asarray(positions, dtype=float) * params.field_side_m
    sink = np.asarray(params.sink_xy_m, dtype=float)
    leader = min(selected, key=lambda v: (np.linalg.norm(p[v] - sink), v))
    start = max(selected, key=lambda v: (np.linalg.norm(p[v] - sink), -v))
    chain = [start]
    unused = set(selected) - {start}
    while unused:
        current = chain[-1]
        # Keep the leader for the end unless it is the only remaining node.
        options = unused - {leader} if leader in unused and len(unused) > 1 else unused
        nxt = min(options, key=lambda v: (np.linalg.norm(p[current] - p[v]), v))
        chain.append(nxt)
        unused.remove(nxt)
    return chain


def _connected_tree_parents(adjacency, selected, sink):
    selected = set(selected)
    parents = {sink: None}
    queue = [sink]
    for v in queue:
        for u in sorted(adjacency[v] & selected):
            if u not in parents:
                parents[u] = v
                queue.append(u)
    return parents


def round_cost(
    selected,
    previous,
    adjacency,
    positions,
    active_cost,
    switch_cost,
    sink,
    mode,
    energy_model="unit",
    radio_parameters=None,
    routing_vertices=None,
):
    """Per-node cost for one service round under unit or first-order radio energy."""
    n = len(adjacency)
    cost = np.zeros(n, dtype=float)
    chosen = set(selected)
    if energy_model == "unit":
        cost[list(chosen)] = active_cost
    elif energy_model == "radio":
        if positions is None:
            raise ValueError("radio energy requires node positions")
        params = radio_parameters or RadioParameters()
        active = chosen - ({sink} if mode == "connected" else set())
        if mode == "connected":
            parents = _connected_tree_parents(adjacency, chosen, sink)
            if set(parents) != chosen:
                return np.full(n, np.inf)
            for u, parent in parents.items():
                if u == sink:
                    continue
                cost[u] += _tx(_distance_m(positions, u, parent, params), params)
                if parent != sink:
                    cost[parent] += _rx(params) + _da(params)
        elif active:
            chain = _nearest_neighbour_chain(active, positions, params)
            routing_vertices = (
                set(range(n)) if routing_vertices is None else set(routing_vertices)
            )
            if not active <= routing_vertices:
                return np.full(n, np.inf)
            for u, v in zip(chain, chain[1:]):
                path = _shortest_path(adjacency, u, v, routing_vertices)
                if path is None:
                    return np.full(n, np.inf)
                for sender, receiver in zip(path, path[1:]):
                    cost[sender] += _tx(
                        _distance_m(positions, sender, receiver, params), params
                    )
                    cost[receiver] += _rx(params)
                # Only the next selected terminal aggregates the incoming
                # chain packet with its own report. Transit relays forward it.
                cost[v] += _da(params)
            leader = chain[-1]
            p = np.asarray(positions, dtype=float)[leader] * params.field_side_m
            distance = float(np.linalg.norm(p - np.asarray(params.sink_xy_m)))
            cost[leader] += _tx(distance, params)
    else:
        raise ValueError(f"unknown energy model: {energy_model}")

    if previous is not None and switch_cost:
        # Under the radio model, graph-path relays outside the dominating set
        # also wake and consume energy. ``previous`` therefore denotes the
        # complete previous working set, not only the previous DS terminals.
        working = set(np.flatnonzero(cost > 0.0))
        newly_active = working - set(previous)
        if mode == "connected":
            newly_active.discard(sink)
        cost[list(newly_active)] += switch_cost
    if mode == "connected":
        cost[sink] = 0.0
    return cost
