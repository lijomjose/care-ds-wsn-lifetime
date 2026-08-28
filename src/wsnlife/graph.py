from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class GraphInstance:
    adjacency: tuple[frozenset[int], ...]
    positions: np.ndarray
    energies: np.ndarray
    seed: int
    topology: str
    target_degree: float

    @property
    def n(self) -> int:
        return len(self.adjacency)

    @property
    def average_degree(self) -> float:
        return float(sum(map(len, self.adjacency)) / self.n)


@lru_cache(maxsize=256)
def _closed_neighbourhoods_cached(adjacency):
    return tuple(frozenset({v, *adjacency[v]}) for v in range(len(adjacency)))


@lru_cache(maxsize=256)
def _closed_neighbourhood_index_cached(adjacency):
    closed = _closed_neighbourhoods_cached(adjacency)
    owners = np.fromiter(
        (v for v, neighbourhood in enumerate(closed) for _ in neighbourhood),
        dtype=np.int32,
    )
    nodes = np.fromiter(
        (u for neighbourhood in closed for u in neighbourhood),
        dtype=np.int32,
    )
    return owners, nodes


def closed_neighbourhoods(adjacency):
    """Return cached closed neighbourhoods for an immutable adjacency tuple."""
    return _closed_neighbourhoods_cached(tuple(adjacency))


def neighbourhood_sums(adjacency, values):
    """Sum node values over every closed neighbourhood in compiled NumPy code."""
    owners, nodes = _closed_neighbourhood_index_cached(tuple(adjacency))
    return np.bincount(
        owners,
        weights=np.asarray(values, dtype=float)[nodes],
        minlength=len(adjacency),
    )


def is_connected(adjacency, vertices=None) -> bool:
    n = len(adjacency)
    allowed = set(range(n)) if vertices is None else set(vertices)
    if not allowed:
        return False
    start = next(iter(allowed))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adjacency[v]:
            if u in allowed and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen == allowed


def connected_components(adjacency):
    unseen = set(range(len(adjacency)))
    components = []
    while unseen:
        start = next(iter(unseen))
        component = {start}
        stack = [start]
        unseen.remove(start)
        while stack:
            v = stack.pop()
            neighbours = set(adjacency[v]) & unseen
            unseen.difference_update(neighbours)
            component.update(neighbours)
            stack.extend(neighbours)
        components.append(component)
    return components


def _bridge_components(adjacency, positions):
    """Add the shortest Euclidean gateways until all components are connected."""
    mutable = [set(x) for x in adjacency]
    components = connected_components(tuple(frozenset(x) for x in mutable))
    if len(components) == 1:
        return _freeze(mutable)
    labels = np.empty(len(mutable), dtype=np.int32)
    for label, component in enumerate(components):
        labels[list(component)] = label

    # Repeatedly taking the shortest edge across the current components is
    # Kruskal's rule. Sorting the candidate edges once and maintaining component
    # membership with union--find is decision-equivalent and avoids rebuilding
    # O(n^2) distance arrays after every bridge.
    upper_u, upper_v = np.triu_indices(len(mutable), 1)
    cross = labels[upper_u] != labels[upper_v]
    candidates_u, candidates_v = upper_u[cross], upper_v[cross]
    delta = positions[candidates_u] - positions[candidates_v]
    distance_sq = np.einsum("ij,ij->i", delta, delta)
    order = np.argsort(distance_sq, kind="stable")
    parent = np.arange(len(components), dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    bridges_needed = len(components) - 1
    for index in order:
        u, v = int(candidates_u[index]), int(candidates_v[index])
        root_u, root_v = find(int(labels[u])), find(int(labels[v]))
        if root_u == root_v:
            continue
        parent[root_v] = root_u
        mutable[u].add(v)
        mutable[v].add(u)
        bridges_needed -= 1
        if bridges_needed == 0:
            break
    return _freeze(mutable)


def is_dominating(adjacency, vertices, required_vertices=None) -> bool:
    """Return whether ``vertices`` dominates the required live-node universe.

    ``required_vertices`` is normally all vertices. Failure experiments pass
    the currently alive vertices so a failed sensor is neither selectable nor
    a service target.
    """
    selected = set(vertices)
    required = (
        set(range(len(adjacency)))
        if required_vertices is None else set(required_vertices)
    )
    if not required:
        return not selected
    if not selected or not selected <= required:
        return False
    covered = set(selected)
    for v in selected:
        covered.update(adjacency[v])
        if required <= covered:
            return True
    return required <= covered


def is_feasible_set(
    adjacency, vertices, mode="dominating", sink=0, required_vertices=None
) -> bool:
    selected = set(vertices)
    required = (
        set(range(len(adjacency)))
        if required_vertices is None else set(required_vertices)
    )
    if not is_dominating(adjacency, selected, required):
        return False
    if mode == "connected":
        return sink in required and sink in selected and is_connected(adjacency, selected)
    return True


def _adjacency_from_positions(positions: np.ndarray, target_degree: float):
    n = len(positions)
    # Vectorized exact equivalent of sorting all pairwise Euclidean distances.
    # Only the target order statistic and the edges below it are required.
    upper_u, upper_v = np.triu_indices(n, 1)
    delta = positions[upper_u] - positions[upper_v]
    distance_sq = np.einsum("ij,ij->i", delta, delta)
    target_edges = max(n - 1, int(round(n * target_degree / 2)))
    kth = min(target_edges - 1, len(distance_sq) - 1)
    cutoff_sq = float(np.partition(distance_sq, kth)[kth])
    cutoff = math.sqrt(cutoff_sq)
    edge_mask = distance_sq <= (cutoff + 1e-12) ** 2
    adjacency = [set() for _ in range(n)]
    for u, v in zip(upper_u[edge_mask], upper_v[edge_mask]):
        adjacency[int(u)].add(int(v))
        adjacency[int(v)].add(int(u))
    return tuple(frozenset(x) for x in adjacency), cutoff


def _sample_positions(rng, n: int, topology: str) -> np.ndarray:
    if topology == "rgg":
        return rng.random((n, 2))
    if topology == "clustered":
        centres = np.array([[0.24, 0.25], [0.75, 0.27], [0.48, 0.75]])
        labels = rng.integers(0, len(centres), n)
        pos = centres[labels] + rng.normal(0.0, 0.13, size=(n, 2))
        return np.clip(pos, 0.0, 1.0)
    raise ValueError(f"unknown topology: {topology}")


def _freeze(adjacency):
    return tuple(frozenset(x) for x in adjacency)


def _sample_er(rng, n, target_degree):
    p = min(max(target_degree / (n - 1), 0.0), 1.0)
    adjacency = [set() for _ in range(n)]
    for u in range(n):
        draws = rng.random(n - u - 1) < p
        for offset in np.flatnonzero(draws):
            v = u + 1 + int(offset)
            adjacency[u].add(v)
            adjacency[v].add(u)
    return _freeze(adjacency)


def _sample_watts_strogatz(rng, n, target_degree, beta=0.1):
    k = min(n - 1, max(2, int(round(target_degree))))
    if k % 2:
        k -= 1
    adjacency = [set() for _ in range(n)]
    for u in range(n):
        for step in range(1, k // 2 + 1):
            v = (u + step) % n
            adjacency[u].add(v)
            adjacency[v].add(u)
    for u in range(n):
        for step in range(1, k // 2 + 1):
            v = (u + step) % n
            if v not in adjacency[u] or rng.random() >= beta:
                continue
            forbidden = adjacency[u] | {u}
            options = [x for x in range(n) if x not in forbidden]
            if not options:
                continue
            w = int(rng.choice(options))
            adjacency[u].remove(v)
            adjacency[v].remove(u)
            adjacency[u].add(w)
            adjacency[w].add(u)
    return _freeze(adjacency)


def _sample_barabasi_albert(rng, n, target_degree):
    m = min(n - 1, max(1, int(round(target_degree / 2))))
    initial = min(n, m + 1)
    adjacency = [set() for _ in range(n)]
    repeated = []
    for u in range(initial):
        for v in range(u + 1, initial):
            adjacency[u].add(v)
            adjacency[v].add(u)
    for v in range(initial):
        repeated.extend([v] * max(len(adjacency[v]), 1))
    for new in range(initial, n):
        if repeated:
            targets = set()
            while len(targets) < min(m, new):
                targets.add(int(rng.choice(repeated)))
        else:
            targets = set(rng.choice(new, size=min(m, new), replace=False).tolist())
        for v in targets:
            adjacency[new].add(v)
            adjacency[v].add(new)
            repeated.extend([new, v])
    return _freeze(adjacency)


def generate_instance(
    n: int,
    target_degree: float,
    seed: int,
    topology: str = "rgg",
    energy_low: int = 20,
    energy_high: int = 100,
    max_attempts: int = 500,
) -> GraphInstance:
    """Generate a connected unit-disk graph with deterministic heterogeneous budgets.

    The radius is the distance threshold giving approximately the requested
    average degree. Entire point sets are resampled when that threshold graph is
    disconnected; edges are never added outside the final geometric radius.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        if topology in {"rgg", "clustered"}:
            positions = _sample_positions(rng, n, topology)
            adjacency, _ = _adjacency_from_positions(positions, target_degree)
        elif topology == "er":
            positions = rng.random((n, 2))
            adjacency = _sample_er(rng, n, target_degree)
        elif topology == "watts_strogatz":
            theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
            positions = np.c_[0.5 + 0.45 * np.cos(theta), 0.5 + 0.45 * np.sin(theta)]
            adjacency = _sample_watts_strogatz(rng, n, target_degree)
        elif topology == "barabasi_albert":
            positions = rng.random((n, 2))
            adjacency = _sample_barabasi_albert(rng, n, target_degree)
        else:
            raise ValueError(f"unknown topology: {topology}")
        if is_connected(adjacency):
            if isinstance(energy_low, float) or isinstance(energy_high, float):
                energies = rng.uniform(float(energy_low), float(energy_high), size=n)
            else:
                energies = rng.integers(energy_low, energy_high + 1, size=n).astype(float)
            return GraphInstance(
                adjacency=adjacency,
                positions=positions,
                energies=energies,
                seed=seed,
                topology=topology,
                target_degree=target_degree,
            )
    # Strongly separated clustered fields can remain disconnected at the
    # requested density even after rejection sampling.  A deterministic
    # shortest-gateway fallback preserves the clusters and changes mean degree
    # by only 2(c-1)/n for c components.
    adjacency = _bridge_components(adjacency, positions)
    if isinstance(energy_low, float) or isinstance(energy_high, float):
        energies = rng.uniform(float(energy_low), float(energy_high), size=n)
    else:
        energies = rng.integers(energy_low, energy_high + 1, size=n).astype(float)
    return GraphInstance(
        adjacency=adjacency,
        positions=positions,
        energies=energies,
        seed=seed,
        topology=topology,
        target_degree=target_degree,
    )
