import numpy as np

from wsnlife.domination import eaa_disjoint_family, gh_mwdds_plus
from wsnlife.exact import capacitated_schedule_optimum
from wsnlife.graph import generate_instance, is_feasible_set
from wsnlife.schedulers import _candidate_score, run_algorithm
from wsnlife.energy import RadioParameters, round_cost
from wsnlife.modern_baselines import (
    eaas_skyline_family, fss_family, mc_cmsa_family, pbig_family,
)


def test_generated_graph_and_schedules_are_valid():
    graph = generate_instance(30, 12, 12345)
    for algorithm in ("D-EAA", "DWG", "MS-RG", "CARE-DS"):
        result = run_algorithm(
            algorithm, graph.adjacency, graph.energies,
            seed=7, random_starts=4, max_rounds=1000,
        )
        assert result.valid
        assert result.lifetime > 0


def test_connected_mode_contains_valid_service_sets():
    graph = generate_instance(30, 12, 54321)
    result = run_algorithm(
        "CARE-DS", graph.adjacency, graph.energies,
        mode="connected", seed=9, random_starts=4,
    )
    assert result.valid and result.lifetime > 0


def test_static_families_are_disjoint_and_dominating():
    graph = generate_instance(25, 10, 999)
    for family in (
        eaa_disjoint_family(graph.adjacency, graph.energies),
        gh_mwdds_plus(graph.adjacency, graph.energies),
    ):
        used = set()
        for ds in family:
            assert is_feasible_set(graph.adjacency, ds)
            assert not (used & set(ds))
            used.update(ds)


def test_exact_small_cycle():
    adjacency = tuple(
        frozenset({(v - 1) % 6, (v + 1) % 6}) for v in range(6)
    )
    optimum, _, result = capacitated_schedule_optimum(adjacency, np.ones(6))
    assert result.success
    assert optimum == 3


def test_modern_reproductions_return_disjoint_dominating_families():
    graph = generate_instance(24, 8, 2026, "er", 2, 6)
    families = (
        pbig_family(graph.adjacency, graph.energies, 1, 0.03, 4),
        mc_cmsa_family(graph.adjacency, 2, 0.03),
        fss_family(graph.adjacency, 3, 0.03, 8),
        eaas_skyline_family(graph.adjacency, graph.energies, 4, 4),
    )
    for family in families:
        used = set()
        assert family
        for ds in family:
            assert is_feasible_set(graph.adjacency, ds)
            assert not (used & set(ds))
            used.update(ds)


def test_failed_nodes_are_removed_from_service_universe():
    adjacency = (
        frozenset({1}),
        frozenset({0, 2}),
        frozenset({1}),
    )
    assert not is_feasible_set(adjacency, {0})
    assert is_feasible_set(adjacency, {0}, required_vertices={0, 1})

    graph = generate_instance(30, 10, 404, "er", 10, 20)
    result = run_algorithm(
        "CARE-DS", graph.adjacency, graph.energies,
        failure_events={2: {5, 6}}, seed=22, random_starts=3,
        refresh_interval=5, max_rounds=100,
    )
    assert result.valid and result.lifetime > 2


def test_first_order_radio_cost_does_not_double_count_tx_electronics():
    adjacency = (frozenset({1}), frozenset({0}))
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    params = RadioParameters(
        packet_bits=100, electronics_j_per_bit=2.0,
        amplifier_j_per_bit_m2=3.0, aggregation_j_per_bit=5.0,
        field_side_m=1.0, sink_xy_m=(0.0, 0.0),
    )
    cost = round_cost(
        {1}, None, adjacency, positions, 0.0, 0.0, 0, "dominating",
        "radio", params,
    )
    # A single active node transmits once to the sink: k(E_elec + eps*d^2).
    assert cost[1] == 100 * (2.0 + 3.0)


def test_ordinary_radio_routes_only_over_graph_edges_and_charges_relays():
    adjacency = (
        frozenset({1}),
        frozenset({0, 2}),
        frozenset({1}),
    )
    # Nodes 0 and 2 are geometrically close but are not adjacent.  Their chain
    # packet must travel through node 1, which must pay receive and transmit.
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.1, 0.0]])
    params = RadioParameters(
        packet_bits=1, electronics_j_per_bit=1.0,
        amplifier_j_per_bit_m2=0.0, aggregation_j_per_bit=0.0,
        field_side_m=1.0, sink_xy_m=(0.0, 0.0),
    )
    cost = round_cost(
        {0, 2}, None, adjacency, positions, 0.0, 0.0, 0, "dominating",
        "radio", params,
    )
    assert cost[1] == 2.0


def test_ordinary_radio_avoids_energy_depleted_relays():
    adjacency = (
        frozenset({1, 3}),
        frozenset({0, 2}),
        frozenset({1, 3}),
        frozenset({0, 2}),
    )
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    params = RadioParameters(
        packet_bits=1, electronics_j_per_bit=1.0,
        amplifier_j_per_bit_m2=0.0, aggregation_j_per_bit=0.0,
        field_side_m=1.0, sink_xy_m=(0.0, 0.0),
    )
    cost = round_cost(
        {0, 2}, None, adjacency, positions, 0.0, 0.0, 0, "dominating",
        "radio", params, routing_vertices={0, 2, 3},
    )
    assert cost[1] == 0.0
    assert cost[3] == 2.0


def test_connected_reserve_ignores_sink_covered_neighbourhoods():
    adjacency = (
        frozenset({1}),
        frozenset({0, 2}),
        frozenset({1}),
    )
    energy = np.array([0.01, 10.0, 8.0])
    cost = np.zeros(3)
    base = np.array([np.inf, np.inf, 18.0])
    score = _candidate_score(
        adjacency, frozenset({0, 1}), energy, cost, None, 1.0, 0.0,
        base_neighbourhood_energy=base, reserve_targets={2},
    )
    assert score[0] == 18.0


def test_radio_candidate_score_includes_relays_and_route_entries():
    adjacency = (
        frozenset({1}),
        frozenset({0, 2}),
        frozenset({1}),
    )
    # Nodes 0 and 2 are DS terminals. Node 1 is a newly entering relay whose
    # small residual budget must determine the working-node bottleneck.
    selected = frozenset({0, 2})
    energy = np.array([10.0, 2.5, 10.0])
    cost = np.array([1.0, 2.0, 1.0])
    score = _candidate_score(
        adjacency, selected, energy, cost, frozenset({0, 2}), 1.0, 0.25,
        variant="CARE-noreserve",
    )
    assert score == (0.5, -0.25, -3)


def test_dwell_matched_control_changes_only_the_dwell_rule():
    graph = generate_instance(36, 12, 20260831, "er", 0.05, 0.20)
    for switch_cost, dwell_scale in ((0.0, 100.0), (1e-4, 0.0)):
        common = dict(
            mode="dominating", active_cost=0.0, switch_cost=switch_cost,
            random_starts=4, seed=31, max_rounds=200,
            positions=graph.positions, energy_model="radio",
            refresh_interval=20, dwell_scale=dwell_scale,
        )
        baseline = run_algorithm(
            "MS-RG-WT", graph.adjacency, graph.energies, **common
        )
        matched = run_algorithm(
            "MS-RG-WT-DW", graph.adjacency, graph.energies, **common
        )
        assert baseline.valid and matched.valid
        assert matched.lifetime == baseline.lifetime
        assert matched.switches == baseline.switches
        assert matched.activations == baseline.activations

    radio_common = dict(
        mode="dominating", active_cost=0.0, switch_cost=1e-4,
        random_starts=4, seed=31, max_rounds=200,
        positions=graph.positions, energy_model="radio",
        refresh_interval=20, dwell_scale=100.0,
    )
    historical = run_algorithm(
        "MS-RG-WT-DW", graph.adjacency, graph.energies, **radio_common
    )
    pinned = run_algorithm(
        "MS-RG-WT-DW", graph.adjacency, graph.energies,
        dwell_rounds_override=51, **radio_common,
    )
    assert historical.lifetime == pinned.lifetime
    assert historical.switches == pinned.switches
