#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.test_core import (
    test_connected_mode_contains_valid_service_sets,
    test_exact_small_cycle,
    test_generated_graph_and_schedules_are_valid,
    test_failed_nodes_are_removed_from_service_universe,
    test_modern_reproductions_return_disjoint_dominating_families,
    test_first_order_radio_cost_does_not_double_count_tx_electronics,
    test_static_families_are_disjoint_and_dominating,
)


def main():
    checks = [
        test_generated_graph_and_schedules_are_valid,
        test_connected_mode_contains_valid_service_sets,
        test_static_families_are_disjoint_and_dominating,
        test_exact_small_cycle,
        test_modern_reproductions_return_disjoint_dominating_families,
        test_failed_nodes_are_removed_from_service_universe,
        test_first_order_radio_cost_does_not_double_count_tx_electronics,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)


if __name__ == "__main__":
    main()
