# Baseline reproduction audit

This record prevents paper-based implementations from being mistaken for the
baseline authors' executable code.

| Label | Source | Status | Main reproduced mechanisms | Deliberate limits |
|---|---|---|---|---|
| PBIG-R | Bouamama, Blum et al., *Sensors* (2022) | Paper-based reproduction | Population, weighted randomized construction, partial destruction, adaptive controls, restart | Reduced population under an explicit time budget; no indexed author code |
| MC-CMSA-R | Rosati, Bouamama and Blum, *Computers & Operations Research* (2024) | Paper-based reconstruction | Six construction scores, merge, exact reduced MILP, component ageing | Candidate dominating sets are reduced-subproblem components; not bit-for-bit equivalent to the assignment-pair implementation |
| FSS-2026-R | Jovanovic and Voss, *Journal of Heuristics* (2026) | Paper-based reproduction | Greedy family construction, elite archive, similarity-fixed cores, adaptive fixed portions, improving swaps, extended local search | Time-bounded neighborhoods, 30 rather than 100 initial GRASP solutions, equal 0.5 s budget |
| EAAS-S4C-MAB-R | Salim et al., *Scientific Reports* (2026) | Paper-based reproduction | Energy-ordered deletion, lifetime/size skyline, tuned selector, four cases, tuned UCB, residual-set validation, conservative fallback | No disclosed author code; common radio model used for all methods |

Validation checks pairwise disjointness for static families, domination for
every family member, common instances, recorded stochastic seeds and budgets,
and retained runtime/validity columns.

The manuscript and repository may say “paper-based reproduction” or “validated
reconstruction.” They must not say “authors' implementation,” “official code,”
or “exact reproduction” unless the authors' executable code is later obtained
and the experiment is rerun.
