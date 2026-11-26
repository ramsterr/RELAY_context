"""Properties the packer claims, asserted against an exhaustive oracle.

The bug this file exists for: the DP value was correct while the reconstructed
set was not, so every feasibility assertion passed. These tests check the
*optimum*, not the budget.
"""
from __future__ import annotations

import itertools
import random

import pytest

from context_handover.core.atoms import AtomType, SemanticAtom
from context_handover.core.budget import TokenBudgetManager

OVERHEAD = 15
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
OPTIONAL_TYPES = [
    AtomType.ENTITY,
    AtomType.TASK,
    AtomType.QUESTION,
    AtomType.RELATION,
    AtomType.BELIEF,
]


def make_atoms(n: int, seed: int, types=None) -> list[SemanticAtom]:
    rng = random.Random(seed)
    types = types or OPTIONAL_TYPES
    atoms = []
    for i in range(n):
        content = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 6)))
        atoms.append(
            SemanticAtom(
                atom_id=f"a{i}",
                atom_type=rng.choice(types),
                content=content,
                canonical_form=content,
                salience=rng.uniform(0.05, 1.0),
            )
        )
    return atoms


def cost_of(atom, mgr) -> int:
    return mgr.count(atom.content) + OVERHEAD


def value_of(atoms, mgr) -> float:
    return sum(mgr._compute_atom_value(a) for a in atoms)


def brute_force(atoms, mgr, budget) -> float:
    best = 0.0
    for r in range(len(atoms) + 1):
        for combo in itertools.combinations(atoms, r):
            if sum(cost_of(a, mgr) for a in combo) <= budget:
                best = max(best, value_of(combo, mgr))
    return best


@pytest.fixture(scope="module")
def mgr() -> TokenBudgetManager:
    return TokenBudgetManager()


@pytest.mark.parametrize("seed", range(6))
def test_selection_equals_brute_force_optimum(mgr, seed):
    atoms = make_atoms(10, seed)
    budget = 120
    chosen = mgr.fit_atoms_to_budget(atoms, token_budget=budget, use_knapsack=True)

    assert sum(cost_of(a, mgr) for a in chosen) <= budget
    assert value_of(chosen, mgr) == pytest.approx(brute_force(atoms, mgr, budget), abs=1e-9)


@pytest.mark.parametrize("n,budget", [(20, 300), (50, 1024), (80, 2000)])
def test_selection_keeps_up_with_the_dp_optimum(mgr, n, budget):
    atoms = make_atoms(n, seed=n)
    chosen = mgr.fit_atoms_to_budget(atoms, token_budget=budget, use_knapsack=True)

    dp = [0.0] * (budget + 1)
    for atom in atoms:
        cost = cost_of(atom, mgr)
        if cost > budget:
            continue
        value = mgr._compute_atom_value(atom)
        for w in range(budget, cost - 1, -1):
            dp[w] = max(dp[w], dp[w - cost] + value)

    assert sum(cost_of(a, mgr) for a in chosen) <= budget
    assert value_of(chosen, mgr) == pytest.approx(dp[budget], abs=1e-9)


def test_large_budget_stays_within_the_real_budget(mgr):
    """The scaled-cost path must not overspend after costs are divided down."""
    atoms = make_atoms(60, seed=7)
    budget = 50_000
    chosen = mgr.fit_atoms_to_budget(atoms, token_budget=budget, use_knapsack=True)

    assert sum(cost_of(a, mgr) for a in chosen) <= budget


def test_mandatory_atoms_survive_optimisation(mgr):
    atoms = make_atoms(8, seed=3)
    forced = SemanticAtom(
        atom_id="c0",
        atom_type=AtomType.CONSTRAINT,
        content="must use RS256",
        canonical_form="must use rs256",
        salience=0.01,
    )
    chosen = mgr.fit_atoms_to_budget(atoms + [forced], token_budget=400, use_knapsack=True)
    assert forced in chosen


def test_budget_too_small_for_any_atom_returns_empty(mgr):
    atoms = make_atoms(5, seed=1)
    assert mgr.fit_atoms_to_budget(atoms, token_budget=1, use_knapsack=True) == []
