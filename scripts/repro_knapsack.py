#!/usr/bin/env python3
"""Compare TokenBudgetManager's knapsack selection against an exhaustive oracle.

The DP value (`dp[budget]`) can be correct while the reconstructed *set* is
wrong. This script checks the thing that ships: the returned atoms.

For n <= 12 every subset is enumerated, so the oracle is exact. For larger n
the DP's own optimal value is the reference, which is enough to expose a
backtrack that walks off the end of its parent chain.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_handover.core.atoms import AtomType, SemanticAtom
from context_handover.core.budget import TokenBudgetManager

OVERHEAD = 15
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def make_atoms(n: int, rng: random.Random, types=None):
    types = types or [AtomType.ENTITY, AtomType.TASK, AtomType.QUESTION, AtomType.RELATION]
    atoms = []
    for i in range(n):
        words = rng.randint(1, 6)
        content = " ".join(rng.choice(WORDS) for _ in range(words))
        atom = SemanticAtom(
            atom_id=f"a{i}",
            atom_type=rng.choice(types),
            content=content,
            canonical_form=content,
            salience=rng.uniform(0.05, 1.0),
        )
        atoms.append(atom)
    return atoms


def value_of(atoms, mgr):
    return sum(mgr._compute_atom_value(a) for a in atoms)


def best_subset(atoms, mgr, budget):
    best_v, best_set = 0.0, ()
    for r in range(len(atoms) + 1):
        for combo in itertools.combinations(atoms, r):
            cost = sum(mgr.count(a.content) + OVERHEAD for a in combo)
            if cost <= budget:
                v = value_of(combo, mgr)
                if v > best_v:
                    best_v, best_set = v, combo
    return best_v, best_set


def dp_optimum(atoms, mgr, budget):
    dp = [0.0] * (budget + 1)
    for atom in atoms:
        cost = mgr.count(atom.content) + OVERHEAD
        if cost > budget:
            continue
        val = mgr._compute_atom_value(atom)
        for w in range(budget, cost - 1, -1):
            if dp[w - cost] + val > dp[w]:
                dp[w] = dp[w - cost] + val
    return dp[budget]


def run(shape, trials, exact_oracle):
    n, budget = shape
    mgr = TokenBudgetManager()
    rng = random.Random(0)
    ratios, fills = [], []
    worst = None
    for _ in range(trials):
        atoms = make_atoms(n, rng)
        chosen = mgr.fit_atoms_to_budget(atoms, token_budget=budget, use_knapsack=True)
        got = value_of(chosen, mgr)
        if exact_oracle and n <= 12:
            ref = best_subset(atoms, mgr, budget)[0]
        else:
            ref = dp_optimum(atoms, mgr, budget)
        ratio = got / ref if ref > 0 else 1.0
        used = sum(mgr.count(a.content) + OVERHEAD for a in chosen)
        ratios.append(ratio)
        fills.append(used / budget)
        if worst is None or ratio < worst[0]:
            worst = (ratio, n, budget, len(chosen), got, ref)
    avg = sum(ratios) / len(ratios)
    return avg, sum(fills) / len(fills), worst


def main() -> int:
    print(f"{'shape':>14} {'value/oracle':>13} {'budget filled':>14}")
    print("-" * 45)
    rows = [((12, 120), 200), ((20, 300), 40), ((50, 1024), 10)]
    for shape, trials in rows:
        avg, fill, worst = run(shape, trials, exact_oracle=True)
        print(f"n={shape[0]:>3} W={shape[1]:<5} {avg:>12.1%} {fill:>13.1%}")
    print()
    print("Worst instance:", f"n={worst[1]} W={worst[2]} selected={worst[3]} "
                            f"value={worst[4]:.3f}/{worst[5]:.3f} = {worst[0]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
