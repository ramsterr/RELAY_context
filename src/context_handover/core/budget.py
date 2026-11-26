from __future__ import annotations
from typing import Optional, List, Tuple
import logging
logger = logging.getLogger(__name__)

#: Tokens each atom costs beyond its own text: the "- [type] " prefix and
#: the newline. Selection and reporting must use the same number, or the
#: reported total looks smaller than the budget it just spent.
ATOM_OVERHEAD = 15

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed — falling back to word estimate")


class TokenBudgetManager:
    """
    Manages token budgeting for context atoms using optimized selection algorithms.
    
    Supports both greedy selection (fast) and bounded knapsack optimization (optimal).
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.encoder = self._load_encoder(model_name) if TIKTOKEN_AVAILABLE else None

    def _load_encoder(self, model_name: str):
        try:
            try:
                return tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning(f"No encoder for {model_name}, using cl100k_base")
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None

    def count(self, text: str) -> int:
        """Count tokens in text using tiktoken or word-based estimation."""
        if self.encoder is None:
            return int(len(text.split()) * 1.4)
        return len(self.encoder.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens for a list of message dictionaries."""
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += 4
        total += 2
        return total

    def _compute_atom_value(self, atom) -> float:
        """
        Compute semantic value score for an atom.
        Combines propagation_score with type-based priority weighting.
        """
        from .atoms import AtomType
        
        # Base value from propagation score (0-1 range)
        base_value = atom.propagation_score
        
        # Type-based multipliers for critical context types
        # Higher weights for more semantically important atom types
        type_weights = {
            AtomType.DECISION: 1.5,      # Critical decisions
            AtomType.CONSTRAINT: 1.4,    # Hard constraints
            AtomType.TASK: 1.3,          # Action items
            AtomType.QUESTION: 1.2,      # Open questions
            AtomType.BELIEF: 1.1,        # Beliefs/assumptions
            AtomType.ENTITY: 1.0,        # Entities/facts
            AtomType.RELATION: 1.0,      # Relationships
        }
        
        weight = type_weights.get(atom.atom_type, 1.0)
        return base_value * weight

    def fit_atoms_to_budget(
        self,
        ranked_atoms: list,
        token_budget: int,
        overhead_per_atom: int = ATOM_OVERHEAD,
        use_knapsack: bool = True,
    ) -> list:
        """
        Select atoms to maximize semantic value within token budget.
        
        Args:
            ranked_atoms: List of candidate atoms
            token_budget: Maximum token limit
            overhead_per_atom: Token overhead per atom for formatting
            use_knapsack: If True, use 0/1 knapsack optimization; else greedy
            
        Returns:
            List of selected atoms optimized for value/token ratio
        """
        from .atoms import AtomType, AtomStatus
        
        if not ranked_atoms:
            return []
        
        # Separate mandatory atoms (always included)
        mandatory_atoms = [
            atom for atom in ranked_atoms
            if atom.atom_type in {AtomType.DECISION, AtomType.CONSTRAINT}
            and atom.status == AtomStatus.ACTIVE
        ]
        
        # Calculate cost of mandatory atoms
        mandatory_tokens = sum(
            self.count(atom.content) + overhead_per_atom
            for atom in mandatory_atoms
        )
        
        if mandatory_tokens > token_budget:
            logger.warning(
                f"Mandatory atoms ({mandatory_tokens} tokens) exceed budget ({token_budget}). "
                "Truncating mandatory set."
            )
            # Fallback: select most critical mandatory atoms only
            mandatory_atoms = sorted(
                mandatory_atoms,
                key=lambda a: self._compute_atom_value(a),
                reverse=True
            )
            selected = []
            tokens_used = 0
            for atom in mandatory_atoms:
                cost = self.count(atom.content) + overhead_per_atom
                if tokens_used + cost <= token_budget:
                    selected.append(atom)
                    tokens_used += cost
            return selected
        
        remaining_budget = token_budget - mandatory_tokens
        
        # Get optional atoms
        optional_atoms = [
            atom for atom in ranked_atoms
            if atom not in mandatory_atoms
            and atom.status == AtomStatus.ACTIVE
        ]
        
        if not optional_atoms:
            return mandatory_atoms
        
        if use_knapsack:
            optional_selected = self._knapsack_select(
                optional_atoms, remaining_budget, overhead_per_atom
            )
        else:
            optional_selected = self._greedy_select(
                optional_atoms, remaining_budget, overhead_per_atom
            )
        
        selected = mandatory_atoms + optional_selected
        
        total_tokens = sum(
            self.count(atom.content) + overhead_per_atom
            for atom in selected
        )
        
        logger.info(
            f"Budget optimization: {len(selected)}/{len(ranked_atoms)} atoms "
            f"in {total_tokens}/{token_budget} tokens "
            f"(knapsack={use_knapsack}, value={sum(self._compute_atom_value(a) for a in selected):.2f})"
        )
        
        return selected

    def _greedy_select(
        self,
        atoms: list,
        budget: int,
        overhead: int,
    ) -> list:
        """Greedy selection by value-per-token ratio."""
        selected = []
        remaining = budget
        
        # Sort by value-per-token ratio (descending)
        atoms_with_ratio = []
        for atom in atoms:
            cost = self.count(atom.content) + overhead
            if cost > 0:
                value = self._compute_atom_value(atom)
                ratio = value / cost
                atoms_with_ratio.append((atom, ratio, cost))
        
        atoms_with_ratio.sort(key=lambda x: x[1], reverse=True)
        
        for atom, ratio, cost in atoms_with_ratio:
            if cost <= remaining:
                selected.append(atom)
                remaining -= cost
        
        return selected

    def _knapsack_select(
        self,
        atoms: list,
        budget: int,
        overhead: int,
    ) -> list:
        """
        0/1 knapsack: each atom is taken or not, once. There are no copies of
        an item to bound, so this is the 0/1 formulation, not the bounded one.

        Time complexity: O(n * W) where n=atoms, W=budget.
        Space complexity: O(n * W) for the decision table, which is what makes
        the reconstruction correct.
        """
        if not atoms:
            return []

        # Precompute costs and values
        items = []
        for atom in atoms:
            cost = self.count(atom.content) + overhead
            if cost <= budget:  # Filter out atoms that exceed budget alone
                value = self._compute_atom_value(atom)
                items.append((atom, cost, value))

        if not items:
            return []

        # Cap the DP width so a multi-million-token budget does not allocate a
        # multi-million-entry table. Scaling down costs can admit a set that
        # overflows the real budget, so the selection is repaired below.
        dp_capacity = min(budget, 10000)
        scale = budget / dp_capacity
        scaled = [
            (atom, max(1, int(cost / scale)), value)
            for atom, cost, value in items
        ]

        selected = [scaled[i][0] for i in self._solve_01(scaled, dp_capacity)]
        if scale > 1.0:
            selected = self._repair_to_budget(selected, items, budget)
        return selected

    @staticmethod
    def _solve_01(items: list, capacity: int) -> list:
        """Exact 0/1 knapsack selection, returning item indices.

        Reconstruction reads a decision table, one row per item. A single
        parent pointer per capacity state is not enough: every later item that
        improves the same state overwrites it, so the parent chain walks into a
        state it never visited and terminates early. The result is feasible and
        much worse than the optimum, which is why nothing crashed.
        """
        n = len(items)
        dp = [0.0] * (capacity + 1)
        keep = [bytearray(capacity + 1) for _ in range(n)]

        for i, (_, cost, value) in enumerate(items):
            taken = keep[i]
            for w in range(capacity, cost - 1, -1):
                candidate = dp[w - cost] + value
                if candidate > dp[w]:
                    dp[w] = candidate
                    taken[w] = 1

        selected = []
        w = capacity
        for i in range(n - 1, -1, -1):
            if keep[i][w]:
                selected.append(i)
                w -= items[i][1]
        selected.reverse()
        return selected

    @staticmethod
    def _repair_to_budget(selected: list, items: list, budget: int) -> list:
        """Make a scaled-cost selection fit the real budget, then refill it."""
        cost_of = {id(atom): cost for atom, cost, _ in items}
        value_of = {id(atom): value for atom, _, value in items}

        chosen = sorted(
            selected,
            key=lambda atom: value_of[id(atom)] / cost_of[id(atom)],
            reverse=True,
        )
        total = sum(cost_of[id(atom)] for atom in chosen)
        while chosen and total > budget:
            total -= cost_of[id(chosen.pop())]

        picked = {id(atom) for atom in chosen}
        for atom, cost, value in sorted(
            items, key=lambda item: item[2] / item[1], reverse=True
        ):
            if id(atom) in picked or total + cost > budget:
                continue
            chosen.append(atom)
            picked.add(id(atom))
            total += cost
        return chosen


class HandoverPackage:
    def __init__(
        self,
        atoms: list,
        budget_manager: TokenBudgetManager,
        max_tokens: int = 8000,
    ):
        self.atoms = atoms
        self.budget_manager = budget_manager
        self.max_tokens = max_tokens
        self.selected_atoms: list = []
        #: Tokens the handover actually spends, including the per-atom prefix
        #: overhead. This is the number to compare against ``max_tokens``.
        self.total_tokens: int = 0
        #: Content tokens only. Smaller than ``total_tokens`` by exactly
        #: ``overhead_tokens``; kept so the two are never conflated.
        self.text_tokens: int = 0
        #: The difference between the two numbers above.
        self.overhead_tokens: int = 0

    def build(self, use_knapsack: bool = True) -> list:
        """
        Build the handover package with optimized atom selection.
        
        Args:
            use_knapsack: If True (default), use 0/1 knapsack optimization.
                         If False, use faster greedy selection.
        """
        ranked = sorted(self.atoms, key=lambda a: a.propagation_score, reverse=True)
        self.selected_atoms = self.budget_manager.fit_atoms_to_budget(
            ranked, self.max_tokens, use_knapsack=use_knapsack
        )
        self.text_tokens = sum(
            self.budget_manager.count(a.content) for a in self.selected_atoms
        )
        self.overhead_tokens = ATOM_OVERHEAD * len(self.selected_atoms)
        self.total_tokens = self.text_tokens + self.overhead_tokens
        return self.selected_atoms

    def to_context_string(self) -> str:
        if not self.selected_atoms:
            return "No context atoms available."
        lines = ["## Context Atoms\n"]
        for atom in self.selected_atoms:
            lines.append(f"- [{atom.atom_type.value}] {atom.content}")
        return "\n".join(lines)