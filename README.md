# RELAY

Memory for an agent that outlives its context window.

---

## The one thing to understand

RELAY does not keep everything. Nothing that fits in a token budget can.

What it promises is narrower, and more useful: **nothing leaves quietly.**

A summary drops a constraint and says nothing. You find out in production.
RELAY drops a constraint and tells you it did, by name and by type, in the same
call that produced the handover:

```json
{
  "selected": 12,
  "tokens": 1980,
  "context": "## Context Atoms\n- [decision] Use RS256 for signing tokens\n- [constraint] Tokens expire after one hour\n...",
  "dropped": [
    {"type": "entity", "content": "staging cluster runs in eu-west-1"},
    {"type": "belief", "content": "the team prefers Postgres"}
  ]
}
```

That is the whole design. Everything else in this repo exists to make that
sentence true: typed atoms, so a type can be protected; a packer, so the budget
is spent on what matters; a ledger, so what was left out is still recorded.

---

## Why the obvious approaches fail

| Approach | What breaks |
|---|---|
| Replay the transcript | Cost grows with every turn, and the constraint from hour one is buried 40k tokens back |
| Summarise it | Unaccountable. The summary may drop "must use RS256" and nobody can tell |
| Retrieve top-k at each turn | Retrieves what is similar to *now*, so it cannot surface the constraint that nothing resembles yet |

The difference is not compression quality. It is whether you can answer the
question: *what did the handover leave behind?*

---

## What it does

1. **Typed atoms.** Seven types — decision, constraint, task, question, entity,
   belief, relation. A type is a policy hook: decisions and constraints are
   placed outside the optimiser, so they are never traded away for cheap filler.
2. **Packs under a token budget.** 0/1 knapsack over `value = salience × type
   weight` and `cost = tokens + 15`. Exact, not greedy: the returned set is the
   optimum for that budget, verified against brute force in the tests.
3. **Reports the loss.** Every atom that did not fit is returned and stored with
   the handover that dropped it.
4. **Measures drift.** Jaccard over the atoms that survived versus the previous
   handover, fused with a structural KL over atom types.

---

## What it does not do

Read this before the feature list.

- **It is not lossless.** The budget is finite, so atoms are always left out.
  The guarantee is that you are told which ones.
- **A dropped atom is not recoverable from RELAY.** It is recorded, not stored
  as text. Keep the transcript if you need the original wording.
- **It is not RAG.** There is no semantic search over history and no embedding
  index in the default path.
- **Deduplication is by canonical form, not meaning.** The same fact phrased two
  different ways becomes two atoms unless the extractor normalises it the same
  way twice.
- **The weights are hand-set priors**, not learned. They encode a belief about
  what matters in a handover. Fitting them is future work.
- **Value is unconditional.** The packer picks the most valuable atoms overall,
  not the ones relevant to what you are about to do next.
- **Extraction quality depends on the extractor.** With an API key it is a
  model; without one it is four regex patterns. `relay_remember` beats both when
  the agent already knows what it just decided.

---

## Install

```bash
git clone https://github.com/ramsterr/RELAY_context
cd RELAY_context
pip install -e .
```

Optional extras:

```bash
pip install -e ".[chroma]"         # persistent vector backend
pip install -e ".[qdrant]"         # production vector backend
pip install -e ".[observability]"  # langfuse + opentelemetry
```

The core install is enough for the plugin. It does not need a database.

---

## Use it as an agent plugin

RELAY ships an MCP server. Any MCP host can call it; these tools then appear to
the agent as `mcp__relay__<tool>`.

```bash
# Wombat
wombat mcp add relay -- relay-mcp

# check it
wombat mcp test relay
```

```text
✓ relay            8 tools
    relay_remember     Remember one fact that must survive this session.
    relay_capture      Extract atoms from raw transcript text.
    relay_atoms        List what is currently remembered.
    relay_handover     Pack the context atoms into a block for the next session.
    relay_drift        How far the context has moved since the last handover.
    relay_losses       What the last handover left behind, by name.
    relay_new_session  End this session and start the next one.
    relay_status       Where state lives and what is in it.
```

State lives in `RELAY_STATE_DIR` (default `~/.relay`), one file per workspace
namespace, so two projects never share memory. Point the server at a different
location with `wombat mcp add relay --env RELAY_STATE_DIR=/path -- relay-mcp`.

### The loop the agent should run

```text
start of session   → relay_handover            put the block in context
during the session → relay_remember            one call per decision/constraint
end of session     → relay_new_session         pack, archive, start clean
when curious       → relay_losses, relay_drift what did we drop, how far did we move
```

---

## Use it as a library

```python
from context_handover import ContextManager

manager = ContextManager(max_tokens=2000)
manager.add_message("Decision: we'll use RS256 for signing tokens.")
manager.add_message("Constraint: tokens expire after one hour.")

package = manager.build_handover_package()
print(package.to_context_string())
print(package.total_tokens)
```

The same object also exposes `compute_drift()`, `create_checkpoint(level)` and
`handover_to_new_session()`.

---

## Command line

The plugin is drivable without an agent, which is also how you debug it:

```bash
relay status
relay remember decision "Use RS256 for signing tokens"
relay atoms --kind constraint
relay handover --max-tokens 800
relay losses
relay drift
relay new-session
relay capture notes.md
relay mcp
```

`relay handover` prints the block on stdout and the accounting on stderr, so it
composes with a pipe.

---

## How it works

```text
messages ──▶ extract ──▶ atoms ──▶ pack under budget ──▶ context block
                            │                              │
                            │                              ├── dropped list
                            └────────── drift ◀────────────┘
```

Four modules, one direction:

| Module | Owns |
|---|---|
| `extraction/extraction.py` | text → typed candidates, LLM first, regex fallback |
| `core/atoms.py`, `core/registry.py` | the atom model, dedupe by canonical form |
| `core/budget.py` | 0/1 knapsack, mandatory atoms, `HandoverPackage` |
| `measurement/` | drift, loss ledger, integrity checks |
| `mcp/` | the store, the server, the tools |

`core/budget.py` imports nothing from the rest of the project, so the packer can
be read and tested on its own.

---

## Numbers worth knowing

- **Budget.** The token budget is the only real number. A 2,000-token handover
  carries about 74 average atoms, because each atom costs its own tokens plus
  15 for the `- [type] ` prefix. The reported total includes that overhead, so
  it is comparable to the budget.
- **Packing cost.** `O(n · W)` time and memory for the decision table, where `W`
  is the budget. At n=50, W=4096 that is 200k cells, about 30 ms in CPython.
- **Packing quality.** Before the reconstruction fix the packer captured 52%,
  35% and 20% of the achievable value at n=12/20/50 while filling only 45%, 27%
  and 12% of the budget. It now captures 100% and fills 95%, 98%, 93%. The
  remainder is granularity: atoms do not divide to fill a budget exactly.
- **Drift thresholds.** composite < 0.10 EXCELLENT, < 0.25 GOOD, < 0.45
  WARNING, above that CRITICAL.

---

## Tests

```bash
python -m pytest -q          # 73 passed
python scripts/repro_knapsack.py   # oracle check: 100% of optimum
python scripts/mcp_smoke.py        # spawns the server over stdio and drives it
```

The suite covers the atom model, dedupe, the packer against a brute-force
oracle, the store, and every MCP tool.

---

## Layout

```text
src/context_handover/
├── core/          atoms, registry, budget, checkpoints
├── extraction/    LLM extraction + regex fallback
├── measurement/   drift, ledger, verification
├── pipeline/      orchestration, retries, trace context
├── storage/       chroma / qdrant / memory backends
├── mcp/           store + server + tools   ← the plugin
└── cli.py         relay status | handover | losses | ...
```

MIT.
