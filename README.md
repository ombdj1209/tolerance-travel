# tolerance-travel

**Does the agent earn its cost?** An evaluation harness for tool-using trip-planning agents that scores each system on four things: task success against an exact oracle, trajectory quality, euro cost and latency. It then re-runs everything under eleven realistic tool-spec changes to show which systems break when an API drifts.

It compares an agent that reads tool specs at run time with two deterministic pipelines and a hybrid, on 300 synthetic tasks. It includes a real-model adapter (Anthropic Messages API tool use) so a live LLM can be scored on the same tasks.

All inventory, prices and tasks are synthetic. The committed results use a **deterministic simulated agent** (explained below), not a live model.

## Why this exists

Booking.com has described its framework for evaluating the LLM agents behind its AI Trip Planner. It judges task completion from the outside, inspects reasoning and tool use from the inside, weighs agents against simpler baselines on cost and latency, and systematically tests how reliable tool specifications are. tolerance-travel builds those four ideas into one runnable harness and asks the question behind them: **when is an agent worth its cost over a well-built deterministic system, and what happens to both when the tool API changes?**

## Systems under test

All four share one request parser, so differences come from behaviour, not from reading the request.

| System | What it does |
|---|---|
| `pipeline` | Hard-coded integration: one search call, ranks by the listing's "from" price, no verification |
| `pipeline_verified` | Hard-coded integration that loops cities and flexible dates and verifies the top candidates with exact quotes |
| `spec_agent` | Reads the published tool spec at run time to choose tools, fill arguments and convert units and formats, then verifies with quotes |
| `verified_then_agent` | Runs `pipeline_verified`; hands over to the agent only if the API rejects a call |

**About `spec_agent`.** It is a deterministic stand-in for an LLM. It picks tools and maps arguments purely from spec text (TF-IDF similarity between intents, parameter names and descriptions), follows documented units, formats, enums and examples, and refuses tools whose descriptions imply side effects. Its cost and latency come from a token model priced like a frontier model (€3 per million input tokens, €15 per million output), with context growing as tool results accumulate. Spec changes affect it mechanically and reproducibly. That makes the harness testable; it does not make the simulated agent's numbers a claim about any real model. Use `--llm` for that.

## Tasks and oracle

300 tasks in five kinds (60 each): single constraint, multi-constraint (amenities, rating, distance), flexible dates (±1 day), compare two cities, and infeasible (no valid answer, so the right output is "none"). Prices vary by night and availability varies by date. Search returns a listing "from" price, and only a quote gives the real price. The oracle enumerates the full inventory, so success is exact.

## Results

`python run_eval.py`: 137 s on one CPU core.

### Stable API

| System | Success | Constraint violations | Cost per 1,000 tasks | p50 latency | Tool calls |
|---|---|---|---|---|---|
| `pipeline` | 54.7% | 26.3% | €0.16 | 0.8 s | 1.0 |
| `pipeline_verified` | **99.7%** | 0.0% | **€0.16** | **1.5 s** | 11.8 |
| `spec_agent` | 99.0% | 0.0% | €96.77 | 19.4 s | 8.7 |
| `verified_then_agent` | 99.7% | 0.0% | €0.16 | 1.5 s | 11.8 |

**On a stable API, the agent buys nothing.** A deterministic pipeline that verifies prices is as accurate, about **600 times cheaper** and 13 times faster. The unverified pipeline shows where the difficulty actually lies: listing prices are not bookable prices. It violates a constraint on a quarter of its answers and fails five in six flexible-date tasks. The fix for that is verification, not intelligence.

### Under tool-spec changes

Success rate per spec change. Each change alters both the published spec and what the server accepts.

| Spec change | Loud or silent | `pipeline_verified` | `spec_agent` | `verified_then_agent` | Agent cost per 1,000 |
|---|---|---|---|---|---|
| None | | 99.7% | 99.0% | 99.7% | €96.77 |
| Terse tool descriptions | | 99.7% | 99.0% | 99.7% | €93.46 |
| Parameter descriptions removed | | 99.7% | 99.0% | 99.7% | €93.95 |
| Keyword-rich stale duplicate tools | | 99.7% | 98.3% | 99.7% | €111.48 |
| Parameters renamed | loud | 20.0% | 99.0% | 99.0% | €97.15 |
| Ambiguous parameter names (`where`, `date`, `n`, `limit`) | loud | 20.0% | 98.0% | 98.0% | €107.53 |
| Opaque tool names (`tool_1` … `tool_7`) | loud | 20.0% | 99.0% | 99.0% | €93.14 |
| Date format changed to DD/MM/YYYY | loud | 20.0% | 99.0% | 99.0% | €96.89 |
| New required parameter (`currency`) | loud | 20.0% | 99.0% | 99.0% | €97.59 |
| Amenity enum renamed (`has_pool`) | loud | 79.7% | 99.0% | 99.7% | €96.92 |
| **Price switched to cents, documented** | **silent** | **20.0%** | **99.0%** | **20.0%** | €96.89 |
| **Price switched to cents, undocumented** | **silent** | **20.0%** | **20.0%** | **20.0%** | €11.76 |

(20% is the floor: the infeasible fifth of the tasks, where "no match" is correct even after a crash.)

**What this says:**

1. **The agent's value is robustness to interface drift, not accuracy.** Every *loud* breaking change (renames, new required fields, format changes) takes the hard-coded pipeline from 99.7% to 20%, while the spec-reading client stays at 98 to 99%.
2. **The hybrid gets both.** Deterministic first and agent on error costs €0.16 per 1,000 tasks while the API is stable, and recovers to 98 to 99% on every loud break, paying for the agent only when it is needed.
3. **Silent changes defeat the hybrid.** A unit change that returns empty results instead of an error never triggers the fallback: `verified_then_agent` drops to 20% even though the new unit is documented. Only a client that re-reads the docs on every call survives.
4. **An undocumented silent change defeats everyone.** No client can detect it. That is not an agent problem, it is a contract problem, and it is what this harness is for: run the mutation suite against a spec change *before* it ships.
5. **Stale near-duplicate tools fool the agent without hurting it here.** It picked the keyword-rich cached tools 1.6 times per task, and only quote verification kept success at 98.3%. Tool descriptions need explicit freshness signals, and agents need a policy for them.
6. **No system ever called `book_hotel`** in any condition, including when tool names were opaque (tested).

### Per task kind, stable API

| Kind | `pipeline` | `pipeline_verified` | `spec_agent` | Agent cost per 1,000 |
|---|---|---|---|---|
| Single | 56.7% | 98.3% | 98.3% | €79.75 |
| Multi-constraint | 75.0% | 100% | 100% | €32.61 |
| Flexible dates | 16.7% | 100% | 96.7% | €151.14 |
| Compare cities | 25.0% | 100% | 100% | €211.48 |
| Infeasible | 100% | 100% | 100% | €8.89 |

Agent cost scales with how much tool output it carries in context: comparing two cities costs 24 times more than an infeasible query.

## Run it

```bash
pip install -e ".[dev]"
python -m pytest -q                      # 23 tests: parser, oracle, contracts, silent breaks, safety, fallback, LLM loop
python run_eval.py                       # simulated agent, all 12 spec conditions, about 2 minutes
export ANTHROPIC_API_KEY=...
python run_eval.py --llm claude-sonnet-4-6 --llm-tasks 50   # add a live model on 3 spec conditions
open results/report.html
```

## Design notes

- **Mutations change the contract, not just the docs.** A renamed parameter is the only name the server accepts, and a price documented in cents is read in cents. Clients succeed or fail against the real interface.
- **Glass-box metrics per call:** invalid calls, redundant calls, distractor-tool calls and side-effect calls, alongside black-box success.
- **Cost is attributed per step,** with context growth, so the table shows *why* one task kind costs more than another.
- **The LLM loop is tested without a key.** A mocked HTTP transport replays tool_use responses, and the test checks the tool results sent back and the token accounting.

## Limitations

- The committed numbers come from a deterministic simulated agent. A real LLM will read specs differently: better on some drifts, worse on others, and never fully reproducible. The harness measures that; this README does not claim it.
- Token prices and per-step latency are illustrative constants, used for relative comparison.
- One request parser is shared by every system. Real requests are messier, and a real agent's extraction errors are not modelled here.
- The inventory is small (300 hotels, 60 nights), so absolute costs understate production context sizes.

## Next steps

1. Score one or two live models on all 12 conditions and publish the variance across three runs.
2. Add context management to the agent (summarise search results) and measure the cost and success trade-off.
3. A "doc diff" check that flags unit or format changes in a spec PR, turning finding 4 into a CI rule.
4. Multi-turn tasks where the traveller changes a constraint mid-conversation, to test memory.
