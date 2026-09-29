"""Evaluation harness: black-box outcome, glass-box trajectory, cost and latency, per system,
per task kind and per spec mutation."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .agents import SpecAgent, Trace, pipeline
from .catalog import Catalog, Task, candidates, parse
from .toolserver import ToolServer, mutated_spec

def fallback(text, srv) -> Trace:
    """Deterministic pipeline first; hand over to the agent only when the API rejects a call."""
    a = pipeline(text, srv, verified=True)
    if not any(not s["ok"] for s in a.steps):
        return a
    b = SpecAgent().run(text, srv)
    b.steps = a.steps + b.steps
    b.tokens_in += a.tokens_in
    b.tokens_out += a.tokens_out
    b.cost_eur += a.cost_eur
    b.latency_s += a.latency_s
    return b


SYSTEMS = {
    "pipeline": lambda text, srv: pipeline(text, srv, verified=False),
    "pipeline_verified": lambda text, srv: pipeline(text, srv, verified=True),
    "spec_agent": lambda text, srv: SpecAgent().run(text, srv),
    "verified_then_agent": fallback,
}


def judge(cat: Catalog, task: Task, tr) -> dict:
    if tr.answer is None:
        success = not task.answer
        violation = False
    else:
        success = tr.answer in task.answer
        feasible = {(h.id, ci) for h, ci, _ in candidates(cat, task.slots)}
        violation = tr.answer not in feasible          # answered with something that breaks a constraint
    calls = [(s["tool"], tuple(sorted((k, str(v)) for k, v in s["args"].items()))) for s in tr.steps]
    return {"success": success, "constraint_violation": violation, "abstained": tr.answer is None,
            "tool_calls": len(tr.steps), "invalid_calls": sum(not s["ok"] for s in tr.steps),
            "redundant_calls": len(calls) - len(set(calls)),
            "distractor_calls": sum(s["tool"] in ("search_hotels_fast", "estimate_price") for s in tr.steps),
            "side_effect_calls": sum(s["tool"] == "book_hotel" for s in tr.steps),
            "tokens_in": tr.tokens_in, "cost_eur": tr.cost_eur, "latency_s": tr.latency_s}


def run(cat: Catalog, tasks, systems=SYSTEMS, mutations=("none",)) -> list[dict]:
    rows = []
    for mut in mutations:
        spec = mutated_spec(mut)
        for name, fn in systems.items():
            for t in tasks:
                srv = ToolServer(cat, spec)
                tr = fn(t.text, srv)
                rows.append({"system": name, "mutation": mut, "task": t.id, "kind": t.kind, **judge(cat, t, tr),
                             "bookings_made": len(srv.bookings)})
    return rows


def aggregate(rows, keys=("system",)) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in sorted(g.items()):
        succ = np.array([r["success"] for r in rs], dtype=float)
        cost = np.array([r["cost_eur"] for r in rs])
        lat = np.array([r["latency_s"] for r in rs])
        out["/".join(k)] = {
            "n": len(rs), "success": float(succ.mean()),
            "success_ci95": float(1.96 * succ.std(ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0,
            "constraint_violation": float(np.mean([r["constraint_violation"] for r in rs])),
            "cost_eur_per_task": float(cost.mean()),
            "cost_eur_per_success": float(cost.sum() / max(1, succ.sum())),
            "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95)),
            "tool_calls": float(np.mean([r["tool_calls"] for r in rs])),
            "invalid_calls": float(np.mean([r["invalid_calls"] for r in rs])),
            "redundant_calls": float(np.mean([r["redundant_calls"] for r in rs])),
            "distractor_calls": float(np.mean([r["distractor_calls"] for r in rs])),
            "side_effect_calls": int(sum(r["side_effect_calls"] for r in rs)),
        }
    return out
