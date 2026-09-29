"""Reproduces every number in the README.

    python run_eval.py                                  # simulated agent, all mutations
    python run_eval.py --quick
    python run_eval.py --llm claude-sonnet-4-6 --llm-tasks 50   # add a real model (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from tolerance_travel.agents import AGENT_MODEL
from tolerance_travel.catalog import Catalog, make_tasks
from tolerance_travel.harness import SYSTEMS, aggregate, run
from tolerance_travel.report import write_report
from tolerance_travel.toolserver import MUTATIONS

OUT = Path(__file__).resolve().parent / "results"


def main(a):
    t0 = time.time()
    cat = Catalog(seed=0)
    tasks = make_tasks(cat, 100 if a.quick else 300, seed=1)
    systems = dict(SYSTEMS)
    if a.llm:
        from tolerance_travel.llm_agent import LLMAgent
        agent = LLMAgent(a.llm, AGENT_MODEL)
        systems["llm:" + a.llm] = agent.run
    rows = run(cat, tasks, systems={k: v for k, v in systems.items() if not k.startswith("llm:")},
               mutations=tuple(MUTATIONS))
    if a.llm:
        rows += run(cat, tasks[: a.llm_tasks], systems={k: v for k, v in systems.items() if k.startswith("llm:")},
                    mutations=("none", "rename_params", "undocumented_cents"))
    res = {"environment": {"python": platform.python_version(), "machine": platform.machine(),
                           "runtime_s": round(time.time() - t0, 1), "quick": a.quick, "llm": a.llm},
           "tasks": {"n": len(tasks), "by_kind": {k: sum(t.kind == k for t in tasks) for k in sorted({t.kind for t in tasks})}},
           "by_system": aggregate([r for r in rows if r["mutation"] == "none"], ("system",)),
           "by_kind": aggregate([r for r in rows if r["mutation"] == "none"], ("kind", "system")),
           "by_mutation": aggregate(rows, ("mutation", "system")),
           "mutations": list(MUTATIONS)}
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    write_report(res, OUT / "report.html")
    for k, v in res["by_mutation"].items():
        print(f"{k:45s} success {v['success']:.3f}  €/1k {1000 * v['cost_eur_per_task']:8.2f}  "
              f"p50 {v['latency_p50_s']:5.1f}s  invalid {v['invalid_calls']:.2f}  distractor {v['distractor_calls']:.2f}  "
              f"side-effects {v['side_effect_calls']}")
    print(f"done in {time.time() - t0:.0f}s -> results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--llm-tasks", type=int, default=50)
    main(ap.parse_args())
