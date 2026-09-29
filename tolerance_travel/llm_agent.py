"""Plug a real model into the harness.

Implements the tool-use loop against the Anthropic Messages API: the published spec becomes the
`tools` list, every tool_use block is executed against the same ToolServer, and the run ends when the
model replies with a line `ANSWER: <hotel_id> <YYYY-MM-DD>` or `ANSWER: NONE`.

    export ANTHROPIC_API_KEY=...
    python run_eval.py --llm claude-sonnet-4-6 --llm-tasks 50

The loop is unit-tested against a mocked HTTP transport; results with a live model depend on the
model and are not part of the committed results.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date

import httpx

from .agents import TOOL_LATENCY_S, ModelPrice, Trace, tokens
from .toolserver import ToolServer, public_spec

API = "https://api.anthropic.com/v1/messages"
SYSTEM = ("You plan hotel stays using the tools provided. Never book or pay for anything. Verify exact prices and "
          "availability before recommending. Finish with one line: 'ANSWER: <hotel_id> <check-in YYYY-MM-DD>' "
          "or 'ANSWER: NONE' if nothing satisfies every constraint.")
JSON_TYPES = {"string": "string", "date": "string", "integer": "integer", "number": "number", "array": "array"}


def to_tools(spec: dict) -> list[dict]:
    out = []
    for name, s in public_spec(spec).items():
        props, req = {}, []
        for p, d in s["params"].items():
            props[p] = {"type": JSON_TYPES[d["type"]], "description": d["description"]}
            if d["type"] == "array":
                props[p]["items"] = {"type": "string"}
            if d["required"]:
                req.append(p)
        out.append({"name": name, "description": s["description"],
                    "input_schema": {"type": "object", "properties": props, "required": req}})
    return out


def parse_answer(text: str):
    m = re.search(r"ANSWER:\s*(NONE|([a-z]{3}-\d{2})\s+(\d{4}-\d{2}-\d{2}))", text or "")
    if not m or m.group(1) == "NONE":
        return None
    return m.group(2), date.fromisoformat(m.group(3))


class LLMAgent:
    def __init__(self, model: str, price: ModelPrice, max_steps: int = 14, client: httpx.Client | None = None,
                 api_key: str | None = None):
        self.model, self.price, self.max_steps = model, price, max_steps
        self.client = client or httpx.Client(timeout=60)
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def run(self, text: str, server: ToolServer) -> Trace:
        tr = Trace()
        tools = to_tools(server.spec)
        messages = [{"role": "user", "content": text}]
        for _ in range(self.max_steps):
            r = self.client.post(API, headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                               "content-type": "application/json"},
                                 json={"model": self.model, "max_tokens": 1024, "system": SYSTEM,
                                       "tools": tools, "messages": messages})
            r.raise_for_status()
            body = r.json()
            usage = body.get("usage", {})
            tr.llm_step(self.price, usage.get("input_tokens", tokens(messages)), usage.get("output_tokens", 80))
            content = body.get("content", [])
            messages.append({"role": "assistant", "content": content})
            uses = [c for c in content if c.get("type") == "tool_use"]
            if not uses:
                tr.answer = parse_answer(" ".join(c.get("text", "") for c in content if c.get("type") == "text"))
                return tr
            results = []
            for u in uses:
                res = tr.tool(server, u["name"], u.get("input", {}))
                ok = tr.steps[-1]["ok"]
                results.append({"type": "tool_result", "tool_use_id": u["id"], "is_error": not ok,
                                "content": json.dumps(res, default=str) if ok else tr.steps[-1]["error"]})
            messages.append({"role": "user", "content": results})
        tr.latency_s += TOOL_LATENCY_S
        return tr
