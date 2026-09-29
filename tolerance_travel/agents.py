"""Three systems under test, all sharing one request parser:

  pipeline           hard-coded integration: one search call, rank by listing price, no verification
  pipeline_verified  hard-coded integration that loops cities and dates and verifies with quotes
  spec_agent         reads the published tool spec at run time to choose tools and fill arguments

spec_agent is a deterministic stand-in for an LLM tool-using agent: it selects tools and maps
arguments purely from spec text (TF-IDF similarity), so spec mutations affect it the way unclear or
changed documentation affects a model, mechanically and reproducibly. `llm_agent.py` plugs a real
model into the same harness.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalog import AMENITIES, Slots, parse
from .toolserver import ToolCallError, ToolServer, public_spec

SIDE_EFFECT_WORDS = re.compile(r"\b(book|booking|pay|charge|purchase|reserve)\b", re.I)


# --------------------------------------------------------------------------- cost model
@dataclass(frozen=True)
class ModelPrice:
    name: str
    eur_per_m_in: float
    eur_per_m_out: float
    s_per_step: float
    s_per_1k_in: float
    s_per_out: float


AGENT_MODEL = ModelPrice("frontier-agent", 3.0, 15.0, 0.6, 0.03, 0.015)
SMALL_MODEL = ModelPrice("small-extractor", 0.25, 1.25, 0.25, 0.01, 0.008)
TOOL_LATENCY_S = 0.08


def tokens(x) -> int:
    return max(1, len(x if isinstance(x, str) else json.dumps(x, default=str)) // 4)


@dataclass
class Trace:
    steps: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_eur: float = 0.0
    answer: tuple | None = None          # (hotel_id, check_in) or None for "no match"

    def llm_step(self, m: ModelPrice, tin: int, tout: int):
        self.tokens_in += tin
        self.tokens_out += tout
        self.cost_eur += (tin * m.eur_per_m_in + tout * m.eur_per_m_out) / 1e6
        self.latency_s += m.s_per_step + m.s_per_1k_in * tin / 1000 + m.s_per_out * tout

    def tool(self, server: ToolServer, name: str, args: dict):
        self.latency_s += TOOL_LATENCY_S
        try:
            res = server.call(name, args)
            self.steps.append({"tool": name, "args": args, "ok": True, "result_tokens": tokens(res)})
            return res
        except ToolCallError as e:
            self.steps.append({"tool": name, "args": args, "ok": False, "error": str(e), "result_tokens": tokens(str(e))})
            return None


def _rank_key(s: Slots, rating, dist, price):
    return {"cheapest": price, "best_rated": -rating, "closest": dist}[s.objective]


# ------------------------------------------------------------------------------ pipelines
def pipeline(text: str, server: ToolServer, verified: bool = False) -> Trace:
    tr = Trace()
    tr.llm_step(SMALL_MODEL, 350, 60)                                  # one structured-extraction call
    s = parse(text)
    cities = s.cities if verified else s.cities[:1]
    offsets = range(-s.flex_days, s.flex_days + 1) if verified else [0]
    best = None
    for city in cities:
        for off in offsets:
            ci = s.check_in + timedelta(days=off)
            args = {"city": city, "check_in": ci.isoformat(), "nights": s.nights}
            if s.max_per_night:
                args["max_price"] = s.max_per_night
            if s.amenities:
                args["amenities"] = list(s.amenities)
            res = tr.tool(server, "search_hotels", args)
            if res is None:
                continue
            listing = [h for h in res if (s.min_rating is None or h["rating"] >= s.min_rating)
                       and (s.max_distance is None or h["distance_km"] <= s.max_distance)]
            listing.sort(key=lambda h: _rank_key(s, h["rating"], h["distance_km"], h["from_price_eur"]))
            for h in listing[: (8 if verified else 1)]:
                price = h["from_price_eur"] * s.nights
                if verified:
                    q = tr.tool(server, "get_quote", {"hotel_id": h["hotel_id"], "check_in": ci.isoformat(), "nights": s.nights})
                    if not q or not q["available"] or (s.max_per_night and q["per_night_eur"] > s.max_per_night):
                        continue
                    price = q["total_eur"]
                key = _rank_key(s, h["rating"], h["distance_km"], price)
                if best is None or key < best[0]:
                    best = (key, (h["hotel_id"], ci))
                if not verified:
                    break
    tr.answer = best[1] if best else None
    return tr


# ----------------------------------------------------------------------------- spec agent
SLOT_TEXT = {
    "city": "city destination location town where name",
    "check_in": "check in arrival start date day yyyy",
    "nights": "number nights length stay duration",
    "max_price": "maximum price budget cap nightly per night eur cost",
    "amenities": "amenities facilities features required values",
    "hotel_id": "hotel property identifier id listing",
}
INTENT = {
    "search": "search find hotels city dates listings rating distance amenities price",
    "quote": "exact total price availability specific hotel stay dates quote",
}


@dataclass
class SpecAgent:
    top_k: int = 8
    avoid_side_effects: bool = True

    def run(self, text: str, server: ToolServer) -> Trace:
        tr = Trace()
        spec = public_spec(server.spec)
        spec_tokens = tokens(spec)
        context = tokens(text) + 400
        s = parse(text)

        def step(extra=0):
            nonlocal context
            tr.llm_step(AGENT_MODEL, context + spec_tokens, 80)
            context += 80 + extra

        tools = [t for t in spec if not (self.avoid_side_effects and SIDE_EFFECT_WORDS.search(t.replace("_", " ") + " " + spec[t]["description"]))]
        docs = [t.replace("_", " ") + " " + spec[t]["description"] for t in tools]
        vec = TfidfVectorizer().fit(docs + list(INTENT.values()) + list(SLOT_TEXT.values()))
        sims = cosine_similarity(vec.transform(list(INTENT.values())), vec.transform(docs))
        pick = {k: tools[int(sims[i].argmax())] for i, k in enumerate(INTENT)}

        def fill(tool: str, values: dict) -> dict:
            params = spec[tool]["params"]
            if not params:
                return {}
            names = list(params)
            ptxt = [n.replace("_", " ") + " " + params[n]["description"] for n in names]
            slots = list(SLOT_TEXT)
            m = cosine_similarity(vec.transform(ptxt), vec.transform([SLOT_TEXT[k] for k in slots]))
            args = {}
            for i, n in enumerate(names):
                j = int(m[i].argmax())
                slot, score = slots[j], m[i, j]
                example = re.search(r"e\.g\. ([A-Za-z0-9_-]+)", params[n]["description"])
                mismatch = params[n]["type"] == "string" and not isinstance(values.get(slot), str)
                if example and params[n]["required"] and (score < 0.1 or mismatch):
                    args[n] = example.group(1)                  # no fitting slot: follow the documented example
                    continue
                if (score < 0.05 and not params[n]["required"]) or slot not in values or values[slot] is None:
                    continue
                args[n] = self.encode(values[slot], params[n])
            return args

        results = {}
        for city in s.cities:
            for off in range(-s.flex_days, s.flex_days + 1):
                ci = s.check_in + timedelta(days=off)
                vals = {"city": city, "check_in": ci, "nights": s.nights, "max_price": s.max_per_night,
                        "amenities": list(s.amenities) or None}
                res = tr.tool(server, pick["search"], fill(pick["search"], vals))
                step(tr.steps[-1]["result_tokens"])
                for h in res or []:
                    if (s.min_rating is None or h.get("rating", 0) >= s.min_rating) and \
                       (s.max_distance is None or h.get("distance_km", 99) <= s.max_distance) and \
                       set(s.amenities) <= set(h.get("amenities", [])):
                        results[(h["hotel_id"], ci)] = h
        ranked = sorted(results.items(), key=lambda kv: _rank_key(s, kv[1]["rating"], kv[1]["distance_km"], kv[1]["from_price_eur"]))
        best = None
        for (hid, ci), h in ranked[: self.top_k * max(1, len(s.cities))]:
            q = tr.tool(server, pick["quote"], fill(pick["quote"], {"hotel_id": hid, "check_in": ci, "nights": s.nights}))
            step(tr.steps[-1]["result_tokens"])
            if not q or not q.get("available") or q.get("total_eur") is None:
                continue
            if s.max_per_night and q["total_eur"] / s.nights > s.max_per_night:
                continue
            key = _rank_key(s, h["rating"], h["distance_km"], q["total_eur"])
            if best is None or key < best[0]:
                best = (key, (hid, ci))
        step()                                                          # final answer
        tr.answer = best[1] if best else None
        return tr

    @staticmethod
    def encode(v, p: dict):
        d = p["description"].lower()
        if isinstance(v, date):
            return v.strftime("%d/%m/%Y") if "dd/mm/yyyy" in d else v.isoformat()
        if isinstance(v, float) and ("cents" in d or "minor units" in d):
            return int(round(v * 100))
        if isinstance(v, list):
            enum = re.findall(r"[a-z_]+", d.split("values:", 1)[1]) if "values:" in d else []
            return [next((e for e in enum if e.endswith(a)), a) for a in v]
        return v


