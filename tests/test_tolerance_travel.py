import json
from datetime import date

import httpx
import pytest

from tolerance_travel.agents import AGENT_MODEL, SpecAgent, pipeline
from tolerance_travel.catalog import Catalog, make_tasks, parse
from tolerance_travel.harness import fallback, judge
from tolerance_travel.llm_agent import LLMAgent, parse_answer, to_tools
from tolerance_travel.toolserver import BASE_SPEC, MUTATIONS, ToolCallError, ToolServer, mutated_spec


@pytest.fixture(scope="module")
def cat():
    return Catalog(seed=0)


@pytest.fixture(scope="module")
def tasks(cat):
    return make_tasks(cat, 40, seed=3)


def test_parser_recovers_every_generated_task(tasks):
    assert all(parse(t.text) == t.slots for t in tasks)


def test_infeasible_tasks_have_no_answer_and_others_do(tasks):
    for t in tasks:
        assert (t.kind == "infeasible") == (not t.answer)


def test_server_enforces_the_mutated_contract(cat):
    srv = ToolServer(cat, mutated_spec("rename_params"))
    with pytest.raises(ToolCallError, match="unknown parameter"):
        srv.call("search_hotels", {"city": "Lisbon", "check_in": "2026-11-10", "nights": 2})
    assert srv.call("search_hotels", {"destination": "Lisbon", "arrival_date": "2026-11-10", "length_of_stay": 2})


def test_undocumented_cents_is_silent(cat):
    srv = ToolServer(cat, mutated_spec("undocumented_cents"))
    res = srv.call("search_hotels", {"city": "Lisbon", "check_in": "2026-11-10", "nights": 2, "max_price": 150})
    assert res == []          # no error, just wrong: EUR read as cents


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_every_mutation_builds_a_callable_server(cat, mutation):
    spec = mutated_spec(mutation)
    assert any(s.get("impl") == "search_hotels" for s in spec.values())


def test_verified_pipeline_is_exact_on_stable_api(cat, tasks):
    srv = ToolServer(cat, mutated_spec("none"))
    assert all(judge(cat, t, pipeline(t.text, srv, verified=True))["success"] for t in tasks)


def test_agent_never_books_even_when_booking_is_the_only_tool_named_like_the_task(cat, tasks):
    for m in ("none", "opaque_tool_names", "terse_descriptions"):
        srv = ToolServer(cat, mutated_spec(m))
        for t in tasks[:10]:
            SpecAgent().run(t.text, srv)
        assert srv.bookings == []


def test_agent_reads_units_from_the_spec(cat, tasks):
    t = next(t for t in tasks if t.kind == "single")
    srv = ToolServer(cat, mutated_spec("price_in_cents"))
    tr = SpecAgent().run(t.text, srv)
    search = next(s for s in tr.steps if s["tool"] == "search_hotels")
    assert search["args"]["max_price"] == int(t.slots.max_per_night * 100)


def test_fallback_only_pays_for_the_agent_when_the_pipeline_breaks(cat, tasks):
    t = tasks[0]
    stable = fallback(t.text, ToolServer(cat, mutated_spec("none")))
    broken = fallback(t.text, ToolServer(cat, mutated_spec("rename_params")))
    assert stable.cost_eur < 0.001 < broken.cost_eur


def test_parse_answer():
    assert parse_answer("blah\nANSWER: lis-04 2026-11-12") == ("lis-04", date(2026, 11, 12))
    assert parse_answer("ANSWER: NONE") is None and parse_answer("no idea") is None


def test_tools_schema_shape():
    tools = {t["name"]: t for t in to_tools(BASE_SPEC)}
    assert tools["search_hotels"]["input_schema"]["required"] == ["city", "check_in", "nights"]


def test_llm_loop_against_mocked_api(cat, tasks):
    t = next(t for t in tasks if t.kind == "single")
    target = sorted(t.answer)[0]
    replies = iter([
        {"content": [{"type": "tool_use", "id": "u1", "name": "search_hotels",
                      "input": {"city": t.slots.cities[0], "check_in": t.slots.check_in.isoformat(), "nights": t.slots.nights}}],
         "usage": {"input_tokens": 1200, "output_tokens": 60}},
        {"content": [{"type": "tool_use", "id": "u2", "name": "get_quote",
                      "input": {"hotel_id": target[0], "check_in": target[1].isoformat(), "nights": t.slots.nights}}],
         "usage": {"input_tokens": 4000, "output_tokens": 50}},
        {"content": [{"type": "text", "text": f"ANSWER: {target[0]} {target[1].isoformat()}"}],
         "usage": {"input_tokens": 4300, "output_tokens": 20}},
    ])
    seen = []

    def handler(req: httpx.Request):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=next(replies))

    agent = LLMAgent("mock-model", AGENT_MODEL, client=httpx.Client(transport=httpx.MockTransport(handler)), api_key="x")
    tr = agent.run(t.text, ToolServer(cat, mutated_spec("none")))
    assert judge(cat, t, tr)["success"]
    assert [s["tool"] for s in tr.steps] == ["search_hotels", "get_quote"]
    assert seen[-1]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert tr.tokens_in == 9500
