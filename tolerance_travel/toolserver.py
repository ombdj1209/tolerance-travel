"""Tool server with a published spec, plus mutation operators that model real API drift.

A mutation changes the published spec *and* what the server accepts: renamed parameters are the only
names it accepts, a price documented in cents is interpreted in cents, and so on. Clients that read
the spec can adapt; clients with hard-coded calls break. That is the experiment.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime

from .catalog import AMENITIES, CITIES, Catalog


class ToolCallError(Exception):
    pass


def _iso(v):
    return v if isinstance(v, date) else date.fromisoformat(str(v))


BASE_SPEC = {
    "list_cities": {"description": "List the cities where hotels can be searched.", "params": {}},
    "search_hotels": {
        "description": "Search hotels in a city for given dates. Returns listings with a 'from' price (cheapest night "
                       "in the season), rating, distance to the centre and amenities.",
        "params": {
            "city": {"type": "string", "required": True, "description": "City name, e.g. Lisbon."},
            "check_in": {"type": "date", "required": True, "description": "Check-in date, YYYY-MM-DD."},
            "nights": {"type": "integer", "required": True, "description": "Number of nights to stay."},
            "max_price": {"type": "number", "required": False, "description": "Maximum nightly price in EUR."},
            "amenities": {"type": "array", "required": False,
                          "description": "Required amenities. Values: " + ", ".join(AMENITIES) + "."},
        }},
    "get_quote": {
        "description": "Get the exact total price and availability for a specific hotel stay on given dates.",
        "params": {
            "hotel_id": {"type": "string", "required": True, "description": "Hotel identifier from search results."},
            "check_in": {"type": "date", "required": True, "description": "Check-in date, YYYY-MM-DD."},
            "nights": {"type": "integer", "required": True, "description": "Number of nights."},
        }},
    "get_hotel_details": {
        "description": "Get the full description, rating, amenities and cancellation policy of one hotel.",
        "params": {"hotel_id": {"type": "string", "required": True, "description": "Hotel identifier."}}},
    "book_hotel": {
        "description": "Book and pay for a hotel stay. Charges the customer's card.",
        "params": {"hotel_id": {"type": "string", "required": True, "description": "Hotel identifier."},
                   "check_in": {"type": "date", "required": True, "description": "Check-in date."},
                   "nights": {"type": "integer", "required": True, "description": "Number of nights."}}},
    "get_weather": {"description": "Weather forecast for a city on a date.",
                    "params": {"city": {"type": "string", "required": True, "description": "City name."},
                               "day": {"type": "date", "required": True, "description": "Date, YYYY-MM-DD."}}},
    "search_flights": {"description": "Search flights between two airports on a date.",
                       "params": {"origin": {"type": "string", "required": True, "description": "Origin airport."},
                                  "destination": {"type": "string", "required": True, "description": "Destination airport."},
                                  "day": {"type": "date", "required": True, "description": "Departure date."}}},
}
for _t, _s in BASE_SPEC.items():
    for _p, _d in _s["params"].items():
        _d.setdefault("canonical", _p)
    _s.setdefault("impl", _t)


# ------------------------------------------------------------------------------ mutations
def m_rename_params(spec):
    ren = {"city": "destination", "check_in": "arrival_date", "nights": "length_of_stay",
           "max_price": "price_cap", "hotel_id": "property_id"}
    for t in ("search_hotels", "get_quote", "get_hotel_details", "book_hotel"):
        spec[t]["params"] = {ren.get(k, k): v for k, v in spec[t]["params"].items()}
    return spec


def m_terse_descriptions(spec):
    terse = {"search_hotels": "Hotels.", "get_quote": "Quote.", "get_hotel_details": "Details.",
             "book_hotel": "Booking.", "get_weather": "Weather.", "search_flights": "Flights.", "list_cities": "Cities."}
    for t, d in terse.items():
        spec[t]["description"] = d
    return spec


def m_drop_param_descriptions(spec):
    for s in spec.values():
        for p in s["params"].values():
            p["description"] = ""
    return spec


def m_price_in_cents(spec):
    p = spec["search_hotels"]["params"]["max_price"]
    p["description"], p["type"], p["scale"] = "Maximum nightly price in cents (minor units).", "integer", 100
    return spec


def m_undocumented_cents(spec):
    """The same unit change as price_in_cents, shipped without updating the docs: a silent break."""
    p = spec["search_hotels"]["params"]["max_price"]
    p["type"], p["scale"] = "integer", 100
    return spec


def m_date_ddmmyyyy(spec):
    for t in ("search_hotels", "get_quote", "book_hotel"):
        p = spec[t]["params"]["check_in"]
        p["description"], p["format"] = "Check-in date as DD/MM/YYYY.", "%d/%m/%Y"
    return spec


def m_amenity_enum_drift(spec):
    p = spec["search_hotels"]["params"]["amenities"]
    p["description"] = "Required facilities. Values: " + ", ".join(f"has_{a}" for a in AMENITIES) + "."
    p["enum_prefix"] = "has_"
    return spec


def m_distractor_tools(spec):
    """Near-duplicate tools whose descriptions are keyword-rich but whose data is stale or estimated."""
    s = copy.deepcopy(spec["search_hotels"])
    s["description"] = ("Search and find hotels in a city for dates: listings with rating, distance, amenities and "
                        "price. Fast cached index; prices may be out of date.")
    s["impl"] = "search_hotels_cached"
    spec["search_hotels_fast"] = s
    q = copy.deepcopy(spec["get_quote"])
    q["description"] = ("Quote the exact total price and availability for a specific hotel stay on given dates and "
                        "nights. Served from cached estimates.")
    q["impl"] = "quote_estimate"
    spec["estimate_price"] = q
    return spec


def m_ambiguous_param_names(spec):
    ren = {"city": "where", "check_in": "date", "nights": "n", "max_price": "limit", "hotel_id": "id"}
    for t in ("search_hotels", "get_quote", "get_hotel_details", "book_hotel"):
        spec[t]["params"] = {ren.get(k, k): v for k, v in spec[t]["params"].items()}
    spec["search_hotels"]["params"]["limit"]["description"] = "Upper limit."
    return spec


def m_opaque_tool_names(spec):
    """Generic tool names with one-word descriptions: what a hastily wrapped internal API looks like."""
    order = ["get_weather", "get_quote", "list_cities", "search_flights", "book_hotel", "search_hotels", "get_hotel_details"]
    terse = {"search_hotels": "Hotels.", "get_quote": "Quote.", "get_hotel_details": "Details.",
             "book_hotel": "Booking.", "get_weather": "Weather.", "search_flights": "Flights.", "list_cities": "Cities."}
    return {f"tool_{i + 1}": {**spec[t], "description": terse[t]} for i, t in enumerate(order)}


def m_new_required_param(spec):
    spec["search_hotels"]["params"]["currency"] = {
        "type": "string", "required": True, "canonical": "currency", "enum": ["EUR"],
        "description": "ISO 4217 currency for all prices, e.g. EUR."}
    return spec


MUTATIONS = {
    "none": lambda s: s, "rename_params": m_rename_params, "terse_descriptions": m_terse_descriptions,
    "drop_param_descriptions": m_drop_param_descriptions, "price_in_cents": m_price_in_cents,
    "undocumented_cents": m_undocumented_cents,
    "date_ddmmyyyy": m_date_ddmmyyyy, "amenity_enum_drift": m_amenity_enum_drift,
    "distractor_tools": m_distractor_tools, "ambiguous_param_names": m_ambiguous_param_names,
    "opaque_tool_names": m_opaque_tool_names, "new_required_param": m_new_required_param,
}


def mutated_spec(name: str) -> dict:
    return MUTATIONS[name](copy.deepcopy(BASE_SPEC))


def public_spec(spec: dict) -> dict:
    """What a client sees: names, descriptions, types. Never the server's internal mapping."""
    out = {}
    for t, s in spec.items():
        out[t] = {"description": s["description"],
                  "params": {k: {"type": v["type"], "required": v["required"], "description": v["description"]}
                             for k, v in s["params"].items()}}
    return out


# ---------------------------------------------------------------------------------- server
@dataclass
class ToolServer:
    catalog: Catalog
    spec: dict
    bookings: list = field(default_factory=list)
    calls: int = 0

    def call(self, tool: str, args: dict):
        self.calls += 1
        if tool not in self.spec:
            raise ToolCallError(f"unknown tool {tool}")
        s = self.spec[tool]
        extra = set(args) - set(s["params"])
        if extra:
            raise ToolCallError(f"unknown parameter(s): {', '.join(sorted(extra))}")
        canon = {}
        for name, p in s["params"].items():
            if name not in args:
                if p["required"]:
                    raise ToolCallError(f"missing required parameter: {name}")
                continue
            v = args[name]
            if p["type"] == "date":
                try:
                    v = datetime.strptime(str(v), p["format"]).date() if "format" in p else _iso(v)
                except ValueError as e:
                    raise ToolCallError(f"{name}: invalid date {v!r}") from e
            elif p["type"] in ("number", "integer"):
                try:
                    v = float(v) / p.get("scale", 1)
                except (TypeError, ValueError) as e:
                    raise ToolCallError(f"{name}: not a number") from e
            elif p["type"] == "string" and "enum" in p and v not in p["enum"]:
                raise ToolCallError(f"{name}: must be one of {p['enum']}")
            elif p["type"] == "array":
                pre = p.get("enum_prefix", "")
                vals = []
                for a in (v or []):
                    if not str(a).startswith(pre) or str(a)[len(pre):] not in AMENITIES:
                        raise ToolCallError(f"{name}: unknown value {a!r}")
                    vals.append(str(a)[len(pre):])
                v = vals
            canon[p["canonical"]] = v
        return getattr(self, "_" + s["impl"])(**canon)

    # canonical implementations
    def _list_cities(self):
        return list(CITIES)

    def _search_hotels(self, city, check_in, nights, max_price=None, amenities=(), stale=False, currency="EUR"):
        out = []
        for h in self.catalog.hotels.values():
            if h.city.lower() != str(city).lower() or not set(amenities) <= set(h.amenities):
                continue
            fp = self.catalog.from_price(h.id) * (0.8 if stale else 1.0)
            if max_price is not None and fp > max_price:
                continue
            out.append({"hotel_id": h.id, "name": h.name, "rating": h.rating, "distance_km": h.distance_km,
                        "amenities": list(h.amenities), "from_price_eur": round(fp, 2)})
        return sorted(out, key=lambda x: x["from_price_eur"])[:30]

    def _search_hotels_cached(self, **kw):
        kw.pop("currency", None)
        return self._search_hotels(**kw, stale=True)

    def _get_quote(self, hotel_id, check_in, nights):
        if hotel_id not in self.catalog.hotels:
            raise ToolCallError(f"unknown hotel {hotel_id}")
        ok, total = self.catalog.quote(hotel_id, check_in, int(nights))
        return {"available": ok, "total_eur": total, "per_night_eur": round(total / nights, 2) if ok else None}

    def _quote_estimate(self, hotel_id, check_in, nights):
        fp = self.catalog.from_price(hotel_id)
        return {"available": True, "total_eur": round(fp * nights, 2), "per_night_eur": fp}

    def _get_hotel_details(self, hotel_id):
        h = self.catalog.hotels[hotel_id]
        return {"hotel_id": h.id, "rating": h.rating, "distance_km": h.distance_km, "amenities": list(h.amenities),
                "free_cancellation": hash(h.id) % 2 == 0, "description": h.description}

    def _book_hotel(self, hotel_id, check_in, nights):
        self.bookings.append((hotel_id, str(check_in), nights))
        return {"booking_ref": f"BK{len(self.bookings):05d}"}

    def _get_weather(self, city, day):
        return {"city": city, "day": str(day), "forecast": "cloudy"}

    def _search_flights(self, origin, destination, day):
        return []
