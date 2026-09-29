"""Synthetic travel inventory, natural-language tasks, an exact oracle, and the request parser.

Prices vary by night; listing pages show a "from" price (the cheapest night in the window), which is
what search returns. Exact prices and availability need a quote. That gap is what separates a
one-shot pipeline from an agent that verifies, and it is modelled on how travel inventory behaves.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

CITIES = ("Amsterdam", "Lisbon", "Barcelona", "Paris", "Berlin", "Rome", "Prague", "Vienna", "Copenhagen", "Porto")
CITY_BASE = dict(zip(CITIES, (190, 120, 140, 210, 115, 150, 95, 130, 180, 100)))
AMENITIES = ("breakfast", "pool", "parking", "pets", "gym")
AMENITY_P = dict(zip(AMENITIES, (0.55, 0.2, 0.35, 0.25, 0.4)))
START = date(2026, 11, 1)
DAYS = 60


def _u(*key) -> float:
    h = hashlib.blake2b("|".join(map(str, key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


@dataclass(frozen=True)
class Hotel:
    id: str
    name: str
    city: str
    base: float
    rating: float
    distance_km: float
    amenities: tuple[str, ...]
    description: str


class Catalog:
    def __init__(self, seed: int = 0, per_city: int = 30):
        r = np.random.default_rng(seed)
        self.seed = seed
        self.hotels: dict[str, Hotel] = {}
        for city in CITIES:
            for i in range(per_city):
                am = tuple(a for a in AMENITIES if r.random() < AMENITY_P[a])
                h = Hotel(f"{city[:3].lower()}-{i:02d}", f"{city} Stay {i:02d}", city,
                          float(CITY_BASE[city] * r.lognormal(0, 0.35)), round(float(np.clip(r.normal(8.0, 0.8), 5.5, 9.9)), 1),
                          round(float(r.gamma(2.0, 1.3)), 1), am,
                          f"A {['quiet', 'central', 'modern', 'classic'][i % 4]} hotel in {city}. " * 6)
                self.hotels[h.id] = h

    def nightly(self, hid: str, d: date) -> float:
        h = self.hotels[hid]
        weekend = 1.25 if d.weekday() in (4, 5) else 1.0
        return round(h.base * weekend * (0.85 + 0.3 * _u(self.seed, hid, d.isoformat(), "p")), 2)

    def available(self, hid: str, d: date) -> bool:
        return _u(self.seed, hid, d.isoformat(), "a") > 0.12

    def quote(self, hid: str, check_in: date, nights: int):
        ds = [check_in + timedelta(days=k) for k in range(nights)]
        if ds[-1] >= START + timedelta(days=DAYS) or ds[0] < START:
            return False, None
        if not all(self.available(hid, d) for d in ds):
            return False, None
        return True, round(sum(self.nightly(hid, d) for d in ds), 2)

    def from_price(self, hid: str) -> float:
        return min(self.nightly(hid, START + timedelta(days=k)) for k in range(DAYS))


# --------------------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class Slots:
    cities: tuple[str, ...]
    check_in: date
    nights: int
    max_per_night: float | None
    amenities: tuple[str, ...] = ()
    min_rating: float | None = None
    max_distance: float | None = None
    objective: str = "cheapest"          # cheapest | best_rated | closest
    flex_days: int = 0


@dataclass(frozen=True)
class Task:
    id: str
    kind: str                            # single | multi_constraint | flex | compare | infeasible
    text: str
    slots: Slots
    answer: frozenset = field(default_factory=frozenset)   # acceptable (hotel_id, check_in) pairs; empty = none exists
    best_value: float | None = None


def candidates(cat: Catalog, s: Slots):
    out = []
    for city in s.cities:
        for h in cat.hotels.values():
            if h.city != city or not set(s.amenities) <= set(h.amenities):
                continue
            if s.min_rating is not None and h.rating < s.min_rating:
                continue
            if s.max_distance is not None and h.distance_km > s.max_distance:
                continue
            for off in range(-s.flex_days, s.flex_days + 1):
                ci = s.check_in + timedelta(days=off)
                ok, total = cat.quote(h.id, ci, s.nights)
                if not ok or (s.max_per_night is not None and total / s.nights > s.max_per_night):
                    continue
                out.append((h, ci, total))
    return out


def objective_value(s: Slots, h: Hotel, total: float) -> float:
    return {"cheapest": total, "best_rated": -h.rating, "closest": h.distance_km}[s.objective]


def oracle(cat: Catalog, s: Slots):
    c = candidates(cat, s)
    if not c:
        return frozenset(), None
    vals = [objective_value(s, h, t) for h, _, t in c]
    best = min(vals)
    tol = abs(best) * 0.01 if s.objective == "cheapest" else 1e-9
    return frozenset((h.id, ci) for (h, ci, _), v in zip(c, vals) if v <= best + tol), best


OBJ_TEXT = {"cheapest": ["the cheapest option", "lowest total price"],
            "best_rated": ["the best rated one", "highest guest rating"],
            "closest": ["the one closest to the centre", "shortest distance to the centre"]}


def render(s: Slots, r: np.random.Generator) -> str:
    d = s.check_in.strftime("%d %B %Y").lstrip("0")
    city = " or ".join(s.cities)
    parts = [r.choice([f"Find me a hotel in {city} from {d} for {s.nights} nights",
                       f"{s.nights} nights in {city} starting {d}",
                       f"I need somewhere to stay in {city}, arriving {d}, {s.nights} nights"])]
    if s.max_per_night:
        parts.append(r.choice([f"under €{s.max_per_night:.0f} a night", f"budget €{s.max_per_night:.0f} per night"]))
    if s.amenities:
        parts.append("with " + " and ".join(s.amenities))
    if s.min_rating:
        parts.append(f"rated at least {s.min_rating:.1f}")
    if s.max_distance:
        parts.append(f"within {s.max_distance:.0f} km of the centre")
    if s.flex_days:
        parts.append(f"my dates can move by {s.flex_days} day either way")
    return ", ".join(parts) + ". I want " + r.choice(OBJ_TEXT[s.objective]) + "."


def make_tasks(cat: Catalog, n: int, seed: int = 0) -> list[Task]:
    r = np.random.default_rng(seed)
    kinds = ("single", "multi_constraint", "flex", "compare", "infeasible")
    out, i = [], 0
    while len(out) < n:
        kind = kinds[len(out) % len(kinds)]
        city = CITIES[r.integers(len(CITIES))]
        cities = (city,) if kind != "compare" else tuple(sorted({city, CITIES[r.integers(len(CITIES))]}))
        if len(cities) == 1 and kind == "compare":
            continue
        ci = START + timedelta(days=int(r.integers(2, DAYS - 10)))
        nights = int(r.integers(1, 5))
        budget = float(round(CITY_BASE[city] * r.uniform(0.9, 1.6), -1))
        s = Slots(cities, ci, nights, budget, objective=("cheapest", "best_rated", "closest")[r.integers(3)])
        if kind == "multi_constraint":
            s = Slots(cities, ci, nights, budget, tuple(sorted(r.choice(AMENITIES, size=int(r.integers(1, 3)), replace=False))),
                      float(round(r.uniform(7.0, 8.5), 1)), float(r.integers(2, 6)), s.objective)
        elif kind == "flex":
            s = Slots(cities, ci, nights, budget, objective="cheapest", flex_days=1)
        elif kind == "infeasible":
            s = Slots(cities, ci, nights, float(round(CITY_BASE[city] * 0.35, -1)), ("pets", "pool"), 9.3, 1.0, s.objective)
        ans, best = oracle(cat, s)
        if kind == "infeasible" and ans:
            continue
        if kind != "infeasible" and not ans:
            continue
        out.append(Task(f"t{i:04d}", kind, render(s, r), s, ans, best))
        i += 1
    return out


# -------------------------------------------------------------------------------- parser
MONTHS = {m: k for k, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"), 1)}


def parse(text: str) -> Slots:
    """Shared by every agent and the baseline, so differences come from behaviour, not reading."""
    t = text.lower()
    cities = tuple(sorted(c for c in CITIES if c.lower() in t))
    m = re.search(r"(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})", t)
    ci = date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    nights = int(re.search(r"(\d+) nights", t).group(1))
    p = re.search(r"€(\d+)", t)
    rating = re.search(r"rated at least (\d+\.\d)", t)
    dist = re.search(r"within (\d+) km", t)
    flex = re.search(r"move by (\d+) day", t)
    am = tuple(sorted(a for a in AMENITIES if re.search(rf"\b{a}\b", t) and "with" in t))
    obj = "cheapest"
    if "rated one" in t or "highest guest rating" in t:
        obj = "best_rated"
    elif "closest" in t or "shortest distance" in t:
        obj = "closest"
    return Slots(cities, ci, nights, float(p.group(1)) if p else None, am,
                 float(rating.group(1)) if rating else None, float(dist.group(1)) if dist else None,
                 obj, int(flex.group(1)) if flex else 0)
