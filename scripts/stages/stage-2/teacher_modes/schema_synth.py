#!/usr/bin/env python3
"""Programmatically verifiable schema_arguments tasks (`schema_synth@v1`): a DESIGN for the maintainer's review. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_synth \\
        --out artifacts/stages/stage-2/teacher_modes/schema_synth_v1 [--audit <json>]

Why it exists: the non-thinking schema_arguments cell of the registered composition is short because the only legal
gold-argument source, Pivot Conversational, is exhausted (audit/schema_arguments_supply_v3.json). This module writes
NEW task material whose gold call is computed by a program from a sampled ground truth -- never by a model -- so it
verifies at the pool's existing levels with no change to the verifier (`tool_episodes.verify`, tool_verification@v2):

* every task is a ``full_episode@v1`` of two turns -- the teacher's own call, the tool's deterministic response,
  the teacher's own answer without a call (gold ``[[call]], [[]]``) -- exactly the shape of Pivot's ``turn1`` tasks;
* step gold is the exact canonical call (name + arguments, sorted keys, `tool_calls.canonical_call`): the tool
  descriptions spell out every convention a value must follow (units, formats, order of arrays, when an optional
  field is present), and each value is either copied verbatim from the request or derived from it by one stated
  rule, so exactly one call is correct;
* the replay table answers that call and nothing else, so an off-gold call ends the episode (recorded execution);
* the schema uses only `tool_calls.CHECKED_KEYWORDS`, so schema validity is a real check.

What the tasks exercise, beyond filling a template: reading a free-form request for the facts a call needs among
distractors (an old order number, a retracted value), turning relative dates and times into calendar values from the
stated current date, converting units and counts ("three dozen", "1.25 kg" -> grams), arithmetic over several facts
(totals, scaled quantities, sums of durations), applying a rule stated in the tool description to set a dependent
field, choosing an enum value from its described meaning, and nesting the results in objects and ordered arrays.

Isolation: a domain is one environment atom (its tool set); domains are assigned whole to splits by a seeded rule,
so no tool, schema or domain is shared between training, calibration, state evaluation and validation. No tool name
or signature equals one of eval-set-1, the pinned public benchmarks or the excluded lineage sources (eleven names
that matched were renamed before any generation; audit/schema_synth_v1_audit.json). The teacher
generates every assistant turn; nothing here is a training target (P17, six_category_design.json :: protocol).
Everything is a pure function of (version, domain, index): the same inputs give the same bytes.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SOURCE = "aadistill/schema_synth"
VERSION = "schema_synth@v1"
#: instances per domain by split, and how domains are assigned to splits (whole domains, `assign_splits`)
SPLIT_DOMAINS = (("state_eval", 3), ("validation", 2), ("calibration", 2), ("reserve", 1))
PER_DOMAIN = {"training": 160, "state_eval": 40, "validation": 20, "calibration": 30, "reserve": 10}
#: the current dates instances are set on (the system prompt states it), and the difficulty mix
DATE_RANGE = (date(2026, 1, 5), date(2026, 12, 18))
LEVELS = ((1, 0.3), (2, 0.4), (3, 0.3))
SYSTEM = "You are a helpful assistant that can call tools.\nCurrent date: {weekday}, {iso}."
OPTIONAL = "Optional: omit it unless the user specifies it."

WEEKDAYS = list(calendar.day_name)
MONTHS = list(calendar.month_name)[1:]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
FIRST = ["Maya", "Daniel", "Priya", "Jonas", "Amara", "Luis", "Hannah", "Kenji", "Sofia", "Omar", "Elena", "Tobias",
         "Grace", "Mateo", "Leila", "Viktor", "Nadia", "Samuel", "Ines", "Rafael", "Chloe", "Arjun", "Freya", "Diego"]
LAST = ["Okafor", "Lindqvist", "Ramanathan", "Becker", "Haddad", "Moreau", "Kowalski", "Tanaka", "Silva", "Novak",
        "Fischer", "Abara", "Castillo", "Nguyen", "Petrov", "Duarte", "Walsh", "Iyer", "Brennan", "Santos"]
COUNTRIES = [("Germany", "DE"), ("Japan", "JP"), ("Brazil", "BR"), ("Canada", "CA"), ("Kenya", "KE"),
             ("Spain", "ES"), ("India", "IN"), ("Australia", "AU"), ("Mexico", "MX"), ("Norway", "NO"),
             ("South Korea", "KR"), ("Italy", "IT"), ("the Netherlands", "NL"), ("Poland", "PL")]
CITIES = {"DE": ["Hamburg", "Leipzig"], "JP": ["Osaka", "Sapporo"], "BR": ["Recife", "Curitiba"],
          "CA": ["Halifax", "Calgary"], "KE": ["Mombasa", "Nakuru"], "ES": ["Valencia", "Bilbao"],
          "IN": ["Pune", "Kochi"], "AU": ["Perth", "Hobart"], "MX": ["Puebla", "Mérida"], "NO": ["Bergen", "Tromsø"],
          "KR": ["Busan", "Daejeon"], "IT": ["Turin", "Bari"], "NL": ["Utrecht", "Groningen"],
          "PL": ["Gdańsk", "Poznań"]}
STREETS = ["Linden Avenue", "Harbor Road", "Maple Street", "Station Lane", "Orchard Way", "Mill Road", "Cedar Court"]


# --- deterministic phrase <-> value toolkit ------------------------------------------------------------------------

def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    raise ValueError(n)


def num(r: random.Random, n: int) -> str:
    """A small count as digits or words."""
    return words(n) if n < 100 and r.random() < 0.5 else str(n)


def iso(d: date) -> str:
    return d.isoformat()


def nice(d: date, r: random.Random) -> str:
    """An absolute date in one of several unambiguous spellings."""
    return r.choice([f"{MONTHS[d.month - 1]} {d.day}, {d.year}", f"{d.day} {MONTHS[d.month - 1]} {d.year}",
                     d.isoformat(), f"{WEEKDAYS[d.weekday()]}, {MONTHS[d.month - 1]} {d.day}, {d.year}"])


def coming(today: date, weekday: int, skip: int = 0) -> date:
    """The first ``weekday`` strictly after today, ``skip`` weeks later."""
    ahead = (weekday - today.weekday()) % 7 or 7
    return today + timedelta(days=ahead + 7 * skip)


def last_of_month(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def next_month(d: date) -> tuple[int, int]:
    return (d.year + (d.month == 12), d.month % 12 + 1)


def date_phrase(r: random.Random, today: date, *, level: int, min_ahead: int = 1, max_ahead: int = 45
                ) -> tuple[str, date, str]:
    """``(phrase, date, operator)``: a future date as the user would say it, resolved against today."""
    kinds = ["absolute"] + (["tomorrow", "in_days", "weeks"] if level >= 1 else []) + \
        (["coming_weekday", "weekday_after_next", "last_of_month", "first_weekday_next_month"] if level >= 2 else [])
    for _ in range(50):
        k = r.choice(kinds)
        if k == "absolute":
            d = today + timedelta(days=r.randint(min_ahead, max_ahead))
            p = nice(d, r)
        elif k == "tomorrow":
            n = r.choice([1, 2])
            d, p = today + timedelta(days=n), ("tomorrow" if n == 1 else "the day after tomorrow")
        elif k == "in_days":
            n = r.randint(3, 20)
            d, p = today + timedelta(days=n), f"in {num(r, n)} days"
        elif k == "weeks":
            n = r.choice([1, 2, 3])
            d, p = today + timedelta(days=7 * n), ["a week from today", "two weeks from today",
                                                    "three weeks from today"][n - 1]
        elif k == "coming_weekday":
            w = r.randrange(7)
            d, p = coming(today, w), f"this coming {WEEKDAYS[w]}"
        elif k == "weekday_after_next":
            w = r.randrange(7)
            d, p = coming(today, w, 1), f"the {WEEKDAYS[w]} after next"
        elif k == "last_of_month":
            if r.random() < 0.5 and last_of_month(today.year, today.month) > today:
                d, p = last_of_month(today.year, today.month), "the last day of this month"
            else:
                d, p = last_of_month(*next_month(today)), "the last day of next month"
        else:
            y, m = next_month(today)
            w = r.randrange(7)
            first = date(y, m, 1)
            d = first + timedelta(days=(w - first.weekday()) % 7)
            p = f"the first {WEEKDAYS[w]} of next month"
        if min_ahead <= (d - today).days <= max(max_ahead, 62):
            return p, d, f"date:{k}"
    d = today + timedelta(days=min_ahead)
    return nice(d, r), d, "date:absolute"


def time_phrase(r: random.Random, *, level: int, lo: int = 7, hi: int = 21) -> tuple[str, str, str]:
    """``(phrase, "HH:MM", operator)`` on a quarter-hour grid."""
    h, m = r.randint(lo, hi), r.choice([0, 15, 30, 45])
    if h == hi and m:
        m = 0
    hh = h % 12 or 12
    ampm = "am" if h < 12 else "pm"
    part = "in the morning" if h < 12 else ("in the afternoon" if h < 18 else "in the evening")
    opts = [(f"{hh}:{m:02d} {ampm}", "time:clock12"), (f"{h:02d}:{m:02d}", "time:clock24")]
    if level >= 2:
        if m == 30:
            opts.append((f"half past {words(hh)} {part}", "time:words"))
        if m == 15:
            opts.append((f"quarter past {words(hh)} {part}", "time:words"))
        if m == 45 and h + 1 != 12:
            nh = h + 1
            npart = "in the morning" if nh < 12 else ("in the afternoon" if nh < 18 else "in the evening")
            opts.append((f"quarter to {words(nh % 12 or 12)} {npart}", "time:words"))
        if m == 0 and h == 12:
            opts.append(("noon", "time:words"))
    p, op = r.choice(opts)
    return p, f"{h:02d}:{m:02d}", op


DURATIONS = {15: ["a quarter of an hour", "15 minutes"], 30: ["half an hour", "30 minutes"],
             45: ["three quarters of an hour", "45 minutes"], 60: ["an hour", "60 minutes", "1 hour"],
             75: ["an hour and fifteen minutes", "75 minutes"], 90: ["an hour and a half", "90 minutes", "1.5 hours"],
             105: ["an hour and 45 minutes", "105 minutes"], 120: ["two hours", "2 hours", "120 minutes"],
             135: ["2 hours 15 minutes", "two and a quarter hours"], 150: ["two and a half hours", "2.5 hours"],
             180: ["three hours", "3 hours"], 210: ["three and a half hours", "3.5 hours"],
             240: ["four hours", "4 hours"]}


def duration_phrase(r: random.Random, choices=tuple(DURATIONS)) -> tuple[str, int, str]:
    m = r.choice(list(choices))
    return r.choice(DURATIONS[m]), m, "duration:minutes"


def count_phrase(r: random.Random, n: int, noun: str, *, level: int) -> tuple[str, str]:
    """A count of ``noun`` (plural noun given), possibly as dozens or groups."""
    if level >= 2 and n in (6, 12, 24, 36, 48) and r.random() < 0.6:
        dz = {6: "half a dozen", 12: "a dozen", 24: "two dozen", 36: "three dozen", 48: "four dozen"}[n]
        return f"{dz} {noun}", "count:dozen"
    if level >= 2 and n >= 8 and r.random() < 0.4:
        for g in (8, 6, 5, 4, 3):
            if n % g == 0 and n // g > 1:
                return f"{num(r, n // g)} packs of {g} {noun}", "count:groups"
    return f"{num(r, n)} {noun}", "count:plain"


def money_text(cents: int, r: random.Random, sym: str = "$") -> str:
    d, c = divmod(cents, 100)
    opts = [f"{sym}{d:,}.{c:02d}" if c else f"{sym}{d:,}"]
    if sym == "$" and c and d < 100:
        opts.append(f"{words(d)} dollars and {words(c)} cents")
    return r.choice(opts)


def phone(r: random.Random) -> tuple[str, str, str]:
    """``(as written, E.164, operator)`` for a US or UK number."""
    if r.random() < 0.6:
        a, b, c = r.randint(201, 989), r.randint(200, 989), r.randint(0, 9999)
        written = r.choice([f"({a}) {b}-{c:04d}", f"{a}-{b}-{c:04d}", f"{a}.{b}.{c:04d}"])
        return written, f"+1{a}{b}{c:04d}", "phone:us"
    a, b = r.randint(1000, 9999), r.randint(1000, 9999)
    return f"020 {a} {b}", f"+4420{a}{b}", "phone:uk"


def person(r: random.Random) -> dict:
    f, ln = r.choice(FIRST), r.choice(LAST)
    return {"first": f, "last": ln, "name": f"{f} {ln}", "email": f"{f.lower()}.{ln.lower()}@{r.choice(['example.com', 'mail.test', 'corp.example'])}"}


def code(r: random.Random, prefix: str, digits: int = 5) -> str:
    return f"{prefix}{r.randint(10 ** (digits - 1), 10 ** digits - 1)}"


def alnum(r: random.Random, n: int = 6) -> str:
    return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(n))


def pick(r: random.Random, xs, k: int) -> list:
    return r.sample(list(xs), k)


# --- schema helpers -------------------------------------------------------------------------------------------------

def obj(props: dict, required: list | None = None, desc: str | None = None) -> dict:
    s = {"type": "object", "properties": props, "required": list(required if required is not None else props),
         "additionalProperties": False}
    if desc:
        s["description"] = desc
    return s


def S(desc: str, **kw) -> dict:
    return {"type": "string", "description": desc, **kw}


def I(desc: str, **kw) -> dict:  # noqa: E743
    return {"type": "integer", "description": desc, **kw}


def B(desc: str) -> dict:
    return {"type": "boolean", "description": desc}


def E(desc: str, values) -> dict:
    return {"type": "string", "enum": list(values), "description": desc}


def A(items: dict, desc: str, **kw) -> dict:
    return {"type": "array", "items": items, "description": desc, **kw}


DATE = {"pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "format": "date"}
TIME = {"pattern": "^[0-9]{2}:[0-9]{2}$"}
E164 = {"pattern": "^\\+[1-9][0-9]{7,14}$"}


def tool(name: str, desc: str, params: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


def simple(fname: str, fdesc: str, /, **props) -> dict:
    return tool(fname, fdesc, obj({k: S(v) for k, v in props.items()}))


@dataclass
class Instance:
    args: dict
    opening: str
    clauses: list
    ops: list
    distractor: str | None = None


@dataclass(frozen=True)
class Domain:
    name: str
    tool: dict
    distractors: tuple
    sample: Callable[[random.Random, date, int], Instance]
    entity: str


# --- the domains ----------------------------------------------------------------------------------------------------

def _shipment(r, today, level):
    s, rc = person(r), person(r)
    sp, se, op_p = phone(r)
    cn, cc = r.choice(COUNTRIES)
    city = r.choice(CITIES[cc])
    street = f"{r.randint(2, 980)} {r.choice(STREETS)}"
    postal = str(r.randint(10000, 99999))
    n_pk = r.choice([1, 1, 2, 3]) if level >= 2 else 1
    pkgs, pk_text = [], []
    for _ in range(n_pk):
        kg = r.choice([0.5, 0.75, 1.25, 2, 2.5, 3.2, 4.75, 6, 8.5, 12])
        dims = [r.choice([20, 25, 30, 35, 40, 50, 60]) for _ in range(3)]
        pkgs.append({"weight_g": int(round(kg * 1000)), "length_cm": dims[0], "width_cm": dims[1],
                     "height_cm": dims[2]})
        pk_text.append(f"{kg:g} kg, {dims[0]} x {dims[1]} x {dims[2]} cm")
    services = {"ECONOMY": "6-10 business days, cheapest", "STANDARD": "3-5 business days",
                "EXPRESS": "1-2 business days", "SAME_DAY": "same-day courier, domestic only"}
    svc = r.choice(["ECONOMY", "STANDARD", "EXPRESS"])
    svc_text = {"ECONOMY": "the cheapest option, even if it takes over a week",
                "STANDARD": "the normal 3-5 business day service",
                "EXPRESS": "whatever gets there within two business days"}[svc]
    value = r.choice([4500, 12000, 38000, 50000, 72500, 150000])
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=14)
    args = {"sender": {"name": s["name"], "phone": se},
            "recipient": {"name": rc["name"], "address": {"street": street, "city": city, "postal_code": postal,
                                                           "country": cc}},
            "packages": pkgs, "service": svc, "pickup_date": iso(d), "declared_value_cents": value,
            "signature_required": value >= 50000}
    pk_clause = ("one box: " + pk_text[0]) if n_pk == 1 else \
        f"{num(r, n_pk)} boxes, in this order: " + "; ".join(pk_text)
    clauses = [f"I'm {s['name']} and you can reach me at {sp}", f"it goes to {rc['name']}, {street}, {postal} {city}, {cn}",
               pk_clause, f"I'd like {svc_text}", f"please have it picked up {dp}",
               f"the contents are worth {money_text(value, r)}"]
    return Instance(args, r.choice(["I need to send a shipment abroad.", "Can you book a parcel shipment for me?"]),
                    clauses, [op_p, op_d, "unit:kg_to_g", "enum:described", "rule:threshold", "nested:array"],
                    r.choice([None, f"My last shipment, {code(r, 'SHP-')}, arrived damaged, but that's a separate issue."]))


SHIPMENT = Domain("parcel_shipment", tool("create_shipment", "Book an international parcel shipment.", obj({
    "sender": obj({"name": S("Sender's full name as written."), "phone": S("Sender phone in E.164 format, e.g. +14155550132.", **E164)}),
    "recipient": obj({"name": S("Recipient's full name as written."), "address": obj({
        "street": S("Street line exactly as written."), "city": S("City."), "postal_code": S("Postal code."),
        "country": S("ISO 3166-1 alpha-2 country code.", pattern="^[A-Z]{2}$")})}),
    "packages": A(obj({"weight_g": I("Weight in grams.", minimum=1), "length_cm": I("Length in cm."),
                       "width_cm": I("Width in cm."), "height_cm": I("Height in cm.")}),
                  "One entry per box, in the order the user lists them.", minItems=1),
    "service": E("ECONOMY: 6-10 business days, cheapest; STANDARD: 3-5 business days; EXPRESS: 1-2 business days; "
                 "SAME_DAY: same-day courier, domestic only.", ["ECONOMY", "STANDARD", "EXPRESS", "SAME_DAY"]),
    "pickup_date": S("Pickup date, YYYY-MM-DD.", **DATE),
    "declared_value_cents": I("Declared value of the contents in US cents.", minimum=0),
    "signature_required": B("Must be true when declared_value_cents is 50000 or more, otherwise false.")})),
    (simple("track_shipment", "Get the status of an existing shipment.", tracking_number="Tracking number."),
     tool("get_shipping_quote", "Estimate a price without booking.", obj({
         "country": S("ISO 3166-1 alpha-2 destination."), "weight_g": I("Total weight in grams.")}))),
    _shipment, "shipment")


def _event(r, today, level):
    title = r.choice(["Q3 budget review", "Onboarding sync", "Design crit", "Vendor call", "Roadmap planning",
                      "Incident retro", "Hiring panel", "Customer demo"])
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=30)
    tp, t, op_t = time_phrase(r, level=level, lo=8, hi=18)
    durp, dur, op_m = duration_phrase(r, (30, 45, 60, 90, 120))
    k = r.choice([1, 2, 3]) if level >= 2 else 1
    people = [person(r) for _ in range(k)]
    args = {"title": title, "date": iso(d), "start_time": t, "duration_minutes": dur,
            "attendees": [p["email"] for p in people]}
    clauses = [f'call it "{title}"', f"it's {dp} at {tp}", f"it should last {durp}",
               "invite " + ", then ".join(f"{p['name']} ({p['email']})" for p in people)]
    ops = [op_d, op_t, op_m, "array:ordered"]
    if level >= 2 and r.random() < 0.6:
        rp, rm, _ = duration_phrase(r, (15, 30, 60))
        args["reminder_minutes_before"] = rm
        clauses.append(f"remind everyone {rp} before it starts")
        ops.append("optional:present")
    if level >= 3 and r.random() < 0.5:
        n = r.choice([3, 4, 6, 8])
        freq = r.choice(["WEEKLY", "DAILY"])
        args["recurrence"] = {"frequency": freq, "count": n}
        clauses.append(f"repeat it {'every week' if freq == 'WEEKLY' else 'every day'} for {num(r, n)} occurrences in total")
        ops.append("nested:optional_object")
    if level >= 2 and r.random() < 0.4:
        wrong = tp
        while wrong == tp:
            wrong = time_phrase(r, level=1, lo=8, hi=18)[0]
        clauses[1] = f"it's {dp} at {wrong} -- sorry, I had the time wrong: same day, but it starts at {tp}"
        ops.append("correction")
    return Instance(args, r.choice(["Please put a meeting on my calendar.", "Set up an event for me."]), clauses, ops)


EVENT = Domain("calendar_event", tool("schedule_calendar_event", "Create a calendar event and send invitations.", obj({
    "title": S("Title exactly as the user gives it."), "date": S("YYYY-MM-DD.", **DATE),
    "start_time": S("24-hour HH:MM.", **TIME), "duration_minutes": I("Length in minutes.", minimum=5),
    "attendees": A(S("Email address."), "Invitee emails in the order the user names them.", minItems=1),
    "reminder_minutes_before": I(f"Minutes before the start to send a reminder. {OPTIONAL}", minimum=1),
    "recurrence": obj({"frequency": E("DAILY or WEEKLY.", ["DAILY", "WEEKLY"]),
                       "count": I("Total number of occurrences, the first included.", minimum=2)},
                      desc=f"Repetition. {OPTIONAL}")},
    required=["title", "date", "start_time", "duration_minutes", "attendees"])),
    (simple("find_free_slot", "Search attendees' calendars for a free slot.", date="YYYY-MM-DD"),
     simple("cancel_event", "Cancel an event.", event_id="Event id.")),
    _event, "event")


def _restaurant(r, today, level):
    adults = r.randint(1, 6)
    kids = r.choice([0, 0, 1, 2, 3]) if level >= 2 else 0
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=21)
    tp, t, op_t = time_phrase(r, level=level, lo=17, hi=21)
    seat = r.choice(["INDOOR", "TERRACE", "BAR", "PRIVATE_ROOM"])
    seat_txt = {"INDOOR": "a regular table inside", "TERRACE": "a table outside on the terrace",
                "BAR": "seats at the counter", "PRIVATE_ROOM": "the separate room you can close off"}[seat]
    ph, e164, op_p = phone(r)
    args = {"party": {"adults": adults, "children": kids}, "date": iso(d), "time": t, "seating": seat,
            "contact_phone": e164}
    total = adults + kids
    party = (f"we're a party of {num(r, total)}, {num(r, kids)} of them kids" if kids else
             "it's just me" if adults == 1 else f"we're {num(r, adults)} adults")
    clauses = [party, f"{dp} at {tp}", f"we'd like {seat_txt}", f"my number is {ph}"]
    ops = [op_d, op_t, op_p, "enum:described", "arithmetic:subtract" if kids else "count:plain"]
    if level >= 2 and r.random() < 0.3:
        wrong = tp
        while wrong == tp:
            wrong = time_phrase(r, level=1, lo=17, hi=21)[0]
        clauses[1] = f"{dp} at {wrong} -- actually no, make that {tp}"
        ops.append("correction")
    if level >= 2 and r.random() < 0.5:
        diets = pick(r, ["VEGETARIAN", "VEGAN", "GLUTEN_FREE", "NUT_ALLERGY"], r.choice([1, 2]))
        names = {"VEGETARIAN": "someone vegetarian", "VEGAN": "a vegan", "GLUTEN_FREE": "someone who can't eat gluten",
                 "NUT_ALLERGY": "a guest with a nut allergy"}
        args["dietary"] = diets
        clauses.append("we have " + " and ".join(names[x] for x in diets))
        ops.append("optional:array")
    return Instance(args, r.choice(["Book us a table, please.", "I want to reserve dinner."]), clauses, ops,
                    r.choice([None, "Last time we were seated at 7 but that doesn't matter now."]))


RESTAURANT = Domain("restaurant_booking", tool("book_restaurant_table", "Reserve a table at the restaurant.", obj({
    "party": obj({"adults": I("Adults.", minimum=1), "children": I("Children under 12; 0 if none.", minimum=0)}),
    "date": S("YYYY-MM-DD.", **DATE), "time": S("24-hour HH:MM.", **TIME),
    "seating": E("INDOOR: dining room; TERRACE: outdoor terrace; BAR: counter seats; PRIVATE_ROOM: closed room.",
                 ["INDOOR", "TERRACE", "BAR", "PRIVATE_ROOM"]),
    "contact_phone": S("E.164 phone, e.g. +14155550132.", **E164),
    "dietary": A(E("Need.", ["VEGETARIAN", "VEGAN", "GLUTEN_FREE", "NUT_ALLERGY"]),
                 f"Dietary needs in the order mentioned. {OPTIONAL}", minItems=1)},
    required=["party", "date", "time", "seating", "contact_phone"])),
    (simple("get_menu", "Fetch today's menu.", section="Menu section."),
     simple("cancel_table_booking", "Cancel a reservation.", reservation_id="Reservation id.")),
    _restaurant, "reservation")


AIRPORTS = [("San Francisco", "SFO"), ("Boston Logan", "BOS"), ("Chicago O'Hare", "ORD"), ("Denver", "DEN"),
            ("Seattle-Tacoma", "SEA"), ("Miami", "MIA"), ("Atlanta", "ATL"), ("Austin", "AUS"), ("Newark", "EWR"),
            ("Phoenix Sky Harbor", "PHX")]


def _flights(r, today, level):
    (on, oc), (dn, dc) = pick(r, AIRPORTS, 2)
    dp, d, op_d = date_phrase(r, today, level=level, min_ahead=3, max_ahead=40)
    adults = r.randint(1, 3)
    kids = r.choice([0, 1, 2]) if level >= 2 else 0
    infants = r.choice([0, 0, 1]) if level >= 3 else 0
    cabin = r.choice(["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS"])
    cabin_txt = {"ECONOMY": "the regular economy cabin", "PREMIUM_ECONOMY": "the extra-legroom cabin between economy and business",
                 "BUSINESS": "business class"}[cabin]
    args = {"origin": oc, "destination": dc, "depart_date": iso(d),
            "passengers": {"adults": adults, "children": kids, "infants": infants}, "cabin": cabin}
    who = [f"{num(r, adults)} {'adult' if adults == 1 else 'adults'}"]
    ages = [r.randint(3, 11) for _ in range(kids)] + [r.choice([0, 1]) for _ in range(infants)]
    if ages:
        who.append("kids aged " + " and ".join(str(a) if a else "eight months" for a in ages))
    clauses = [f"from {on} ({oc}) to {dn} ({dc})", f"leaving {dp}", " plus ".join(who), f"in {cabin_txt}"]
    ops = [op_d, "enum:described", "rule:age_bands" if infants else "count:plain"]
    if r.random() < 0.5:
        n = r.choice([3, 4, 5, 7, 10])
        args["return_date"] = iso(d + timedelta(days=n))
        clauses.append(f"coming back {num(r, n)} days after I leave")
        ops.append("arithmetic:date_offset")
    if level >= 2 and r.random() < 0.5:
        args["max_stops"] = r.choice([0, 1])
        clauses.append("nonstop only" if args["max_stops"] == 0 else "at most one layover")
        ops.append("optional:present")
    return Instance(args, r.choice(["Find me some flights.", "Search flights please."]), clauses, ops)


FLIGHTS = Domain("flight_search", tool("find_flight_options", "Search available flights.", obj({
    "origin": S("IATA code.", pattern="^[A-Z]{3}$"), "destination": S("IATA code.", pattern="^[A-Z]{3}$"),
    "depart_date": S("YYYY-MM-DD.", **DATE),
    "return_date": S(f"YYYY-MM-DD for a round trip. {OPTIONAL}", **DATE),
    "passengers": obj({"adults": I("Age 12+.", minimum=1), "children": I("Ages 2-11.", minimum=0),
                       "infants": I("Under 2 years.", minimum=0)}),
    "cabin": E("ECONOMY, PREMIUM_ECONOMY or BUSINESS.", ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS"]),
    "max_stops": I(f"Maximum layovers. {OPTIONAL}", minimum=0, maximum=2)},
    required=["origin", "destination", "depart_date", "passengers", "cabin"])),
    (simple("get_booking", "Look up a booking.", record_locator="Six-character locator."),
     simple("seat_map", "Show the seat map of a flight.", flight_number="Flight number.")),
    _flights, "search")


def _hotel(r, today, level):
    hid = code(r, "HTL-", 4)
    dp, d, op_d = date_phrase(r, today, level=level, min_ahead=2, max_ahead=40)
    nights = r.randint(1, 9)
    n_rooms = r.choice([1, 1, 2]) if level >= 2 else 1
    rooms, rt = [], []
    for _ in range(n_rooms):
        typ = r.choice(["SINGLE", "DOUBLE", "TWIN", "SUITE"])
        g = {"SINGLE": 1, "DOUBLE": r.choice([1, 2]), "TWIN": 2, "SUITE": r.choice([2, 3, 4])}[typ]
        rooms.append({"type": typ, "guests": g})
        rt.append({"SINGLE": "a single room for one", "DOUBLE": f"a room with one big bed for {num(r, g)}",
                   "TWIN": "a room with two separate beds for two",
                   "SUITE": f"a suite for {num(r, g)}"}[typ])
    args = {"hotel_id": hid, "check_in": iso(d), "check_out": iso(d + timedelta(days=nights)), "rooms": rooms}
    clauses = [f"at hotel {hid}", f"arriving {dp}", f"staying {num(r, nights)} night{'s' if nights > 1 else ''}",
               "we need " + ("; and ".join(rt) if n_rooms > 1 else rt[0])]
    ops = [op_d, "arithmetic:date_offset", "enum:described", "nested:array"]
    if level >= 2 and r.random() < 0.5:
        args["breakfast"] = r.random() < 0.5
        clauses.append("with breakfast included" if args["breakfast"] else "without breakfast")
        ops.append("optional:present")
    return Instance(args, r.choice(["I'd like to book a hotel.", "Reserve rooms for me."]), clauses, ops, r.choice([None, f"(We stayed at {code(r, 'HTL-', 4)} last year.)"]))


HOTEL = Domain("hotel_booking", tool("reserve_hotel_rooms", "Book rooms at a hotel.", obj({
    "hotel_id": S("Hotel id as given."), "check_in": S("YYYY-MM-DD.", **DATE), "check_out": S("YYYY-MM-DD.", **DATE),
    "rooms": A(obj({"type": E("SINGLE (1 bed, 1 guest), DOUBLE (1 large bed), TWIN (2 beds), SUITE.",
                              ["SINGLE", "DOUBLE", "TWIN", "SUITE"]), "guests": I("Guests in the room.", minimum=1)}),
               "Rooms in the order the user lists them.", minItems=1),
    "breakfast": B(f"Breakfast included. {OPTIONAL}")}, required=["hotel_id", "check_in", "check_out", "rooms"])),
    (simple("hotel_availability", "Check availability.", hotel_id="Hotel id."),
     simple("read_hotel_reviews", "Read reviews.", hotel_id="Hotel id.")),
    _hotel, "booking")


PRODUCE = [("eggs", "SKU-EGG"), ("bagels", "SKU-BGL"), ("lemons", "SKU-LMN"), ("yogurt cups", "SKU-YGT"),
           ("bread rolls", "SKU-RLL"), ("limes", "SKU-LIM"), ("apples", "SKU-APL"), ("cans of tomatoes", "SKU-TOM")]


def _grocery(r, today, level):
    k = r.choice([1, 2, 3]) if level >= 2 else r.choice([1, 2])
    items, texts = [], []
    for name, sku in pick(r, PRODUCE, k):
        n = r.choice([6, 12, 24, 36, 8, 10, 15, 16, 20])
        t, _ = count_phrase(r, n, name, level=level)
        items.append({"sku": sku, "quantity": n})
        texts.append(t)
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=7)
    win = r.choice(["MORNING", "AFTERNOON", "EVENING"])
    wtxt = {"MORNING": "before noon", "AFTERNOON": "between noon and 5", "EVENING": "after 5 pm"}[win]
    args = {"items": items, "delivery": {"date": iso(d), "window": win}, "allow_substitutions": r.random() < 0.5}
    clauses = ["I need " + ", ".join(texts[:-1]) + (" and " if k > 1 else "") + texts[-1],
               f"deliver {dp}, {wtxt}",
               "substitutions are fine" if args["allow_substitutions"] else "please don't substitute anything"]
    catalog = "; ".join(f"{n} = {s}" for n, s in PRODUCE)
    return Instance(args, f"Grocery order please. (Catalog codes: {catalog}.)", clauses,
                    [op_d, "count:dozen", "enum:described", "array:ordered", "lookup:catalog"])


GROCERY = Domain("grocery_order", tool("place_grocery_order", "Place a grocery delivery order.", obj({
    "items": A(obj({"sku": S("Catalog code."), "quantity": I("Number of units.", minimum=1)}),
               "Items in the order the user mentions them.", minItems=1),
    "delivery": obj({"date": S("YYYY-MM-DD.", **DATE),
                     "window": E("MORNING (8-12), AFTERNOON (12-17), EVENING (17-21).",
                                 ["MORNING", "AFTERNOON", "EVENING"])}),
    "allow_substitutions": B("Whether the store may substitute out-of-stock items.")})),
    (simple("search_grocery_catalog", "Search the catalog.", query="Search text."),
     simple("order_status", "Status of an order.", order_id="Order id.")),
    _grocery, "order")


def _medication(r, today, level):
    med = r.choice(["amoxicillin", "ibuprofen", "metformin", "prednisone", "cetirizine", "omeprazole"])
    every = r.choice([6, 8, 12])
    start_h = r.choice([6, 7, 8]) if every != 6 else r.choice([0, 6])
    times = sorted(f"{(start_h + every * i) % 24:02d}:00" for i in range(24 // every))
    tablet_mg = r.choice([250, 400, 500, 1000])
    half = level >= 2 and r.random() < 0.5
    dose = tablet_mg // 2 if half else tablet_mg
    days = r.choice([5, 7, 10, 14, 21])
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=5)
    args = {"medication": med, "dose_mg": dose, "times": times, "start_date": iso(d), "duration_days": days,
            "with_food": r.random() < 0.5}
    st = f"{start_h % 12 or 12} {'am' if start_h < 12 else 'pm'}" if start_h else "midnight"
    clauses = [f"it's {med}", (f"half of a {tablet_mg} mg tablet each time" if half else f"{tablet_mg} mg each time"),
               f"every {num(r, every)} hours, the first dose at {st}",
               f"starting {dp}", ("for two weeks" if days == 14 else ("for three weeks" if days == 21 else f"for {num(r, days)} days")),
               "it has to be taken with food" if args["with_food"] else "it should be taken on an empty stomach"]
    return Instance(args, r.choice(["Set up medication reminders for me.", "I need a pill reminder schedule."]),
                    clauses, [op_d, "arithmetic:schedule", "array:sorted", "arithmetic:halve" if half else "count:plain"])


MEDICATION = Domain("medication_reminders", tool("create_medication_schedule", "Create a dose reminder schedule.", obj({
    "medication": S("Medication name in lowercase."), "dose_mg": I("Dose per intake in milligrams.", minimum=1),
    "times": A(S("24-hour HH:MM.", **TIME), "Daily intake times sorted from earliest to latest (00:00 first).", minItems=1),
    "start_date": S("YYYY-MM-DD.", **DATE), "duration_days": I("Number of days.", minimum=1),
    "with_food": B("True if it must be taken with food.")})),
    (simple("drug_interactions", "Check interactions.", medication="Name."),
     simple("refill_request", "Request a refill.", prescription_id="Prescription id.")),
    _medication, "schedule")


def _expenses(r, today, level):
    emp = code(r, "E", 6)
    k = r.choice([2, 3, 4]) if level >= 2 else 2
    cats = {"MEALS": ["dinner with the client", "lunch", "breakfast at the airport"], "TRAVEL": ["the train ticket", "a taxi"],
            "LODGING": ["the hotel"], "SUPPLIES": ["printer paper", "a USB cable"]}
    rows, texts, total = [], [], 0
    for _ in range(k):
        cat = r.choice(list(cats))
        what = r.choice(cats[cat])
        cents = r.randrange(800, 42000, 25)
        back = r.randint(1, 20)
        d = today - timedelta(days=back)
        rows.append({"date": iso(d), "category": cat, "amount_cents": cents})
        texts.append(f"{what} on {nice(d, r)}: {money_text(cents, r)}")
        total += cents
    rows.sort(key=lambda x: x["date"])
    args = {"employee_id": emp, "expenses": rows, "total_cents": total}
    clauses = [f"my employee id is {emp}", "the items are -- " + "; ".join(texts)]
    return Instance(args, "File an expense report for my recent trip.", clauses,
                    ["arithmetic:sum", "array:sorted", "enum:described", "unit:money_cents"],
                    r.choice([None, f"(Ignore report {code(r, 'R', 5)} from last month, that one is already paid.)"]))


EXPENSES = Domain("expense_report", tool("submit_expense_report", "Submit an expense report.", obj({
    "employee_id": S("Employee id."),
    "expenses": A(obj({"date": S("YYYY-MM-DD.", **DATE),
                       "category": E("MEALS (food and drink), TRAVEL (transport), LODGING (hotels), SUPPLIES (office items).",
                                     ["MEALS", "TRAVEL", "LODGING", "SUPPLIES"]),
                       "amount_cents": I("Amount in US cents.", minimum=1)}),
                  "Expenses sorted by date, earliest first (keep the user's order for the same date).", minItems=1),
    "total_cents": I("Sum of all amount_cents.", minimum=1)})),
    (simple("expense_policy", "Read the expense policy.", topic="Topic."),
     simple("report_status", "Status of a report.", report_id="Report id.")),
    _expenses, "report")


def _thermostat(r, today, level):
    zone = r.choice(["LIVING", "BEDROOM", "OFFICE", "BASEMENT"])
    days_kind = r.choice(["WEEKDAYS", "WEEKEND", "ALL"])
    days = {"WEEKDAYS": ["MON", "TUE", "WED", "THU", "FRI"], "WEEKEND": ["SAT", "SUN"],
            "ALL": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]}[days_kind]
    k = r.choice([2, 3]) if level >= 2 else 2
    hours = sorted(r.sample(range(5, 23), k))
    periods, texts = [], []
    for h in hours:
        temp = r.choice([16.5, 18, 19.5, 20, 21, 21.5, 22.5])
        periods.append({"start": f"{h:02d}:00", "target_tenths_c": int(round(temp * 10))})
        texts.append(f"{temp:g}°C from {h % 12 or 12} {'am' if h < 12 else 'pm'}")
    r.shuffle(texts)
    args = {"zone": zone, "days": days, "periods": periods}
    clauses = [f"for the {zone.lower()} zone", {"WEEKDAYS": "Monday to Friday", "WEEKEND": "on Saturdays and Sundays",
                                               "ALL": "every day of the week"}[days_kind],
               "set " + ", ".join(texts)]
    return Instance(args, "Program my thermostat.", clauses,
                    ["unit:tenths", "array:sorted", "enum:expand_days", "time:clock12"])


THERMOSTAT = Domain("thermostat_program", tool("set_heating_program", "Set a heating program for a zone.", obj({
    "zone": E("Zone.", ["LIVING", "BEDROOM", "OFFICE", "BASEMENT"]),
    "days": A(E("Day code.", ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]),
              "Days the program applies to, in calendar order MON..SUN.", minItems=1),
    "periods": A(obj({"start": S("24-hour HH:MM.", **TIME), "target_tenths_c": I("Target temperature in tenths of a degree Celsius (21.5 C = 215).")}),
                 "Periods sorted by start time.", minItems=1)})),
    (simple("current_temperature", "Read the current temperature.", zone="Zone."),
     simple("set_away_mode", "Enable away mode.", until="YYYY-MM-DD")),
    _thermostat, "program")


REGIONS = {"us-east": "Virginia", "us-west": "Oregon", "eu-central": "Frankfurt", "eu-west": "Dublin",
           "ap-northeast": "Tokyo", "ap-south": "Mumbai"}


def _vm(r, today, level):
    region = r.choice(list(REGIONS))
    name = f"{r.choice(['build', 'etl', 'web', 'cache', 'batch'])}-{r.randint(1, 99):02d}"
    cores = r.choice([2, 4, 8, 16])
    gb = r.choice([4, 8, 16, 32, 64])
    k = r.choice([1, 2]) if level >= 2 else 1
    disks = [{"size_gib": r.choice([50, 100, 200, 500]), "type": r.choice(["SSD", "HDD"])} for _ in range(k)]
    args = {"name": name, "region": region, "cpu_cores": cores, "memory_mib": gb * 1024, "disks": disks}
    dtxt = [f"a {d['size_gib']} GB {'solid-state' if d['type'] == 'SSD' else 'spinning'} disk" for d in disks]
    clauses = [f'name it "{name}"', f"put it in the data center in {REGIONS[region]}", f"{num(r, cores)} cores",
               f"{gb} GB of memory", " and ".join(dtxt)]
    ops = ["enum:described", "unit:gb_to_mib", "nested:array"]
    if level >= 2 and r.random() < 0.6:
        tags = {"team": r.choice(["data", "infra", "ml", "web"]), "env": r.choice(["dev", "staging", "prod"])}
        args["tags"] = tags
        clauses.append(f"tag it team={tags['team']} and env={tags['env']}")
        ops.append("optional:object")
    return Instance(args, "Spin up a virtual machine.", clauses, ops)


VM = Domain("vm_provisioning", tool("create_instance", "Create a virtual machine.", obj({
    "name": S("Instance name as given."),
    "region": E("us-east (Virginia), us-west (Oregon), eu-central (Frankfurt), eu-west (Dublin), ap-northeast (Tokyo), ap-south (Mumbai).",
                list(REGIONS)),
    "cpu_cores": I("vCPU count.", minimum=1), "memory_mib": I("Memory in MiB; 1 GB in a request means 1024 MiB.", minimum=512),
    "disks": A(obj({"size_gib": I("Size; 1 GB in a request means 1 GiB.", minimum=1), "type": E("SSD or HDD.", ["SSD", "HDD"])}),
               "Disks in the order the user lists them.", minItems=1),
    "tags": obj({"team": S("Team tag."), "env": S("Environment tag.")}, desc=f"Labels. {OPTIONAL}")},
    required=["name", "region", "cpu_cores", "memory_mib", "disks"])),
    (simple("list_instances", "List instances in a region.", region="Region."),
     simple("stop_instance", "Stop an instance.", instance_id="Instance id.")),
    _vm, "instance")


def _orders_query(r, today, level):
    statuses = pick(r, ["PENDING", "PAID", "SHIPPED", "CANCELLED", "REFUNDED"], r.choice([1, 2, 3]))
    after = today - timedelta(days=r.choice([7, 14, 30, 60, 90]))
    sort_f, sort_d = r.choice(["created_at", "total_cents"]), r.choice(["ASC", "DESC"])
    limit = r.choice([10, 20, 25, 50, 100])
    fields = pick(r, ["order_id", "customer_email", "total_cents", "status", "created_at"], r.choice([2, 3, 4]))
    args = {"filters": {"status_in": statuses, "created_after": iso(after)}, "sort": {"field": sort_f, "direction": sort_d},
            "limit": limit, "fields": fields}
    days = (today - after).days
    clauses = ["orders that are " + " or ".join(s.lower() for s in statuses),
               f"created after {nice(after, r)}" if level < 2 else f"from the last {num(r, days)} days (created after that date)",
               ("newest first" if (sort_f, sort_d) == ("created_at", "DESC") else
                "oldest first" if sort_f == "created_at" else
                "biggest totals first" if sort_d == "DESC" else "smallest totals first"),
               f"only the top {limit}", "show " + ", ".join(f.replace("_", " ") for f in fields)]
    ops = ["array:ordered", "enum:described", "date:back_offset" if level >= 2 else "date:absolute"]
    if level >= 2 and r.random() < 0.5:
        m = r.choice([2000, 5000, 10000, 25000])
        args["filters"]["min_total_cents"] = m
        clauses.append(f"with a total of at least {money_text(m, r)}")
        ops.append("optional:present")
    return Instance(args, "Pull an orders report from the database.", clauses, ops)


ORDERS_QUERY = Domain("orders_report", tool("query_orders", "Query the orders table.", obj({
    "filters": obj({"status_in": A(E("Status.", ["PENDING", "PAID", "SHIPPED", "CANCELLED", "REFUNDED"]),
                                   "Statuses in the order mentioned.", minItems=1),
                    "created_after": S("YYYY-MM-DD, exclusive.", **DATE),
                    "min_total_cents": I(f"Minimum order total in cents. {OPTIONAL}", minimum=0)},
                   required=["status_in", "created_after"]),
    "sort": obj({"field": E("Sort field.", ["created_at", "total_cents"]), "direction": E("ASC or DESC.", ["ASC", "DESC"])}),
    "limit": I("Maximum rows.", minimum=1, maximum=500),
    "fields": A(E("Column.", ["order_id", "customer_email", "total_cents", "status", "created_at"]),
                "Columns in the order mentioned.", minItems=1)})),
    (simple("describe_table", "Show a table's columns.", table="Table name."),
     simple("export_csv", "Export a saved query.", query_id="Query id.")),
    _orders_query, "query")


def _tickets(r, today, level):
    eid = code(r, "EVT-", 5)
    prices = {"GENERAL": r.choice([3500, 4500, 5500]), "RESERVED": r.choice([8000, 9500]), "VIP": r.choice([15000, 18000])}
    k = r.choice([1, 2]) if level >= 2 else 1
    tiers = pick(r, list(prices), k)
    rows, texts, total = [], [], 0
    for t in tiers:
        q = r.randint(1, 6)
        rows.append({"tier": t, "quantity": q})
        texts.append(f"{num(r, q)} {t.lower()}")
        total += q * prices[t]
    args = {"event_id": eid, "tickets": rows, "delivery": r.choice(["MOBILE", "PRINT_AT_HOME", "WILL_CALL"]),
            "expected_total_cents": total}
    dtxt = {"MOBILE": "send them to my phone app", "PRINT_AT_HOME": "a PDF I can print",
            "WILL_CALL": "I'll pick them up at the box office"}[args["delivery"]]
    ptxt = ", ".join(f"{t.lower()} {money_text(p, r)}" for t, p in prices.items())
    clauses = [f"for event {eid}", f"prices are {ptxt} each", "I want " + " and ".join(texts), dtxt]
    return Instance(args, "Buy concert tickets for me.", clauses,
                    ["arithmetic:sum_product", "enum:described", "nested:array"])


TICKETS = Domain("ticket_purchase", tool("buy_tickets", "Buy event tickets.", obj({
    "event_id": S("Event id as given."),
    "tickets": A(obj({"tier": E("GENERAL, RESERVED or VIP.", ["GENERAL", "RESERVED", "VIP"]), "quantity": I("Count.", minimum=1)}),
                 "One entry per tier, in the order mentioned.", minItems=1),
    "delivery": E("MOBILE (app), PRINT_AT_HOME (PDF), WILL_CALL (box office pickup).", ["MOBILE", "PRINT_AT_HOME", "WILL_CALL"]),
    "expected_total_cents": I("Sum of quantity x tier price in cents, used to confirm the charge.", minimum=1)})),
    (simple("get_event_info", "Event information.", event_id="Event id."),
     simple("seat_availability", "Remaining seats.", event_id="Event id.")),
    _tickets, "order")


def _car(r, today, level):
    loc = code(r, "LOC", 3)
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=30)
    tp, t, op_t = time_phrase(r, level=level, lo=8, hi=18)
    days = r.randint(1, 10)
    seats = r.choice([2, 4, 5, 7])
    cls = {2: "COMPACT", 4: "COMPACT", 5: "SEDAN", 7: "MINIVAN"}[seats]
    k = r.choice([1, 2]) if level >= 2 else 1
    drivers = []
    for _ in range(k):
        p = person(r)
        drivers.append({"name": p["name"], "age": r.randint(21, 70)})
    args = {"pickup": {"location": loc, "date": iso(d), "time": t},
            "dropoff": {"location": loc, "date": iso(d + timedelta(days=days)), "time": t},
            "car_class": cls, "drivers": drivers}
    clauses = [f"pick up at {loc} {dp} at {tp}", f"return it to the same place {num(r, days)} days later at the same time",
               f"it needs to seat {num(r, seats)}",
               "drivers: " + "; ".join(f"{x['name']}, {x['age']}" for x in drivers)]
    return Instance(args, "Reserve a rental car.", clauses,
                    [op_d, op_t, "arithmetic:date_offset", "enum:by_rule", "nested:array"])


CAR = Domain("car_rental", tool("reserve_car", "Reserve a rental car.", obj({
    "pickup": obj({"location": S("Location code."), "date": S("YYYY-MM-DD.", **DATE), "time": S("HH:MM.", **TIME)}),
    "dropoff": obj({"location": S("Location code."), "date": S("YYYY-MM-DD.", **DATE), "time": S("HH:MM.", **TIME)}),
    "car_class": E("COMPACT seats up to 4, SEDAN up to 5, MINIVAN up to 7: choose the smallest class that seats everyone.",
                   ["COMPACT", "SEDAN", "MINIVAN"]),
    "drivers": A(obj({"name": S("Full name."), "age": I("Age in years.", minimum=18)}), "Drivers in the order listed.", minItems=1)})),
    (simple("rental_locations", "List rental locations.", city="City."),
     simple("rental_status", "Status of a rental.", rental_id="Rental id.")),
    _car, "rental")


def _recipe(r, today, level):
    base, target = r.choice([(2, 4), (4, 6), (4, 2), (6, 9), (3, 6), (8, 4)])
    ings = pick(r, [("flour", "g"), ("sugar", "g"), ("butter", "g"), ("milk", "ml"), ("rice", "g"), ("stock", "ml"),
                    ("olive oil", "ml"), ("lentils", "g")], r.choice([2, 3, 4]))
    rows, texts = [], []
    for name, unit in ings:
        q = r.choice([60, 120, 180, 240, 300, 360, 480, 600])
        rows.append({"name": name, "quantity": q * target // base, "unit": unit})
        texts.append(f"{q} {unit} {name}")
    title = r.choice(["Sunday pancakes", "Lentil soup", "Rice pudding", "Shortbread"])
    args = {"title": title, "servings": target, "ingredients": rows}
    clauses = [f'my "{title}" recipe serves {num(r, base)}', "it uses " + ", ".join(texts),
               f"save a version for {num(r, target)} people with the amounts scaled"]
    return Instance(args, "Save a scaled recipe to my cookbook.", clauses,
                    ["arithmetic:scale", "array:ordered"])


RECIPE = Domain("recipe_scaling", tool("save_recipe", "Save a recipe card.", obj({
    "title": S("Recipe title as given."), "servings": I("Servings of this version.", minimum=1),
    "ingredients": A(obj({"name": S("Ingredient."), "quantity": I("Amount for these servings."),
                          "unit": E("g or ml.", ["g", "ml"])}), "Ingredients in the order given.", minItems=1)})),
    (simple("find_saved_recipes", "Search recipes.", query="Text."),
     simple("nutrition_facts", "Nutrition of an ingredient.", ingredient="Name.")),
    _recipe, "recipe")


def _invoice(r, today, level):
    c = person(r)
    k = r.choice([1, 2, 3])
    items = []
    for _ in range(k):
        desc = r.choice(["Design workshop", "Logo revisions", "Hosting (annual)", "Copywriting", "Site audit",
                         "Training session"])
        items.append({"description": desc, "quantity": r.randint(1, 5), "unit_price_cents": r.randrange(5000, 90000, 500)})
    net = r.choice([14, 30, 45])
    args = {"client": {"name": c["name"], "email": c["email"]}, "line_items": items,
            "due_date": iso(today + timedelta(days=net)), "currency": r.choice(["USD", "EUR", "GBP"])}
    sym = {"USD": "$", "EUR": "€", "GBP": "£"}[args["currency"]]
    clauses = [f"bill {c['name']} at {c['email']}",
               "; ".join(f"{i['quantity']} x {i['description']} at {money_text(i['unit_price_cents'], r, sym)} each" for i in items),
               f"payment terms net {net} from today"]
    ops = ["arithmetic:date_offset", "unit:money_cents", "enum:currency_symbol", "nested:array"]
    if level >= 2 and r.random() < 0.5:
        args["discount_percent"] = r.choice([5, 10, 15, 20])
        clauses.append(f"give them {args['discount_percent']}% off")
        ops.append("optional:present")
    return Instance(args, "Create an invoice.", clauses, ops)


INVOICE = Domain("invoicing", tool("issue_client_invoice", "Create and send an invoice.", obj({
    "client": obj({"name": S("Client name."), "email": S("Client email.")}),
    "line_items": A(obj({"description": S("Exactly as given."), "quantity": I("Units.", minimum=1),
                         "unit_price_cents": I("Price per unit in minor units (cents/pence).", minimum=1)}),
                    "Lines in the order given.", minItems=1),
    "due_date": S("YYYY-MM-DD: issue date (today) plus the payment term in days.", **DATE),
    "currency": E("USD ($), EUR (€), GBP (£).", ["USD", "EUR", "GBP"]),
    "discount_percent": I(f"Whole-invoice discount. {OPTIONAL}", minimum=1, maximum=100)},
    required=["client", "line_items", "due_date", "currency"])),
    (simple("invoice_status", "Status of an invoice.", invoice_number="Number."),
     simple("list_clients", "List clients.", query="Name filter.")),
    _invoice, "invoice")


def _pet(r, today, level):
    name = r.choice(["Biscuit", "Luna", "Miso", "Pepper", "Juniper", "Toast"])
    species = r.choice(["DOG", "CAT"])
    kg = r.choice([3.5, 4.2, 5.25, 6]) if species == "CAT" else r.choice([6, 8.75, 12.5, 22, 31.25])
    years = r.randint(0, 9)
    if years == 0:
        months = r.choice([3, 6, 9])
        age_txt = f"{num(r, months)} months old"
    else:
        months = r.choice([0, 3, 6])
        age_txt = (f"{num(r, years)} and a half years old" if months == 6 else
                   f"{num(r, years)} years and {num(r, months)} months old" if months else
                   f"{num(r, years)} year{'s' if years > 1 else ''} old")
    age_m = years * 12 + months
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=21)
    tp, t, op_t = time_phrase(r, level=level, lo=9, hi=16)
    svcs = pick(r, ["BATH", "HAIRCUT", "NAIL_TRIM", "TEETH"], r.choice([1, 2, 3]))
    ph, e164, op_p = phone(r)
    args = {"pet": {"name": name, "species": species, "weight_g": int(round(kg * 1000)), "age_months": age_m},
            "services": svcs, "date": iso(d), "time": t, "owner_phone": e164}
    sv = {"BATH": "a bath", "HAIRCUT": "a trim of the coat", "NAIL_TRIM": "nails clipped", "TEETH": "teeth brushed"}
    clauses = [f"my {species.lower()} {name} is {age_txt} and weighs {kg:g} kg", "needs " + ", ".join(sv[s] for s in svcs),
               f"{dp} at {tp}", f"call me at {ph}"]
    return Instance(args, "Book a grooming appointment.", clauses,
                    [op_d, op_t, op_p, "unit:kg_to_g", "arithmetic:age_months", "array:ordered"])


PET = Domain("pet_grooming", tool("book_grooming", "Book a pet grooming appointment.", obj({
    "pet": obj({"name": S("Pet name."), "species": E("DOG or CAT.", ["DOG", "CAT"]), "weight_g": I("Weight in grams."),
                "age_months": I("Age in whole months.", minimum=0)}),
    "services": A(E("BATH, HAIRCUT (coat trim), NAIL_TRIM, TEETH (brushing).", ["BATH", "HAIRCUT", "NAIL_TRIM", "TEETH"]),
                  "Services in the order mentioned.", minItems=1),
    "date": S("YYYY-MM-DD.", **DATE), "time": S("HH:MM.", **TIME), "owner_phone": S("E.164.", **E164)})),
    (simple("groomer_availability", "Free slots.", date="YYYY-MM-DD"),
     simple("vaccination_record", "Fetch a pet's vaccinations.", pet_id="Pet id.")),
    _pet, "appointment")


PLANS = {"STARTER": "up to 3 seats, no SSO, no audit logs", "TEAM": "up to 25 seats, SSO, no audit logs",
         "BUSINESS": "up to 200 seats, SSO and audit logs", "ENTERPRISE": "unlimited seats, SSO and audit logs"}
#: seat cap, SSO, audit logs: the cheapest plan meeting every need wins (`PLANS` order is the price order)
PLAN_RULES = {"STARTER": (3, False, False), "TEAM": (25, True, False), "BUSINESS": (200, True, True),
              "ENTERPRISE": (10 ** 9, True, True)}


def _subscription(r, today, level):
    acct = code(r, "AC-", 6)
    seats = r.choice([2, 3, 5, 12, 25, 40, 150, 320])
    sso, audit = r.random() < 0.5, r.random() < 0.3
    plan = next(p for p, (cap, has_sso, has_audit) in PLAN_RULES.items()
                if seats <= cap and (has_sso or not sso) and (has_audit or not audit))
    cycle = r.choice(["MONTHLY", "ANNUAL"])
    y, m = next_month(today)
    args = {"account_id": acct, "plan": plan, "seats": seats, "billing_cycle": cycle, "effective_date": iso(date(y, m, 1))}
    clauses = [f"account {acct}", f"we now have {num(r, seats)} people who need seats",
               "we need single sign-on" if sso else "we don't need single sign-on",
               "we must have audit logs" if audit else "audit logs aren't required",
               "bill us once a year" if cycle == "ANNUAL" else "keep monthly billing",
               "switch at the start of next month"]
    return Instance(args, "Change our subscription to the cheapest plan that fits.", clauses,
                    ["enum:by_rule", "date:first_of_next_month"])


SUBSCRIPTION = Domain("subscription_change", tool("change_plan", "Change a subscription plan.", obj({
    "account_id": S("Account id."),
    "plan": E("; ".join(f"{k}: {v}" for k, v in PLANS.items()) + " (listed cheapest first).", list(PLANS)),
    "seats": I("Seats.", minimum=1), "billing_cycle": E("MONTHLY or ANNUAL.", ["MONTHLY", "ANNUAL"]),
    "effective_date": S("YYYY-MM-DD.", **DATE)})),
    (simple("current_plan", "Show the current plan.", account_id="Account id."),
     simple("billing_history", "List invoices.", account_id="Account id.")),
    _subscription, "change")


def _timesheet(r, today, level):
    monday = today - timedelta(days=today.weekday() + 7)
    proj = pick(r, ["PRJ-ALPHA", "PRJ-DELTA", "PRJ-ORBIT", "PRJ-LUMEN"], r.choice([1, 2]))
    entries, texts, total = [], [], 0
    for i in sorted(r.sample(range(5), r.choice([3, 4, 5]))):
        p = r.choice(proj)
        if r.random() < 0.6:
            hours = r.choice([8, 9, 10])
            mins, txt = hours * 60, f"{'an' if hours == 8 else 'a'} {words(hours)}-hour day"
        else:
            txt, mins, _ = duration_phrase(r, (120, 150, 180, 210, 240))
        entries.append({"date": iso(monday + timedelta(days=i)), "project": p, "minutes": mins})
        texts.append(f"{WEEKDAYS[i]}: {txt} on {p}")
        total += mins
    args = {"week_start": iso(monday), "entries": entries, "overtime": total > 2400}
    clauses = ["; ".join(texts)]
    return Instance(args, "Log my hours from last week.", clauses,
                    ["date:week_start", "duration:minutes", "rule:threshold", "array:sorted"])


TIMESHEET = Domain("timesheet", tool("submit_timesheet", "Submit a weekly timesheet.", obj({
    "week_start": S("Monday of the week, YYYY-MM-DD.", **DATE),
    "entries": A(obj({"date": S("YYYY-MM-DD.", **DATE), "project": S("Project code."), "minutes": I("Minutes.", minimum=1)}),
                 "Entries in date order.", minItems=1),
    "overtime": B("True when the minutes add up to more than 2400 for the week.")})),
    (simple("project_codes", "List project codes.", team="Team."),
     simple("leave_balance", "Leave balance.", employee_id="Employee id.")),
    _timesheet, "timesheet")


def _lab(r, today, level):
    pid = code(r, "PT", 7)
    k = r.choice([1, 2, 3])
    rows, texts = [], []
    for _ in range(k):
        typ = r.choice(["BLOOD", "URINE", "SALIVA"])
        ml = r.choice([0.5, 1, 1.5, 2, 2.5, 5])
        rows.append({"type": typ, "volume_ul": int(round(ml * 1000))})
        texts.append(f"{ml:g} mL of {typ.lower()}")
    tp, t, op_t = time_phrase(r, level=level, lo=6, hi=11)
    args = {"patient_id": pid, "samples": rows, "collected_at": {"date": iso(today), "time": t},
            "fasting": r.random() < 0.5, "priority": r.choice(["ROUTINE", "URGENT"])}
    clauses = [f"patient {pid}", "we drew " + ", ".join(texts), f"collected this morning at {tp}",
               "the patient fasted overnight" if args["fasting"] else "the patient had breakfast first",
               "results needed today" if args["priority"] == "URGENT" else "no rush on these"]
    return Instance(args, "Register lab samples.", clauses,
                    [op_t, "unit:ml_to_ul", "date:today", "enum:described", "nested:array"])


LAB = Domain("lab_samples", tool("register_samples", "Register collected lab samples.", obj({
    "patient_id": S("Patient id."),
    "samples": A(obj({"type": E("BLOOD, URINE or SALIVA.", ["BLOOD", "URINE", "SALIVA"]), "volume_ul": I("Volume in microliters.")}),
                 "Samples in the order listed.", minItems=1),
    "collected_at": obj({"date": S("YYYY-MM-DD.", **DATE), "time": S("HH:MM.", **TIME)}),
    "fasting": B("Patient was fasting."), "priority": E("ROUTINE or URGENT (same-day results).", ["ROUTINE", "URGENT"])})),
    (simple("lab_results", "Fetch results.", accession="Accession number."),
     simple("test_catalog", "Look up a test.", name="Test name.")),
    _lab, "accession")


LANGS = [("French", "fr"), ("German", "de"), ("Japanese", "ja"), ("Portuguese", "pt"), ("Korean", "ko"),
         ("Spanish", "es"), ("Italian", "it"), ("Dutch", "nl"), ("Polish", "pl")]


def _translation(r, today, level):
    src = ("English", "en")
    tg = pick(r, LANGS, r.choice([1, 2, 3]))
    words_n = r.choice([800, 1200, 2500, 4000, 6000])
    dp, d, op_d = date_phrase(r, today, level=level, min_ahead=3, max_ahead=30)
    dom = r.choice(["LEGAL", "MEDICAL", "MARKETING", "TECHNICAL"])
    args = {"source_lang": "en", "target_langs": [c for _, c in tg], "word_count": words_n, "deadline": iso(d),
            "domain": dom}
    dtxt = {"LEGAL": "a contract", "MEDICAL": "patient information leaflets", "MARKETING": "ad copy for a campaign",
            "TECHNICAL": "a user manual for a router"}[dom]
    clauses = [f"it's {dtxt} in {src[0]}", "into " + ", then ".join(n for n, _ in tg), f"about {words_n:,} words",
               f"needed by {dp}"]
    ops = [op_d, "lookup:language_codes", "enum:described", "array:ordered"]
    if level >= 2 and r.random() < 0.4:
        args["certified"] = True
        clauses.append("we need a certified translation")
        ops.append("optional:present")
    return Instance(args, "I need a translation job set up.", clauses, ops)


TRANSLATION = Domain("translation_job", tool("create_translation_job", "Order a translation.", obj({
    "source_lang": S("ISO 639-1 code.", pattern="^[a-z]{2}$"),
    "target_langs": A(S("ISO 639-1 code.", pattern="^[a-z]{2}$"), "Targets in the order mentioned.", minItems=1),
    "word_count": I("Word count.", minimum=1), "deadline": S("YYYY-MM-DD.", **DATE),
    "domain": E("LEGAL, MEDICAL, MARKETING or TECHNICAL.", ["LEGAL", "MEDICAL", "MARKETING", "TECHNICAL"]),
    "certified": B(f"Certified translation. {OPTIONAL}")}, required=["source_lang", "target_langs", "word_count", "deadline", "domain"])),
    (simple("quote_translation", "Price estimate.", word_count="Words."),
     simple("job_status", "Status.", job_id="Job id.")),
    _translation, "job")


def _parking(r, today, level):
    lot = r.choice(["P1", "P2", "P3", "P4"])
    plate_raw = f"{alnum(r, 3)} {r.randint(100, 999)}"
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=14)
    tp, t, op_t = time_phrase(r, level=level, lo=6, hi=20)
    durp, dur, op_m = duration_phrase(r)
    vt = r.choice(["CAR", "MOTORCYCLE", "VAN"])
    args = {"lot": lot, "vehicle": {"plate": plate_raw.replace(" ", "").upper(), "type": vt},
            "start": {"date": iso(d), "time": t}, "duration_minutes": dur, "ev_charging": r.random() < 0.4}
    clauses = [f"lot {lot}", f"my {vt.lower()}, plate {plate_raw.lower() if level >= 2 else plate_raw}",
               f"from {dp} at {tp}", (f"for {duration_phrase(r, [m for m in DURATIONS if m != dur])[0]}, or rather "
                                      f"{durp} to be safe" if level >= 2 and r.random() < 0.3 else f"for {durp}"),
               "I need a charger for my electric car" if args["ev_charging"] else "no charging needed"]
    return Instance(args, "Reserve a parking spot.", clauses,
                    [op_d, op_t, op_m, "normalize:plate", "nested:object"])


PARKING = Domain("parking_reservation", tool("reserve_parking", "Reserve a parking space.", obj({
    "lot": E("Lot.", ["P1", "P2", "P3", "P4"]),
    "vehicle": obj({"plate": S("Plate in uppercase with no spaces.", pattern="^[A-Z0-9]+$"),
                    "type": E("CAR, MOTORCYCLE or VAN.", ["CAR", "MOTORCYCLE", "VAN"])}),
    "start": obj({"date": S("YYYY-MM-DD.", **DATE), "time": S("HH:MM.", **TIME)}),
    "duration_minutes": I("Minutes.", minimum=15), "ev_charging": B("Needs an EV charging space.")})),
    (simple("lot_occupancy", "Free spaces in a lot.", lot="Lot."),
     simple("extend_parking", "Extend a reservation.", permit="Permit id.")),
    _parking, "permit")


def _warehouse(r, today, level):
    frm, to = pick(r, ["WH-NORTH", "WH-SOUTH", "WH-EAST", "WH-WEST"], 2)
    k = r.choice([1, 2, 3])
    lines, texts = [], []
    for _ in range(k):
        sku = r.choice(["HZ-", "GN-"]) + str(r.randint(1000, 9999))
        per = r.choice([6, 12, 24])
        cases = r.randint(2, 15)
        lines.append({"sku": sku, "units": per * cases})
        texts.append(f"{num(r, cases)} cases of {sku} ({per} per case)")
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=10)
    args = {"from_site": frm, "to_site": to, "lines": lines, "ship_date": iso(d),
            "hazmat": any(x["sku"].startswith("HZ-") for x in lines)}
    clauses = [f"move stock from {frm} to {to}", "; ".join(texts), f"ship {dp}"]
    ops = [op_d, "arithmetic:multiply", "rule:prefix", "nested:array"]
    if level >= 2 and r.random() < 0.3:
        other = date_phrase(r, today, level=1, max_ahead=10)
        if other[1] != d:
            clauses[2] = f"ship {other[0]} -- no, scratch that, ship {dp}"
            ops.append("correction")
    return Instance(args, "Create a stock transfer.", clauses, ops)


WAREHOUSE = Domain("warehouse_transfer", tool("create_transfer", "Transfer stock between warehouses.", obj({
    "from_site": S("Origin site."), "to_site": S("Destination site."),
    "lines": A(obj({"sku": S("SKU."), "units": I("Individual units (cases x units per case).", minimum=1)}),
               "Lines in the order given.", minItems=1),
    "ship_date": S("YYYY-MM-DD.", **DATE),
    "hazmat": B("True when any SKU starts with HZ- (hazardous goods).")})),
    (simple("stock_level", "Units on hand.", sku="SKU."),
     simple("transfer_status", "Status.", transfer_id="Transfer id.")),
    _warehouse, "transfer")


def _library(r, today, level):
    card = code(r, "LIB", 8)
    k = r.choice([1, 2, 3])
    isbns = []
    for _ in range(k):
        digits = "978" + "".join(str(r.randint(0, 9)) for _ in range(10))
        isbns.append(digits)
    written = [f"{s[:3]}-{s[3]}-{s[4:8]}-{s[8:12]}-{s[12]}" for s in isbns]
    branch = r.choice(["CENTRAL", "RIVERSIDE", "HILLTOP"])
    notify = r.choice(["EMAIL", "SMS"])
    args = {"card_number": card, "isbns": isbns, "pickup_branch": branch, "notify_by": notify}
    clauses = [f"my card is {card}", "the books are " + ", ".join(written),
               f"I'll collect them at the {branch.lower()} branch", "text me" if notify == "SMS" else "email me when they're in"]
    return Instance(args, "Place holds on some books.", clauses,
                    ["normalize:isbn", "array:ordered", "enum:described"])


LIBRARY = Domain("library_holds", tool("place_holds", "Place holds on books.", obj({
    "card_number": S("Library card."), "isbns": A(S("ISBN-13 digits only, no hyphens.", pattern="^[0-9]{13}$"),
                                                  "In the order given.", minItems=1),
    "pickup_branch": E("Branch.", ["CENTRAL", "RIVERSIDE", "HILLTOP"]), "notify_by": E("EMAIL or SMS.", ["EMAIL", "SMS"])})),
    (simple("search_catalog", "Search books.", query="Text."),
     simple("renew_loan", "Renew a loan.", loan_id="Loan id.")),
    _library, "hold")


def _payment(r, today, level):
    acct_raw = f"{r.randint(1000, 9999)} {r.randint(1000, 9999)} {r.randint(10, 99)}"
    cents = r.randrange(2000, 40000, 1)
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=20)
    last4 = f"{r.randint(0, 9999):04d}"
    mtype = r.choice(["CARD", "BANK"])
    args = {"account_number": acct_raw.replace(" ", ""), "amount_cents": cents, "pay_date": iso(d),
            "method": {"type": mtype, "last4": last4}}
    wrong = cents + r.choice([-500, -100, 100, 1000])
    pay = (f"pay {money_text(wrong, r)} -- sorry, I misread the bill, it's {money_text(cents, r)}"
           if level >= 2 and r.random() < 0.3 else f"pay {money_text(cents, r)}")
    clauses = [f"my utility account is {acct_raw}", pay, f"on {dp}",
               f"use my {'card' if mtype == 'CARD' else 'bank account'} ending in {last4}"]
    ops = [op_d, "normalize:digits", "unit:money_cents", "nested:object"]
    if level >= 2 and r.random() < 0.4:
        dom = r.choice([1, 5, 15, 28])
        args["autopay"] = {"day_of_month": dom}
        clauses.append(f"and set up autopay on day {dom} of every month")
        ops.append("optional:object")
    return Instance(args, "Pay my electricity bill.", clauses, ops,
                    r.choice([None, f"My previous payment of {money_text(r.randrange(2000, 40000), r)} went through fine."]))


PAYMENT = Domain("utility_payment", tool("pay_bill", "Pay a utility bill.", obj({
    "account_number": S("Digits only.", pattern="^[0-9]+$"), "amount_cents": I("Amount in cents.", minimum=1),
    "pay_date": S("YYYY-MM-DD.", **DATE),
    "method": obj({"type": E("CARD or BANK.", ["CARD", "BANK"]), "last4": S("Last four digits.", pattern="^[0-9]{4}$")}),
    "autopay": obj({"day_of_month": I("Day.", minimum=1, maximum=28)}, desc=f"Recurring payment. {OPTIONAL}")},
    required=["account_number", "amount_cents", "pay_date", "method"])),
    (simple("bill_balance", "Current balance.", account_number="Account."),
     simple("usage_history", "Usage history.", account_number="Account.")),
    _payment, "payment")


def _conference(r, today, level):
    p = person(r)
    cutoff = today + timedelta(days=r.choice([-10, -3, 5, 20]))
    student = r.random() < 0.3
    ttype = "STUDENT" if student else ("EARLY_BIRD" if today <= cutoff else "STANDARD")
    ws = pick(r, ["WS-101", "WS-204", "WS-317", "WS-422"], r.choice([0, 1, 2]))
    size = r.choice(["S", "M", "L", "XL"])
    args = {"attendee": {"name": p["name"], "email": p["email"]}, "ticket_type": ttype, "workshops": ws, "tshirt_size": size}
    clauses = [f"register {p['name']} ({p['email']})", "they're a full-time student" if student else "they're not a student",
               ("workshops " + " and ".join(ws)) if ws else "no workshops", f"t-shirt size {size}"]
    tool_note = f"EARLY_BIRD applies to non-students registering on or before {iso(cutoff)}"
    return Instance(args, f"Register someone for the conference (note: {tool_note}).", clauses,
                    ["enum:by_rule", "date:compare", "array:ordered"])


CONFERENCE = Domain("conference_registration", tool("register_attendee", "Register a conference attendee.", obj({
    "attendee": obj({"name": S("Full name."), "email": S("Email.")}),
    "ticket_type": E("STUDENT (full-time students), EARLY_BIRD (others, if registering on or before the early-bird "
                     "cutoff), STANDARD (otherwise).", ["STUDENT", "EARLY_BIRD", "STANDARD"]),
    "workshops": A(S("Workshop code."), "Workshop codes in the order mentioned; empty if none."),
    "tshirt_size": E("Size.", ["S", "M", "L", "XL"])})),
    (simple("conference_agenda", "Agenda.", day="Day."),
     simple("cancel_registration", "Cancel.", registration_id="Id.")),
    _conference, "registration")


def _claim(r, today, level):
    pol = f"POL-{alnum(r, 8)}"
    back = r.randint(1, 20)
    d = today - timedelta(days=back)
    tp, t, op_t = time_phrase(r, level=level, lo=6, hi=22)
    typ = r.choice(["THEFT", "WATER_DAMAGE", "FIRE", "ACCIDENT"])
    k = r.choice([1, 2, 3])
    items = [{"description": r.choice(["laptop", "bicycle", "sofa", "camera", "rug", "television"]),
              "value_cents": r.randrange(5000, 250000, 100)} for _ in range(k)]
    ttxt = {"THEFT": "someone broke in and took things", "WATER_DAMAGE": "a pipe burst and flooded the room",
            "FIRE": "a small kitchen fire spread", "ACCIDENT": "something got knocked over and broke"}[typ]
    when = f"{num(r, back)} days ago" if back > 1 else "yesterday"
    args = {"policy_number": pol, "incident": {"date": iso(d), "time": t, "type": typ}, "items": items}
    clauses = [f"policy {pol}", f"it happened {when} around {tp}", ttxt,
               "lost: " + "; ".join(f"{i['description']} worth {money_text(i['value_cents'], r)}" for i in items)]
    ops = [op_t, "date:back_offset", "enum:described", "nested:array"]
    if typ == "THEFT" and r.random() < 0.7:
        rep = code(r, "PR-", 6)
        args["police_report"] = rep
        clauses.append(f"the police report number is {rep}")
        ops.append("optional:present")
    return Instance(args, "File an insurance claim.", clauses, ops)


CLAIM = Domain("insurance_claim", tool("file_claim", "File a home insurance claim.", obj({
    "policy_number": S("Policy number."),
    "incident": obj({"date": S("YYYY-MM-DD.", **DATE), "time": S("HH:MM.", **TIME),
                     "type": E("THEFT, WATER_DAMAGE, FIRE or ACCIDENT.", ["THEFT", "WATER_DAMAGE", "FIRE", "ACCIDENT"])}),
    "items": A(obj({"description": S("Item, one lowercase word as given."), "value_cents": I("Value in cents.", minimum=1)}),
               "Items in the order listed.", minItems=1),
    "police_report": S(f"Police report number. {OPTIONAL}")}, required=["policy_number", "incident", "items"])),
    (simple("claim_status", "Status.", claim_id="Claim id."),
     simple("policy_coverage", "Coverage details.", policy_number="Policy.")),
    _claim, "claim")


def _bug(r, today, level):
    proj = r.choice(["WEB", "API", "IOS", "DATA"])
    summary = r.choice(["Login button unresponsive on Safari", "CSV export drops last row", "Timeout on bulk upload",
                        "Dark mode toggle resets after reload", "Search returns duplicates"])
    sev_key = r.choice(["CRITICAL", "MAJOR", "MINOR"])
    sev_txt = {"CRITICAL": "nobody can use the product at all", "MAJOR": "a main feature is broken but there's a workaround",
               "MINOR": "it's cosmetic"}[sev_key]
    comps = pick(r, ["auth", "billing", "export", "search", "ui"], r.choice([1, 2]))
    vers = sorted(r.sample(["2.3.0", "2.3.1", "2.4.0", "2.4.2", "3.0.0"], r.choice([1, 2])))
    args = {"project": proj, "summary": summary, "severity": sev_key, "components": comps, "affected_versions": vers}
    clauses = [f"project {proj}", f'summary: "{summary}"', sev_txt, "components: " + ", ".join(comps),
               "it shows up in " + " and ".join(reversed(vers))]
    return Instance(args, "Open a bug ticket.", clauses,
                    ["enum:described", "array:ordered", "array:sorted"])


BUG = Domain("bug_tracker", tool("file_bug_ticket", "Create a bug ticket.", obj({
    "project": E("Project key.", ["WEB", "API", "IOS", "DATA"]), "summary": S("Exactly as given."),
    "severity": E("CRITICAL (product unusable), MAJOR (main feature broken, workaround exists), MINOR (cosmetic).",
                  ["CRITICAL", "MAJOR", "MINOR"]),
    "components": A(S("Component."), "In the order mentioned.", minItems=1),
    "affected_versions": A(S("Semantic version."), "Sorted ascending.", minItems=1)})),
    (simple("search_tickets", "Search tickets.", query="Text."),
     simple("assign_ticket", "Assign a ticket.", ticket="Ticket key.")),
    _bug, "ticket")


def _campaign(r, today, level):
    name = r.choice(["Spring sale", "Product launch", "Webinar invite", "Win-back"])
    dp, d, op_d = date_phrase(r, today, level=level, max_ahead=21)
    tp, t, op_t = time_phrase(r, level=level, lo=7, hi=19)
    tz = r.choice(["America/New_York", "Europe/Berlin", "Asia/Tokyo"])
    city = {"America/New_York": "New York", "Europe/Berlin": "Berlin", "Asia/Tokyo": "Tokyo"}[tz]
    segs = pick(r, ["NEW", "ACTIVE", "LAPSED", "VIP"], r.choice([1, 2]))
    args = {"name": name, "send_at": {"date": iso(d), "time": t, "timezone": tz}, "segments": segs}
    stxt = {"NEW": "new signups", "ACTIVE": "active customers", "LAPSED": "people who haven't bought in a while",
            "VIP": "our top spenders"}
    clauses = [f'campaign "{name}"', f"send it {dp} at {tp} {city} time", "to " + " and ".join(stxt[s] for s in segs)]
    ops = [op_d, op_t, "enum:described", "lookup:timezone"]
    if level >= 2 and r.random() < 0.4:
        args["ab_test"] = {"variants": r.choice([2, 3]), "test_percent": r.choice([10, 20])}
        clauses.append(f"A/B test {num(r, args['ab_test']['variants'])} subject lines on {args['ab_test']['test_percent']}% of the list first")
        ops.append("optional:object")
    return Instance(args, "Schedule an email campaign.", clauses, ops)


CAMPAIGN = Domain("email_campaign", tool("schedule_campaign", "Schedule an email campaign.", obj({
    "name": S("Campaign name as given."),
    "send_at": obj({"date": S("YYYY-MM-DD.", **DATE), "time": S("HH:MM.", **TIME),
                    "timezone": E("IANA zone.", ["America/New_York", "Europe/Berlin", "Asia/Tokyo"])}),
    "segments": A(E("NEW (new signups), ACTIVE, LAPSED (no purchase recently), VIP (top spenders).", ["NEW", "ACTIVE", "LAPSED", "VIP"]),
                  "In the order mentioned.", minItems=1),
    "ab_test": obj({"variants": I("Subject-line variants.", minimum=2), "test_percent": I("Percent of list.", minimum=1, maximum=50)},
                   desc=f"A/B test. {OPTIONAL}")}, required=["name", "send_at", "segments"])),
    (simple("campaign_stats", "Stats.", campaign_id="Id."),
     simple("list_segments", "Segments.", filter="Filter.")),
    _campaign, "campaign")


def _workout(r, today, level):
    k = r.choice([2, 3])
    ex = pick(r, ["SQUAT", "BENCH_PRESS", "DEADLIFT", "ROW", "OVERHEAD_PRESS"], k)
    rows, texts = [], []
    for e in ex:
        sets, reps = r.choice([3, 4, 5]), r.choice([5, 8, 10, 12])
        kg = r.choice([20, 32.5, 40, 47.5, 60, 72.5, 100])
        rows.append({"exercise": e, "sets": sets, "reps": reps, "load_g": int(round(kg * 1000))})
        texts.append(f"{e.replace('_', ' ').lower()} {sets}x{reps} at {kg:g} kg")
    durp, dur, op_m = duration_phrase(r, (45, 60, 75, 90))
    back = r.choice([0, 1])
    args = {"date": iso(today - timedelta(days=back)), "exercises": rows, "duration_minutes": dur}
    clauses = ["today" if back == 0 else "yesterday", ", ".join(texts), f"the whole session took {durp}"]
    return Instance(args, "Log a workout.", clauses,
                    ["date:back_offset", "unit:kg_to_g", op_m, "nested:array"])


WORKOUT = Domain("workout_log", tool("log_workout", "Log a strength workout.", obj({
    "date": S("YYYY-MM-DD.", **DATE),
    "exercises": A(obj({"exercise": E("Exercise.", ["SQUAT", "BENCH_PRESS", "DEADLIFT", "ROW", "OVERHEAD_PRESS"]),
                        "sets": I("Sets.", minimum=1), "reps": I("Reps per set.", minimum=1), "load_g": I("Load in grams.")}),
                   "In the order performed.", minItems=1),
    "duration_minutes": I("Session length.", minimum=1)})),
    (simple("personal_records", "PRs.", exercise="Exercise."),
     simple("weekly_volume", "Volume summary.", week="YYYY-MM-DD")),
    _workout, "workout")


def _crm(r, today, level):
    cid = code(r, "C-", 6)
    ph, e164, op_p = phone(r)
    cn, cc = r.choice(COUNTRIES)
    city = r.choice(CITIES[cc])
    ch = r.choice(["EMAIL", "PHONE", "SMS"])
    updates = {"phone": e164, "preferred_channel": ch}
    clauses = [f"contact {cid}", f"new phone {ph}",
               {"EMAIL": "they prefer email now", "PHONE": "they want calls", "SMS": "texts only"}[ch]]
    ops = [op_p, "enum:described", "nested:object"]
    if level >= 2 and r.random() < 0.6:
        street = f"{r.randint(2, 980)} {r.choice(STREETS)}"
        updates["address"] = {"street": street, "city": city, "country": cc}
        clauses.append(f"they moved to {street}, {city}, {cn}")
        ops.append("optional:object")
    args = {"contact_id": cid, "updates": updates}
    return Instance(args, "Update a customer record.", clauses, ops,
                    r.choice([None, f"(Their old number was {phone(r)[0]}.)"]))


CRM = Domain("crm_update", tool("update_contact", "Update a CRM contact.", obj({
    "contact_id": S("Contact id."),
    "updates": obj({"phone": S("E.164.", **E164), "preferred_channel": E("EMAIL, PHONE or SMS.", ["EMAIL", "PHONE", "SMS"]),
                    "address": obj({"street": S("Street as written."), "city": S("City."),
                                    "country": S("ISO 3166-1 alpha-2.", pattern="^[A-Z]{2}$")}, desc=f"New address. {OPTIONAL}")},
                   required=["phone", "preferred_channel"])})),
    (simple("get_contact", "Fetch a contact.", contact_id="Id."),
     simple("merge_contacts", "Merge duplicates.", contact_id="Id.")),
    _crm, "contact")


DOMAINS: tuple[Domain, ...] = (SHIPMENT, EVENT, RESTAURANT, FLIGHTS, HOTEL, GROCERY, MEDICATION, EXPENSES, THERMOSTAT,
                               VM, ORDERS_QUERY, TICKETS, CAR, RECIPE, INVOICE, PET, SUBSCRIPTION, TIMESHEET, LAB,
                               TRANSLATION, PARKING, WAREHOUSE, LIBRARY, PAYMENT, CONFERENCE, CLAIM, BUG, CAMPAIGN,
                               WORKOUT, CRM)


# --- assembly -------------------------------------------------------------------------------------------------------

def assign_splits(domains=DOMAINS) -> dict[str, str]:
    """Whole domains to splits: domains sorted by sha256(version|name); the first ones go to the held-out splits in
    `SPLIT_DOMAINS` order, the rest to training."""
    names = sorted((d.name for d in domains), key=lambda n: hashlib.sha256(f"{VERSION}|{n}".encode()).hexdigest())
    out, i = {}, 0
    for split, k in SPLIT_DOMAINS:
        for n in names[i:i + k]:
            out[n] = split
        i += k
    for n in names[i:]:
        out[n] = "training"
    return out


def rng(domain: str, index: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{VERSION}|{domain}|{index}".encode()).hexdigest(), 16))


def compose(r: random.Random, inst: Instance) -> str:
    clauses = list(inst.clauses)
    style = r.choice(["prose", "prose", "notes"])
    if style == "notes":
        body = inst.opening + " Details:\n" + "\n".join(f"- {c[0].upper()}{c[1:]}" for c in clauses)
    else:
        if r.random() < 0.5 and len(clauses) > 2:
            head, tail = clauses[:1], clauses[1:]
            r.shuffle(tail)
            clauses = head + tail
        joined = ". ".join(c[0].upper() + c[1:] for c in clauses) + "."
        body = inst.opening + " " + joined
    if inst.distractor:
        body = (body + ("\n" if style == "notes" else " ") + inst.distractor if r.random() < 0.5
                else inst.distractor + " " + body)
    return body + r.choice(["", " Thanks!", " Thank you.", ""])


def response_of(domain: Domain, args: dict) -> str:
    h = hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()
    return json.dumps({"status": "ok", f"{domain.entity}_id": f"{domain.entity[:3].upper()}-{h[:8]}"})


def task(domain: Domain, index: int, split: str) -> dict:
    from stages.teacher_modes import tool_episodes as TE
    from stages.teacher_modes import tool_pool as TP
    r = rng(domain.name, index)
    today = DATE_RANGE[0] + timedelta(days=r.randrange((DATE_RANGE[1] - DATE_RANGE[0]).days))
    lvl = r.choices([x for x, _ in LEVELS], [w for _, w in LEVELS])[0]
    inst = domain.sample(r, today, lvl)
    call = {"name": domain.tool["function"]["name"], "arguments": inst.args}
    from aadistill.data.tool_calls import canonical_call
    tools = [domain.tool, *domain.distractors]
    order = list(range(len(tools)))
    r.shuffle(order)
    row = f"{domain.name}/{index}"
    return {
        "task_id": TE.task_id_of(SOURCE, VERSION, row),
        "category": "schema_arguments", "category_detail": None,
        "source": {"dataset": SOURCE, "revision": VERSION, "config": domain.name, "row": str(index),
                   "subset": "full", "verification_class": "G2",
                   "generator": {"level": lvl, "operators": sorted(set(inst.ops)), "current_date": iso(today)},
                   "recorded": {"steps": 2, "calls": 1, "ends_on_call": False},
                   "verification": {"levels": ["protocol", "schema_validity", "recorded_execution", "step_gold"],
                                    "step_gold": True, "task_success_verifier": None,
                                    "gold": "computed by the generator from the sampled ground truth; no model output"}},
        "split": split, "environment": f"schema-synth:{domain.name}",
        "atoms": [f"environment:schema-synth:{domain.name}"],
        "family": TP.family_of([f"environment:schema-synth:{domain.name}"]),
        "system": SYSTEM.format(weekday=WEEKDAYS[today.weekday()], iso=iso(today)),
        "tools": [tools[i] for i in order],
        "messages": [{"role": "user", "content": compose(r, inst)}],
        "later_user_turns": [],
        "backend": {"kind": "replay", "table": {canonical_call(call): response_of(domain, inst.args)}, "conflicts": []},
        "gold_steps": [[[call]], [[]]],
        "final_check": {"kind": "none"}, "max_turns": 2,
    }


def generate(per_domain: dict = PER_DOMAIN, domains=DOMAINS) -> list[dict]:
    splits = assign_splits(domains)
    out = []
    for d in domains:
        split = splits[d.name]
        for i in range(per_domain[split]):
            out.append(task(d, i, split))
    return out


def write(tasks: list[dict], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n" for t in tasks)
    (out / "tasks.jsonl").write_text(text, encoding="utf-8")
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    man = {"schema": "aadistill.stage2.teacher_modes.schema_synth/v1", "version": VERSION, "source": SOURCE,
           "generator_sha256": code_sha, "tasks": len(tasks),
           "tasks_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "domains_by_split": {s: sorted(n for n, x in assign_splits().items() if x == s)
                                for s in sorted(set(assign_splits().values()))},
           "per_domain": PER_DOMAIN, "census": dict(Counter(t["split"] for t in tasks)),
           "authorizes": "nothing: a design for the maintainer's review; no task enters the canonical pool"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


#: the extension's pool id: tool_pool@v1 plus this source (`tool_pool.extend`), a DESIGN until the maintainer admits
#: the source; the canonical pool directory is never written
EXTENSION_POOL_ID = "tool_pool@v1+schema_synth@v1"


def build_extension(base_dir: Path, tasks_path: Path, out: Path) -> dict:
    """``tool_pool.extend`` over the written tasks with the canonical pool's screens: eval-set-1 (T1), the pinned
    benchmarks and lineage sources (T2/T3), each teacher's first-prompt length bound."""
    from stages.teacher_modes import tool_pool as TP
    rows = TP.TE.read_tasks(tasks_path)
    proto = json.loads(TP.PROTOCOL.read_text())
    ref = TP.benchmark_reference(TP.eval_set_reference(TP.Reference()))
    checkers = TP.length_checkers(proto)
    sources = {"schema_synth": {"converted_sha256": TP._sha_file(tasks_path), "tasks": len(rows),
                                "datasets": [f"{SOURCE}@{VERSION}"],
                                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    return TP.extend(base_dir, rows, out, pool_id=EXTENSION_POOL_ID, sources=sources, reference=ref,
                     prompt_lengths=checkers, extra={"reference": ref.summary(),
                                                     "length_checkers": [f.__name__ for f in checkers]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="where tasks.jsonl and manifest.json are written")
    ap.add_argument("--extend", type=Path, help="a base pool directory: also write the extension to --pool-out")
    ap.add_argument("--pool-out", type=Path)
    args = ap.parse_args(argv)
    man = write(generate(), args.out)
    print(json.dumps({k: man[k] for k in ("tasks", "tasks_sha256", "census", "domains_by_split")}, indent=1))
    if args.extend:
        doc = build_extension(args.extend, args.out / "tasks.jsonl", args.pool_out)
        print(json.dumps({"content_sha256": doc["content_sha256"], "admitted": doc["addition"]["admitted"],
                          "excluded_by": doc["addition"]["excluded_by"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
