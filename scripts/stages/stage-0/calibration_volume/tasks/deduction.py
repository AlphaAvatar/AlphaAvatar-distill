"""Task-contained deduction: multi-step reasoning over facts the task supplies.

Every puzzle is checked by brute force: an instance is kept only when exactly
one world satisfies the stated facts, so the answer follows from the prompt
alone and needs no outside knowledge.
"""
from __future__ import annotations

import itertools
import json
import random

from .common import (ANSWER_INSTRUCTION, PEOPLE, Family, Generator, Unsatisfiable,
                     user)

_ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh"}

# --- ordering ----------------------------------------------------------------

_OKN = {"L1": (3, 0), "L2": (5, 1), "L3": (7, 2)}  # entities, redundant facts


def _ordering_family(fid: str):
    def make(rng: random.Random, difficulty: str) -> dict:
        n, extra = _OKN[difficulty]
        names = rng.sample(PEOPLE, n)
        order = names[:]  # order[0] is the tallest / first / leftmost
        rng.shuffle(order)
        facts, constraints = [], []
        if fid == "order.taller":
            pairs = [(order[i], order[i + 1]) for i in range(n - 1)]
            for _ in range(extra):
                i, j = sorted(rng.sample(range(n), 2))
                pairs.append((order[i], order[j]))
            rng.shuffle(pairs)
            for a, b in pairs:
                if rng.random() < 0.5:
                    facts.append(f"{a} is taller than {b}.")
                else:
                    facts.append(f"{b} is shorter than {a}.")
                constraints.append(lambda p, a=a, b=b: p.index(a) < p.index(b))
            k = rng.randint(1, n)
            q = f"Who is the {_ORDINAL[k]} tallest?"
            gold = order[k - 1]
        elif fid == "order.race":
            pairs = [(order[i], order[i + 1]) for i in range(n - 1)]
            for _ in range(extra):
                i, j = sorted(rng.sample(range(n), 2))
                pairs.append((order[i], order[j]))
            rng.shuffle(pairs)
            for a, b in pairs:
                gap = order.index(b) - order.index(a)
                if gap == 1 and rng.random() < 0.5:
                    facts.append(f"{a} finished immediately before {b}.")
                    constraints.append(lambda p, a=a, b=b: p.index(b) - p.index(a) == 1)
                else:
                    facts.append(f"{a} finished ahead of {b}.")
                    constraints.append(lambda p, a=a, b=b: p.index(a) < p.index(b))
            k = rng.randint(1, n)
            q = f"Who finished in {_ORDINAL[k]} place?"
            gold = order[k - 1]
        elif fid == "order.seating":
            # a row of seats, left to right; adjacency and end facts
            for i in range(n - 1):
                a, b = order[i], order[i + 1]
                facts.append(f"{a} sits immediately to the left of {b}.")
                constraints.append(lambda p, a=a, b=b: p.index(b) - p.index(a) == 1)
            drop = rng.randrange(n - 1)
            facts.pop(drop)
            constraints.pop(drop)
            end = rng.choice((0, n - 1))
            side = "left" if end == 0 else "right"
            facts.append(f"{order[end]} sits at the {side} end.")
            constraints.append(lambda p, who=order[end], end=end: p.index(who) == end)
            for _ in range(extra):
                i, j = sorted(rng.sample(range(n), 2))
                facts.append(f"{order[i]} sits somewhere to the left of {order[j]}.")
                constraints.append(lambda p, a=order[i], b=order[j]: p.index(a) < p.index(b))
            rng.shuffle(facts)
            k = rng.randint(1, n)
            q = f"Counting from the left, who sits in seat number {k}?"
            gold = order[k - 1]
        elif fid == "order.ages":
            base = rng.randint(10, 40)
            ages = {order[-1]: base}
            for i in range(n - 2, -1, -1):
                ages[order[i]] = ages[order[i + 1]] + rng.randint(1, 6)
            anchor = rng.choice(names)
            facts.append(f"{anchor} is {ages[anchor]} years old.")
            links = [(order[i], order[i + 1]) for i in range(n - 1)]
            rng.shuffle(links)
            for a, b in links:
                d = ages[a] - ages[b]
                facts.append(f"{a} is {d} years older than {b}." if rng.random() < 0.5
                             else f"{b} is {d} years younger than {a}.")
            target = rng.choice([x for x in names if x != anchor])
            q = f"How old is {target}?"
            prompt = ("Read the facts and answer the question.\n\n" + "\n".join(facts)
                      + f"\n\n{q}\n\n" + ANSWER_INSTRUCTION)
            return {"messages": user(prompt), "answer": str(ages[target]),
                    "answer_type": "int"}
        else:
            raise KeyError(fid)
        sols = [p for p in itertools.permutations(names) if all(c(p) for c in constraints)]
        if len(sols) != 1:
            raise Unsatisfiable
        prompt = ("Read the facts and answer the question.\n\n" + "\n".join(facts)
                  + f"\n\n{q} Answer with the name only.\n\n" + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": gold, "answer_type": "name"}

    cls = {"order.ages": "numeric_chain"}.get(fid, "total_order")
    return Family(fid, cls, make)


ORDERING = Generator(
    "ordering@v1", "task_contained_deduction", "name",
    tuple(_ordering_family(f) for f in ("order.taller", "order.race",
                                        "order.seating", "order.ages")),
    description="recover a total order (or a value chain) from stated relations")

# --- entity tracking ---------------------------------------------------------

_ITEMS = ("apple", "key", "coin", "ring", "shell", "stone", "card", "pen",
          "cup", "bell", "leaf", "rope")
_TKN = {"L1": (3, 2), "L2": (4, 4), "L3": (5, 7)}  # boxes, operations
_TRACK_OPS = {
    "track.move": ("move",),
    "track.swap": ("swap",),
    "track.move_swap": ("move", "swap"),
    "track.move_remove": ("move", "remove"),
    "track.add_remove": ("add", "remove"),
    "track.all": ("move", "swap", "add", "remove"),
}


def _tracking_family(fid: str):
    allowed = _TRACK_OPS[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        n_boxes, n_ops = _TKN[difficulty]
        boxes = [f"box {i + 1}" for i in range(n_boxes)]
        pool = list(_ITEMS)
        rng.shuffle(pool)
        state = {b: [] for b in boxes}
        for b in boxes:
            state[b].append(pool.pop())
        for _ in range(rng.randint(0, n_boxes - 1)):
            state[rng.choice(boxes)].append(pool.pop())
        init = [f"{b.capitalize()} contains {' and '.join('the ' + x for x in sorted(v))}."
                for b, v in state.items()]
        steps = []
        for _ in range(n_ops):
            op = rng.choice(allowed)
            if op == "move":
                src = rng.choice([b for b in boxes if state[b]])
                item = rng.choice(state[src])
                dst = rng.choice([b for b in boxes if b != src])
                state[src].remove(item)
                state[dst].append(item)
                steps.append(f"The {item} is moved from {src} to {dst}.")
            elif op == "swap":
                a, b = rng.sample(boxes, 2)
                state[a], state[b] = state[b], state[a]
                steps.append(f"The contents of {a} and {b} are swapped.")
            elif op == "add":
                if not pool:
                    raise Unsatisfiable
                item, b = pool.pop(), rng.choice(boxes)
                state[b].append(item)
                steps.append(f"A {item} is put into {b}.")
            elif op == "remove":
                cands = [b for b in boxes if state[b]]
                b = rng.choice(cands)
                item = rng.choice(state[b])
                state[b].remove(item)
                steps.append(f"The {item} is taken out of {b} and thrown away.")
        if rng.random() < 0.5:
            b = rng.choice(boxes)
            q = f"Which items are in {b} at the end? List them, or write none."
            gold = json.dumps(sorted(state[b]))
            atype, ans = "name_set", (", ".join(sorted(state[b])) or "none")
        else:
            present = [(x, b) for b in boxes for x in state[b]]
            if not present:
                raise Unsatisfiable
            x, b = rng.choice(present)
            q = f"Which box contains the {x} at the end? Answer with the box number only."
            atype, ans = "int", str(int(b.split()[1]))
            gold = ans
        del gold
        prompt = ("At the start:\n" + "\n".join(init) + "\n\nThen, in order:\n"
                  + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
                  + f"\n\n{q}\n\n" + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": ans, "answer_type": atype}

    return Family(fid, "state_tracking", make)


ENTITY_TRACKING = Generator(
    "entity_tracking@v1", "task_contained_deduction", "name_set",
    tuple(_tracking_family(f) for f in _TRACK_OPS),
    description="track object locations through a sequence of stated operations")

# --- knights and knaves ------------------------------------------------------

_KKN = {"L1": 2, "L2": 3, "L3": 4}
_STATEMENT_SETS = {
    # "X is a knight/knave" alone is invariant under flipping everyone, so the
    # identity family carries a disjunction that breaks the flip.
    "kk.identity": ("is_knight", "is_knave", "or_knight"),
    "kk.same_type": ("same", "different"),
    "kk.counting": ("at_least_one_knave", "exactly_one_knight"),
    "kk.mixed": ("is_knight", "is_knave", "same", "different", "at_least_one_knave"),
}


def _kk_family(fid: str):
    kinds = _STATEMENT_SETS[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        n = _KKN[difficulty]
        names = rng.sample(PEOPLE, n)
        truth = {p: rng.random() < 0.5 for p in names}
        stmts = []
        for speaker in names:
            kind = rng.choice(kinds)
            other = rng.choice([p for p in names if p != speaker])
            if kind == "is_knight":
                text, f = f"{other} is a knight.", (lambda w, o=other: w[o])
            elif kind == "is_knave":
                text, f = f"{other} is a knave.", (lambda w, o=other: not w[o])
            elif kind == "same":
                text, f = (f"{other} and I are the same kind.",
                           lambda w, s=speaker, o=other: w[s] == w[o])
            elif kind == "different":
                text, f = (f"{other} and I are different kinds.",
                           lambda w, s=speaker, o=other: w[s] != w[o])
            elif kind == "or_knight":
                text, f = (f"At least one of {other} and I is a knight.",
                           lambda w, s=speaker, o=other: w[s] or w[o])
            elif kind == "at_least_one_knave":
                text, f = ("At least one of us is a knave.",
                           lambda w: not all(w.values()))
            else:
                text, f = ("Exactly one of us is a knight.",
                           lambda w: sum(w.values()) == 1)
            stmts.append((speaker, text, f))

        def holds(w):
            return all(w[s] == f(w) for s, _t, f in stmts)

        worlds = [dict(zip(names, bits)) for bits in itertools.product((True, False), repeat=n)]
        sols = [w for w in worlds if holds(w)]
        if len(sols) != 1:
            raise Unsatisfiable
        knights = sorted(p for p, k in sols[0].items() if k)
        prompt = ("On an island, knights always tell the truth and knaves always lie. "
                  "Everyone is either a knight or a knave.\n\n"
                  + "\n".join(f"{s} says: \"{t}\"" for s, t, _f in stmts)
                  + "\n\nWhich of them are knights? List their names, or write none.\n\n"
                  + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": ", ".join(knights) or "none"}

    return Family(fid, "truth_assignment", make)


KNIGHTS_KNAVES = Generator(
    "knights_knaves@v1", "task_contained_deduction", "name_set",
    tuple(_kk_family(f) for f in _STATEMENT_SETS),
    description="find the unique truth assignment consistent with all statements")

# --- graph paths ---------------------------------------------------------------

_TOWNS = ("Alder", "Birch", "Cedar", "Dune", "Elm", "Fern", "Glen", "Heath",
          "Iris", "Juniper", "Kestrel", "Linden")
_GKN = {"L1": (4, 4), "L2": (6, 8), "L3": (8, 13)}  # towns, roads


def _bfs(adj, src):
    dist, frontier = {src: 0}, [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def _graph_family(fid: str):
    def make(rng: random.Random, difficulty: str) -> dict:
        n, m = _GKN[difficulty]
        towns = rng.sample(_TOWNS, n)
        edges = set()
        while len(edges) < m:
            a, b = rng.sample(towns, 2)
            edges.add((a, b))
        adj = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
        src = rng.choice(towns)
        dist = _bfs(adj, src)
        if fid == "graph.shortest":
            far = [t for t, d in dist.items() if d >= 2]
            if not far:
                raise Unsatisfiable
            dst = rng.choice(far)
            q = (f"What is the smallest number of roads needed to travel from {src} "
                 f"to {dst}?")
            ans, atype = str(dist[dst]), "int"
        elif fid == "graph.reach_count":
            q = (f"How many OTHER towns can be reached from {src} by following roads?")
            ans, atype = str(len(dist) - 1), "int"
            if len(dist) - 1 < 1:
                raise Unsatisfiable
        elif fid == "graph.unreachable":
            unreach = sorted(set(towns) - set(dist))
            if not unreach or len(unreach) > 4:
                raise Unsatisfiable
            q = (f"Which towns can NOT be reached from {src}? List them, or write none.")
            ans, atype = ", ".join(unreach), "name_set"
        else:
            raise KeyError(fid)
        roads = sorted(edges)
        rng.shuffle(roads)
        prompt = ("The roads below are all one-way.\n\n"
                  + "\n".join(f"There is a road from {a} to {b}." for a, b in roads)
                  + f"\n\n{q}\n\n" + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": ans, "answer_type": atype}

    return Family(fid, "graph_search", make)


GRAPH_PATHS = Generator(
    "graph_paths@v1", "task_contained_deduction", "int",
    tuple(_graph_family(f) for f in ("graph.shortest", "graph.reach_count",
                                    "graph.unreachable")),
    description="search a stated one-way road network")

GENERATORS = (ORDERING, ENTITY_TRACKING, KNIGHTS_KNAVES, GRAPH_PATHS)
