"""The v2 generators (maintainer-authorized 2026-10-19), and v1 left untouched.

Each test protects a measurement. v1 built every frozen task set (eval-set-1's
`dev_v1`, `calibration_v1`, `calibration_v2`), so a v1 instance that changes
silently rewrites a historical result. Each v2 property is one eval-set-1 audit
finding, checked from the PROMPT TEXT a model sees rather than from the
generator's own bookkeeping, and each check is also shown to catch the defect
in v1 -- a check that cannot fail v1 proves nothing about v2.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from stages.calibration_volume.tasks import registry as R
from stages.calibration_volume.tasks.common import (CANON, DIFFICULTIES, PEOPLE,
                                                    build_instance)

REPO = Path(__file__).resolve().parents[5]
DEV = REPO / "data/eval-set/eval-set-1/fluid_reasoning_720/dev_v1.jsonl"
STAGE_SETS = REPO / "data/stages/stage-0/calibration_volume/tasks"
#: every set built from the v1 roster
FROZEN_SETS = (DEV, STAGE_SETS / "calibration_v1.jsonl", STAGE_SETS / "calibration_v2.jsonl")
#: the hash seed under which dev_v1 is reproducible (eval-set-1 README)
FROZEN_HASH_SEED = "14"
TOUCHED = {"ordering@v1": "ordering@v2", "graph_paths@v1": "graph_paths@v2",
           "string_induction@v1": "string_induction@v2",
           "letter_analogy@v1": "letter_analogy@v2"}
SEEDS = range(120)


def _prompt(t) -> str:
    return t["messages"][0]["content"]


def _instances(gen_id: str, families=None):
    gen = R.BY_ID.get(gen_id) or R.BY_ID_V2[gen_id]
    for fam in gen.families:
        if families and fam.family_id not in families:
            continue
        for d in DIFFICULTIES:
            for s in SEEDS:
                yield build_instance(gen, fam, d, s)


def _names(text: str) -> list[str]:
    return re.split(r", | and ", text)


# --- v1 is unchanged -----------------------------------------------------------

_REGEN = """
import json, sys
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/scripts"]
from stages.calibration_volume.build_task_sets import build
for role, per_cell, per_int, out in json.loads(sys.argv[2]):
    tasks = build(role, per_cell, per_int, None)
    with open(out, "w") as fh:
        fh.write("".join(json.dumps(t, sort_keys=True) + "\\n" for t in tasks))
"""


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """Every frozen set rebuilt from today's v1 code the way build_task_sets
    built it, in a fresh process under the frozen hash seed: {name: (frozen
    bytes, rebuilt bytes, manifest)}."""
    out_dir = tmp_path_factory.mktemp("regen")
    sets = {}
    for jsonl in FROZEN_SETS:
        man = json.loads(jsonl.with_suffix(".manifest.json").read_text())
        assert "--generators" not in man["command"], jsonl.name
        sets[jsonl.name] = (jsonl, man, out_dir / jsonl.name)
    jobs = [[m["role"], m["per_cell"], m["per_cell_interactive"], str(out)]
            for _p, m, out in sets.values()]
    env = {**os.environ, "PYTHONHASHSEED": FROZEN_HASH_SEED}
    subprocess.run([sys.executable, "-c", _REGEN, str(REPO), json.dumps(jobs)],
                   env=env, check=True, timeout=300)
    return {name: (path.read_bytes(), out.read_bytes(), man)
            for name, (path, man, out) in sets.items()}


def test_touched_v1_generators_regenerate_their_frozen_rows(regenerated):
    for name, (frozen, rebuilt, _man) in regenerated.items():
        def rows(body):
            return [l for l in body.decode().splitlines()
                    if json.loads(l)["generator_id"] in TOUCHED]
        want = rows(frozen)
        assert want, name
        assert rows(rebuilt) == want, name
    dev_gens = {json.loads(l)["generator_id"] for l in regenerated[DEV.name][0].decode().splitlines()}
    assert set(TOUCHED) <= dev_gens


def test_every_frozen_set_regenerates_byte_for_byte(regenerated):
    import hashlib
    for name, (frozen, rebuilt, man) in regenerated.items():
        assert hashlib.sha256(frozen).hexdigest() == man["sha256"], name
        assert rebuilt == frozen, name


# --- ordering@v2: seating names everyone and states the seat count --------------

_INTRO = re.compile(r"^(\d+) people sit in a row of (\d+) seats, numbered 1 to (\d+) "
                    r"from the left, one person per seat: (.+)\.$")
_SEAT_FACTS = (
    (re.compile(r"^(\w+) sits immediately to the left of (\w+)\.$"),
     lambda a, b: lambda p: p.index(b) - p.index(a) == 1),
    (re.compile(r"^(\w+) sits somewhere to the left of (\w+)\.$"),
     lambda a, b: lambda p: p.index(a) < p.index(b)),
    (re.compile(r"^(\w+) sits at the (left|right) end\.$"),
     lambda a, side: lambda p: p.index(a) == (0 if side == "left" else len(p) - 1)),
)


def _solve_seating(prompt: str) -> list[tuple[str, ...]]:
    """Every arrangement of exactly the stated people in exactly the stated
    seats that satisfies every stated fact."""
    lines = prompt.split("\n\n")[1].split("\n")
    m = _INTRO.match(lines[0])
    assert m, lines[0]
    n_people, n_seats, last, listed = m.groups()
    names = _names(listed)
    assert int(n_people) == int(n_seats) == int(last) == len(names) == len(set(names))
    constraints = []
    for line in lines[1:]:
        hits = [(pat.match(line), mk) for pat, mk in _SEAT_FACTS]
        (g, mk), = [(h, mk) for h, mk in hits if h]
        assert g.group(1) in names and (g.group(2) in names or g.group(2) in ("left", "right"))
        constraints.append(mk(*g.groups()))
    return [p for p in itertools.permutations(names) if all(c(p) for c in constraints)]


def test_ordering_v2_seating_is_complete_and_unique():
    n = 0
    for t in _instances("ordering@v2", {"order.seating"}):
        prompt = _prompt(t)
        sols = _solve_seating(prompt)
        assert len(sols) == 1, t["task_id"]
        k = int(re.search(r"seat number (\d+)\?", prompt).group(1))
        assert sols[0][k - 1] == t["answer"], t["task_id"]
        named = set(_names(_INTRO.match(prompt.split("\n\n")[1].split("\n")[0]).group(4)))
        assert {w for w in re.findall(r"[A-Z][a-z]+", prompt) if w in PEOPLE} <= named
        n += 1
    assert n == len(DIFFICULTIES) * len(SEEDS)
    # the defect the check exists for: a v1 seating gold that the prompt never names
    assert any(t["answer"] not in _prompt(t)
               for t in _instances("ordering@v1", {"order.seating"}))


def test_ordering_v2_other_families_name_their_answer():
    for t in _instances("ordering@v2", {"order.taller", "order.race"}):
        assert t["answer"] in _prompt(t), t["task_id"]


# --- graph_paths@v2: the printed universe is the gold's universe ------------------

_UNIVERSE = re.compile(r"^There are (\d+) towns: (.+)\.$", re.M)
_ROAD = re.compile(r"^There is a road from (\w+) to (\w+)\.$", re.M)


def _bfs(roads, src):
    dist, frontier = {src: 0}, [src]
    while frontier:
        nxt = []
        for u in frontier:
            for a, b in roads:
                if a == u and b not in dist:
                    dist[b] = dist[u] + 1
                    nxt.append(b)
        frontier = nxt
    return dist


def _graph_answer_from_prompt(t):
    """(answer recomputed from the printed facts alone, printed universe)."""
    prompt = _prompt(t)
    roads = _ROAD.findall(prompt)
    src = re.search(r"from (\w+)", prompt.split("\n\n")[-2]).group(1)
    m = _UNIVERSE.search(prompt)
    # v1 names no universe: a reader can only know the towns it prints
    universe = set(_names(m.group(2))) if m else {src, *(x for r in roads for x in r)}
    if m:
        assert int(m.group(1)) == len(universe)
    assert src in universe and {x for r in roads for x in r} <= universe
    dist = _bfs(roads, src)
    if t["family_id"] == "graph.shortest":
        dst = re.search(r"to (\w+)\?", prompt).group(1)
        return str(dist[dst]), universe
    if t["family_id"] == "graph.reach_count":
        return str(len(dist) - 1), universe
    return ", ".join(sorted(universe - set(dist))) or "none", universe


def test_graph_v2_gold_follows_from_the_printed_facts():
    for t in _instances("graph_paths@v2"):
        assert _UNIVERSE.search(_prompt(t)), t["task_id"]
        got, universe = _graph_answer_from_prompt(t)
        assert CANON[t["answer_type"]](got) == CANON[t["answer_type"]](t["answer"]), t["task_id"]
        if t["family_id"] == "graph.unreachable":
            assert set(_names(t["answer"])) <= universe
    # the defect the check exists for: a v1 unreachable gold naming a town on no road
    assert any(CANON["name_set"](_graph_answer_from_prompt(t)[0]) != CANON["name_set"](t["answer"])
               for t in _instances("graph_paths@v1", {"graph.unreachable"}))


_DIGEST = """
import hashlib, json, sys
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/scripts"]
from stages.calibration_volume.tasks import registry as R
from stages.calibration_volume.tasks.common import DIFFICULTIES, build_instance
out = {}
for gid in json.loads(sys.argv[2]):
    g, h = (R.BY_ID.get(gid) or R.BY_ID_V2[gid]), hashlib.sha256()
    for f in g.families:
        for d in DIFFICULTIES:
            for s in range(int(sys.argv[3])):
                h.update((json.dumps(build_instance(g, f, d, s), sort_keys=True) + "\\n").encode())
    out[gid] = h.hexdigest()
print(json.dumps(out))
"""


def test_v2_generators_do_not_depend_on_the_hash_seed():
    gids = [*TOUCHED.values(), "graph_paths@v1"]
    runs = []
    for hash_seed in ("1", "2"):
        res = subprocess.run([sys.executable, "-c", _DIGEST, str(REPO), json.dumps(gids),
                              str(len(SEEDS))], env={**os.environ, "PYTHONHASHSEED": hash_seed},
                             capture_output=True, text=True, check=True, timeout=300)
        runs.append(json.loads(res.stdout))
    for gid in TOUCHED.values():
        assert runs[0][gid] == runs[1][gid], gid
    # the defect the check exists for: v1 graphs move with the hash seed
    assert runs[0]["graph_paths@v1"] != runs[1]["graph_paths@v1"]


# --- string_induction@v2: every changed query letter is shown changed -------------

def _minimal_evidence_answer(t) -> str:
    """The audit's competing reading: change only the letters seen changed."""
    prompt = _prompt(t)
    examples = re.findall(r"^([a-z]+) -> ([a-z]*)$", prompt, re.M)
    query = re.search(r"Apply the same rule to: ([a-z]+)$", prompt, re.M).group(1)
    assert examples
    if t["family_id"] == "str.remove_vowels":
        removed = {c for x, y in examples for c in set(x) if x.count(c) > y.count(c)}
        return "".join(c for c in query if c not in removed)
    mapping = {a: b for x, y in examples for a, b in zip(x, y) if a != b}
    return "".join(mapping.get(c, c) for c in query)


def _y_vowel_reading_differs(t) -> bool:
    """The audit's other competing reading: 'y' counts as a vowel. True when it
    fits every example and still gives another answer for the query."""
    prompt = _prompt(t)
    examples = re.findall(r"^([a-z]+) -> ([a-z]*)$", prompt, re.M)
    query = re.search(r"Apply the same rule to: ([a-z]+)$", prompt, re.M).group(1)
    vowels = set("aeiouy")
    if t["family_id"] == "str.remove_vowels":
        def alt(s):
            return "".join(c for c in s if c not in vowels)
    else:
        def alt(s):
            return "".join("x" if c in vowels else c for c in s)
    return all(alt(x) == y for x, y in examples) and alt(query) != t["answer"]


def test_string_v2_minimal_evidence_reading_agrees_with_the_gold():
    fams = {"str.remove_vowels", "str.vowel_to_x"}
    n = 0
    for t in _instances("string_induction@v2", fams):
        assert _minimal_evidence_answer(t) == t["answer"], t["task_id"]
        assert not _y_vowel_reading_differs(t), t["task_id"]
        n += 1
    assert n == len(fams) * len(DIFFICULTIES) * len(SEEDS)
    for fam in sorted(fams):  # the defects the checks exist for, in each family
        v1 = list(_instances("string_induction@v1", {fam}))
        assert any(_minimal_evidence_answer(t) != t["answer"] for t in v1), fam
        assert any(_y_vowel_reading_differs(t) for t in v1), fam


# --- letter_analogy@v2: no leaked query, no repeated demonstration ----------------

def _letter_defects(t) -> list[str]:
    prompt = _prompt(t)
    demos = re.findall(r"^If ([a-z]+) changes to ([a-z]+),$", prompt, re.M)
    query = re.search(r"^what does ([a-z]+) change to\?$", prompt, re.M).group(1)
    assert demos
    out = []
    if query in {x for x, _y in demos}:
        out.append("query_is_an_example")
    if len(set(demos)) != len(demos):
        out.append("repeated_demo")
    return out


def test_letter_v2_query_is_new_and_demos_are_distinct():
    for t in _instances("letter_analogy@v2"):
        assert not _letter_defects(t), t["task_id"]
    seen = {d for t in _instances("letter_analogy@v1") for d in _letter_defects(t)}
    assert seen == {"query_is_an_example", "repeated_demo"}


# --- the v2 roster ---------------------------------------------------------------

def test_v2_roster_replaces_exactly_the_four_and_keeps_the_split():
    assert len(R.GENERATORS_V2) == len(R.GENERATORS) == len(R.BY_ID_V2)
    assert not any(g.generator_id.endswith("@v2") for g in R.GENERATORS)
    replaced = {}
    for g1, g2 in zip(R.GENERATORS, R.GENERATORS_V2):
        assert R.family_split(g2) == R.family_split(g1), g2.generator_id
        assert (g1.generator_id in R.EVAL_ONLY_GENERATORS) == \
            (g2.generator_id in R.EVAL_ONLY_GENERATORS), g2.generator_id
        if g2 is g1:
            continue
        replaced[g1.generator_id] = g2.generator_id
        assert (g2.axis, g2.answer_type, g2.interactive) == (g1.axis, g1.answer_type, g1.interactive)
        assert [(f.family_id, f.transformation_class) for f in g2.families] == \
            [(f.family_id, f.transformation_class) for f in g1.families]
    assert replaced == TOUCHED
    assert R.EVAL_ONLY_GENERATORS <= set(R.BY_ID) | set(R.BY_ID_V2)
    assert {"graph_paths@v2", "letter_analogy@v2"} <= R.EVAL_ONLY_GENERATORS
    assert not {"ordering@v2", "string_induction@v2"} & R.EVAL_ONLY_GENERATORS
