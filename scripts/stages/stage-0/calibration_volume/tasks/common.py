"""Shared vocabulary of the procedural reasoning tasks: families, instances, answers.

A **generator** is one task format (its prompt layout and answer type). A
**rule family** is one hidden rule, transformation or structure a generator can
pose; it is the unit evaluation isolation is checked by. A family also names its
**transformation class**, a coarser grouping (e.g. "reorder") reported so a
reader can see whether an unseen family is also an unseen kind of rule.

Every instance is built from a seeded `random.Random`, so the same
(generator, family, difficulty, seed) always produces the same bytes. Nothing
here touches a model.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

ANSWER_INSTRUCTION = "Put your final answer inside \\boxed{}."

AXES = (
    "inductive_rule_discovery",
    "compositional_generalization",
    "task_contained_deduction",
    "abstract_analogical",
    "interactive_adaptation",
)

DIFFICULTIES = ("L1", "L2", "L3")


class Unsatisfiable(Exception):
    """This seed cannot produce an unambiguous instance; the caller redraws."""


@dataclass(frozen=True)
class Family:
    family_id: str
    transformation_class: str
    make: Callable[[random.Random, str], dict]
    note: str = ""


@dataclass(frozen=True)
class Generator:
    generator_id: str
    axis: str
    answer_type: str
    families: tuple[Family, ...]
    interactive: bool = False
    description: str = ""
    extra: dict = field(default_factory=dict)

    def family(self, family_id: str) -> Family:
        for f in self.families:
            if f.family_id == family_id:
                return f
        raise KeyError(f"{self.generator_id}: no family {family_id!r}")


def instance_id(generator_id: str, family_id: str, difficulty: str, seed: int) -> str:
    raw = f"{generator_id}|{family_id}|{difficulty}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def build_instance(gen: Generator, fam: Family, difficulty: str, seed: int,
                   *, max_redraws: int = 200) -> dict:
    """One deterministic instance; redraws inside the seed's own stream."""
    rng = random.Random(f"{gen.generator_id}|{fam.family_id}|{difficulty}|{seed}")
    for _ in range(max_redraws):
        try:
            body = fam.make(rng, difficulty)
        except Unsatisfiable:
            continue
        task = {
            "task_id": instance_id(gen.generator_id, fam.family_id, difficulty, seed),
            "generator_id": gen.generator_id,
            "family_id": fam.family_id,
            "transformation_class": fam.transformation_class,
            "axis": gen.axis,
            "difficulty": difficulty,
            "seed": seed,
            "answer_type": gen.answer_type,
            "interactive": gen.interactive,
            **body,
        }
        task["content_sha256"] = content_hash(task)
        if task.get("answer") is not None and gold_key(task) is None:
            raise RuntimeError(f"{task['task_id']}: its own answer does not "
                               f"canonicalize as {task['answer_type']!r}")
        return task
    raise RuntimeError(f"{gen.generator_id}/{fam.family_id}/{difficulty}/{seed}: "
                       f"no unambiguous instance in {max_redraws} draws")


def content_hash(task: dict) -> str:
    """Hash of what a model sees plus the answer, independent of ids and seeds."""
    seen = {"messages": task.get("messages"), "answer": task.get("answer"),
            "env": task.get("env")}
    return hashlib.sha256(json.dumps(seen, sort_keys=True).encode()).hexdigest()


# --- answer canonicalization -------------------------------------------------

_INT = re.compile(r"-?\d+")


def canon_int(text: str | None) -> str | None:
    if text is None:
        return None
    t = text.replace(",", "").replace("\\!", "").strip().strip("$").strip()
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    nums = _INT.findall(t)
    if len(nums) != 1:
        return None
    return str(int(nums[0]))


def canon_int_list(text: str | None) -> str | None:
    if text is None:
        return None
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    t = t.replace("\\left", "").replace("\\right", "").replace("\\[", "[").replace("\\]", "]")
    if re.search(r"[A-Za-z]", t.replace("emptyset", "")):
        return None
    nums = _INT.findall(t)
    if not nums and not re.search(r"\[\s*\]|\\emptyset|\(\s*\)", t):
        return None
    return json.dumps([int(n) for n in nums])


def canon_word(text: str | None) -> str | None:
    if text is None:
        return None
    t = re.sub(r"\\text(tt|rm|bf)?\{([^}]*)\}", r"\2", text)
    t = t.replace("\\_", "_").strip().strip("$").strip().strip("\"'`").strip()
    t = re.sub(r"\s+", " ", t)
    return t.lower() if t else None


def canon_token_seq(text: str | None) -> str | None:
    """A sequence of uppercase action tokens, order-sensitive, separators free."""
    if text is None:
        return None
    t = re.sub(r"\\text(tt|rm|bf)?\{([^}]*)\}", r"\2", text)
    toks = re.findall(r"[A-Z]+", t.upper())
    return " ".join(toks) if toks else None


def canon_grid(text: str | None) -> str | None:
    """Rows of digits; rows separated by '/', newlines or LaTeX line breaks."""
    if text is None:
        return None
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    t = t.replace("\\\\", "/").replace("\n", "/").replace(";", "/")
    t = re.sub(r"\\begin\{[a-z]*matrix\}|\\end\{[a-z]*matrix\}", "/", t)
    t = t.replace("&", " ")
    rows = [re.findall(r"\d", r) for r in t.split("/")]
    rows = ["".join(r) for r in rows if r]
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    return "/".join(rows)


def canon_name(text: str | None) -> str | None:
    if text is None:
        return None
    t = re.sub(r"\\text(tt|rm|bf)?\{([^}]*)\}", r"\2", text)
    words = re.findall(r"[A-Za-z]+", t)
    return words[0].lower() if len(words) == 1 else None


def canon_name_set(text: str | None) -> str | None:
    if text is None:
        return None
    t = re.sub(r"\\text(tt|rm|bf)?\{([^}]*)\}", r"\2", text)
    words = sorted({w.lower() for w in re.findall(r"[A-Za-z]+", t)
                    if w.lower() not in {"and", "none", "only", "the", "a", "an"}})
    if re.search(r"\bnone\b|\\emptyset", t.lower()) and not words:
        return "[]"
    return json.dumps(words)


CANON: dict[str, Callable[[str | None], str | None]] = {
    "int": canon_int,
    "int_list": canon_int_list,
    "word": canon_word,
    "token_seq": canon_token_seq,
    "grid": canon_grid,
    "name": canon_name,
    "name_set": canon_name_set,
}


def gold_key(task: dict) -> str:
    return CANON[task["answer_type"]](task["answer"])


def fmt_list(xs: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def fmt_grid(g: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def grid_answer(g: list[list[int]]) -> str:
    return " / ".join("".join(str(c) for c in row) for row in g)


NONSENSE_SYLLABLES = ("da", "ke", "lo", "mi", "nu", "po", "ra", "si", "tu", "ve",
                      "wo", "zi", "fe", "gu", "bi", "ha", "jo", "ki", "le", "mo")


def nonsense_word(rng: random.Random, syllables: int = 2) -> str:
    return "".join(rng.choice(NONSENSE_SYLLABLES) for _ in range(syllables))


def distinct_nonsense(rng: random.Random, n: int, syllables: int = 2) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        w = nonsense_word(rng, syllables)
        if w not in out:
            out.append(w)
    return out


PEOPLE = ("Ava", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun",
          "Kai", "Lena", "Milo", "Nia", "Omar", "Pia", "Quin", "Rosa", "Sami", "Tao")


def user(content: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": content}]
