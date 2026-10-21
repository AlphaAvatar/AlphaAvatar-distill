"""Inductive rule discovery: infer an unstated rule from examples, apply it.

Three generators, each with a library of hidden rules. An instance is kept only
when EVERY rule in its generator's library that is consistent with the shown
examples agrees on the query: the answer is then determined by the examples
under the library, not by a guess about which rule the author meant.

`string_induction@v2` (maintainer-authorized 2026-10-19) adds the evidence the
library check cannot supply: a letter-class rule must be SHOWN acting on every
letter it changes in the query. v1, which built the frozen sets, is unchanged.
"""
from __future__ import annotations

import random
from typing import Callable

from .common import (ANSWER_INSTRUCTION, Family, Generator, Unsatisfiable,
                     fmt_list, user)

# --- integer lists -----------------------------------------------------------

ListFn = Callable[[list[int]], object]

_KNOBS = {  # examples, min len, max len, max value
    "L1": (5, 3, 5, 9),
    "L2": (4, 4, 7, 20),
    "L3": (3, 5, 8, 50),
}


def _list_rules() -> dict[str, tuple[str, list[tuple[str, ListFn]]]]:
    """family -> (transformation class, [(param label, fn), ...])."""
    def p(fn_by_k, ks):
        return [(f"k={k}", fn_by_k(k)) for k in ks]

    return {
        "list.reverse": ("reorder", [("", lambda x: x[::-1])]),
        "list.sort_asc": ("reorder", [("", lambda x: sorted(x))]),
        "list.sort_desc": ("reorder", [("", lambda x: sorted(x, reverse=True))]),
        "list.rotate_left": ("reorder", p(lambda k: (lambda x: x[k % len(x):] + x[:k % len(x)]), (1, 2))),
        "list.swap_ends": ("reorder", [("", lambda x: [x[-1]] + x[1:-1] + [x[0]] if len(x) > 1 else x)]),
        "list.add_k": ("elementwise", p(lambda k: (lambda x: [v + k for v in x]), (1, 2, 3, 4, 5))),
        "list.mul_k": ("elementwise", p(lambda k: (lambda x: [v * k for v in x]), (2, 3))),
        "list.square": ("elementwise", [("", lambda x: [v * v for v in x])]),
        "list.mod_k": ("elementwise", p(lambda k: (lambda x: [v % k for v in x]), (2, 3, 5))),
        "list.keep_even": ("filter", [("", lambda x: [v for v in x if v % 2 == 0])]),
        "list.keep_odd": ("filter", [("", lambda x: [v for v in x if v % 2 == 1])]),
        "list.keep_gt": ("filter", p(lambda k: (lambda x: [v for v in x if v > k]), (3, 5, 10))),
        "list.drop_first": ("filter", p(lambda k: (lambda x: x[k:]), (1, 2))),
        "list.take_first": ("filter", p(lambda k: (lambda x: x[:k]), (2, 3))),
        "list.dedupe": ("filter", [("", lambda x: list(dict.fromkeys(x)))]),
        "list.sum": ("aggregate", [("", lambda x: sum(x))]),
        "list.length": ("aggregate", [("", lambda x: len(x))]),
        "list.max": ("aggregate", [("", lambda x: max(x))]),
        "list.min": ("aggregate", [("", lambda x: min(x))]),
        "list.count_even": ("aggregate", [("", lambda x: sum(v % 2 == 0 for v in x))]),
        "list.duplicate_each": ("structural", [("", lambda x: [v for v in x for _ in (0, 1)])]),
        "list.append_sum": ("structural", [("", lambda x: x + [sum(x)])]),
        "list.prepend_length": ("structural", [("", lambda x: [len(x)] + x)]),
        "list.cumsum": ("structural", [("", lambda x: [sum(x[:i + 1]) for i in range(len(x))])]),
        "list.pairwise_sums": ("structural", [("", lambda x: [x[i] + x[i + 1] for i in range(len(x) - 1)])]),
    }


LIST_RULES = _list_rules()


def _consistent_answers(rules: dict, examples, query) -> set:
    answers = set()
    for _fam, (_cls, variants) in rules.items():
        for _label, fn in variants:
            try:
                if all(fn(list(x)) == y for x, y in examples):
                    out = fn(list(query))
                    answers.add(tuple(out) if isinstance(out, list) else out)
            except (IndexError, ValueError, ZeroDivisionError):
                continue
    return answers


def _list_family(fid: str) -> Family:
    cls, variants = LIST_RULES[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        n_ex, lo, hi, vmax = _KNOBS[difficulty]
        _label, fn = rng.choice(variants)

        def draw():
            return [rng.randint(0, vmax) for _ in range(rng.randint(lo, hi))]

        examples = []
        for _ in range(n_ex):
            x = draw()
            examples.append((x, fn(list(x))))
        query = draw()
        gold = fn(list(query))
        # degenerate instances teach nothing: an identity-looking example set
        if all(x == y for x, y in examples):
            raise Unsatisfiable
        if isinstance(gold, list) and not gold:
            raise Unsatisfiable
        answers = _consistent_answers(LIST_RULES, examples, query)
        if len(answers) != 1:
            raise Unsatisfiable
        lines = [f"Input: {fmt_list(x)} -> Output: "
                 f"{fmt_list(y) if isinstance(y, list) else y}" for x, y in examples]
        prompt = ("Each input list below was turned into its output by the same "
                  "hidden rule.\n\n" + "\n".join(lines) +
                  f"\n\nApply the same rule to this input:\nInput: {fmt_list(query)}\n\n"
                  + ANSWER_INSTRUCTION)
        return {"messages": user(prompt),
                "answer": fmt_list(gold) if isinstance(gold, list) else str(gold),
                "answer_type": "int_list" if isinstance(gold, list) else "int"}

    return Family(fid, cls, make)


LIST_INDUCTION = Generator(
    "list_induction@v1", "inductive_rule_discovery", "int_list",
    tuple(_list_family(f) for f in LIST_RULES),
    description="hidden integer-list function from input/output pairs")


# --- strings -----------------------------------------------------------------

_VOWELS = set("aeiou")
_SKNOBS = {"L1": (5, 3, 5), "L2": (4, 4, 7), "L3": (3, 6, 9)}


def _shift(c: str, k: int) -> str:
    return chr((ord(c) - 97 + k) % 26 + 97)


def _string_rules() -> dict[str, tuple[str, list[tuple[str, Callable[[str], str]]]]]:
    return {
        "str.reverse": ("reorder", [("", lambda s: s[::-1])]),
        "str.sort_letters": ("reorder", [("", lambda s: "".join(sorted(s)))]),
        "str.rotate_left": ("reorder", [(f"k={k}", (lambda k: lambda s: s[k:] + s[:k])(k)) for k in (1, 2)]),
        "str.swap_ends": ("reorder", [("", lambda s: s[-1] + s[1:-1] + s[0] if len(s) > 1 else s)]),
        "str.caesar": ("substitution", [(f"k={k}", (lambda k: lambda s: "".join(_shift(c, k) for c in s))(k)) for k in (1, 2, 3)]),
        "str.caesar_back": ("substitution", [(f"k={k}", (lambda k: lambda s: "".join(_shift(c, -k) for c in s))(k)) for k in (1, 2)]),
        "str.vowel_to_x": ("substitution", [("", lambda s: "".join("x" if c in _VOWELS else c for c in s))]),
        "str.remove_vowels": ("filter", [("", lambda s: "".join(c for c in s if c not in _VOWELS))]),
        "str.keep_even_positions": ("filter", [("", lambda s: s[::2])]),
        "str.drop_last": ("filter", [(f"k={k}", (lambda k: lambda s: s[:-k])(k)) for k in (1, 2)]),
        "str.first_k": ("filter", [(f"k={k}", (lambda k: lambda s: s[:k])(k)) for k in (2, 3)]),
        "str.duplicate_each": ("structural", [("", lambda s: "".join(c + c for c in s))]),
        "str.mirror": ("structural", [("", lambda s: s + s[::-1])]),
        "str.append_first": ("structural", [("", lambda s: s + s[0])]),
        "str.double_last": ("structural", [("", lambda s: s + s[-1])]),
        "str.wrap_first": ("structural", [("", lambda s: s[0] + s + s[0])]),
    }


STRING_RULES = _string_rules()

#: Rules that change one letter class and leave the rest alone. Under them
#: "change only the letters seen changed" fits the examples too, so v2 keeps an
#: instance only when every letter the rule changes in the query is shown
#: changed in some example -- the minimal-evidence reading then agrees with the
#: gold (eval-set-1 audit: `str.remove_vowels` items showing only 'o' removed).
#: The class's commonest wider reading counts 'y' as a vowel, so a query with a
#: 'y' also needs an example that shows a 'y' left unchanged.
_LETTER_CLASS_FAMILIES = frozenset({"str.remove_vowels", "str.vowel_to_x"})


def _string_family(fid: str, version: int) -> Family:
    cls, variants = STRING_RULES[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        n_ex, lo, hi = _SKNOBS[difficulty]
        _label, fn = rng.choice(variants)

        def draw():
            return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz")
                           for _ in range(rng.randint(lo, hi)))

        examples = [(x, fn(x)) for x in (draw() for _ in range(n_ex))]
        query = draw()
        gold = fn(query)
        if all(x == y for x, y in examples) or not gold or gold == query:
            raise Unsatisfiable
        if version >= 2 and fid in _LETTER_CLASS_FAMILIES:
            shown = {c for x, _y in examples for c in x if fn(c) != c}
            if any(fn(c) != c and c not in shown for c in query):
                raise Unsatisfiable
            if "y" in query and not any(c == "y" == fn(c) for x, _y in examples for c in x):
                raise Unsatisfiable
        answers = set()
        for _f, (_c, vs) in STRING_RULES.items():
            for _l, g in vs:
                if all(g(x) == y for x, y in examples):
                    answers.add(g(query))
        if len(answers) != 1:
            raise Unsatisfiable
        lines = [f"{x} -> {y}" for x, y in examples]
        prompt = ("Each word on the left was turned into the word on its right by "
                  "the same hidden rule.\n\n" + "\n".join(lines) +
                  f"\n\nApply the same rule to: {query}\n\n" + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": gold}

    return Family(fid, cls, make)


STRING_INDUCTION = Generator(
    "string_induction@v1", "inductive_rule_discovery", "word",
    tuple(_string_family(f, 1) for f in STRING_RULES),
    description="hidden letter-string rewrite from word pairs")

STRING_INDUCTION_V2 = Generator(
    "string_induction@v2", "inductive_rule_discovery", "word",
    tuple(_string_family(f, 2) for f in STRING_RULES),
    description="as string_induction@v1; letter-class rules shown on every query letter they change")


# --- number sequences --------------------------------------------------------

def _fit_arith(t):
    d = t[1] - t[0]
    return t[-1] + d if all(t[i + 1] - t[i] == d for i in range(len(t) - 1)) else None


def _fit_geom(t):
    if t[0] == 0 or t[1] % t[0]:
        return None
    r = t[1] // t[0]
    if r in (0, 1):
        return None
    return t[-1] * r if all(t[i + 1] == t[i] * r for i in range(len(t) - 1)) else None


def _fit_quadratic(t):
    d = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    dd = [d[i + 1] - d[i] for i in range(len(d) - 1)]
    if len(dd) < 2 or len(set(dd)) != 1 or dd[0] == 0:
        return None
    return t[-1] + d[-1] + dd[0]


def _fit_fib(t):
    if len(t) < 4 or not all(t[i] == t[i - 1] + t[i - 2] for i in range(2, len(t))):
        return None
    return t[-1] + t[-2]


def _fit_alternating(t):
    d = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    if len(d) < 4 or d[0] == d[1]:
        return None
    if all(d[i] == d[i % 2] for i in range(len(d))):
        return t[-1] + d[len(d) % 2]
    return None


def _fit_interleave(t):
    ev, od = t[0::2], t[1::2]
    if len(ev) < 3 or len(od) < 2:
        return None
    a, b = _fit_arith(ev), _fit_arith(od)
    if a is None or b is None or ev[1] - ev[0] == od[1] - od[0]:
        return None
    return a if len(t) % 2 == 0 else b


def _fit_affine(t):
    """a_n = r * a_{n-1} + c with integer r >= 2."""
    if len(t) < 4 or t[1] == t[0]:
        return None
    for r in (2, 3):
        c = t[1] - r * t[0]
        if c != 0 and all(t[i + 1] == r * t[i] + c for i in range(len(t) - 1)):
            return r * t[-1] + c
    return None


SEQ_FITS = {
    "seq.arithmetic": ("additive", _fit_arith),
    "seq.geometric": ("multiplicative", _fit_geom),
    "seq.quadratic": ("additive", _fit_quadratic),
    "seq.fibonacci_like": ("recurrence", _fit_fib),
    "seq.alternating_steps": ("additive", _fit_alternating),
    "seq.interleaved": ("interleave", _fit_interleave),
    "seq.affine_recurrence": ("recurrence", _fit_affine),
}

_QKNOBS = {"L1": (7, 1), "L2": (6, 2), "L3": (5, 3)}  # shown terms, scale


def _seq_maker(fid: str):
    def gen_terms(rng, n, s):
        if fid == "seq.arithmetic":
            a, d = rng.randint(0, 10 * s), rng.choice([x for x in range(-4 * s, 5 * s + 1) if x])
            return [a + d * i for i in range(n + 1)]
        if fid == "seq.geometric":
            a, r = rng.randint(1, 3 * s), rng.choice((2, 3))
            return [a * r ** i for i in range(n + 1)]
        if fid == "seq.quadratic":
            a, b, c = rng.choice([1, 2] if s < 3 else [1, 2, 3]), rng.randint(-3, 3), rng.randint(0, 9)
            return [a * i * i + b * i + c for i in range(n + 1)]
        if fid == "seq.fibonacci_like":
            x, y = rng.randint(1, 4 * s), rng.randint(1, 4 * s)
            out = [x, y]
            while len(out) < n + 1:
                out.append(out[-1] + out[-2])
            return out
        if fid == "seq.alternating_steps":
            a = rng.randint(0, 10 * s)
            p, q = rng.sample([x for x in range(-5 * s, 6 * s) if x], 2)
            out = [a]
            for i in range(n):
                out.append(out[-1] + (p if i % 2 == 0 else q))
            return out
        if fid == "seq.interleaved":
            a, b = rng.randint(0, 10 * s), rng.randint(0, 10 * s)
            d1, d2 = rng.sample([x for x in range(-3 * s, 4 * s) if x], 2)
            return [(a + d1 * (i // 2)) if i % 2 == 0 else (b + d2 * (i // 2)) for i in range(n + 1)]
        if fid == "seq.affine_recurrence":
            a, r, c = rng.randint(1, 3 * s), rng.choice((2, 3)), rng.choice([x for x in range(-3, 4) if x])
            out = [a]
            while len(out) < n + 1:
                out.append(r * out[-1] + c)
            return out
        raise KeyError(fid)

    def make(rng: random.Random, difficulty: str) -> dict:
        n, s = _QKNOBS[difficulty]
        terms = gen_terms(rng, n, s)
        shown, gold = terms[:n], terms[n]
        if abs(gold) > 10 ** 6:
            raise Unsatisfiable
        answers = {fit(shown) for _f, (_c, fit) in SEQ_FITS.items()} - {None}
        if answers != {gold}:
            raise Unsatisfiable
        prompt = ("The numbers below follow one hidden rule.\n\n"
                  + ", ".join(str(t) for t in shown)
                  + "\n\nWhat is the next number?\n\n" + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": str(gold), "answer_type": "int"}

    return make


SEQUENCE_INDUCTION = Generator(
    "sequence_induction@v1", "inductive_rule_discovery", "int",
    tuple(Family(f, c, _seq_maker(f)) for f, (c, _fit) in SEQ_FITS.items()),
    description="next term of a sequence under a hidden generating rule")

GENERATORS = (LIST_INDUCTION, STRING_INDUCTION, SEQUENCE_INDUCTION)
GENERATORS_V2 = (LIST_INDUCTION, STRING_INDUCTION_V2, SEQUENCE_INDUCTION)
