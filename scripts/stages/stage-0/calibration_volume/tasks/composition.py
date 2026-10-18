"""Compositional generalization: apply and combine rules in unseen compositions.

Every primitive is DEFINED in the prompt, under a fresh nonsense name, so no
prior knowledge of the names exists. A family is a composition STRUCTURE; the
calibration and evaluation splits use disjoint structures (see the registry),
which is what makes an evaluation composition unseen.
"""
from __future__ import annotations

import random

from .common import (ANSWER_INSTRUCTION, Family, Generator, Unsatisfiable,
                     distinct_nonsense, fmt_list, user)

# --- defined list operations -------------------------------------------------

_PRIMS = (
    ("reverses the order of the list", lambda x: x[::-1]),
    ("sorts the list from smallest to largest", lambda x: sorted(x)),
    ("adds 2 to every number", lambda x: [v + 2 for v in x]),
    ("subtracts 1 from every number", lambda x: [v - 1 for v in x]),
    ("doubles every number", lambda x: [2 * v for v in x]),
    ("removes the first number", lambda x: x[1:]),
    ("removes the last number", lambda x: x[:-1]),
    ("moves the first number to the end", lambda x: x[1:] + x[:1]),
    ("keeps only the even numbers", lambda x: [v for v in x if v % 2 == 0]),
    ("appends the sum of the list at the end", lambda x: x + [sum(x)]),
    ("repeats the last number once more at the end", lambda x: x + x[-1:]),
    ("swaps the first and the last number", lambda x: (x[-1:] + x[1:-1] + x[:1]) if len(x) > 1 else x),
)

_OKNOBS = {"L1": (3, True, 3, 4, 9), "L2": (4, True, 4, 6, 20), "L3": (5, False, 5, 7, 30)}


def _defined_ops_family(fid: str, cls: str):
    def make(rng: random.Random, difficulty: str) -> dict:
        n_ops, worked, lo, hi, vmax = _OKNOBS[difficulty]
        prims = rng.sample(_PRIMS, n_ops)
        names = distinct_nonsense(rng, n_ops)
        ops = list(zip(names, prims))
        x = [rng.randint(0, vmax) for _ in range(rng.randint(lo, hi))]

        def app(i, v):
            out = ops[i][1][1](v)
            if not out:
                raise Unsatisfiable
            return out

        if fid == "comp.chain2":
            i, j = rng.sample(range(n_ops), 2)
            expr, gold = f"{names[i]}({names[j]}(X))", app(i, app(j, x))
        elif fid == "comp.chain3":
            i, j, k = rng.sample(range(n_ops), 3)
            expr, gold = f"{names[i]}({names[j]}({names[k]}(X)))", app(i, app(j, app(k, x)))
        elif fid == "comp.chain4":
            if n_ops < 4:
                ids = [rng.randrange(n_ops) for _ in range(4)]
            else:
                ids = rng.sample(range(n_ops), 4)
            v = x
            for i in reversed(ids):
                v = app(i, v)
            expr = "".join(f"{names[i]}(" for i in ids) + "X" + ")" * 4
            gold = v
        elif fid == "comp.repeat":
            i = rng.randrange(n_ops)
            n = rng.choice((2, 3))
            v = x
            for _ in range(n):
                v = app(i, v)
            expr = "".join(f"{names[i]}(" for _ in range(n)) + "X" + ")" * n
            gold = v
        elif fid == "comp.conditional":
            i, j, k = rng.sample(range(n_ops), 3)
            cond = x[0] % 2 == 0
            mid = app(i, x) if cond else app(j, x)
            gold = app(k, mid)
            expr = (f"first apply {names[i]} if the first number of X is even, or "
                    f"{names[j]} if it is odd; then apply {names[k]} to the result")
        elif fid == "comp.conditional_chain":
            i, j, k = rng.sample(range(n_ops), 3)
            mid = app(k, x)
            cond = len(mid) % 2 == 0
            gold = app(i, mid) if cond else app(j, mid)
            expr = (f"first apply {names[k]} to X; then, if the result has an even "
                    f"number of elements apply {names[i]}, otherwise apply {names[j]}")
        else:
            raise KeyError(fid)
        if any(abs(v) > 999 for v in gold) or gold == x:
            raise Unsatisfiable
        lines = [f"- {n} {p[0]}." for n, p in ops]
        demo = ""
        if worked:
            ex = [rng.randint(0, 9) for _ in range(4)]
            demo = "\nFor example:\n" + "\n".join(
                f"  {n}({fmt_list(ex)}) = {fmt_list(p[1](ex))}" for n, p in ops) + "\n"
        how = ("Nested expressions are evaluated from the inside out." if "(" in expr
               else "")
        prompt = ("Here are some operations on lists of numbers:\n" + "\n".join(lines)
                  + "\n" + demo + (f"\n{how}\n" if how else "\n")
                  + f"\nLet X = {fmt_list(x)}. Compute: {expr}\n\n" + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": fmt_list(gold)}

    return Family(fid, cls, make)


DEFINED_OPS = Generator(
    "defined_ops@v1", "compositional_generalization", "int_list",
    tuple(_defined_ops_family(f, c) for f, c in (
        ("comp.chain2", "nesting"), ("comp.chain3", "nesting"), ("comp.chain4", "nesting"),
        ("comp.repeat", "iteration"), ("comp.conditional", "conditional"),
        ("comp.conditional_chain", "conditional"))),
    description="evaluate a composition of operations defined in the prompt")


# --- a SCAN-like command language -------------------------------------------

_ACTIONS = ("JUMP", "WALK", "RUN", "LOOK", "TURN", "SWIM")
_CONNECTIVES = {
    "seq": ("means: do the left part, then the right part", lambda a, b: a + b),
    "rev": ("means: do the right part, then the left part", lambda a, b: b + a),
    "sur": ("means: do the left part, then the right part, then the left part again",
            lambda a, b: a + b + a),
}
_MODS = {
    "twice": ("placed after a command means: do that command two times", 2),
    "thrice": ("placed after a command means: do that command three times", 3),
}
# family -> the connectives the QUERY uses (left to right)
_SCAN_FAMILIES = {
    "scan.seq_rev": ("seq", "rev"),
    "scan.rev_seq": ("rev", "seq"),
    "scan.seq_sur": ("seq", "sur"),
    "scan.sur_seq": ("sur", "seq"),
    "scan.rev_sur": ("rev", "sur"),
    "scan.sur_rev": ("sur", "rev"),
    "scan.sur_sur": ("sur", "sur"),
    "scan.rev_rev": ("rev", "rev"),
}
_SCKNOBS = {"L1": (3, 0.0), "L2": (4, 0.5), "L3": (5, 0.9)}  # verbs, P(modifier on a term)


def _scan_family(fid: str):
    cons = _SCAN_FAMILIES[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        n_verbs, p_mod = _SCKNOBS[difficulty]
        verbs = distinct_nonsense(rng, n_verbs + len(_CONNECTIVES) + len(_MODS))
        vnames, cnames, mnames = (verbs[:n_verbs], verbs[n_verbs:n_verbs + 3],
                                  verbs[n_verbs + 3:])
        acts = rng.sample(_ACTIONS, n_verbs)
        cmap = dict(zip(_CONNECTIVES, cnames))
        mmap = dict(zip(_MODS, mnames))

        def term():
            i = rng.randrange(n_verbs)
            words, out = [vnames[i]], [acts[i]]
            if rng.random() < p_mod:
                m = rng.choice(list(_MODS))
                words.append(mmap[m])
                out = out * _MODS[m][1]
            return words, out

        t = [term() for _ in range(len(cons) + 1)]
        words, value = list(t[0][0]), t[0][1]
        for c, (w, v) in zip(cons, t[1:]):
            words += [cmap[c]] + w
            value = _CONNECTIVES[c][1](value, v)
        if len(value) > 24:
            raise Unsatisfiable
        defs = [f"- \"{vnames[i]}\" means {acts[i]}." for i in range(n_verbs)]
        defs += [f"- \"{mmap[m]}\" {_MODS[m][0]}." for m in _MODS]
        defs += [f"- \"A {cmap[c]} B\" {_CONNECTIVES[c][0]}." for c in _CONNECTIVES]
        demos = [f"  {vnames[0]} {mmap['twice']} -> {acts[0]} {acts[0]}",
                 f"  {vnames[1]} {cmap['seq']} {vnames[0]} -> {acts[1]} {acts[0]}"]
        prompt = ("A small command language is defined as follows:\n" + "\n".join(defs)
                  + "\nModifiers apply to the single command just before them. "
                    "Connectives are applied from left to right, each to everything "
                    "built so far and the next command.\n"
                  + "Examples:\n" + "\n".join(demos)
                  + f"\n\nWrite the action sequence for: {' '.join(words)}\n\n"
                  + "Give the actions in order, separated by spaces. " + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": " ".join(value)}

    return Family(fid, "connective_pair", make)


SCAN_LANGUAGE = Generator(
    "scan_language@v1", "compositional_generalization", "token_seq",
    tuple(_scan_family(f) for f in _SCAN_FAMILIES),
    description="interpret an unseen combination of in-context-defined connectives")


# --- counterfactual arithmetic ----------------------------------------------

_OPS = {
    "cf.affine": ("a {s} b means 2 times a, plus b", lambda a, b: 2 * a + b),
    "cf.affine_minus": ("a {s} b means a, minus 2 times b", lambda a, b: a - 2 * b),
    "cf.product_plus": ("a {s} b means a times b, plus a", lambda a, b: a * b + a),
    "cf.absdiff": ("a {s} b means the larger of a and b minus the smaller", lambda a, b: abs(a - b)),
    "cf.concat": ("a {s} b means the number formed by writing a and then b", lambda a, b: int(f"{a}{b}") if a >= 0 and b >= 0 else None),
    "cf.min_plus": ("a {s} b means the smaller of a and b, plus 3", lambda a, b: min(a, b) + 3),
    "cf.square_sum": ("a {s} b means a times a, plus b", lambda a, b: a * a + b),
}
_SYMS = ("#", "@", "&", "%", "~")
_CFKNOBS = {"L1": (1, 9), "L2": (2, 9), "L3": (3, 12)}  # nested operator applications


def _cf_family(fid: str):
    text, fn = _OPS[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        depth, vmax = _CFKNOBS[difficulty]
        others = rng.sample([k for k in _OPS if k != fid], 1)
        syms = rng.sample(_SYMS, 2)
        table = {syms[0]: (text, fn), syms[1]: _OPS[others[0]]}

        def build(d):
            if d == 0:
                v = rng.randint(1, vmax)
                return str(v), v
            sym = syms[0] if d == depth else rng.choice(syms)
            ls, lv = build(d - 1)
            rs, rv = str(rng.randint(1, vmax)), None
            rv = int(rs)
            if rng.random() < 0.5:
                ls, lv, rs, rv = rs, rv, ls, lv
            out = table[sym][1](lv, rv)
            if out is None or abs(out) > 10 ** 5:
                raise Unsatisfiable
            return f"({ls} {sym} {rs})", out

        expr, gold = build(depth)
        expr = expr[1:-1] if expr.startswith("(") else expr
        defs = [f"- {t.format(s=s)}." for s, (t, _f) in table.items()]
        prompt = ("In this puzzle two symbols have new meanings:\n" + "\n".join(defs)
                  + "\nParentheses are evaluated first.\n\n"
                  + f"Compute: {expr}\n\n" + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": str(gold), "answer_type": "int"}

    return Family(fid, "operator_redefinition", make)


COUNTERFACTUAL_ARITH = Generator(
    "counterfactual_arith@v1", "compositional_generalization", "int",
    tuple(_cf_family(f) for f in _OPS),
    description="evaluate nested expressions under redefined operators")

GENERATORS = (DEFINED_OPS, SCAN_LANGUAGE, COUNTERFACTUAL_ARITH)
