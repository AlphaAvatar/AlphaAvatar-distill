"""Project-side answer checks for third-party modules with no single gold string. $0.

    from stages.reasoning_data.checkers import CHECKERS, out_of_domain
    CHECKERS["reasoning_gym/n_queens"](task, final_text) -> Check

A module whose answer is a board, a move list, a plan, a derivation or a math
expression cannot be judged by comparing one string with the generator's gold:
several answers are valid, or one value has many spellings. Each checker here
reads the BOXED final answer (`aadistill.data.verify.boxed_answer`) and the
task's own data -- the generator's prompt and gold, carried on the task -- and
decides by construction:

* ``reasoning_gym/n_queens`` -- the board's size, every pre-placed queen kept,
  n queens in all, none attacking another (any valid completion);
* ``reasoning_gym/tower_of_hanoi`` -- every move simulated: legal, and all
  disks end on the target peg (any legal solution);
* ``reasoning_core/planning`` -- the plan simulated under the stated actions
  (closed world: unlisted facts are false; deletes, then adds): every
  precondition holds when its action runs, the goal holds at the end, and the
  plan has the length the prompt's cue states ("exactly k actions");
* ``reasoning_core/logic_derivation`` -- every derivation line is a step of the
  generator's reference derivation (same rule, same input facts, same
  conclusion; line numbers are resolved, not compared), each input available
  when used, and the last line derives the target. A proof made of reference
  steps, not any valid proof: a different valid route is refused;
* ``mathematics_dataset/*`` -- value equality through a SAFE polynomial parser
  (numbers, fractions, ``\\frac``, single-letter variables, + - * / ^ and
  parentheses, integer powers; no eval): a polynomial for
  ``calculus__differentiate_composed``, a rational for
  ``probability__swr_p_sequence``, the multiset of roots for
  ``algebra__polynomial_roots_composed`` (a factorization item is out of
  domain), the ordered list for ``comparison__sort_composed``.

Nothing here evaluates teacher text as code or calls a library's scorer
(`external_modules.json :: verification_policy`): a parse that fails is an
``incorrect`` answer, never a guess. `out_of_domain` says which generator
items a checker cannot judge; `external_tasks` drops those before any task is
written, so no paid candidate is generated for an item that could never be
accepted.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from fractions import Fraction

from aadistill.data.target_quality import Check
from aadistill.data.verify import boxed_answer


def _prompt(task: dict) -> str:
    return "\n".join(m.get("content") or "" for m in task["messages"] if m.get("role") == "user")


_WRAPPERS = re.compile(r"\\(?:text|mathrm|textbf|texttt|mathtt|mbox)\s*\{([^{}]*)\}")


def _plain(s: str) -> str:
    """Boxed LaTeX to plain lines: text wrappers unwrapped, ``\\\\`` a line break."""
    prev = None
    while prev != s:
        prev, s = s, _WRAPPERS.sub(r"\1", s)
    s = s.replace("\\\\", "\n").replace("$", "")
    for sp in ("\\quad", "\\qquad", "\\,", "\\;", "\\!", "\\ "):
        s = s.replace(sp, " ")
    return s


def _boxed(final: str, scope: str) -> tuple[str | None, Check | None]:
    got = boxed_answer(final or "")
    if got is None:
        return None, Check("incorrect", "no boxed answer", scope)
    return _plain(got), None


# --------------------------------------------------------------------------- n_queens

NQ_SCOPE = ("the boxed board: the stated size, every pre-placed queen kept, n queens in all, "
            "none attacking another (any valid completion)")


def _nq_problem(prompt: str) -> tuple[int, list[str]]:
    m = re.search(r"board of size (\d+) x (\d+)", prompt)
    if not m or m.group(1) != m.group(2):
        raise ValueError("no square board size in the prompt")
    n = int(m.group(1))
    rows = [ln.replace(" ", "") for ln in prompt.splitlines() if re.fullmatch(r"[Q_](?: [Q_])*", ln.strip())]
    if len(rows) != n or any(len(r) != n for r in rows):
        raise ValueError("the prompt's board does not have the stated size")
    return n, rows


def _attack_free(cells: list[tuple[int, int]]) -> bool:
    rows, cols = Counter(r for r, _ in cells), Counter(c for _, c in cells)
    d1, d2 = Counter(r - c for r, c in cells), Counter(r + c for r, c in cells)
    return all(v == 1 for cnt in (rows, cols, d1, d2) for v in cnt.values())


def n_queens(task: dict, final: str) -> Check:
    text, bad = _boxed(final, NQ_SCOPE)
    if bad:
        return bad
    n, given = _nq_problem(_prompt(task))
    lines = [re.sub(r"[\s&|]", "", ln) for ln in text.splitlines()]
    board = [ln for ln in lines if ln]
    if len(board) != n or any(not re.fullmatch(r"[Q_]+", r) or len(r) != n for r in board):
        return Check("incorrect", f"not a {n} x {n} board of Q and _", NQ_SCOPE)
    queens = [(r, c) for r, row in enumerate(board) for c, ch in enumerate(row) if ch == "Q"]
    fixed = [(r, c) for r, row in enumerate(given) for c, ch in enumerate(row) if ch == "Q"]
    if any(rc not in queens for rc in fixed):
        return Check("incorrect", "a pre-placed queen was removed or moved", NQ_SCOPE)
    if len(queens) != n:
        return Check("incorrect", f"{len(queens)} queens, not {n}", NQ_SCOPE)
    if not _attack_free(queens):
        return Check("incorrect", "two queens attack each other", NQ_SCOPE)
    return Check("correct", "", NQ_SCOPE)


# --------------------------------------------------------------------------- tower_of_hanoi

HANOI_SCOPE = "every boxed move simulated: each legal, and all disks end on the target peg (any legal solution)"
_MOVE = re.compile(r"move\s+disk\s+(\d+)\s+from\s+peg\s+(\d+)\s+to\s+peg\s+(\d+)", re.I)


def _hanoi_problem(prompt: str) -> tuple[int, int, int, int]:
    m = re.search(r"with (\d+) disks and (\d+) pegs", prompt)
    g = re.search(r"Move all disks from Peg (\d+) to Peg (\d+)", prompt)
    if not (m and g):
        raise ValueError("no disk / peg statement in the prompt")
    n, pegs, src, dst = int(m.group(1)), int(m.group(2)), int(g.group(1)), int(g.group(2))
    if not (1 <= src <= pegs and 1 <= dst <= pegs and src != dst and n >= 1):
        raise ValueError("inconsistent pegs")
    return n, pegs, src, dst


def tower_of_hanoi(task: dict, final: str) -> Check:
    text, bad = _boxed(final, HANOI_SCOPE)
    if bad:
        return bad
    n, pegs, src, dst = _hanoi_problem(_prompt(task))
    moves = [tuple(int(x) for x in m) for m in _MOVE.findall(text)]
    if not moves:
        return Check("incorrect", "no move lines", HANOI_SCOPE)
    state = {p: [] for p in range(1, pegs + 1)}
    state[src] = list(range(n, 0, -1))
    for k, (disk, a, b) in enumerate(moves):
        if a not in state or b not in state or a == b:
            return Check("incorrect", f"move {k + 1}: no such peg pair {a}->{b}", HANOI_SCOPE)
        if not state[a] or state[a][-1] != disk:
            return Check("incorrect", f"move {k + 1}: disk {disk} is not on top of peg {a}", HANOI_SCOPE)
        if state[b] and state[b][-1] < disk:
            return Check("incorrect", f"move {k + 1}: disk {disk} onto a smaller disk", HANOI_SCOPE)
        state[b].append(state[a].pop())
    if state[dst] != list(range(n, 0, -1)):
        return Check("incorrect", f"after {len(moves)} moves the disks are not all on peg {dst}", HANOI_SCOPE)
    return Check("correct", "", HANOI_SCOPE)


# --------------------------------------------------------------------------- planning

PLAN_SCOPE = ("the boxed plan simulated under the stated actions (closed world; deletes, then adds): every "
              "precondition holds when its action runs, the goal holds at the end, and the plan has the "
              "length the prompt's cue states")
_ATOM = re.compile(r"\w+\([^()]*\)")
_LIT = re.compile(r"(!?)(\w+\([^()]*\))")


def _atom(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _plan_problem(prompt: str) -> dict:
    init = re.search(r"Initial true facts:(.*?)All other facts are false", prompt, re.S)
    goal = re.search(r"^Goal:(.*)$", prompt, re.M)
    if not (init and goal):
        raise ValueError("no initial facts or goal in the prompt")
    actions: dict[str, tuple] = {}
    for line in prompt.splitlines():
        m = re.fullmatch(r"\s*(\w+\([^()]*\)):(.*?)->(.*)", line)
        if not m:
            continue
        name = _atom(m.group(1))
        if name in actions:
            raise ValueError(f"action {name} is defined twice")
        actions[name] = ([(neg == "!", _atom(a)) for neg, a in _LIT.findall(m.group(2))],
                         [(neg == "!", _atom(a)) for neg, a in _LIT.findall(m.group(3))])
    if not actions:
        raise ValueError("no actions in the prompt")
    cue = re.search(r"exactly (\d+) actions?", prompt)
    return {"init": {_atom(a) for a in _ATOM.findall(init.group(1))}, "actions": actions,
            "goal": [(neg == "!", _atom(a)) for neg, a in _LIT.findall(goal.group(1))],
            "length": int(cue.group(1)) if cue else None}


def planning(task: dict, final: str) -> Check:
    text, bad = _boxed(final, PLAN_SCOPE)
    if bad:
        return bad
    p = _plan_problem(_prompt(task))
    plan = [_atom(a) for a in _ATOM.findall(text)]
    if not plan:
        return Check("incorrect", "no grounded actions", PLAN_SCOPE)
    if p["length"] is not None and len(plan) != p["length"]:
        return Check("incorrect", f"{len(plan)} actions, the cue states {p['length']}", PLAN_SCOPE)
    state = set(p["init"])
    for k, a in enumerate(plan):
        if a not in p["actions"]:
            return Check("incorrect", f"step {k + 1}: {a} is not a listed action", PLAN_SCOPE)
        pre, eff = p["actions"][a]
        if any((f in state) == neg for neg, f in pre):
            return Check("incorrect", f"step {k + 1}: a precondition of {a} does not hold", PLAN_SCOPE)
        state -= {f for neg, f in eff if neg}
        state |= {f for neg, f in eff if not neg}
    if any((f in state) == neg for neg, f in p["goal"]):
        return Check("incorrect", "the goal does not hold after the plan", PLAN_SCOPE)
    return Check("correct", "", PLAN_SCOPE)


# --------------------------------------------------------------------------- logic_derivation

LOGIC_SCOPE = ("every boxed derivation line is a step of the generator's reference derivation (same rule, same "
               "input facts, same conclusion), each input available when used, and the last line derives the "
               "target: a proof made of reference steps, not any valid proof")
_LINE = re.compile(r"(?:\d+[.)]\s+)?(\d+)\s*:\s*([@\d,\s]*?)\s*(?:=>|⇒|\\Rightarrow|\\implies)\s*(.+)")


def _sent(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .")


def _logic_problem(prompt: str) -> tuple[dict[int, str], str]:
    prem = re.search(r"Premise:\s*\n(.*?)\n\s*Target:\s*\n(.*?)\n", prompt, re.S)
    if not prem:
        raise ValueError("no premises or target in the prompt")
    premises = {int(m.group(1)): _sent(m.group(2))
                for m in re.finditer(r"^\s*(\d+):\s*(.+?)\s*$", prem.group(1), re.M)}
    return premises, _sent(prem.group(2))


def _derivation(text: str, premises: dict[int, str]) -> list[tuple[int, frozenset, str]]:
    """The derivation's steps as (rule, input facts, conclusion); ValueError when
    a line does not parse or cites an unavailable input."""
    steps: list[tuple[int, frozenset, str]] = []
    for raw in re.split(r"\n|;", text):
        if not raw.strip():
            continue
        m = _LINE.fullmatch(raw.strip())
        if not m:
            raise ValueError(f"not a derivation line: {raw.strip()[:60]}")
        facts = []
        for ref in re.findall(r"@?\d+", m.group(2)):
            if ref.startswith("@"):
                j = int(ref[1:])
                if j >= len(steps):
                    raise ValueError(f"{ref} cites a line not yet derived")
                facts.append(steps[j][2])
            elif int(ref) in premises:
                facts.append(premises[int(ref)])
            else:
                raise ValueError(f"no premise {ref}")
        if int(m.group(1)) not in premises:
            raise ValueError(f"no rule premise {m.group(1)}")
        steps.append((int(m.group(1)), frozenset(facts), _sent(m.group(3))))
    return steps


def logic_derivation(task: dict, final: str) -> Check:
    text, bad = _boxed(final, LOGIC_SCOPE)
    if bad:
        return bad
    premises, target = _logic_problem(_prompt(task))
    reference = set(_derivation(task["answer"], premises))
    try:
        steps = _derivation(text, premises)
    except ValueError as exc:
        return Check("incorrect", f"malformed derivation: {exc}", LOGIC_SCOPE)
    if not steps:
        return Check("incorrect", "no derivation lines", LOGIC_SCOPE)
    off = [k for k, s in enumerate(steps) if s not in reference]
    if off:
        return Check("incorrect", f"line {off[0] + 1} is not a reference step", LOGIC_SCOPE)
    if steps[-1][2] != target:
        return Check("incorrect", "the last line does not derive the target", LOGIC_SCOPE)
    return Check("correct", "", LOGIC_SCOPE)


# --------------------------------------------------------------------------- mathematics

#: polynomial: {((var, power), ...): coefficient}, zero terms removed
Poly = dict[tuple[tuple[str, int], ...], Fraction]
_MAX_POWER, _MAX_TERMS = 64, 4096
_TOK = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|([A-Za-z])|(\*\*|[-+*/()]))")


def _frac_to_div(s: str) -> str:
    """``\\frac{a}{b}`` -> ``((a)/(b))``, brace-balanced, innermost last."""
    for cmd in ("\\dfrac", "\\tfrac", "\\frac"):
        while (i := s.find(cmd)) != -1:
            args, j = [], i + len(cmd)
            for _ in range(2):
                while j < len(s) and s[j] == " ":
                    j += 1
                if j >= len(s) or s[j] != "{":
                    raise ValueError("malformed \\frac")
                depth, k = 0, j
                while k < len(s):
                    depth += {"{": 1, "}": -1}.get(s[k], 0)
                    if depth == 0:
                        break
                    k += 1
                if depth:
                    raise ValueError("unbalanced \\frac")
                args.append(s[j + 1:k])
                j = k + 1
            s = s[:i] + f"(({args[0]})/({args[1]}))" + s[j:]
    return s


def _math_text(s: str) -> str:
    s = _plain(s).replace("−", "-").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\left", "").replace("\\right", "")
    s = _frac_to_div(s).replace("{", "(").replace("}", ")").replace("^", "**")
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    if "\\" in s:
        raise ValueError("an unsupported LaTeX command")
    return s.strip().rstrip(".")


def _mul(a: Poly, b: Poly) -> Poly:
    out: dict = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            powers = Counter(dict(ma))
            powers.update(dict(mb))
            key = tuple(sorted(powers.items()))
            out[key] = out.get(key, 0) + ca * cb
    out = {k: v for k, v in out.items() if v}
    if len(out) > _MAX_TERMS:
        raise ValueError("too many terms")
    return out


def _add(a: Poly, b: Poly, sign: int = 1) -> Poly:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + sign * v
    return {k: v for k, v in out.items() if v}


def _const(p: Poly) -> Fraction | None:
    if not p:
        return Fraction(0)
    return p.get((), None) if set(p) == {()} else None


def parse_poly(text: str) -> Poly:
    """A polynomial from math text, or ValueError. Grammar: sums of products of
    numbers, single-letter variables and parenthesized sums; ``**`` takes a
    constant integer 0..64; ``/`` divides by a nonzero constant only. Juxtaposed
    factors multiply (``2x``, ``3(x+1)``)."""
    s = _math_text(text)
    toks, pos = [], 0
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(f"unexpected {s[pos:pos + 10]!r}")
        pos = m.end()
        if m.group(1):
            toks.append(("num", Fraction(m.group(1))))
        elif m.group(2):
            toks.append(("var", m.group(2)))
        elif m.group(3):
            toks.append(("op", m.group(3)))
    if not toks:
        raise ValueError("empty")
    k = 0

    def peek():
        return toks[k] if k < len(toks) else ("end", None)

    def take():
        nonlocal k
        t = peek()
        k += 1
        return t

    def expr() -> Poly:
        out = term()
        while peek() in (("op", "+"), ("op", "-")):
            sign = 1 if take()[1] == "+" else -1
            out = _add(out, term(), sign)
        return out

    def term() -> Poly:
        out = unary()
        while True:
            t = peek()
            if t == ("op", "*"):
                take()
                out = _mul(out, unary())
            elif t == ("op", "/"):
                take()
                d = _const(unary())
                if d is None or d == 0:
                    raise ValueError("division by a non-constant or zero")
                out = {m: c / d for m, c in out.items()}
            elif t[0] == "var" or t == ("op", "("):
                out = _mul(out, power())
            else:
                return out

    def unary() -> Poly:
        if peek() in (("op", "-"), ("op", "+")):
            sign = -1 if take()[1] == "-" else 1
            return {m: sign * c for m, c in unary().items()}
        return power()

    def power() -> Poly:
        base = atom()
        if peek() == ("op", "**"):
            take()
            e = _const(unary())
            if e is None or e.denominator != 1 or not 0 <= e <= _MAX_POWER:
                raise ValueError("a power that is not a constant integer 0..64")
            out: Poly = {(): Fraction(1)}
            for _ in range(int(e)):
                out = _mul(out, base)
            return out
        return base

    def atom() -> Poly:
        t = take()
        if t[0] == "num":
            return {(): t[1]} if t[1] else {}
        if t[0] == "var":
            return {((t[1], 1),): Fraction(1)}
        if t == ("op", "("):
            inner = expr()
            if take() != ("op", ")"):
                raise ValueError("unbalanced parentheses")
            return inner
        raise ValueError(f"unexpected {t[1]!r}")

    out = expr()
    if k < len(toks):
        raise ValueError(f"trailing {toks[k][1]!r}")
    return out


def _parts(text: str) -> list[str]:
    s = _plain(text).strip()
    if len(s) >= 2 and s[0] + s[-1] in ("()", "[]", "{}"):
        s = s[1:-1]
    s = s.replace("\\{", "").replace("\\}", "")
    return [p.strip() for p in re.split(r",|;|\band\b", s) if p.strip()]


def _value(text: str) -> Fraction:
    v = _const(parse_poly(text))
    if v is None:
        raise ValueError("not a number")
    return v


EXPR_SCOPE = "the boxed expression equals the generator's gold as a polynomial (safe parser; no evaluation of text)"
RATIONAL_SCOPE = "the boxed value equals the generator's gold rational exactly (safe parser)"
ROOTS_SCOPE = "the boxed roots are the generator's roots as a multiset of exact rationals (safe parser)"
SORT_SCOPE = "the boxed list is the generator's ordered list: same names, same exact values, same order"


def _math_check(scope: str, same: Callable[[str, str], bool]) -> Callable[[dict, str], Check]:
    def check(task: dict, final: str) -> Check:
        got = boxed_answer(final or "")
        if got is None:
            return Check("incorrect", "no boxed answer", scope)
        try:
            ok = same(got, task["answer"])
        except (ValueError, RecursionError) as exc:
            return Check("incorrect", f"unparseable answer: {exc}", scope)
        return Check("correct", "", scope) if ok else Check("incorrect", "wrong answer", scope)
    return check


def _same_poly(got: str, gold: str) -> bool:
    return parse_poly(got) == parse_poly(gold)


def _same_roots(got: str, gold: str) -> bool:
    return Counter(_value(p) for p in _parts(got)) == Counter(_value(p) for p in _parts(gold))


def _sort_item(p: str):
    s = _math_text(p)
    return ("name", s) if re.fullmatch(r"[A-Za-z]", s) else ("value", _value(p))


def _same_sorted(got: str, gold: str) -> bool:
    split = lambda t: [p for p in re.split(r",|;|>|<|\\gt|\\lt", _plain(t)) if p.strip()]  # noqa: E731
    return [_sort_item(p) for p in split(got)] == [_sort_item(p) for p in split(gold)]


CHECKERS: dict[str, Callable[[dict, str], Check]] = {
    "reasoning_gym/n_queens": n_queens,
    "reasoning_gym/tower_of_hanoi": tower_of_hanoi,
    "reasoning_core/planning": planning,
    "reasoning_core/logic_derivation": logic_derivation,
    "mathematics_dataset/calculus__differentiate_composed": _math_check(EXPR_SCOPE, _same_poly),
    "mathematics_dataset/probability__swr_p_sequence": _math_check(
        RATIONAL_SCOPE, lambda got, gold: _value(got) == _value(gold)),
    "mathematics_dataset/algebra__polynomial_roots_composed": _math_check(ROOTS_SCOPE, _same_roots),
    "mathematics_dataset/comparison__sort_composed": _math_check(SORT_SCOPE, _same_sorted),
}


def _gold_parses(fn: Callable[[str], object]) -> Callable[[str, str], str | None]:
    def reason(gold: str, prompt: str) -> str | None:
        try:
            fn(gold)
        except ValueError as exc:
            return f"the gold is outside the checker's grammar ({exc})"
        return None
    return reason


def _roots_domain(gold: str, prompt: str) -> str | None:
    if "factor" in prompt.lower():
        return "a factorization item: the project has no factored-form checker"
    try:
        [_value(p) for p in _parts(gold)]
    except ValueError as exc:
        return f"the gold is not a list of rationals ({exc})"
    return None


def _problem_parses(fn: Callable[[str], object]) -> Callable[[str, str], str | None]:
    def reason(gold: str, prompt: str) -> str | None:
        try:
            fn(prompt)
        except ValueError as exc:
            return f"the prompt does not parse ({exc})"
        return None
    return reason


def _logic_domain(gold: str, prompt: str) -> str | None:
    try:
        _derivation(gold, _logic_problem(prompt)[0])
    except ValueError as exc:
        return f"the reference derivation does not parse ({exc})"
    return None


_DOMAIN: dict[str, Callable[[str, str], str | None]] = {
    "reasoning_gym/n_queens": _problem_parses(_nq_problem),
    "reasoning_gym/tower_of_hanoi": _problem_parses(_hanoi_problem),
    "reasoning_core/planning": _problem_parses(_plan_problem),
    "reasoning_core/logic_derivation": _logic_domain,
    "mathematics_dataset/calculus__differentiate_composed": _gold_parses(parse_poly),
    "mathematics_dataset/probability__swr_p_sequence": _gold_parses(_value),
    "mathematics_dataset/algebra__polynomial_roots_composed": _roots_domain,
    "mathematics_dataset/comparison__sort_composed": _gold_parses(
        lambda g: [_sort_item(p) for p in g.split(",")]),
}


def out_of_domain(key: str, gold: str, prompt: str) -> str | None:
    """Why the checker for `key` cannot judge this generator item, or None."""
    return _DOMAIN[key](str(gold), prompt)
