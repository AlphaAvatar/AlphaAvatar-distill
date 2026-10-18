"""Interactive adaptation: acquire information through actions and feedback.

Static and interactive protocols are never conflated (AGENTS.md P10.2): an
interactive task is a multi-turn EPISODE. The model replies, the environment
reads the last action line of the reply's final answer, and answers with an
observation, until the episode ends. Each environment is deterministic given its
spec, so an episode replays exactly from the transcript.

An instance carries the full `env` spec (hidden parameters included) for the
scorer; the model only ever sees `messages[0]` and the observations.
"""
from __future__ import annotations

import random
import re
from collections import deque

from .common import Family, Generator, Unsatisfiable, user

_LINE = {
    "query": re.compile(r"QUERY:\s*(-?\d+)", re.I),
    "answer": re.compile(r"ANSWER:\s*(-?\d+)", re.I),
    "press": re.compile(r"PRESS:\s*([A-Za-z])\b", re.I),
    "guess": re.compile(r"GUESS:\s*(\d+)", re.I),
}


def last_action(text: str, kinds: tuple[str, ...]):
    """(kind, value) of the LAST action line in `text`, or (None, None)."""
    best = (None, None, -1)
    for k in kinds:
        for m in _LINE[k].finditer(text or ""):
            if m.start() > best[2]:
                best = (k, m.group(1), m.start())
    return best[0], best[1]


# --- environments ------------------------------------------------------------

class ProbeFunctionEnv:
    """A hidden integer function; ask f(x) for chosen x, then answer f(target)."""

    def __init__(self, spec: dict):
        self.spec, self.left = spec, spec["queries"]
        self.done, self.success, self.invalid, self.used = False, False, 0, 0
        self.turns = 0

    @staticmethod
    def f(spec: dict, x: int) -> int:
        k, p = spec["family"], spec["params"]
        if k == "probe.affine":
            return p["a"] * x + p["b"]
        if k == "probe.quadratic":
            return x * x + p["b"]
        if k == "probe.mod_scale":
            return (x % p["m"]) * p["k"]
        if k == "probe.threshold":
            return p["lo"] if x < p["t"] else x + p["hi"]
        if k == "probe.digit_sum":
            return sum(int(c) for c in str(abs(x))) + p["c"]
        if k == "probe.floor_div":
            return x // p["k"] + p["c"]
        raise KeyError(k)

    def step(self, reply: str) -> str | None:
        self.turns += 1
        kind, val = last_action(reply, ("query", "answer"))
        if kind == "answer":
            self.done = True
            self.success = int(val) == self.f(self.spec, self.spec["target"])
            return None
        if kind == "query" and self.left > 0:
            x = int(val)
            if not self.spec["lo"] <= x <= self.spec["hi"]:
                self.invalid += 1
                return (f"You may only ask about integers from {self.spec['lo']} to "
                        f"{self.spec['hi']}. You have {self.left} questions left.")
            self.left -= 1
            self.used += 1
            tail = (f"You have {self.left} questions left." if self.left else
                    f"You have no questions left. Reply with ANSWER: <value> for "
                    f"f({self.spec['target']}).")
            return f"f({x}) = {self.f(self.spec, x)}. {tail}"
        if kind == "query":
            self.invalid += 1
            return (f"You have no questions left. Reply with ANSWER: <value> for "
                    f"f({self.spec['target']}).")
        self.invalid += 1
        return ("I could not find a QUERY or ANSWER line. End your reply with "
                "QUERY: <integer> or ANSWER: <integer>.")

    def result(self) -> dict:
        return {"success": self.success, "done": self.done, "queries_used": self.used,
                "invalid_actions": self.invalid, "turns": self.turns}


class ButtonsEnv:
    """A counter and buttons with hidden effects; reach the goal value."""

    def __init__(self, spec: dict):
        self.spec, self.value = spec, spec["start"]
        self.left, self.done, self.success = spec["presses"], False, False
        self.invalid, self.turns, self.pressed = 0, 0, 0

    @staticmethod
    def apply(spec: dict, button: str, v: int) -> int:
        op, arg = spec["buttons"][button]
        if op == "add":
            v = v + arg
        elif op == "mul":
            v = v * arg
        elif op == "rev":
            v = int(str(abs(v))[::-1]) * (1 if v >= 0 else -1)
        if spec.get("modulus"):
            v %= spec["modulus"]
        return v

    def step(self, reply: str) -> str | None:
        self.turns += 1
        kind, val = last_action(reply, ("press",))
        if kind != "press" or val.upper() not in self.spec["buttons"]:
            self.invalid += 1
            return ("I could not find a valid PRESS line. End your reply with "
                    f"PRESS: <one of {', '.join(sorted(self.spec['buttons']))}>.")
        self.value = self.apply(self.spec, val.upper(), self.value)
        self.left -= 1
        self.pressed += 1
        if self.value == self.spec["goal"]:
            self.done, self.success = True, True
            return None
        if self.left == 0:
            self.done = True
            return None
        return (f"The counter now shows {self.value}. The goal is {self.spec['goal']}. "
                f"You have {self.left} presses left.")

    def result(self) -> dict:
        return {"success": self.success, "done": self.done, "presses_used": self.pressed,
                "invalid_actions": self.invalid, "turns": self.turns,
                "final_value": self.value}


class CodeEnv:
    """Guess a hidden digit code from structured feedback."""

    def __init__(self, spec: dict):
        self.spec, self.left = spec, spec["guesses"]
        self.done, self.success, self.invalid, self.turns, self.used = False, False, 0, 0, 0

    @staticmethod
    def feedback(spec: dict, guess: str) -> str:
        code, kind = spec["code"], spec["family"]
        exact = sum(a == b for a, b in zip(guess, code))
        if kind == "code.pegs":
            common = sum(min(guess.count(d), code.count(d)) for d in set(guess))
            return (f"{exact} digit(s) correct and in the right place, "
                    f"{common - exact} correct digit(s) in the wrong place.")
        if kind == "code.exact_only":
            return f"{exact} digit(s) correct and in the right place."
        if kind == "code.position_hints":
            hints = []
            for i, (a, b) in enumerate(zip(guess, code), 1):
                hints.append(f"position {i}: " + ("correct" if a == b else
                                                  "higher" if b > a else "lower"))
            return "; ".join(hints) + "."
        raise KeyError(kind)

    def step(self, reply: str) -> str | None:
        self.turns += 1
        kind, val = last_action(reply, ("guess",))
        n, digits = self.spec["length"], self.spec["digits"]
        if kind != "guess" or len(val) != n or any(c not in digits for c in val):
            self.invalid += 1
            return (f"I could not find a valid GUESS line. End your reply with GUESS: "
                    f"followed by {n} digits from {digits[0]} to {digits[-1]}.")
        self.used += 1
        self.left -= 1
        if val == self.spec["code"]:
            self.done, self.success = True, True
            return None
        if self.left == 0:
            self.done = True
            return None
        return f"{self.feedback(self.spec, val)} You have {self.left} guesses left."

    def result(self) -> dict:
        return {"success": self.success, "done": self.done, "guesses_used": self.used,
                "invalid_actions": self.invalid, "turns": self.turns}


ENVIRONMENTS = {"probe_function": ProbeFunctionEnv, "buttons": ButtonsEnv,
                "code": CodeEnv}


def make_env(spec: dict):
    return ENVIRONMENTS[spec["kind"]](spec)


def max_turns(spec: dict) -> int:
    """Hard episode length: the action budget plus room for format errors."""
    budget = {"probe_function": spec.get("queries", 0) + 1,
              "buttons": spec.get("presses", 0),
              "code": spec.get("guesses", 0)}[spec["kind"]]
    return budget + 2


# --- generators ----------------------------------------------------------------

_PKN = {"L1": 7, "L2": 5, "L3": 4}


def _probe_family(fid: str):
    def make(rng: random.Random, difficulty: str) -> dict:
        if fid == "probe.affine":
            params = {"a": rng.randint(2, 5), "b": rng.randint(-9, 9)}
        elif fid == "probe.quadratic":
            params = {"b": rng.randint(-9, 9)}
        elif fid == "probe.mod_scale":
            params = {"m": rng.randint(3, 9), "k": rng.randint(1, 4)}
        elif fid == "probe.threshold":
            params = {"t": rng.randint(20, 80), "lo": rng.randint(0, 9), "hi": rng.randint(1, 9)}
        elif fid == "probe.digit_sum":
            params = {"c": rng.randint(0, 5)}
        elif fid == "probe.floor_div":
            params = {"k": rng.randint(2, 7), "c": rng.randint(-5, 5)}
        else:
            raise KeyError(fid)
        spec = {"kind": "probe_function", "family": fid, "params": params,
                "lo": 0, "hi": 100, "queries": _PKN[difficulty],
                "target": rng.randint(101, 999)}
        target_val = ProbeFunctionEnv.f(spec, spec["target"])
        prompt = (
            "There is a hidden function f that takes an integer and returns an integer. "
            f"You may ask for the value of f at any integer from {spec['lo']} to "
            f"{spec['hi']}: end your reply with a line QUERY: <integer>, and I will tell "
            f"you the value. You can ask at most {spec['queries']} questions, one per "
            "reply.\n\n"
            f"Your goal is to find f({spec['target']}). When you are ready, end your "
            "reply with a line ANSWER: <integer>. Your first ANSWER ends the game.")
        return {"messages": user(prompt), "env": spec, "answer": str(target_val),
                "answer_type": "int"}

    return Family(fid, "hidden_function", make)


PROBE_FUNCTION = Generator(
    "probe_function@v1", "interactive_adaptation", "int",
    tuple(_probe_family(f) for f in ("probe.affine", "probe.quadratic", "probe.mod_scale",
                                    "probe.threshold", "probe.digit_sum",
                                    "probe.floor_div")),
    interactive=True,
    description="choose informative queries to identify a hidden function")

_BKN = {"L1": 12, "L2": 10, "L3": 8}


def _bfs_len(spec, start, goal, limit):
    seen, q = {start: 0}, deque([start])
    while q:
        v = q.popleft()
        if v == goal:
            return seen[v]
        if seen[v] >= limit:
            continue
        for b in spec["buttons"]:
            w = ButtonsEnv.apply(spec, b, v)
            if w not in seen and abs(w) < 10 ** 5:
                seen[w] = seen[v] + 1
                q.append(w)
    return None


def _buttons_family(fid: str):
    def make(rng: random.Random, difficulty: str) -> dict:
        if fid == "act.additive":
            a, b, c = rng.sample(range(2, 10), 3)
            buttons = {"A": ("add", a), "B": ("add", b), "C": ("add", -c)}
            modulus = None
        elif fid == "act.multiply":
            buttons = {"A": ("mul", 2), "B": ("add", rng.randint(1, 5)),
                       "C": ("add", -rng.randint(1, 5))}
            modulus = None
        elif fid == "act.reverse":
            buttons = {"A": ("rev", 0), "B": ("add", rng.randint(2, 9)),
                       "C": ("add", -rng.randint(1, 4))}
            modulus = None
        elif fid == "act.modular":
            a, b = rng.sample(range(3, 12), 2)
            buttons = {"A": ("add", a), "B": ("add", b), "C": ("add", -1)}
            modulus = 20
        else:
            raise KeyError(fid)
        names = list(buttons)
        rng.shuffle(names)
        buttons = {new: buttons[old] for new, old in zip("ABC", names)}
        presses = _BKN[difficulty]
        spec = {"kind": "buttons", "family": fid, "buttons": buttons, "modulus": modulus,
                "start": rng.randint(0, 19) if modulus else rng.randint(1, 30),
                "presses": presses}
        goal = rng.randint(0, 19) if modulus else rng.randint(1, 80)
        spec["goal"] = goal
        d = _bfs_len(spec, spec["start"], goal, presses)
        if d is None or d < 3 or d > presses - 4:
            raise Unsatisfiable
        spec["optimal_presses"] = d
        wrap = (f" The counter only shows values from 0 to {modulus - 1}: it wraps "
                "around." if modulus else "")
        prompt = (
            f"A counter shows {spec['start']}. There are three buttons, A, B and C. "
            "Each button changes the counter in its own fixed way, but you are not told "
            f"how.{wrap} Your goal is to make the counter show exactly {goal}.\n\n"
            f"You can press one button per reply, at most {presses} times in total: end "
            "each reply with a line PRESS: <A, B or C>. After each press I will tell you "
            "what the counter shows. The game ends as soon as the counter shows the goal.")
        return {"messages": user(prompt), "env": spec, "answer": None}

    return Family(fid, "hidden_dynamics", make)


HIDDEN_BUTTONS = Generator(
    "hidden_buttons@v1", "interactive_adaptation", "episode",
    tuple(_buttons_family(f) for f in ("act.additive", "act.multiply", "act.reverse",
                                      "act.modular")),
    interactive=True,
    description="discover hidden action effects by trying them, then plan to a goal")

_CKN = {"L1": (3, "123456", False, 10), "L2": (3, "123456", True, 9),
        "L3": (4, "123456", True, 9)}


def _code_family(fid: str):
    def make(rng: random.Random, difficulty: str) -> dict:
        n, digits, repeats, guesses = _CKN[difficulty]
        code = ("".join(rng.choice(digits) for _ in range(n)) if repeats
                else "".join(rng.sample(digits, n)))
        spec = {"kind": "code", "family": fid, "code": code, "length": n,
                "digits": digits, "guesses": guesses, "repeats": repeats}
        rule = {"code.pegs": "how many digits are correct and in the right place, and "
                             "how many are correct but in the wrong place",
                "code.exact_only": "how many digits are correct and in the right place",
                "code.position_hints": "for each position, whether your digit is "
                                       "correct, or whether the hidden digit is higher "
                                       "or lower"}[fid]
        prompt = (
            f"I am thinking of a {n}-digit code. Each digit is from {digits[0]} to "
            f"{digits[-1]}; digits {'may' if repeats else 'do not'} repeat.\n\n"
            f"Guess it: end each reply with a line GUESS: <{n} digits>. After each guess "
            f"I will tell you {rule}. You have {guesses} guesses. The game ends when "
            "you guess the code.")
        return {"messages": user(prompt), "env": spec, "answer": None}

    return Family(fid, "feedback_search", make)


CODE_BREAKER = Generator(
    "code_breaker@v1", "interactive_adaptation", "episode",
    tuple(_code_family(f) for f in ("code.pegs", "code.exact_only",
                                   "code.position_hints")),
    interactive=True,
    description="identify a hidden code from structured feedback")

GENERATORS = (PROBE_FUNCTION, HIDDEN_BUTTONS, CODE_BREAKER)
