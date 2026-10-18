"""Abstract and analogical reasoning: map a demonstrated relation onto a new case.

Two generators. Letter analogies (Copycat-style) operate on runs of consecutive
letters; grid transformations operate on small digit grids. Both keep an
instance only when every rule in the library consistent with the demonstrations
agrees on the query. These are this project's own generators and families; no
ARC-AGI task, solution or trajectory is used or derived from (AGENTS.md P10.2).
"""
from __future__ import annotations

import random

from .common import (ANSWER_INSTRUCTION, Family, Generator, Unsatisfiable,
                     fmt_grid, grid_answer, user)

_A = "abcdefghijklmnopqrstuvwxyz"

# --- letter analogies ----------------------------------------------------------


def _succ(c, k=1):
    i = _A.index(c) + k
    if not 0 <= i < 26:
        raise ValueError
    return _A[i]


_LETTER_RULES = {
    "letter.succ_last": ("successor", lambda s: s[:-1] + _succ(s[-1])),
    "letter.pred_first": ("successor", lambda s: _succ(s[0], -1) + s[1:]),
    "letter.succ_all": ("successor", lambda s: "".join(_succ(c) for c in s)),
    "letter.extend_run": ("length", lambda s: s + _succ(s[-1], _A.index(s[-1]) - _A.index(s[-2]))),
    "letter.shrink_run": ("length", lambda s: s[:-1]),
    "letter.reverse": ("order", lambda s: s[::-1]),
    "letter.double_last": ("repetition", lambda s: s + s[-1]),
    "letter.double_all": ("repetition", lambda s: "".join(c + c for c in s)),
    "letter.swap_halves": ("order", lambda s: s[len(s) // 2:] + s[:len(s) // 2]),
    "letter.mirror_first": ("repetition", lambda s: s[0] + s + s[0]),
}
_LKN = {"L1": (3, 3, 4, 1), "L2": (2, 3, 5, 1), "L3": (2, 4, 6, 2)}  # demos, lo, hi, max step


def _letter_family(fid: str):
    cls, fn = _LETTER_RULES[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        n_demo, lo, hi, step_max = _LKN[difficulty]

        def run():
            length = rng.randint(lo, hi)
            step = rng.randint(1, step_max)
            start = rng.randint(1, 25 - step * length)
            return "".join(_A[start + step * i] for i in range(length))

        try:
            demos = [(s, fn(s)) for s in (run() for _ in range(n_demo))]
            query = run()
            gold = fn(query)
        except (ValueError, IndexError):
            raise Unsatisfiable
        answers = set()
        for _f, (_c, g) in _LETTER_RULES.items():
            try:
                if all(g(x) == y for x, y in demos):
                    answers.add(g(query))
            except (ValueError, IndexError):
                continue
        if answers != {gold}:
            raise Unsatisfiable
        lines = [f"If {x} changes to {y}," for x, y in demos]
        prompt = ("\n".join(lines) + f"\nwhat does {query} change to?\n\n"
                  + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": gold}

    return Family(fid, cls, make)


LETTER_ANALOGY = Generator(
    "letter_analogy@v1", "abstract_analogical", "word",
    tuple(_letter_family(f) for f in _LETTER_RULES),
    description="Copycat-style analogies over runs of consecutive letters")

# --- digit grids ---------------------------------------------------------------


def _flip_h(g):
    return [row[::-1] for row in g]


def _flip_v(g):
    return g[::-1]


def _transpose(g):
    return [list(r) for r in zip(*g)]


def _rot_cw(g):
    return [list(r) for r in zip(*g[::-1])]


def _rot_ccw(g):
    return [list(r) for r in zip(*g)][::-1]


def _gravity(g):
    h, w = len(g), len(g[0])
    out = [[0] * w for _ in range(h)]
    for c in range(w):
        col = [g[r][c] for r in range(h) if g[r][c]]
        for i, v in enumerate(reversed(col)):
            out[h - 1 - i][c] = v
    return out


def _shift_right(g):
    return [[row[-1]] + row[:-1] for row in g]


def _fill_rows(g):
    out = []
    for row in g:
        nz = [v for v in row if v]
        out.append([nz[0]] * len(row) if len(set(nz)) == 1 else row[:])
    return out


def _mirror_lr(g):
    w = len(g[0])
    return [row[: (w + 1) // 2] + row[: w // 2][::-1] for row in g]


def _border(g):
    h, w = len(g), len(g[0])
    return [[(8 if (r in (0, h - 1) or c in (0, w - 1)) and not g[r][c] else g[r][c])
             for c in range(w)] for r in range(h)]


def _count_row(g):
    n = sum(1 for row in g for v in row if v)
    w = len(g[0])
    if n > w * len(g):
        raise ValueError
    flat = [1] * n + [0] * (w * len(g) - n)
    return [flat[i * w:(i + 1) * w] for i in range(len(g))]


def _recolor_factory(a, b):
    return lambda g: [[b if v == a else v for v in row] for row in g]


_GRID_RULES = {
    "grid.flip_h": ("symmetry", [_flip_h]),
    "grid.flip_v": ("symmetry", [_flip_v]),
    "grid.transpose": ("symmetry", [_transpose]),
    "grid.rotate_cw": ("rotation", [_rot_cw]),
    "grid.rotate_ccw": ("rotation", [_rot_ccw]),
    "grid.gravity_down": ("physics", [_gravity]),
    "grid.shift_right": ("translation", [_shift_right]),
    "grid.fill_rows": ("completion", [_fill_rows]),
    "grid.mirror_lr": ("completion", [_mirror_lr]),
    "grid.border": ("completion", [_border]),
    "grid.count_cells": ("counting", [_count_row]),
    "grid.recolor": ("color", [_recolor_factory(a, b) for a in range(1, 8)
                               for b in range(1, 10) if a != b]),
}
_GKN = {"L1": (3, 3, 3, 0.35), "L2": (2, 4, 4, 0.4), "L3": (2, 5, 5, 0.45)}


def _grid_family(fid: str):
    cls, variants = _GRID_RULES[fid]

    def make(rng: random.Random, difficulty: str) -> dict:
        n_demo, h, w, dens = _GKN[difficulty]
        fn = rng.choice(variants)
        colors = rng.sample(range(1, 8), 3)

        def grid():
            g = [[(rng.choice(colors) if rng.random() < dens else 0) for _ in range(w)]
                 for _ in range(h)]
            if fid == "grid.recolor":
                # make sure the recolored color is present
                r, c = rng.randrange(h), rng.randrange(w)
                g[r][c] = recolor_src
            return g

        recolor_src = None
        if fid == "grid.recolor":
            k = variants.index(fn)
            pairs = [(a, b) for a in range(1, 8) for b in range(1, 10) if a != b]
            recolor_src = pairs[k][0]
            if recolor_src not in colors:
                colors[0] = recolor_src
        try:
            demos = [(g, fn(g)) for g in (grid() for _ in range(n_demo))]
            query = grid()
            gold = fn(query)
        except (ValueError, IndexError):
            raise Unsatisfiable
        if any(x == y for x, y in demos) or gold == query:
            raise Unsatisfiable
        answers = set()
        for _f, (_c, vs) in _GRID_RULES.items():
            for g in vs:
                try:
                    if all(g(x) == y for x, y in demos):
                        answers.add(grid_answer(g(query)))
                except (ValueError, IndexError):
                    continue
        if answers != {grid_answer(gold)}:
            raise Unsatisfiable
        blocks = []
        for i, (x, y) in enumerate(demos, 1):
            blocks.append(f"Example {i} input:\n{fmt_grid(x)}\nExample {i} output:\n{fmt_grid(y)}")
        prompt = ("Each example input grid was turned into its output grid by the same "
                  "hidden rule. 0 is empty.\n\n" + "\n\n".join(blocks)
                  + f"\n\nTest input:\n{fmt_grid(query)}\n\n"
                  "Write the test output grid with its rows separated by \" / \", "
                  "for example \\boxed{120 / 003 / 450}. " + ANSWER_INSTRUCTION)
        return {"messages": user(prompt), "answer": grid_answer(gold)}

    return Family(fid, cls, make)


GRID_TRANSFORM = Generator(
    "grid_transform@v1", "abstract_analogical", "grid",
    tuple(_grid_family(f) for f in _GRID_RULES),
    description="infer a grid transformation from demonstrations (own generator, not ARC data)")

GENERATORS = (LETTER_ANALOGY, GRID_TRANSFORM)
