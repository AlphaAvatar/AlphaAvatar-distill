"""The project checkers accept what is valid, refuse a wrong answer, a near miss
and a malformed one, and never evaluate teacher text."""
from __future__ import annotations

import pytest

from stages.reasoning_data import checkers as CK
from stages.reasoning_data import external_export as EX
from stages.reasoning_data import external_tasks as ET
from stages.reasoning_data.process_check import reference_steps


def task(prompt: str, answer: str = "") -> dict:
    return {"messages": [{"role": "user", "content": prompt + "\n\nPut your final answer inside \\boxed{}."}],
            "answer": answer}


def boxed(s: str) -> str:
    return "Here it is.\n\\boxed{" + s + "}"


def state(key: str, t: dict, answer: str) -> str:
    return CK.CHECKERS[key](t, boxed(answer)).state


# --------------------------------------------------------------------------- n_queens

NQ = task("Given the below board of size 8 x 8 your job is to place 6 queen(s) on the board such that no two "
          "queens attack each other.\n" + "\n".join(["_ _ _ _ _ _ _ _"] * 5 + ["Q _ _ _ _ _ _ _",
                                                   "_ _ _ _ _ _ _ _", "_ _ _ Q _ _ _ _"]))
NQ_GOLD = ["__Q_____", "_____Q__", "_Q______", "______Q_", "____Q___", "Q_______", "_______Q", "___Q____"]


def board(rows: list[str]) -> str:
    return "\n".join(" ".join(r) for r in rows)


def test_n_queens_accepts_any_valid_completion_and_refuses_the_rest():
    key = "reasoning_gym/n_queens"
    assert state(key, NQ, board(NQ_GOLD)) == "correct"
    empty4 = task("Given the below board of size 4 x 4 your job is to place 4 queen(s) on the board such "
                  "that no two queens attack each other.\n" + "\n".join(["_ _ _ _"] * 4))
    assert state(key, empty4, board(["_Q__", "___Q", "Q___", "__Q_"])) == "correct"     # the other solution
    assert state(key, empty4, board(["__Q_", "Q___", "___Q", "_Q__"])) == "correct"
    moved = [r[1:] + r[0] if i == 5 else r for i, r in enumerate(NQ_GOLD)]               # a fixed queen moved
    assert state(key, NQ, board(moved)) == "incorrect"
    other = ["Q_______", "____Q___", "_______Q", "_____Q__", "__Q_____", "______Q_", "_Q______", "___Q____"]
    assert state(key, NQ, board(other)) == "incorrect"            # a valid 8-queens board without (5, 0)
    attack = list(NQ_GOLD)
    attack[0] = "Q_______"                                                               # same column as row 5
    assert state(key, NQ, board(attack)) == "incorrect"
    fewer = list(NQ_GOLD)
    fewer[0] = "________"
    assert state(key, NQ, board(fewer)) == "incorrect"
    assert state(key, NQ, board(NQ_GOLD[:7])) == "incorrect"                             # wrong size
    assert state(key, NQ, board(NQ_GOLD + ["________"])) == "incorrect"                  # valid queens, 9 rows
    assert state(key, NQ, board(NQ_GOLD).replace("Q", "X", 1)) == "incorrect"            # malformed
    assert CK.n_queens(NQ, board(NQ_GOLD)).state == "incorrect"                          # not boxed


# --------------------------------------------------------------------------- tower_of_hanoi

HANOI = task("Solve the Tower of Hanoi problem with 3 disks and 3 pegs.\nMove all disks from Peg 3 to Peg 2 "
             "following the rules:\n- Only one disk can be moved at a time.")
HANOI_GOLD = ["1 3 2", "2 3 1", "1 2 1", "3 3 2", "1 1 3", "2 1 2", "1 3 2"]


def moves(ms: list[str]) -> str:
    return "\n".join("Move disk {} from Peg {} to Peg {}".format(*m.split()) for m in ms)


def test_tower_of_hanoi_simulates_every_move():
    key = "reasoning_gym/tower_of_hanoi"
    assert state(key, HANOI, moves(HANOI_GOLD)) == "correct"
    detour = HANOI_GOLD[:1] + ["1 2 1", "1 1 2"] + HANOI_GOLD[1:]                        # legal, not optimal
    assert state(key, HANOI, moves(detour)) == "correct"
    assert state(key, HANOI, moves(HANOI_GOLD[:-1])) == "incorrect"                      # not finished
    swapped = [HANOI_GOLD[1], HANOI_GOLD[0]] + HANOI_GOLD[2:]                             # disk 2 is not on top
    assert state(key, HANOI, moves(swapped)) == "incorrect"
    onto_smaller = ["1 3 2", "2 3 2"] + HANOI_GOLD[2:]
    assert state(key, HANOI, moves(onto_smaller)) == "incorrect"
    two = task("Solve the Tower of Hanoi problem with 2 disks and 3 pegs.\nMove all disks from Peg 1 to Peg 3")
    assert state(key, two, moves(["1 1 2", "2 1 2", "2 2 3", "1 2 3"])) == "incorrect"   # 2 onto 1, undone later
    assert state(key, two, moves(["1 1 2", "2 1 3", "1 2 3"])) == "correct"
    assert state(key, two, moves(["2 1 2", "1 1 3", "2 2 3"])) == "incorrect"            # right pegs, wrong disks
    assert state(key, HANOI, moves(["1 3 4"] + HANOI_GOLD[1:])) == "incorrect"           # no peg 4
    assert state(key, HANOI, "first disk 1 to the middle") == "incorrect"


# --------------------------------------------------------------------------- planning

PLAN = task("Initial true facts: active(delta), cool(amber), cool(elm). All other facts are false.\n\n"
            "Actions (preconditions -> effects; !fact means false):\n"
            "drain(elm,grove): charged(indigo),cool(elm),linked(linen) -> warm(coral)\n"
            "guide(kestrel,delta): cool(elm),!charged(indigo),!linked(linen) -> warm(coral)\n"
            "drain(jade,grove): cool(elm),!charged(indigo),!linked(linen) -> linked(linen)\n"
            "open(linen,coral): cool(elm),linked(linen),!charged(indigo) -> charged(indigo),!cool(amber)\n"
            "drain(coral,amber): charged(indigo),cool(elm),linked(linen) -> !linked(linen)\n\n"
            "Goal: charged(indigo),!linked(linen).\nCue: exactly 3 actions.\nReturn one grounded action per line.")
PLAN_GOLD = ["drain(jade,grove)", "open(linen,coral)", "drain(coral,amber)"]


def test_planning_simulates_the_plan_under_the_stated_actions():
    key = "reasoning_core/planning"
    assert state(key, PLAN, "\n".join(PLAN_GOLD)) == "correct"
    assert state(key, PLAN, "1. drain(jade, grove)\\\\ 2. open(linen, coral)\\\\ 3. drain(coral, amber)") == "correct"
    assert state(key, PLAN, "\n".join(PLAN_GOLD[1:] + PLAN_GOLD[:1])) == "incorrect"     # precondition fails
    assert state(key, PLAN, "\n".join(PLAN_GOLD[:2])) == "incorrect"                     # cue length, goal
    assert state(key, PLAN, "\n".join(["guide(kestrel,delta)"] + PLAN_GOLD)) == "incorrect"   # goal, not the cue
    skipped = ["open(linen,coral)", "drain(coral,amber)", "guide(kestrel,delta)"]          # goal, 3 actions,
    assert state(key, PLAN, "\n".join(skipped)) == "incorrect"                          # preconditions unmet
    assert state(key, PLAN, "\n".join(PLAN_GOLD[:2] + ["drain(elm,grove)"])) == "incorrect"
    assert state(key, PLAN, "\n".join(PLAN_GOLD[:2] + ["fly(coral,amber)"])) == "incorrect"
    twice = PLAN["messages"][0]["content"].replace("drain(coral,amber): charged",
                                                   "drain(jade,grove): charged")
    assert "defined twice" in CK.out_of_domain(key, "", twice)


# --------------------------------------------------------------------------- logic_derivation

LOGIC = task("Premise:\n0: elena is a parent of farah.\n1: farah is a parent of bruno.\n2: bruno is a parent "
             "of david.\n3: clara is minor.\n4: For all x, y, if x is a parent of y, then x is an ancestor of "
             "y.\n5: From x is a parent of y and y is an ancestor of z, it follows that x is an ancestor of z."
             "\n\nTarget:\nelena is an ancestor of david.\n\nGive derivation lines as Rule: Input... => "
             "Deduction.\nUse premise IDs and @0, @1, ... for derived lines.",
             "4: 2 => bruno is an ancestor of david\n5: 1 @0 => farah is an ancestor of david\n"
             "5: 0 @1 => elena is an ancestor of david")


def test_logic_derivation_accepts_reference_steps_only():
    key = "reasoning_core/logic_derivation"
    assert state(key, LOGIC, LOGIC["answer"]) == "correct"
    assert state(key, LOGIC, "4: 2 => Bruno is an ancestor of David.\n5: @0, 1 => farah is an ancestor of "
                             "david\n5: @1 0 => elena is an ancestor of david") == "correct"   # input order, case
    lines = LOGIC["answer"].split("\n")
    assert state(key, LOGIC, "\n".join(lines[:2])) == "incorrect"                        # target not derived
    assert state(key, LOGIC, LOGIC["answer"].replace("4: 2", "4: 1")) == "incorrect"     # wrong input fact
    assert state(key, LOGIC, LOGIC["answer"].replace("5: 1 @0", "4: 1 @0")) == "incorrect"   # wrong rule
    assert state(key, LOGIC, LOGIC["answer"].replace("@1", "@2")) == "incorrect"         # a line not derived
    assert state(key, LOGIC, "5: 0 @0 => elena is an ancestor of david") == "incorrect"
    assert state(key, LOGIC, "elena is an ancestor of david") == "incorrect"             # malformed


# --------------------------------------------------------------------------- mathematics

def test_the_polynomial_parser_is_exact_and_evaluates_nothing():
    p = CK.parse_poly
    assert p("2(x+1)^2") == p("2*x**2 + 4*x + 2") == p("2x^{2}+4x+2")
    assert p("\\frac{1}{2}x - \\frac{3}{4}") == p("x/2 - 3/4") == p("0.5*x - 0.75")
    assert p("9336*j**2 - 6*j + 4") != p("9336*j**2 - 6*j - 4")
    for bad in ("__import__('os').system('true')", "x**y", "1/x", "\\sqrt{x}", "x**-1", "(x+1", "2x2"):
        with pytest.raises(ValueError):
            p(bad)


def test_mathematics_checkers_compare_values_not_spellings():
    d = "mathematics_dataset/calculus__differentiate_composed"
    t = task("What is the derivative of -350*q**2 + 326*q wrt q?", "326 - 8*q")
    assert state(d, t, "-8q + 326") == "correct" and state(d, t, "q'(q) = 326 - 8 q") == "correct"
    for wrong in ("8q + 326", "326 - 8*q**2", "326", "\\sqrt{q}", "exec('x')"):
        assert state(d, t, wrong) == "incorrect", wrong
    r = "mathematics_dataset/probability__swr_p_sequence"
    t = task("What is prob of sequence oe ...?", "5/72")
    assert state(r, t, "\\frac{5}{72}") == "correct" and state(r, t, "10/144") == "correct"
    assert state(r, t, "0.0694") == "incorrect" and state(r, t, "72/5") == "incorrect"
    roots = "mathematics_dataset/algebra__polynomial_roots_composed"
    t = task("Determine x, given that -7/2*x**2 + 3 - g*x = 0.", "-3, 2/7")
    assert state(roots, t, "x = 2/7, x = -3") == "correct" and state(roots, t, "\\{-3, \\frac{2}{7}\\}") == "correct"
    for wrong in ("-3", "-3, 2/7, 1", "3, 2/7", "-3, -3, 2/7"):
        assert state(roots, t, wrong) == "incorrect", wrong
    assert "factorization" in CK.out_of_domain(roots, "4*v*(v - 1)*(v + 2)", "Factor f(v).")
    s = "mathematics_dataset/comparison__sort_composed"
    t = task("Put f, 1/8, m, 2 in descending order.", "2, m, f, 1/8")
    assert state(s, t, "2, m, f, 0.125") == "correct" and state(s, t, "2 > m > f > 1/8") == "correct"
    for wrong in ("m, 2, f, 1/8", "2, m, 1/8, f", "2, m, f", "2, n, f, 1/8"):
        assert state(s, t, wrong) == "incorrect", wrong


# --------------------------------------------------------------------------- PrOntoQA and the export

PQ_Q = ("Lorpuses are bright. Each lorpus is a zumpus. Every zumpus is not cold. Each zumpus is a shumpus. "
        "Shumpuses are not large. Every sterpus is large. Wren is a grimpus. Wren is a lorpus.")
PQ_CHAIN = ["Wren is a lorpus.", "Each lorpus is a zumpus.", "Wren is a zumpus.", "Each zumpus is a shumpus.",
            "Wren is a shumpus.", "Shumpuses are not large.", "Wren is not large."]


def test_prontoqa_reference_conclusions_are_the_intermediate_derived_facts():
    c = ET.reference_conclusions(PQ_Q, PQ_CHAIN)
    assert c == ["Wren is a zumpus.", "Wren is a shumpus."]                              # premises and last out
    assert reference_steps(" ".join(PQ_CHAIN), c).state == "partial"
    assert reference_steps("Wren is a lorpus, so Wren is not a zumpus.", c).state == "contradicted"
    quoting = "Is it true that Wren is large? Wren is a zumpus, Wren is a shumpus, so Wren is not large."
    assert reference_steps(quoting, c).state == "partial"                                # the query is not a step


def test_a_gold_the_checker_cannot_judge_is_dropped_before_any_task_exists():
    mod = {"answer_type": "checker"}
    row = {"source": "mathematics_dataset", "module": "algebra__polynomial_roots_composed",
           "gold": "-4*(c + 1)**3", "prompt": "Let p = 3. Factor p*c**3 + 5.", "gold_in_prompt": False}
    assert ET.drop_reason(row, mod).startswith("outside the project checker's domain")
    assert ET.drop_reason({**row, "gold": "-1, 8", "prompt": "Let v(q) = 0. What is q?"}, mod) is None


def test_every_checker_module_in_the_config_has_a_checker_and_none_needs_one():
    for (src, mod), row in ET.candidates().items():
        assert row.get("answer_type") != "needs_checker", (src, mod)
        if row.get("answer_type") == "checker":
            assert f"{src}/{mod}" in CK.CHECKERS, (src, mod)


def test_an_entry_seed_depends_on_its_module_and_level_never_its_position():
    a = EX.entry_seed(4_100_000, "reasoning_gym", "n_queens", 0)
    assert a == EX.entry_seed(4_100_000, "reasoning_gym", "n_queens", 0)
    assert EX.entry_seed(4_100_000, "reasoning_gym", "n_queens", 2) == a + 200_000
    assert a != EX.entry_seed(4_100_000, "reasoning_gym", "tower_of_hanoi", 0)
    assert a < 2 ** 31
