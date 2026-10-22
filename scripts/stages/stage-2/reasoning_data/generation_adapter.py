#!/usr/bin/env python3
"""From generation records to training sessions: verify, assess, select, emit. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.generation_adapter \
        --variant <corpus id> --tasks <pool.jsonl> ... --records <pass records.jsonl> ... \
        --out <sessions.jsonl> [--review]
    # $0 validation on the Stage-0 teacher records (real teacher outputs):
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.generation_adapter --validate-stage0

THE ENGINE is not here. The paid step runs the driver that produced Stage-0's
teacher records, `scripts/stages/stage-0/calibration_volume/run_battery.py
--mode generate`: the teacher's recommended sampling, one seeded request per
turn, and real action -> observation loops for episodes (the environment reads
each reply's action and its observation becomes the next user turn). It writes
generation records and executes no model-written code, so the generation pod's
credentials, network and volume never meet teacher programs.

THIS MODULE then, on the dev box:

1. refuses to process records for a corpus variant whose source plan is not
   ADMITTED (`source_plan`: a source the maintainer has not admitted), unless
   ``--review``, which labels its output review-only;
2. verifies each candidate, keeping every check in its own record
   (`aadistill.data.target_quality` v2):
   * project static tasks: the battery's v2 scorer (boxed answer vs gold) for
     the answer; text consistency as an observation; steps ``not_assessed``
     (no step checker exists for these families);
   * episodes: environment REPLAY for the answer (goal reached) and for the
     ``action_legality`` observation; the natural-language reasoning is not
     verified by a legal action sequence, so steps stay ``not_assessed``;
   * external static tasks: the boxed final answer against the generator's
     gold with the module's declared comparator (`external_modules.json ::
     answer_types`), or, for answer type ``checker``, the module's project
     checker (`checkers.CHECKERS`: a validity check or a simulation, never an
     evaluation of the answer text); a module with neither (``needs_checker``)
     is ``unverifiable`` -- never accepted -- rather than guessed at; a library's
     own scorer is never used. A task that ships reference conclusions
     (PrOntoQA) also gets the step check (`process_check.reference_steps`),
     and its group requires it (``partial``);
   * supporting dataset tasks: the check their ``verifier`` field names
     (`supporting_verify.VERIFIERS`: boxed math against the human gold, the
     declared tests in the probed sandbox, an explicit abstention);
3. accepts under the group's policy (episodes: an illegal action rejects) and
   selects the lowest accepted candidate index per unit, never by length;
4. writes sessions in `build_master_ladder`'s input contract: a static task is
   one session; an accepted episode is its turn-expanded sessions, each
   declaring ``unit_sessions``, so the packer admits the episode whole or not at
   all.

ONE TRAJECTORY, VERIFIED AND TRAINED. Before anything is assessed, every
record must carry an explicit ``candidate_index`` (the generation pass that
wrote it; never defaulted) that no other record of its task shares, and must be
internally the trajectory it claims (`consistency`): its prompt is the task's,
every assistant message is exactly ``split_generation`` of that turn's raw
output, every observation is the next user turn, and -- given a tokenizer --
the recorded output token ids decode to the raw text. A record that fails any
of these stops the run: it is a defect of the harness that wrote it, and
nothing from it is assessed or trained. The verdict and its record then travel
TOGETHER: the selected candidate's own record is the one turned into sessions
(no second lookup by index), and every session carries the record's identity
(``target``: its canonical sha256, the candidate index and each turn's output
token sha256), which the packer writes into the pack (``targets.jsonl``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import process_check as PC
from stages.reasoning_data import source_plan as SP

from aadistill.data.target_quality import AcceptancePolicy, Check, assess, select_target, summarize
from aadistill.data.verify import boxed_answer
from aadistill.evaluation.behavior import split_generation

ENDS = {"natural": ("eos",), "length": ("length_limited", "length", "context_limit"),
        "degeneration": ("degeneration",)}
STATIC = AcceptancePolicy()
EPISODE = AcceptancePolicy(reject_observations={"action_legality": ("illegal_action",)})
PROCESS_GATED = AcceptancePolicy(require_steps=("partial", "verified"))
OUT = A.LOG_DIR / "audit/generation_adapter_validation.json"


class Unparseable(ValueError):
    """The answer is not ONE complete value of its answer type."""


_WRAP = re.compile(r"\\(?:text|textrm|mathrm|textbf|texttt|mathtt|mbox)\s*\{([^{}]*)\}")


def _unwrap(s) -> str:
    """LaTeX text wrappers removed (a boxed ``\\text{False}`` is the answer False),
    surrounding ``$`` and backticks dropped; nothing else is rewritten."""
    out, prev = str(s), None
    while prev != out:
        prev, out = out, _WRAP.sub(r"\1", out)
    return out.strip().strip("$`").strip()


def _ws(s: str) -> str:
    return " ".join(s.split())


_DEC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUMBER = re.compile(_DEC)
_RATIO = re.compile(rf"({_DEC})\s*/\s*({_DEC})")
_TEXFRAC = re.compile(r"([+-]?)\\[dt]?frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")
_LHS = re.compile(r"[A-Za-z]\s*=\s*")


def parse_number(s) -> Fraction:
    """ONE number literal and nothing else -- an integer, a decimal, ``a/b`` or
    ``\\frac{a}{b}``, signed, optionally after ``x =`` -- as an exact rational.
    ``not 7``, ``7 apples``, ``3+4`` and ``1, 2`` are Unparseable."""
    t = _unwrap(s).replace("\u2212", "-").replace("\\left", "").replace("\\right", "").strip()
    t = t[:-1].strip() if t.endswith(".") and not _NUMBER.fullmatch(t) else t
    if (m := _LHS.match(t)):
        t = t[m.end():].strip()
    try:
        if (m := _TEXFRAC.fullmatch(t)):
            v = Fraction(int(m.group(2)), int(m.group(3)))
            return -v if m.group(1) == "-" else v
        if (m := _RATIO.fullmatch(t)):
            return Fraction(m.group(1)) / Fraction(m.group(2))
        if _NUMBER.fullmatch(t):
            return Fraction(t)
    except ZeroDivisionError as exc:
        raise Unparseable("division by zero") from exc
    raise Unparseable(f"not one number: {t[:40]!r}")


def _list_body(s) -> str:
    t = _unwrap(s).replace("\\{", "{").replace("\\}", "}").strip()
    t = t[:-1].strip() if t.endswith(".") else t
    if len(t) >= 2 and t[0] + t[-1] in ("()", "[]", "{}"):
        t = t[1:-1].strip()
    return t


def parse_number_list(s) -> list[Fraction]:
    """Every element of a ``,``/``;``/``and``-separated list parsed as one number;
    an empty element or a non-number anywhere is Unparseable."""
    t = _list_body(s)
    parts = re.split(r"\s*(?:,|;|\band\b)\s*", t)
    if not t or any(not p for p in parts):
        raise Unparseable("an empty element")
    return [parse_number(p) for p in parts]


def _tokens(s, sep=r"[\s,]+") -> list[str]:
    t = _list_body(s)
    toks = [x for x in re.split(sep, t) if x]
    if not toks:
        raise Unparseable("empty")
    return toks


def parse_int_set(s) -> frozenset[int]:
    toks = _tokens(s, r"[\s,;]+")
    if not all(re.fullmatch(r"\d+", x) for x in toks):
        raise Unparseable("an element that is not a non-negative integer")
    if len(set(toks)) != len(toks):
        raise Unparseable("a repeated element")
    return frozenset(int(x) for x in toks)


def parse_assignment(s) -> dict[str, str]:
    """``A=1,B=2``: every part a letter = digit pair, no letter twice."""
    t = re.sub(r"\s*=\s*", "=", _list_body(s))
    parts = [x for x in re.split(r"[\s,;]+", t) if x]
    pairs = [re.fullmatch(r"([A-Za-z])=(\d)", x) for x in parts]
    if not parts or not all(pairs):
        raise Unparseable("a part that is not letter=digit")
    keys = [m.group(1) for m in pairs]
    if len(set(keys)) != len(keys):
        raise Unparseable("a letter assigned twice")
    return {m.group(1): m.group(2) for m in pairs}


_REPR_TOK = re.compile(r"""\s*(?:('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)"""
                       r"""|([A-Za-z_]\w*)|([\[\](){}:,]))""")


def parse_repr(s) -> list[tuple[str, str]]:
    """A Python literal's tokens, whitespace between tokens ignored, nothing
    evaluated: strings keep their content (either quote style), numbers their
    literal text (``2`` and ``2.0`` differ, as their reprs do), names (True,
    None) their case."""
    t, pos, out = _unwrap(s), 0, []
    while pos < len(t):
        if t[pos:].strip() == "":
            break
        m = _REPR_TOK.match(t, pos)
        if not m or m.end() == pos:
            raise Unparseable(f"not a Python literal at {t[pos:pos + 12]!r}")
        pos = m.end()
        kind = next(k for k, g in zip("snip", m.groups()) if g is not None)
        tok = m.group(1)[1:-1] if kind == "s" else m.group("snip".index(kind) + 1)
        out.append((kind, tok))
    if not out:
        raise Unparseable("empty")
    return out


_LATEX_ENV = re.compile(r"\\begin\{[a-z]*\*?\}(?:\{[^{}]*\})?|\\end\{[a-z]*\*?\}|\\hline")


def parse_grid(s) -> list[list[str]]:
    """Rows of cells, structure kept: ``[[a,b],[c,d]]``, or one row per line
    (``\\\\`` too, LaTeX array wrappers dropped, ``&`` a cell separator). Text
    outside the rows, an empty row, or a lone flat list for a multi-row grid
    stays what it is and is compared as such."""
    t = _LATEX_ENV.sub("", _unwrap(s)).strip()
    if t.startswith("["):
        if not t.endswith("]"):
            raise Unparseable("unbalanced brackets")
        inner = t[1:-1].strip()
        if "[" in inner or "]" in inner:
            rows = re.findall(r"\[([^\[\]]*)\]", inner)
            if re.sub(r"\[[^\[\]]*\]", "", inner).strip(" ,\n\t") or not rows:
                raise Unparseable("text outside the rows")
        else:
            rows = [inner]
    else:
        rows = [r for r in re.split(r"\n|\\\\", t) if r.strip()]
    cells = [[c for c in re.split(r"[\s,&]+", r.strip()) if c] for r in rows]
    if not cells or any(not r for r in cells):
        raise Unparseable("an empty row")
    return cells


def _text_exact(s) -> str:
    out = _ws(_unwrap(s))
    if not out:
        raise Unparseable("empty")
    return out


def _text_ci(s) -> str:
    return _text_exact(s).casefold().strip(" .")


def _bool(s) -> str:
    v = _text_ci(s)
    if v not in ("true", "false"):
        raise Unparseable("not True/False")
    return v


DECIMAL_TOL = Fraction(1, 10_000) + Fraction(1, 10 ** 12)


def _same_decimals(got, gold) -> bool:
    a, b = sorted(parse_number_list(got)), sorted(parse_number_list(gold))
    return len(a) == len(b) and all(abs(x - y) <= DECIMAL_TOL for x, y in zip(a, b))


#: answer_type -> (boxed answer, gold) -> equal. Each PARSES THE WHOLE ANSWER and
#: raises `Unparseable` when it is not one complete value of its type; nothing
#: is extracted from a larger text. A comparator, never a score. Which module
#: uses which type, and why its case or order matters, is
#: `external_modules.json :: answer_types` and each module's row.
COMPARE = {
    "text": lambda got, gold: _text_exact(got) == _text_exact(gold),
    "text_ci": lambda got, gold: _text_ci(got) == _text_ci(gold),
    "bool": lambda got, gold: _bool(got) == _bool(gold),
    "number": lambda got, gold: parse_number(got) == parse_number(gold),
    "decimal_set": _same_decimals,
    "int_set": lambda got, gold: parse_int_set(got) == parse_int_set(gold),
    "token_seq": lambda got, gold: _tokens(got) == _tokens(gold),
    "assignment": lambda got, gold: parse_assignment(got) == parse_assignment(gold),
    "repr": lambda got, gold: parse_repr(got) == parse_repr(gold),
    "matrix": lambda got, gold: parse_grid(got) == parse_grid(gold),
}


def compares_equal(kind: str, got, gold) -> bool:
    """`COMPARE` with an unparseable answer read as unequal (for self-checks)."""
    try:
        return bool(COMPARE[kind](got, gold))
    except Unparseable:
        return False


def perturbations(kind: str, gold: str) -> list[str]:
    """Structurally wrong neighbours of a gold that its comparator must refuse:
    a dropped or added element, a reorder where order matters, a reshape, a
    case flip where case matters, trailing words."""
    g = str(gold)
    out = [f"{g} or something else"]
    if kind in ("number",):
        out += [str(parse_number(g) + 1), f"not {g}", f"{g}, {g}"]
    elif kind == "decimal_set":
        v = parse_number_list(g)
        out += [", ".join(str(float(x)) for x in v[:-1]) or "1e9", g + ", 12345.6789",
                ", ".join(str(float(x) + Fraction(1, 1000)) for x in v)]
    elif kind == "int_set":
        v = sorted(parse_int_set(g))
        out += [" ".join(map(str, v[:-1])) or "999", " ".join(map(str, v + [max(v) + 1]))]
    elif kind == "token_seq":
        v = _tokens(g)
        out += [" ".join(v[:-1]) or "x", " ".join(v + v[-1:])]
        if v != v[::-1]:
            out.append(" ".join(v[::-1]))
    elif kind == "assignment":
        d = parse_assignment(g)
        k = sorted(d)[0]
        out += [",".join(f"{a}={(int(b) + 1) % 10 if a == k else b}" for a, b in sorted(d.items())),
                ",".join(f"{a}={b}" for a, b in sorted(d.items())[1:]) or "Q=1"]
    elif kind == "repr":
        out += [g[:-1] + ", 0" + g[-1:] if g[-1:] in "])}" else g + "0", g.replace("-", "", 1) if "-" in g else "-" + g]
    elif kind == "matrix":
        rows = parse_grid(g)
        flat = [c for r in rows for c in r]
        out.append(" ".join(flat) if len(rows) > 1 else "\n".join(flat))
        out.append("\n".join(" ".join(r) for r in rows[:-1]) or "x")
        if len(rows) > 1 and len({len(r) for r in rows}) == 1 and len(rows) == len(rows[0]):
            t = ["\n".join(" ".join(r[i] for r in rows) for i in range(len(rows[0])))]
            if parse_grid(t[0]) != rows:
                out += t
    elif kind == "text" and g.swapcase() != g:
        out.append(g.swapcase())
    elif kind == "bool":
        out.append("False" if _bool(g) == "true" else "True")
    return out


def external_answer(task: dict, final: str) -> Check:
    kind = task.get("answer_type")
    scope = f"the boxed final answer against the generator's gold ({kind} comparator)"
    if kind == "checker":
        from stages.reasoning_data.checkers import CHECKERS
        key = f"{task.get('source')}/{task.get('module')}"
        if key not in CHECKERS:
            return Check("unverifiable", f"no project checker for {key}", "none")
        return CHECKERS[key](task, final or "")
    if kind not in COMPARE:
        return Check("unverifiable", f"no comparator for answer type {kind!r}", scope)
    got = boxed_answer(final or "")
    if got is None:
        return Check("incorrect", "no boxed answer", scope)
    try:
        same = COMPARE[kind](got, task["answer"])
    except Unparseable as exc:
        return Check("incorrect", f"unparseable answer: {exc}", scope)
    return Check("correct", "", scope) if same else Check("incorrect", "wrong answer", scope)


def _stop(rec: dict) -> str:
    return next((t["stop_reason"] for t in rec["turns"] if t["stop_reason"] != "eos"), "eos")


def _protocol(rec: dict) -> str | None:
    return None if rec["turns"] and all(A.split_raw(t.get("raw", "")) for t in rec["turns"]) else "think_not_closed"


def _policy(task: dict) -> AcceptancePolicy:
    if task.get("interactive"):
        return EPISODE
    if task.get("module") and not task.get("verifier") and task.get("reference_conclusions") is not None:
        return PROCESS_GATED
    return STATIC


def _checks(task: dict, rec: dict) -> tuple[Check, Check, dict[str, Check]]:
    """(answer, steps, observations) of one record, every check apart."""
    from stages.calibration_volume import score_battery as SB
    obs: dict[str, Check] = {}
    steps = PC.NO_STEP_CHECKER
    if task.get("interactive"):
        s = SB.score_episode(task, rec)
        inv = int((s.get("env") or {}).get("invalid_actions", 0))
        obs["action_legality"] = Check("all_legal" if inv == 0 else "illegal_action",
                                       f"invalid_actions={inv}", "environment replay of every action")
        ans = Check("correct" if s["correct"] else "incorrect", "" if s["correct"] else "goal not reached",
                    "the environment's goal reached on replay")
    elif task.get("generator_id"):
        s = SB.score_static(task, rec, "v2")
        ok = bool(s.get("answer_correct"))
        ans = Check("correct" if ok else "incorrect", "" if ok else str(s.get("reason", "")),
                    "the boxed final answer against the generator's gold (battery protocol v2)")
        parts = A.split_raw(rec["turns"][-1].get("raw", "")) if rec["turns"] else None
        if parts:
            obs["text_consistency"] = PC.text_consistency(
                parts[0], s.get("extracted"), task["answer_type"],
                prompt=" ".join(m.get("content") or "" for m in task["messages"]))
    elif task.get("verifier"):
        from stages.reasoning_data.supporting_verify import VERIFIERS
        parts = A.split_raw(rec["turns"][-1].get("raw", "")) if rec["turns"] else None
        ans = (VERIFIERS[task["verifier"]](task, *parts) if parts and task["verifier"] in VERIFIERS else
               Check("unverifiable", f"no verifier {task['verifier']!r}" if parts else "think not closed", "none"))
    elif task.get("module"):
        parts = A.split_raw(rec["turns"][-1].get("raw", "")) if rec["turns"] else None
        ans = external_answer(task, parts[1] if parts else "")
        if task.get("reference_conclusions") is not None:
            steps = PC.reference_steps(parts[0] if parts else "", task["reference_conclusions"])
    else:
        ans = Check("unverifiable", f"no verifier for source {task.get('source')!r}", "none")
    return ans, steps, obs


VERIFIER_ERROR_SCOPE = "the verifier raised on this record; nothing about the answer is known"


def candidate(task: dict, rec: dict, index: int):
    """One generation record -> its CandidateRecord, every check apart. A
    verifier that RAISES on one record costs that candidate only: its answer is
    ``unverifiable`` (``verifier error: ...``, counted in the rejection reasons,
    never accepted) and the batch goes on. Anything outside the checks -- the
    ending, the protocol, the assessment -- is not caught."""
    try:
        ans, steps, obs = _checks(task, rec)
    except Exception as exc:  # noqa: BLE001 -- one verifier fault must not end the batch
        ans = Check("unverifiable", f"verifier error: {type(exc).__name__}: {str(exc)[:160]}",
                    VERIFIER_ERROR_SCOPE)
        steps, obs = PC.NO_STEP_CHECKER, {}
    return assess(task["task_id"], index, stop_reason=_stop(rec), protocol_reason=_protocol(rec),
                  answer=ans, steps=steps, observations=obs, policy=_policy(task),
                  end_reasons=ENDS, generated_tokens=sum(int(t["generated_tokens"]) for t in rec["turns"]))


class RecordError(ValueError):
    """Generation records that cannot be assessed: a missing or repeated
    candidate identity, or a record that is not the trajectory it claims."""


def record_sha256(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def candidate_index(rec: dict) -> int:
    i = rec.get("candidate_index")
    if isinstance(i, bool) or not isinstance(i, int) or i < 0:
        raise RecordError(f"{rec.get('task_id')}: candidate_index {i!r} is not an explicit non-negative integer")
    return i


def consistency(task: dict, rec: dict, tokenizer=None) -> list[str]:
    """Every way `rec` can fail to be the one trajectory it claims. Empty when
    its prompt is the task's (after the run's system message, or the task's
    own), each assistant message is exactly
    `split_generation` of that turn's raw output (answer and reasoning), each
    observation is the next user turn, and (with a tokenizer) each turn's
    output token ids decode to its raw text."""
    bad = []
    msgs, turns = rec.get("messages") or [], rec.get("turns") or []
    if rec.get("task_id") != task["task_id"]:
        bad.append("task id differs")
    if not msgs or msgs[0].get("role") != "system":
        bad.append("no system message first")
    # a task's own system message is the run's (generation_plan runs it with --system)
    prompt = task["messages"] if task["messages"][:1] and task["messages"][0]["role"] == "system" else \
        [msgs[0] if msgs else None, *task["messages"]]
    n = len(prompt)
    if msgs[:n] != prompt:
        bad.append("prompt messages differ from the task's")
    rest = msgs[n:]
    if [m.get("role") for m in rest] != ["assistant"] + ["user", "assistant"] * (len(turns) - 1) or not turns:
        bad.append(f"message roles do not match {len(turns)} turns")
        return bad
    for k, t in enumerate(turns):
        a = rest[2 * k]
        parts = split_generation(t.get("raw", ""), think_preopened=bool(t.get("think_preopened", True)))
        if a.get("content") != parts["answer"] or t.get("answer") != parts["answer"]:
            bad.append(f"turn {k}: assistant content is not the raw output's answer")
        if a.get("reasoning_content", "") != parts.get("think", ""):
            bad.append(f"turn {k}: assistant reasoning is not the raw output's reasoning")
        if k + 1 < len(turns) and rest[2 * k + 1].get("content") != t.get("observation"):
            bad.append(f"turn {k}: the next user turn is not this turn's observation")
        if tokenizer is not None and tokenizer.decode(t.get("output_token_ids") or [],
                                                      skip_special_tokens=False) != t.get("raw"):
            bad.append(f"turn {k}: output token ids do not decode to the raw text")
    return bad


def sessions_for(task: dict, rec: dict, stratum: str) -> list[dict]:
    """The accepted record as packer sessions: turn k is the history through the
    k-th assistant message, and every turn declares the unit's size and carries
    the record's identity (`target`)."""
    history = rec["messages"]
    at = [i for i, m in enumerate(history) if m["role"] == "assistant"]
    fam = f"{task['generator_id']}/{task['family_id']}" if task.get("generator_id") else task.get("family_id")
    target = {"record_sha256": record_sha256(rec), "candidate_index": candidate_index(rec),
              "output_token_ids_sha256": [_ids_sha256(t["output_token_ids"]) for t in rec["turns"]]}
    return [{"id": f"{task['task_id']}#t{k}", "unit_id": task["task_id"], "unit_sessions": len(at),
             "stratum": stratum, "family": fam,
             "attributes": {"source": task.get("source", ""), "difficulty": task.get("difficulty", "")},
             "messages": history[:idx + 1], "data_type": stratum, "target": target}
            for k, idx in enumerate(at)]


def adapt(tasks: dict[str, dict], records: list[dict], stratum_of, *, tokenizer=None) -> dict:
    """Refuse the records unless each has an explicit, unique candidate identity
    and is consistent (`RecordError`, every problem listed); then assess each,
    select per unit, and emit the selected candidate's OWN record as sessions."""
    by_unit: dict[str, list] = {}
    problems, seen = [], set()
    for rec in records:
        tid = rec.get("task_id")
        task = tasks.get(tid)
        if task is None:
            problems.append(f"{tid}: record for a task in no given pool")
            continue
        try:
            idx = candidate_index(rec)
        except RecordError as e:
            problems.append(str(e))
            continue
        if (tid, idx) in seen:
            problems.append(f"{tid}: candidate_index {idx} appears more than once")
            continue
        seen.add((tid, idx))
        bad = consistency(task, rec, tokenizer)
        if bad:
            problems.append(f"{tid}#c{idx}: " + "; ".join(bad))
            continue
        by_unit.setdefault(tid, []).append((idx, rec))
    if problems:
        raise RecordError(f"{len(problems)} record problem(s), nothing assessed: " + " | ".join(problems[:20]))
    out, cands, selected = [], [], {}
    for uid, rows in sorted(by_unit.items()):
        pairs = [(candidate(tasks[uid], rec, idx), rec) for idx, rec in rows]
        cands += [c for c, _ in pairs]
        best = select_target([c for c, _ in pairs])
        if best is None:
            continue
        rec = next(r for c, r in pairs if c is best)
        sess = sessions_for(tasks[uid], rec, stratum_of(tasks[uid]))
        selected[uid] = sess[0]["target"]
        out += sess
    return {"sessions": out, "candidates": cands, "selected": selected,
            "summary": summarize(cands, group_of=lambda c: tasks[c.unit_id].get("axis") or "external")}


def stage0_records() -> list[dict]:
    """Stage-0's teacher records as this contract reads them. teacher_cal wrote
    ONE candidate per task and no index; that is checked here (a repeated task
    refuses) and the index 0 is then stated explicitly, never defaulted."""
    recs = A.teacher_cal_records()
    counts = Counter(r["task_id"] for r in recs)
    if max(counts.values()) != 1:
        raise RecordError(f"teacher_cal repeats tasks: {[t for t, n in counts.items() if n > 1][:5]}")
    return [{**r, "candidate_index": 0} for r in recs]


def validate_stage0() -> dict:
    """Run the whole post-generation path on the 672 Stage-0 teacher records and
    pack the result with the shared builder; compare with base_pool_v1."""
    from stages.reasoning_data import build_master_ladder as B
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    tasks = A.calibration_tasks()
    recs = stage0_records()
    tok = B.load_tokenizer()
    got = adapt(tasks, recs, lambda t: SP.stratum_of(t, spec), tokenizer=tok)
    rendered, excluded, _ = B.render_all(tok, got["sessions"], block_len=spec["block_len"])
    admitted = {r.source_id for r in rendered.values()}
    base = {it["unit_id"] for it in A.base_pool_items() if it.get("pool_class") == "primary"}
    accepted = {c.unit_id for c in got["candidates"] if c.accepted_for_training}
    return {"schema": "aadistill.stage2.reasoning_data.generation_adapter_validation/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": {"records": "Stage-0 teacher_cal.jsonl (sha256-verified by assets.teacher_cal_records)",
                      "n_records": len(recs),
                      "candidate_identity": "one record per task (checked); candidate_index 0 stated explicitly",
                      "consistency": ("every record passed `consistency` with the tokenizer: prompt = the task's, "
                                      "assistant messages = split_generation(raw), observations = next user "
                                      "turns, output token ids decode to raw")},
            "contract": got["summary"],
            "accepted_units": len(accepted),
            "sessions_written": len(got["sessions"]),
            "packer_admission": {"admitted_units": len(admitted), **B.exclusion_summary(excluded)},
            "against_base_pool_v1": {"base_pool_units": len(base), "both": len(admitted & base),
                                     "admitted_not_in_base_pool": sorted(admitted - base)[:20],
                                     "in_base_pool_not_admitted": sorted(base - admitted)[:20]},
            "reading": ("the adapter + whole-unit packer reproduce base_pool_v1's unit set when 'both' equals both "
                        "counts; any difference is listed by unit"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-stage0", action="store_true")
    ap.add_argument("--variant")
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--records", nargs="*", default=[], help="one file per candidate pass")
    ap.add_argument("--out")
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args(argv)
    if args.validate_stage0:
        doc = validate_stage0()
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
        print(OUT.relative_to(A.REPO), doc["packer_admission"], doc["against_base_pool_v1"]["both"])
        return 0
    if not (args.variant and args.tasks and args.records and args.out):
        ap.error("--variant, --tasks, --records and --out are required")
    plan = SP.plan(args.variant)
    if not plan["admitted"] and not args.review:
        raise SystemExit(f"{args.variant}: {plan['status']}; pass --review for a review-only output")
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    tasks = {t["task_id"]: t for p in args.tasks for t in A.jsonl(Path(p))}
    from stages.reasoning_data.build_master_ladder import load_tokenizer
    got = adapt(tasks, [r for p in args.records for r in A.jsonl(Path(p))],
                lambda t: SP.stratum_of(t, spec), tokenizer=load_tokenizer())
    out = Path(args.out)
    out.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in got["sessions"]))
    meta = {"corpus_id": args.variant, "review_only": not plan["admitted"],
            "records_sha256": {p: A.sha256_file(Path(p)) for p in args.records},
            "sessions_sha256": A.sha256_file(out),
            "tasks_sha256": {p: A.sha256_file(Path(p)) for p in args.tasks},
            "selected_targets": got["selected"],
            "contract": got["summary"], "by_stratum": dict(Counter(s["stratum"] for s in got["sessions"]))}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(out, meta["review_only"], meta["by_stratum"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
