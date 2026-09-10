#!/usr/bin/env python3
"""Eight ownership violations the literal and semantic gates could not see.

    PYTHONPATH=src:scripts python scripts/architecture/core_ownership.py [--write]

Both existing gates reported ZERO while all eight defects below were live in
`src/aadistill`. They were not evaded; they were out of vocabulary:

* `architecture_inventory.py` reads LITERALS and asks who owns them. A phase
  name inside a longer enforcement sentence is not a path, a repo id or a
  sha256, so nothing looked at it.
* `semantic_hardcode.py` reads SHAPE -- seed-like ints, capability tuples,
  dataclass defaults that construct a policy object. A default of `True`, or of
  the string `"correct_overall"`, has the shape of an ordinary parameter.

So this checks a third thing: **who owns the decision a line encodes.** A
serialized payload naming a phase, a plan id that is a study, a metric default
that is a study's choice, a scoring rule inside a planning function, a core read
of `configs/`, an import-time deployment load, a driver's stage NUMBER in a
marker policy, and a result-shape branch that fails open on an unrecognized
object.

Each rule exists because a real defect survived the other two gates, and
`tests/architecture/test_core_ownership.py` restores that exact defect and
requires this to flag it. Deliberately not flagged: mathematical constants,
schema versions, generic terminal types, and the operator taxonomy -- the false
positives of the first semantic pass.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "src/aadistill"

RULES = (
    "phase_in_payload",
    "concrete_id_default",
    "experiment_default",
    "scoring_semantics",
    "core_reads_repository",
    "import_time_deployment_load",
    "numeric_stage_policy",
    "untyped_result_branch",
    "instance_prose",
)

# ---------------------------------------------------------------------------
# rule 9: instance prose in DOCSTRINGS AND COMMENTS
#
# The other eight rules read code and skip prose, which is correct for them: a
# phase name in a `raise` explains a refusal to a human. But the same file's
# docstring can carry this project's entire run history -- which experiment
# measured what, at which probe size, on which seeds, for how many dollars --
# and a reusable core that documents itself by one campaign's incidents is
# owned by that campaign whether or not any of it is executable.
#
# Deliberately NARROW. It matches an instance IDENTITY (a run label, a phase,
# an attempt number), an instance MEASUREMENT this project made, an instance
# SEED name, and a campaign COST. It does not match generic mathematics, schema
# versions, algorithm terminology, model-family capabilities, or the AGENTS.md
# principle numbers (`P3`, `P10`) that state the project's rules rather than
# its results.
# ---------------------------------------------------------------------------

#: A run label as this project spells them -- `E7`, `E6b`, `P2` -- when used as
#: an identity. `P0`-`P4` collide with AGENTS.md principle numbers, so those are
#: disambiguated by context below rather than by the pattern.
RUN_LABEL = re.compile(r"(?<![A-Za-z0-9_.])(E[1-9][0-9]?[ab]?|P[0-4])(?=[ :,)./]|$)")
#: A phase, an attempt, or a numbered session of one.
PHASE_ATTEMPT = re.compile(
    r"\bPhase[- ][ABC]\b|\bAttempt[- ]?\d+\b|\bC1 (?:attempt|session|run)\b")
#: Measurements THIS project made, at the sizes it made them.
INSTANCE_MEASURE = re.compile(
    r"\b0\.86\s?M\b|\b0\.1290\b|[−-]5\.22\b|\b190 (?:prompts|items)\b|"
    r"\b170 are\b|\b= ?340\b|\b170 correctness")
#: This project's seed names.
SEED_LABEL = re.compile(r"\bseeds? s[abc]\b|\bs[abc] ?\+ ?s[abc]\b|\bon seed s[abc]\b")
#: A campaign dollar figure. AGENTS.md P12.1: a task's dollar amounts live in
#: that task's governance artifact, never in reusable core.
CAMPAIGN_COST = re.compile(r"\$\s?\d+\.\d{2,4}")

#: `P3`/`P10`-style references to AGENTS.md principles are the project's RULES,
#: not its results, and a core module may cite the rule it implements.
PRINCIPLE_CITATION = re.compile(
    r"AGENTS\.md[^.]*|\bP\d+(?:\.\d+)?/P\d+|\bP\d+(?:\.\d+)?\)|principle")

INSTANCE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("run label", RUN_LABEL),
    ("phase/attempt", PHASE_ATTEMPT),
    ("measurement", INSTANCE_MEASURE),
    ("seed", SEED_LABEL),
    ("cost", CAMPAIGN_COST),
)

#: Modules whose BYTES are pinned by the accepted real-CUDA engineering
#: validation of 2026-09-10 (execution SHA 7027a8f4). The review that accepted
#: that evidence also directed that this exact surface not be modified, so its
#: prose is reported separately rather than rewritten: editing it would put the
#: reporting tip's copy of a validated file out of step with the commit the
#: evidence names, for a comment.
#:
#: This is a DEFERRAL, not an exemption. `scan()` counts these findings and
#: `main()` prints them; they are excluded only from the `total`, which is the
#: number the merge gate reads. Removing a file from this tuple must make its
#: findings count, and adding one must be justified by the CUDA acceptance --
#: `tests/architecture/test_core_ownership.py` pins the membership.
CUDA_VALIDATED_SURFACE: tuple[str, ...] = (
    "src/aadistill/initialization/operators/attention_activation.py",
    "src/aadistill/initialization/statistics/attention.py",
    "src/aadistill/initialization/device.py",
    "src/aadistill/initialization/planning/fixed_path.py",
    "src/aadistill/initialization/adapters/__init__.py",
    "src/aadistill/initialization/adapters/qwen3.py",
)

#: A phase/attempt/session NAME, as this project spells them.
PHASE_NAME = re.compile(r"\bPhase [ABC]\b|\bphase_[abc]\b|\bAttempt \d|\bC[01] "
                        r"(?:attempt|session)\b")
#: Repository trees a reusable core must not name.
REPO_TREE = re.compile(r"(^|[\"'/ ])(configs?|logs)/")
#: Bindings whose value is an identity of a STUDY rather than of a format.
ID_FIELD = re.compile(r"(^|_)(plan_id|experiment_id|study_id|battery_id|"
                      r"run_id|session_id)$")
#: Study-choice fields: a metric NAME or a control-design switch.
EXPERIMENT_FIELD = re.compile(
    r"(^|_)(primary_metric|secondary_metric|feasibility_metric|"
    r"feasibility_min|include_canonical_control|ranking_metric|"
    r"objective_metric)$")
#: Format identities, which are NOT study identities.
FORMAT_FIELD = re.compile(r"(^|_)(schema|version|contract|spec_id|kind)$")


def _docstrings(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            d = ast.get_docstring(n, clean=False)
            if d:
                out.add(d)
    return out


def _raised_strings(tree: ast.AST) -> set[int]:
    """Line numbers of strings inside a `raise`. A refusal explains itself to a
    human and is not stored, so a phase name there is prose, not a payload."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Raise):
            for c in ast.walk(n):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.add(id(c))
    return out


def _dict_value_strings(tree: ast.AST) -> dict[int, str]:
    """Strings that are VALUES in a dict literal -- i.e. serialized payload."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                key = k.value if isinstance(k, ast.Constant) else "?"
                for c in ast.walk(v):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        out[id(c)] = str(key)
    return out


class Walker(ast.NodeVisitor):
    def __init__(self, path: str, source: str):
        self.path = path
        self.tree = ast.parse(source)
        self.docs = _docstrings(self.tree)
        self.raised = _raised_strings(self.tree)
        self.payload = _dict_value_strings(self.tree)
        self.findings: list[dict] = []
        self._class: list[str] = []

    def add(self, rule, node, name, why):
        self.findings.append({"rule": rule, "path": self.path,
                              "line": getattr(node, "lineno", 0),
                              "name": name, "why": why})

    # -- 1. a phase name inside a serialized payload --------------------------
    def visit_Dict(self, node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values):
            key = k.value if isinstance(k, ast.Constant) else "?"
            for c in ast.walk(v):
                if not (isinstance(c, ast.Constant) and isinstance(c.value, str)):
                    continue
                if c.value in self.docs or id(c) in self.raised:
                    continue
                if PHASE_NAME.search(c.value):
                    self.add("phase_in_payload", c, str(key),
                             f"the serialized value of {key!r} names a phase; a "
                             "generic serializer must take that sentence from "
                             "the application policy")
        self.generic_visit(node)

    # -- 2/3. dataclass field defaults ---------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class.append(node.name)
        for stmt in node.body:
            if not (isinstance(stmt, ast.AnnAssign) and stmt.value is not None
                    and isinstance(stmt.target, ast.Name)):
                continue
            name = stmt.target.id
            if self._is_required(stmt.value):
                continue
            if ID_FIELD.search(name) and isinstance(stmt.value, ast.Constant) \
                    and isinstance(stmt.value.value, str) and stmt.value.value:
                self.add("concrete_id_default", stmt, name,
                         f"{name} defaults to {stmt.value.value!r}, a concrete "
                         "identity; anything built without one silently claims it")
            if EXPERIMENT_FIELD.search(name) and not FORMAT_FIELD.search(name):
                self.add("experiment_default", stmt, name,
                         f"{name} is a study's choice, not a generic parameter; "
                         "a default makes one experiment the silent answer")
        self.generic_visit(node)
        self._class.pop()

    @staticmethod
    def _is_required(value: ast.AST) -> bool:
        """`field(kw_only=True)` with no default is a REQUIRED field."""
        if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "field":
            return not any(k.arg == "default" or k.arg == "default_factory"
                           for k in value.keywords)
        return False

    # -- 4/6/7/8. function-level rules ---------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self._function(node)
        self.generic_visit(node)

    def _function(self, node) -> None:
        args = {a.arg for a in node.args.args + node.args.kwonlyargs}

        # 4. a concrete scoring rule inside a planning function
        if node.name.startswith("score_"):
            for c in ast.walk(node):
                if isinstance(c, ast.BoolOp) and isinstance(c.op, ast.And):
                    names = {n.id for n in ast.walk(c) if isinstance(n, ast.Name)}
                    if len(names & {"usable", "scorer_correct", "scorable",
                                    "correct"}) >= 3 and "rule" not in args:
                        self.add("scoring_semantics", c, node.name,
                                 "the correctness rule is decided here; it is a "
                                 "scientific decision about one battery and must "
                                 "be supplied by the caller")
                        break

        # 7. a stage NUMBER in a policy helper.
        #
        # Scoped to infrastructure/runtime/governance on purpose. `Stage 1` in
        # `initialization/` is this PROJECT's workflow vocabulary -- AGENTS.md
        # defines stages 0-6 and a whole module is named `stage1_import` -- so
        # flagging it would be trading a false negative for a false positive.
        # What the maintainer named is a driver's MARKER number leaking into a
        # generic marker policy, which is a different thing in a different layer.
        infra = any(part in self.path for part in
                    ("/infrastructure/", "/runtime/", "/governance/"))
        if infra and re.search(r"stage\d", node.name):
            self.add("numeric_stage_policy", node, node.name,
                     f"{node.name} names a driver's stage number; generic "
                     "infrastructure must not know one driver's numbering")
        elif infra:
            for c in ast.walk(node):
                if (isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Attribute)
                        and c.func.attr == "get" and len(c.args) == 1
                        and isinstance(c.args[0], ast.Constant)
                        and isinstance(c.args[0].value, str)
                        and c.args[0].value.isdigit()
                        and re.search(r"stage", ast.unparse(c.func.value))):
                    self.add("numeric_stage_policy", c, node.name,
                             f"reads stage {c.args[0].value!r} by number")

        # 8. an isinstance branch whose fallthrough asserts success
        if node.name.endswith("_ok") or node.name.startswith("fetch_result"):
            has_isinstance = any(
                isinstance(c, ast.Call) and getattr(c.func, "id", "") == "isinstance"
                for c in ast.walk(node))
            tail = node.body[-1]
            if has_isinstance and isinstance(tail, ast.Return) \
                    and isinstance(tail.value, ast.Constant) \
                    and tail.value.value is True:
                self.add("untyped_result_branch", tail, node.name,
                         "an object of unrecognized shape returns True; a "
                         "generic runner cannot know that an unknown object "
                         "means a completed transfer, so this fails OPEN")

    # -- 5/6. module-level reads ---------------------------------------------
    def check_module(self) -> None:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                    and n.value not in self.docs and REPO_TREE.search(n.value) \
                    and not n.value.startswith("src/"):
                self.add("core_reads_repository", n, n.value[:60],
                         "a reusable core naming a repository tree depends on "
                         "the layout it is supposed to be independent of")
        for stmt in self.tree.body:
            targets = ([stmt.target] if isinstance(stmt, ast.AnnAssign)
                       else getattr(stmt, "targets", []))
            value = getattr(stmt, "value", None)
            if not targets or not isinstance(value, ast.Call):
                continue
            fn = getattr(value.func, "id", "") or getattr(value.func, "attr", "")
            called = next((f for f in ast.walk(self.tree)
                           if isinstance(f, ast.FunctionDef) and f.name == fn), None)
            if called is None:
                continue
            body = ast.unparse(called)
            if "read_text" in body or "json.load" in body or "open(" in body:
                self.add("import_time_deployment_load", stmt, fn,
                         f"{fn}() reads a file and is called at IMPORT time, so "
                         "importing this module reads deployment configuration")


def prose_lines(source: str) -> list[tuple[int, str, str]]:
    """`(lineno, "doc"|"cmt", text)` for every docstring and comment line.

    Line-granular on purpose: a module docstring is one string, and reporting
    the whole thing would name a file rather than a sentence.
    """
    lines = source.splitlines()
    kind: dict[int, str] = {}
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)) and ast.get_docstring(n, clean=False):
            body0 = n.body[0]
            for ln in range(body0.lineno, (body0.end_lineno or body0.lineno) + 1):
                kind[ln] = "doc"
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("#"):
            kind[i] = "cmt"
    return [(ln, k, lines[ln - 1]) for ln, k in sorted(kind.items())]


def scan_prose(path: str, source: str) -> list[dict]:
    """Rule 9. One finding per matched instance value, with its line."""
    out = []
    for ln, kind, text in prose_lines(source):
        cited = PRINCIPLE_CITATION.search(text)
        for label, rx in INSTANCE_PATTERNS:
            for m in rx.finditer(text):
                # `P4` inside "AGENTS.md P4, P5" is the rule this module obeys.
                if label == "run label" and cited:
                    continue
                out.append({
                    "rule": "instance_prose", "path": path, "line": ln,
                    "name": f"{kind}:{m.group(0)}",
                    "why": (f"a {label} from this project's run history "
                            f"({m.group(0)!r}) documents reusable core with one "
                            "campaign's instance data; relocate it to "
                            "docs/core-provenance.md and keep the mechanism"),
                })
    return out


def scan_source(path: str, source: str) -> list[dict]:
    w = Walker(path, source)
    w.visit(w.tree)
    w.check_module()
    return w.findings + scan_prose(path, source)


def scan(root: Path = CORE) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*.py")):
        out.extend(scan_source(str(p.relative_to(REPO)), p.read_text()))
    return out


def partition(found: list[dict]) -> tuple[list[dict], list[dict]]:
    """`(counted, deferred)`. Deferred is the CUDA-validated surface's prose.

    Split rather than filtered: a deferred finding is still a finding, and it
    is still printed. What it is excluded from is the gate's total.
    """
    deferred = [f for f in found if f["path"] in CUDA_VALIDATED_SURFACE
                and f["rule"] == "instance_prose"]
    ids = {id(f) for f in deferred}
    return [f for f in found if id(f) not in ids], deferred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    counted, deferred = partition(scan())
    by_rule: dict[str, int] = {r: 0 for r in RULES}
    for f in counted:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
        print(f"  {f['rule']:28} {f['path']}:{f['line']}  {f['name']}")
    for f in deferred:
        print(f"  DEFERRED {f['rule']:19} {f['path']}:{f['line']}  {f['name']}")
    doc = {"schema": "aadistill.architecture_core_ownership/v2",
           "_contract": ("Ownership violations the literal and semantic gates "
                         "cannot see. Each rule is pinned by a RESTORED real "
                         "defect in tests/architecture/test_core_ownership.py."),
           "root": "src/aadistill", "by_rule": by_rule,
           "total": len(counted), "findings": counted,
           "_deferred": ("Instance prose inside the real-CUDA-validated "
                         "execution surface. Reported, not rewritten: the "
                         "review that accepted execution SHA 7027a8f4 directed "
                         "that this surface not be modified. Not an exemption "
                         "-- removing a path from CUDA_VALIDATED_SURFACE makes "
                         "its findings count again."),
           "deferred_surface": list(CUDA_VALIDATED_SURFACE),
           "deferred_total": len(deferred), "deferred_findings": deferred,
           "authorizes": "nothing"}
    print(json.dumps(by_rule, indent=1))
    print(f"total: {len(counted)}   deferred (CUDA surface): {len(deferred)}")
    if args.write:
        out = REPO / "logs/architecture_core_ownership.json"
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
