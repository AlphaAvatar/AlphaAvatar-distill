#!/usr/bin/env python3
"""Detect CURRENT-EXPERIMENT policy in `src/aadistill`, from the syntax tree.

    PYTHONPATH=src:scripts python scripts/architecture/semantic_hardcode.py

The literal detectors next door answer "is there a hash, a path, a repo id in
the core". They report zero and are still not enough: an experiment can be
hardcoded without a single literal of those kinds. `SEED_SA = 20260726` is an
int. `expected=("gsm8k", "math_verified", ...)` is a tuple of ordinary words.
`catastrophic: Rule = CATASTROPHIC_V1` is a dataclass default. Each of those was
in the core, and each is one study's decision.

So these rules look at SHAPE and ROLE, not at spelling:

* a **seed-like** binding -- a name that says seed, holding a concrete int;
* a **phase/attempt/C1 policy** property or class, by what it decides;
* a **concrete capability tuple** used as a value -- short lowercase task names
  in a field or default that a schema will range over;
* **hardware SKU, VRAM or price** policy, by shape (`L40S`, `40 GiB`, `$1.09/h`);
* a **current battery / replay / recovery identity** named as a value;
* a **dataclass default that instantiates policy** -- the specific defect that
  let `SuccessiveHalvingPlan()` silently be this study;
* a **source-file list owned by a current experiment**.

What they must NOT do is fire on generic algorithm parameters. `z = 1.96` is a
normal quantile. `chunk = 512` is a batch size. `version: int = 1` is a schema
version. `temperature` is a distillation knob. Every rule below is written to
separate those from an instance of the current experiment, and
`tests/architecture/test_semantic_hardcode.py` pins both directions -- a rule
that only ever fires is as useless as one that never does.

Comments are absent from the AST by construction and docstrings are excluded
explicitly, so no finding here can come from prose.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = REPO_ROOT / "src" / "aadistill"
SCHEMA = "aadistill.semantic_hardcode/v1"

#: A name that means "seed". `seed` alone is too broad -- `seed_everything` is a
#: function, `seeded` is an adjective -- so this matches the whole identifier or
#: a clear suffix/prefix boundary.
SEED_NAME = re.compile(r"(^|_)SEEDS?($|_)", re.I)

#: A concrete seed VALUE. The seeds in this project are dates-as-ints
#: (20260726); a generic default of 0, 1 or 42 is a placeholder, not a study.
def _seedish(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 1000


#: Names that decide something about a named phase, attempt or C1 stage.
#: Both spellings: `phase_a_authorized` and `PhaseAAuthorization`. The class
#: names in this repository are CamelCase without separators, so a rule that
#: only understood snake_case would have missed every one of them.
PHASE_POLICY = re.compile(
    r"(^|_)(phase_[a-z0-9]+|attempt_?\d*|c1|c0|stage_[a-z0-9]+|rung\d*)($|_)", re.I)
PHASE_CAMEL = re.compile(r"(^|[a-z0-9])(Phase[A-Z0-9]|Attempt\d|C[01][A-Z_]|Rung\d)")


def _phasey(name: str) -> bool:
    return bool(PHASE_POLICY.search(name) or PHASE_CAMEL.search(name))

#: A hardware SKU, a VRAM quantity, or a price. Shapes, not a vendor list.
SKU = re.compile(r"\b([ARLHVT]\d{2,3}[SXL]?|H\d{3}|MI\d{3})\b")
VRAM = re.compile(r"\b\d{1,3}\s?(GiB|GB)\b", re.I)
PRICE = re.compile(r"\$\s?\d+\.\d{2,4}|per_hour|price_per_hour|usd_per_hour", re.I)

#: A short lowercase task name, of the kind a capability schema ranges over.
TASKY = re.compile(r"^[a-z][a-z0-9]{1,14}(_[a-z0-9]{1,14})*$")

#: A tuple of task names is only a CAPABILITY tuple when the binding says so.
#: `COMPONENTS`, `SPLITS`, `GATES` and `BEHAVIOR_GROUPS` are the mechanism's own
#: vocabulary -- what a metric is made of -- and naming them is what those
#: modules are for. `E8A_DOMAINS` and `expected=` are a study's task list.
CAPABILITY_NAME = re.compile(
    r"(^|_)(capabilit\w*|domains?|tasks?|expected|batter(y|ies)|subsets?)($|_)", re.I)

#: Names whose VALUE would be one experiment's identity.
IDENTITY_NAME = re.compile(
    r"(^|_)(battery|replay|recovery|probe|checkpoint|artifact|control)"
    r".*(_id|_ids|_sha256|_digest|_asset|_path|_paths|_v\d+)($|_)", re.I)

#: Generic knobs that are algorithm parameters, never a current experiment.
GENERIC_NAMES = {
    "version", "schema_version", "set_version", "layout_version", "z",
    "temperature", "chunk", "chunk_size", "batch_size", "seq_len", "timeout",
    "max_retries", "poll_seconds", "epsilon", "eps", "atol", "rtol", "seed_offset",
}

#: A name that ends in one of these is a TYPE or a mechanism, not an instance.
MECHANISM_SUFFIX = ("Error", "Rule", "Schema", "Policy", "Plan", "Spec",
                    "Protocol", "Registry", "Adapter", "Aggregation")


def _name_of(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.arg):
        return target.arg
    return None


def _const_tuple(node: ast.AST) -> list[object] | None:
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts and all(
            isinstance(e, ast.Constant) for e in node.elts):
        return [e.value for e in node.elts]
    return None


class SemanticWalker(ast.NodeVisitor):
    """Every finding carries the binding it came from, and why it fired."""

    def __init__(self, rel: str, src: str) -> None:
        self.rel = rel
        self.stack: list[str] = []
        self.findings: list[dict] = []
        self._prose: set[int] = set()
        tree = ast.parse(src)
        self._collect_prose(tree)
        #: Module-level CONSTANT -> what builds it. A default naming a constant
        #: is only a policy instance when the constant is CONSTRUCTED from a
        #: mechanism type. `MAIN_RELAY = _main_relay()` reads the deployment's
        #: configuration; `CATASTROPHIC_V1 = CatastrophicCapabilityRule(...)` is
        #: an instance of the current study, and only the second is the defect.
        self._const_builder: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                built = getattr(fn, "id", getattr(fn, "attr", ""))
                for tgt in node.targets:
                    nm = _name_of(tgt)
                    if nm:
                        self._const_builder[nm] = built

    def _collect_prose(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                self._prose.add(id(node.value))

    def _report(self, rule: str, node: ast.AST, name: str, value: object,
                why: str) -> None:
        self.findings.append({
            "path": self.rel, "rule": rule, "owner": ".".join(self.stack) or "<module>",
            "name": name, "line": getattr(node, "lineno", 0),
            "value": str(value)[:120], "why": why,
        })

    # -- scope ------------------------------------------------------------
    def _scoped(self, node, kind: str):
        self.stack.append(node.name)
        if kind == "class":
            self._class_body(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node): self._scoped(node, "function")
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for dec in node.decorator_list:
            nm = getattr(dec, "id", getattr(getattr(dec, "func", None), "id", ""))
            if nm == "dataclass" or (isinstance(dec, ast.Attribute)
                                     and dec.attr == "dataclass"):
                self._dataclass_defaults(node)
        if _phasey(node.name):
            self._report("phase_policy_class", node, node.name, node.name,
                         "a class named for one phase, attempt or C1 stage")
        self._scoped(node, "class")

    def _class_body(self, node) -> None:
        """Properties that decide a named phase's permission."""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_prop = any(getattr(d, "id", "") == "property"
                              for d in item.decorator_list)
                #: A permission decision, not a derived count. `rung1_probes`
                #: computes how many probes a rung has; `allows_phase_a` decides
                #: whether a phase may run, and that is the study's call.
                permissiony = re.match(
                    r"(allows?|automatic|refuse|permits?|can|may)_", item.name) \
                    or item.name.endswith(("_authorized", "_permitted",
                                           "_reachable", "_allowed"))
                if is_prop and permissiony and PHASE_POLICY.search(item.name):
                    self._report("phase_policy_property", item, item.name,
                                 item.name,
                                 "a property whose name decides something about "
                                 "one phase, attempt or C1 stage")

    def _dataclass_defaults(self, node) -> None:
        """A default that INSTANTIATES policy, rather than naming a type.

        `catastrophic: Rule = CATASTROPHIC_V1` and
        `aggregation: Agg = POOLED_COUNTS_V2` are the shape: a field whose
        default is a concrete instance of the current study. A default of
        `None`, `()`, a literal, or `field(default_factory=list)` is not.
        """
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or item.value is None:
                continue
            name = _name_of(item.target) or ""
            default = item.value
            #: A DEFAULT_/DENY_/NULL_ sentinel is the safe empty value, and a
            #: *_VERSION / *_ID is a schema identity -- neither is an instance of
            #: the current experiment, which is what this rule is for.
            SENTINEL = ("DEFAULT_", "DENY_", "NULL_", "NONE_", "EMPTY_", "NO_")
            IDENTITYISH = ("_VERSION", "_ID", "_SCHEMA")
            builder = self._const_builder.get(
                default.id if isinstance(default, ast.Name) else "", "")
            if (isinstance(default, ast.Name) and default.id.isupper()
                    and not default.id.startswith(SENTINEL)
                    and not default.id.endswith(IDENTITYISH)
                    and builder.endswith(MECHANISM_SUFFIX)):
                self._report("dataclass_default_instantiates_policy", item, name,
                             default.id,
                             "a field defaulting to a concrete instance, so an "
                             "object built without arguments becomes it")
            elif isinstance(default, ast.Call):
                fn = default.func
                nm = getattr(fn, "id", getattr(fn, "attr", ""))
                if nm.endswith(MECHANISM_SUFFIX) and default.args or (
                        nm.endswith(MECHANISM_SUFFIX) and default.keywords):
                    self._report("dataclass_default_instantiates_policy", item,
                                 name, nm,
                                 "a field defaulting to a constructed policy "
                                 "object with concrete arguments")

    # -- assignments -------------------------------------------------------
    def visit_AnnAssign(self, node):
        self._assignment(node, _name_of(node.target), node.value)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for t in node.targets:
            self._assignment(node, _name_of(t), node.value)
        self.generic_visit(node)

    def _assignment(self, node, name: str | None, value: ast.AST | None) -> None:
        if not name or value is None or name in GENERIC_NAMES:
            return
        const = value.value if isinstance(value, ast.Constant) else None

        if SEED_NAME.search(name) and _seedish(const):
            self._report("concrete_seed", node, name, const,
                         "a seed-named binding holding a concrete value; a "
                         "study's seed is the study's, not the mechanism's")

        if IDENTITY_NAME.search(name) and isinstance(const, str) and const:
            self._report("current_experiment_identity", node, name, const,
                         "a name that means 'which battery/replay/recovery "
                         "artifact', holding one")

        items = _const_tuple(value)
        if (items and len(items) >= 3 and CAPABILITY_NAME.search(name)
                and all(isinstance(i, str) and TASKY.match(i) for i in items)):
            self._report("concrete_capability_tuple", node, name, items,
                         "a tuple of short task names used as a value; which "
                         "capabilities exist is a study's decision")

        for text in ([const] if isinstance(const, str) else []) + (
                [i for i in (items or []) if isinstance(i, str)]):
            if SKU.search(text) or VRAM.search(text):
                self._report("hardware_policy", node, name, text,
                             "a GPU SKU or VRAM quantity; which hardware a run "
                             "needs is the run's fact")
            if PRICE.search(text):
                self._report("price_policy", node, name, text,
                             "a price or per-hour rate")

        #: `attempt = 0`, `rung = 1`, `stage = 3` are indices and counters, and
        #: every loop over stages has one. A phase-named binding is POLICY when
        #: it carries a study-sized value: a string, or an int too large to be
        #: an ordinal.
        if PHASE_POLICY.search(name) and const is not None and (
                isinstance(const, str) and const
                or (isinstance(const, int) and not isinstance(const, bool)
                    and abs(const) > 1000)):
            self._report("phase_policy_constant", node, name, const,
                         "a constant named for one phase, attempt or C1 stage, "
                         "holding a study-sized value rather than an ordinal")

    # -- keyword defaults in signatures ------------------------------------
    def visit_arguments(self, node):
        for arg, default in zip(node.args[len(node.args) - len(node.defaults):],
                                node.defaults):
            if isinstance(default, ast.Constant):
                self._assignment(default, arg.arg, default)
        for arg, default in zip(node.kwonlyargs, node.kw_defaults):
            if isinstance(default, ast.Constant):
                self._assignment(default, arg.arg, default)
        self.generic_visit(node)


def scan(root: Path = CORE) -> list[dict]:
    out: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        src = path.read_text()
        w = SemanticWalker(rel, src)
        w.visit(ast.parse(src, filename=rel))
        out.extend(w.findings)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    findings = scan()
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    doc = {
        "schema": SCHEMA,
        "_contract": (
            "CURRENT-EXPERIMENT policy found in src/aadistill by SHAPE and ROLE "
            "rather than by keyword. Reports ownership and decides nothing; "
            "tests/architecture/test_semantic_hardcode.py turns it into a gate "
            "and pins both directions. AUTHORIZES NOTHING."),
        "root": "src/aadistill",
        "counts": by_rule,
        "total": len(findings),
        "findings": findings,
        "authorizes": "nothing",
    }
    print(json.dumps({"total": doc["total"], **by_rule}, indent=1))
    for f in findings[:20]:
        print(f"  {f['rule']:38} {f['path'][14:]:34} {f['name']}")
    if args.write:
        (REPO_ROOT / "logs/architecture_semantic_hardcode.json").write_text(
            json.dumps(doc, indent=1) + "\n")
        print("wrote logs/architecture_semantic_hardcode.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
