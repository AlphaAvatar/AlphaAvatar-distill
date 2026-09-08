#!/usr/bin/env python3
"""What is in `src/aadistill`, and which of it is generic. AST, not grep.

    PYTHONPATH=src python scripts/architecture/inventory.py --write

The maintainer's rule is that the algorithm core must be reusable and that
experiment and model instances live in configuration and scripts. Deciding
whether the current tree obeys that needs a mechanical answer, not a reading:
`src/aadistill` is ~34k lines over 60+ modules, and the words that would
identify a violation — "phase_c1", "attempt", a repo id, a seed — also appear in
the prose explaining why something is generic.

So every judgement here comes from the syntax tree:

* **string and number literals** are collected with their *owner* — the module,
  class and function they are assigned in, and whether they are a docstring or a
  comment (comments are not in the AST at all, which is the point);
* **import edges** come from `ast.Import` / `ast.ImportFrom`, not from a regex,
  so `# imports aadistill.autoinit` in a docstring cannot register as an edge;
* **attribute chains** like `model.model.layers` are matched as attribute
  nodes, so `"model.model.layers"` inside a string — which is how the adapters
  legitimately *describe* what they encapsulate — is not a violation;
* **path literals** are recognised by shape (a string containing `logs/`,
  `configs/`, `artifacts/`, `~`, or an absolute prefix) and reported with the
  binding they belong to.

The output is `logs/architecture_inventory.json` and is consumed by
`tests/architecture/`, which turns the findings into binding gates. Nothing here
decides policy; it reports ownership, and the tests decide.
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
OUT = "logs/architecture_inventory.json"
SCHEMA = "aadistill.architecture_inventory/v1"

# --- what counts as an instance literal -------------------------------------
#
# Each pattern is a CLAIM about a shape, not about a word. `sha256`-shaped hex
# is 64 hex chars; a HF repo id is `owner/name` with no spaces and no path
# separator beyond the first; a seed is a bare int over a threshold in an
# assignment. None of these can be satisfied by prose, which is why they are
# shapes rather than keywords.

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
#: A Hugging Face repo id: `owner/name`, no further separators, not a path
#: fragment and not a schema string. Without the exclusions this matched
#: `logs/runs`, `governance/grant.json` and `aadistill.x/v1` — every one a
#: false positive, and a gate built on those would be noise.
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Shapes that LOOK like `owner/name` and are not a model repo id. Without these
#: the rule reported MIME types, a TCP port spec and a slash-joined label -- and
#: a gate whose findings are mostly noise stops being read, which is worse than
#: no gate. Every exclusion is a form, not a specific string: `application/json`
#: is excluded because it is a MIME type, not because it is that MIME type.
NOT_REPO_ID = re.compile(
    r"(^(logs|configs|artifacts|data|scripts|src|tests|docs|governance|runtime"
    r"|evidence|closeout|stage[0-9])/)|(\.(json|jsonl|py|md|sh|txt|yaml|yml|"
    r"safetensors|bin|log)$)|(/v[0-9]+$)"
    #: MIME types: a registered IANA top-level type followed by a subtype.
    r"|(^(application|text|image|audio|video|multipart|message|model|font)/)"
    #: A port specification, as Docker and RunPod write them: `22/tcp`.
    r"|(^[0-9]+/(tcp|udp)$)"
    #: An alternation label -- `E6/E6b`, `sa/sb` -- where both sides share a
    #: prefix or are single tokens of the same short shape. A model repo id
    #: names an owner and a model, which do not look like that.
    r"|(^[A-Za-z][A-Za-z0-9]{0,4}/[A-Za-z][A-Za-z0-9]{0,4}$)")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
#: A path a core module should not be naming: run evidence, artifact storage,
#: someone's home directory, a temp location. `configs/` is deliberately NOT
#: here -- it is where the maintainer's rules say experiment data belongs, so a
#: core module reading a declared configuration path is the intended design
#: rather than a violation of it. `scripts/` and `data/` stay, because a core
#: module naming an executable or a dataset location is reaching outward.
PATHY = re.compile(r"(^|[\"' ])(logs/|artifacts/|data/|scripts/|~/|/home/|/tmp/)")

#: Model-family attribute chains. A core module reaching through these is
#: bypassing the adapter; an adapter doing it is the adapter's whole job.
#: Registrations that declare framework vocabulary rather than a concrete
#: instance. Kept in step with tests/architecture/test_core_boundaries.py.
TAXONOMY_REGISTRATIONS = frozenset({"register_kind"})

FAMILY_ATTRS = {
    ("model", "layers"), ("self_attn",), ("o_proj",), ("q_proj",), ("k_proj",),
    ("v_proj",), ("mlp",), ("gate_proj",), ("up_proj",), ("down_proj",),
    ("embed_tokens",), ("input_layernorm",), ("post_attention_layernorm",),
}

#: Vocabulary that names a particular experiment rather than a mechanism.
EXPERIMENT_TOKENS = ("phase_a", "phase_b", "phase_c", "c1_", "_c1", "attempt",
                     "continuation", "rung", "e8", "e6b", "e1_kd")

CLASSES = (
    "generic_architecture_spec", "model_family_adapter", "generic_statistics",
    "generic_operator", "generic_planning_execution",
    "generic_governance_runtime", "model_instance_recipe",
    "experiment_instance", "historical_compatibility",
)


def module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT / "src").with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class Walker(ast.NodeVisitor):
    """Collects owned facts. Every record carries the binding it came from."""

    def __init__(self, mod: str, src: str):
        self.mod = mod
        self.lines = src.splitlines()
        self.stack: list[str] = []
        self.imports: list[dict] = []
        self.literals: list[dict] = []
        self.attr_chains: list[dict] = []
        self.defs: list[dict] = []
        self.import_time_calls: list[dict] = []
        self._docstrings: set[int] = set()
        self._attr_seen: dict[int, dict] = {}

    # -- scope tracking -----------------------------------------------------
    def _scoped(self, node, kind):
        self.defs.append({"name": node.name, "kind": kind,
                          "qualname": ".".join([*self.stack, node.name]),
                          "line": node.lineno})
        doc = ast.get_docstring(node, clean=False)
        if doc is not None and node.body and isinstance(node.body[0], ast.Expr):
            self._docstrings.add(id(node.body[0].value))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node): self._scoped(node, "function")
    def visit_AsyncFunctionDef(self, node): self._scoped(node, "function")
    def visit_ClassDef(self, node): self._scoped(node, "class")

    # -- imports ------------------------------------------------------------
    def visit_Import(self, node):
        for a in node.names:
            self.imports.append({"target": a.name, "line": node.lineno,
                                 "relative": 0})
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        target = node.module or ""
        if node.level:
            base = self.mod.rsplit(".", node.level) if "." in self.mod else [self.mod]
            prefix = ".".join(self.mod.split(".")[:-node.level])
            target = f"{prefix}.{target}" if target else prefix
        self.imports.append({"target": target, "line": node.lineno,
                             "relative": node.level or 0,
                             "names": [a.name for a in node.names]})
        self.generic_visit(node)

    # -- literals, with their owner ----------------------------------------
    def visit_Constant(self, node):
        if id(node) in self._docstrings:
            return                      # a docstring is prose, not a value
        owner = ".".join(self.stack) or "<module>"
        if isinstance(node.value, str):
            v = node.value
            if len(v) > 400:
                return
            kinds = []
            if HEX64.match(v): kinds.append("sha256")
            elif HEX16.match(v): kinds.append("hex16")
            if GIT_SHA.match(v): kinds.append("git_sha")
            if REPO_ID.match(v) and not NOT_REPO_ID.search(v):
                kinds.append("repo_id_shape")
            if PATHY.search(v): kinds.append("path")
            low = v.lower()
            hits = [t for t in EXPERIMENT_TOKENS if t in low]
            if hits: kinds.append("experiment_token")
            if kinds:
                self.literals.append({"value": v[:200], "kinds": kinds,
                                      "owner": owner, "line": node.lineno,
                                      "experiment_tokens": hits})
        elif isinstance(node.value, bool):
            pass
        elif isinstance(node.value, int) and node.value > 100000:
            self.literals.append({"value": node.value, "kinds": ["big_int"],
                                  "owner": owner, "line": node.lineno,
                                  "experiment_tokens": []})
        elif isinstance(node.value, float) and 0.01 < node.value < 10000:
            self.literals.append({"value": node.value, "kinds": ["float"],
                                  "owner": owner, "line": node.lineno,
                                  "experiment_tokens": []})
        self.generic_visit(node)

    # -- family attribute access -------------------------------------------
    def visit_Attribute(self, node):
        chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        chain.reverse()
        #: One record per access site, not one per matching probe. A chain like
        #: `self_attn.q_proj.weight` matches two probes and contains three
        #: nested Attribute nodes, so the naive form counted a single access up
        #: to six times and inflated the finding count ~4x.
        hit = next((".".join(probe) for probe in FAMILY_ATTRS
                    if any(tuple(chain[i:i + len(probe)]) == probe
                           for i in range(len(chain) - len(probe) + 1))), None)
        if hit is not None:
            key = (node.lineno, node.col_offset)
            prev = self._attr_seen.get(node.lineno)
            text = ".".join(chain)
            if prev is None or len(text) > len(prev["chain"]):
                rec = {"chain": text, "probe": hit,
                       "owner": ".".join(self.stack) or "<module>",
                       "line": node.lineno}
                self._attr_seen[node.lineno] = rec
        self.generic_visit(node)


def classify(mod: str, rel: str, facts: dict) -> str:
    """One label per module, from its path and what it owns."""
    if "/adapters/" in rel:
        return "model_family_adapter"
    if "/operators/" in rel:
        return "generic_operator"
    name = rel.rsplit("/", 1)[-1]
    low = name.lower()
    if any(t in low for t in ("phase_a", "phase_b", "c1_", "continuation",
                              "recovery_continuation", "measurement")):
        return "experiment_instance"
    if low in ("arch.py", "spec.py"):
        return "generic_architecture_spec"
    if low in ("collect.py", "attention_stats.py", "stats.py", "contribution.py",
               "project.py", "sandwich.py", "reweight.py"):
        return "generic_statistics"
    if low in ("search.py", "fixed_path.py", "ranking.py", "state.py",
               "run_layout.py", "planner.py"):
        return "generic_planning_execution"
    if low in ("authorization.py", "artifact.py", "manifest.py", "cost.py",
               "post_freeze.py", "pod_environment.py", "device.py",
               "device_handoff.py", "cpu_test_env.py", "leaf_durability.py"):
        return "generic_governance_runtime"
    return "generic_planning_execution"


def scan() -> dict:
    modules = {}
    for path in sorted(CORE.rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        src = path.read_text()
        tree = ast.parse(src, filename=rel)
        w = Walker(module_name(path), src)
        doc = ast.get_docstring(tree, clean=False)
        if doc is not None and tree.body and isinstance(tree.body[0], ast.Expr):
            w._docstrings.add(id(tree.body[0].value))
        w.visit(tree)

        # import-time calls at module scope (registration side effects)
        import_time = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                fn = node.value.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                import_time.append({"call": nm, "line": node.lineno})
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if "register" in nm:
                    import_time.append({"call": nm, "line": node.lineno})

        facts = {
            "module": w.mod,
            "path": rel,
            "lines": len(src.splitlines()),
            "imports": w.imports,
            "internal_imports": sorted({i["target"] for i in w.imports
                                        if i["target"].startswith("aadistill")}),
            "literals": w.literals,
            "family_attribute_access": sorted(w._attr_seen.values(),
                                              key=lambda r: r["line"]),
            "definitions": w.defs,
            "import_time_calls": import_time,
        }
        facts["classification"] = classify(w.mod, rel, facts)
        modules[rel] = facts
    return modules


def edges(modules: dict) -> dict:
    """Import graph over the core, plus the cycles in it."""
    by_mod = {m["module"]: rel for rel, m in modules.items()}
    out = {}
    for rel, m in modules.items():
        targets = set()
        for t in m["internal_imports"]:
            # attribute imports resolve to the longest known module prefix
            parts = t.split(".")
            while parts:
                cand = ".".join(parts)
                if cand in by_mod and cand != m["module"]:
                    targets.add(cand)
                    break
                parts.pop()
        out[m["module"]] = sorted(targets)

    # package-level edges, to find reverse/cyclic dependencies
    def pkg(mod: str) -> str:
        p = mod.split(".")
        return ".".join(p[:3]) if len(p) > 2 else mod

    pkg_edges: dict[str, set[str]] = {}
    for src_mod, dsts in out.items():
        a = pkg(src_mod)
        for d in dsts:
            b = pkg(d)
            if a != b:
                pkg_edges.setdefault(a, set()).add(b)
    cycles = sorted({tuple(sorted((a, b))) for a, bs in pkg_edges.items()
                     for b in bs if a in pkg_edges.get(b, set())})
    return {"module_edges": out,
            "package_edges": {k: sorted(v) for k, v in pkg_edges.items()},
            "package_cycles": [list(c) for c in cycles]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    modules = scan()
    graph = edges(modules)

    findings = {
        "sha256_literals": [], "repo_id_literals": [], "path_literals": [],
        "experiment_named_modules": [], "family_access_outside_adapters": [],
        "import_time_registration": [], "import_time_taxonomy": [],
        "big_int_literals": [],
    }
    for rel, m in modules.items():
        is_adapter = m["classification"] == "model_family_adapter"
        for lit in m["literals"]:
            entry = {"path": rel, "owner": lit["owner"], "line": lit["line"],
                     "value": lit["value"]}
            if "sha256" in lit["kinds"] or "git_sha" in lit["kinds"]:
                findings["sha256_literals"].append(entry)
            if "repo_id_shape" in lit["kinds"]:
                findings["repo_id_literals"].append(entry)
            if "path" in lit["kinds"]:
                findings["path_literals"].append(entry)
            if "big_int" in lit["kinds"]:
                findings["big_int_literals"].append(entry)
        if m["classification"] == "experiment_instance":
            findings["experiment_named_modules"].append(rel)
        if not is_adapter and m["family_attribute_access"]:
            for a in m["family_attribute_access"]:
                findings["family_access_outside_adapters"].append(
                    {"path": rel, **a})
        for c in m["import_time_calls"]:
            if c["call"] in TAXONOMY_REGISTRATIONS:
                #: Framework VOCABULARY, reported separately. Registering an
                #: operator kind is not an instance registering itself: there is
                #: nothing for a caller to inject, and counting it alongside the
                #: real thing made the two rules disagree about the same tree.
                findings["import_time_taxonomy"].append({"path": rel, **c})
            elif "register" in c["call"]:
                findings["import_time_registration"].append({"path": rel, **c})

    doc = {
        "schema": SCHEMA,
        "_contract": ("A mechanical inventory of src/aadistill, derived from the "
                      "syntax tree. Docstrings are excluded from literal "
                      "collection and comments are not in the AST at all, so no "
                      "finding here can be produced by prose. Reports ownership; "
                      "decides no policy. AUTHORIZES NOTHING."),
        "root": "src/aadistill",
        "counts": {
            "modules": len(modules),
            "lines": sum(m["lines"] for m in modules.values()),
            "classifications": {c: sum(1 for m in modules.values()
                                       if m["classification"] == c)
                                for c in CLASSES},
            **{k: len(v) for k, v in findings.items()},
            "package_cycles": len(graph["package_cycles"]),
        },
        "findings": findings,
        "graph": graph,
        "modules": modules,
        "authorizes": "nothing",
    }
    if args.write:
        (REPO_ROOT / OUT).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT}")
    print(json.dumps(doc["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
