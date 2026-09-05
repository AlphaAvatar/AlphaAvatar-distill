#!/usr/bin/env python
"""Every predicate in the C1-selected suite that can decide differently on a pod.

C1 attempt 5 died for `$0.3150` because two tests were guarded by
`skipif(AAD_SYNTHETIC_HF_TOKEN)` — the SIMULATOR's flag. The pod is
behaviourally identical to the simulation while carrying none of its markers, so
the guard skipped where the assertion held and ran where it could not. That is a
CLASS, not an incident: any predicate whose premise differs between the dev box
and the pod can decide differently there, and the sweep cannot see it.

This walks the AST of every module the C1 session actually selects, finds every
skip construct, and classifies the signal its condition reads. Anything that can
see a different premise on a pod must carry an explicit classification; anything
unclassified is reported and fails the audit.

    PYTHONPATH=src python scripts/autoinit/audit_skip_predicates.py [--write]

`--write` refreshes `logs/c1_skip_predicate_audit.json`. Entirely at `$0`: it
reads source and never imports the modules under audit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests/pod"))

RECORD = "logs/c1_skip_predicate_audit.json"
REGISTRY = "configs/autoinit/c1_skip_predicate_classification.json"
SCHEMA = "aadistill.autoinit.c1_skip_predicate_audit/v1"

#: What a condition can read, and whether a pod sees the same thing.
#:
#: `same_on_pod` means the premise travels in the bundle or is derived from the
#: manifest, so the dev box, the simulation and the pod agree by construction.
#: `differs_on_pod` means the answer depends on the machine, and the sweep's
#: verdict is therefore not the pod's.
SIGNALS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "simulator_marker": (
        ("AAD_SYNTHETIC_HF_TOKEN", "PODSIM_"),
        "differs_on_pod",
        "THE ATTEMPT-5 DEFECT. Set by simulate_pod_env.sh and by nothing else, "
        "so it is false on the pod even when the condition it stands for holds. "
        "Never legitimate as a staging or artifact premise."),
    "unstaged_artifact": (
        ("aad-artifacts", "aad-scratch", "artifacts/stage3/corpus_v2",
         "artifacts/eval/battery_v2", "artifacts/stage0", "artifacts/stage2"),
        "differs_on_pod",
        "an artifact path C1's manifest does not stage; present on the dev box "
        "and absent on the pod"),
    "home_directory": (
        ("Path.home()", 'environ["HOME"]', 'environ.get("HOME"'),
        "differs_on_pod",
        "the pod's $HOME is empty; the dev box's holds caches and credentials"),
    "absolute_devbox_path": (
        ("/home/ecs-user",),
        "differs_on_pod",
        "an absolute dev-box path, which does not exist on a pod"),
    "gpu": (
        ("cuda.is_available", "nvidia-smi", "device_count", "torch.cuda"),
        "differs_on_pod",
        "the pod has an L40S and the dev box has none, so this decides the "
        "OPPOSITE way there — it skips here and runs on the pod"),
    "credential": (
        ("HF_TOKEN", "hf_token", "api_key", "API_KEY", "token()"),
        "differs_on_pod",
        "the pod exports a real token; the simulation injects a synthetic one "
        "and the bare dev box may have neither"),
    "optional_dependency": (
        ("importorskip", "shutil.which", "ImportError", "ModuleNotFoundError",
         "find_spec"),
        "differs_on_pod",
        "the pod's image is not the dev box's venv"),
    "network": (
        ("snapshot_download", "hf_hub_download", "requests.", "urlopen",
         "HfApi("),
        "differs_on_pod",
        "reachability and cache state differ per machine"),
    "filesystem_premise": (
        (".exists()", ".is_dir()", ".is_file()", ".glob(", ".rglob(",
         "os.path.exists", "listdir"),
        "differs_on_pod",
        "an existence check against a path this audit could not resolve to "
        "staged or tracked content; a pod holds only what the manifest stages"),
    "staged_artifact": (
        ("artifacts/stage1/qwen3_0p6b_init_v0", "c1_confirmation_v1",
         "artifacts/stage3/c1"),
        "same_on_pod",
        "C1's manifest stages exactly this, so the premise travels"),
    "repository_content": (
        ("logs/", "configs/", "src/", "scripts/", "tests/", "AGENTS.md",
         "README.md", "pyproject.toml"),
        "same_on_pod",
        "tracked content the bundle checkout carries"),
}

#: Signals whose presence in a condition is decided BEFORE the weaker ones.
#: Order matters: a condition naming both a simulator marker and a repo path is
#: a simulator-marker predicate.
PRECEDENCE = ("simulator_marker", "unstaged_artifact", "absolute_devbox_path",
              "home_directory", "gpu", "credential", "optional_dependency",
              "network", "staged_artifact", "repository_content",
              "filesystem_premise")


def c1_selected_modules(repo: Path) -> tuple[list[Path], list[str]]:
    """The modules the C1 session actually runs, from its own manifest."""
    from session_specs import load_session_launcher, session_args
    mod = load_session_launcher("autoinit_c1_launch")
    ignores = list(mod.spec(session_args(mod)).setup.test_ignores)
    files = sorted(p for p in (repo / "tests").rglob("test_*.py")
                   if str(p.relative_to(repo)) not in ignores)
    return files, ignores


def _seg(src: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(src, node) or ""
    except Exception:                                     # pragma: no cover
        return ""


def _enclosing(tree: ast.AST, node: ast.AST) -> str:
    """The test function a construct sits in, or '<module>' for a decorator."""
    best = "<module>"
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(fn, "end_lineno", fn.lineno)
            if fn.lineno <= node.lineno <= end:
                best = fn.name
    return best


def _module_constants(tree: ast.AST, src: str) -> dict[str, str]:
    """`NAME = <expr>` at module level, as source text.

    Most conditions are a NAME, not a literal — `if not BATTERY.is_dir()`.
    Classifying the condition text alone therefore learns nothing, which is why
    a first draft of this audit reported 82 of 95 predicates unrecognised.
    """
    out: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and node.value is not None:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = _seg(src, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            out[node.target.id] = _seg(src, node.value)
    return out


def _local_assignments(tree: ast.AST, src: str, line: int) -> dict[str, str]:
    """`name = <expr>` inside the function that owns `line`.

    Most existence checks bind a local first — `path = REPO / pe.RECORD_PATH` —
    so without this the audit sees `not path.is_file()` and learns nothing about
    which path.
    """
    out: dict[str, str] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (fn.lineno <= line <= getattr(fn, "end_lineno", fn.lineno)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and node.value is not None:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        out[tgt.id] = _seg(src, node.value)
    return out


def known_attributes() -> dict[str, str]:
    """Module constants named through an alias, read from the real module.

    Derived, never transcribed: if `RECORD_PATH` moves, this moves with it.
    """
    from aadistill.autoinit import pod_environment as pe
    # Quoted, because the expansion is read back by a literal-path regex: an
    # unquoted value expands and then resolves to nothing.
    return {k: f'"{v}"' for k, v in (
        ("pe.RECORD_PATH", pe.RECORD_PATH), ("RECORD_PATH", pe.RECORD_PATH),
        ("A.RECORD", RECORD), ("A.REGISTRY", REGISTRY))}


def _expand(condition: str, consts: dict[str, str], depth: int = 3) -> str:
    """Inline module constants the condition names, so the signal is visible."""
    text = condition
    for _ in range(depth):
        names = {n for n in consts if n in text}
        added = "".join(f" | {n}={consts[n]}" for n in sorted(names)
                        if f"{n}={consts[n]}" not in text)
        if not added:
            break
        text += added
    return text


PATH_LITERAL = re.compile(r"""["']([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+)["']""")


def repo_inventory(repo: Path) -> tuple[set[str], set[str]]:
    """(tracked, staged): what the bundle carries, and what setup stages.

    The authoritative answer to "does a pod hold this path?", replacing a
    keyword guess with the two mechanisms that actually put files on a pod.
    """
    from aadistill.autoinit import staging_contract as sc
    from session_specs import load_session_launcher, session_args
    out = subprocess.run(["git", "-C", str(repo), "ls-files"],
                         capture_output=True, text=True, check=True)
    tracked = {p for p in out.stdout.split("\n") if p.strip()}
    mod = load_session_launcher("autoinit_c1_launch")
    contract = sc.derive_contract(mod.spec(session_args(mod)).setup,
                                  session_id="autoinit-c1")
    staged = set(sc.staged_files(contract, repo))
    staged |= {a["staged_tree"] for a in contract["local_assets"]}
    staged |= {r["dest"] for r in contract["relay_inputs"] if r.get("dest")}
    return tracked, staged


def _reaches_pod(path: str, tracked: set[str], staged: set[str]) -> bool:
    """Is this repo-relative path on a pod — tracked, or staged by the manifest?"""
    if path in tracked or path in staged:
        return True
    pre = path.rstrip("/") + "/"
    return (any(t.startswith(pre) for t in tracked)
            or any(s.startswith(pre) or pre.startswith(s.rstrip("/") + "/")
                   for s in staged))


def classify(condition: str, tracked: set[str] | None = None,
             staged: set[str] | None = None) -> tuple[str, str, str]:
    """(signal, verdict, why) for one predicate's condition source.

    Environment signals win first — a credential or a GPU check is about the
    machine whatever paths it also names. Only then are path premises RESOLVED
    against the two mechanisms that put a file on a pod.
    """
    for name in PRECEDENCE:
        if name in ("filesystem_premise", "staged_artifact", "repository_content"):
            continue
        needles, verdict, why = SIGNALS[name]
        if any(n in condition for n in needles):
            return name, verdict, why

    if any(n in condition for n in SIGNALS["filesystem_premise"][0]):
        if tracked is None:
            return SIGNALS["filesystem_premise"][0][0], "differs_on_pod", ""
        paths = set(PATH_LITERAL.findall(condition))
        if not paths:
            return ("filesystem_premise_unresolved", "unknown",
                    "an existence check whose path this audit could not read "
                    "from source; a human must say what it inspects")
        missing = sorted(p for p in paths if not _reaches_pod(p, tracked, staged))
        if missing:
            return ("filesystem_premise_offpod", "differs_on_pod",
                    "inspects " + ", ".join(missing[:4])
                    + " — neither tracked nor staged, so absent on a pod")
        return ("filesystem_premise_onpod", "same_on_pod",
                "every path it inspects is tracked or staged, so the premise "
                "travels to the pod")

    for name in ("staged_artifact", "repository_content"):
        needles, verdict, why = SIGNALS[name]
        if any(n in condition for n in needles):
            return name, verdict, why
    return ("unrecognised", "unknown",
            "no known signal matched; a human must say whether a pod sees the "
            "same premise")


def predicates_in(path: Path, repo: Path, tracked: set[str] | None = None,
                  staged: set[str] | None = None) -> list[dict]:
    src = path.read_text()
    # A mutation test audits a module OUTSIDE the repo, to prove this refuses an
    # unregistered premise. `relative_to` raises on that, which would make the
    # mutation look like a pass.
    rel = (str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path))
    tree = ast.parse(src)
    found: list[dict] = []

    for node in ast.walk(tree):
        # 1. pytest.skip(...) inside a function, with its guarding condition.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("skip", "importorskip", "xfail")
                and _seg(src, node.func).startswith("pytest.")):
            cond, best = "", None
            for parent in ast.walk(tree):
                if isinstance(parent, ast.If):
                    end = getattr(parent, "end_lineno", parent.lineno)
                    if parent.lineno <= node.lineno <= end:
                        # innermost wins: the smallest enclosing span
                        span = end - parent.lineno
                        if best is None or span < best:
                            best, cond = span, _seg(src, parent.test)
            kind = ("importorskip" if node.func.attr == "importorskip"
                    else f"pytest.{node.func.attr}")
            found.append({
                "file": rel, "line": node.lineno, "kind": kind,
                "function": _enclosing(tree, node),
                "condition": (cond or _seg(src, node))[:400],
            })
        # 2. skipif / skipUnless marks, condition first argument.
        if (isinstance(node, ast.Call) and "skipif" in _seg(src, node.func)):
            cond = _seg(src, node.args[0]) if node.args else _seg(src, node)
            found.append({
                "file": rel, "line": node.lineno, "kind": "skipif",
                "function": _enclosing(tree, node),
                "condition": cond[:400],
            })

    consts = _module_constants(tree, src)
    attrs = known_attributes()
    for f in found:
        scope = {**consts, **_local_assignments(tree, src, f["line"]), **attrs}
        f["expanded"] = _expand(f["condition"], scope)[:900]
        sig, verdict, why = classify(f["expanded"], tracked, staged)
        f["signal"], f["verdict"], f["why"] = sig, verdict, why
        f["nodeid"] = (f"{f['file']}::{f['function']}"
                       if f["function"] != "<module>" else f["file"])
    return found


def known_classification(nodeid: str) -> str | None:
    """Groups the readiness contract already names, by nodeid."""
    from aadistill.autoinit import pod_environment as pe
    for group, members in (
            ("renderer_parity", pe.RENDERER_PARITY_NODEIDS),
            ("battery_source", pe.BATTERY_SOURCE_NODEIDS),
            ("host_local_c1", pe.HOST_LOCAL_C1_NODEIDS),
            ("devbox_only", pe.DEVBOX_ONLY_NODEIDS),
            ("known_non_environment", pe.KNOWN_NON_ENVIRONMENT_SKIPS)):
        if any(nodeid == m or m.startswith(nodeid + "[") or nodeid.startswith(m)
               for m in members):
            return group
    return None


def audit(repo: Path = REPO) -> dict:
    files, ignores = c1_selected_modules(repo)
    tracked, staged = repo_inventory(repo)
    predicates: list[dict] = []
    for f in files:
        predicates.extend(predicates_in(f, repo, tracked, staged))

    for p in predicates:
        p["readiness_group"] = known_classification(p["nodeid"])

    differs = [p for p in predicates if p["verdict"] == "differs_on_pod"]
    unknown = [p for p in predicates if p["verdict"] == "unknown"]
    by_signal: dict[str, int] = {}
    for p in predicates:
        by_signal[p["signal"]] = by_signal.get(p["signal"], 0) + 1

    # A predicate that can decide differently on a pod is ACCOUNTED FOR when the
    # audit resolves its signal to a whole class the readiness contract already
    # reasons about, when it carries a readiness group, or when a human has
    # registered it explicitly.
    registry = json.loads((repo / REGISTRY).read_text())
    registered = registry["predicates"]
    AUTO = ("gpu", "optional_dependency", "network", "credential")
    for p in predicates:
        key = f"{p['file']}:{p['line']}"
        p["registered_as"] = registered.get(key, {}).get("class")

    needs_a_word = [p for p in predicates
                    if p["verdict"] != "same_on_pod" and p["signal"] not in AUTO]
    unaccounted = [p for p in needs_a_word
                   if not p["readiness_group"] and not p["registered_as"]]
    live_keys = {f"{p['file']}:{p['line']}" for p in needs_a_word}
    stale = sorted(k for k in registered if k not in live_keys)

    body = {
        "schema": SCHEMA,
        "_what_this_is":
            "every skip construct in the C1-SELECTED suite, with the signal its "
            "condition reads and whether a pod sees the same premise. Written "
            "after attempt 5 died on a predicate keyed to the simulator.",
        "modules_scanned": len(files),
        "test_ignores": ignores,
        "n_predicates": len(predicates),
        "by_signal": dict(sorted(by_signal.items())),
        "n_differs_on_pod": len(differs),
        "n_unrecognised": len(unknown),
        "n_registered": len(registered),
        "registry": REGISTRY,
        "registered_by_class": {
            c: sorted(k for k, v in registered.items() if v["class"] == c)
            for c in sorted(registry["classes"])},
        "stale_registry_entries": stale,
        "unaccounted": sorted(
            {(p["nodeid"], p["signal"]) for p in unaccounted}),
        "unrecognised": sorted({p["nodeid"] for p in unknown}),
        "predicates": sorted(predicates, key=lambda p: (p["file"], p["line"])),
        "signal_definitions": {k: {"verdict": v[1], "why": v[2]}
                               for k, v in SIGNALS.items()},
        "verdict": ("PASS" if not unaccounted and not stale else "REVIEW"),
    }
    body["digest"] = hashlib.sha256(json.dumps(
        {k: v for k, v in body.items() if k != "digest"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rec = audit()
    print(f"{rec['n_predicates']} predicates across {rec['modules_scanned']} "
          f"C1-selected modules")
    for sig, n in rec["by_signal"].items():
        print(f"  {sig:<22} {n:>4}   {SIGNALS.get(sig, ((), '', 'unmatched'))[1]}")
    print(f"differs_on_pod : {rec['n_differs_on_pod']}")
    print(f"unrecognised   : {rec['n_unrecognised']}")
    print(f"registered     : {rec['n_registered']}")
    print(f"stale entries  : {len(rec['stale_registry_entries'])}")
    for k in rec["stale_registry_entries"]:
        print(f"    STALE {k}")
    print(f"unaccounted    : {len(rec['unaccounted'])}")
    for nodeid, sig in rec["unaccounted"]:
        print(f"    {sig:<20} {nodeid}")
    print(f"verdict        : {rec['verdict']}")
    if a.write:
        (REPO / RECORD).write_text(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {RECORD} ({rec['digest'][:12]}…)")
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
