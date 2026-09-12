#!/usr/bin/env python3
"""Move loose `logs/` material into the experiment that owns it, and record it.

    PYTHONPATH=src python scripts/consolidate/relocate_logs.py           # plan
    PYTHONPATH=src python scripts/consolidate/relocate_logs.py --write

`logs/` root had accumulated 198 loose files whose only organising principle was
a filename prefix. This moves the ones that can move and states, per exception,
what actually holds the rest in place.

**What may move.** A file no executable names. Nothing in `src/`, `scripts/`,
`tests/` or `configs/` reads or writes it, so moving it cannot change what any
run does; only prose points at it, and prose is rewritten here in the same
commit.

**What may not, and why — per file, not as a blanket.**

* *named by an executable* — a producer or consumer resolves this exact path.
  Moving it means moving the writer too, which for anything inside the C1
  harness moves a frozen digest and owes three declarations. Those moves are
  made deliberately and one at a time, not swept along with a tidy-up.
* *pinned by a hash-bearing record* — a consumed authorization, a
  preregistration or a closeout records this path together with a digest. The
  path is part of the evidence.
* *inside a run directory* — owned by that run's manifest.

The point of the distinction is that "cannot move" is a claim with a reason
attached, checkable against the tree, rather than a precaution.

**Nothing is copied.** A move is `git mv`: one path, one owner, full history.
Copying would leave two editable records of one fact, which is the condition
this cleanup exists to end.

**Nothing is deleted.** This script has no delete path at all.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where a logs/ path may be named from. `src/` is absent deliberately: the core
#: must name no log path, and a reference appearing there is a finding, not an
#: input to this plan.
REFERENCE_ROOTS = ("scripts", "tests", "configs")

PATH_LITERAL = re.compile(r'["\']((?:logs|\./logs)/[A-Za-z0-9_./*<>{}-]+)["\']')

#: Filename prefix -> the experiment that owns the file. Read in order. These
#: are conventions this repository already used in `record_run_index.py`; they
#: are written down once more here because a MOVE must not be decided by a
#: guess, and an unmatched file is left alone rather than filed somewhere
#: plausible.
ATTRIBUTION: tuple[tuple[str, str], ...] = (
    (r"^autoinit_c1_|^c1_|^phase_c1_|^phase_c0_|^phase_c_", "phase_c1"),
    (r"^autoinit_phase_a|^phase_a_", "phase_a"),
    (r"^autoinit_phase_b|^phase_b_", "phase_b"),
    (r"^autoinit_continuation_b|^continuation_b", "continuation_b"),
    (r"^autoinit_recovery_continuation|^recovery_", "recovery_continuation"),
    (r"^autoinit_measurement", "measurement"),
    (r"^device_canary|^autoinit_device", "device_canary"),
    (r"^e[0-9]", "early_experiments"),
    (r"^architecture_", "architecture"),
    (r"^storage_|^scratch_|^checkpoint_|^derived_cache|^log_inventory"
     r"|^relay_mirror", "maintenance"),
)

#: Stays at `logs/` root: these ARE the entry points, and an entry point that
#: has to be found through a subdirectory is not one.
ENTRY_POINTS = frozenset({
    "README.md", "STATE.md", "current_state.json", "CATALOG.md",
    "PHASE_INDEX.md", "decisions.md", "BUDGET_LEDGER.md",
    "supported_models.md", "EXPERIMENTS.md", "EXPERIMENT_INDEX.md",
    "artifact_manifests.md",
})

#: Experiment-level material lives beside the runs it explains, not in a
#: parallel tree: `logs/experiments/<id>/`. Runs stay in `logs/runs/`.
DEST_ROOT = "logs/experiments"
MAINTENANCE = "logs/maintenance"


def git(*args: str, root: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout


def executable_references(root: Path) -> dict[str, list[str]]:
    """Every `logs/` path named by a string literal in the executable tree."""
    refs: dict[str, set[str]] = collections.defaultdict(set)
    for top in REFERENCE_ROOTS:
        for f in (root / top).rglob("*"):
            if not f.is_file() or f.suffix not in (".py", ".json", ".sh", ".md"):
                continue
            try:
                text = f.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for m in PATH_LITERAL.finditer(text):
                refs[m.group(1).removeprefix("./")].add(
                    f.relative_to(root).as_posix())
    return {k: sorted(v) for k, v in refs.items()}


def index_registered(root: Path) -> dict[str, str]:
    """Paths the run index registers as a component, with a digest beside them.

    A SEPARATE check, because the index does not store path and digest in one
    object: `components` maps role -> path and `component_digests` maps role ->
    digest, so a walker looking for a dict carrying both finds neither. This
    ran first without it and moved 24 registered components, and the link fixer
    then edited 15 READMEs inside registered directories -- 39 pieces of
    historical evidence whose recorded digest no longer described them.

    A registered component is evidence AT A PATH. It does not move, and files
    inside a registered directory do not change, because the digest covers the
    directory.
    """
    out: dict[str, str] = {}
    p = root / "logs/runs/index.json"
    if not p.is_file():
        return out
    idx = json.loads(p.read_text())
    for e in idx.get("runs", []):
        for role, rel in (e.get("components") or {}).items():
            out[rel] = (f"{e['experiment_id']}/{e['run_id']}::{role} in "
                        "logs/runs/index.json, with a recorded digest")
    return out


def cited_by_frozen_evidence(root: Path) -> dict[str, str]:
    """Paths named in TEXT inside a registered (digest-covered) directory.

    Not a link and not a hash -- a plain citation, of the kind a session record
    or a run README makes: "grant | logs/autoinit_..._grant.json". Moving the
    target leaves immutable evidence pointing at nothing, and the evidence
    cannot be corrected, because correcting it changes the digest that makes it
    evidence. So the target is what holds still.

    Six files were moved and moved back for this reason. They were not caught by
    the executable or digest checks: nothing executes a session record, and the
    citation carries no hash of its own.
    """
    out: dict[str, str] = {}
    p = root / "logs/runs/index.json"
    if not p.is_file():
        return out
    idx = json.loads(p.read_text())
    dirs = sorted({rel for e in idx.get("runs", [])
                   for rel in (e.get("components") or {}).values()
                   if (root / rel).is_dir()})
    texts: list[tuple[str, str]] = []
    for d in dirs:
        for f in (root / d).rglob("*"):
            if not f.is_file():
                continue
            try:
                texts.append((f.relative_to(root).as_posix(), f.read_text()))
            except (OSError, UnicodeDecodeError):
                continue
    for cand in sorted((root / "logs").iterdir()):
        if not cand.is_file():
            continue
        rel = cand.relative_to(root).as_posix()
        for where, body in texts:
            if rel in body:
                out[rel] = f"cited in {where}, which a recorded digest covers"
                break
    return out


def pinned_paths(root: Path) -> dict[str, list[str]]:
    """Paths a hash-bearing record names, so the path is part of the evidence."""
    pinned: dict[str, set[str]] = collections.defaultdict(set)

    def note(p, why):
        if isinstance(p, str) and p.startswith("logs/"):
            pinned[p].add(why)

    def walk(o, src, trail=""):
        if isinstance(o, dict):
            has_digest = any(k in o for k in ("sha256", "digest", "hash"))
            for key in ("path", "file", "record", "artifact"):
                if has_digest:
                    note(o.get(key), f"{src}{trail}")
            for k, v in o.items():
                walk(v, src, f"{trail}.{k}")
        elif isinstance(o, list):
            for i in o:
                walk(i, src, trail)
        elif isinstance(o, str) and o.startswith("logs/") and (
                "sha256" in trail or "digest" in trail):
            note(o, f"{src}{trail}")

    for f in sorted((root / "logs").rglob("*.json")):
        try:
            walk(json.loads(f.read_text()), f.relative_to(root).as_posix())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    return {k: sorted(v) for k, v in pinned.items()}


def attribute(name: str) -> str | None:
    for pat, exp in ATTRIBUTION:
        if re.match(pat, name):
            return exp
    return None


def destination(name: str, experiment: str) -> str:
    if experiment == "maintenance":
        return f"{MAINTENANCE}/{name}"
    return f"{DEST_ROOT}/{experiment}/{name}"


def plan(root: Path = REPO_ROOT) -> dict:
    refs = executable_references(root)
    pins = pinned_paths(root)
    registered = index_registered(root)
    cited = cited_by_frozen_evidence(root)
    moves, exceptions = [], []
    for p in sorted((root / "logs").iterdir()):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        name = p.name
        if name in ENTRY_POINTS:
            exceptions.append({"path": rel, "reason": "entry_point",
                               "detail": "a top-level entry point; it stays "
                                         "findable at the root"})
            continue
        if rel in refs:
            exceptions.append({"path": rel, "reason": "named_by_executable",
                               "detail": f"resolved by {len(refs[rel])} site(s): "
                                         + ", ".join(refs[rel][:3]),
                               "sites": refs[rel]})
            continue
        if rel in registered:
            exceptions.append({"path": rel, "reason": "index_registered",
                               "detail": registered[rel]})
            continue
        if rel in cited:
            exceptions.append({"path": rel, "reason": "cited_by_frozen_evidence",
                               "detail": cited[rel]})
            continue
        if rel in pins:
            exceptions.append({"path": rel, "reason": "pinned_by_record",
                               "detail": "recorded with a digest by "
                                         + ", ".join(pins[rel][:2]),
                               "records": pins[rel]})
            continue
        exp = attribute(name)
        if not exp:
            exceptions.append({"path": rel, "reason": "unattributed",
                               "detail": "no naming convention claims it; left "
                                         "in place rather than filed by guess"})
            continue
        moves.append({"from": rel, "to": destination(name, exp),
                      "experiment": exp})
    return {"moves": moves, "exceptions": exceptions,
            "counts": {
                "loose_before": len(moves) + len(exceptions),
                "moving": len(moves),
                "staying": len(exceptions),
                "by_reason": dict(collections.Counter(
                    e["reason"] for e in exceptions)),
                "by_experiment": dict(collections.Counter(
                    m["experiment"] for m in moves)),
            }}


def rewrite_prose(root: Path, moves: list[dict]) -> list[str]:
    """Repoint every prose mention of a moved path. Text only."""
    table = {m["from"]: m["to"] for m in moves}
    #: Longest first, so `logs/x.json` cannot be rewritten inside
    #: `logs/x.json.bak`.
    keys = sorted(table, key=len, reverse=True)
    touched = []
    for f in sorted(root.rglob("*.md")):
        if ".git" in f.parts or "aad-artifacts" in f.parts:
            continue
        try:
            text = old = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for k in keys:
            if k in text:
                text = text.replace(k, table[k])
        if text != old:
            f.write_text(text)
            touched.append(f.relative_to(root).as_posix())
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--record", default="logs/maintenance/log_relocation.json")
    a = ap.parse_args()

    p = plan()
    c = p["counts"]
    print(f"loose files at logs/ root : {c['loose_before']}")
    print(f"  moving                  : {c['moving']}  {c['by_experiment']}")
    print(f"  staying                 : {c['staying']}  {c['by_reason']}")
    if not a.write:
        print("\n(plan only; pass --write)")
        return 0

    for m in p["moves"]:
        dest = REPO_ROOT / m["to"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        git("mv", m["from"], m["to"])
    touched = rewrite_prose(REPO_ROOT, p["moves"])
    print(f"\nmoved {len(p['moves'])} file(s); repointed prose in "
          f"{len(touched)} document(s)")

    rec = REPO_ROOT / a.record
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps({
        "schema": "aadistill.log_relocation/v1",
        "_what_this_is": (
            "one relocation of loose `logs/` material into the experiment that "
            "owns it. Every move is old path -> new path, so a citation in an "
            "older document can be followed forward. Nothing was copied and "
            "nothing was deleted."),
        "counts": c,
        "moves": p["moves"],
        "prose_repointed": touched,
        "exceptions": p["exceptions"],
        "_exceptions_rule": (
            "each exception states what holds it: an entry point, a live "
            "executable reference with the naming sites listed, a path recorded "
            "beside a digest, or no convention claiming it. `for safety` is not "
            "a reason and does not appear."),
        "authorizes": "nothing",
    }, indent=1) + "\n")
    print(f"recorded {a.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
