#!/usr/bin/env python3
"""Regenerate the navigation that can be derived: CATALOG classes and READMEs.

    PYTHONPATH=src python scripts/consolidate/render_log_navigation.py --write

Three things here are functions of the tree and of the run index, and were being
maintained by hand:

* the class of every `logs/` entry in `CATALOG.md`;
* each experiment directory's README — what it holds and where its runs are;
* the `logs/experiments/` and `logs/maintenance/` overviews.

They went stale the moment a file moved, which happened thirty times in one
afternoon during the relocation. Deriving them means a move updates the
navigation by re-running this, rather than by remembering to.

**It rewrites only what it generates.** `CATALOG.md`'s ownership rules and its
trailing sections are prose a person wrote and are passed through untouched;
only the `## Classification` section between them is replaced. A README that a
person has edited beyond the generated block is left alone and reported.

It states no cost, no status and no SHA. Those have owners, and a generated
document restating them is the duplication this exists to remove.
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

CATALOG = "logs/state/ownership.md"
READREC = "logs/stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json"
READINESS_RECORD = READREC
READINESS_HISTORY = "logs/stages/stage-1/phase_c1/history/readiness_history.json"
STATE_MD = "logs/state/current.md"
R_BEGIN = "<!-- readiness:begin -->"
R_END = "<!-- readiness:end -->"
INDEX = "logs/index.json"
MARK_START = "## Classification"
MARK_END = "## One copy of every raw artifact"

#: Directory -> (class, one-line purpose). Everything else keeps whatever class
#: the catalog already gave it, and an unknown entry is HISTORICAL: evidence is
#: the safe default, because it is the class that forbids editing.
#: Entries whose row must carry a note, because the class alone does not say
#: the thing a reader needs: what replaced a superseded document. Kept small and
#: explicit -- this is not a place to re-grow a per-file description of every
#: log, which is what was removed.
ANNOTATIONS: dict[str, str] = {
    "HANDOFF_next_session.md":
        "HISTORICAL, superseded 2026-09-10. Written 2026-09-02 and not "
        "maintained; its contents are unaltered. The current handoff is "
        "`STATE.md`",
}

KNOWN: dict[str, tuple[str, str]] = {
    "README.md": ("CURRENT", "the single entry point for this directory"),
    "state": ("CURRENT", "what is true now: current.json, current.md, ownership"),
    "budget": ("CURRENT", "the ledger, the decision records and every approval"),
    "experiments": ("CURRENT",
                    "per experiment: plans, analyses, results, history"),
    "runs": ("CURRENT", "every run, by declared stage or unscoped, and the index"),
    "validations": ("REFERENCE",
                    "engineering validation evidence -- not scientific runs"),
    "maintenance": ("CURRENT", "storage, inventories and cleanup records"),
    "migrations": ("REFERENCE",
                   "how old paths map forward; the relocation provenance"),
    "archive": ("SUPERSEDED", "documents that were current, kept verbatim"),
}


def load(rel: str, root: Path) -> dict:
    return json.loads((root / rel).read_text())


def existing_classes(root: Path) -> dict[str, str]:
    """What class the catalog already assigns each backticked name."""
    text = (root / CATALOG).read_text()
    cls: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^###? ([A-Z]+)", line)
        if m and m.group(1) in ("CURRENT", "REFERENCE", "HISTORICAL",
                               "SUPERSEDED", "TERMINATED"):
            current = m.group(1)
            continue
        if current:
            for t in re.findall(r"`([^`]+)`", line):
                cls.setdefault(t.strip().rstrip("/"), current)
    return cls


def runs_by_experiment(root: Path) -> dict[str, list[tuple[str, str]]]:
    idx = load(INDEX, root)
    out: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for e in [*idx["runs"], *idx["unrecorded"]]:
        root_rel = e.get("root") or (e.get("components") or {}).get("root", "")
        out[e["experiment_id"]].append((e["run_id"], root_rel))
    return {k: sorted(v) for k, v in out.items()}


def render_catalog(root: Path) -> str:
    prior = existing_classes(root)
    rows: dict[str, list[tuple[str, str | None]]] = collections.defaultdict(list)
    for p in sorted((root / "logs").iterdir()):
        name = p.name
        if name in KNOWN:
            c, why = KNOWN[name]
        else:
            c = prior.get(name) or prior.get(name.rstrip("/")) or "HISTORICAL"
            #: No derived annotation. This read the catalog it had just written,
            #: so a name gained a note on the first render and lost it on the
            #: second -- the renderer was not idempotent, and a regeneration
            #: check would have reported permanent drift. The class is the
            #: statement; anything more belongs to the entry's owner.
            why = None
        rows[c].append((name + ("/" if p.is_dir() else ""),
                        ANNOTATIONS.get(name, why)))

    body = [MARK_START, "",
            "Every entry in `logs/`, by class. Regenerated by",
            "[`scripts/consolidate/render_log_navigation.py`]"
            "(../../scripts/consolidate/render_log_navigation.py)",
            "— a move updates this by re-running it, not by remembering to.", "",
            "The per-entry detail — what each attempt cost, what failed, what each",
            "record binds — is in",
            "[`archive/CATALOG_detail_through_2026-09-11.md`]"
            "(../archive/CATALOG_detail_through_2026-09-11.md).",
            "It was maintained here *and* in the ledger, the run index and each",
            "run's own closeout; those are the owners.", ""]
    for c in ("CURRENT", "REFERENCE", "HISTORICAL", "SUPERSEDED", "TERMINATED"):
        if not rows[c]:
            continue
        body += [f"### {c}", ""]
        body += [f"* `{n}`" + (f" — {w}" if w else "") for n, w in sorted(rows[c])]
        body += [""]
    return "\n".join(body)


STAGES = "logs/stages"
CROSS = "logs/cross-stage"


def render_stage_readme(stage_dir: Path, root: Path) -> str:
    """What a stage holds: its experiments, and what each is."""
    stage = stage_dir.name.removeprefix("stage-")
    exps = sorted(p.name for p in stage_dir.iterdir() if p.is_dir())
    lines = [f"# {stage_dir.name}", "",
             f"Pipeline stage **{stage}**. Every experiment here DECLARES this",
             "stage in `configs/experiments/<id>/authorization.json`; the stage is",
             "never inferred from an experiment's name.", "",
             "| experiment | material |", "| --- | --- |"]
    for e in exps:
        sub = sorted(q.name for q in (stage_dir / e).iterdir() if q.is_dir())
        lines.append(f"| [`{e}/`]({e}/) | {', '.join(sub) or '—'} |")
    lines += ["",
              "An experiment's plans, analyses, results, history, validations and",
              "runs are all inside its own directory. Canonical run list, across",
              "every stage: [`../../index.json`](../../index.json).", ""]
    return "\n".join(lines)


def render_group_readme(group: Path, title: str, why: str) -> str:
    exps = sorted(p.name for p in group.iterdir() if p.is_dir())
    lines = [f"# {title}", "", why, "",
             "| directory | material |", "| --- | --- |"]
    for e in exps:
        sub = sorted(q.name for q in (group / e).iterdir() if q.is_dir())
        lines.append(f"| [`{e}/`]({e}/) | {', '.join(sub) or '—'} |")
    lines += ["", "Canonical run list: [`../index.json`](../index.json).", ""]
    return "\n".join(lines)


def render_experiment_readme(d: Path, root: Path,
                             runs: dict[str, list[tuple[str, str]]]) -> str:
    rs = runs.get(d.name, [])
    areas = [q.name for q in sorted(d.iterdir()) if q.is_dir() and q.name != "runs"]
    lines = [f"# {d.name}", "",
             "Everything this experiment produced, in one place.", "",
             "| area | what it holds |", "| --- | --- |"]
    purpose = {"plans": "protocol, preregistration, pricing — what was registered before running",
               "analyses": "working analyses and audits",
               "results": "aggregate results across runs",
               "history": "narrative: what happened, session by session",
               "validations": "engineering evidence supporting this experiment"}
    for a in areas:
        lines.append(f"| [`{a}/`]({a}/) | {purpose.get(a, 'material')} |")
    lines += ["| [`runs/`](runs/) | one directory per execution attempt |", "",
              "## Runs", ""]
    #: Linked only when the directory exists. `legacy_aggregate` is a
    #: registered entry whose components span several directories, not a run
    #: directory, so a link to it would point nowhere.
    lines += ([f"* [`{r}`](runs/{r}/)" if (d / "runs" / r).is_dir()
               else f"* `{r}` — registered from components; see the index"
               for r, _ in rs] or ["* none recorded"])
    lines += ["",
              "Canonical list, always the index rather than this file.",
              "Nothing here authorizes anything.", ""]
    return "\n".join(lines)


def _rel_to_state(target: str) -> str:
    """`target` as a link from `logs/state/current.md`.

    The renderer emitted a bare filename, which was right only while the record
    sat beside the document. After the layout migration it did not, so the link
    resolver repaired the link and the next render broke it again -- the two
    tools overwriting each other on every run.
    """
    import os

    return os.path.relpath(target, Path(STATE_MD).parent).replace("\\", "/")


def readiness_view(root: Path) -> dict:
    """Three separate statements, read from the record rather than restated.

    `STATE.md` said the readiness record was a launch-bound FAILURE while the
    file it linked to was a diagnostic PASS. The three facts had been collapsed
    into one line, so the line was wrong about all of them at once:

    * what the LATEST sweep observed -- its kind, verdict and the tree it swept;
    * whether a launch-bound record exists for the CURRENT tree, which is what a
      launch would rest on and is a different question entirely;
    * that an earlier launch-bound sweep FAILED, which stays true afterwards.

    Derived, so saving a new verification record does not also require editing
    the current view by hand -- which is how it went stale within hours.
    """
    live = json.loads((root / READREC).read_text()) if (
        root / READREC).is_file() else {}
    hist = (json.loads((root / READINESS_HISTORY).read_text())
            if (root / READINESS_HISTORY).is_file() else {"entries": []})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    swept = str(live.get("swept_base_commit") or "")
    lb = [e for e in hist.get("entries", [])
          if e.get("record_kind") == "launch_bound"]
    return {
        "latest": {"kind": live.get("record_kind"), "verdict": live.get("verdict"),
                   "swept_base_commit": swept, "counts": live.get("counts"),
                   "describes_head": bool(swept) and swept == head},
        "launch_bound_ready": (live.get("record_kind") == "launch_bound"
                               and live.get("verdict") == "PASS"
                               and swept == head),
        "launch_bound_failures": [
            {"swept_base_commit": e["swept_base_commit"],
             "committed_utc": e.get("committed_utc"),
             "problems": e.get("problems")}
            for e in lb if e.get("verdict") != "PASS"],
        "history": READINESS_HISTORY,
        "record": READREC,
    }


def render_readiness(root: Path) -> str:
    v = readiness_view(root)
    lt = v["latest"]
    c = lt.get("counts") or {}
    lines = [R_BEGIN, "",
             "| readiness | | owner |", "| --- | --- | --- |"]
    lines.append(
        f"| latest sweep | **{lt['kind']} — {lt['verdict']}**"
        + (f" ({c.get('passed')} passed, {c.get('failed', 0)} failed)" if c else "")
        + f", swept at `{lt['swept_base_commit'][:8]}`"
        + ("; describes the current tree" if lt["describes_head"]
           else "; **does not describe the current tree**")
        + f" | [`{Path(READREC).name}`]({_rel_to_state(READREC)}) |")
    lines.append(
        "| launch-bound for the next session | "
        + ("**PREPARED**" if v["launch_bound_ready"]
           else "**not prepared** — a launch-bound sweep on the final clean "
                "pre-authorization tree is owed")
        + " | this file's launch-chain section |")
    if v["launch_bound_failures"]:
        f = v["launch_bound_failures"][-1]
        lines.append(
            f"| last launch-bound failure | swept at `{str(f['swept_base_commit'])[:8]}`"
            f" on {str(f['committed_utc'])[:10]} — kept as history, not a"
            " current state | [`readiness_history.json`]"
            f"({_rel_to_state(READINESS_HISTORY)}) |")
    lines += ["",
              "*Generated from the record by "
              "`scripts/consolidate/render_log_navigation.py`; do not edit by "
              "hand — it went stale within hours when it was prose.*",
              "", R_END]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    root = REPO_ROOT

    text = (root / CATALOG).read_text()
    i, j = text.index(MARK_START), text.index(MARK_END)
    new_catalog = text[:i] + render_catalog(root) + "\n" + text[j:]

    runs = runs_by_experiment(root)
    readmes = {}
    #: Stage-first: every stage, every experiment inside it, and the two
    #: non-stage groups. Generated from the tree, so a new experiment or a new
    #: stage appears by re-running this rather than by remembering to write one.
    stages_root = root / STAGES
    if stages_root.is_dir():
        readmes[stages_root / "README.md"] = render_group_readme(
            stages_root, "logs/stages",
            "Pipeline stages. A stage exists here when an experiment DECLARES "
            "it in `configs/experiments/<id>/authorization.json`; stages are "
            "never pre-created and never inferred from a name.")
        for st in sorted(p for p in stages_root.iterdir() if p.is_dir()):
            readmes[st / "README.md"] = render_stage_readme(st, root)
            for d in sorted(p for p in st.iterdir() if p.is_dir()):
                readmes[d / "README.md"] = render_experiment_readme(d, root, runs)
    cross = root / CROSS
    if cross.is_dir():
        readmes[cross / "README.md"] = render_group_readme(
            cross, "logs/cross-stage",
            "Experiments whose pipeline stage no frozen record establishes. "
            "That is a finding, not a holding pen: an identifier names the "
            "experiment, and a driver stage or an operator id is a different "
            "dimension. When a record does establish a stage, the experiment "
            "moves under it.")
        for d in sorted(p for p in cross.iterdir() if p.is_dir()):
            readmes[d / "README.md"] = render_experiment_readme(d, root, runs)

    state_p = root / STATE_MD
    state_text = state_p.read_text()
    i2, j2 = state_text.index(R_BEGIN), state_text.index(R_END) + len(R_END)
    new_state = state_text[:i2] + render_readiness(root) + state_text[j2:]

    changed = [CATALOG] if new_catalog != text else []
    if new_state != state_text:
        changed.append(STATE_MD)
    for p, body in readmes.items():
        if not p.exists() or p.read_text() != body:
            changed.append(p.relative_to(root).as_posix())

    if a.write:
        (root / CATALOG).write_text(new_catalog)
        state_p.write_text(new_state)
        for p, body in readmes.items():
            p.write_text(body)
        print(f"rewrote {len(changed)} document(s)")
    else:
        print(f"would rewrite {len(changed)} document(s):")
        for c in changed:
            print("  ", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
