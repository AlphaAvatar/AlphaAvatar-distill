#!/usr/bin/env python3
"""Regenerate the navigation that can be derived: CATALOG classes and READMEs.

    PYTHONPATH=src python scripts/consolidate/render_log_navigation.py --write

Three things here are functions of the tree and of the run index, and were being
maintained by hand:

* the class of every `logs/` entry in `CATALOG.md`;
* each experiment directory's README — what it holds and where its runs are;
* the stage READMEs, the stages index and the `logs/shared/` overview.

They went stale the moment a file moved, which happened thirty times in one
afternoon during the reorganisation. Deriving them means a move updates the
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
SNAPSHOT = "logs/state/current.json"
LOGS_README = "logs/README.md"
R_BEGIN = "<!-- readiness:begin -->"
R_END = "<!-- readiness:end -->"
S_BEGIN = "<!-- stages:begin -->"
S_END = "<!-- stages:end -->"
INDEX = "logs/index.json"
MARK_START = "## Classification"
MARK_END = "## One copy of every raw artifact"

#: Directory -> (class, one-line purpose). Everything else keeps whatever class
#: the catalog already gave it, and an unknown entry is HISTORICAL: evidence is
#: the safe default, because it is the class that forbids editing.
#: Entries whose row must carry a note, because the class alone does not say
#: the thing a reader needs. Empty today: `logs/` holds no superseded document
#: that a reader has to be routed away from, because a superseded document is
#: deleted and git history keeps it. Kept as the place such a note would go --
#: not as somewhere to re-grow a per-file description of every log.
ANNOTATIONS: dict[str, str] = {}

KNOWN: dict[str, tuple[str, str]] = {
    "README.md": ("CURRENT", "the single entry point for this directory"),
    "state": ("CURRENT", "what is true now: current.json, current.md, ownership"),
    "budget": ("CURRENT", "the ledger, the decision records and every approval"),
    #: Stage-first: `experiments/`, `runs/` and `validations/` are not
    #: top-level axes -- an experiment's plans, analyses, results, history,
    #: validations and runs are all inside its own directory, under the stage
    #: that owns it.
    "stages": ("CURRENT",
               "stage -> experiment -> run; where every experiment record lives"),
    "index.json": ("CURRENT", "every run, across every stage"),
    "shared": ("CURRENT",
               "records owned by no single experiment; each states whether it "
               "is stage-neutral"),
    "maintenance": ("CURRENT",
                    "storage, inventories, cleanup and source relocations"),
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
            "Per-entry detail is deliberately absent. What an attempt cost, what",
            "failed and what a record binds already have owners — the ledger,",
            "the run index and each run's own closeout — and restating them here",
            "meant maintaining every one of them twice.", ""]
    for c in ("CURRENT", "REFERENCE", "HISTORICAL", "SUPERSEDED", "TERMINATED"):
        if not rows[c]:
            continue
        body += [f"### {c}", ""]
        body += [f"* `{n}`" + (f" — {w}" if w else "") for n, w in sorted(rows[c])]
        body += [""]
    return "\n".join(body)


STAGES = "logs/stages"
STAGE_INDEX = "logs/stages/index.json"

#: Directories under a stage that are the STAGE's own areas rather than one of
#: its experiments. `history/` holds a record that spans the stage's experiments
#: — Stage 3's pre-layout experiment chronology is the only one today. They are
#: named here so that enumerating experiments does not pick them up, and so the
#: stage index is not asked for a row that would have to invent one.
STAGE_AREAS = ("history",)

STAGE_AREA_PURPOSE = {
    "history": "records spanning this stage's experiments, kept verbatim",
}

#: What each stage IS, in the pipeline's own terms (AGENTS.md section 4). The
#: index says which experiments belong to a stage; this says what the stage
#: is for, which no amount of attribution can derive.
STAGE_PURPOSE = {
    "0": ("Initialization warm-up data collection",
          "collect the teacher signals a teacher-aware student initialization "
          "needs (AGENTS.md 4.2)."),
    "1": ("Projection and structural initialization",
          "build a student checkpoint from teacher structure and teacher "
          "activations rather than random weights (AGENTS.md 4.3)."),
    "2": ("Offline warm-up data collection",
          "prepare the offline training data the initialized student is taught "
          "with (AGENTS.md 4.4)."),
    "3": ("Student recovery",
          "recover the initialized student offline, before on-policy training "
          "(AGENTS.md 4.5)."),
}

#: Rows that are experiments; everything else is stage work of another kind.
_EXPERIMENT = "experiment"


def attribution(root: Path) -> dict:
    """The generated stage index. The single source for stage navigation.

    Rendering from the index rather than from the directory listing is the
    point: a directory listing cannot say that `e2` ran and produced no logs,
    and a stage README that omits it is wrong in a way nothing detects.
    """
    return load(STAGE_INDEX, root)


def experiment_dirs(stage_dir: Path) -> list[Path]:
    """The experiment directories a stage holds — its own areas excluded."""
    return sorted(p for p in stage_dir.iterdir()
                  if p.is_dir() and p.name not in STAGE_AREAS)


def _stage_rows(root: Path, stage: str) -> list[dict]:
    return [r for r in attribution(root)["experiments"]
            if r["stage_id"] == stage]


def _material(stage_dir: Path, exp_id: str) -> str:
    d = stage_dir / exp_id
    if not d.is_dir():
        return "—"
    return ", ".join(sorted(q.name for q in d.iterdir() if q.is_dir())) or "—"


def _stage_description(root: Path, stage: str) -> dict:
    for s in attribution(root).get("stages") or []:
        if s["stage_id"] == stage:
            return s
    return {}


def _cited(items) -> list[str]:
    """Bullets for a list of `E(path, says)` dicts or plain strings.

    Paths outside `logs/` are rendered as **code spans, never links**. Most of
    what a stage points at — a corpus, a mixture, a checkpoint — is untracked
    and absent from a fresh clone, and a link to it would be reported broken by
    the repository's own link check while being a perfectly correct pointer at
    a canonical owner.
    """
    out = []
    for item in items or []:
        if isinstance(item, dict):
            out.append(f"* `{item['path']}` — {item['says']}")
        else:
            out.append(f"* {item}")
    return out


def render_stage_readme(stage_dir: Path, root: Path) -> str:
    """What a stage IS and what it holds — derived from the stage index.

    `logs/stages/` is the pipeline's stage-level entry point, so this answers
    the stage-level questions first — purpose, inputs, data, outputs — and only
    then lists experiments and runs. A stage that produced no experiment-run
    logs is not an empty stage; Stage 0 and Stage 2 both produced material that
    every later stage depends on, and a README saying "no logs yet" reports the
    absence of one kind of material as the absence of the stage.

    It is a NAVIGATION layer. Revisions, sample counts, hashes and full configs
    stay with their canonical owners in `configs/`, in the dataset manifests
    beside their data, and in the artifact manifests; this points at them.
    """
    stage = stage_dir.name.removeprefix("stage-")
    rows = _stage_rows(root, stage)
    desc = _stage_description(root, stage)
    title, what = STAGE_PURPOSE.get(stage, (f"Stage {stage}", ""))
    lines = [f"# {stage_dir.name} — {title}", "",
             f"Pipeline stage **{stage}**: {what}", "",
             "Generated from [`../index.json`](../index.json). It records the",
             "evidence behind every claim below; a path in a code span is a",
             "pointer to its canonical owner, which is where the revisions,",
             "counts, hashes and configs live. Stage membership is read from",
             "repository facts — a config's stage field, the",
             "`configs/stage<N>/` directory it lives in, a declared `stage_id`,",
             "or an experiment's own subject and product — never guessed from a",
             "name.", ""]

    if desc.get("purpose"):
        lines += ["## Purpose", "", desc["purpose"], ""]
    if desc.get("inputs"):
        lines += ["## Inputs", "",
                  "What this stage consumes, and from where." if stage == "0"
                  else "What this stage receives from the pipeline before it.",
                  ""] + _cited(desc["inputs"]) + [""]
    if desc.get("data") or desc.get("groups"):
        lines += ["## Data", ""]
        lines += _cited(desc.get("data")) + [""] if desc.get("data") else []
        if desc.get("groups"):
            lines += ["Groups, as the manifests name them:", "",
                      "| group | what it is |", "| --- | --- |"]
            lines += [f"| `{g}` | {w} |" for g, w in desc["groups"]]
            lines.append("")
        if desc.get("splits"):
            lines += ["Every group is split three ways:", ""]
            lines += [f"* {s}" for s in desc["splits"]] + [""]
        if desc.get("sources"):
            lines += ["Source families. The manifests own each one's dataset",
                      "id, revision, license and sample count:", ""]
            lines += [f"* {s}" for s in desc["sources"]] + [""]
    if desc.get("execution"):
        lines += ["## How it runs", "",
                  "Only what helps to understand the stage; the config is the",
                  "source of truth.", ""] + _cited(desc["execution"]) + [""]
    if desc.get("outputs"):
        lines += ["## Outputs", ""] + _cited(desc["outputs"]) + [""]

    areas = [p for p in sorted(stage_dir.iterdir())
             if p.is_dir() and p.name in STAGE_AREAS]
    if areas:
        lines += ["## This stage's own areas", "",
                  "Material that belongs to the stage rather than to one of its",
                  "experiments.", "",
                  "| area | what it holds |", "| --- | --- |"]
        lines += [f"| [`{p.name}/`]({p.name}/) | "
                  f"{STAGE_AREA_PURPOSE.get(p.name, 'material')} |"
                  for p in areas]
        lines.append("")

    activity = [r for r in rows if r["classification"] in
                ("pipeline-activity", "engineering-measurement",
                 "experiment-spanning")]
    if activity:
        lines += ["## Pipeline activity", "",
                  "The stage's own work, and measurements supporting it. Not",
                  "studies *of* it.", "",
                  "| what | kind | logs | status |",
                  "| --- | --- | --- | --- |"]
        for r in activity:
            has = (stage_dir / r["experiment_id"]).is_dir()
            where = (f"[`{r['experiment_id']}/`]({r['experiment_id']}/)" if has
                     else (r.get("canonical_path") or "none"))
            lines.append(f"| {r['title']} | `{r['classification']}` | {where} | "
                         f"{r['status']} |")
        lines.append("")
        for r in activity:
            if r.get("blocked_by"):
                lines += [f"`{r['experiment_id']}` is stage-{stage} material that stays "
                          f"where it is: {r['blocked_by']}. Moving it would mean "
                          "editing frozen-set members to tidy a directory. "
                          "Declared rather than left looking stage-neutral.", ""]

    exps = [r for r in rows if r["classification"] == _EXPERIMENT]
    if not exps:
        lines += ["## Experiments", "",
                  "**None.** No experiment has taken this stage as its subject,",
                  "so there are no experiment-run logs. That is not the same as",
                  "an empty stage: the material above is this stage's real",
                  "output, and no run, experiment or manifest is invented to",
                  "fill the gap. If experiment logs appear later, they get",
                  "directories here.", ""]

    arms = [r for r in rows if r["classification"] in ("arm", "alias")]
    if exps:
        lines += ["## Experiments", "",
                  "What each one asked of this stage.", "",
                  "| experiment | what it asked | logs | status |",
                  "| --- | --- | --- | --- |"]
        for r in exps:
            has = (stage_dir / r["experiment_id"]).is_dir()
            where = f"[`{r['experiment_id']}/`]({r['experiment_id']}/)" if has else "none"
            lines.append(f"| `{r['experiment_id']}` — {r['title']} | "
                         f"{r.get('question') or '—'} | {where} | "
                         f"{r['status']} |")
        lines.append("")
        missing = [r for r in exps if not (stage_dir / r["experiment_id"]).is_dir()]
        if missing:
            lines += ["An experiment with no `logs` directory RAN, and produced no",
                      "log files of its own — its evidence is its configs, its",
                      "artifacts and the historical index. It is listed here",
                      "because omitting it would misreport the stage; it gets no",
                      "directory because an empty one would assert material that",
                      "does not exist. Its evidence is in the stage index:", ""]
            for r in missing:
                lines.append(f"* `{r['experiment_id']}` — " + "; ".join(
                    e["path"] for e in r["evidence"][:4]))
            lines.append("")
    if arms:
        lines += ["### Arms and aliases, filed with their experiment", "",
                  "| name | is | filed under |", "| --- | --- | --- |"]
        for r in arms:
            dest = r.get("canonical_path") or "—"
            lines.append(f"| `{r['experiment_id']}` | {r['classification']} — {r['title']} | "
                         f"`{dest}` |")
        lines += ["",
                  "Not counted as experiments. See the stage index for why.", ""]

    act = desc.get("activity") or {}
    lines += ["## Runs", "",
              f"**{act.get('runs', 0)}** run(s) are registered for this stage's",
              "experiments. An experiment's plans, analyses, results, history,",
              "validations and runs are all inside its own directory; the",
              "canonical run list, across every stage, is",
              "[`../../index.json`](../../index.json).", ""]

    cfgs = desc.get("canonical_configs") or []
    lines += ["## Canonical configs", ""]
    if cfgs:
        lines += ["`configs/` is the source of truth. A run's manifest records",
                  "the config path and hash it ran under.", ""]
        lines += [f"* `{c}`" for c in cfgs] + [""]
    else:
        lines += ["None declared for this stage's rows.", ""]

    mans = desc.get("canonical_data_manifests") or []
    outs = [o for o in (desc.get("outputs") or []) if isinstance(o, dict)]
    lines += ["## Canonical data and artifact manifests", ""]
    if mans or outs:
        lines += ["A dataset manifest lives beside the data it describes, and",
                  "an artifact lives outside git with its manifest. Neither is",
                  "copied here.", ""]
        lines += [f"* `{m}`" for m in mans]
        lines += [f"* `{o['path']}`" for o in outs] + [""]
    else:
        lines += ["None: this stage's material is code and logs.", ""]

    decisions = [x for r in rows for x in r.get("decisions") or []]
    if decisions:
        lines += ["## Decisions", ""] + [f"* {x}" for x in decisions] + [""]

    if desc.get("status"):
        lines += ["## Current status", "", desc["status"], "",
                  "Money, authorizations and what is running right now are not",
                  "here: they belong to [`../../budget/`](../../budget/) and",
                  "[`../../state/`](../../state/).", ""]
    return "\n".join(lines)


def _plural(n: int, one: str, many: str) -> str:
    """`n one` or `n many`; empty when there is nothing to report."""
    return f"{n} {one if n == 1 else many}" if n else ""


def render_stages_index(root: Path, *, base: str = "") -> str:
    """`logs/stages/README.md`, and the same block inside `logs/README.md`.

    A stage is listed because the repository holds evidence that it happened,
    never because AGENTS.md numbers stages 0-6. Stages 4, 5 and 6 appear when
    they do, and are not pre-created.

    `base` is how to reach `logs/stages/` from the document being written --
    empty from inside it, `stages/` from `logs/README.md`. Emitting correct
    links here is what stopped the renderer and the link-fixer from overwriting
    each other's idea of the same link.
    """
    up = "../" if not base else ""
    doc = attribution(root)
    acts = {s["stage_id"]: s.get("activity") or {}
            for s in doc.get("stages") or []}
    #: Experiments are one KIND of material, not the measure of a stage.
    #: Reporting only "experiments: 0" made Stage 0 and Stage 2 read as empty
    #: while they held the statistics cache and the mixtures every later stage
    #: consumes, so what a stage produced is reported beside what studied it.
    lines = ["| stage | what it is | its own work | experiments | runs |",
             "| --- | --- | --- | --- | --- |"]
    for key in doc["stages_present"]:
        sid = key.removeprefix("stage-")
        a = acts.get(sid, {})
        title = STAGE_PURPOSE.get(sid, (f"Stage {sid}", ""))[0]
        own = " · ".join(x for x in (
            _plural(a.get("pipeline_activity", 0), "pipeline activity",
                    "pipeline activities"),
            _plural(a.get("canonical_configs", 0), "config", "configs"),
            _plural(a.get("canonical_data_manifests", 0), "data manifest",
                    "data manifests")) if x) or "—"
        n, withlogs = a.get("experiments", 0), a.get("experiments_with_logs", 0)
        if not n:
            note = "**none** — its output is data and artifacts"
        elif withlogs == n:
            note = f"{n}, all with logs"
        else:
            note = f"{n}, of which {n - withlogs} produced no logs of their own"
        lines.append(f"| [`{key}/`]({base}{key}/) | {title} | {own} | {note} | "
                     f"{a.get('runs', 0)} |")
    t = doc["totals"]
    #: `base` already says how to reach `logs/stages/` from the document being
    #: written, so the index link is the same expression from both.
    m = f"{base}index.json"
    label = "stages/index.json" if base else "index.json"
    lines += ["",
              f"**{t['experiment_total']} experiments — {t['single_stage']} in one "
              f"stage, {t['true_cross_stage']} genuinely cross-stage, "
              f"{t['unresolved']} unresolved.** "
              f"{t['stage_neutral']} stage-neutral infrastructure entries are "
              f"counted separately, in [`shared/`]({up}shared/).", "",
              f"Generated from [`{label}`]({m}), which carries the",
              "evidence for every row."]
    return "\n".join(lines)


def render_shared_readme(root: Path) -> str:
    """`logs/shared/README.md` — what is genuinely stage-neutral, and what is not.

    Without this the directory reads as "everything in here belongs to no
    stage", which is false: most of it is Stage-1 AutoInit program material
    pinned in place by frozen executable sources. Naming the exception is the
    difference between a known compromise and a quiet misfiling.
    """
    doc = attribution(root)
    neutral = [r for r in doc["experiments"] if r["classification"] == "stage-neutral"]
    pinned = [r for r in doc["experiments"]
              if r["classification"] != "stage-neutral" and
              (r.get("canonical_path") or "").startswith("logs/shared")]
    lines = ["# logs/shared", "",
             "Records owned by no single experiment. **Not** a shelf: every",
             "entry below states whether it is stage-neutral, and if it is not,",
             "which stage owns it and why it is still here.", "",
             "## Genuinely stage-neutral", "",
             "The subject is the machine, the provider or the transport — not",
             "any pipeline stage.", "",
             "| what | where | status |", "| --- | --- | --- |"]
    for r in neutral:
        d = (r.get("canonical_path") or "").removeprefix("logs/shared/")
        lines.append(f"| {r['title']} | [`{d}/`]({d}/) | {r['status']} |")
    lines.append("")
    if pinned:
        lines += ["## Here, but not stage-neutral", ""]
        for r in pinned:
            lines += [f"**{r['title']}** — stage **{r['stage_id']}**, status "
                      f"`{r['status']}`.", "",
                      r["classification_reason"], ""]
    lines += ["Recorded in [`../stages/index.json`](../stages/index.json),",
              "which carries the evidence for every row.", ""]
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
    if (d / "runs").is_dir():
        lines.append("| [`runs/`](runs/) | one directory per execution attempt |")
    else:
        #: An experiment whose runs were registered from components spread
        #: across several directories has no `runs/` of its own. Linking one
        #: would point at nothing.
        lines.append("| runs | registered from components; see the index |")
    lines += ["", "## Runs", ""]
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
    #: READREC is a POINTER now, not the record: a run owns its readiness
    #: evidence so that the next sweep cannot overwrite what the previous one
    #: launched under.
    #:
    #: Read the POINTER ONLY, and deliberately do not follow it. Merging the
    #: record made this view a function of a file every sweep rewrites, so
    #: `current.md` and `current.json` went stale the instant a launch-bound
    #: sweep finished — and regenerating them puts tracked changes into a tree
    #: whose lineage permits exactly one. The pointer is navigation and moves
    #: only when navigation is maintained; the record is the authority, and the
    #: launch gate resolves it directly from run_id and stage_id.
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
    #: Stage-first, and driven by the STAGE INDEX rather than by the directory
    #: listing. A listing cannot say that `e2` ran and left no runs of its own,
    #: or that stage 0 and stage 2 exist with no experiment-run logs at all --
    #: so a README derived from one is wrong in a way nothing detects.
    stages_root = root / STAGES
    doc = attribution(root)
    #: A stage directory is created when the index shows the stage happened.
    #: That is not pre-building: stage 0 and stage 2 have configs, builders,
    #: datasets, artifacts and decisions. Stages with no evidence are absent.
    for key in doc["stages_present"]:
        (stages_root / key).mkdir(parents=True, exist_ok=True)
    if stages_root.is_dir():
        readmes[stages_root / "README.md"] = "\n".join([
            "# logs/stages", "",
            "Pipeline stages. **Stage → Experiment → Run.**", "",
            "A stage is here because the repository holds evidence that it",
            "happened — configs, builders, datasets, artifacts, decisions or",
            "experiments — not because AGENTS.md numbers stages 0–6. Stages 4, 5",
            "and 6 will appear the same way, and are not pre-created.", "",
            render_stages_index(root), "",
            "A stage with no experiment-run logs is still a real stage. Its",
            "README says so and points at where its material lives; no run,",
            "experiment or manifest is invented to fill it.", ""])
        for st in sorted(p for p in stages_root.iterdir() if p.is_dir()):
            readmes[st / "README.md"] = render_stage_readme(st, root)
            for d in experiment_dirs(st):
                readmes[d / "README.md"] = render_experiment_readme(d, root, runs)
    shared = root / "logs/shared"
    if shared.is_dir():
        readmes[shared / "README.md"] = render_shared_readme(root)

    #: `current.json.latest_verification` says `_derived_by: readiness_view`
    #: and nothing derived it. It was hand-maintained, so it still read
    #: `diagnostic` at `30f6ceed` while the live record was a launch-bound
    #: sweep at `b6354353` -- the exact drift `_three_separate_facts` was
    #: written about. It points rather than copies: the record owns counts and
    #: digests, and the test that forbids duplicating them still holds.
    snap_p = root / SNAPSHOT
    snap = json.loads(snap_p.read_text())
    v = readiness_view(root)
    lv = dict(snap.get("latest_verification") or {})
    lv["latest_sweep"] = {
        "kind": v["latest"]["kind"],
        "verdict": v["latest"]["verdict"],
        "swept_base_commit": v["latest"]["swept_base_commit"],
        "describes_head": v["latest"]["describes_head"],
    }
    lv["launch_bound_ready"] = v["launch_bound_ready"]
    lv["launch_bound_failures"] = len(v["launch_bound_failures"])
    new_snapshot = dict(snap)
    new_snapshot["latest_verification"] = lv

    #: `latest_run` was hand-maintained too, and went stale the same way: it
    #: still named attempt 13 "PREPARED, not executed" after that attempt was
    #: consumed and closed and attempt 14 existed. Derived from the run index,
    #: which is itself derived from the tree.
    #:
    #: "Latest" is the highest attempt number of the experiment the snapshot
    #: already names -- ordering by run id rather than by mtime, because a file
    #: touched by a regeneration is not a newer run. `state` comes from the
    #: index entry, so a prepared run cannot read as an executed one.
    idx = load(INDEX, root)
    prior = dict(snap.get("latest_run") or {})
    exp = prior.get("experiment_id")
    if exp:
        entries = [e for e in [*idx["runs"], *idx["unrecorded"]]
                   if e["experiment_id"] == exp]

        def _n(e):
            digits = "".join(c for c in e["run_id"] if c.isdigit())
            return (int(digits) if digits else -1, e["run_id"])

        if entries:
            newest = max(entries, key=_n)
            root_rel = (newest.get("root")
                        or (newest.get("components") or {}).get("root"))
            prior["run_id"] = newest["run_id"]
            prior["root"] = root_rel
            prior["state"] = (
                newest["why"] if newest in idx["unrecorded"]
                else "recorded: the run wrote its own manifest")
            new_snapshot["latest_run"] = prior
    snap_body = json.dumps(new_snapshot, indent=1) + "\n"

    state_p = root / STATE_MD
    state_text = state_p.read_text()
    i2, j2 = state_text.index(R_BEGIN), state_text.index(R_END) + len(R_END)
    new_state = state_text[:i2] + render_readiness(root) + state_text[j2:]

    #: `logs/README.md`'s stage table comes from the same index as the stage
    #: READMEs. Two hand-maintained copies of one mapping is how the root README
    #: came to describe a stage layout that had stopped being true.
    logs_readme = root / LOGS_README
    lr_text = logs_readme.read_text()
    i3, j3 = lr_text.index(S_BEGIN) + len(S_BEGIN), lr_text.index(S_END)
    new_logs_readme = (lr_text[:i3] + "\n" + render_stages_index(root, base="stages/")
                       + "\n" + lr_text[j3:])

    changed = [CATALOG] if new_catalog != text else []
    if snap_body != snap_p.read_text():
        changed.append(SNAPSHOT)
    if new_state != state_text:
        changed.append(STATE_MD)
    if new_logs_readme != lr_text:
        changed.append(LOGS_README)
    for p, body in readmes.items():
        if not p.exists() or p.read_text() != body:
            changed.append(p.relative_to(root).as_posix())

    if a.write:
        (root / CATALOG).write_text(new_catalog)
        snap_p.write_text(snap_body)
        state_p.write_text(new_state)
        logs_readme.write_text(new_logs_readme)
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
