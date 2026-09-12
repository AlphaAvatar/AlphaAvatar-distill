#!/usr/bin/env python3
"""Which pipeline stage each experiment belongs to, and the evidence for it.

    PYTHONPATH=src python scripts/consolidate/stage_attribution.py
    PYTHONPATH=src python scripts/consolidate/stage_attribution.py --json

`cross-stage` means **this experiment genuinely spans several pipeline
stages** — not "the migration did not find a `stage_id` field". Using it for
the second is how six historical experiments ended up in one bucket that
explained nothing.

So attribution is rebuilt from repository facts, in the order below. The first
source that decides an experiment wins, and the source is recorded with the
answer so a reader can check it rather than trust it:

1. an explicit `stage` / `stage_id` in the experiment's own configuration;
2. the stage directory the configuration lives in — `configs/stage<N>/<id>/`;
3. a run manifest or preregistration that states a stage;
4. the driver's artifact lineage: what it reads and what it writes.

**Lineage needs a rule, because every autoinit experiment touches two stages.**
Phase A reads `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, measures with
`artifacts/stage3/recovery_search_v2`, and writes probes under
`artifacts/stage3/...`. Read naively that is "cross-stage" — and it would be
wrong, because the repository has already settled the question in a declaration:
**Phase C1 declares `stage_id: "1"`** while doing exactly the same thing. It
selects a Stage-1 initialization operator and measures candidates with Stage-3
recovery probes.

The probe is the *instrument*, not the subject. An experiment whose product is a
Stage-1 initialization decision is a Stage-1 experiment however it measures, and
an experiment whose product is a recovered checkpoint is Stage 3 however it was
initialized. That rule is applied uniformly here, including to the one
experiment that declares its own stage — so the declaration and the lineage rule
agree on C1, which is what makes the rule checkable rather than convenient.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: `stage3_recovery` -> "3". The configs spell the stage as a name.
STAGE_NAME = re.compile(r"^stage(\d+)[_-]")


def from_config_field(experiment_id: str, root: Path) -> tuple[str, str] | None:
    """Source 1: the experiment's configuration states its stage."""
    p = root / f"configs/experiments/{experiment_id}/authorization.json"
    if p.is_file():
        try:
            sid = json.loads(p.read_text()).get("stage_id")
        except (OSError, json.JSONDecodeError):
            sid = None
        if sid:
            return sid, f"{p.relative_to(root)} declares stage_id {sid!r}"
    return None


def from_config_location(experiment_id: str, root: Path) -> tuple[str, str] | None:
    """Sources 2 and 3: `configs/stage<N>/<id>/`, and the `stage` field inside."""
    for d in sorted(root.glob("configs/stage*")):
        cand = d / experiment_id
        if not cand.exists():
            continue
        m = re.match(r"stage(\d+)$", d.name)
        if not m:
            continue
        stage = m.group(1)
        inner = next(iter(sorted(cand.glob("*.json"))), None)
        detail = f"its configuration lives in configs/{d.name}/{experiment_id}/"
        if inner:
            try:
                doc = json.loads(inner.read_text())
                #: Not every config in a stage directory is an object -- some
                #: are lists of arms. The stage is established by the DIRECTORY
                #: either way; this only enriches the reason.
                named = doc.get("stage") if isinstance(doc, dict) else None
            except (OSError, json.JSONDecodeError):
                named = None
            if named:
                detail += f", and {inner.name} names stage {named!r}"
        return stage, detail
    return None


def from_named_inside_stage_config(experiment_id: str, root: Path
                                   ) -> tuple[str, str] | None:
    """An experiment with no directory of its own, named inside one that has.

    `e6` has no `configs/stage3/e6/`, but `configs/stage3/e6b/` names it; `e8a`
    is arm A of `e8` and appears as `configs/stage3/e8/artifacts_a.json`. The
    stage is established by the configuration that names them, which is a
    repository fact and not an inference from the shared letter.
    """
    for d in sorted(root.glob("configs/stage*")):
        m = re.match(r"stage(\d+)$", d.name)
        if not m or not d.is_dir():
            continue
        for f in sorted(d.rglob("*.json")):
            if re.match(rf"{re.escape(experiment_id)}[_.]", f.name):
                return m.group(1), (f"named by {f.relative_to(root)}, which is "
                                    f"inside configs/{d.name}/")
    return None


#: Source 4, applied with the instrument rule stated in the module docstring.
#: Each entry records WHAT the experiment produces, which is what decides it.
LINEAGE: dict[str, tuple[str, str]] = {
    "phase_a": ("1",
                "its product is a Stage-1 initialization decision: the driver "
                "reads artifacts/stage1/qwen3_0p6b_init_v0/checkpoint and "
                "state_eval_v1 and searches initialization operator paths. It "
                "measures candidates with artifacts/stage3/recovery_search_v2, "
                "which is the instrument, not the subject -- the same shape "
                "phase_c1 DECLARES as stage 1"),
    "phase_b": ("1",
                "behavioural selection among the Stage-1 initialization "
                "candidates Phase A produced; the driver reads "
                "artifacts/stage1/state_eval_v1 and its result is which "
                "initialization path wins"),
    "continuation_b": ("1",
                       "the continuation of Phase B's selection, reading the "
                       "same artifacts/stage1/state_eval_v1 manifest; it "
                       "resolves a Stage-1 selection and produces no recovered "
                       "checkpoint"),
    "recovery_continuation": ("1",
                              "the continuation of Phase A's search: reads "
                              "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint "
                              "and the stage-3 battery as its instrument, and "
                              "its product is the Stage-1 path selection"),
    "measurement": ("1",
                    "the `autoinit.causal_depth_measurement` plan: it prices "
                    "the causal-depth INITIALIZATION operator's runtime for the "
                    "autoinit search. Depth compression is a Stage-1 structural "
                    "operator; the E8a comparison is the baseline it is measured "
                    "against, not its subject"),
    "e6": ("3",
           "the predecessor of e6b within the same series: "
           "configs/stage3/e6b/provenance.json names its parents as "
           "configs/stage3/e4/ and configs/stage3/e1/, so the whole lineage is "
           "inside configs/stage3/, every member of which declares "
           "stage 'stage3_recovery'"),
    "e8a": ("3",
            "arm A of experiment e8, not a separate experiment: "
            "configs/stage3/e8/arms.json defines the arms and "
            "artifacts_a.json / completion_markers_a.json are its files. Its "
            "material belongs with e8, which is stage 3 by config location"),
    "phase_c1": ("1",
                 "declared, and the lineage agrees: a fixed-path ATTENTION "
                 "isolation whose product is a Stage-1 initialization operator "
                 "decision, measured with stage-3 confirmation probes"),
}

#: Genuinely not an experiment: infrastructure capability with no single owner.
NOT_AN_EXPERIMENT = {
    "device_canary": ("shared",
                      "a provider/device canary: it proves a machine can run "
                      "the stack at all, and is used by whichever session needs "
                      "it. No stage owns it"),
}


def attribute(experiment_id: str, root: Path = REPO_ROOT) -> dict:
    if experiment_id in NOT_AN_EXPERIMENT:
        dest, why = NOT_AN_EXPERIMENT[experiment_id]
        return {"experiment_id": experiment_id, "stage_ids": [], "home": dest,
                "source": "not an experiment", "reason": why,
                "confidence": "high"}
    for source, fn in (("explicit stage_id", from_config_field),
                       ("config location", from_config_location)):
        got = fn(experiment_id, root)
        if got:
            stage, why = got
            return {"experiment_id": experiment_id, "stage_ids": [stage],
                    "home": f"stages/stage-{stage}", "source": source,
                    "reason": why, "confidence": "high"}
    got = from_named_inside_stage_config(experiment_id, root)
    if got:
        stage, why = got
        return {"experiment_id": experiment_id, "stage_ids": [stage],
                "home": f"stages/stage-{stage}", "source": "named in a stage config",
                "reason": why, "confidence": "high"}
    if experiment_id in LINEAGE:
        stage, why = LINEAGE[experiment_id]
        return {"experiment_id": experiment_id, "stage_ids": [stage],
                "home": f"stages/stage-{stage}", "source": "artifact lineage",
                "reason": why, "confidence": "high"}
    return {"experiment_id": experiment_id, "stage_ids": [], "home": None,
            "source": None,
            "reason": "no repository fact decides this; it is UNRESOLVED and "
                      "must not be filed as cross-stage, which means something "
                      "else",
            "confidence": "none"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("experiments", nargs="*")
    a = ap.parse_args()
    ids = a.experiments or sorted({
        *(p.name for p in (REPO_ROOT / "logs/cross-stage").glob("*")
          if p.is_dir()),
        *(p.name for st in (REPO_ROOT / "logs/stages").glob("stage-*")
          for p in st.iterdir() if p.is_dir()),
    })
    out = [attribute(i) for i in ids]
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    for r in out:
        where = r["home"] or "UNRESOLVED"
        print(f"{r['experiment_id']:24s} {where:18s} [{r['source']}]")
        print(f"    {r['reason']}")
    unresolved = [r["experiment_id"] for r in out if not r["home"]]
    print(f"\nunresolved: {len(unresolved)} {unresolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
