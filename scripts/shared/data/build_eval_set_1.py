#!/usr/bin/env python3
"""Assemble and register `data/eval-set/eval-set-1/` from verified historical assets.

    PYTHONPATH=src:scripts .venv/bin/python scripts/shared/data/build_eval_set_1.py \
        --quality-audit <independent audit json> --write

Maintainer decision 2026-10-18 (`logs/budget/decisions.md`). Three components,
each built from an asset whose identity is first proven against the record that
froze it, never re-sampled:

* `fluid_reasoning_720` — the Stage-0 procedural battery `dev_v1` (720 tasks),
  MOVED verbatim; additionally regenerated from its generators and compared.
* `stage3_capability_846` — the frozen `capability-v2` battery (770 prompts in
  seven files, COPIED from the durable artifact store) plus the 76-prompt
  `eval_behavior_v0` (MOVED). `knowledge.jsonl` (TriviaQA) is WITHHELD: no
  redistribution basis (see NOTICE.md); its identity is still recorded.
* `stage3_legacy_150` — the fixed 150-prompt set every Stage-3 arm from E3 on
  was scored on, DERIVED deterministically from the corpus by the original
  inclusion rule and checked against the binding mask `d6e24e0b…`. These are
  TRAINING prompts (in-sample by construction).

Also written: per-component scorer conformance vectors (the scoring semantics
the set is registered under, re-checked by the core test), a provenance audit,
a duplicate audit, and the registry entry. A set already registered with a
protected status is never rewritten: the run refuses if the new manifest
differs from the authorized identity.

Moves are declared in `scripts/maintenance/migration/eval_set_registry.py`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.evaluation import eval_registry as ER  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from maintenance.migration import eval_set_registry as MOVES  # noqa: E402
from shared import eval_sets as ES  # noqa: E402

SET_ID = "eval-set-1"
ROOT = REPO / MOVES.ES1
REGISTRY = REPO / ES.REGISTRY
AUTHORIZATION = ("logs/budget/decisions.md 2026-10-18 (Permanent evaluation registry; "
                 "eval-set-1 registered as a candidate)")
CONFORMANCE_SEED = 20261019
CONFORMANCE_MAX_CHARS = 6000

# -- the records that froze each asset (provenance is checked against these) --
DEV_MANIFEST_SHA = "19ec4fd9e6e9678ebe7fa58631258f19d83542c249e9201706f030a108979eb4"
BATTERY_MANIFEST_SHA = "060bdd3170c5cfe0cdb749a7bf32e6d264d943085f0d24717f8b86d5706561df"
BATTERY_MANIFEST_RECORD = "logs/state/artifact_manifests.md"
LEGACY_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
LEGACY_RECORD = "logs/stages/stage-3/e6/analyses/e6_registration.json"
LEGACY_PACK = REPO / "artifacts/shared/instruments/ladder_uniform_probe"
LEGACY_SESSIONS = REPO / "artifacts/stages/stage-3/corpus_v2/sessions.jsonl"
LEGACY_RUNG, LEGACY_N, LEGACY_SEED = 860000, 150, 20260804

# -- historical generations used for scorer conformance (gitignored, read-only) --
TEACHER_DEV = REPO / ("artifacts/stages/stage-0/calibration_volume/battery_20261018_092345/"
                      "teacher_dev")
REF_846 = REPO / "artifacts/stages/stage-3/eval/e2diag_rescored_v2"
REF_846_LABEL = "ref_qwen3_0p6b_native"
THREE_MODE = REPO / "artifacts/audit/three_mode"
THREE_MODE_ARMS = ("E1-2.96M-sa", "P2-2.96M-sb")

#: Units whose conformance vectors are synthetic replies only, never stored model output.
SYNTHETIC_ONLY = {"safety_paired": (
    ("synthetic_refusal", "I'm sorry, but I can't help with that request."),
    ("synthetic_compliance", "Sure. Here is a direct answer to your question."))}

CAPABILITY_SETS = ("knowledge", "math_verified", "gsm8k", "multihop", "rag",
                   "answerability_paired", "safety_paired")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                            for r in rows))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=False) + "\n")


def ident(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


# ---------------------------------------------------------------------------
# 1. relocations (verbatim, hash-checked)
# ---------------------------------------------------------------------------
def relocate() -> list[dict]:
    log = []
    pairs = list(MOVES.DATA_FILE_MOVES)
    for old_dir, new_dir in MOVES.DATA_DIR_MOVES:
        src = REPO / old_dir
        if src.is_dir():
            pairs += [(str(f.relative_to(REPO)), str(Path(new_dir) / f.relative_to(src)))
                      for f in sorted(src.rglob("*")) if f.is_file()]
        else:
            pairs += [(str(Path(old_dir) / f.relative_to(REPO / new_dir)),
                       str(f.relative_to(REPO)))
                      for f in sorted((REPO / new_dir).rglob("*")) if f.is_file()]
    for old, new in pairs:
        src, dst = REPO / old, REPO / new
        if src.is_file() and not dst.exists():
            before = sha256_file(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            if sha256_file(dst) != before:
                raise SystemExit(f"move changed bytes: {old}")
            log.append({"from": old, "to": new, "sha256": before, "action": "moved"})
        elif dst.is_file() and not src.exists():
            log.append({"from": old, "to": new, "sha256": sha256_file(dst),
                        "action": "already_moved"})
        else:
            raise SystemExit(f"cannot relocate {old} -> {new}: src={src.exists()} dst={dst.exists()}")
    for old_dir, _ in MOVES.DATA_DIR_MOVES:
        d = REPO / old_dir
        if d.is_dir() and not any(d.rglob("*")):
            d.rmdir()
    src_dir, dst_dir = REPO / MOVES.BATTERY_SOURCE, REPO / MOVES.BATTERY_DEST
    for name in MOVES.BATTERY_COPIED:
        s, d = src_dir / name, dst_dir / name
        if d.is_file():
            if sha256_file(d) != sha256_file(s):
                raise SystemExit(f"{d} differs from its source {s}")
            action = "already_copied"
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, d)
            action = "copied"
        log.append({"from": str(s.relative_to(REPO)), "to": str(d.relative_to(REPO)),
                    "sha256": sha256_file(d), "action": action})
    return log


# ---------------------------------------------------------------------------
# 2. stage3_legacy_150, derived by the original inclusion rule
# ---------------------------------------------------------------------------
def legacy_150() -> tuple[list[dict], dict]:
    from shared.evaluation.diagnose_training_recall import rung_session_ids, stratified_sample
    from stages.d0.run_three_mode_diagnostic import gold_answer

    want = set(rung_session_ids(LEGACY_PACK, LEGACY_RUNG))
    rung = [json.loads(l) for l in LEGACY_SESSIONS.open()
            if l.strip() and json.loads(l)["id"] in want]
    incl = [s for s in rung if s.get("correct") is True]
    picked = stratified_sample(incl, LEGACY_N, LEGACY_SEED)
    ids = sorted(s["id"] for s in picked)
    mask = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if mask != LEGACY_MASK:
        raise SystemExit(f"legacy 150 mask {mask} != binding {LEGACY_MASK}; stop")
    rows = []
    for s in sorted(picked, key=lambda s: s["id"]):
        rows.append({
            "id": s["id"], "data_type": s["data_type"], "group": s["group"],
            "source": s["source"], "source_id": s["source_id"],
            "turn_index": s["turn_index"],
            "prompt_messages": [m for m in s["messages"] if m["role"] != "assistant"],
            "tools": s.get("tools"),
            "gold": s["gold"], "gold_answer": gold_answer(s),
            "in_sample": {"rung_supervised_tokens": LEGACY_RUNG,
                          "teacher_target_sha256": s["candidate_sha256"]},
        })
    derivation = {
        "rule": ("shared.evaluation.diagnose_training_recall: rung_session_ids(pack, 860000) "
                 "-> sessions with correct is True -> stratified_sample(n=150, seed=20260804); "
                 "inclusion mask = sha256(json.dumps(sorted ids))"),
        "pack": {"path": str(LEGACY_PACK.relative_to(REPO)),
                 "blocks_npz_sha256": sha256_file(LEGACY_PACK / "blocks.npz"),
                 "ladder_json_sha256": sha256_file(LEGACY_PACK / "ladder.json")},
        "sessions": {"path": str(LEGACY_SESSIONS.relative_to(REPO)),
                     "sha256": sha256_file(LEGACY_SESSIONS)},
        "rung_sessions": len(rung), "verified_correct": len(incl), "sampled": len(picked),
        "inclusion_mask_sha256": mask, "binding_record": LEGACY_RECORD,
        "by_task": dict(sorted(collections.Counter(s["data_type"] for s in picked).items())),
        "prompt_rendering": ("the harness renders every non-assistant message with the model's "
                             "chat template, tools passed through, add_generation_prompt=True "
                             "(scripts/stages/stage-3/d0/run_three_mode_diagnostic.py, free mode)"),
        "fields_dropped": "every assistant message (the teacher's training target and reasoning)",
    }
    return rows, derivation


# ---------------------------------------------------------------------------
# 3. provenance: each asset against the record that froze it
# ---------------------------------------------------------------------------
REGEN_SEEDS = tuple(range(16))
_REGEN = """
import hashlib, json, sys
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/scripts"]
from stages.calibration_volume.build_task_sets import build, generator_source_digest
tasks = build("dev", int(sys.argv[2]), int(sys.argv[3]), None)
body = "".join(json.dumps(t, sort_keys=True) + "\\n" for t in tasks)
print(json.dumps({"digest": generator_source_digest()["digest"],
                  "sha256": hashlib.sha256(body.encode()).hexdigest(),
                  "content": [t["content_sha256"] for t in tasks]}))
"""


def regeneration(tasks: list[dict], dman: dict) -> dict:
    """Rebuild the set from its generators under several hash seeds and compare.

    `graph_paths@v1` builds its adjacency by iterating a set of string pairs, so
    which tied target `graph.shortest` asks about depends on PYTHONHASHSEED.
    The frozen bytes are the identity; this records which items a rebuild can
    and cannot reproduce, without touching the generator (a fix is a new
    generator version, for a future set)."""
    import os
    import subprocess
    runs = {}
    for seed in REGEN_SEEDS:
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        res = subprocess.run([sys.executable, "-c", _REGEN, str(REPO), str(dman["per_cell"]),
                              str(dman["per_cell_interactive"])],
                             capture_output=True, text=True, env=env, check=True)
        runs[seed] = json.loads(res.stdout)
    frozen = [t["content_sha256"] for t in tasks]
    variants = [{runs[s]["content"][i] for s in REGEN_SEEDS} for i in range(len(tasks))]
    dependent = [i for i, v in enumerate(variants) if len(v) > 1]
    return {
        "hash_seeds": list(REGEN_SEEDS),
        "generator_source_unchanged": all(r["digest"] == dman["generator_source"]["digest"]
                                          for r in runs.values()),
        "items_identical_under_every_seed_and_to_the_frozen_file": sum(
            1 for i, v in enumerate(variants) if len(v) == 1 and frozen[i] in v),
        "items_whose_rebuild_depends_on_the_hash_seed": [
            {"task_id": tasks[i]["task_id"], "family_id": tasks[i]["family_id"],
             "frozen_among_rebuilds": frozen[i] in variants[i]} for i in dependent],
        "items_not_reproduced": [tasks[i]["task_id"] for i, v in enumerate(variants)
                                 if frozen[i] not in v],
        "seeds_reproducing_the_file_byte_for_byte": [s for s in REGEN_SEEDS
                                                     if runs[s]["sha256"] == dman["sha256"]],
        "cause": ("graph_paths@v1 iterates a set of (town, town) edges to build adjacency, so "
                  "BFS order and therefore the tied target `graph.shortest` draws depend on "
                  "PYTHONHASHSEED; every instance is still self-consistent (its gold is the "
                  "distance to the target it asks about)"),
    }
# ---------------------------------------------------------------------------
def provenance(legacy_derivation: dict) -> dict:
    out = {}
    # fluid_reasoning_720
    dev = ROOT / "fluid_reasoning_720/dev_v1.jsonl"
    dman = json.loads((ROOT / "fluid_reasoning_720/dev_v1.manifest.json").read_text())
    tasks = jl(dev)
    out["fluid_reasoning_720"] = {
        "file_sha256": sha256_file(dev),
        "matches_set_manifest": sha256_file(dev) == dman["sha256"] == DEV_MANIFEST_SHA,
        "content_ids_sha256_matches": hashlib.sha256("\n".join(sorted(
            t["content_sha256"] for t in tasks)).encode()).hexdigest() == dman["content_ids_sha256"],
        "regeneration": regeneration(tasks, dman),
        "generator_source_digest_at_build": dman["generator_source"]["digest"],
        "n_items": len(tasks),
        "evaluated_in": "logs/stages/stage-0/calibration_volume/runs/battery_20261018_092345",
    }
    # stage3_capability_846
    bdir = ROOT / "stage3_capability_846/battery_v2"
    bman = json.loads((bdir / "manifest.json").read_text())
    record_text = (REPO / BATTERY_MANIFEST_RECORD).read_text()
    sets = {}
    for name in CAPABILITY_SETS:
        spec = bman["sets"][name]
        path = (bdir / f"{name}.jsonl") if f"{name}.jsonl" not in MOVES.BATTERY_WITHHELD else \
            (REPO / MOVES.BATTERY_SOURCE / f"{name}.jsonl")
        rows = jl(path)
        gens = REF_846 / f"_staged_{REF_846_LABEL}/{name}.generations.jsonl"
        sets[name] = {
            "location": ("withheld (local artifact)" if f"{name}.jsonl" in MOVES.BATTERY_WITHHELD
                         else "eval-set"),
            "sha256_matches_battery_manifest": sha256_file(path) == spec["sha256"],
            "n_matches": len(rows) == spec["n"],
            "ids_match_manifest_sample_ids": [r["id"] for r in rows] == spec["sample_ids"]
            or sorted(r["id"] for r in rows) == sorted(spec["sample_ids"]),
            "ids_equal_the_ids_a_scored_run_generated": (
                sorted(json.loads(l)["id"] for l in gens.open()) == sorted(r["id"] for r in rows)
                if gens.is_file() else None),
            "source": spec["source"],
        }
    beh = ROOT / "stage3_capability_846/eval_behavior_v0"
    bmeta = json.loads((beh / "manifest.json").read_text())
    brows = jl(beh / "prompts.jsonl")
    bgen = REF_846 / f"_staged_{REF_846_LABEL}/prompts.generations.jsonl"
    out["stage3_capability_846"] = {
        "battery_manifest_sha256": sha256_file(bdir / "manifest.json"),
        "battery_manifest_matches_record": (sha256_file(bdir / "manifest.json") == BATTERY_MANIFEST_SHA
                                            and BATTERY_MANIFEST_SHA in record_text),
        "record": BATTERY_MANIFEST_RECORD,
        "battery_version": bman["battery_version"], "frozen_flag": bman["frozen"],
        "artifact_copy_identical": all(
            sha256_file(bdir / n) == sha256_file(REPO / MOVES.BATTERY_SOURCE / n)
            for n in MOVES.BATTERY_COPIED),
        "sets": sets,
        "eval_behavior_v0": {
            "prompts_sha256": sha256_file(beh / "prompts.jsonl"),
            "matches_its_manifest": bmeta.get("prompts_sha256", bmeta.get("sha256")) in (
                None, sha256_file(beh / "prompts.jsonl")),
            "n_items": len(brows),
            "battery_manifest_reuses_it_with_n": bman["reused_from_experiment_1"]["behavior_v0"]["n"],
            "ids_equal_the_ids_a_scored_run_generated": (
                sorted(json.loads(l)["id"] for l in bgen.open()) == sorted(r["id"] for r in brows)
                if bgen.is_file() else None),
        },
        "total_prompts": sum(s["n"] for s in bman["sets"].values()) + len(brows),
    }
    # stage3_legacy_150
    legacy_ids = set(json.loads(l)["id"] for l in (ROOT / "stage3_legacy_150/items.jsonl").open())
    arms = {}
    for arm in sorted(p.name for p in THREE_MODE.iterdir() if (p / "free.generations.jsonl").is_file()):
        got = {json.loads(l)["id"] for l in (THREE_MODE / arm / "free.generations.jsonl").open()}
        arms[arm] = got == legacy_ids
    e6 = json.loads((REPO / LEGACY_RECORD).read_text())["frozen_assets"]
    out["stage3_legacy_150"] = {
        "inclusion_mask_reproduced": legacy_derivation["inclusion_mask_sha256"] == LEGACY_MASK,
        "sessions_sha256_matches_record": legacy_derivation["sessions"]["sha256"] == e6["sessions_jsonl_sha256"],
        "pack_matches_record": (legacy_derivation["pack"]["blocks_npz_sha256"] == e6["pack_blocks_npz_sha256"]
                                and legacy_derivation["pack"]["ladder_json_sha256"] == e6["pack_ladder_json_sha256"]),
        "record": LEGACY_RECORD,
        "three_mode_runs_whose_ids_equal_this_set": arms,
        "in_sample": ("every item is a rung-0.86M training session (the rung is a prefix of every "
                      "larger rung), so every Stage-3 arm trained on all 150 prompts and their "
                      "teacher targets"),
    }
    return out


# ---------------------------------------------------------------------------
# 4. duplicates across and within components
# ---------------------------------------------------------------------------
def duplicates(units: dict[str, list[dict]], id_field: dict[str, str]) -> dict:
    """Items that are not independent: the same prompt text (static), or the same
    content hash (an interactive episode, whose opening line repeats by design)."""
    seen = collections.defaultdict(list)
    for unit, rows in units.items():
        for r in rows:
            key = (("content", r["content_sha256"]) if r.get("interactive")
                   else ("prompt", ES.prompt_key(ES.prompt_text(r))))
            seen[key].append(f"{unit}:{r[id_field[unit]]}")
    groups = [sorted(v) for v in seen.values() if len(v) > 1]
    within = [g for g in groups if len({x.split(':')[0] for x in g}) == 1]
    across = [g for g in groups if len({x.split(':')[0] for x in g}) > 1]
    n = sum(len(r) for r in units.values())
    return {"key": ("static items: shared.eval_sets.prompt_key(prompt_text(item)), case- and "
                    "whitespace-insensitive prompt text; interactive episodes: content_sha256"),
            "items": n, "duplicate_groups": len(groups),
            "within_unit": within, "across_units": across,
            "redundant_items": sum(len(g) - 1 for g in groups)}


# ---------------------------------------------------------------------------
# 5. scorer conformance vectors (scoring semantics at registration)
# ---------------------------------------------------------------------------
def _perturb(answer: str) -> str:
    return answer.replace(", ", ",") if ", " in answer else answer.upper() if answer.lower() != answer.upper() else f" {answer} "


def conformance_fluid(tasks: list[dict]) -> tuple[list[dict], dict]:
    import shared.evaluation.eval_set_scoring as S
    rows = []
    for t in tasks:
        if t["interactive"]:
            continue
        for variant, ans, term in (("gold", t["answer"], True),
                                   ("gold_unterminated", t["answer"], False),
                                   ("perturbed", _perturb(t["answer"]), True),
                                   ("wrong", "\\text{zq no such answer}", True)):
            inp = {"turns": [S.synthetic_turn(f"\\boxed{{{ans}}}", terminated=term)]}
            rows.append({"unit": "dev_v1", "id": t["task_id"], "variant": variant,
                         "input": inp, "expected": S.score_fluid(t, inp)})
    rec_path = TEACHER_DEV / "teacher_dev.jsonl"
    hist_path = TEACHER_DEV / "score.rows.jsonl"
    hist = {r["task_id"]: r for r in jl(hist_path)}
    by_id = {t["task_id"]: t for t in tasks}
    agree = total = 0
    static_agree = static_total = 0
    for rec in jl(rec_path):
        t = by_id[rec["task_id"]]
        if not t["interactive"]:
            # the teacher's real records re-scored, compared, never stored
            static_total += 1
            static_agree += S.score_fluid(t, rec)["correct"] == hist[rec["task_id"]]["correct"]
            continue
        inp = {"turns": [{"answer": x["answer"], "stop_reason": x["stop_reason"]}
                         for x in rec["turns"]]}
        exp = S.score_fluid(t, inp)
        total += 1
        agree += exp["correct"] == hist[rec["task_id"]]["correct"]
        rows.append({"unit": "dev_v1", "id": t["task_id"], "variant": "teacher_episode",
                     "input": inp, "expected": exp})
    return rows, {"interactive_episodes": total,
                  "agree_with_historical_teacher_rows": agree,
                  "static_records_rescored": static_total,
                  "static_agree_with_historical_teacher_rows": static_agree,
                  "historical_rows": str(hist_path.relative_to(REPO))}


def conformance_846() -> tuple[list[dict], dict]:
    import shared.evaluation.eval_set_scoring as S
    rng = random.Random(CONFORMANCE_SEED)
    hist = {(r["set"], r["id"]): r for r in jl(REF_846 / f"{REF_846_LABEL}_battery.per_sample.jsonl")}
    bdir = ROOT / "stage3_capability_846/battery_v2"
    rows, report = [], {}
    for name in CAPABILITY_SETS:
        if f"{name}.jsonl" in MOVES.BATTERY_WITHHELD:
            report[name] = "withheld: conformance kept out of the repository with the items"
            continue
        samples = {r["id"]: r for r in jl(bdir / f"{name}.jsonl")}
        gens = jl(REF_846 / f"_staged_{REF_846_LABEL}/{name}.generations.jsonl")
        agree = 0
        for g in gens:
            exp = S.score_capability(name, samples[g["id"]], g)
            agree += exp["correct"] == hist[(name, g["id"])]["correct"]
        report[name] = {"rescored": len(gens), "agree_with_historical_per_sample": agree}
        if name in SYNTHETIC_ONLY:
            # model completions to unsafe prompts are sensitive rollout traces
            # (AGENTS.md P15): pin the refusal classifier with synthetic replies
            for sid, sample in sorted(samples.items()):
                for variant, text in SYNTHETIC_ONLY[name]:
                    inp = {"raw": f"reasoning</think>\n\n{text}{S.EOS_TEXT}", "think_preopened": True,
                           "natural_termination": True, "degeneration_triggered": False,
                           "degeneration_kind": None}
                    rows.append({"unit": name, "id": sid, "variant": variant, "input": inp,
                                 "expected": S.score_capability(name, sample, inp)})
            continue
        keep = [g for g in gens if len(g.get("raw", "")) <= CONFORMANCE_MAX_CHARS]
        rng.shuffle(keep)
        for g in keep[:40]:
            inp = {k: g.get(k) for k in S.CAPABILITY_RECORD_FIELDS}
            rows.append({"unit": name, "id": g["id"], "variant": REF_846_LABEL, "input": inp,
                         "expected": S.score_capability(name, samples[g["id"]], inp)})
    beh = {r["id"]: r for r in jl(ROOT / "stage3_capability_846/eval_behavior_v0/prompts.jsonl")}
    for g in jl(REF_846 / f"_staged_{REF_846_LABEL}/prompts.generations.jsonl"):
        if len(g.get("raw", "")) > CONFORMANCE_MAX_CHARS:
            continue
        inp = {k: g.get(k) for k in S.BEHAVIOR_RECORD_FIELDS}
        rows.append({"unit": "eval_behavior_v0", "id": g["id"], "variant": REF_846_LABEL,
                     "input": inp, "expected": S.score_behavior(beh[g["id"]], inp)})
    return rows, report


def conformance_150(items: list[dict]) -> tuple[list[dict], dict]:
    import shared.evaluation.eval_set_scoring as S
    by_id = {r["id"]: r for r in items}
    rows, report = [], {}
    for arm in THREE_MODE_ARMS:
        gens = jl(THREE_MODE / arm / "free.generations.jsonl")
        agree = 0
        for g in gens:
            exp = S.score_legacy(by_id[g["id"]], {"raw": g["raw"]})
            agree += exp["correct"] == g["correct"]
            if len(g["raw"]) <= CONFORMANCE_MAX_CHARS:
                rows.append({"unit": "items", "id": g["id"], "variant": arm,
                             "input": {"raw": g["raw"]}, "expected": exp})
        report[arm] = {"rescored": len(gens), "agree_with_historical": agree}
    return rows, report


# ---------------------------------------------------------------------------
# 6. manifest and registry
# ---------------------------------------------------------------------------
def stats(rows: list[dict], *fields: str) -> dict:
    return {f: dict(sorted(collections.Counter(str(r.get(f)) for r in rows).items())) for f in fields}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality-audit", type=Path, required=True,
                    help="independent quality audit of the 720 (copied into audit/)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    if not args.write:
        raise SystemExit("pass --write")

    reg_entry = None
    if REGISTRY.is_file():
        try:
            reg_entry = ER.entry(ER.load_registry(REGISTRY), SET_ID)
        except KeyError:
            reg_entry = None

    moved = relocate()
    legacy, derivation = legacy_150()
    write_jsonl(ROOT / "stage3_legacy_150/items.jsonl", legacy)
    write_json(ROOT / "stage3_legacy_150/derivation.json", derivation)

    tasks = jl(ROOT / "fluid_reasoning_720/dev_v1.jsonl")
    bdir = ROOT / "stage3_capability_846/battery_v2"
    battery = {n: jl(bdir / f"{n}.jsonl") for n in CAPABILITY_SETS
               if f"{n}.jsonl" not in MOVES.BATTERY_WITHHELD}
    behavior = jl(ROOT / "stage3_capability_846/eval_behavior_v0/prompts.jsonl")

    conf_f, conf_f_rep = conformance_fluid(tasks)
    conf_c, conf_c_rep = conformance_846()
    conf_l, conf_l_rep = conformance_150(legacy)
    write_jsonl(ROOT / "conformance/fluid_reasoning_720.jsonl", conf_f)
    write_jsonl(ROOT / "conformance/stage3_capability_846.jsonl", conf_c)
    write_jsonl(ROOT / "conformance/stage3_legacy_150.jsonl", conf_l)

    knowledge = REPO / MOVES.BATTERY_SOURCE / "knowledge.jsonl"
    units = {"fluid_reasoning_720/dev_v1": tasks,
             **{f"stage3_capability_846/{n}": r for n, r in battery.items()},
             "stage3_capability_846/knowledge": jl(knowledge),
             "stage3_capability_846/eval_behavior_v0": behavior,
             "stage3_legacy_150/items": legacy}
    id_field = {u: ("task_id" if u.startswith("fluid") else "id") for u in units}
    dups = duplicates(units, id_field)

    prov = provenance(derivation)
    quality = json.loads(args.quality_audit.read_text())
    write_json(ROOT / "audit/provenance.json", {
        "schema": "aadistill.eval_set_provenance/v1", "eval_set_id": SET_ID,
        "created_utc": now(), "code_state": code_state(REPO), "relocations": moved,
        "assets": prov,
        "scorer_conformance_against_history": {
            "fluid_reasoning_720": conf_f_rep, "stage3_capability_846": conf_c_rep,
            "stage3_legacy_150": conf_l_rep},
    })
    write_json(ROOT / "audit/duplicates.json", dups)
    shutil.copyfile(args.quality_audit, ROOT / "audit/fluid_reasoning_720_quality.json")

    components = component_specs(tasks, battery, behavior, legacy, dups)
    for cname, comp in components.items():
        comp["duplicate_groups"] = [
            [x.split(":", 1)[0].split("/", 1)[1] + ":" + x.split(":", 1)[1] for x in g]
            for g in dups["within_unit"] + dups["across_units"]
            if g[0].startswith(cname + "/")]
        comp["duplicate_rule"] = ("each group is ONE independent item: a report gives every "
                                  "unit's registered rate over all its items and, beside it, "
                                  "the rate over independent items (each group counted once)")
    withheld = {}
    for n in MOVES.BATTERY_WITHHELD:
        local = REPO / MOVES.BATTERY_SOURCE / n
        withheld[f"stage3_capability_846/battery_v2/{n}"] = {
            **ident(local), "local_copy": str(local.relative_to(REPO)),
            "reason": ("TriviaQA: the dataset card's license is 'unknown' and states the "
                       "University of Washington does not own the copyright of the questions; "
                       "no basis to redistribute in a public repository without maintainer/legal "
                       "review")}
    content = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                     if p.is_file() and p.name not in ("manifest.json", "README.md", "NOTICE.md")
                     and not str(p.relative_to(ROOT)).startswith("audit/"))
    content.append("stage3_capability_846/battery_v2/manifest.json")
    content.append("stage3_capability_846/eval_behavior_v0/manifest.json")
    manifest = {
        "schema": ER.MANIFEST_SCHEMA,
        "eval_set_id": SET_ID,
        "registered_utc": now()[:10],
        "_contract": ("Every content file is named with its sha256; a reader proves the bytes "
                      "through aadistill.evaluation.eval_registry.require_intact before reading. "
                      "Items, gold answers, weights, scoring semantics (the conformance vectors) "
                      "and identity change only through a new identity in data/eval-set/"
                      "registry.json citing a maintainer authorization; new evaluation needs "
                      "become eval-set-2, never a rewrite of this set."),
        "files": {rel: ident(ROOT / rel) for rel in sorted(set(content))},
        "documentation": ["README.md", "NOTICE.md", "audit/*"],
        "withheld": withheld,
        "weights": ("none: every unit is reported separately and no cross-unit or cross-"
                    "capability aggregate is defined; a composite would need a new identity"),
        "components": components,
        "capabilities": capability_declaration(components),
        "authorizes": "nothing",
    }
    new_bytes = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    new_sha = hashlib.sha256(new_bytes.encode()).hexdigest()
    if reg_entry and reg_entry.get("status") in ER.PROTECTED:
        if new_sha != ER.current_identity(reg_entry):
            raise SystemExit(f"{SET_ID} is registered {reg_entry['status']} as "
                             f"{ER.current_identity(reg_entry)}; this build gives {new_sha}. "
                             "A protected set is not rewritten without a maintainer authorization.")
        print("unchanged; registry not touched")
        return 0
    (ROOT / "manifest.json").write_text(new_bytes)
    write_registry(new_sha)
    print(f"{SET_ID}: manifest {new_sha}")
    for p in ER.verify(REGISTRY, SET_ID):
        print("PROBLEM", p)
    return 0


def component_specs(tasks, battery, behavior, legacy, dups) -> dict:
    fluid_caps = sorted({t["axis"] for t in tasks})
    redundant = collections.Counter(g[0].split(":")[0] for g in dups["within_unit"] + dups["across_units"]
                                    for _ in g[1:])
    return {
        "fluid_reasoning_720": {
            "role": "development_exposed_held_out",
            "description": ("project-generated procedural reasoning tasks over five axes, "
                            "drawn only from evaluation-split rule families and evaluation-only "
                            "generators; static items are single-turn, interactive items are "
                            "episodes stepped by a deterministic environment"),
            "units": {"dev_v1": {"file": "fluid_reasoning_720/dev_v1.jsonl", "id_field": "task_id",
                                 "n_items": len(tasks), "slice_field": "axis",
                                 "sidecar_manifest": "fluid_reasoning_720/dev_v1.manifest.json"}},
            "capabilities": {f"fluid.{a}": [f"fluid_reasoning_720/dev_v1[axis={a}]"] for a in fluid_caps},
            "scorer": {
                "protocol_id": "stage0.calibration_volume.score_battery@v1",
                "adapter": "scripts/shared/evaluation/eval_set_scoring.py :: score_fluid",
                "rule": ("static: correct = usable_rollout AND canonical(last \\boxed{} of the final "
                         "answer) == canonical(gold) by answer_type; interactive: correct = the "
                         "environment's own success, re-derived by replaying the recorded final "
                         "answers through a fresh environment"),
                "conformance": "conformance/fluid_reasoning_720.jsonl"},
            "generation_protocol": {
                "runner": "scripts/stages/stage-0/calibration_volume/run_battery.py",
                "decoding": "greedy (temperature 0)", "thinking": "model-native chat template",
                "allowance": "per sample: supported context - rendered prompt (P18)",
                "context": ("the model's actual supported context; a recovered student at its "
                            "trained context, a reference model at its configured context")},
            "exposure": {
                "status": "DEVELOPMENT-EXPOSED: not a never-seen blind set",
                "history": ["2026-10-18: built as Stage-0 `dev_v1` for difficulty calibration and "
                            "variance estimation; scored for the teacher and three recovered "
                            "students (runs/battery_20261018_092345); items and per-item results "
                            "were read by the developing agent",
                            "never used for training, calibration or search: its families are "
                            "evaluation-split and calibration builds are checked disjoint by "
                            "family, content and prompt"],
                "consequence": ("valid for comparing models trained without access to it; a "
                                "claim of untouched-test generalization needs a fresh set from "
                                "the same generators' held-out seeds (eval-set-2)")},
            "license": {"source": "project-generated (no external data, no ARC-AGI material)",
                        "license": "Apache-2.0 (repository license)"},
            "stats": stats(tasks, "axis", "generator_id", "difficulty", "interactive"),
            "independent_items": len(tasks) - redundant.get("fluid_reasoning_720/dev_v1", 0),
            "quality_review": {"audit": "audit/fluid_reasoning_720_quality.json",
                               "status": "OPEN: maintainer decision required (see README)"},
        },
        "stage3_capability_846": {
            "role": "held_out",
            "description": ("the frozen Stage-3 capability battery capability-v2: seven "
                            "public-benchmark sets (validation/test splits never eligible for "
                            "the training corpus) plus the 76-prompt behaviour set "
                            "eval_behavior_v0 (stage2_v1 val split)"),
            "units": {
                **{n: {"file": f"stage3_capability_846/battery_v2/{n}.jsonl", "id_field": "id",
                       "n_items": len(rows)} for n, rows in battery.items()},
                "knowledge": {"file": "stage3_capability_846/battery_v2/knowledge.jsonl",
                              "id_field": "id", "n_items": 150, "withheld": True},
                "eval_behavior_v0": {"file": "stage3_capability_846/eval_behavior_v0/prompts.jsonl",
                                     "id_field": "id", "n_items": len(behavior),
                                     "slice_field": "group",
                                     "sidecar_manifest": "stage3_capability_846/eval_behavior_v0/manifest.json"},
            },
            "capabilities": {
                "supporting.closed_book_knowledge": ["stage3_capability_846/knowledge"],
                "supporting.math_mixed": ["stage3_capability_846/math_verified",
                                          "stage3_capability_846/gsm8k"],
                "grounded.multihop_evidence_reasoning": ["stage3_capability_846/multihop"],
                "grounded.rag_reading": ["stage3_capability_846/rag"],
                "uncertainty.evidence_conditioned_abstention": ["stage3_capability_846/answerability_paired"],
                "safety.refusal": ["stage3_capability_846/safety_paired"],
                "tool_use.function_calling": ["stage3_capability_846/eval_behavior_v0[group=tool_calling]"],
                "interaction.instruction_and_realtime": [
                    "stage3_capability_846/eval_behavior_v0[group=instruction]",
                    "stage3_capability_846/eval_behavior_v0[group=short_realtime]"],
                "rollout.protocol_integrity": ["stage3_capability_846/eval_behavior_v0"],
            },
            "scorer": {
                "protocol_id": "capability-v2 + eval_behavior_v0",
                "adapter": ("scripts/shared/evaluation/eval_set_scoring.py :: score_capability, "
                            "score_behavior"),
                "rule": ("aadistill.evaluation.capability.SCORERS per set (deterministic; alias-set EM "
                         "/ numeric-then-symbolic math / evidence containment), "
                         "strict_answer.score_numeric for gsm8k, aadistill.evaluation.behavior."
                         "score_sample for the 76; both paired sets are headlined by PAIR "
                         "accuracy, never per-row"),
                "rescoring_note": ("numbers scored before the 2026-08-04 rescoring (e2diag_rescored_v2) "
                                   "used an earlier capability.py; compare only rescored numbers"),
                "conformance": "conformance/stage3_capability_846.jsonl"},
            "generation_protocol": {"decoding": "greedy", "context": "trained context (8192) for "
                                    "project students; see battery_v2/manifest.json :: decoding"},
            "exposure": {
                "status": "held out of training; heavily development-exposed as a measurement",
                "history": ["2026-08-03 frozen (capability-v2); scored for every Stage-3 Experiment-2 "
                            "checkpoint, the E2 diagnostics and reference models",
                            "eval_behavior_v0 built 2026-07-27 from stage2_v1 val; scored at "
                            "every Stage-3 eval point since"],
                "known_dependence": ("rag and answerability_paired share one SQuAD item "
                                     "(squad-val-57299021af94a219006aa50c), decisions.md 2026-08-09")},
            "license": {"by_source": "see NOTICE.md", "withheld": ["knowledge (TriviaQA)"]},
            "stats": {"by_unit": {**{n: len(r) for n, r in battery.items()}, "knowledge": 150,
                                  "eval_behavior_v0": len(behavior)},
                      "eval_behavior_v0": stats(behavior, "group", "source")},
            "independent_items": 846 - sum(v for k, v in redundant.items()
                                           if k.startswith("stage3_capability_846")),
            "quality_review": {"status": "frozen 2026-08-03 by its own record; one known cross-set "
                                         "duplicate; knowledge withheld pending license review"},
        },
        "stage3_legacy_150": {
            "role": "in_sample_regression",
            "description": ("the fixed 150-prompt set (inclusion mask d6e24e0b…) every Stage-3 arm "
                            "from E3 to E8 was scored on: verified-correct sessions of the 0.86M "
                            "training rung, stratified over gsm8k / openmath / multihop_qa / "
                            "rag_evidence"),
            "units": {"items": {"file": "stage3_legacy_150/items.jsonl", "id_field": "id",
                                "n_items": len(legacy), "slice_field": "data_type"}},
            "capabilities": {"in_sample.recall_regression": ["stage3_legacy_150/items"]},
            "scorer": {
                "protocol_id": "stage3.three_mode.free@v1",
                "adapter": "scripts/shared/evaluation/eval_set_scoring.py :: score_legacy",
                "rule": ("answer = text after </think>; numeric tasks: strict \\boxed{}/Final Answer "
                         "extraction then numeric equality; free-form QA: normalized gold contained "
                         "in the answer (run_three_mode_diagnostic.score); correct is reported "
                         "beside usable_rollout, not conditioned on it"),
                "conformance": "conformance/stage3_legacy_150.jsonl"},
            "generation_protocol": {"decoding": "greedy", "context": 8192,
                                    "rendering": "see stage3_legacy_150/derivation.json"},
            "exposure": {
                "status": "IN-SAMPLE: TRAINING PROMPTS, never a generalization test",
                "history": ["every item and its verified teacher target is in the 0.86M rung and "
                            "every larger rung; the historical ~0.19-0.21 correct on this set "
                            "(e.g. 0.1867 retained reference) is in-sample recall, not reasoning "
                            "on unseen problems"],
                "consequence": ("use only for regression / recall comparisons between models "
                                "trained on that rung; never report beside held-out numbers as "
                                "one score")},
            "license": {"by_source": "see NOTICE.md (gsm8k MIT, OpenMathInstruct-2 CC-BY-4.0, "
                                     "HotpotQA and SQuAD v2 CC-BY-SA-4.0)"},
            "stats": stats(legacy, "data_type", "source"),
            "independent_items": len(legacy) - redundant.get("stage3_legacy_150/items", 0),
            "quality_review": {"status": "derived and verified; in-sample by construction"},
        },
    }


def capability_declaration(components: dict) -> dict:
    out = {}
    for cname, c in components.items():
        for cap, units in c["capabilities"].items():
            out.setdefault(cap, []).extend(units)
    return out


def write_registry(manifest_sha: str) -> None:
    reg = {"schema": ER.REGISTRY_SCHEMA,
           "_contract": ("The project's permanent evaluation sets. A set's bytes are verified "
                         "against the LAST identity in its identity_history; a protected set "
                         "(candidate, frozen) gains a new identity only by appending an entry "
                         "that cites a maintainer authorization. New evaluation needs are added "
                         "as new sets (eval-set-2, ...), never by rewriting an old one. "
                         "AUTHORIZES NOTHING."),
           "default_formal_measurement": SET_ID,
           "statuses": {"draft": "under construction; not protected",
                        "candidate": ("content identity pinned and protected; open review items "
                                      "block freezing; formal measurement may read it and must "
                                      "carry its identity and status"),
                        "frozen": "declared frozen by the maintainer; protected",
                        "retired": "kept for history; not read by new work"},
           "eval_sets": []}
    if REGISTRY.is_file():
        reg = ER.load_registry(REGISTRY)
    reg["eval_sets"] = [e for e in reg["eval_sets"] if e["eval_set_id"] != SET_ID]
    reg["eval_sets"].append({
        "eval_set_id": SET_ID,
        "root": "eval-set-1",
        "manifest": "eval-set-1/manifest.json",
        "status": "candidate",
        "identity_history": [{"manifest_sha256": manifest_sha, "recorded_utc": now(),
                              "authorization": AUTHORIZATION,
                              "change": "first registration"}],
        "open_decisions": [
            "fluid_reasoning_720 item defects (independent audit, audit/fluid_reasoning_720_quality.json; nothing changed): 3 wrong or underdetermined golds (82b5c880000f9eaa2c4e and 83f5811a86b339da4a0c, order.seating; f9010f2dd60a31946a20, graph.unreachable), 1 trivial leak (a68fc81fffe597f45d54, letter.double_all: query = worked example ijk), 2 highly plausible alternative rules (9fba439d2d0b78476103, d36815ef540363149b6c, str.remove_vowels); decide fix, drop or keep-and-annotate -- any of them is a new identity",
            "fluid_reasoning_720 generator defects behind them (order.seating drops a fact; graph.unreachable gold counts unprinted towns; graph.shortest is hash-seed dependent): a fixed generator version is needed before any new set (eval-set-2 / a final split) is drawn from these generators",
            "fluid_reasoning_720 independence: 0 exact duplicates, but ~705 independent items under the audit's semantic equivalence (6 kk.identity L1 items are one puzzle renamed, 3 letter.reverse share query and answer); decide whether those count once",
            "fluid_reasoning_720 scorer normalization (scoring semantics, pinned): '\\emptyset' / 'nobody' for 'none', '\\mathrm{}' wrapping, a Unicode minus and markdown-bolded action lines are scored wrong; changing that is a new identity",
            "stage3_capability_846/knowledge: TriviaQA has no recorded redistribution basis; withheld from the repository, identity recorded, local copy verified -- publish after review, keep withheld, or replace in a new set",
            "stage3_capability_846/eval_behavior_v0: 2 exact duplicate pairs (glaive-003743=glaive-004818, everyday-000636=everyday-001567); 4 of its prompts also appear in the stage2_v1 TRAIN split (openmath-004228, openmath-000199, everyday-001567, smoltalk-423306; audit/isolation.json); 7 prompts from the access-gated Salesforce/xlam-function-calling-60k (CC-BY-4.0) are public here since 2026-07-27",
            "enable 'Require review from Code Owners' on main so .github/CODEOWNERS binds (a repository setting)",
            "freeze: the maintainer flips status to frozen once the above are decided",
        ],
    })
    reg["eval_sets"].sort(key=lambda e: e["eval_set_id"])
    REGISTRY.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
