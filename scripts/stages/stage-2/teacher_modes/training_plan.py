#!/usr/bin/env python3
"""The final 3 x 2 training matrix as a PROPOSAL: every unit's recovery config with its lineage block, validated. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.training_plan \\
        --write logs/stages/stage-2/teacher_modes/plans/training_plan_proposal.json

Maintainer 2026-10-31 (Teacher-Student Lineage Invariant and Final 3 x 2 Training Matrix): A1 (the hybrid D4
full-search Top-1; the hybrid in thinking mode as KD teacher and data), A2 (the same Top-1; non-thinking), C1
(Thinking-2507's D4 Top-1; Thinking-2507; D4-Augmented's own pack), seeds 20260726 and 20260801. Six units, no
other. Every unit gets E1's / D4-Augmented's recipe with the registered objective -- CE 0.25 + KD 1.0, T 1, KD on
the assistant target positions only, Top-200 + tail -- and a ``lineage`` block (aadistill.training.lineage) that
config validation (here), launch admission (`admit`) and the trainer at runtime check, failing closed.

A1/A2's Top-1 does not exist yet: their configs carry ``student_path``, the initialization digest and its
provenance record as ``TO_BE_SET_BY_THE_FULL_SEARCH``, and `admit` refuses them until the search's committed record
binds it. Nothing here trains, launches or authorizes anything.

Tokenizers (maintainer review 2026-11-01): a searched Top-1 is a model artifact and carries no tokenizer files, so
every unit names its tokenizer as a separate, hash-pinned source (`TOKENIZERS`): A1/A2 the hybrid's own files at
its pinned revision, staged by ``--stage-tokenizers`` from the snapshot and verified file by file; C1 the PCA init
directory every E-series and D4-Augmented run trained with (Thinking-2507's). The config carries the vocabulary
hash the trainer's tokenizer contract requires (``tokenizer_sha256``) and the lineage binds the files, which tell
the two teachers' tokenizers apart where their vocabularies do not (the chat template).

`admission_report` is the launch admission run unit by unit without raising: every lineage fact the dev box can
observe (the KD teacher, the teacher snapshot where its shards are cached, the initialization's bytes, effective
config and search record, the tokenizer through the trainer's own contract, the pack) and each unit's problems.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.training import lineage as LIN  # noqa: E402
from stages.reasoning_data import assets as A  # noqa: E402

E1 = REPO / "configs/stages/stage-3/e1/e1_r2960k_sa_pca.json"
H1PA = REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"
ADMISSION = REPO / "logs/stages/stage-2/teacher_modes/data/token_matched_admission.json"
C1_AUDIT = REPO / "logs/stages/stage-2/teacher_modes/audit/c1_identity_audit.json"
HYBRID = REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json"
T2507_BINDING = REPO / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
SEEDS = (20260726, 20260801)
UNSET = "TO_BE_SET_BY_THE_FULL_SEARCH"
D4_SEARCH = REPO / "logs/stages/stage-0/calibration_volume/runs/dose_d4_20261020_212310/evidence/d4/search/search_record.json"
#: the tokenizer sources, hash-pinned file by file (the hybrid's from teacher_identity.json :: files_sha256)
HYBRID_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
TOKENIZERS = {"hybrid": "artifacts/stages/stage-2/teacher_modes/tokenizers/qwen3_4b_1cfa9a72",
              "thinking_2507": "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint"}
#: Thinking-2507's tokenizer as the PCA init directory holds it (tokenizer.json as d4_top1_materialization.json ::
#: tokenizer_binding records it; the template is Thinking-2507's own, 3802169b...)
T2507_TOKENIZER_FILES = {"tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
                         "tokenizer_config.json": "8fa82a4ba512c8bee7c1c5e82b9a71ddbef362e4665be5c8f7ce0afd78af129a",
                         "chat_template.jinja": "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"}
SCHEMA = "aadistill.stage2.teacher_modes.training_plan_proposal/v1"
#: the cost owner: per-arm seconds per optimizer step, hours, dollars and deadlines, from v1's and D4-Augmented's
#: measured L40S steps (2026-11-01 re-estimate)
BUDGET = REPO / "logs/stages/stage-2/teacher_modes/plans/budget_v2.json"
OBJECTIVE = {"ce_weight": 0.25, "kd_weight": 1.0, "kd_temperature": 1.0, "kd_scope": "assistant",
             "kd_support": {"support_id": "reference_topk_tail_v1", "top_k": 200}}


def schedule(blocks: int, batch: int = 2, epochs: int = 3) -> dict:
    """E1's builder rule (h1pa.json :: recipe.derive): T = ceil(3 x sequences / 2); warmup max(10, round(0.05 T));
    eval every max(25, T // 8); save every max(200, T // 2)."""
    t = math.ceil(epochs * blocks / batch)
    return {"total_steps": t, "warmup_steps": max(10, round(0.05 * t)), "min_lr_frac": 0.1,
            "eval_every": max(25, t // 8), "save_every": max(200, t // 2)}


def teachers() -> dict:
    h = json.loads(HYBRID.read_text())
    b = json.loads(T2507_BINDING.read_text())
    hist = h["historical_teacher"]
    arch = h["architecture"]
    return {
        "hybrid": {"model_id": h["model"], "revision": h["revision"],
                   "weights_sha256": {s["file"]: s["lfs_sha256"] for s in h["weight_shards"]},
                   "tokenizer_sha256": h["vocabulary_sha256"], "chat_template_sha256": h["chat_template_sha256"],
                   "rope_theta": float(arch["rope_theta"]), "max_position_embeddings": int(arch["max_position_embeddings"]),
                   "vocab_size": int(arch["vocab_size"])},
        "thinking_2507": {"model_id": hist["model"], "revision": hist["revision"],
                          "weights_sha256": dict(b["expected_shard_sha256"]),
                          "tokenizer_sha256": hist["vocabulary_sha256"],
                          "chat_template_sha256": hist["chat_template_sha256"],
                          "rope_theta": float(hist["rope_theta"]), "max_position_embeddings": int(hist["max_position_embeddings"]),
                          "vocab_size": int(arch["vocab_size"])}}


def tokenizer_files() -> dict:
    """teacher key -> {file: sha256} of its pinned tokenizer source."""
    h = json.loads(HYBRID.read_text())
    return {"hybrid": {n: h["files_sha256"][n] for n in HYBRID_TOKENIZER_FILES},
            "thinking_2507": dict(T2507_TOKENIZER_FILES)}


def stage_tokenizers(snapshot_dir: Path, root: Path = REPO) -> dict:
    """Copy the hybrid's tokenizer files from its snapshot at the pinned revision into `TOKENIZERS['hybrid']`,
    refusing any file whose sha256 is not the pinned one; returns {file: sha256} as staged."""
    import shutil
    want = tokenizer_files()["hybrid"]
    dest = root / TOKENIZERS["hybrid"]
    dest.mkdir(parents=True, exist_ok=True)
    for n, sha in want.items():
        got = A.sha256_file(Path(snapshot_dir) / n)
        if got != sha:
            raise SystemExit(f"{snapshot_dir}/{n} is {got[:12]}, the pinned hybrid tokenizer file is {sha[:12]}")
        shutil.copyfile(Path(snapshot_dir) / n, dest / n)
    return LIN.tokenizer_files(dest, want)


def units(adm: dict | None, c1: dict) -> list[dict]:
    e1 = json.loads(E1.read_text())
    h = json.loads(H1PA.read_text())
    ts = teachers()
    tf = tokenizer_files()
    out = []
    arms = {}
    for arm, mode, tmode in (("A1", "H-T", "thinking"), ("A2", "H-N", "non_thinking")):
        a = (adm or {}).get("arms", {}).get(arm) or {}
        pack = a.get("pack") or {}
        arms[arm] = {"teacher": ts["hybrid"], "mode": tmode, "data_dir": f"artifacts/stages/stage-2/teacher_modes/packs_tm/{arm}",
                     "packing": "ladder", "context_strategy": "document_isolated",
                     "files_sha256": pack.get("files_sha256") or {}, "blocks": a.get("sequences"),
                     "rung": a.get("supervised_tokens"), "init_sha256": UNSET, "student_path": UNSET,
                     "provenance": {"record": UNSET, "record_sha256": UNSET},
                     "tokenizer_source": TOKENIZERS["hybrid"], "tokenizer_files": tf["hybrid"],
                     "val_blocks": int((pack.get("n_val_blocks") or 16)), "accounting": a}
    ds = h["datasets"]["e1_2p96M_plus_fluid"]
    arms["C1"] = {"teacher": ts["thinking_2507"], "mode": "thinking", "data_dir": ds["pack"], "packing": "ladder",
                  "context_strategy": "cross_context (D4-Augmented's own pack, kept and disclosed)",
                  "files_sha256": {"blocks.npz": ds["blocks_sha256"], "audit.jsonl": ds["audit_sha256"],
                                   "ladder.json": ds["ladder_sha256"]},
                  "blocks": ds["rung"]["n_blocks"], "rung": ds["rung"]["actual_supervised_tokens"],
                  "init_sha256": c1["initialization"]["model_safetensors_sha256"],
                  "student_path": h["initializations"]["d4_top1"]["path"],
                  "provenance": {"record": str(D4_SEARCH.relative_to(REPO)), "record_sha256": A.sha256_file(D4_SEARCH)},
                  "tokenizer_source": TOKENIZERS["thinking_2507"], "tokenizer_files": tf["thinking_2507"],
                  "val_blocks": 16, "accounting": {}}
    if h["initializations"]["pca"]["path"] != TOKENIZERS["thinking_2507"]:
        raise SystemExit("C1's tokenizer source is not the PCA init directory E1 and D4-Augmented trained with")
    for arm, a in arms.items():
        for seed in SEEDS:
            cfg = copy.deepcopy(e1)
            cfg.update(run_name=f"research_a_v2_{arm}_s{seed}", seed=seed, data_dir=a["data_dir"],
                       packing=a["packing"], student_path=a["student_path"], tokenizer_source=a["tokenizer_source"],
                       tokenizer_sha256=a["teacher"]["tokenizer_sha256"],
                       teacher={"model_id": a["teacher"]["model_id"], "revision": a["teacher"]["revision"],
                                "dtype": "bfloat16"}, loss=dict(OBJECTIVE), rung=a["rung"], val_blocks=a["val_blocks"],
                       out_dir=f"/runpod-volume/teacher_modes/v2_units/{arm}-s{seed}/train")
            cfg.pop("_purpose", None)
            if a["blocks"]:
                sch = schedule(int(a["blocks"]))
                cfg["schedule"] = {k: sch[k] for k in ("total_steps", "warmup_steps", "min_lr_frac")}
                cfg["intervals"] = {**cfg["intervals"], "eval_every": sch["eval_every"], "eval_blocks": a["val_blocks"]}
                cfg["checkpoint"] = {**cfg["checkpoint"], "save_every": sch["save_every"], "bind_identity": True}
            if a["context_strategy"] == "document_isolated":   #: the token-document packs declare it (write_pack)
                cfg["batch"] = {**cfg["batch"], "context_strategy": "document_isolated"}
            cfg["lineage"] = {"teacher": a["teacher"], "mode": a["mode"],
                              "tokenizer": {"files_sha256": a["tokenizer_files"]},
                              "initialization": {"root_teacher": {k: a["teacher"][k] for k in ("model_id", "revision")},
                                                 "model_safetensors_sha256": a["init_sha256"],
                                                 "provenance": dict(a["provenance"])},
                              "data": [{"files_sha256": a["files_sha256"], "mode": a["mode"],
                                        "teacher": {k: a["teacher"][k] for k in ("model_id", "revision")}}]}
            out.append({"unit": f"{arm}-s{seed}", "arm": arm, "seed": seed, "config": cfg, "arm_facts": a})
    return out


def validate(us: list[dict]) -> dict:
    """Config validation: each unit's declared lineage against its own config (KD teacher, declared identities)."""
    res = {}
    for u in us:
        cfg = u["config"]
        bad = LIN.problems(cfg["lineage"], kd_teacher=cfg["teacher"])
        if not cfg.get("tokenizer_source") or not cfg.get("tokenizer_sha256"):
            bad.append("the config names no tokenizer source or no tokenizer sha256 (the trainer refuses both)")
        if cfg.get("tokenizer_sha256") != cfg["lineage"]["teacher"].get("tokenizer_sha256"):
            bad.append("the config's tokenizer sha256 is not its lineage teacher's vocabulary hash")
        res[u["unit"]] = {"problems": bad, "pending_full_search": UNSET in json.dumps(cfg)}
    return res


def observe(u: dict, root: Path = REPO, snapshots: dict | None = None) -> tuple[dict, list[str]]:
    """What launch admission can observe of one unit here, and what it could not (pending)."""
    from aadistill.models.tokenizer_contract import TokenizerContractError, resolve_training_tokenizer
    from shared.training.initialization_provenance import ProvenanceError, read as read_provenance
    cfg, lin = u["config"], u["config"]["lineage"]
    pending, obs = [], {"kd_teacher": cfg["teacher"]}
    snap = (snapshots or {}).get(cfg["teacher"]["model_id"])
    if snap is not None:
        obs["teacher_snapshot"] = LIN.snapshot_facts(snap)
    else:
        pending.append("the teacher snapshot's shards are not cached here: the trainer checks them on the pod")
    if UNSET in (cfg["student_path"], lin["initialization"]["model_safetensors_sha256"]):
        pending.append("the initialization is the full search's Top-1, which does not exist yet")
    else:
        sp = root / cfg["student_path"]
        obs["student_weights_sha256"] = (A.sha256_file(sp / "model.safetensors") if (sp / "model.safetensors").is_file()
                                         else "missing")
        obs["student_config"] = LIN.effective_config(sp) if (sp / "config.json").is_file() else {}
        prov = lin["initialization"]["provenance"]
        try:
            obs["initialization_provenance"] = read_provenance(root / prov["record"], prov["record_sha256"],
                                                               lin["initialization"]["model_safetensors_sha256"])
        except ProvenanceError as exc:
            obs["initialization_provenance"] = {"error": str(exc)}
    dd = root / cfg["data_dir"]
    obs["pack_files_sha256"] = {n: A.sha256_file(dd / n) for n in ("blocks.npz", "ladder.json", "audit.jsonl")
                                if (dd / n).is_file()}
    ts = root / cfg["tokenizer_source"]
    obs["tokenizer_files_sha256"] = LIN.tokenizer_files(ts, lin["tokenizer"]["files_sha256"])
    try:
        _tok, contract = resolve_training_tokenizer(student_path=ts, tokenizer_source=ts,
                                                    expected_sha256=cfg["tokenizer_sha256"], repo_root=root)
        obs["tokenizer_sha256"] = contract["tokenizer_sha256"]
    except TokenizerContractError as exc:
        obs["tokenizer_sha256"] = f"refused: {exc}"
    return obs, pending


def admission_report(us: list[dict], root: Path = REPO, snapshots: dict | None = None) -> dict:
    """Launch admission, unit by unit, without raising: the facts observed, the problems, what is pending."""
    out = {}
    for u in us:
        obs, pending = observe(u, root, snapshots)
        out[u["unit"]] = {"problems": LIN.problems(u["config"]["lineage"], **obs), "pending": pending,
                          "observed": sorted(obs)}
    return out


def admit(us: list[dict], root: Path = REPO, snapshots: dict | None = None) -> None:
    """Launch admission (for the future training launcher): every unit's lineage against the COMMITTED bytes and
    records -- fail closed, all units reported; a unit with anything pending is refused too."""
    rows, bad = [], []
    for u in us:
        obs, pending = observe(u, root, snapshots)
        bad += [f"{u['unit']}: pending: {p}" for p in pending if "teacher snapshot" not in p]
        rows.append((u["unit"], u["config"]["lineage"], obs))
    try:
        LIN.require_all(rows)
    except LIN.LineageError as exc:
        bad.insert(0, str(exc))
    if bad:
        raise LIN.LineageError("; ".join(bad))


def local_snapshots() -> dict:
    """model id -> its cached snapshot directory at the pinned revision, when the weight shards are cached here."""
    from huggingface_hub.constants import HF_HUB_CACHE
    out = {}
    for t in teachers().values():
        d = Path(HF_HUB_CACHE) / f"models--{t['model_id'].replace('/', '--')}" / "snapshots" / t["revision"]
        if all(os.path.lexists(d / n) for n in t["weights_sha256"]) and (d / "config.json").is_file():
            out[t["model_id"]] = d
    return out


def accounting(u: dict, budget: dict) -> dict:
    """The unit's scale from its pack, and its hours and dollars (lower / expected / deadline) from the budget record,
    whose steps must be the pack's."""
    a = u["arm_facts"]
    blocks = a["blocks"]
    if not blocks:
        return {"pending": "the arm's pack does not exist yet"}
    t = math.ceil(3 * int(blocks) / 2)
    arm = u["arm"]
    if int(budget["3_training"]["steps"][arm]) != t:
        raise SystemExit(f"{arm}: the budget prices {budget['3_training']['steps'][arm]} optimizer steps, the pack "
                         f"gives {t}: re-run the budget")
    cost = budget["3_training"]["units"][arm]
    acc = dict(a.get("accounting") or {})
    return {"packed_sequences": blocks, "optimizer_steps": t, "unique_supervised_tokens": a["rung"],
            "supervised_token_exposure": 3 * int(a["rung"]),
            "hours": {k: v["hours"] for k, v in cost.items()}, "usd": {k: v["usd"] for k, v in cost.items()},
            "deadline_hours": budget["3_training"]["per_unit_deadlines_hours"][arm],
            **{k: acc[k] for k in ("non_padding_tokens", "padding_share", "training_flops", "documents", "tasks",
                                   "by_category", "category_share_points") if k in acc}}


A1_PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/training_plan_a1.json"
#: where the full search's Top-1 lives on the A1 phase's network volume (full_search.persist_top1)
TOP1_ON_VOLUME = "/runpod-volume/teacher_modes/search_a1/top1"


def a1_plan(search_record: Path, root: Path = REPO) -> dict:
    """A1's two units with the full search's Top-1 bound (maintainer 2026-11-01, 'Execute A1 First'): the committed
    search record names the root it ran on and the checkpoint it selected; the units' initialization is that
    checkpoint (on the volume), its sha256 and the record itself as provenance. Nothing else changes."""
    from shared.training.initialization_provenance import read as read_provenance
    rel = str(Path(search_record).resolve().relative_to(Path(root).resolve()))
    doc = json.loads((Path(root) / rel).read_text())
    if doc.get("status") != "COMPLETE":
        raise SystemExit(f"{rel} is {doc.get('status')!r}: no Top-1 before the search completes")
    top1 = doc["top1"]
    sha = top1["files_sha256"]["model.safetensors"]
    prov = read_provenance(Path(root) / rel, A.sha256_file(Path(root) / rel), sha)
    adm = json.loads(ADMISSION.read_text())
    us = [u for u in units(adm, json.loads(C1_AUDIT.read_text())) if u["arm"] == "A1"]
    out = []
    for u in us:
        cfg = u["config"]
        cfg["student_path"] = TOP1_ON_VOLUME
        cfg["lineage"]["initialization"].update(model_safetensors_sha256=sha,
                                                provenance={"record": rel, "record_sha256": A.sha256_file(Path(root) / rel)})
        bad = LIN.problems(cfg["lineage"], kd_teacher=cfg["teacher"], initialization_provenance=prov)
        if bad or UNSET in json.dumps(cfg):
            raise SystemExit(f"{u['unit']}: {bad or 'an unset field remains'}")
        out.append({"unit": u["unit"], "arm": "A1", "seed": u["seed"], "config": cfg})
    t = teachers()["hybrid"]
    return {"schema": "aadistill.stage2.teacher_modes.training_plan_a1/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "decided_by": "maintainer 2026-11-01 'Research A v2: Execute A1 First' (phases C and D)",
            "teacher": {"model_id": t["model_id"], "revision": t["revision"]}, "seed_order": list(SEEDS),
            "search_record": {"path": rel, "sha256": A.sha256_file(Path(root) / rel)},
            "top1": {"on_volume": TOP1_ON_VOLUME, "model_safetensors_sha256": sha, "state_id": top1.get("state_id"),
                     "artifact_digest": top1.get("artifact_digest")},
            "units": out, "authorizes": "nothing by itself: the train_a1 kind's one-use artifact funds the session"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    ap.add_argument("--stage-tokenizers", default=None, metavar="SNAPSHOT",
                    help="first copy the hybrid's pinned tokenizer files from this snapshot directory")
    ap.add_argument("--a1-from-search", default=None, metavar="SEARCH_RECORD",
                    help="write training_plan_a1.json from the committed full-search record instead")
    a = ap.parse_args(argv)
    if a.a1_from_search:
        doc = a1_plan(Path(a.a1_from_search))
        A1_PLAN.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(json.dumps({"units": [u["unit"] for u in doc["units"]], "top1": doc["top1"]}, indent=1))
        return 0
    if not a.write:
        ap.error("--write is required unless --a1-from-search is given")
    if a.stage_tokenizers:
        stage_tokenizers(Path(a.stage_tokenizers))
    adm = json.loads(ADMISSION.read_text()) if ADMISSION.is_file() else None
    c1 = json.loads(C1_AUDIT.read_text())
    us = units(adm, c1)
    val = validate(us)
    adm_rep = admission_report(us, snapshots=local_snapshots())
    budget = json.loads(BUDGET.read_text())
    acc = {u["unit"]: accounting(u, budget) for u in us}
    total = budget["3_training"]["six_trainings_usd"]
    rec = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "status": "PROPOSED for the maintainer's review; nothing trains before an explicit authorization",
           "matrix": {"A1": "hybrid Top-1 + hybrid thinking", "A2": "hybrid Top-1 + hybrid non-thinking",
                      "C1": "Thinking-2507 D4 Top-1 + Thinking-2507 + D4-Augmented"}, "seeds": list(SEEDS),
           "objective": OBJECTIVE, "baseline_recipe": {"path": str(E1.relative_to(REPO)), "sha256": A.sha256_file(E1)},
           "tokenizers": {k: {"source": TOKENIZERS[k], "files_sha256": v} for k, v in tokenizer_files().items()},
           "units": [{"unit": u["unit"], "config": u["config"], "lineage_validation": val[u["unit"]],
                      "launch_admission": adm_rep[u["unit"]], "accounting": acc[u["unit"]]} for u in us],
           "cost": {"six_trainings_usd": total, "owner": str(BUDGET.relative_to(REPO)),
                    "budget_sha256": A.sha256_file(BUDGET)},
           "authorizes": "nothing"}
    Path(a.write).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"validation": val, "launch_admission": adm_rep, "usd": total}, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
