#!/usr/bin/env python3
"""On-pod driver of one formal Stage-0 volume dose: search -> recovery -> eval-set-1.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/dose_driver.py \
        --treatment d1 --run-id <run> --auth <authorization.json> --plan dose_d1 \
        --status <file> --volume /runpod-volume

One dose of the preregistered study (`plans/volume_preregistration.json`):

* S -- the D1 full search over the dose's two profiles, in its own process
  (`search_driver.py`), under the treatment's STABLE search identity, with the
  validated execution options (DEPTH prefix reuse, device-side save/reload
  check) and every state's bytes kept;
* R -- the committed Top-1 recovered once, the frozen recipe
  (`e1_p1_kd_heavy@0.86M`) at its own seed, by the C1 trainer;
* E -- the recovered student measured on EVERY item of eval-set-1 (P18: the
  trained 8,192-token context; action protocol v2), scored by
  `shared.evaluation.eval_set_scoring` under the frozen set's formal protocols.

DURABILITY (maintainer 2026-10-19): the search's journal and every materialized
state are mirrored to the network volume while the search runs
(`volume_sync.py`), and each finished stage's products are pushed there before
the next starts. A replacement session for the same dose pulls the mirror back
to the same path first: the search engine restores every journaled state from
its bytes (re-identified, never trusted), a trained student on the volume is
evaluated rather than retrained (P8.4 state 2), and finished evaluation
generations are rescored rather than regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

AUDIT = "artifacts/audit/stage0_calvol_dose"
SEARCH_AUDIT = "artifacts/audit/stage0_calvol_search"
PREREG = "logs/stages/stage-0/calibration_volume/plans/volume_preregistration.json"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
SET_ID = "eval-set-1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu() -> dict:
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
        return dict(zip(("name", "memory", "driver"), [x.strip() for x in line.split(",")]))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def sync(mode: str, src: Path, dst: Path, log: Path) -> dict:
    out = subprocess.run([sys.executable, str(REPO / "scripts/stages/stage-0/calibration_volume/volume_sync.py"),
                          mode, "--src", str(src), "--dst", str(dst), "--log", str(log)],
                         capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"rc": out.returncode, "stderr": out.stderr[-500:]}


def retire_completed_states(study_dir: Path, treatment: str, keep_k: int = 1) -> dict:
    """Free the volume for the next dose: delete a COMPLETE dose's search states
    except its committed Top-1 and, with `keep_k` > 1, the rest of its distinct
    Top-`keep_k` (the 2026-10-20 extension recovers them), leaving a tombstone
    that names every retired state by id, digest and size (maintainer
    2026-10-19: one dose's states fill most of the 300 GB volume; AGENTS.md 2.5).

    The states exist so a lost pod resumes instead of recomputing; a dose whose
    record says COMPLETE has no resume left and nothing else reads them. A dose
    that is not COMPLETE is refused, never trimmed. Idempotent."""
    prev = study_dir / treatment
    tomb = prev / "retired_states.json"
    if tomb.is_file() and json.loads(tomb.read_text()).get("status") == "retired":
        return json.loads(tomb.read_text())
    rec = prev / "dose_out" / "dose_record.json"
    status = json.loads(rec.read_text()).get("status") if rec.is_file() else None
    if status != "COMPLETE":
        raise RuntimeError(f"dose {treatment} is {status or 'absent'} on the volume; only a "
                           "COMPLETE dose's states may be retired")
    unit = prev / "search_unit"
    selection = json.loads((unit / "stage1_selection.json").read_text())
    keep = {r["state_id"] for r in selection["selected"]}
    if keep_k > 1:
        from stages.calibration_volume.finalists import distinct_finalists, journal_index
        keep |= {f["state_id"] for f in distinct_finalists(
            selection, journal_index(unit / "search" / "states.jsonl"), keep_k)}
    digests = {}
    for line in (unit / "search" / "states.jsonl").read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            digests[s["state_id"]] = {
                "artifact_digest": s.get("artifact_digest")
                or (s.get("artifact") or {}).get("artifact_digest"),
                "checkpoint_sha256": s.get("checkpoint_sha256")}
    retired = []
    for d in sorted((unit / "search" / "states").iterdir()):
        if d.name in keep or not d.is_dir():
            continue
        retired.append({"state_id": d.name, **digests.get(d.name, {}),
                        "bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file())})
    doc = {"schema": "aadistill.stage0.calibration_volume.retired_states/v1",
           "treatment": treatment, "status": "retiring",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "why": ("the dose is COMPLETE, so its intermediate states have no resume or "
                   "analysis consumer; the next dose needs the volume (maintainer decision "
                   "2026-10-19). Rebuildable: the search is deterministic from its journal."),
           "kept": sorted(keep), "retired": retired,
           "retired_bytes": sum(r["bytes"] for r in retired),
           "identities": "search_unit/search/states.jsonl, kept beside this file"}
    tomb.write_text(json.dumps(doc, indent=1) + "\n")
    for r in retired:
        shutil.rmtree(unit / "search" / "states" / r["state_id"], ignore_errors=True)
    doc["status"] = "retired"
    tomb.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def release_relayed_finalists(study_dir: Path, treatment: str, manifest: dict, *,
                              api, repo: str, intent: Path) -> dict:
    """Free the volume of an earlier dose's retained Top-k once the relay holds it:
    every file the manifest names for a candidate is checked against what the hub
    STORES (LFS sha256 for the shard, git blob id of the volume copy for the
    small files), then the candidate's search state and rebuilt copy are deleted,
    leaving a tombstone. The probe sessions read these candidates from the relay
    (`finalists.probe_units`), so the volume copies have no consumer left
    (AGENTS.md 2.5, P8.4); on 2026-10-20 the d4 search filled the 300 GB volume
    beside them. Any candidate the relay does not hold refuses the whole dose
    before a byte is deleted. Idempotent.

    The volume this frees is full, so nothing can be written to it before the
    deletion (the first d4 resume died opening the tombstone): the record of what
    is about to go is written to `intent`, off the volume, and the tombstone
    follows the deletion."""
    from stages.calibration_volume.finalists import (_git_blob_sha1, _sha256, _stored,
                                                     journal_index)
    prev = study_dir / treatment
    tomb = prev / "released_finalists.json"
    try:
        if json.loads(tomb.read_text()).get("status") == "released":
            shutil.copyfile(tomb, intent)
            return json.loads(tomb.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass        # absent, or cut short by a write the full volume refused
    unit = prev / "search_unit"
    states = journal_index(unit / "search" / "states.jsonl")
    released = []
    for c in manifest["candidates"][treatment]:
        sid = c["state_id"]
        rows = [f for f in manifest["files"] if f["treatment"] == treatment and f["state_id"] == sid]
        shard = [f for f in rows if f["is_shard"]]
        if len(shard) != 1 or shard[0]["sha256"] != c["checkpoint_sha256"]:
            raise RuntimeError(f"{treatment} {sid}: the manifest names no relay shard "
                               f"{c['checkpoint_sha256'][:12]}")
        state_dir = unit / "search" / "states" / sid
        rebuilt = prev / "finalists" / sid
        if states.get(sid, {}).get("checkpoint_path"):
            ck = Path(states[sid]["checkpoint_path"])
            ck_dir = state_dir / ck.relative_to(ck.parents[1]).relative_to(sid)
        else:
            ck_dir = state_dir
        src = rebuilt if rebuilt.is_dir() else ck_dir
        for f in rows:
            got = list(api.get_paths_info(repo, [f["path"]], repo_type="model"))
            have = _stored(got[0]) if got else None
            if f["is_shard"]:
                ok = bool(have) and have["sha256"] == f["sha256"]
            elif src.is_dir():
                local = src / f["name"]
                ok = (bool(have) and local.is_file() and _sha256(local) == f["sha256"]
                      and have["blob_id"] == _git_blob_sha1(local))
            else:       # an interrupted release already deleted it: nothing left to compare
                ok = bool(have)
            if not ok:
                raise RuntimeError(f"{f['path']}: the relay holds {have}, not sha256 "
                                   f"{f['sha256'][:12]}; nothing of {treatment} was released")
        released.append({"state_id": sid, "rank": c.get("rank"),
                         "checkpoint_sha256": c["checkpoint_sha256"],
                         "relay_paths": sorted(f["path"] for f in rows),
                         "dirs": [str(d.relative_to(study_dir)) for d in (state_dir, rebuilt)
                                  if d.is_dir()],
                         "bytes": sum(p.stat().st_size for d in (state_dir, rebuilt)
                                      if d.is_dir() for p in d.rglob("*") if p.is_file())})
    doc = {"schema": "aadistill.stage0.calibration_volume.released_finalists/v1",
           "treatment": treatment, "status": "releasing", "repo": repo,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "why": ("the relay holds every file of these candidates, verified against the hub "
                   "just before deletion, and the probe sessions read them from there; the "
                   "volume copies had no consumer left and the volume needed the room"),
           "released": released, "released_bytes": sum(r["bytes"] for r in released)}
    intent.write_text(json.dumps(doc, indent=1) + "\n")
    for r in released:
        for d in r["dirs"]:
            shutil.rmtree(study_dir / d, ignore_errors=True)
    doc["status"] = "released"
    write_after_freeing(tomb, json.dumps(doc, indent=1) + "\n")
    intent.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def write_after_freeing(path: Path, text: str) -> None:
    """A tombstone written just after deleting from a full volume, whose quota may
    lag the deletion."""
    for wait in (0, 5, 15, 40, 60):
        time.sleep(wait)
        try:
            return write_atomic(path, text)
        except OSError as exc:
            if exc.errno not in (122, 28) or wait == 60:    # EDQUOT, ENOSPC
                raise


def retire_optimizer_states(study_dir: Path, treatment: str, *, intent: Path) -> dict:
    """Delete a COMPLETE dose's recovery optimizer states (`trainer_state.pt`, 3.5 GB
    each) from the volume, leaving a tombstone with each file's sha256 and size.
    Their one reader is the trainer resuming that same run; a COMPLETE dose is never
    trained again (the preregistered retry_policy) and its evaluated student --
    `model/` -- stays (AGENTS.md 2.5; 2026-10-20 the d4 resume needed the room).
    The intent is written off the full volume, the tombstone after the deletion.
    Idempotent."""
    prev = study_dir / treatment
    tomb = prev / "retired_optimizer_states.json"
    try:
        if json.loads(tomb.read_text()).get("status") == "retired":
            shutil.copyfile(tomb, intent)
            return json.loads(tomb.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    rec = prev / "dose_out" / "dose_record.json"
    status = json.loads(rec.read_text()).get("status") if rec.is_file() else None
    if status != "COMPLETE":
        raise RuntimeError(f"dose {treatment} is {status or 'absent'} on the volume; only a "
                           "COMPLETE dose's optimizer states may be retired")
    files = sorted((prev / "recovery" / "checkpoints").glob("*/trainer_state.pt"))
    retired = [{"path": str(f.relative_to(study_dir)), "bytes": f.stat().st_size,
                "sha256": sha(f)} for f in files]
    doc = {"schema": "aadistill.stage0.calibration_volume.retired_optimizer_states/v1",
           "treatment": treatment, "status": "retiring",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "why": ("the dose is COMPLETE: its recovery is never resumed, so the optimizer "
                   "state has no reader; the trained student it produced is kept"),
           "retired": retired, "retired_bytes": sum(r["bytes"] for r in retired)}
    intent.write_text(json.dumps(doc, indent=1) + "\n")
    for f in files:
        f.unlink()
    doc["status"] = "retired"
    write_after_freeing(tomb, json.dumps(doc, indent=1) + "\n")
    intent.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def volume_room(volume: Path, size_gb: float, dose_dir: Path) -> dict:
    """Room left on the network volume: the size the provider sold less what `du`
    finds on the mount (`df` there reports the shared cluster, not the volume).
    Temporary files an interrupted `volume_sync` left under `dose_dir` are removed
    first; they are never a real file's only copy."""
    stale = [p for p in dose_dir.rglob("*.partial") if p.is_file()]
    for p in stale:
        p.unlink()
    used = int(subprocess.run(["du", "-sb", str(volume)], capture_output=True, text=True,
                              check=True).stdout.split()[0])
    return {"size_bytes": int(size_gb * 1e9), "used_bytes": used,
            "free_bytes": int(size_gb * 1e9) - used,
            "stale_partials_removed": [str(p.relative_to(dose_dir)) for p in stale]}


def read_products(path: Path) -> tuple[dict, str | None]:
    """The dose's product record on the volume. One that does not parse was cut short
    by a failed write (2026-10-20: d4's, to a full volume), so nothing it would
    claim was made durable: it is set aside and the dose resumes unrecorded."""
    if not path.is_file():
        return {}, None
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError:
        aside = path.with_name(f"{path.name}.torn-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        os.replace(path, aside)
        return {}, aside.name


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(text)
    os.replace(tmp, path)


def require_pushed(res: dict, what: str) -> dict:
    """A push whose bytes did not all reach the volume must not be recorded as durable."""
    if "error" in res or "rc" in res:
        raise RuntimeError(f"the push of {what} to the volume failed: {res}")
    return res


def train_recovery(student_dir: Path, rec_out: Path, cfg: Path, log: Path, *, env: dict,
                   run_name: str, seed: int, purpose: str) -> Path:
    """The frozen recipe (`e1_p1_kd_heavy@0.86M`) on `student_dir` by the C1 trainer.
    Only the C1 override set -- run identity, paths, student and seed -- may differ
    from the recipe, and that is checked rather than assumed."""
    from stages.phase_c1.autoinit_c1_driver import (C1_PROBE_OVERRIDES, FROZEN_RECIPE,
                                                    PACK_DIR, TRAINER, trained_model_dir)
    frozen = json.loads(FROZEN_RECIPE.read_text())
    derived = {**frozen, "run_name": run_name, "out_dir": str(rec_out.relative_to(REPO)),
               "data_dir": PACK_DIR, "student_path": str(student_dir), "seed": int(seed),
               "_purpose": purpose}
    outside = sorted(k for k in set(frozen) | set(derived) if frozen.get(k) != derived.get(k))
    if set(outside) - set(C1_PROBE_OVERRIDES):
        raise RuntimeError(f"the derived recovery config differs outside "
                           f"{sorted(C1_PROBE_OVERRIDES)}: {outside}")
    cfg.write_text(json.dumps(derived, indent=2) + "\n")
    shutil.rmtree(rec_out, ignore_errors=True)
    with log.open("w") as lf:
        rc = subprocess.run([TRAIN_PY, str(TRAINER), "--config", str(cfg)], cwd=REPO,
                            env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"the trainer exited {rc}; see {log.name}")
    return trained_model_dir(rec_out)


def evaluate_all(model_dir: Path, out: Path, vol_gens: Path, label: str, prereg: dict,
                 env: dict, *, package: dict | str | None = None,
                 template_kwargs: dict | None = None) -> tuple[dict, list]:
    """Every item of eval-set-1 on the recovered student, then its formal report.
    Generations already on the volume (`vol_gens`) are rescored, never regenerated;
    each finished unit is pushed there before the next starts.

    ``package``: None packages the checkpoint with the historical evaluation
    tokenizer (Stage-1/3's Thinking-2507 sidecars); a dict ``{"tokenizer_source",
    "sidecar_sha256"}`` with that tokenizer's files instead; ``"as_is"`` evaluates
    a directory that already carries its own tokenizer (a teacher snapshot).
    ``template_kwargs`` go to both runners' ``--chat-template-kwargs`` (a hybrid
    template's ``enable_thinking``); absent, neither runner is passed any."""
    from shared import eval_sets as ES
    from stages.phase_c1.autoinit_c1_driver import TOKENIZER_SIDECAR_SHA256, TOKENIZER_SOURCE
    from stages.phase_c1.packaging import build_evaluation_package
    pkg = out / "package"
    if package == "as_is":
        pkg = Path(model_dir)
    else:
        shutil.rmtree(pkg, ignore_errors=True)
        src = package or {"tokenizer_source": TOKENIZER_SOURCE, "sidecar_sha256": TOKENIZER_SIDECAR_SHA256}
        build_evaluation_package(model_dir, tokenizer_source=src["tokenizer_source"], dest=pkg,
                                 expected_sidecar_sha256=src["sidecar_sha256"],
                                 sidecars=tuple(src["sidecar_sha256"]))
    kw = (["--chat-template-kwargs", json.dumps(template_kwargs, sort_keys=True)]
          if template_kwargs is not None else [])
    gens = out / "generations"
    gens.mkdir(exist_ok=True)
    sync("pull", vol_gens, gens, out / "sync.log")
    ctx = str(prereg["evaluation"]["trained_context"])
    #: a decoding protocol the preregistration names; absent, both runners keep
    #: their frozen greedy sampling
    decoding = prereg["evaluation"].get("decoding")
    m = ES.manifest(SET_ID)
    runs = []
    fluid = gens / "fluid_reasoning_720__dev_v1.jsonl"
    if not fluid.is_file():
        tmp = out / "fluid_tmp.jsonl"
        with (out / "eval_fluid.log").open("w") as lf:
            rc = subprocess.run(
                [VLLM_PY, "scripts/stages/stage-0/calibration_volume/run_battery.py",
                 "--model", str(pkg), "--label", label,
                 "--tasks", str(ES.unit_path(SET_ID, "fluid_reasoning_720", "dev_v1")),
                 "--mode", "eval", "--out", str(tmp), "--trained-context", ctx,
                 "--action-protocol", prereg["evaluation"]["action_protocol"]]
                + (["--decoding", str(REPO / decoding), "--seed-namespace", "fluid_reasoning_720__dev_v1"]
                   if decoding else []) + kw,
                cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"run_battery exited {rc}; see eval_fluid.log")
        shutil.copyfile(tmp.with_suffix(".summary.json"), gens / "fluid_reasoning_720__dev_v1.summary.json")
        os.replace(tmp, fluid)
        sync("push", gens, vol_gens, out / "sync.log")
        runs.append("fluid")
    else:   #: the eval unit's log exists either way (the launcher fetches it)
        (out / "eval_fluid.log").write_text(f"{fluid.name}: restored from the volume, "
                                            "rescored, not regenerated\n")
    units = {u: ES.unit_path(SET_ID, "stage3_capability_846", u)
             for u in m["components"]["stage3_capability_846"]["units"]}
    legacy = legacy_prompts(out / "legacy_150.prompts.jsonl")
    todo = {f"stage3_capability_846__{u}": p for u, p in units.items() if p is not None}
    todo["stage3_legacy_150__items"] = legacy
    missing = {k: p for k, p in todo.items() if not (gens / f"{k}.jsonl").is_file()}
    if missing:
        staged = out / "uncapped_in"
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir()
        for k, p in missing.items():
            shutil.copyfile(p, staged / f"{k}.jsonl")
        raw = out / "uncapped_out"
        with (out / "eval_capability.log").open("w") as lf:
            rc = subprocess.run(
                [VLLM_PY, "scripts/shared/evaluation/uncapped_eval.py", "--model", str(pkg),
                 "--label", label, "--prompts", *sorted(str(staged / f"{k}.jsonl") for k in missing),
                 "--out-dir", str(raw), "--trained-context", ctx]
                + (["--decoding", str(REPO / decoding)] if decoding else []) + kw,
                cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"uncapped_eval exited {rc}; see eval_capability.log")
        for k in missing:
            shutil.copyfile(raw / f"{k}.json", gens / f"{k}.summary.json")
            os.replace(raw / f"{k}.generations.jsonl", gens / f"{k}.jsonl")
        sync("push", gens, vol_gens, out / "sync.log")
        runs.append("capability")
    else:
        (out / "eval_capability.log").write_text(
            f"{sorted(todo)}: restored from the volume, rescored, not regenerated\n")
    import shared.evaluation.eval_set_scoring as S
    rep = S.report(gens, label, {}, SET_ID, rows_out=out / "rows")
    return rep, runs


def candidate_dirs(unit: Path, finalists_dir: Path, k: int) -> list[dict]:
    """Each distinct Top-`k` leaf of the dose whose search unit is `unit`, with the
    directory that holds its checkpoint: a rebuilt one under `finalists_dir`,
    otherwise the search's own state, found under `unit` by its journaled path."""
    from stages.calibration_volume.finalists import distinct_finalists, journal_index
    states = journal_index(unit / "search" / "states.jsonl")
    out = []
    for f in distinct_finalists(json.loads((unit / "stage1_selection.json").read_text()),
                                states, k):
        rebuilt = finalists_dir / f["state_id"]
        if rebuilt.is_dir():
            d = rebuilt
        else:
            ck = Path(states[f["state_id"]]["checkpoint_path"])
            d = unit / "search" / "states" / ck.relative_to(ck.parents[1])
        out.append({**f, "dir": str(d)})
    return out


def legacy_prompts(dest: Path) -> Path:
    """The 150 in uncapped_eval's prompt shape: the item's own messages (system
    included, so the evaluator does not inject a second one), its tools, and the
    `prompt_text` its in-run scorer reads (d1 attempt 2 died on that KeyError
    after generating every capability unit)."""
    from shared import eval_sets as ES
    rows = []
    for it in ES.items(SET_ID, "stage3_legacy_150", "items"):
        rows.append({"id": it["id"], "group": it.get("data_type"), "source": it.get("source"),
                     "messages": it["prompt_messages"], "tools": it.get("tools"),
                     "prompt_text": ES.prompt_text(it)})
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return dest


def main(argv=None) -> int:
    from stages.calibration_volume import governance as G

    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--image-digest", default=None)
    args = ap.parse_args(argv)
    plan = G.plan_doc(args.plan)
    prereg = json.loads((REPO / PREREG).read_text())
    treat = prereg["treatments_by_id"][args.treatment]
    out = REPO / AUDIT / args.treatment
    out.mkdir(parents=True, exist_ok=True)
    vol = Path(args.volume) / "stage0_calvol" / prereg["study_id"] / args.treatment
    status = Path(args.status)
    search_unit = REPO / SEARCH_AUDIT / args.treatment
    env = {**os.environ, "PYTHONPATH": "src:scripts"}
    if args.image_digest:
        env["AADISTILL_IMAGE_DIGEST"] = args.image_digest
    record = {"schema": "aadistill.stage0.calibration_volume.dose_record/v1",
              "treatment": args.treatment, "run_id": args.run_id,
              "search_run_id": treat["search_run_id"], "volume_dir": str(vol),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       capture_output=True, text=True).stdout.strip(),
              "gpu": gpu(), "stages": {}, "status": "STARTED"}

    def mark(m: str) -> None:
        with status.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        (out / "dose_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    def done_unit(name: str, path: Path) -> None:
        (out / "records").mkdir(exist_ok=True)
        (out / "records" / f"{name}.done").write_text(sha(path) + "\n")

    def stage(name: str):
        record["stages"][name] = {"t0": time.time(), "status": "running"}
        save()
        mark(f"STAGE_START:{name}")
        return record["stages"][name]

    def finish(st: dict, name: str, **kw) -> None:
        st.update(kw, seconds=round(time.time() - st.pop("t0"), 1), status="ok")
        save()
        mark(f"STAGE_OK:{name}")

    mark(f"DRIVER_START:{args.treatment}")
    try:
        # -- A: contract ---------------------------------------------------
        st = stage("A_contract")
        auth = G.load_for("dose", args.auth)
        auth.require_plan(G.plan_hash(args.plan))
        record["authorization_id"] = auth.authorization_id
        from aadistill.evaluation import eval_registry as ER
        from shared import eval_sets as ES
        es_identity = ER.current_identity(ER.entry(ES.registry(), SET_ID))
        if es_identity != prereg["evaluation"]["eval_set_identity"]:
            raise RuntimeError(f"{SET_ID} identity {es_identity} != preregistered "
                               f"{prereg['evaluation']['eval_set_identity']}")
        ES.manifest(SET_ID)                                  # bytes vs identity
        knowledge = ES.unit_path(SET_ID, "stage3_capability_846", "knowledge")
        if subprocess.run(["mountpoint", "-q", args.volume]).returncode != 0:
            raise RuntimeError(f"no network volume mounted at {args.volume}")
        vol.mkdir(parents=True, exist_ok=True)
        finish(st, "A_contract", eval_set_identity=es_identity,
               knowledge_staged=knowledge is not None)

        #: The Top-4 extension (maintainer 2026-10-20, plans/top4_extension.json).
        #: Its stages never fail the preregistered dose: each records its own
        #: outcome, which the closeout reports.
        ext = plan.get("top4_extension") or {}
        record["extension"] = {}

        def extension(name: str, fn) -> None:
            st = stage(name)
            try:
                st.update(fn())
                finish(st, name)
            except Exception as exc:  # noqa: BLE001
                st.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:2000],
                          traceback=traceback.format_exc()[-4000:],
                          seconds=round(time.time() - st.pop("t0"), 1))
                save()
                mark(f"STAGE_FAILED:{name}")
            record["extension"][name] = st["status"]
            save()

        def rebuild() -> dict:
            done = {}
            for t, rel in ext["rebuild"].items():
                rec = out / f"rebuild_{t}.json"
                with (out / "rebuild.log").open("a") as lf:
                    rc = subprocess.run(
                        [TRAIN_PY, "scripts/stages/stage-0/calibration_volume/rebuild_driver.py",
                         "--plan", rel, "--record", str(rec),
                         "--work", str(REPO / "artifacts/stages/stage-0/calibration_volume/rebuild" / t),
                         "--dest", str(vol.parent / t / "finalists")],
                        cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                doc = json.loads(rec.read_text()) if rec.is_file() else {}
                done[t] = {"rc": rc, "status": doc.get("status"),
                           "adopted": [e["state_id"] for e in doc.get("leaves", []) if e.get("adopted")]}
                if rc != 0:
                    raise RuntimeError(f"the rebuild of {t} exited {rc}: {done[t]}")
            return {"rebuilt": done}

        def export(treatments) -> dict:
            from shared.deployment import MAIN_RELAY
            from stages.calibration_volume.finalists import export_candidates
            token = (os.environ.get("HF_TOKEN") or Path("/workspace/hf/token").read_text()).strip()
            man = out / "finalists_manifest.json"
            doc = json.loads(man.read_text()) if man.is_file() else {"candidates": {}, "files": []}
            for t in treatments:
                cands = candidate_dirs(vol.parent / t / "search_unit", vol.parent / t / "finalists",
                                       int(ext["k"]))
                files = export_candidates(cands, study=prereg["study_id"], treatment=t,
                                          repo=MAIN_RELAY, token=token)
                doc["candidates"][t] = [{k: v for k, v in c.items() if k != "dir"} for c in cands]
                doc["files"] = [r for r in doc["files"] if r["treatment"] != t] + files
            for rec in sorted(out.glob("rebuild_*.json")):
                doc.setdefault("rebuild", {})[rec.stem.removeprefix("rebuild_")] = json.loads(rec.read_text())
            man.write_text(json.dumps(doc, indent=1, default=str) + "\n")
            done_unit("finalists", man)
            return {"exported": {t: [c["state_id"] for c in doc["candidates"][t]] for t in treatments},
                    "manifest_sha256": sha(man)}

        # -- V: room on the volume (the plan names the doses to retire) -------
        if plan.get("retire_states_of"):
            st = stage("V_retire")
            done = {}
            for t in plan["retire_states_of"]:
                doc = retire_completed_states(vol.parent, t, keep_k=int(ext.get("k", 1)))
                shutil.copyfile(vol.parent / t / "retired_states.json",
                                out / f"retired_states_{t}.json")
                done[t] = {"retired": len(doc["retired"]), "bytes": doc["retired_bytes"],
                           "kept": doc["kept"]}
            finish(st, "V_retire", doses=done)
        if plan.get("release_relayed_finalists"):
            rel = plan["release_relayed_finalists"]
            st = stage("V_release")
            man = REPO / rel["manifest"]
            if sha(man) != rel["manifest_sha256"]:
                raise RuntimeError(f"{rel['manifest']} does not hash to the plan's "
                                   f"{rel['manifest_sha256'][:12]}")
            from huggingface_hub import HfApi
            from shared.deployment import MAIN_RELAY
            token = (os.environ.get("HF_TOKEN") or Path("/workspace/hf/token").read_text()).strip()
            done = {}
            for t in rel["treatments"]:
                doc = release_relayed_finalists(vol.parent, t, json.loads(man.read_text()),
                                                api=HfApi(token=token), repo=MAIN_RELAY,
                                                intent=out / f"released_finalists_{t}.json")
                done[t] = {"released": [r["state_id"] for r in doc["released"]],
                           "bytes": doc["released_bytes"]}
            finish(st, "V_release", doses=done)
        if plan.get("retire_optimizer_states"):
            st = stage("V_optim")
            done = {}
            for t in plan["retire_optimizer_states"]:
                doc = retire_optimizer_states(vol.parent, t,
                                              intent=out / f"retired_optimizer_states_{t}.json")
                done[t] = {"retired": [r["path"] for r in doc["retired"]],
                           "bytes": doc["retired_bytes"]}
            finish(st, "V_optim", doses=done)
        if plan.get("volume_room_gb"):
            st = stage("V_room")
            room = volume_room(Path(args.volume), plan["hardware"]["network_volume"]["size_gb"], vol)
            if room["free_bytes"] < plan["volume_room_gb"] * 1e9:
                raise RuntimeError(f"the volume has {room['free_bytes'] / 1e9:.1f} GB free; this "
                                   f"session needs {plan['volume_room_gb']} GB ({room})")
            finish(st, "V_room", **room)

        # -- B, X: the earlier doses' finalists, rebuilt where retired, then shipped
        if ext.get("rebuild"):
            extension("B_rebuild", rebuild)
        if ext.get("export_prior"):
            extension("X_export_prior", lambda: export(ext["export_prior"]))

        # -- S: search (resumable) ------------------------------------------
        products = vol / "products.json"
        prod, torn = read_products(products)
        st = stage("S_search")
        if torn:
            st["products_set_aside"] = torn
        if prod.get("search", {}).get("complete"):
            #: P8.4: a finished search is records plus ONE consumed checkpoint
            #: (the Top-1, and only while it is untrained). The other ~90 states
            #: stay on the volume.
            search_unit.mkdir(parents=True, exist_ok=True)
            for f in (vol / "search_unit").glob("*.json"):
                shutil.copyfile(f, search_unit / f.name)
            pulled = {"records": sorted(f.name for f in search_unit.glob("*.json"))}
            if sha(search_unit / "search_record.json") != prod["search"]["search_record_sha256"]:
                raise RuntimeError("the search record on the volume is not the one the product "
                                   "record names")
            if not prod.get("recovery", {}).get("complete"):
                sel = json.loads((search_unit / "stage1_selection.json").read_text())
                ck = Path(sel["selected"][0]["checkpoint_path"])
                pulled["top1"] = sync("pull", vol / "search_unit" / ck.relative_to(search_unit),
                                      ck, out / "sync.log")
            finish(st, "S_search", resumed_complete=True, pulled=pulled)
        else:
            pulled = sync("pull", vol / "search_unit", search_unit, out / "sync.log")
            stop = out / "mirror.stop"
            stop.unlink(missing_ok=True)
            mirror = subprocess.Popen(
                [sys.executable, str(REPO / "scripts/stages/stage-0/calibration_volume/volume_sync.py"),
                 "push", "--src", str(search_unit), "--dst", str(vol / "search_unit"),
                 "--loop", str(plan.get("mirror_seconds", 120)), "--until", str(stop),
                 "--log", str(out / "mirror.log")], cwd=REPO, env=env)
            cmd = [TRAIN_PY, "scripts/stages/stage-0/calibration_volume/search_driver.py",
                   "--treatment", args.treatment, "--run-id", args.run_id,
                   "--search-run-id", treat["search_run_id"], "--auth", args.auth,
                   "--auth-kind", "dose",
                   "--plan", args.plan, "--treatments-doc", PREREG,
                   "--profiles-doc", prereg["profiles_doc"], "--status", args.status,
                   *(["--prefix-reuse"] if prereg["execution"]["depth_prefix_reuse"] else []),
                   *(["--validate-on-device"] if prereg["execution"]["validate_on_search_device"] else []),
                   "--keep-pruned-weights"]
            with (out / "search.log").open("a") as lf:
                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf,
                                    stderr=subprocess.STDOUT).returncode
            stop.write_text("stop\n")
            mirror.wait(timeout=3600)
            final = require_pushed(sync("push", search_unit, vol / "search_unit", out / "sync.log"),
                                   "the search unit (the volume's journal stops at the last "
                                   "good mirror pass)")
            srec = json.loads((search_unit / "search_record.json").read_text())
            if rc != 0 or srec.get("status") != "COMPLETE":
                raise RuntimeError(f"search_driver exited {rc} with status {srec.get('status')}; "
                                   "the journal and states are on the volume for a resume")
            prod["search"] = {"complete": True,
                              "search_record_sha256": sha(search_unit / "search_record.json")}
            write_atomic(products, json.dumps(prod, indent=1) + "\n")
            finish(st, "S_search", pulled_before=pulled, final_push=final,
                   n_resumed=srec["stages"]["C_beam_search"].get("n_resumed"))
        #: a unit restored from the volume was secured by the session that made
        #: it; marking it here sends the launcher after files it never pulled
        if not st.get("resumed_complete"):
            done_unit("search", search_unit / "search_record.json")
        srec = json.loads((search_unit / "search_record.json").read_text())
        record["search"] = {"commit": srec.get("commit"), "session": srec.get("session")}
        save()
        if ext.get("export_own"):
            extension("X_export", lambda: export([args.treatment]))

        # -- R: recovery (resumable) ------------------------------------------
        st = stage("R_recovery")
        selection = json.loads((search_unit / "stage1_selection.json").read_text())
        top1 = selection["selected"][0]
        top1_dir = Path(top1["checkpoint_path"])
        model_vol = vol / "recovery"
        rec_out = REPO / "artifacts/stages/stage-0/calibration_volume/formal" / args.treatment / "recovery"
        if prod.get("recovery", {}).get("complete"):
            sync("pull", model_vol, rec_out, out / "sync.log")
            from stages.phase_c1.autoinit_c1_driver import trained_model_dir
            model_dir = trained_model_dir(rec_out)
            if sha(model_dir / "model.safetensors") != prod["recovery"]["model_sha256"]:
                raise RuntimeError("the recovered student on the volume does not hash to its record")
            finish(st, "R_recovery", resumed_complete=True, model_dir=str(model_dir))
        else:
            from stages.phase_c1.autoinit_c1_driver import FROZEN_RECIPE
            if json.loads(FROZEN_RECIPE.read_text())["seed"] != prereg["recovery"]["seed"]:
                raise RuntimeError("the frozen recipe's seed is not the preregistered seed")
            cfg = out / "recovery_config.json"
            model_dir = train_recovery(
                top1_dir, rec_out, cfg, out / "recovery.log", env=env,
                run_name=f"s0v_formal_{args.treatment}", seed=prereg["recovery"]["seed"],
                purpose=(f"Stage-0 volume formal dose {args.treatment}: the committed "
                         "Top-1 recovered by the frozen recipe at its own seed; only run "
                         "identity, pack path and student_path differ from it"))
            msha = sha(model_dir / "model.safetensors")
            pushed = require_pushed(sync("push", rec_out, model_vol, out / "sync.log"),
                                    "the recovered student")
            if (rec_out / "train_log.jsonl").is_file():
                shutil.copyfile(rec_out / "train_log.jsonl", out / "train_log.jsonl")
            prod["recovery"] = {"complete": True, "model_sha256": msha,
                                "model_bytes": (model_dir / "model.safetensors").stat().st_size,
                                "config_sha256": sha(cfg), "top1_state": top1.get("state_id")}
            write_atomic(products, json.dumps(prod, indent=1) + "\n")
            finish(st, "R_recovery", model_dir=str(model_dir), model_sha256=msha, pushed=pushed,
                   top1=top1, config=str(cfg.relative_to(REPO)))
        (out / "recovery_products.json").write_text(json.dumps(prod.get("recovery"), indent=1) + "\n")
        if not st.get("resumed_complete"):
            done_unit("recovery", out / "recovery_products.json")

        # -- E: eval-set-1, every item --------------------------------------------
        st = stage("E_eval")
        from stages.phase_c1.autoinit_c1_driver import trained_model_dir
        rep, runs = evaluate_all(trained_model_dir(rec_out), out, vol / "generations",
                                 f"s0v_formal_{args.treatment}", prereg, env)
        (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
        sync("push", out, vol / "dose_out", out / "sync.log")
        finish(st, "E_eval", generated=runs,
               fluid_formal=rep["units"]["fluid_reasoning_720/dev_v1"].get("correct"),
               report_sha256=sha(out / "report.json"))
        done_unit("eval", out / "report.json")
        record["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        try:
            sync("push", out, vol / "dose_out", out / "sync.log")
        except Exception:  # noqa: BLE001
            pass
        mark("ALL_DONE" if record["status"] == "COMPLETE" else "RUN_FAILED")
    return 0 if record["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
