"""What every SOB1 script reads the same way: the design, the one evaluation protocol (8,192 tokens, both modes) in the
shape `reasoning_inheritance.driver.fluid` / `capability` take it, the evaluation package rule, and where the round's
bytes live (the relay prefix root, the pod's root, the dev box's store). Nothing here runs a model.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DESIGN = REPO / "configs/stages/stage-1/single_operator_budget/design.json"
SELECTION = REPO / "configs/stages/stage-1/single_operator_budget/selection.json"
STRUCTURES = REPO / "logs/stages/stage-1/single_operator_budget/plans/structures.json"
LOGS = REPO / "logs/stages/stage-1/single_operator_budget"


class SOB1Error(RuntimeError):
    pass


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def design() -> dict:
    return read_json(DESIGN)


def storage() -> dict:
    return design()["storage"]


def relay_root() -> str:
    return storage()["relay_prefix_root"]


def pod_root() -> str:
    return storage()["pod_root"]


def local_store() -> Path:
    return Path(os.path.expanduser(storage()["local_store"]))


def protocol() -> dict:
    """The registered evaluation protocol (`design.json :: evaluation.protocol`), one for the untrained arms, the
    teacher and the recovered arms, in the runners' shape."""
    ev = design()["evaluation"]["protocol"]
    return {"eval_set": ev["eval_set"], "decoding": ev["decoding"]["path"], "context": {"tokens": ev["context"]["tokens"]},
            "action_protocol": ev["action_protocol"], "system_message": ev["system_message"],
            "degeneration_check_every": ev["degeneration_check_every"],
            "modes": {m: {"template_kwargs": v["template_kwargs"]} for m, v in ev["modes"].items()},
            "package": ev["package"]["rule"]}


def statistics_input() -> dict:
    """Where the residual-width build reads the sealed teacher statistics on the pod, and the relay object setup
    stages there (`selection.json :: residual_width.statistics`, verified by its sha256)."""
    st = read_json(SELECTION)["residual_width"]["statistics"]
    dest = design()["phase_1"]["inputs"]["teacher_residual_statistics"]["dest"]
    return {"repo": st["relay"]["repo"], "path": st["relay"]["path"], "sha256": st["sha256"], "bytes": st["bytes"],
            "dest": dest, "file": st["file"], "local": REPO / dest / st["file"]}


def mode_dirs() -> dict[str, str]:
    return {m: v["dir"] for m, v in design()["evaluation"]["protocol"]["modes"].items()}


def package(model_dir: Path, dest: Path) -> dict:
    """The evaluated checkpoint: its weights and config linked, and the TEACHER's tokenizer files and generation config
    (`evaluation.protocol.package.sidecars`), each hashed against the teacher identity -- one rule for an untrained
    arm, the teacher itself and a recovered arm, so the three are read through the same tokenizer, template and stop
    ids. A model directory's own copies of those files are never used."""
    from shared.pod.teacher_cache import local_snapshot
    from stages.teacher_parent_initialization.relay import sha256_file
    d = design()
    rule = d["evaluation"]["protocol"]["package"]
    ident = read_json(REPO / d["teacher"]["identity"]["path"])
    snap = Path(local_snapshot(d["teacher"]["model"], d["teacher"]["revision"]))
    model_dir, dest = Path(model_dir), Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    linked = []
    for p in sorted(model_dir.iterdir()):
        if p.is_file() and p.name not in rule["sidecars"]:
            os.symlink(p.resolve(), dest / p.name)
            linked.append(p.name)
    side = {}
    for n in rule["sidecars"]:
        got = sha256_file(snap / n)
        if got != ident["files_sha256"][n]:
            raise SOB1Error(f"the teacher's {n} hashes to {got[:12]}, the identity pins {ident['files_sha256'][n][:12]}")
        shutil.copyfile(snap / n, dest / n)
        side[n] = got
    #: the weights these generations come from, by content (completion.py checks them against the init record or the
    #: teacher identity)
    weights = {n: sha256_file(dest / n) for n in linked if n.endswith(".safetensors")}
    return {"model_dir": str(model_dir), "rule": rule["rule"], "linked": linked, "sidecars": side,
            "teacher_snapshot": str(snap), "weights_sha256": weights}


# ---- an arm's durable products: one manifest for build, upload, restore, takeover and local retention -------------

#: the checkpoint's files every arm saves (`adapter.save`): the weights and the config
INIT_FILES = ("model.safetensors", "config.json")
PRODUCT_GROUPS = ("init", "mapping")


def width_declared(arm: str) -> bool:
    """Whether the design declares a residual-width projection for ``arm`` (its build runs RESIDUAL_WIDTH): only then
    does its mapping own a tensor file."""
    return "RESIDUAL_WIDTH" in design()["arms"][arm]["build"]


def mapping_manifest(saved: dict, directory: Path, arm: str) -> dict:
    """The mapping's files, from `frozen_mapping.save`'s own record: ``mapping.json`` always, and the tensor file
    that record names (``tensor_file``) exactly when it holds a width projection -- which must be exactly when the
    design declares one for ``arm``. Each file is hashed on the disk; the tensor file must hash to what save recorded."""
    from aadistill.initialization.planning import frozen_mapping as FM
    from stages.teacher_parent_initialization.relay import sha256_file
    tf, width = saved.get("tensor_file"), saved.get("width") is not None
    if width != (tf is not None) or width != width_declared(arm):
        raise SOB1Error(f"{arm}: the saved mapping {'holds' if width else 'holds no'} width projection and "
                        f"{'names' if tf else 'names no'} tensor file; the design declares "
                        f"{'one' if width_declared(arm) else 'none'}")
    names = [FM.MAPPING_FILE] + ([tf["name"]] if tf else [])
    missing = [n for n in names if not (Path(directory) / n).is_file()]
    if missing:
        raise SOB1Error(f"{arm}: the saved mapping lacks {missing} in {directory}")
    files = {n: sha256_file(Path(directory) / n) for n in names}
    if tf and files[tf["name"]] != tf["sha256"]:
        raise SOB1Error(f"{arm}: {tf['name']} hashes to {files[tf['name']][:12]}, save recorded {tf['sha256'][:12]}")
    return {"content_digest": saved["content_digest"], "width_projection": width, "files": names,
            "files_sha256": files}


def products(rec: dict) -> dict[str, dict[str, str]]:
    """An arm's durable products by relay group, read from its build or init record: what is uploaded, restored,
    taken over and retained, each relative path with its sha256."""
    return {"init": {f"init/{n}": s for n, s in (rec.get("checkpoint", {}).get("files_sha256") or {}).items()},
            "mapping": {f"mapping/{n}": s for n, s in (rec.get("mapping", {}).get("files_sha256") or {}).items()}}


def required_products(arm: str) -> dict[str, set[str]]:
    """The files each group must hold for ``arm``: the checkpoint's two, ``mapping.json``, and the tensor file only
    where the design declares a width projection."""
    from aadistill.initialization.planning import frozen_mapping as FM
    return {"init": {f"init/{n}" for n in INIT_FILES},
            "mapping": {f"mapping/{FM.MAPPING_FILE}"} | ({f"mapping/{FM.TENSOR_FILE}"} if width_declared(arm) else set())}


def product_problems(rec: dict | None, arm: str) -> dict[str, str]:
    """Why ``rec`` (an arm's init record) does not show every product durable, by ``<arm>/<group>``; empty when it
    does. Each group's manifest must be exactly the required files (`required_products`), the checkpoint's weights
    the record's own shard, and the relay copy verified with exactly that manifest's files and hashes. An empty
    manifest or an empty relay list proves nothing."""
    if rec is None:
        return {f"{arm}/{g}": "no init record" for g in PRODUCT_GROUPS}
    if rec.get("status") != "COMPLETE" or rec.get("arm") != arm:
        return {f"{arm}/{g}": f"the init record is {rec.get('status')!r} for {rec.get('arm')!r}" for g in PRODUCT_GROUPS}
    man, need, bad = products(rec), required_products(arm), {}
    shard = ((rec.get("checkpoint") or {}).get("shards") or [{}])[0].get("sha256")
    for g in PRODUCT_GROUPS:
        relay = (rec.get("relay") or {}).get(g) or {}
        if set(man[g]) != need[g]:
            bad[f"{arm}/{g}"] = f"the manifest names {sorted(man[g])}, the arm needs {sorted(need[g])}"
        elif g == "init" and man[g]["init/model.safetensors"] != shard:
            bad[f"{arm}/{g}"] = "the manifest's weights are not the checkpoint's shard"
        elif relay.get("verified") is not True:
            bad[f"{arm}/{g}"] = "the relay copy is not verified"
        elif relay.get("files_sha256") != man[g]:
            bad[f"{arm}/{g}"] = (f"the relay copy holds {sorted(relay.get('files_sha256') or {})}, not the manifest's "
                                 f"{sorted(man[g])} with its hashes")
    return bad
