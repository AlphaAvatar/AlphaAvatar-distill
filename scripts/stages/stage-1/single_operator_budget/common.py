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
