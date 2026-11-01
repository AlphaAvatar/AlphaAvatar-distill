"""Teacher-student lineage: one teacher identity from a student's initialization root to its online KD teacher.

A student is built from its own teacher's weights and distilled by that same teacher. A training config that
carries a ``lineage`` block declares that one teacher and what must descend from it:

    "lineage": {
      "teacher": {"model_id": ..., "revision": ..., "weights_sha256": {shard: sha256}, "tokenizer_sha256": ...,
                  "chat_template_sha256": ..., "rope_theta": ..., "max_position_embeddings": ..., "vocab_size": ...},
      "mode": <the teacher mode the targets and KD use, or null>,
      "tokenizer": {"files_sha256": {name: sha256}},
      "initialization": {"root_teacher": {"model_id": ..., "revision": ...}, "model_safetensors_sha256": ...,
                         "provenance": {"record": <the search record that produced it>, "record_sha256": ...}},
      "data": [{"files_sha256": {name: sha256}, "teacher": {"model_id": ..., "revision": ...}, "mode": ...}]
    }

`problems` compares the declaration with what a caller OBSERVED -- the config's KD teacher, the loaded teacher's
identity and snapshot, the student's EFFECTIVE config and weight bytes, the initialization's provenance as its
search recorded it, the tokenizer files, the pack's bytes, a resumed checkpoint's lineage stamp -- and returns
every mismatch; `require` raises on any. A fact the caller did not observe is not checked, so each layer (config
validation, launch admission, the trainer at runtime) passes what it can see; a fact it did observe that leaves a
declared field unstated is a problem, never a match. Nothing here names a model, a revision or a path: every
instance comes from the caller (P3).

The tokenizer is bound by its FILES, not only its vocabulary: two teachers can share a vocabulary (and its hash)
and differ in their chat template, which lives in the tokenizer files. The initialization is bound by what the
search that produced it RECORDED -- its root teacher's id and shard-set hash and the checkpoint's sha256 --
rather than by the model id a config declares for it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

IDENTITY_KEYS = ("model_id", "revision")
#: what tells two teachers of one vocabulary and geometry apart, and must descend to the student unchanged
STRUCTURAL_KEYS = ("rope_theta", "max_position_embeddings", "vocab_size")


class LineageError(ValueError):
    """The run's teacher lineage is not one lineage."""


def _ident(x: Mapping | None) -> tuple:
    return tuple((x or {}).get(k) for k in IDENTITY_KEYS)


def rope_theta(cfg: Mapping) -> float | None:
    """A model config's RoPE base, whichever spelling the transformers version wrote."""
    v = cfg.get("rope_theta")
    if v is None:
        v = (cfg.get("rope_parameters") or {}).get("rope_theta")
    return float(v) if v is not None else None


def shard_set_sha256(weights_sha256: Mapping[str, str]) -> str:
    """One hash over a teacher's weight shards: sha256 of ``"<shard>:<sha256>\n"`` in shard-name order -- the
    root-teacher hash an initialization search records beside the root's id."""
    return hashlib.sha256("".join(f"{k}:{weights_sha256[k]}\n" for k in sorted(weights_sha256)).encode()).hexdigest()


def stamp(lineage: Mapping, *, initialization_sha256: str | None) -> dict:
    """What a checkpoint written under ``lineage`` carries, so a resume can prove its descent: the lineage's digest
    and the initialization bytes the run verified when it started (carried forward unchanged by every resume)."""
    return {"lineage_sha256": lineage_sha256(lineage), "teacher": dict(zip(IDENTITY_KEYS, _ident(lineage.get("teacher")))),
            "mode": lineage.get("mode"), "initialization_sha256": initialization_sha256}


def lineage_sha256(lineage: Mapping) -> str:
    return hashlib.sha256(json.dumps(lineage, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def declared(lineage: Mapping) -> list[str]:
    """Problems with the declaration itself: it must name one teacher and bind each descendant to it."""
    bad = []
    t = lineage.get("teacher") or {}
    if not all(t.get(k) for k in IDENTITY_KEYS):
        bad.append("lineage.teacher must name model_id and revision")
    for k in STRUCTURAL_KEYS:
        if t.get(k) is None:
            bad.append(f"lineage.teacher must state {k}: an unstated field cannot be compared")
    if not (lineage.get("tokenizer") or {}).get("files_sha256"):
        bad.append("lineage.tokenizer must bind the tokenizer's files by sha256")
    init = lineage.get("initialization") or {}
    if _ident(init.get("root_teacher")) != _ident(t):
        bad.append(f"the initialization's root teacher {_ident(init.get('root_teacher'))} is not the lineage's "
                   f"teacher {_ident(t)}")
    if not init.get("model_safetensors_sha256"):
        bad.append("lineage.initialization must bind the student initialization's weights by sha256")
    prov = init.get("provenance") or {}
    if not (prov.get("record") and prov.get("record_sha256")):
        bad.append("lineage.initialization.provenance must name the search record that produced it, by sha256")
    data = lineage.get("data")
    if not data:
        bad.append("lineage.data must bind every training pack")
    for k, d in enumerate(data or []):
        if _ident(d.get("teacher")) != _ident(t):
            bad.append(f"pack {k}: generated by {_ident(d.get('teacher'))}, not the lineage's teacher {_ident(t)}")
        if lineage.get("mode") is not None and d.get("mode") != lineage.get("mode"):
            bad.append(f"pack {k}: mode {d.get('mode')!r}, the lineage's is {lineage.get('mode')!r}")
        if not d.get("files_sha256"):
            bad.append(f"pack {k}: no files_sha256")
    return bad


def problems(lineage: Mapping, *, kd_teacher: Mapping | None = None, loaded_teacher: Mapping | None = None,
             teacher_snapshot: Mapping | None = None, student_config: Mapping | None = None,
             student_weights_sha256: str | None = None, pack_files_sha256: Mapping | None = None,
             tokenizer_sha256: str | None = None, tokenizer_files_sha256: Mapping | None = None,
             initialization_provenance: Mapping | None = None,
             checkpoint_lineage: Mapping | None = None) -> list[str]:
    """Every way the observed facts disagree with the lineage (empty: one lineage).

    kd_teacher: the config's teacher block; loaded_teacher: the loader's identity record (model_id, revision,
    tokenizer_sha256, vocab_size); teacher_snapshot: {"weights_sha256": {shard: sha}, "chat_template_sha256",
    "rope_theta", "max_position_embeddings"} read from the snapshot actually loaded; student_config: the
    student's EFFECTIVE config (`effective_config`, or the loaded model's ``config.to_dict()``) -- a raw
    config.json that omits a field is read by the loader with its library default; student_weights_sha256: the
    initialization's model.safetensors sha256; pack_files_sha256: the training pack's files; tokenizer_sha256:
    the vocabulary hash of the tokenizer the student trains with; tokenizer_files_sha256: that tokenizer source's
    files; initialization_provenance: {"root_teacher_id", "root_teacher_sha256", "checkpoint_sha256"} as the
    search record names them; checkpoint_lineage: a resumed checkpoint's `stamp` ({} when it carries none)."""
    bad = declared(lineage)
    t = lineage.get("teacher") or {}
    if kd_teacher is not None and _ident(kd_teacher) != _ident(t):
        bad.append(f"the online KD teacher {_ident(kd_teacher)} is not the lineage's teacher {_ident(t)}")
    if loaded_teacher is not None:
        if _ident(loaded_teacher) != _ident(t):
            bad.append(f"the loaded teacher {_ident(loaded_teacher)} is not the lineage's teacher {_ident(t)}")
        for k in ("tokenizer_sha256", "vocab_size"):
            if t.get(k) is not None and loaded_teacher.get(k) is not None and loaded_teacher[k] != t[k]:
                bad.append(f"the loaded teacher's {k} {loaded_teacher[k]} is not the lineage's {t[k]}")
    if teacher_snapshot is not None:
        want = t.get("weights_sha256") or {}
        got = teacher_snapshot.get("weights_sha256") or {}
        if want and got != want:
            bad.append(f"the teacher's weight shards {sorted(got.items())[:3]} are not the lineage's "
                       f"{sorted(want.items())[:3]}")
        for k in ("chat_template_sha256", "rope_theta", "max_position_embeddings"):
            if t.get(k) is not None and teacher_snapshot.get(k) is not None and teacher_snapshot[k] != t[k]:
                bad.append(f"the teacher snapshot's {k} {teacher_snapshot[k]} is not the lineage's {t[k]}")
    if student_config is not None:
        for k, got in (("rope_theta", rope_theta(student_config)),
                       ("max_position_embeddings", student_config.get("max_position_embeddings")),
                       ("vocab_size", student_config.get("vocab_size"))):
            if t.get(k) is None:
                continue
            if got is None:
                bad.append(f"the student's config states no {k}; its lineage teacher's is {t[k]}: a missing field "
                           "is not a match")
            elif float(got) != float(t[k]):
                bad.append(f"the student's {k} {got} is not its lineage teacher's {t[k]}: a student of another root")
    init = lineage.get("initialization") or {}
    if student_weights_sha256 is not None and student_weights_sha256 != init.get("model_safetensors_sha256"):
        bad.append(f"the student initialization's weights {student_weights_sha256[:12]} are not the lineage's "
                   f"{str(init.get('model_safetensors_sha256'))[:12]}")
    if initialization_provenance is not None:
        prov = initialization_provenance
        want_id = "@".join(str(x) for x in _ident(t))
        if prov.get("root_teacher_id") != want_id:
            bad.append(f"the initialization's search recorded root {prov.get('root_teacher_id')}, not the lineage's "
                       f"teacher {want_id}")
        shards = t.get("weights_sha256") or {}
        if not shards or prov.get("root_teacher_sha256") != shard_set_sha256(shards):
            bad.append(f"the initialization's search recorded root shards {str(prov.get('root_teacher_sha256'))[:12]}, "
                       "not the lineage teacher's weight shards")
        if prov.get("checkpoint_sha256") != init.get("model_safetensors_sha256"):
            bad.append(f"the search recorded checkpoint {str(prov.get('checkpoint_sha256'))[:12]}, not the "
                       f"lineage's initialization {str(init.get('model_safetensors_sha256'))[:12]}")
    if tokenizer_files_sha256 is not None:
        want = dict((lineage.get("tokenizer") or {}).get("files_sha256") or {})
        if dict(tokenizer_files_sha256) != want:
            diff = sorted(k for k in set(want) | set(tokenizer_files_sha256)
                          if want.get(k) != dict(tokenizer_files_sha256).get(k))
            bad.append(f"the tokenizer source's files differ from the lineage's in {diff}")
    if checkpoint_lineage is not None:
        ck = checkpoint_lineage
        if not ck:
            bad.append("the resumed checkpoint carries no lineage stamp: its descent cannot be shown")
        else:
            if ck.get("lineage_sha256") != lineage_sha256(lineage):
                bad.append(f"the resumed checkpoint was written under lineage {str(ck.get('lineage_sha256'))[:12]}, "
                           f"not this one {lineage_sha256(lineage)[:12]}")
            if ck.get("initialization_sha256") != init.get("model_safetensors_sha256"):
                bad.append(f"the resumed checkpoint's run verified initialization "
                           f"{str(ck.get('initialization_sha256'))[:12]}, not the lineage's "
                           f"{str(init.get('model_safetensors_sha256'))[:12]}")
    if pack_files_sha256 is not None:
        known = [dict(d.get("files_sha256") or {}) for d in lineage.get("data") or []]
        if dict(pack_files_sha256) not in known:
            bad.append("the training pack's files are not a pack the lineage binds")
    if tokenizer_sha256 is not None and t.get("tokenizer_sha256") is not None and tokenizer_sha256 != t["tokenizer_sha256"]:
        bad.append(f"the training tokenizer {tokenizer_sha256[:12]} is not the lineage teacher's "
                   f"{t['tokenizer_sha256'][:12]}")
    return bad


def effective_config(model_dir) -> dict:
    """The config a loader actually builds for ``model_dir`` (library defaults filled in), as a dict."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(str(model_dir)).to_dict()


def tokenizer_files(directory, names: Iterable[str]) -> dict[str, str]:
    """sha256 of each named file of a tokenizer source ("missing" for an absent one, so it cannot match)."""
    from pathlib import Path
    d = Path(directory)
    return {n: (hashlib.sha256((d / n).read_bytes()).hexdigest() if (d / n).is_file() else "missing") for n in names}


def require(lineage: Mapping, **observed) -> None:
    """`problems`, fail closed."""
    bad = problems(lineage, **observed)
    if bad:
        raise LineageError("teacher-student lineage refused: " + "; ".join(bad))


def require_all(lineages: Iterable[tuple[str, Mapping, Mapping]]) -> None:
    """Many runs at once (a launch's units): every (name, lineage, observed) must pass; all problems are reported."""
    bad = [f"{name}: {p}" for name, lin, obs in lineages for p in problems(lin, **obs)]
    if bad:
        raise LineageError("teacher-student lineage refused: " + "; ".join(bad))


def snapshot_facts(snapshot_dir) -> dict:
    """What a teacher snapshot directory IS: each weight shard's sha256 (an LFS blob's name when the file links to
    one, else its bytes hashed), sha256 of tokenizer_config's chat_template, the RoPE base and position limit."""
    import re
    from pathlib import Path
    d = Path(snapshot_dir)
    shards = {}
    for p in sorted(d.glob("*.safetensors")):
        name = Path(p.resolve()).name if p.is_symlink() else ""
        if re.fullmatch(r"[0-9a-f]{64}", name):
            shards[p.name] = name
        else:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            shards[p.name] = h.hexdigest()
    cfg = json.loads((d / "config.json").read_text())
    tc = json.loads((d / "tokenizer_config.json").read_text()) if (d / "tokenizer_config.json").is_file() else {}
    tmpl = tc.get("chat_template")
    if tmpl is None and (d / "chat_template.jinja").is_file():
        tmpl = (d / "chat_template.jinja").read_text()
    return {"weights_sha256": shards, "rope_theta": rope_theta(cfg),
            "max_position_embeddings": cfg.get("max_position_embeddings"),
            "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest() if isinstance(tmpl, str) else None}
