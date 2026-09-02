"""Package a trained C1 probe for evaluation without mutating it.

The frozen generation protocol declares `tokenizer_source = "the evaluated
checkpoint"`, and `generation_runtime_comparability@v2` makes that field
material. So passing `--tokenizer` to the evaluator would not be an
infrastructure detail — it would rewrite a declared field of the protocol and
make every probe incomparable.

Phase A satisfied the rule by copying the tokenizer sidecars *into* the trained
checkpoint. That works and it mutates the scientific artifact: a checkpoint's
`artifact_digest` folds in `tokenizer_sha256`, so the thing evaluated is no
longer byte-identical to the thing trained, and which bytes a probe was scored
against becomes unrecoverable if the copy is ever wrong.

C1 keeps the rule true a different way. The **evaluation package** is a separate
directory holding the trained model files and the frozen tokenizer sidecars; it
*is* "the evaluated checkpoint" as far as the evaluator is concerned, so
`tokenizer_source` stays exactly what Stage 0 attested. The training checkpoint
is never written to, and this module proves that rather than promising it: the
model directory is hashed before and after, and a single changed byte is an
error.

Model files are hard-linked when the filesystem allows it, so a package costs
inodes rather than a second copy of the weights. `uncapped_eval.py` is untouched.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_file


class C1PackagingError(RuntimeError):
    """The evaluation package cannot be built, or the checkpoint was touched."""


#: Files a tokenizer source must supply for the evaluator to load one. Presence
#: is decided by looking, never by calling the loader and seeing whether it
#: throws — `AutoTokenizer.from_pretrained` on a model-only directory returns a
#: ONE-TOKEN vocabulary instead of raising, which cost Phase-A attempt 11 a probe.
EVAL_TOKENIZER_SIDECARS = ("tokenizer.json", "tokenizer_config.json",
                           "chat_template.jinja")


def _listing(d: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file()}


def build_evaluation_package(model_dir: str | Path, *,
                             tokenizer_source: str | Path,
                             dest: str | Path,
                             expected_sidecar_sha256: Mapping[str, str],
                             ) -> dict[str, Any]:
    """A directory the evaluator may treat as "the evaluated checkpoint".

    Fail-closed in both directions: a sidecar whose bytes are not the pinned ones
    is refused before anything is linked, and the training checkpoint's listing
    must be byte-identical afterwards.
    """
    model_dir, src, dest = Path(model_dir), Path(tokenizer_source), Path(dest)
    if not model_dir.is_dir():
        raise C1PackagingError(f"{model_dir} is not a directory")
    if not (model_dir / "config.json").is_file():
        raise C1PackagingError(
            f"{model_dir} has no config.json; it is not a checkpoint")

    #: Check the sidecars BEFORE touching anything, so a bad tokenizer never
    #: produces a half-built package that a later step might use.
    landed = []
    for name in EVAL_TOKENIZER_SIDECARS:
        want = expected_sidecar_sha256.get(name)
        if not want:
            raise C1PackagingError(
                f"no expected sha256 declared for {name}; a source that loads is "
                "not thereby the right source")
        p = src / name
        if not p.is_file():
            raise C1PackagingError(
                f"the frozen evaluation tokenizer is incomplete: {p} is missing")
        got = sha256_file(p)
        if got != want:
            raise C1PackagingError(
                f"{p} hashes to {got} but the protocol pins {want}")
        landed.append({"file": name, "sha256": got})

    before = _listing(model_dir)
    carried = [n for n in EVAL_TOKENIZER_SIDECARS if n in before]
    if carried:
        raise C1PackagingError(
            f"{model_dir} already carries {carried}. The trainer writes no "
            "tokenizer, so these came from somewhere else and which bytes the "
            "probe would be scored against is ambiguous; refusing.")

    dest.mkdir(parents=True, exist_ok=True)
    linked = []
    for p in sorted(model_dir.iterdir()):
        if not p.is_file():
            continue
        target = dest / p.name
        if target.exists():
            target.unlink()
        try:
            os.link(p, target)
            how = "hardlink"
        except OSError:
            shutil.copyfile(p, target)
            how = "copy"
        linked.append({"file": p.name, "how": how})
    for name in EVAL_TOKENIZER_SIDECARS:
        shutil.copyfile(src / name, dest / name)

    after = _listing(model_dir)
    if after != before:
        moved = sorted(k for k in set(before) | set(after)
                       if before.get(k) != after.get(k))
        raise C1PackagingError(
            f"the training checkpoint {model_dir} changed while packaging: "
            f"{moved}. The scientific artifact must be identical to the one that "
            "was trained.")

    package = _listing(dest)
    for entry in landed:
        if package.get(entry["file"]) != entry["sha256"]:
            raise C1PackagingError(
                f"{dest / entry['file']} does not carry the pinned bytes after "
                "packaging")
    return {
        "schema": "aadistill.autoinit.c1_evaluation_package/v1",
        "package": str(dest),
        "model_dir": str(model_dir),
        "tokenizer_source": str(src),
        "tokenizer_sidecars": landed,
        "model_files": linked,
        "package_listing": package,
        "checkpoint_unmodified": True,
        "checkpoint_listing_sha256": before,
        "tokenizer_source_rule": (
            "the evaluated checkpoint — satisfied by evaluating THIS package, "
            "not by copying tokenizer files into the training checkpoint"),
    }
