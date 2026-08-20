"""Which tokenizer a training run uses, stated rather than inferred.

Phase-A attempt 11 lost a Stage-2 probe to this, after a 180-minute search had
already succeeded. The chain was:

1. a searched leaf is a **model** artifact: `save_pretrained()` writes weights,
   `config.json` and `generation_config.json`, and no tokenizer files. That is
   correct — the search consumes pre-tokenized calibration items and never needs
   a tokenizer, and its `CheckpointIdentity.artifact_digest` folds in
   `tokenizer_sha256` only when one is present, so adding tokenizer files to an
   already-measured leaf would change the identity its search metrics hang on;
2. the trainer inferred its tokenizer from `student_path`;
3. `AutoTokenizer.from_pretrained()` on such a directory **does not raise**. It
   returns a tokenizer whose vocabulary has **one token**.

Measured on transformers 5.x: vocabulary size 1, hash `42d8c56b2d86cf7b…`
against the teacher's `7781771acc3798ee…`. The only reason attempt 11 stopped
rather than training against a one-token vocabulary is that `train_stage3.py`
compared the teacher's tokenizer to the student's and refused.

So the repair is not to give the producer a tokenizer. It is to stop the
consumer inferring one:

* the tokenizer source is **explicit** and separate from `student_path`;
* it is loaded through the real loader and its identity is **verified against a
  frozen hash**, so a source that loads is not thereby accepted;
* if the student checkpoint happens to carry tokenizer files, they must **agree**;
* every failure refuses **before** training or data construction begins.

Presence is decided by looking for the files, never by calling the loader and
seeing whether it throws — which is precisely the check attempt 11 proved does
not hold.

This module holds no model-specific constant (AGENTS.md P3). The canonical
source and the expected hash are data supplied by the recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .teacher import tokenizer_hash

__all__ = [
    "TokenizerContractError",
    "TOKENIZER_FILES",
    "carries_tokenizer_files",
    "resolve_training_tokenizer",
]


class TokenizerContractError(RuntimeError):
    """The tokenizer a run would use is absent, unverified or contradictory."""


#: What makes a directory a tokenizer source. Any one of these is enough for
#: `AutoTokenizer` to load something real; none of them present is the case that
#: silently yields a one-token vocabulary.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "tokenizer.model", "spiece.model")


def carries_tokenizer_files(path: str | Path) -> bool:
    """Does this directory hold tokenizer assets?

    A file check, deliberately. `try: AutoTokenizer.from_pretrained(p)` is not a
    presence test — attempt 11 established that it succeeds on a directory with
    only `config.json` and hands back a one-token vocabulary.
    """
    d = Path(path)
    return any((d / name).is_file() for name in TOKENIZER_FILES)


def _load(source: Path, loader: Any):
    if loader is None:
        from transformers import AutoTokenizer

        loader = AutoTokenizer.from_pretrained
    try:
        return loader(source)
    except Exception as exc:                       # noqa: BLE001 - reported as-is
        raise TokenizerContractError(
            f"the declared tokenizer source {source} could not be loaded: "
            f"{type(exc).__name__}: {exc}") from exc


def resolve_training_tokenizer(
    *,
    student_path: str | Path,
    tokenizer_source: str | Path | None,
    expected_sha256: str | None,
    repo_root: str | Path = ".",
    loader: Any = None,
):
    """The tokenizer this run will use, or a refusal.

    `student_path` supplies the weights. `tokenizer_source` supplies the
    tokenizer, and the two are allowed to differ: a searched leaf is a model
    artifact and carries no tokenizer by design.

    Refuses when the source is absent, when it does not exist, when its identity
    does not match `expected_sha256`, or when the student checkpoint carries
    tokenizer files that disagree with it. Returns `(tokenizer, report)`.
    """
    root = Path(repo_root)
    student = Path(student_path)
    if not student.is_absolute():
        student = root / student

    if tokenizer_source is None or not str(tokenizer_source).strip():
        raise TokenizerContractError(
            "no tokenizer source declared. The tokenizer is a separate "
            "dependency from the student weights and must be named explicitly: "
            "inferring it from student_path is what cost Phase-A attempt 11 a "
            "Stage-2 probe, because a model-only checkpoint yields a one-token "
            "vocabulary instead of an error.")
    if not str(expected_sha256 or "").strip():
        raise TokenizerContractError(
            "no expected tokenizer sha256 declared. A source that loads is not "
            "thereby the right source; the frozen recovery protocol pins a "
            "tokenizer identity and the run must be checked against it.")

    source = Path(tokenizer_source)
    if not source.is_absolute():
        source = root / source
    if not source.exists():
        raise TokenizerContractError(
            f"the declared tokenizer source {source} does not exist")
    if source.is_dir() and not carries_tokenizer_files(source):
        raise TokenizerContractError(
            f"the declared tokenizer source {source} holds none of "
            f"{list(TOKENIZER_FILES)}. Loading it would not fail — it would "
            "produce a one-token vocabulary — so it is refused here instead.")

    tokenizer = _load(source, loader)
    actual = tokenizer_hash(tokenizer)
    if actual != expected_sha256:
        raise TokenizerContractError(
            f"tokenizer identity mismatch: {source} hashes to {actual} but the "
            f"protocol pins {expected_sha256}. Refusing before training.")

    student_carries = student.is_dir() and carries_tokenizer_files(student)
    student_sha = None
    if student_carries:
        student_sha = tokenizer_hash(_load(student, loader))
        if student_sha != actual:
            raise TokenizerContractError(
                f"the student checkpoint {student} carries its own tokenizer "
                f"hashing to {student_sha}, which disagrees with the declared "
                f"source {source} at {actual}. One of the two is wrong and this "
                "run cannot decide which; refusing before training.")

    return tokenizer, {
        "tokenizer_source": str(source),
        "tokenizer_sha256": actual,
        "expected_sha256": expected_sha256,
        "student_path": str(student),
        "student_carries_tokenizer_files": student_carries,
        "student_tokenizer_sha256": student_sha,
        "vocab_size": len(tokenizer.get_vocab()),
    }
