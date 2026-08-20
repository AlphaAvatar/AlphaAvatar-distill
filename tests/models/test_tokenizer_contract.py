"""The tokenizer a recovery run uses is declared, not inferred.

Phase-A attempt 11 reached Stage 2 after a successful 180-minute search and lost
the probe to this. A searched leaf is a model artifact: `save_pretrained()`
writes weights and config and no tokenizer files, and that is correct — the
search consumes pre-tokenized items, and `artifact_digest` folds in
`tokenizer_sha256` only when one is present, so adding tokenizer files to an
already-measured leaf would move the identity its search metrics hang on.

What was wrong is the consumer inferring its tokenizer from `student_path`.
`AutoTokenizer.from_pretrained` on a model-only directory **does not raise**: it
returns a one-token vocabulary. These tests pin the repaired contract and, in
`test_the_silent_one_token_fallback_is_real`, the library behaviour that makes it
necessary — because if transformers ever starts raising there, the reason for
this whole module changes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.models.teacher import tokenizer_hash  # noqa: E402
from aadistill.models.tokenizer_contract import (  # noqa: E402
    TokenizerContractError, carries_tokenizer_files, resolve_training_tokenizer,
)

CANONICAL = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
#: The identity the frozen recovery protocol pins
#: (`compare_recovery_fingerprints.phase_a_protocol`).
FROZEN_SHA = "7781771acc3798ee454c1253c751f930eb1c18c1c3df62e2552cc6f1d394f654"

pytestmark = pytest.mark.skipif(
    not (CANONICAL / "tokenizer.json").is_file(),
    reason="the canonical initialization is not staged here")


@pytest.fixture(scope="module")
def model_only_leaf(tmp_path_factory) -> Path:
    """Exactly what `Qwen3Adapter.save()` leaves behind: config, no tokenizer."""
    d = tmp_path_factory.mktemp("leaf")
    for name in ("config.json", "generation_config.json"):
        if (CANONICAL / name).is_file():
            shutil.copy(CANONICAL / name, d / name)
    assert not carries_tokenizer_files(d)
    return d


# --- the five required cases ------------------------------------------------

def test_model_only_leaf_with_the_canonical_tokenizer_is_accepted(model_only_leaf):
    tok, report = resolve_training_tokenizer(
        student_path=model_only_leaf, tokenizer_source=CANONICAL,
        expected_sha256=FROZEN_SHA, repo_root=REPO)
    assert report["tokenizer_sha256"] == FROZEN_SHA
    assert report["student_carries_tokenizer_files"] is False
    assert report["student_tokenizer_sha256"] is None
    # And it is a real tokenizer, not the one-token stand-in.
    assert report["vocab_size"] == 151_669
    assert len(tok.get_vocab()) == 151_669


def test_model_only_leaf_with_no_tokenizer_source_is_refused(model_only_leaf):
    """The attempt-11 configuration. It must not reach training."""
    with pytest.raises(TokenizerContractError, match="no tokenizer source"):
        resolve_training_tokenizer(
            student_path=model_only_leaf, tokenizer_source=None,
            expected_sha256=FROZEN_SHA, repo_root=REPO)
    for empty in ("", "   "):
        with pytest.raises(TokenizerContractError, match="no tokenizer source"):
            resolve_training_tokenizer(
                student_path=model_only_leaf, tokenizer_source=empty,
                expected_sha256=FROZEN_SHA, repo_root=REPO)


def test_a_wrong_tokenizer_source_is_refused(model_only_leaf):
    """Loading successfully is not the test; the identity is."""
    with pytest.raises(TokenizerContractError, match="tokenizer identity mismatch"):
        resolve_training_tokenizer(
            student_path=model_only_leaf, tokenizer_source=CANONICAL,
            expected_sha256="0" * 64, repo_root=REPO)


def test_a_checkpoint_carrying_a_conflicting_tokenizer_is_refused(tmp_path):
    """The student may carry a tokenizer — but then it has to agree."""
    from transformers import AutoTokenizer

    student = tmp_path / "student"; student.mkdir()
    for name in ("config.json", "generation_config.json"):
        if (CANONICAL / name).is_file():
            shutil.copy(CANONICAL / name, student / name)
    # A tokenizer that LOADS CLEANLY and differs. Mutating the BPE vocabulary
    # instead would break the merge table, and the contract would then refuse
    # for the wrong reason — "could not be loaded" rather than "disagrees" —
    # which would leave the disagreement path untested.
    other = AutoTokenizer.from_pretrained(CANONICAL)
    other.add_tokens(["<|aad_contract_test_token|>"])
    other.save_pretrained(student)
    assert carries_tokenizer_files(student)
    assert tokenizer_hash(AutoTokenizer.from_pretrained(student)) != FROZEN_SHA

    with pytest.raises(TokenizerContractError, match="carries its own tokenizer"):
        resolve_training_tokenizer(
            student_path=student, tokenizer_source=CANONICAL,
            expected_sha256=FROZEN_SHA, repo_root=REPO)


def test_the_canonical_control_resolves_to_the_same_hash_as_before():
    """The control's student path IS the canonical init and carries its own
    tokenizer. Nothing about this repair may move the identity the frozen
    recovery protocol pins, or every historical comparison is invalidated."""
    from transformers import AutoTokenizer

    direct = tokenizer_hash(AutoTokenizer.from_pretrained(CANONICAL))
    assert direct == FROZEN_SHA, "the canonical tokenizer itself moved"

    tok, report = resolve_training_tokenizer(
        student_path=CANONICAL, tokenizer_source=CANONICAL,
        expected_sha256=FROZEN_SHA, repo_root=REPO)
    assert tokenizer_hash(tok) == FROZEN_SHA
    assert report["student_carries_tokenizer_files"] is True
    assert report["student_tokenizer_sha256"] == FROZEN_SHA


# --- why the contract exists ------------------------------------------------

def test_the_silent_one_token_fallback_is_real(model_only_leaf):
    """The library behaviour this module exists for, pinned.

    If transformers ever starts raising here, this test fails and the module's
    justification has to be re-derived rather than assumed. That is the point:
    the contract rests on a measured fact about a dependency, and the fact is
    checked rather than remembered.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_only_leaf)   # does NOT raise
    assert len(tok.get_vocab()) == 1, (
        "AutoTokenizer no longer returns a one-token vocabulary for a "
        "model-only directory; re-derive the contract's justification")
    assert tokenizer_hash(tok) != FROZEN_SHA


def test_presence_is_decided_by_files_not_by_calling_the_loader(model_only_leaf):
    """`carries_tokenizer_files` must not be implemented as try/except around
    the loader — that is exactly the check attempt 11 proved does not hold."""
    assert carries_tokenizer_files(CANONICAL) is True
    assert carries_tokenizer_files(model_only_leaf) is False
    src = (REPO / "src/aadistill/models/tokenizer_contract.py").read_text()
    body = src[src.index("def carries_tokenizer_files"):src.index("def _load")]
    # The docstring EXPLAINS the trap, so it names `from_pretrained`. Strip it:
    # a source check that matches its own prose proves nothing.
    code = body.split('"""')[-1]
    assert "from_pretrained" not in code and "except" not in code, code


def test_a_missing_expected_hash_is_refused(model_only_leaf):
    """A source with no pinned identity is not a contract."""
    with pytest.raises(TokenizerContractError, match="expected tokenizer sha256"):
        resolve_training_tokenizer(
            student_path=model_only_leaf, tokenizer_source=CANONICAL,
            expected_sha256=None, repo_root=REPO)


def test_a_nonexistent_source_is_refused(model_only_leaf, tmp_path):
    with pytest.raises(TokenizerContractError, match="does not exist"):
        resolve_training_tokenizer(
            student_path=model_only_leaf, tokenizer_source=tmp_path / "nope",
            expected_sha256=FROZEN_SHA, repo_root=REPO)


def test_a_source_directory_without_tokenizer_files_is_refused_before_loading(
        model_only_leaf):
    """Pointing the SOURCE at a model-only directory is the same trap one level
    up, and is refused without ever calling the loader."""
    calls = []
    with pytest.raises(TokenizerContractError, match="holds none of"):
        resolve_training_tokenizer(
            student_path=model_only_leaf, tokenizer_source=model_only_leaf,
            expected_sha256=FROZEN_SHA, repo_root=REPO,
            loader=lambda p: calls.append(p))
    assert not calls, "the loader was called on a directory with no tokenizer files"


# --- the call site, not just the contract -----------------------------------

def test_the_trainer_declares_its_tokenizer_source_and_does_not_infer_it():
    """A contract nothing calls correctly is not a contract.

    Mutating `train_stage3.py` to pass `student_path` as the tokenizer source
    passed every test above — they exercise the function, not its caller. This
    reads the real call site.
    """
    import ast

    src = (REPO / "scripts/training/train_stage3.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "resolve_training_tokenizer"]
    assert len(calls) == 1, f"expected exactly one resolution site, found {len(calls)}"
    kw = {k.arg: k.value for k in calls[0].keywords}

    assert set(kw) >= {"student_path", "tokenizer_source", "expected_sha256"}
    src_arg = ast.unparse(kw["tokenizer_source"])
    assert "student_path" not in src_arg, (
        f"the trainer infers its tokenizer from the student weights: {src_arg}. "
        "That is the attempt-11 defect; the two are separate dependencies.")
    assert "tokenizer_source" in src_arg
    assert "tokenizer_sha256" in ast.unparse(kw["expected_sha256"])


def test_the_tokenizer_is_resolved_before_any_data_is_built():
    """Refusing after the ladder is packed still wastes the pod's time; the
    contract has to fail at the cheapest possible point."""
    src = (REPO / "scripts/training/train_stage3.py").read_text()
    resolve_at = src.index("resolve_training_tokenizer(")
    for later in ("loading {source} from", "data_dir = REPO_ROOT", "load_teacher("):
        assert src.index(later) > resolve_at, (
            f"{later!r} happens before the tokenizer contract is checked")


#: The recovery configs the CURRENT paid path executes. Historical experiment
#: configs are deliberately NOT in this set: their file hashes are pinned by
#: `tests/training/test_e{3,4,8b}_*.py` because reproducing a recorded result
#: means reproducing the config that produced it (AGENTS.md P4). Editing 53 of
#: them to fix a Phase-A bug would have broken that tie — and the pins caught it.
#:
#: A historical config re-run today refuses with the contract's message, which is
#: strictly better than the previous behaviour of silently training against a
#: one-token vocabulary. Adding the two fields is then a one-line, deliberate act
#: by whoever re-runs it, not a silent rewrite of the record.
ACTIVE_RECOVERY_CONFIGS = ("configs/stage3/e1/e1_r0860k_sa_pca.json",)


def test_every_active_recovery_config_declares_the_contract():
    """The trainer refuses without it, so an active config missing it is a
    config that cannot run — better found here than on a pod."""
    missing = []
    for rel in ACTIVE_RECOVERY_CONFIGS:
        cfg = json.loads((REPO / rel).read_text())
        if not cfg.get("tokenizer_source") or not cfg.get("tokenizer_sha256"):
            missing.append(rel)
    assert not missing, f"active recovery configs with no tokenizer contract: {missing}"


def test_the_historical_configs_are_left_alone():
    """Their hashes are pinned to recorded results. If this ever fails, someone
    has rewritten an experiment's config, and the recorded result no longer
    describes the run that produced it."""
    import glob

    edited = []
    for p in sorted(glob.glob(str(REPO / "configs/stage3/*/*.json"))):
        rel = str(Path(p).relative_to(REPO))
        if rel in ACTIVE_RECOVERY_CONFIGS:
            continue
        cfg = json.loads(Path(p).read_text())
        if isinstance(cfg, dict) and "tokenizer_source" in cfg:
            edited.append(rel)
    assert not edited, (
        f"historical configs gained a tokenizer contract: {edited}. Their file "
        "hashes are pinned to recorded results; adding fields breaks the tie "
        "between a result and the config that produced it.")


def test_the_frozen_recipe_pins_the_protocols_tokenizer():
    """`e1_r0860k_sa_pca.json` is what every Phase-A probe derives from, and the
    recovery protocol fingerprint pins `tokenizer_sha256`. If the recipe named a
    different one, probes would be comparable to nothing."""
    cfg = json.loads((REPO / "configs/stage3/e1/e1_r0860k_sa_pca.json").read_text())
    assert cfg["tokenizer_sha256"] == FROZEN_SHA
    assert cfg["tokenizer_source"] == "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    # And it is NOT the student path, which for a searched leaf carries nothing.
    assert cfg["tokenizer_source"] != cfg["student_path"] or (
        REPO / cfg["student_path"] / "tokenizer.json").is_file()
