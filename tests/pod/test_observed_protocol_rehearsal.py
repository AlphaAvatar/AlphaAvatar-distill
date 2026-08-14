"""The six targeted scenarios: what happens when a run cannot prove what it did.

Companion to `test_autoinit_preflight_rehearsal.py`, which rehearses the *staging*
— that a bad machine never reaches the permanent controls. This file rehearses the
*verification*: that a control which did not run the attested protocol, or a set of
rollouts that cannot establish the protocol they were produced under, is rejected
rather than characterized.

    1. observed Stage-2 recovery protocol mismatch      -> control rejected
    2. missing Stage-2 material protocol field          -> fail closed
    3. observed Stage-3 generation protocol mismatch    -> characterization rejected
    4. missing Stage-3 material generation field        -> fail closed
    5. state_eval frozen identity mismatch              -> blocked before measurement
    6. recovery_search identity / scoring-contract drift -> blocked before it

Scenarios 1 and 2 run the **real trainer** end to end on a toy ladder and then the
**real** `Driver.verify_control` over its artifacts. That matters more than it
looks: the thing being tested is a contract between two programs — the trainer
writes the evidence, the verifier reads it — and a fixture written by the test
would satisfy the reader while proving nothing about the writer. The one field
this offline path cannot exercise is the teacher identity, which comes from a Hub
lookup; scenario 2 uses exactly that gap rather than simulating one, and the
success path injects the teacher block from a real run's manifest, marked where it
happens.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.generation import (  # noqa: E402
    GENERATION_V1_DECLARED,
    ObservedGenerationError,
    declared_generation_protocol,
    generation_runtime_fingerprint,
    observe_generation_protocol,
)
from aadistill.autoinit.recovery import (  # noqa: E402
    ObservedProtocolError,
    RecoveryProtocolFingerprint,
    observe_recovery_protocol,
)

DRIVER_PATH = REPO / "scripts/pod/autoinit_preflight_driver.py"
TYPES = ["alpha", "beta", "gamma"]

#: A teacher identity block of the shape `load_teacher` records. Taken from a
#: real Stage-3 run manifest; the offline trainer run cannot produce one because
#: pinning a revision is a Hub call, and the suite does not download.
TEACHER_BLOCK = {
    "model_id": "Qwen/Qwen3-4B-Thinking-2507",
    "revision": "768f209d9ea81521153ed38c47d515654e938aea",
    "dtype": "bfloat16", "device": "cuda",
    "attn_implementation": "sdpa", "num_parameters": 4022468096,
}


# --- a real run, produced by the real trainer -------------------------------


def write_pack(d: Path, n_blocks=12, block_len=16, rungs=(40, 80)) -> None:
    import numpy as np

    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 64, size=(n_blocks, block_len), dtype=np.int32)
    ce = np.zeros((n_blocks, block_len), dtype=bool)
    ce[:, block_len // 2:] = True
    np.savez_compressed(d / "blocks.npz", input_ids=ids, ce_mask=ce,
                        content_mask=np.ones((n_blocks, block_len), dtype=bool))
    with open(d / "audit.jsonl", "w") as f:
        for i in range(n_blocks):
            f.write(json.dumps({
                "unpadded_length": block_len, "padding_length": 0,
                "terminal_truncated": False,
                "sessions": [{"session_id": f"s{i}", "data_type": TYPES[i % 3],
                              "supervised_retained": 10}]}) + "\n")
    (d / "ladder.json").write_text(json.dumps({
        "block_len": block_len, "n_blocks": n_blocks,
        "declared_mixture": {t: 1 / 3 for t in TYPES},
        "rungs": [{"target_supervised_tokens": r, "reachable": True,
                   "n_blocks": r // 10, "actual_supervised_tokens": r}
                  for r in rungs]}))


def write_student(d: Path) -> None:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, intermediate_size=48,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        tie_word_embeddings=True, max_position_embeddings=128)).float(
    ).save_pretrained(d)


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """Train a toy control with the real `train_stage3.py` and return its tree."""
    tmp = tmp_path_factory.mktemp("observed")
    pack, student = tmp / "pack", tmp / "student"
    out = tmp / "artifacts/stage3/preflight_ctl_toy"
    write_pack(pack)
    write_student(student)
    cfg = {
        "stage": "stage3_recovery", "run_name": "preflight_ctl_toy",
        "student_path": str(student), "teacher": None,
        "data_dir": str(pack), "groups": None, "packing": "ladder", "rung": 40,
        "val_blocks": 2, "block_len": 16, "dtype": "float32",
        "autocast_bf16": False, "gradient_checkpointing": False,
        "device": "cpu", "seed": 20260726, "trainable_patterns": "all",
        "loss": {"ce_weight": 1.0, "kd_weight": 0.0, "kd_temperature": 1.0,
                 "kd_scope": "assistant"},
        "optim": {"lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.95],
                  "eps": 1e-8, "grad_clip": 1.0},
        "schedule": {"total_steps": 3, "warmup_steps": 1, "min_lr_frac": 0.1},
        "batch": {"blocks_per_step": 2, "micro_blocks": 1},
        "checkpoint": {"save_every": 3, "keep_last": 1},
        "intervals": {"log_every": 1, "eval_every": 0, "eval_blocks": 0},
        "out_dir": str(out),
    }
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/training/train_stage3.py"),
         "--config", str(tmp / "cfg.json")],
        capture_output=True, text=True, timeout=900,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
             "HOME": str(tmp), "OMP_NUM_THREADS": "2",
             "AADISTILL_IMAGE_DIGEST": "sha256:rehearsal"})
    assert rc.returncode == 0, (rc.stdout + rc.stderr)[-3000:]
    return {"root": tmp, "out": out, "pack": pack, "student": student}


def with_teacher(run: dict, dest: Path, **manifest_edits) -> Path:
    """Copy a real run and give it the teacher block the offline path cannot."""
    shutil.copytree(run["out"], dest, dirs_exist_ok=True)
    manifest = json.loads((dest / "run_manifest.json").read_text())
    manifest["teacher"] = dict(TEACHER_BLOCK)
    for key, value in manifest_edits.items():
        target, _, leaf = key.rpartition(".")
        node = manifest
        for part in filter(None, target.split(".")):
            node = node[part]
        if value is _DELETE:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    (dest / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


class _Delete:
    pass


_DELETE = _Delete()


# --- scenario 1: observed Stage-2 protocol mismatch -------------------------


def test_1_an_observed_protocol_that_differs_rejects_the_permanent_control(
        real_run, tmp_path):
    """A control whose run differs from the attestation is not a control."""
    run_dir = with_teacher(real_run, tmp_path / "run")
    attested = observe_recovery_protocol(run_dir, repo_root=Path("/"),
                                         strict=True).protocol

    # The same run, one material field different: a learning rate the attested
    # protocol does not describe. Nothing else moves.
    drifted = with_teacher(real_run, tmp_path / "drifted",
                           **{"execution.optimizer_defaults": {
                               "lr": 2e-3, "betas": [0.9, 0.95], "eps": 1e-8,
                               "weight_decay": 0.01},
                              "config": {**json.loads(
                                  (real_run["out"] / "run_manifest.json").read_text()
                              )["config"], "optim": {
                                  "lr": 2e-3, "weight_decay": 0.01,
                                  "betas": [0.9, 0.95], "eps": 1e-8,
                                  "grad_clip": 1.0}}})
    observed = observe_recovery_protocol(drifted, repo_root=Path("/"),
                                         strict=True).protocol
    comparison = observed.compare(attested)
    assert not comparison["protocol_identical"]
    assert [m["field"] for m in comparison["mismatched_fields"]] == ["lr"]

    # And the driver's own Stage-2 verification refuses it.
    record = run_verify_control(real_run, tmp_path, drifted, attested)
    assert record["protocol_verified"] is False
    assert "OBSERVED protocol differs" in record["problem"]
    assert record["control_binding"] is None, (
        "a rejected control was still bound as if it were one")


def test_1b_the_matching_control_verifies_and_binds_three_hashes(real_run, tmp_path):
    """The success path: observed == attested, and the binding is complete."""
    run_dir = with_teacher(real_run, tmp_path / "run")
    attested = observe_recovery_protocol(run_dir, repo_root=Path("/"),
                                         strict=True).protocol
    record = run_verify_control(real_run, tmp_path, run_dir, attested)
    assert record["protocol_verified"] is True, record["problem"]
    binding = record["control_binding"]
    assert binding["observed_protocol_fingerprint"] == attested.fingerprint
    assert len(binding["checkpoint_weights_sha256"]) == 64
    assert len(binding["probe_id"]) == 64
    assert record["observed_vs_attested"]["protocol_identical"] is True
    # The comparison is not a tautology: it compares two independently built
    # objects, and every material field is populated from the run's artifacts.
    assert record["observed_protocol"]["evidence"]["missing_fields"] == []
    assert (record["observed_protocol"]["evidence"]["pack_blocks_sha256_recomputed"]
            == record["observed_protocol"]["evidence"][
                "pack_blocks_sha256_recorded_by_run"])


def run_verify_control(real_run, tmp_path: Path, run_dir: Path,
                       attested: RecoveryProtocolFingerprint) -> dict:
    """Drive the driver's real `verify_control` over a real run tree."""
    import importlib.util

    from aadistill.infrastructure.manifest import sha256_file

    spec = importlib.util.spec_from_file_location("preflight_driver_obs", DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_driver_obs"] = mod
    spec.loader.exec_module(mod)
    name = run_dir.name
    repo = tmp_path / f"repo_{name}"
    (repo / "artifacts/stage3").mkdir(parents=True, exist_ok=True)
    if not (repo / f"artifacts/stage3/{name}").exists():
        shutil.copytree(run_dir, repo / f"artifacts/stage3/{name}")
    mod.REPO = repo
    mod.AUDIT = tmp_path / f"audit_{name}"
    mod.AUDIT.mkdir(parents=True, exist_ok=True)
    mod.PINNED = {**mod.PINNED, "canonical_init_weights": (
        str(real_run["student"]),
        sha256_file(real_run["student"] / "model.safetensors"))}

    driver = mod.Driver.__new__(mod.Driver)
    driver.a = type("A", (), {"control_minutes": 1.0})()
    driver.attested = attested
    return mod.Driver.verify_control(driver, name, 20260726,
                                     tmp_path / "cfg.json", 1.0)


# --- scenario 2: a missing Stage-2 material field ---------------------------


def test_2_a_missing_material_protocol_field_fails_closed(real_run, tmp_path):
    """No evidence is not "the same"; it is a refusal to certify."""
    # The natural case, not a simulated one: an offline run has no teacher
    # identity, because pinning a revision is a Hub call.
    with pytest.raises(ObservedProtocolError) as exc:
        observe_recovery_protocol(real_run["out"], repo_root=Path("/"), strict=True)
    message = str(exc.value)
    for field in ("teacher_id", "teacher_revision", "teacher_dtype", "teacher_attn"):
        assert field in message
    assert "tautology" in message

    # And each of the fields the *trainer* is responsible for recording, one at a
    # time. Every one of these was a value the old forensic helper would have
    # supplied from a default or a constant.
    for path, field in (("execution.kd_chunk", "kd_chunk"),
                        ("execution.optimizer", "optimizer"),
                        ("execution.lr_schedule", "lr_schedule"),
                        ("execution.block_ordering", "block_ordering"),
                        ("execution.resume_semantics", "resume_semantics"),
                        ("execution.trainer_source", "trainer_source_digest"),
                        ("execution.runtime_digest", "runtime_digest"),
                        ("ladder.blocks_sha256", "pack_blocks_sha256")):
        target = with_teacher(real_run, tmp_path / f"missing_{field}",
                              **{path: _DELETE})
        with pytest.raises(ObservedProtocolError, match=field):
            observe_recovery_protocol(target, repo_root=Path("/"), strict=True)
        # Non-strict is the forensic path: it reports the gap as unknown and
        # `compare` can then never call it matched.
        lax = observe_recovery_protocol(target, repo_root=Path("/"), strict=False)
        assert field in lax.protocol.unverifiable
        other = observe_recovery_protocol(
            with_teacher(real_run, tmp_path / f"ref_{field}"),
            repo_root=Path("/"), strict=True).protocol
        assert not lax.protocol.compare(other)["protocol_identical"]


def test_2b_a_run_that_did_not_finish_is_not_a_control(real_run, tmp_path):
    run_dir = with_teacher(real_run, tmp_path / "short")
    completion = json.loads((run_dir / "run_completion.json").read_text())
    completion["final_step"] = 2
    completion["completed_all_steps"] = False
    (run_dir / "run_completion.json").write_text(json.dumps(completion))
    with pytest.raises(ObservedProtocolError, match="stopped at step 2 of 3"):
        observe_recovery_protocol(run_dir, repo_root=Path("/"), strict=True)


def test_2c_the_pack_hash_is_recomputed_not_trusted(real_run, tmp_path):
    """The run's recorded pack hash is checked against the pack on disk."""
    run_dir = with_teacher(real_run, tmp_path / "packswap")
    tampered = tmp_path / "packswap_pack"
    shutil.copytree(real_run["pack"], tampered)
    (tampered / "blocks.npz").write_bytes(
        (tampered / "blocks.npz").read_bytes() + b"\x00")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    manifest["config"]["data_dir"] = str(tampered)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ObservedProtocolError, match="pack on disk hashes to"):
        observe_recovery_protocol(run_dir, repo_root=Path("/"), strict=True)

    # And a pack that is not on this machine cannot be re-hashed at all.
    manifest["config"]["data_dir"] = "/nonexistent/pack"
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ObservedProtocolError, match="cannot be re-hashed"):
        observe_recovery_protocol(run_dir, repo_root=Path("/"), strict=True)


def test_2d_the_strict_path_does_not_inherit_the_permissive_helpers_defaults():
    """`historical_protocol` backfills; the strict path must not, ever."""
    source = (REPO / "src/aadistill/autoinit/recovery.py").read_text()
    reconstruct = source[source.index("def observe_recovery_protocol"):
                         source.index("class RecoveryProbeIdentity")]
    for backfill in ('"AdamW"', '512', 'PACK_BLOCKS_SHA', 'cosine to min_lr_frac'):
        assert backfill not in reconstruct, (
            f"the strict reconstruction contains the literal {backfill!r}; a "
            "value the verifier supplies to itself is not evidence")
    driver = DRIVER_PATH.read_text()
    assert "observe_recovery_protocol(out_dir" in driver
    assert "RecoveryProbeIdentity(\n                    protocol=observed.protocol" in driver, (
        "the probe identity must be built from the OBSERVED protocol; building "
        "it from the attested one compares the attestation with itself")


# --- scenarios 3 and 4: the generation protocol ------------------------------


def summary(**overrides) -> dict:
    """A rollout summary of the shape `uncapped_eval.py` writes."""
    runtime = generation_runtime_fingerprint("sha256:rehearsal")
    declared = GENERATION_V1_DECLARED
    out = {
        "label": "ctl", "prompts": "gsm8k.jsonl", "n_samples": 30,
        "protocol": declared["protocol"], "system_message": declared["system_message"],
        "thinking_mode": declared["thinking_mode"],
        "stop_ids": [151643, 151645],
        "degeneration_stop": declared["degeneration_stop"],
        "tokenizer_source": "/workspace/aad/artifacts/stage3/ctl/checkpoints/step_1/model",
        "tokenizer_source_rule": declared["tokenizer_source"],
        "tokenizer_sha256": "t" * 64,
        "chat_template_sha256": "c" * 64,
        "context_resolution": {
            "resolved_context": 8192, "context_source": "trained_block_len",
            "trained_context": 8192, "context_len_override": None,
            "rule": declared["context_resolution_rule"]},
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                     "detokenize": False,
                     "max_tokens_rule": declared["max_tokens_rule"],
                     "degeneration_check_every": 256},
        "libraries": {"transformers": runtime.transformers_version, "vllm": "0.11.2"},
        "engine": {"vllm_version": "0.11.2", "dtype": declared["dtype"],
                   "gpu_memory_utilization": 0.90, "max_num_seqs": 256,
                   "max_num_batched_tokens": 8192, "enforce_eager": False,
                   "max_model_len": 8192},
        "identity": {
            "generation_source_digest": "g" * 64,
            "generation_source_set_version": 1,
            "degeneration_source_digest": "d" * 64,
            "runtime": runtime.as_dict(), "runtime_digest": runtime.digest,
            "system_injection_rule": declared["system_injection_rule"],
            "stop_id_derivation_rule": declared["stop_id_derivation_rule"],
            "chat_template_kwargs_json": "{}"},
    }
    for key, value in overrides.items():
        target, _, leaf = key.rpartition(".")
        node = out
        for part in filter(None, target.split(".")):
            node = node[part]
        if value is _DELETE:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return out


def attested_generation():
    """The Stage-0 attested fingerprint for the summaries above."""
    return observe_generation_protocol([summary()]).protocol


def test_3_an_observed_generation_mismatch_rejects_the_characterization():
    attested = attested_generation()
    # The engine's scheduler cap moved — a different effective batch, which
    # moves throughput and can move behaviour, and which nothing else records.
    observed = observe_generation_protocol(
        [summary(**{"engine.max_num_seqs": 64})]).protocol
    comparison = observed.compare(attested)
    assert not comparison["identical"]
    assert [m["field"] for m in comparison["mismatched_fields"]] == ["max_num_seqs"]

    for field, override in (("temperature", {"sampling.temperature": 0.7}),
                            ("resolved_context",
                             {"context_resolution.resolved_context": 4096}),
                            ("stop_token_ids", {"stop_ids": [151645]}),
                            ("degeneration_check_every",
                             {"sampling.degeneration_check_every": 64}),
                            ("chat_template_sha256",
                             {"chat_template_sha256": "e" * 64}),
                            ("runtime_digest",
                             {"identity.runtime_digest": "r" * 64})):
        drifted = observe_generation_protocol([summary(**override)]).protocol
        assert [m["field"] for m in drifted.compare(attested)["mismatched_fields"]] \
            == [field], field


def test_3b_sets_of_one_evaluation_must_agree_with_each_other():
    with pytest.raises(ObservedGenerationError, match="not generated under one"):
        observe_generation_protocol([
            summary(prompts="gsm8k.jsonl"),
            summary(prompts="tool.jsonl", **{"engine.max_num_seqs": 64})])


def test_4_a_missing_material_generation_field_fails_closed():
    for path, field in (("sampling.temperature", "temperature"),
                        ("sampling.top_p", "top_p"),
                        ("sampling.top_k", "top_k"),
                        ("sampling.detokenize", "detokenize"),
                        ("tokenizer_sha256", "tokenizer_sha256"),
                        ("engine.max_num_batched_tokens", "max_num_batched_tokens"),
                        ("engine.enforce_eager", "enforce_eager"),
                        ("identity.runtime_digest", "runtime_digest"),
                        ("identity.generation_source_digest",
                         "generation_source_digest"),
                        ("stop_ids", "stop_token_ids"),
                        ("context_resolution.context_source", "context_source")):
        with pytest.raises(ObservedGenerationError, match=field):
            observe_generation_protocol([summary(**{path: _DELETE})])
        # Not merely absent: a null is the same statement, and the same refusal.
        with pytest.raises(ObservedGenerationError, match=field):
            observe_generation_protocol([summary(**{path: None})])


def test_4b_a_missing_field_is_never_taken_from_the_expected_fingerprint():
    """The one substitution that would make every comparison pass."""
    attested = attested_generation()
    lax = observe_generation_protocol(
        [summary(**{"identity.runtime_digest": _DELETE})], strict=False).protocol
    assert lax.runtime_digest is None
    assert "runtime_digest" in lax.unmaterialized_fields()
    comparison = lax.compare(attested)
    assert not comparison["identical"]
    assert not comparison["both_materialized"]
    assert "runtime_digest" in comparison["unmaterialized_fields"]


def test_4c_every_material_field_is_actually_written_by_the_generator():
    """The writer/consumer contract: the paths must exist in what is written.

    A mapping that names a key the generator never writes fails closed on a real
    pod at Stage 3 — after both permanent controls have been paid for.
    """
    from aadistill.autoinit.generation import (
        NULLABLE_SUMMARY_FIELDS, SUMMARY_FIELD_PATHS,
    )

    sys.path.insert(0, str(REPO / "scripts/evaluation"))
    source = (REPO / "scripts/evaluation/uncapped_eval.py").read_text()
    runtime_keys = set(generation_runtime_fingerprint(None).as_dict())

    for field, path in {**SUMMARY_FIELD_PATHS, **NULLABLE_SUMMARY_FIELDS}.items():
        leaf = path.split(".")[-1]
        if path.startswith("identity.runtime."):
            assert leaf in runtime_keys, f"{field}: {leaf} is not a runtime field"
        else:
            assert f'"{leaf}"' in source, (
                f"{field}: uncapped_eval.py writes no {leaf!r} key, so this "
                "mapping cannot be satisfied by a real evaluation")

    # And every field of the fingerprint is covered by the mapping: a field in
    # the identity that nothing observes is a field the observed side invents.
    declared = set(GENERATION_V1_DECLARED)
    covered = set(SUMMARY_FIELD_PATHS) | set(NULLABLE_SUMMARY_FIELDS)
    assert declared - covered == set(), f"unobserved identity fields: {declared - covered}"


def test_4d_the_declared_protocol_and_the_generator_share_one_definition():
    """Two copies of a rule string is the defect this pair of files had."""
    source = (REPO / "scripts/evaluation/uncapped_eval.py").read_text()
    for constant in ("MAX_TOKENS_RULE", "CONTEXT_RESOLUTION_RULE",
                     "SYSTEM_INJECTION_RULE", "STOP_ID_DERIVATION_RULE",
                     "GENERATION_DTYPE", "TOKENIZER_SOURCE_CHECKPOINT"):
        assert constant in source, (
            f"{constant} is restated in uncapped_eval.py rather than imported; "
            "two copies of a protocol string drift, and this pair already had")
    assert '"per sample: context - prompt' not in source
    # The declared fingerprint is unchanged by that refactor.
    assert declared_generation_protocol().fingerprint == (
        "f4ac744867af5818826ab6e8da85213556ef90b1759374211d94854c3c780c73")


# --- scenarios 5 and 6: the frozen assets -----------------------------------


FROZEN_PRESENT = (REPO / "artifacts/stage3/recovery_search_v2/manifest.json").is_file()
frozen_only = pytest.mark.skipif(
    not FROZEN_PRESENT, reason="frozen assets are local artifacts, not tracked in git")


def frozen_repo(tmp_path: Path) -> Path:
    """A minimal repo the frozen-asset verifier can run against."""
    repo = tmp_path / "repo"
    (repo / "artifacts/stage1").mkdir(parents=True)
    (repo / "artifacts/stage3").mkdir(parents=True)
    shutil.copytree(REPO / "artifacts/stage1/state_eval_v1",
                    repo / "artifacts/stage1/state_eval_v1")
    shutil.copytree(REPO / "artifacts/stage3/recovery_search_v2",
                    repo / "artifacts/stage3/recovery_search_v2")
    from aadistill.autoinit.recovery import RECOVERY_SCORING_FILES_V2
    for rel in RECOVERY_SCORING_FILES_V2:
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    return repo


def verify_frozen(repo: Path) -> tuple[int, dict]:
    out = repo / "report.json"
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/autoinit/verify_frozen_assets.py"),
         "--repo", str(repo), "--out", "report.json"],
        capture_output=True, text=True, cwd=REPO, timeout=300,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    return rc.returncode, json.loads(out.read_text())


@frozen_only
def test_5_a_state_eval_identity_mismatch_blocks_before_measurement(tmp_path):
    repo = frozen_repo(tmp_path)
    assert verify_frozen(repo)[0] == 0, "the unmodified assets must verify"

    # One prompt changed. The content hash is over the loaded items, so this is
    # exactly the failure that must not reach a measurement.
    items = repo / "artifacts/stage1/state_eval_v1/items.jsonl"
    lines = items.read_text().splitlines()
    row = json.loads(lines[0])
    row["text" if "text" in row else list(row)[-1]] = "tampered"
    lines[0] = json.dumps(row)
    items.write_text("\n".join(lines) + "\n")
    rc, report = verify_frozen(repo)
    assert rc == 1
    assert any("state_eval_v1.items_sha256" in p for p in report["problems"])
    assert report["passed"] is False


@frozen_only
def test_5b_a_manifest_edited_to_match_itself_is_still_caught(tmp_path):
    """Self-consistency is not identity — the point of the preregistered pins."""
    from aadistill.infrastructure.manifest import sha256_json

    repo = frozen_repo(tmp_path)
    path = repo / "artifacts/stage1/state_eval_v1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["content_sha256"] = "0" * 64
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_json(manifest)
    path.write_text(json.dumps(manifest, indent=2))
    rc, report = verify_frozen(repo)
    assert rc == 1
    problems = " ".join(report["problems"])
    assert "content_sha256" in problems and "manifest_sha256" in problems
    # It is internally consistent and still rejected: the constants are what it
    # is compared against.
    assert report["assets"]["state_eval_v1"]["manifest_self_consistent"] is True


@frozen_only
def test_6_a_recovery_search_or_scoring_drift_blocks_characterization(tmp_path):
    repo = frozen_repo(tmp_path)
    battery = repo / "artifacts/stage3/recovery_search_v2"
    lines = (battery / "gsm8k.jsonl").read_text().splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), "answer": "tampered"})
    (battery / "gsm8k.jsonl").write_text("\n".join(lines) + "\n")
    manifest = json.loads((battery / "manifest.json").read_text())
    manifest["content_sha256"] = "0" * 64
    (battery / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rc, report = verify_frozen(repo)
    assert rc == 1
    assert any("recovery_search_v2.content_sha256" in p for p in report["problems"])

    # Separately: the scoring contract. A changed scorer with untouched prompts
    # is the failure @v1 could not see.
    repo2 = frozen_repo(tmp_path / "b")
    scorer = repo2 / "src/aadistill/evaluation/capability.py"
    scorer.write_text(scorer.read_text() + "\n# behaviour change\n")
    rc, report = verify_frozen(repo2)
    assert rc == 1
    assert any("scoring contract" in p for p in report["problems"])
    assert report["scoring_contract"]["match"] is False


def test_the_driver_verifies_generations_before_it_scores_them():
    """Order matters: a rejected protocol must stop the characterization."""
    driver = DRIVER_PATH.read_text()
    stage3 = driver[driver.index("def stage3"):driver.index("# -- run ---")]
    verify_at = stage3.index("self.observed_generation(gen_dir, name)")
    score_at = stage3.index("score_recovery_search.py")
    assert verify_at < score_at, (
        "the rollouts are scored before their generation protocol is checked; a "
        "mismatch would then be found only after the numbers exist")
    # The evaluation protocol each result binds to is built from the OBSERVED
    # generation fingerprint, not copied from the Stage-0 object.
    assert "generation=gen_protocols[name]" in stage3
    assert "observed_eval.require_comparable(self.evaluation_protocol" in stage3
    assert 'result["evaluation_protocol_hash"]' in stage3
    # And Stage 0 no longer fills the generation protocol's library versions
    # from the training venv.
    stage0 = driver[driver.index("def stage0"):driver.index("def observe_engine")]
    declared = stage0[stage0.index("declared_generation_protocol().materialized("):]
    declared = declared[:declared.index(")")]
    for trainer_field in ("runtime.transformers_version", "runtime.torch_version",
                          "runtime.digest"):
        assert trainer_field not in declared, (
            "the generation protocol is being materialized from the TRAINING "
            f"runtime ({trainer_field}); rollouts run in the vLLM environment "
            "and an observed reconstruction could never match it")


def test_6b_setup_blocks_on_the_frozen_asset_gate_before_stage_1():
    setup = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    assert "verify_frozen_assets.py" in setup
    assert "exit 91" in setup and "FROZEN_ASSETS_FAILED" in setup
    assert setup.index("verify_frozen_assets.py") < setup.index("mark ASSETS_READY")


def test_the_derived_control_config_overrides_exactly_three_fields(tmp_path):
    """Stage 2's first action, and it had never executed either."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "preflight_driver_cfg", DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_driver_cfg"] = mod
    spec.loader.exec_module(mod)
    mod.AUDIT = tmp_path / "audit"

    driver = mod.Driver.__new__(mod.Driver)
    name, seed, frozen_rel = mod.CONTROLS[0]
    path = mod.Driver.control_config(driver, name, frozen_rel)

    frozen = json.loads((REPO / frozen_rel).read_text())
    derived = json.loads(path.read_text())
    changed = {k for k in set(frozen) | set(derived)
               if frozen.get(k) != derived.get(k)}
    assert changed == {"out_dir", "data_dir", "run_name", "_purpose"}
    assert derived["out_dir"] == f"artifacts/stage3/{name}"
    # The pack path the preregistration and the attested protocol both pin.
    assert derived["data_dir"] == "artifacts/stage3/ladder_uniform_probe"
    assert derived["seed"] == seed
    # Everything that defines the recovery is untouched.
    for key in ("loss", "optim", "schedule", "batch", "rung", "block_len",
                "trainable_patterns", "teacher", "student_path", "dtype"):
        assert derived[key] == frozen[key], key


def test_the_generation_path_is_smoke_tested_before_any_control_is_trained():
    """Stage 3's path cannot be rehearsed on CPU, so it is rehearsed on the pod.

    The 2026-08-13 session trained both permanent controls and *then* discovered
    that generation failed. The evaluator needs vLLM and a GPU, so no test here
    can execute it; the substitute is a two-prompt run of the whole path — the
    evaluator, its summaries, the observed-protocol reconstruction and the
    comparison against the attestation — inside Stage 1, where a failure costs
    minutes instead of two control runs.
    """
    driver = DRIVER_PATH.read_text()
    assert "def generation_smoke" in driver
    stage1 = driver[driver.index("def stage1"):driver.index("def generation_smoke")]
    assert "self.generation_smoke()" in stage1
    # In Stage 1, i.e. before Stage 2 can start.
    assert driver.index('gates["generation_smoke"]') < driver.index("def stage2")
    smoke = driver[driver.index("def generation_smoke"):driver.index("def repeatability")]
    # It must run the real evaluator and the real reconstruction, not a stub.
    assert "uncapped_eval.py" in smoke
    assert "observe_generation_protocol(summaries, strict=True)" in smoke
    assert "self.evaluation_protocol.generation" in smoke
    # And it must never be mistaken for a measurement.
    assert '"is_measurement": False' in smoke
