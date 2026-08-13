"""The rollout layer's identity: what turned a checkpoint into stored text.

Three protocols sit between a recovered checkpoint and a materialized threshold,
and until now only two of them had an identity:

    recovered checkpoint
        |  RecoveryProtocolFingerprint      -- how it was trained
        v
    generation / rollout
        |  <- this module                   -- how it was made to speak
        v
    stored outputs
        |  recovery_search_scoring@v2       -- how they became numbers
        v
    metrics

The middle layer decides as much as either neighbour. Temperature, the resolved
context, the stop-token set, the degeneration check interval, the chat template
and the engine's scheduler defaults all move `usable_rollout_rate` — and
`usable_rollout_rate` is what Stage 3 freezes the feasibility floor from. A
control and a searched candidate generated under different vLLM builds or a
different `max_num_seqs` are not comparable, and nothing in the record would have
said so.

**No new rollout algorithm is defined here.** The semantics are exactly
`scripts/evaluation/uncapped_eval.py` — P18 unrestricted generation within the
effective context, greedy, semantic degeneration stop, complete raw output
retained. This module only names and hashes what that script does, and refuses to
call two runs comparable when any of it differs.

`RecoveryEvaluationProtocol` then joins the three identities into the single hash
every control and every later probe binds to.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_file, sha256_json

#: The implementation whose behaviour this fingerprint claims to describe. If the
#: generator changes, the digest must move, or the identity is a decoration.
GENERATION_SOURCE_FILES_V1: tuple[str, ...] = (
    "scripts/evaluation/uncapped_eval.py",       # the rollout driver itself
    "src/aadistill/evaluation/degeneration.py",  # the semantic stop
    "src/aadistill/evaluation/behavior.py",      # split/score of the stored text
)
GENERATION_PROTOCOL_ID = "recovery_generation"
GENERATION_PROTOCOL_VERSION = 1


class GenerationProtocolError(RuntimeError):
    """Two generations are not comparable, or an identity is not materialized."""


def generation_source_digest(repo_root: str | Path = ".", *,
                             files: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Aggregate digest over the declared generation implementation."""
    root = Path(repo_root)
    declared = tuple(files) if files is not None else GENERATION_SOURCE_FILES_V1
    entries = []
    for rel in sorted(declared):
        path = root / rel
        if not path.is_file():
            raise GenerationProtocolError(
                f"declared generation source {rel!r} is missing; refusing to "
                "produce a digest over a smaller generator than the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "files": entries,
            "rule": "sha256 over sorted 'path:sha256' lines of the declared set"}


@dataclass(frozen=True)
class RecoveryGenerationProtocolFingerprint:
    """Everything that must be identical for two rollouts to be comparable.

    Deliberately **excludes** the checkpoint being generated from: that is the
    treatment, exactly as in `RecoveryProtocolFingerprint`. A generation identity
    containing the model would mark every pair of arms as mismatched and be
    useless for its one job.

    Fields that are only knowable once the engine is up — the vLLM version, the
    scheduler defaults that actually cap the batch, the resolved stop ids — are
    `None` until observed, and `require_materialized()` refuses to declare a
    match while any of them is unknown. Unknown on both sides is not verified
    identical; that rule cost this project a real defect once already.
    """

    # implementation
    generation_source_digest: str | None
    generation_source_set_version: int
    # engine + libraries
    vllm_version: str | None
    transformers_version: str | None
    torch_version: str | None
    dtype: str
    gpu_memory_utilization: float
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    enforce_eager: bool | None
    # tokenizer + template
    tokenizer_source: str
    tokenizer_sha256: str | None
    chat_template_sha256: str | None
    # prompt protocol
    protocol: str
    system_message: str | None
    system_injection_rule: str
    chat_template_kwargs_json: str
    thinking_mode: str
    # context
    trained_context: int
    context_len_override: int | None
    resolved_context: int | None
    context_source: str | None
    context_resolution_rule: str
    # sampling
    temperature: float
    top_p: float
    top_k: int
    detokenize: bool
    max_tokens_rule: str
    # stopping
    stop_token_ids: tuple[int, ...] | None
    stop_id_derivation_rule: str
    degeneration_stop: bool
    degeneration_check_every: int
    degeneration_source_digest: str | None
    # runtime
    runtime_digest: str | None

    protocol_id: str = GENERATION_PROTOCOL_ID
    version: int = GENERATION_PROTOCOL_VERSION

    #: Fields that must carry a real value before two protocols may be called
    #: identical. Everything here is an *observation*, not a choice, which is why
    #: it cannot be filled in from the config alone.
    MATERIALIZATION_REQUIRED: tuple[str, ...] = (
        "generation_source_digest", "vllm_version", "transformers_version",
        "torch_version", "tokenizer_sha256", "chat_template_sha256",
        "resolved_context", "stop_token_ids", "degeneration_source_digest",
        "runtime_digest", "max_num_seqs")

    def identity(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()
               if k not in ("MATERIALIZATION_REQUIRED",)}
        if self.stop_token_ids is not None:
            out["stop_token_ids"] = list(self.stop_token_ids)
        return out

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.identity())

    def unmaterialized_fields(self) -> tuple[str, ...]:
        return tuple(f for f in self.MATERIALIZATION_REQUIRED
                     if getattr(self, f) is None)

    @property
    def is_materialized(self) -> bool:
        return not self.unmaterialized_fields()

    def require_materialized(self, *, context: str = "") -> None:
        missing = self.unmaterialized_fields()
        if missing:
            where = f" ({context})" if context else ""
            raise GenerationProtocolError(
                f"generation protocol is not materialized{where}: "
                f"{', '.join(missing)} unknown. These are engine observations; "
                "run one wave and record them before declaring two rollouts "
                "comparable.")

    def compare(self, other: "RecoveryGenerationProtocolFingerprint") -> dict[str, Any]:
        mine, theirs = self.identity(), other.identity()
        unknown_fields = sorted(set(self.unmaterialized_fields())
                                | set(other.unmaterialized_fields()))
        matched, mismatched, unknown = [], [], []
        for key in sorted(set(mine) | set(theirs)):
            if key in unknown_fields:
                unknown.append({"field": key, "self": mine.get(key),
                                "other": theirs.get(key)})
            elif mine.get(key) == theirs.get(key):
                matched.append(key)
            else:
                mismatched.append({"field": key, "self": mine.get(key),
                                   "other": theirs.get(key)})
        return {"fingerprint_self": self.fingerprint,
                "fingerprint_other": other.fingerprint,
                "matched_fields": matched, "mismatched_fields": mismatched,
                "unverifiable_fields": unknown,
                "unmaterialized_fields": unknown_fields,
                "both_materialized": self.is_materialized and other.is_materialized,
                "identical": not mismatched and not unknown}

    def materialized(self, **observations: Any) -> "RecoveryGenerationProtocolFingerprint":
        """Fill observed fields once, refusing to overwrite a differing value."""
        from dataclasses import replace

        for key, value in observations.items():
            if not hasattr(self, key):
                raise GenerationProtocolError(f"unknown generation field {key!r}")
            current = getattr(self, key)
            if current is not None and current != value:
                raise GenerationProtocolError(
                    f"observed {key}={value!r} contradicts the declared "
                    f"{current!r}; that is generation-protocol drift, not an "
                    "update")
        return replace(self, **observations)

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity(),
                "generation_protocol_fingerprint": self.fingerprint,
                "is_materialized": self.is_materialized,
                "unmaterialized_fields": list(self.unmaterialized_fields()),
                "excluded_by_design": ["the checkpoint being generated from"],
                "why_excluded": ("the checkpoint is the treatment; a generation "
                                 "identity containing it would mark every pair "
                                 "of arms as mismatched")}


#: The declared half of the protocol: everything chosen in advance, matching
#: `uncapped_eval.py`'s defaults exactly. The observed half is filled at Stage 0
#: / the first wave. Values here are choices; anything left `None` is an
#: observation.
GENERATION_V1_DECLARED = dict(
    generation_source_digest=None,
    generation_source_set_version=1,
    vllm_version=None, transformers_version=None, torch_version=None,
    dtype="bfloat16", gpu_memory_utilization=0.90,
    max_num_seqs=None, max_num_batched_tokens=None, enforce_eager=None,
    tokenizer_source="the evaluated checkpoint",
    tokenizer_sha256=None, chat_template_sha256=None,
    protocol="project",
    system_message="You are a helpful Assistant.",
    system_injection_rule=("injected only when the sample carries no system turn "
                           "and protocol == project; a sample's own system prompt "
                           "is preserved"),
    chat_template_kwargs_json="{}",
    thinking_mode="template-default (not overridden)",
    trained_context=8192, context_len_override=None,
    resolved_context=None, context_source=None,
    context_resolution_rule=("min(trained_context, architectural context); the "
                             "effective context is the TRAINED context and this "
                             "is not a 262K-context evaluation"),
    temperature=0.0, top_p=1.0, top_k=-1, detokenize=False,
    max_tokens_rule=("per sample: resolved_context - len(prompt_ids); P18 "
                     "unrestricted, never a chosen token budget"),
    stop_token_ids=None,
    stop_id_derivation_rule=("sorted union of config.eos_token_id, "
                             "generation_config.eos_token_id, <|im_end|> and "
                             "tokenizer.eos_token_id"),
    degeneration_stop=True, degeneration_check_every=256,
    degeneration_source_digest=None,
    runtime_digest=None,
)


def declared_generation_protocol() -> RecoveryGenerationProtocolFingerprint:
    return RecoveryGenerationProtocolFingerprint(**GENERATION_V1_DECLARED)


@dataclass(frozen=True)
class RecoveryEvaluationProtocol:
    """Generation + scoring + battery: the one hash a probe's metrics bind to.

    A control measured under this hash and a searched candidate measured under a
    different one are not comparable, however similar they look. Stage 3
    materializes thresholds from the control's numbers, so this is the identity
    that makes those thresholds mean anything later.
    """

    generation: RecoveryGenerationProtocolFingerprint
    scoring_contract: str          # e.g. "recovery_search_scoring@v2"
    scoring_digest: str
    battery_artifact: str          # "recovery_search_v1"
    battery_manifest_sha256: str
    battery_content_sha256: str
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def evaluation_protocol_hash(self) -> str:
        return sha256_json({
            "generation_protocol_fingerprint": self.generation.fingerprint,
            "scoring_contract": self.scoring_contract,
            "scoring_digest": self.scoring_digest,
            "battery_artifact": self.battery_artifact,
            "battery_manifest_sha256": self.battery_manifest_sha256,
            "battery_content_sha256": self.battery_content_sha256})

    def require_materialized(self, *, context: str = "") -> None:
        self.generation.require_materialized(context=context)

    def matched_against(self, other: "RecoveryEvaluationProtocol") -> dict[str, Any]:
        gen = self.generation.compare(other.generation)
        same_scoring = (self.scoring_contract == other.scoring_contract
                        and self.scoring_digest == other.scoring_digest)
        same_battery = (self.battery_artifact == other.battery_artifact
                        and self.battery_manifest_sha256 == other.battery_manifest_sha256
                        and self.battery_content_sha256 == other.battery_content_sha256)
        ok = gen["both_materialized"] and gen["identical"] and same_scoring \
            and same_battery
        return {"generation_identical": gen["identical"],
                "generation_materialized": gen["both_materialized"],
                "same_scoring_contract": same_scoring,
                "same_battery": same_battery,
                "comparable": ok,
                "generation_comparison": gen,
                "verdict": ("COMPARABLE: same generation protocol, same scoring "
                            "contract, same battery." if ok else
                            "NOT COMPARABLE: " + "; ".join(filter(None, [
                                None if gen["both_materialized"] else
                                "generation protocol not materialized ("
                                + ", ".join(gen["unmaterialized_fields"]) + ")",
                                None if gen["identical"] else
                                "generation protocol differs",
                                None if same_scoring else "scoring contract differs",
                                None if same_battery else "battery differs"])))}

    def require_comparable(self, other: "RecoveryEvaluationProtocol", *,
                           context: str = "") -> None:
        verdict = self.matched_against(other)
        if not verdict["comparable"]:
            where = f" ({context})" if context else ""
            raise GenerationProtocolError(
                f"evaluation protocols are not comparable{where}: "
                f"{verdict['verdict']}")

    def as_dict(self) -> dict[str, Any]:
        return {"evaluation_protocol_hash": self.evaluation_protocol_hash,
                "generation": self.generation.as_dict(),
                "scoring_contract": self.scoring_contract,
                "scoring_digest": self.scoring_digest,
                "battery": {"artifact": self.battery_artifact,
                            "manifest_sha256": self.battery_manifest_sha256,
                            "content_sha256": self.battery_content_sha256},
                "binding": ("every control and every searched recovery probe must "
                            "bind to this exact hash; a probe measured under a "
                            "different one is not comparable to the thresholds "
                            "materialized from these controls"),
                "notes": dict(self.notes)}
