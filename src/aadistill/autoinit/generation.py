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
from collections.abc import Mapping, Sequence
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

#: The protocol's descriptive fields, defined once and emitted by the generator
#: into every summary it writes.
#:
#: These used to be literals here only, restated in prose inside
#: `uncapped_eval.py` — and the two had already drifted: this module declared the
#: max-tokens rule as "per sample: resolved_context - len(prompt_ids); P18
#: unrestricted, never a chosen token budget" while the generator wrote "per
#: sample: context - prompt; P18 unrestricted, never a chosen budget". Two
#: sentences describing the same behaviour, and an observed-vs-declared
#: comparison would have called them a protocol mismatch. One definition, used by
#: both sides.
SYSTEM_INJECTION_RULE = ("injected only when the sample carries no system turn "
                         "and protocol == project; a sample's own system prompt "
                         "is preserved")
CONTEXT_RESOLUTION_RULE = ("min(trained_context, architectural context); the "
                           "effective context is the TRAINED context and this "
                           "is not a 262K-context evaluation")
MAX_TOKENS_RULE = ("per sample: resolved_context - len(prompt_ids); P18 "
                   "unrestricted, never a chosen token budget")
STOP_ID_DERIVATION_RULE = ("sorted union of config.eos_token_id, "
                           "generation_config.eos_token_id, <|im_end|> and "
                           "tokenizer.eos_token_id")
#: The rollout dtype, passed to the engine and recorded. One name, so the
#: summary cannot describe a dtype the engine was not asked for.
GENERATION_DTYPE = "bfloat16"
#: What `tokenizer_source` means when the evaluator was given no `--tokenizer`.
#: The raw path is *not* usable as an identity: it contains the checkpoint being
#: evaluated, which differs between every arm by construction.
TOKENIZER_SOURCE_CHECKPOINT = "the evaluated checkpoint"


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
    #: Every field declared material to comparability that is only knowable once
    #: the live engine starts. A field cannot be both "part of the comparison"
    #: and "allowed to stay null": that is the `None == None` hole, and it is the
    #: reason the tool capability read a structural zero for every arm once
    #: already. If a vLLM version genuinely cannot expose one of these, the
    #: choice is explicit — Stage 0 fails closed, or the field is formally
    #: removed from the comparable protocol in a new fingerprint version with the
    #: reason recorded. Silently accepting None on both sides is not an option.
    MATERIALIZATION_REQUIRED: tuple[str, ...] = (
        "generation_source_digest", "vllm_version", "transformers_version",
        "torch_version", "tokenizer_sha256", "chat_template_sha256",
        "resolved_context", "context_source", "stop_token_ids",
        "degeneration_source_digest", "runtime_digest",
        "max_num_seqs", "max_num_batched_tokens", "enforce_eager")

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

    @classmethod
    def from_run_summaries(cls, summaries: Sequence[Mapping[str, Any]], *,
                           strict: bool = True,
                           ) -> "RecoveryGenerationProtocolFingerprint":
        """Reconstruct the protocol the stored rollouts were **actually** made under.

        Reads `uncapped_eval.py`'s per-set summaries and nothing else: not the
        attested fingerprint, not this module's declared defaults. Every material
        field must be present in every summary, and every summary of one
        evaluation must agree with the others field by field — a wave that
        changed engine settings between sets did not measure one protocol.

        A missing field raises rather than being taken from the expected
        fingerprint. That substitution is the failure this exists to prevent: it
        would make the comparison pass by construction, and the passing
        comparison is what the thresholds are then materialized under.
        """
        return observe_generation_protocol(summaries, strict=strict).protocol

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
    dtype=GENERATION_DTYPE, gpu_memory_utilization=0.90,
    max_num_seqs=None, max_num_batched_tokens=None, enforce_eager=None,
    tokenizer_source=TOKENIZER_SOURCE_CHECKPOINT,
    tokenizer_sha256=None, chat_template_sha256=None,
    protocol="project",
    system_message="You are a helpful Assistant.",
    system_injection_rule=SYSTEM_INJECTION_RULE,
    chat_template_kwargs_json="{}",
    thinking_mode="template-default (not overridden)",
    trained_context=8192, context_len_override=None,
    resolved_context=None, context_source=None,
    context_resolution_rule=CONTEXT_RESOLUTION_RULE,
    temperature=0.0, top_p=1.0, top_k=-1, detokenize=False,
    max_tokens_rule=MAX_TOKENS_RULE,
    stop_token_ids=None,
    stop_id_derivation_rule=STOP_ID_DERIVATION_RULE,
    degeneration_stop=True, degeneration_check_every=256,
    degeneration_source_digest=None,
    runtime_digest=None,
)


def declared_generation_protocol() -> RecoveryGenerationProtocolFingerprint:
    return RecoveryGenerationProtocolFingerprint(**GENERATION_V1_DECLARED)


def generation_runtime_fingerprint(image_digest: str | None):
    """The runtime the **rollouts** execute under, observed in this process.

    Not the trainer's runtime. Generation runs in the vLLM environment, which is
    a different interpreter with different torch and transformers versions from
    the training environment on the same pod — so filling this side of the
    identity from the training venv, as Stage 0 originally did, describes a stack
    that never generated a token. Both the Stage-0 engine probe and every
    evaluation wave call this function, in that environment, and the digests are
    therefore comparable.

    `image_digest` cannot be observed from inside a container; it comes from the
    launcher via `AADISTILL_IMAGE_DIGEST`, the same source Stage 0 uses.
    """
    import os

    from .recovery import RuntimeEnvironmentFingerprint

    return RuntimeEnvironmentFingerprint.observe(
        image_digest=image_digest,
        attention_backend=f"vllm:{os.environ.get('VLLM_ATTENTION_BACKEND', 'default')}")


class ObservedGenerationError(GenerationProtocolError):
    """The stored rollouts do not establish the protocol they were made under."""


#: Where each material field lives in a summary written by `uncapped_eval.py`.
#: Dotted paths, resolved against the summary itself. Everything the fingerprint
#: compares is here: a field that is part of the identity and *not* required from
#: the evidence would be a field the observed side is free to invent.
SUMMARY_FIELD_PATHS: dict[str, str] = {
    "generation_source_digest": "identity.generation_source_digest",
    "generation_source_set_version": "identity.generation_source_set_version",
    "degeneration_source_digest": "identity.degeneration_source_digest",
    "runtime_digest": "identity.runtime_digest",
    "vllm_version": "engine.vllm_version",
    "transformers_version": "identity.runtime.transformers_version",
    "torch_version": "identity.runtime.torch_version",
    "dtype": "engine.dtype",
    "gpu_memory_utilization": "engine.gpu_memory_utilization",
    "max_num_seqs": "engine.max_num_seqs",
    "max_num_batched_tokens": "engine.max_num_batched_tokens",
    "enforce_eager": "engine.enforce_eager",
    "tokenizer_source": "tokenizer_source_rule",
    "tokenizer_sha256": "tokenizer_sha256",
    "chat_template_sha256": "chat_template_sha256",
    "protocol": "protocol",
    "system_injection_rule": "identity.system_injection_rule",
    "chat_template_kwargs_json": "identity.chat_template_kwargs_json",
    "thinking_mode": "thinking_mode",
    "trained_context": "context_resolution.trained_context",
    "resolved_context": "context_resolution.resolved_context",
    "context_source": "context_resolution.context_source",
    "context_resolution_rule": "context_resolution.rule",
    "temperature": "sampling.temperature",
    "top_p": "sampling.top_p",
    "top_k": "sampling.top_k",
    "detokenize": "sampling.detokenize",
    "max_tokens_rule": "sampling.max_tokens_rule",
    "stop_token_ids": "stop_ids",
    "stop_id_derivation_rule": "identity.stop_id_derivation_rule",
    "degeneration_stop": "degeneration_stop",
    "degeneration_check_every": "sampling.degeneration_check_every",
}

#: Material fields that are legitimately null-valued rather than absent.
#: `context_len_override` is `None` whenever the context was not overridden,
#: which is the case the whole protocol assumes; it is required to be *present*
#: and is allowed to be null.
NULLABLE_SUMMARY_FIELDS = {
    "context_len_override": "context_resolution.context_len_override",
    # `--protocol native` legitimately records no system message. Required to be
    # present, allowed to be null: the *comparison* should then say "the system
    # message differs from the attested protocol", which is the accurate
    # diagnosis, rather than "no evidence".
    "system_message": "system_message",
}


def _dig(summary: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    node: Any = summary
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


@dataclass(frozen=True)
class ObservedGenerationProtocol:
    protocol: RecoveryGenerationProtocolFingerprint
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"observed_generation_protocol": self.protocol.as_dict(),
                "observed_generation_fingerprint": self.protocol.fingerprint,
                "evidence": self.evidence}


def observe_generation_protocol(summaries: Sequence[Mapping[str, Any]], *,
                                strict: bool = True) -> ObservedGenerationProtocol:
    """Rebuild one generation protocol from the summaries of one evaluation run."""
    if not summaries:
        raise ObservedGenerationError(
            "no rollout summaries; there is no evidence of how these "
            "generations were produced")

    missing: list[str] = []
    disagreements: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    labels = [s.get("label") for s in summaries]
    sets = [s.get("prompts") for s in summaries]

    paths = {**SUMMARY_FIELD_PATHS, **NULLABLE_SUMMARY_FIELDS}
    for field_name, path in paths.items():
        seen: list[tuple[Any, Any]] = []
        for summary in summaries:
            present, value = _dig(summary, path)
            if not present:
                missing.append(
                    f"{field_name} (summary {summary.get('prompts') or '?'} has "
                    f"no {path})")
                continue
            if value is None and field_name not in NULLABLE_SUMMARY_FIELDS:
                missing.append(
                    f"{field_name} (summary {summary.get('prompts') or '?'} "
                    f"records {path} as null)")
                continue
            if field_name == "stop_token_ids":
                value = tuple(value)
            seen.append((summary.get("prompts"), value))
        if not seen:
            continue
        distinct = {json_key(v) for _, v in seen}
        if len(distinct) > 1:
            disagreements.append({
                "field": field_name,
                "values": {str(where): value for where, value in seen}})
        values[field_name] = seen[0][1]

    if strict and missing:
        raise ObservedGenerationError(
            f"{len(missing)} material generation field(s) are not established by "
            "the stored rollouts, so the protocol they were produced under "
            "cannot be reconstructed:\n  - " + "\n  - ".join(missing)
            + "\n\nFilling any of them from the attested fingerprint would make "
              "the comparison pass by construction. Fail closed instead.")
    if strict and disagreements:
        raise ObservedGenerationError(
            "the sets of this evaluation were not generated under one protocol: "
            + "; ".join(f"{d['field']} differs across sets ({d['values']})"
                        for d in disagreements))

    declared = dict(GENERATION_V1_DECLARED)
    unknown = sorted({m.split(" (")[0] for m in missing})
    observed = {k: v for k, v in values.items()}
    for key in unknown:
        observed.pop(key, None)
        declared[key] = None
    protocol = RecoveryGenerationProtocolFingerprint(
        **{**{k: None for k in declared}, **observed,
           # Version fields are structural, not observed; they identify which
           # fingerprint schema this object is, not what the run did.
           "protocol_id": GENERATION_PROTOCOL_ID,
           "version": GENERATION_PROTOCOL_VERSION})
    return ObservedGenerationProtocol(
        protocol=protocol,
        evidence={
            "strict": strict,
            "n_summaries": len(summaries),
            "labels": sorted({str(x) for x in labels}),
            "sets": [str(x) for x in sets],
            "field_paths": paths,
            "missing_fields": missing,
            "cross_set_disagreements": disagreements,
            "rule": ("every material field is read from the rollout summaries "
                     "themselves; all sets of one evaluation must agree; nothing "
                     "is taken from the declared or attested fingerprint"),
        })


def json_key(value: Any) -> str:
    """A comparable key for heterogeneous summary values (lists included)."""
    import json as _json

    try:
        return _json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


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
    battery_artifact: str          # "recovery_search_v2"
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
