"""The Phase-C1 scoring identity: what `c1_confirmation_scoring@v1` binds.

`recovery_search_scoring@v2` cannot score `c1_confirmation_v1`, in two
independent places, both inside `scripts/autoinit/score_recovery_search.py`:

1. `BATTERY_MANIFEST_SHA256` / `BATTERY_CONTENT_SHA256` are module constants
   checked unconditionally, so `--battery` cannot reach a different asset;
2. the result builder reads `manifest["metrics"]`, and the C1 battery manifest
   deliberately has no such key — verified by execution, with the pins overridden
   in memory, which raised `KeyError: 'metrics'` *after* scoring 150 rows.

Both frozen assets stay untouched. `recovery_search_scoring@v2` remains the
Phase-A/B scoring identity at `808080a7…`, and the C1 battery keeps
`content_sha256 = a285d61f…` and `manifest_sha256 = e6ff5cf5…`. A second identity
is declared here instead.

**The split is deliberate and is the whole design.** The battery defines the
examples and their distribution; this contract defines the metric semantics. The
C1 manifest is therefore not asked for a `metrics` block — `C1_METRIC_CONTRACT`
below is that block, and it lives in source under a hash rather than inside an
asset that a future battery build could quietly reshape.

**The numbers do not change.** Every correctness, usability, row-composition,
aggregation and capability rule is imported from the frozen implementation, and
`scripts/autoinit/score_c1_confirmation.py` restates only the *iteration* over
sets. That restatement is made safe by an admission gate rather than by review:
`tests/autoinit/test_c1_confirmation_scoring.py` scores real retained
`recovery_search_v2` generations through both paths and requires equality of
every material numerical field. If any differed, C1 could not use this scorer,
because the C0 power analysis and the SESOI were computed under the historical
semantics.

**The closure is wider than V2's, on purpose.** `RECOVERY_SCORING_FILES_V2` omits
three files that directly decide numbers — `audit_tool_scoring.py` (the OpenAI
envelope translation every tool verdict passes through), `data/tools.py` (tool
normalization, whose absence once made the whole tool capability structurally
zero) and `data/verify.py` (`boxed_answer` / `normalize_math`, which `capability`
and `strict_answer` both call to decide correctness). Phase-A/B's historical
contract is NOT rewritten to add them — that would move a digest bound in
finished records — but the hole is not repeated here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_file

SCHEMA = "aadistill.autoinit.c1_confirmation_result/v1"

C1_SCORING_CONTRACT_ID = "c1_confirmation_scoring"
C1_SCORING_CONTRACT_VERSION = 1

#: The semantic ancestor. Recorded so a reader can see that this identity is a
#: new *binding*, not a new *metric*.
C1_SCORING_SEMANTIC_PARENT = "recovery_search_scoring@v2"
C1_SCORING_SEMANTIC_PARENT_DIGEST = (
    "808080a7c5d88d5a66760fd0d7eeabc5451c096ad0819f8c5663a0b8224660be")
C1_SCORING_SEMANTIC_DELTA = (
    "battery and result-schema binding only; correctness, usability, row "
    "composition, aggregation and capability semantics are the imported frozen "
    "implementations and are unchanged, which the historical numerical-"
    "equivalence gate demonstrates rather than asserts")


class C1ScoringError(RuntimeError):
    """The battery this scorer was pointed at is not the frozen C1 battery."""


# --- the frozen C1 battery, pinned here and nowhere else --------------------

C1_BATTERY_PATH = "artifacts/stage3/c1_confirmation_v1"
C1_BATTERY_MANIFEST_SHA256 = (
    "e6ff5cf536d515e6b466f2fc945c4368c89637d481c8ce1993ef3f3cf0909e42")
C1_BATTERY_CONTENT_SHA256 = (
    "a285d61f88de9da85e87818786cce8d350f03246365ff946207c61a6464fee3c")

#: name -> (n, domain, scorable). Pinned so a battery that still hashes
#: correctly but was staged half-copied cannot be scored as if whole.
C1_BATTERY_SETS: dict[str, tuple[int, str, bool]] = {
    "gsm8k": (150, "reasoning_math", True),
    "math_verified": (150, "reasoning_math", True),
    "multihop": (150, "rag_multihop", True),
    "rag": (150, "rag_multihop", True),
    "knowledge": (150, "general", True),
    "tool": (100, "tool", True),
    "code": (100, "code", False),
}
C1_N_PROMPTS = 950
C1_N_SCORABLE_PROMPTS = 850

#: The metric semantics, which the C1 battery manifest deliberately does not
#: carry. Preserved EXACTLY from the frozen recovery semantics: the denominators
#: are the load-bearing part, and they are the reason `code` can be behaviour-only
#: without depressing every arm identically.
C1_METRIC_CONTRACT: dict[str, Any] = {
    "contract": f"{C1_SCORING_CONTRACT_ID}@v{C1_SCORING_CONTRACT_VERSION}",
    "defined_by": "source, not the battery manifest",
    "correct_overall": {
        "numerator": "correct on scorable prompts",
        "denominator": "n_scorable",
        "denominator_value": C1_N_SCORABLE_PROMPTS,
    },
    "usable_rollout_rate": {
        "numerator": "usable on ALL prompts",
        "denominator": "n",
        "denominator_value": C1_N_PROMPTS,
    },
    "correct_given_usable": {
        "numerator": "correct on scorable prompts",
        "denominator": "usable_scorable",
        "role": "diagnostic; never reorders",
    },
    "scorable_sets": sorted(k for k, v in C1_BATTERY_SETS.items() if v[2]),
    "behaviour_only_sets": sorted(
        k for k, v in C1_BATTERY_SETS.items() if not v[2]),
    "correct_implies_usable": (
        "by construction, via the frozen score_recovery_row contract"),
    "tool_usability": (
        "generic usable_rollout AND tool_call_emitted AND tool_call_parsed AND "
        "tool_name_valid"),
    "capability_schema": "CAPABILITY_SCHEMA_V1 (six scorable capabilities)",
    "no_weighted_scalar": (
        "usable_rollout_rate and correct_overall are reported separately and are "
        "never combined; usable_rollout is blind to correctness by construction"),
}


#: Every file whose bytes can change a C1 numerical score. `infrastructure/
#: manifest.py` is deliberately absent: it hashes results, and can move
#: `result_sha256`, but cannot move any count or rate.
C1_SCORING_FILES_V1: tuple[str, ...] = (
    # the C1 binding: pins, validation, result schema, this contract
    "scripts/autoinit/score_c1_confirmation.py",
    "src/aadistill/autoinit/c1_scoring.py",
    # the historical implementation whose rules it imports
    "scripts/autoinit/score_recovery_search.py",
    "src/aadistill/autoinit/recovery.py",
    "src/aadistill/evaluation/usable_rollout.py",
    "src/aadistill/evaluation/strict_answer.py",
    "src/aadistill/evaluation/behavior.py",
    "src/aadistill/evaluation/capability.py",
    # the three V2 leaves out, each of which decides numbers
    "scripts/autoinit/audit_tool_scoring.py",
    "src/aadistill/data/tools.py",
    "src/aadistill/data/verify.py",
)


def c1_scoring_contract(repo_root: str | Path = ".", *,
                        files: Sequence[str] | None = None) -> dict[str, Any]:
    """The aggregate C1 scoring digest a result binds to.

    Same convention and same failure mode as every other source-digest set here:
    a missing declared file raises rather than yielding a digest over a smaller
    scorer than the one that runs.
    """
    root = Path(repo_root)
    declared = tuple(files) if files is not None else C1_SCORING_FILES_V1
    entries = []
    for rel in sorted(declared):
        path = root / rel
        if not path.is_file():
            raise C1ScoringError(
                f"declared C1 scoring source {rel!r} is missing; refusing to "
                "produce a contract digest that describes a smaller scorer than "
                "the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {
        "contract": f"{C1_SCORING_CONTRACT_ID}@v{C1_SCORING_CONTRACT_VERSION}",
        "contract_id": C1_SCORING_CONTRACT_ID,
        "version": C1_SCORING_CONTRACT_VERSION,
        "digest": digest,
        "files": entries,
        "rule": ("sha256 over sorted 'path:sha256' lines of the declared C1 "
                 "scoring source set"),
        "semantic_parent": C1_SCORING_SEMANTIC_PARENT,
        "semantic_parent_digest": C1_SCORING_SEMANTIC_PARENT_DIGEST,
        "semantic_delta": C1_SCORING_SEMANTIC_DELTA,
        "closure_note": (
            "wider than RECOVERY_SCORING_FILES_V2, which omits "
            "audit_tool_scoring.py, data/tools.py and data/verify.py although all "
            "three decide numbers. The historical contract is not rewritten; the "
            "hole is not repeated."),
    }


def validate_c1_battery(manifest: dict[str, Any], *,
                        manifest_sha256: str) -> dict[str, Any]:
    """Refuse anything that is not exactly the frozen C1 battery.

    Every check is an equality against a value pinned in this module, so a
    battery that was rebuilt, half-staged, re-mixed or relabelled fails here
    rather than producing a number that looks like a result.
    """
    problems: list[str] = []
    if manifest_sha256 != C1_BATTERY_MANIFEST_SHA256:
        problems.append(
            f"battery manifest is {manifest_sha256} but this scorer pins "
            f"{C1_BATTERY_MANIFEST_SHA256}")
    if manifest.get("content_sha256") != C1_BATTERY_CONTENT_SHA256:
        problems.append(
            f"battery content is {manifest.get('content_sha256')} but this "
            f"scorer pins {C1_BATTERY_CONTENT_SHA256}")
    sets = manifest.get("sets") or {}
    if sorted(sets) != sorted(C1_BATTERY_SETS):
        problems.append(
            f"battery sets are {sorted(sets)}, not {sorted(C1_BATTERY_SETS)}")
    else:
        for name, (n, domain, _) in C1_BATTERY_SETS.items():
            spec = sets[name]
            if spec.get("n") != n:
                problems.append(f"{name}: manifest says n={spec.get('n')}, pinned {n}")
            if spec.get("domain") != domain:
                problems.append(
                    f"{name}: domain {spec.get('domain')!r}, pinned {domain!r}")
    scorable = set(manifest.get("scorable_sets") or ())
    behaviour = set(manifest.get("behaviour_only_sets") or ())
    want_scorable = {k for k, v in C1_BATTERY_SETS.items() if v[2]}
    if scorable != want_scorable:
        problems.append(
            f"scorable sets are {sorted(scorable)}, pinned {sorted(want_scorable)}")
    if behaviour != {"code"}:
        problems.append(
            f"behaviour-only sets are {sorted(behaviour)}, pinned ['code']")
    if manifest.get("n_prompts") != C1_N_PROMPTS:
        problems.append(f"n_prompts {manifest.get('n_prompts')}, pinned {C1_N_PROMPTS}")
    if manifest.get("n_scorable_prompts") != C1_N_SCORABLE_PROMPTS:
        problems.append(
            f"n_scorable_prompts {manifest.get('n_scorable_prompts')}, pinned "
            f"{C1_N_SCORABLE_PROMPTS}")
    if problems:
        raise C1ScoringError(
            "this is not the frozen C1 confirmation battery: " + "; ".join(problems))
    return {
        "artifact": manifest["artifact"],
        "role": manifest["role"],
        "version": manifest["version"],
        "manifest_sha256": manifest_sha256,
        "manifest_sha256_convention": (
            "sha256_json over the manifest with manifest_sha256 removed; NOT "
            "sha256_file of the raw bytes"),
        "content_sha256": manifest["content_sha256"],
        "n_prompts": manifest["n_prompts"],
        "n_scorable_prompts": manifest["n_scorable_prompts"],
        "has_metrics_key": "metrics" in manifest,
        "metrics_source": (
            "C1_METRIC_CONTRACT in aadistill.autoinit.c1_scoring; the C1 battery "
            "manifest carries no metrics block and is not asked for one"),
    }
