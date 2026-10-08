"""D1's behavioural rungs: the arms, the probes, and what binds each of them.

The frozen facts of the screening and confirmation sessions, in one place, so
the driver and the launcher derive them rather than each carrying a copy. Every
one is READ from the record that owns it:

    probe and seed counts      the design's `behavioural_design`
    the recovery recipe        `experiments.recipes.E1_KD_HEAVY_0860K`
    the candidate membership   the maintainer's retention decision
    the incumbent              the design's `incumbent`
    the battery                the realized D-series family manifest

Nothing here decides anything. The Top-4 field was frozen by the retention
decision of 2026-10-07, the recipe and the seeds by the design, and the battery
by the source-policy decision of 2026-10-03 that built the D-series family
before any D1 outcome existed.

WHY THE BATTERY IS NOT THE EXHAUSTED ONE. `d1_evidence_capacity.json` reports
`batteries_remaining: 0`, binding on MATH-500 with 70 eligible rows against 150
per battery. That is the correct answer to a HISTORICAL question -- how many
further batteries the original C1 source populations can yield -- and it is not
this session's battery. The realized family drew `d1_screening` and
`d1_confirmation` from prospectively widened sources while preserving the frozen
stratum balance exactly, and its own record carries
`capacity_source_blocker: CLOSED`. A reader who followed the capacity record
reported D1 as blocked on a battery it already had, which is why `require_*`
below checks the realized bytes rather than any claim about them.

WHAT THE WIDENED SOURCES COST, stated because it is a real limitation and not a
footnote: D-series absolute scores are NOT directly interchangeable with the
historical C1 absolute scores, because three capacity-limited strata draw from
wider populations. D1 does not need them to be. Both arms of each rung are
measured on the SAME role under the same generation and scoring protocol, and
the estimand is the within-family paired candidate-minus-B difference. The C0
SESOI of 0.010 is carried forward as an EXPLICIT ASSUMPTION about that
difference, not as a re-measurement on the new population.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]

DESIGN_REL = "logs/stages/stage-1/phase_d1/plans/d1_design.json"
RETENTION_REL = ("logs/stages/stage-1/phase_d1/decisions/"
                 "post_search_finalist_retention.json")
FAMILY_REL = "logs/shared/analyses/autoinit_d_series_battery_family.json"
MANIFEST_REL = "logs/shared/analyses/autoinit_d_series_family_manifest.json"
BATTERY_ROOT_REL = "artifacts/stage3/d_series_behavioural_v1"

#: The seven strata and their per-battery counts, frozen by C0 and preserved
#: exactly by the D-series family. `correct_overall` is a mean over THIS
#: mixture, which is why a battery with a different balance measures a
#: different quantity and why the family preserved it rather than shrinking it.
FROZEN_STRATA: dict[str, int] = {
    "code": 100, "gsm8k": 150, "knowledge": 150, "math_verified": 150,
    "multihop": 150, "rag": 150, "tool": 100,
}

#: Where each arm's identity-verified bytes live on the development host. The
#: search secured q1 and q3 itself; q2 and q4 were rematerialized and verified
#: three times (on the pod against each step's pin, on arrival, and again
#: here). B is C1's frozen treatment, still standing after C2 closed without
#: promotion and C3 returned NO_GO.
ARM_SOURCES: dict[str, str] = {
    "q1": "/home/ecs-user/aad-scratch/d1_search_20261006_210210/products",
    "q3": "/home/ecs-user/aad-scratch/d1_search_20261006_210210/products",
    "q2": "/home/ecs-user/aad-scratch/d1_replay_002/products",
    "q4": "/home/ecs-user/aad-scratch/d1_replay_002/products",
    "B": "/home/ecs-user/aad-artifacts/autoinit/phase_a",
}


class D1BehaviouralError(RuntimeError):
    """A behavioural precondition does not hold. Raised before anything is
    priced or created, so a reader sees the premise rather than a traceback."""


def _read(rel: str, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    path = Path(repo_root) / rel
    if not path.is_file():
        raise D1BehaviouralError(f"{rel} is missing; it owns a frozen fact "
                                 "this session cannot derive without it")
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Arm:
    """One initialization under test, and where its verified bytes are.

    `quality_position` is `None` for the incumbent: B is not a candidate of the
    D1 search and has no position in its quality order. Conflating the two
    would make the incumbent look like a fifth candidate.
    """

    arm_id: str
    state_id: str
    artifact_digest: str
    quality_position: int | None
    checkpoint_dir: str
    role: str

    @property
    def is_incumbent(self) -> bool:
        return self.role == "incumbent"


@dataclass(frozen=True)
class Probe:
    """One (arm, seed) training-and-evaluation unit.

    The seed is the REPLICATE and the initialization is the treatment. Two
    probes at one seed differ only in which checkpoint they started from, and
    that is what makes their difference attributable.
    """

    probe_id: str
    arm_id: str
    seed: int
    rung: str
    checkpoint_dir: str
    battery_role: str


def design(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    return _read(DESIGN_REL, repo_root)


def behavioural_design(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    return design(repo_root)["behavioural_design"]


C1_EXECUTION_PREREG_REL = ("logs/stages/stage-1/phase_c1/plans/"
                           "execution_preregistration.json")


def excluded_seeds(repo_root: str | Path = REPO_ROOT) -> tuple[int, ...]:
    """Every seed a D1 draw must avoid, READ from the records that own them.

    `sa/sb/sc` because the incumbent B was itself selected under them, so
    reusing them leaves a winner's-curse channel -- C0 says so in as many
    words. And C1's three MATERIALIZED confirmation seeds, because B's
    standing result was measured on them: a D1 rung that reused them would
    compare a fresh candidate against an incumbent on the incumbent's own
    seeds.

    A missing C1 preregistration is NOT silently fine. It would mean the
    exclusion set is unknown, and an unknown exclusion set is how a draw
    collides with a selection seed without anyone noticing.
    """
    prereg = _read(
        "logs/stages/stage-1/phase_c1/plans/phase_c0_preregistration.json",
        repo_root)
    historical = prereg["confirmation_seeds"]["historical_seeds_excluded"]
    out = {int(v) for k, v in historical.items() if k in ("sa", "sb", "sc")}
    path = Path(repo_root) / C1_EXECUTION_PREREG_REL
    if not path.is_file():
        raise D1BehaviouralError(
            f"{C1_EXECUTION_PREREG_REL} is missing, so C1's materialized "
            "confirmation seeds cannot be excluded from a D1 draw")
    c1 = json.loads(path.read_text())
    values = (c1.get("seeds") or {}).get("values")
    if not isinstance(values, list) or not values:
        raise D1BehaviouralError(
            "C1's execution preregistration carries no materialized "
            "`seeds.values`; this session will not guess which seeds it used")
    out |= {int(v) for v in values}
    return tuple(sorted(out))


def derive_seeds(rung: str, repo_root: str | Path = REPO_ROOT
                 ) -> tuple[int, ...]:
    """This rung's exact seeds, derived from an identity frozen before it ran.

        H_i    = SHA256(design_hash + ":phase-d1:<rung>-seed:" + decimal(i))
        seed_i = uint32_be(H_i[0:4]) mod 2**31

    advancing past any collision with an excluded seed, an earlier draw of this
    rung, or -- for confirmation -- any screening seed.

    WHY A DERIVATION AND NOT A CHOICE. C0 fixes the seed COUNT and requires
    freshness, and is explicit that its exact values are "DELIBERATELY NOT SET
    HERE... no prospective deterministic seed-selection rule exists that could
    choose them without reference to future candidate results", so they "MUST
    be materialized and hash-bound in the execution preregistration BEFORE any
    candidate behavioural result exists". This is that materialization, and it
    runs before this rung has measured anything.

    THE BASE IS D1'S OWN `design_hash` -- the scientific preimage of the frozen
    design, which did not move across two spend bookings or a ceiling
    amendment. So the values are fixed by a document that predates every D1
    behavioural result and no discretion is left to exercise: human-chosen
    seeds cannot be shown to be independent of anything.

    THE RULE IS C1'S, with D1's domain separator. Inheriting the shape keeps
    the rounds comparable in the way C0 asks for; the separator keeps the draws
    distinct.
    """
    import hashlib

    if rung not in ("screening", "confirmation"):
        raise D1BehaviouralError(f"unknown rung {rung!r}")
    count = int(behavioural_design(repo_root)[f"{rung}_seeds"])
    if count < 1:
        raise D1BehaviouralError(f"the design asks for {count} {rung} seeds")
    exclude = set(excluded_seeds(repo_root))
    if rung == "confirmation":
        exclude |= set(derive_seeds("screening", repo_root))
    base = design(repo_root)["design_hash"]
    seeds: list[int] = []
    i = 0
    while len(seeds) < count:
        digest = hashlib.sha256(
            f"{base}:phase-d1:{rung}-seed:{i}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big") % (2 ** 31)
        if seed not in exclude and seed not in seeds:
            seeds.append(seed)
        i += 1
        if i > 10_000:
            raise D1BehaviouralError("seed derivation failed to converge")
    return tuple(seeds)


def screening_seeds(repo_root: str | Path = REPO_ROOT) -> tuple[int, ...]:
    return derive_seeds("screening", repo_root)


def confirmation_seeds(repo_root: str | Path = REPO_ROOT) -> tuple[int, ...]:
    return derive_seeds("confirmation", repo_root)


def arms(repo_root: str | Path = REPO_ROOT) -> tuple[Arm, ...]:
    """The five arms: the frozen Top-4 field plus the incumbent.

    Candidate membership is READ from the maintainer's retention decision, so
    this session cannot re-select; the incumbent is read from the design.
    """
    root = Path(repo_root)
    decision = _read(RETENTION_REL, repo_root)
    frozen = decision["the_frozen_behavioural_finalists"]
    if frozen["rule_applied"] != "quality_only":
        raise D1BehaviouralError(
            "the retention decision does not declare quality-only retention; "
            "the candidate field this session measures is not the frozen one")
    out: list[Arm] = []
    for member in frozen["members"]:
        position = int(member["quality_position"])
        arm_id = f"q{position}"
        source = ARM_SOURCES.get(arm_id)
        if source is None:
            raise D1BehaviouralError(
                f"{arm_id} has no declared checkpoint source; a probe cannot "
                "be trained from a checkpoint nobody can name")
        out.append(Arm(arm_id=arm_id, state_id=member["state_id"],
                       artifact_digest=member["artifact_digest"],
                       quality_position=position,
                       checkpoint_dir=str(Path(source) / member["state_id"]),
                       role="candidate"))
    incumbent = design(repo_root)["incumbent"]
    out.append(Arm(
        arm_id="B", state_id=incumbent["state_id"],
        artifact_digest=incumbent["artifact_digest"],
        quality_position=None,
        checkpoint_dir=str(Path(ARM_SOURCES["B"]) / incumbent["state_id"]),
        role="incumbent"))
    out.sort(key=lambda a: (a.is_incumbent, a.quality_position or 0))
    return tuple(out)


def probes(rung: str, repo_root: str | Path = REPO_ROOT) -> tuple[Probe, ...]:
    """Every probe of one rung, in a deterministic order.

    The count is CHECKED against the design rather than assumed: the design
    says screening is 10 probes, and 5 arms x 2 seeds is the only way to reach
    it with this field. A schedule that silently produced a different number
    would be a different experiment at the same price.
    """
    if rung not in ("screening", "confirmation"):
        raise D1BehaviouralError(f"unknown rung {rung!r}")
    seeds = (screening_seeds(repo_root) if rung == "screening"
             else confirmation_seeds(repo_root))
    field = arms(repo_root)
    role = f"d1_{rung}"
    out = tuple(
        Probe(probe_id=f"d1_{rung}_{a.arm_id}_s{seed}", arm_id=a.arm_id,
              seed=seed, rung=rung, checkpoint_dir=a.checkpoint_dir,
              battery_role=role)
        for a in field for seed in seeds)
    declared = int(behavioural_design(repo_root)[f"{rung}_probes"])
    if rung == "screening" and len(out) != declared:
        raise D1BehaviouralError(
            f"the schedule yields {len(out)} {rung} probes "
            f"({len(field)} arms x {len(seeds)} seeds) and the design declares "
            f"{declared}. One of them is wrong, and a schedule that disagreed "
            "with the priced design would run a different experiment at the "
            "same cost.")
    return out


def battery_role(role: str, repo_root: str | Path = REPO_ROOT
                 ) -> dict[str, Any]:
    """One realized role of the D-series family, with its identities.

    Raises when the role is not realized on disk or its bytes do not match the
    manifest. The whole point of a realized family is that a session consumes
    BYTES rather than a promise, and the capacity-record misreading showed how
    easily a claim about evidence substitutes for the evidence.
    """
    import hashlib

    root = Path(repo_root)
    family = _read(FAMILY_REL, repo_root)
    manifest = _read(MANIFEST_REL, repo_root)
    if family["status"] != "BUILT / VERIFIED":
        raise D1BehaviouralError(
            f"the D-series family reports status {family['status']!r}; this "
            "session consumes realized batteries, not a plan for them")
    if role not in manifest["roles"]:
        raise D1BehaviouralError(
            f"the realized family has no role {role!r}; it has "
            f"{sorted(manifest['roles'])}")
    declared = manifest["roles"][role]
    if declared["per_stratum"] != FROZEN_STRATA:
        raise D1BehaviouralError(
            f"{role} has stratum balance {declared['per_stratum']} and the "
            f"frozen mixture is {FROZEN_STRATA}. `correct_overall` is a mean "
            "over the mixture, so a different balance measures a different "
            "quantity.")
    base = root / BATTERY_ROOT_REL
    files: list[dict[str, Any]] = []
    for rel, rec in sorted(manifest["output_files"].items()):
        if not rel.startswith(f"{role}/"):
            continue
        path = base / rel
        if not path.is_file():
            raise D1BehaviouralError(f"{rel} is declared by the manifest and "
                                     "is not realized on disk")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != rec["sha256"]:
            raise D1BehaviouralError(
                f"{rel}: realized bytes hash to {got[:12]} and the manifest "
                f"declares {rec['sha256'][:12]}")
        files.append({"path": rel, "sha256": got, "n_items": rec["n_items"]})
    if len(files) != len(FROZEN_STRATA):
        raise D1BehaviouralError(
            f"{role} realized {len(files)} stratum files and the frozen "
            f"mixture has {len(FROZEN_STRATA)}")
    return {
        "family_id": family["family_id"],
        "family_content_id": manifest["family_content_id"],
        "allocation_rule_id": manifest["allocation_rule_id"],
        "role": role,
        "item_ids_sha256": declared["item_ids_sha256"],
        "n_prompts": declared["n_prompts"],
        "n_scorable": declared["n_scorable"],
        "per_stratum": dict(declared["per_stratum"]),
        "root": f"{BATTERY_ROOT_REL}/{role}",
        "files": files,
        "_capacity_blocker": family["capacity_source_blocker"],
        "_not_the_exhausted_pool": (
            "`d1_evidence_capacity.json` reports batteries_remaining 0 for the "
            "ORIGINAL C1 source populations, binding on MATH-500. This role "
            "was drawn from prospectively widened sources preserving the "
            "frozen stratum balance exactly, and the family's own record "
            "carries capacity_source_blocker CLOSED."),
    }


def roles_are_disjoint(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The two D1 roles share no prompt. HALF THE VALIDITY CONDITION.

    An advancing candidate is selected on the screening role and confirmed on
    the confirmation role, and the confirmation estimate is unbiased only
    because it does not reuse the prompts the selection saw. Checked by reading
    the realized item ids, not by trusting that two hashes differ.
    """
    root = Path(repo_root) / BATTERY_ROOT_REL
    shared: dict[str, int] = {}
    counts: dict[str, int] = {}
    for stratum in sorted(FROZEN_STRATA):
        def ids(role: str) -> set[str]:
            path = root / role / f"{stratum}.jsonl"
            if not path.is_file():
                raise D1BehaviouralError(f"{role}/{stratum}.jsonl is missing")
            return {json.loads(line)["id"]
                    for line in path.read_text().splitlines() if line.strip()}

        a, b = ids("d1_screening"), ids("d1_confirmation")
        counts[stratum] = len(a)
        overlap = a & b
        if overlap:
            shared[stratum] = len(overlap)
    if shared:
        raise D1BehaviouralError(
            f"the two D1 roles share prompts: {shared}. An advancing candidate "
            "selected on the screening role would then be confirmed partly on "
            "prompts the selection already saw.")
    return {"disjoint": True, "per_stratum_screening_ids": counts}


def require_arms_present(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Every arm's bytes exist and carry the identity the frozen record names.

    At `$0`, before anything is priced. A probe trained from the wrong
    checkpoint is a measurement of something nobody asked about, and it costs
    its full training time to discover.
    """
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.initialization.specs.artifact import identify_checkpoint

    register_builtin_adapters()
    adapter = get_adapter("qwen3")
    out: list[dict[str, Any]] = []
    for arm in arms(repo_root):
        directory = Path(arm.checkpoint_dir)
        if not directory.is_dir():
            raise D1BehaviouralError(
                f"{arm.arm_id}: no checkpoint at {directory}")
        config = json.loads((directory / "config.json").read_text())
        spec = ArchSpec.of("qwen3", {
            key: config[key] for key in (
                "hidden_size", "num_hidden_layers", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "vocab_size", "tie_word_embeddings")})
        identity = identify_checkpoint(
            directory, adapter=adapter, spec=spec,
            num_parameters=adapter.param_count(spec))
        #: The frozen record carries a 12-hex prefix for the incumbent and a
        #: full digest for the candidates, so the comparison is by prefix --
        #: on the RECORDED value's own length, never truncated to make a
        #: mismatch pass.
        want = arm.artifact_digest
        if not identity.artifact_digest.startswith(want):
            raise D1BehaviouralError(
                f"{arm.arm_id} at {directory} has artifact_digest "
                f"{identity.artifact_digest[:len(want)]} and the frozen record "
                f"names {want}. This is not the checkpoint that was selected.")
        out.append({"arm": arm.arm_id, "role": arm.role,
                    "quality_position": arm.quality_position,
                    "state_id": arm.state_id,
                    "artifact_digest": identity.artifact_digest,
                    "recorded_prefix": want,
                    "checkpoint_dir": str(directory),
                    "num_parameters": identity.num_parameters})
    return {"n_arms": len(out), "arms": out}


def session_contract(rung: str, repo_root: str | Path = REPO_ROOT
                     ) -> dict[str, Any]:
    """Everything this rung is bound to, derived and checked. `$0`.

    One document the driver asserts on the pod and the launcher writes before
    a resource exists, so "what was this session supposed to measure" has one
    answer that was true before it measured anything.
    """
    from experiments.recipes import E1_KD_HEAVY_0860K as recipe

    bd = behavioural_design(repo_root)
    role = f"d1_{rung}"
    seeds = (screening_seeds(repo_root) if rung == "screening"
             else confirmation_seeds(repo_root))
    scheduled = probes(rung, repo_root)
    return {
        "schema": "aadistill.phase_d1.behavioural_contract/v1",
        "rung": rung,
        "design_hash": design(repo_root)["design_hash"],
        "recovery_recipe": {
            "recipe_id": recipe.recipe_id,
            "declared_in_design": bd and design(repo_root)["recovery_recipe"],
            "tokens": recipe.tokens, "pack": recipe.pack,
            "pack_sha256": recipe.pack_sha256,
            "ce_weight": recipe.ce_weight, "kd_weight": recipe.kd_weight,
            "temperature": recipe.temperature, "kd_scope": recipe.kd_scope,
            "block_len": recipe.block_len,
        },
        "seeds": list(seeds),
        "n_probes": len(scheduled),
        "n_probes_declared": int(bd[f"{rung}_probes"]),
        "probes": [{"probe_id": p.probe_id, "arm": p.arm_id, "seed": p.seed}
                   for p in scheduled],
        "arms": require_arms_present(repo_root),
        "battery": battery_role(role, repo_root),
        "roles_disjoint": roles_are_disjoint(repo_root),
        "endpoint": bd["decision_rule"]["primary_endpoint"],
        "estimand": bd["decision_rule"]["estimand"],
        "inference": bd["decision_rule"]["inference"],
        "guardrails": bd["decision_rule"]["guardrails"],
        "rule": bd["decision_rule"]["rule"],
        "sesoi": bd["sesoi"],
        "_sesoi_is_carried_forward_not_remeasured": (
            "0.010 is C0's, and D-series absolute scores are not directly "
            "interchangeable with the historical C1 absolute scores because "
            "three capacity-limited strata draw from wider populations. The "
            "SESOI is carried forward as an explicit assumption about the "
            "within-family PAIRED difference, which is what this rung "
            "estimates: both arms are measured on the same role under the "
            "same protocol."),
        "_decides_nothing": (
            "this contract is derived from frozen records. The candidate field "
            "is the retention decision's, the recipe and seed counts are the "
            "design's, the battery is the realized family's, and the endpoint "
            "and SESOI are C0's."),
    }


__all__ = ["ARM_SOURCES", "Arm", "D1BehaviouralError", "FROZEN_STRATA",
           "Probe", "arms", "battery_role", "behavioural_design",
           "confirmation_seeds", "derive_seeds", "design", "excluded_seeds",
           "probes",
           "require_arms_present", "roles_are_disjoint", "screening_seeds",
           "session_contract"]
