"""The Phase-C2 Search-1 space, and a structural bound on what it costs.

    PYTHONPATH=src:scripts python -m experiments.phase_c2.search_space

Zero cost. It loads no model, reads no checkpoint and needs no GPU: the
branching comes from the real registry and the real `applicable_implementations`
and `expansion_profiles`, and the per-expansion minutes come from
Phase-B attempt 5's committed telemetry.

**What the space is.** Four operator kinds, one implementation each, ORDER FREE.
ATTENTION is fixed to the C1-selected `attention.activation_importance_v1` and
branches over both materialized profiles; DEPTH, FFN and RESIDUAL_WIDTH are held
at the Phase-B incumbent's mixtures. That restriction is the reason
`SearchConfig.impl_profiles` exists: putting a second profile in `profiles` used
to branch all four kinds, which is a full factorial — a different and far more
expensive experiment than the one C2 asks for.

**Why the bound is structural.** `children_max x mean node cost` is what
authorized Phase-B attempt 3 at a 1.91-7.51 h projection; it ran 9.08 h and did
not finish. Node cost varies with the parent's geometry and, much more
importantly, the beam decides HOW MANY expensive nodes there are. So `bound()`
does not multiply averages. It enumerates the real class space and takes the
exact minimum and maximum over every beam the ranking policy could return,
which is a claim about arithmetic rather than about how the Pareto front is
likely to fall.

`replay_phase_b()` feeds Phase-B attempt 5's real beams through the same
branching model and predicts every level's expansion count exactly (10/46/19/7)
and its minutes to within 0.2%, because a model that cannot reproduce the past
cannot bound the future.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aadistill.initialization.specs.arch import ArchSpec
from aadistill.initialization.planning.ranking import SCHEDULE_V1

#: The branching-and-cost machinery, shared with every other search instance.
#: Extracted when the C2 full joint re-search became a second consumer; the
#: arithmetic is unchanged and this module keeps its own numbers.
from experiments.search_cost_model import (
    Bound, CostModel, SearchSpace, compositions, decomposition, named_profiles,
    walk_leaves,
)
from experiments.search_cost_model import bound as _bound
from experiments.search_cost_model import level_children as _level_children
from experiments.search_cost_model import price as _price
from experiments.search_cost_model import trajectory as _trajectory

# --- identity ---------------------------------------------------------------
#
# Every value below is resolved from committed evidence, and
# `tests/autoinit/test_phase_c2_search_space.py` re-derives each one rather than
# restating it. Copying an identity out of a handoff message is how a search
# gets configured for a path nobody ran.

#: `logs/stages/stage-1/phase_b/runs/attempt5/stage1_selection.json`: the frozen
#: Phase-B winner `fe9683e6a9c783bbc6fe276a78c851c6` is
#: `DEPTH(domain_balanced) -> FFN(domain_balanced) -> RESIDUAL_WIDTH(reasoning_heavy@v2)
#: -> ATTENTION(calib.none@v1)`, artifact `c313d1b4081b…`.
PHASE_B_WINNER_STATE_ID = "fe9683e6a9c783bbc6fe276a78c851c6"
PHASE_B_WINNER_ARTIFACT_DIGEST = (
    "c313d1b4081b9a3b410dddf7a29ebcaad8dd0759179d51e1d761238c1743a2a6")

#: The C1 treatment, from `plans/execution_preregistration.json:fixed_path`. It
#: is the baseline C2 candidates are compared against, and it is a LEAF OF THIS
#: SEARCH: same order, same three held mixtures, ATTENTION on domain_balanced.
#: The search therefore re-derives it rather than importing it — deterministic
#: search plus identical seed reproduces a prior checkpoint byte for byte.
C1_TREATMENT_SPEC_HASH = (
    "3a233a9017b3b8a717ff18fc1aaa171dad84f36adc97920765d181ca98c53612")
C1_TREATMENT_PATH_LABEL = (
    "DEPTH(calib.domain_balanced@v1)->FFN(calib.domain_balanced@v1)->"
    "RESIDUAL_WIDTH(calib.reasoning_heavy@v2)->ATTENTION(calib.domain_balanced@v1)")

#: The four implementations, one per kind. `attention.weight_proxy_v0`,
#: `depth.positional_v0` and `composite.stage1_sandwich_v0` are deliberately
#: ABSENT: C2 does not reopen the Phase-B implementation search, and the
#: historical ATTENTION control is an anchor rather than a branch. Absent from
#: `allowed_impls` is the only way to say that — `BeamSearch._allowed_impl_ids`
#: falls back to the ENTIRE registry when it is None, which is how Phase A and
#: Phase B came to search everything registered.
C2_ALLOWED_IMPLS = (
    "attention.activation_importance_v1",
    "depth.causal_kl_greedy_v1",
    "ffn.activation_importance_v0",
    "width.global_pca_v0",
)

#: The two materialized mixtures. `calib.reasoning_heavy@v1` and
#: `calib.stage0_current@v1` are declared but unbuilt, and `resolve()` refuses
#: them, so they cannot be search branches.
C2_PROFILE_IDS = ("calib.domain_balanced@v1", "calib.reasoning_heavy@v2")

#: Search-1 varies ATTENTION's mixture and holds the other three at the
#: incumbent's values. Varying all four is the P=2 factorial Phase B ran; it
#: cost 9.08 h without finishing and is not what C2 asks.
C2_IMPL_PROFILES: dict[str, tuple[str, ...]] = {
    "depth.causal_kl_greedy_v1": ("calib.domain_balanced@v1",),
    "ffn.activation_importance_v0": ("calib.domain_balanced@v1",),
    "width.global_pca_v0": ("calib.reasoning_heavy@v2",),
    "attention.activation_importance_v1": C2_PROFILE_IDS,
}

#: Derived from the committed Phase-A journal: a level-0 ATTENTION child's
#: `arch_spec` is the teacher in every field its operator did not modify, and
#: its `steps[0].trace.q_heads` is `[32, 16]` — the parent's query-head count and
#: the target's. The test re-derives this from that record.
TEACHER_GEOMETRY: dict[str, Any] = dict(
    hidden_size=2560, num_hidden_layers=36, intermediate_size=9728,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True)

PHASE_B_TELEMETRY = (
    "logs/stages/stage-1/phase_b/runs/attempt5/search_telemetry.jsonl")
PHASE_B_RESULT = "logs/stages/stage-1/phase_b/runs/attempt5/search_result.json"

#: `NVIDIA L40S` `securePrice`, quoted live 2026-09-15: `$1.09/h`, stock Medium.
#: **`securePrice`, not `communityPrice`.** The same query returned `0.79` for
#: community, and reporting that figure once had a maintainer act on a price the
#: launcher would never pay — `check_gpu_offered` reads `securePrice` and aborts
#: above `--max-price`.
#:
#: For DISPLAY only. A launch re-quotes: an hour-old price is not a price.
PRICE_PER_HOUR_LAST_QUOTED = 1.09


# --- measured cost ----------------------------------------------------------
#
# Minutes per EXPANSION, end to end: operator + parent load + materialize +
# identify + canonical reload + validate + state_eval. Computed from
# PHASE_B_TELEMETRY, split by whether the parent is the root, because the
# teacher-width parent is the expensive one. The test recomputes the whole table
# from that file, so these are a cache and not a transcription.
#
# `max` is what `bound()` uses. A ceiling built on `mean` is not a ceiling.
MEASURED_MINUTES: dict[str, dict[str, float]] = {
    "depth.causal_kl_greedy_v1":    {"root_max": 34.35, "deeper_max": 31.10,
                                     "root_mean": 31.67, "deeper_mean": 26.65},
    "width.global_pca_v0":          {"root_max": 3.93, "deeper_max": 3.22,
                                     "root_mean": 3.82, "deeper_mean": 1.85},
    "ffn.activation_importance_v0": {"root_max": 2.59, "deeper_max": 2.91,
                                     "root_mean": 2.55, "deeper_mean": 1.71},
    "attention.weight_proxy_v0":    {"root_max": 3.84, "deeper_max": 3.86,
                                     "root_mean": 3.84, "deeper_mean": 1.44},
    "depth.positional_v0":          {"root_max": 3.73, "deeper_max": 2.56,
                                     "root_mean": 3.73, "deeper_mean": 1.43},
    "composite.stage1_sandwich_v0": {"root_max": 3.88, "deeper_max": 3.88,
                                     "root_mean": 3.84, "deeper_mean": 3.84},
}

#: `attention.activation_importance_v1` HAS NEVER RUN INSIDE A SEARCH. C1 ran it
#: through `fixed_path` only, where stage F took **14.4 s** end to end
#: (`runs/attempt18/evidence/c1_evidence.json`: stage E finished 06:04:49Z,
#: stage F 06:05:03Z) on an already-narrowed 28-layer/1024-hidden parent.
#:
#: So its in-search cost is priced from the most expensive MEASURED
#: ACTIVATION_STATS operator, `width.global_pca_v0`, times this factor. The
#: factor is a margin, not a measurement, and it is generous in the direction
#: the arithmetic can afford: per token the attention second moment accumulates
#: 36 x 32 x 128^2 = 18.9M MAC against the residual covariance's
#: 36 x 2560^2 = 236M, i.e. ~12x LESS accumulation work, and it reads no
#: statistics cache (the operator collects directly, by design, because its
#: quantities belong to ATTENTION_STATS_SPEC).
#:
#: This is the one unmeasured input to the price. It is called out here, in
#: `bound().unmeasured`, and in the plan document, rather than blended in.
ATTENTION_ACTIVATION_PROXY_IMPL = "width.global_pca_v0"
ATTENTION_ACTIVATION_PROXY_FACTOR = 1.5


#: This instance's cost model, handed to the shared machinery. The table and the
#: proxy rule are Search-1's facts; the arithmetic over them is not, and lives in
#: `experiments.search_cost_model`.
COST = CostModel(
    minutes=MEASURED_MINUTES,
    proxies={"attention.activation_importance_v1": (
        ATTENTION_ACTIVATION_PROXY_IMPL, ATTENTION_ACTIVATION_PROXY_FACTOR)},
    source=("Phase-B attempt 5 telemetry, plus one proxied row. See "
            "MEASURED_MINUTES and ATTENTION_ACTIVATION_PROXY_*."))


def minutes_for(impl_id: str, *, root: bool, statistic: str = "max") -> float:
    """Minutes for one expansion of `impl_id` on a root / deeper parent."""
    return COST.minutes_for(impl_id, root=root, statistic=statistic)


# --- the class space --------------------------------------------------------
#
# A state's expansion cost and branching depend only on WHICH implementations it
# has applied, never on which calibration profile fed them: every operator sets
# its field to the target's value, so the geometry is a function of the applied
# set. Two states differing only in ATTENTION's mixture are therefore one class
# with multiplicity two — and multiplicity is kept, because each of them is
# expanded and paid for separately.


#: `Space` is `SearchSpace` with this instance's family bound. The type, the
#: branching model and the bound all moved to `experiments.search_cost_model`
#: unchanged; what stays here is which implementations, which mixtures and which
#: measured minutes — the parts that are Search-1's rather than arithmetic.
def Space(*, allowed_impls, profile_ids, impl_profiles, teacher, target,
          family: str = "qwen3") -> SearchSpace:
    return SearchSpace(allowed_impls=tuple(allowed_impls),
                       profile_ids=tuple(profile_ids),
                       impl_profiles=impl_profiles, teacher=teacher,
                       target=target, family=family)


#: Kept as module names because callers and tests use them, and because a search
#: instance should be able to ask its own questions without importing two
#: modules. Each is the shared function with this instance's cost model applied
#: and this project's default schedule filled in.
_fake_profiles = named_profiles


def level_children(space, beam, *, statistic: str = "max"):
    return _level_children(space, beam, COST, statistic=statistic)


def bound(space, *, beam_width: int | None = None,
          warmup_levels: int | None = None, statistic: str = "max",
          max_depth: int | None = None) -> Bound:
    return _bound(
        space, COST, statistic=statistic, max_depth=max_depth,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=(SCHEDULE_V1.warmup_levels if warmup_levels is None
                       else warmup_levels))


# --- the two configured spaces ---------------------------------------------


def c2_search1_space() -> Space:
    return Space(
        allowed_impls=C2_ALLOWED_IMPLS,
        profile_ids=C2_PROFILE_IDS,
        impl_profiles=dict(C2_IMPL_PROFILES),
        teacher=ArchSpec.of("qwen3", TEACHER_GEOMETRY),
        target=_target_spec())


def phase_b_space() -> Space:
    """Phase B as it actually ran: the whole registry, P=2, no restriction.

    The back-test. `replay_phase_b()` checks that the observed run falls
    inside this space's bound; a model that cannot contain what happened is not
    a model of what might.
    """
    from aadistill.initialization.operators.register import BUILTIN_OPERATORS

    return Space(
        allowed_impls=tuple(sorted(i.impl_id for i in BUILTIN_OPERATORS)),
        profile_ids=C2_PROFILE_IDS,
        impl_profiles=None,
        teacher=ArchSpec.of("qwen3", TEACHER_GEOMETRY),
        target=_target_spec())


def _target_spec() -> ArchSpec:
    from phase_a_frozen import TARGET_GEOMETRY

    return ArchSpec.of("qwen3", TARGET_GEOMETRY)


DEPTH_IMPL = "depth.causal_kl_greedy_v1"

#: Non-search pod time for a search session, in minutes. Each figure is the one
#: this repository already plans with (`scripts/pod/autoinit_preflight_launch.py`
#: uses `setup_minutes=45.0`, `transfer_minutes=6.0` and the same named phases),
#: NOT the warm-image observations. C1 attempt 18 set up in 6 minutes; the same
#: script on the same image and card has also taken 8.5 and over 150. Budgeting
#: the lucky number is how a session runs out of money during teardown.
SESSION_PHASE_MINUTES: tuple[tuple[str, float], ...] = (
    ("setup_and_asset_staging", 45.0),
    ("bundle_transfer", 6.0),
    ("teacher_fetch_and_verify", 8.0),
    ("machine_gates", 22.0),
    ("selection_commit_and_artifact_manifest", 8.0),
    ("artifact_synchronization", 6.0),
)


def trajectory(space, *, prefer_depth: bool, beam_width: int | None = None,
               warmup_levels: int | None = None,
               statistic: str = "max") -> dict[str, Any]:
    """Cost ONE nominated beam trajectory. See the shared model's docstring.

    `prefer_depth` is this instance's name for `prefer_costly`: DEPTH *is* the
    expensive operator in this cost table, and the shared model now DERIVES that
    rather than being told, so `DEPTH_IMPL` below is an assertion about the table
    rather than an input to the arithmetic.
    """
    out = _trajectory(
        space, COST, prefer_costly=prefer_depth, statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=(SCHEDULE_V1.warmup_levels if warmup_levels is None
                       else warmup_levels))
    return {**out, "prefer_depth": out.pop("prefer_costly")}


def price(space, *, price_per_hour: float, authorized_usd: float,
          beam_width: int | None = None, statistic: str = "max"):
    """A `BudgetPlan` for a search-only session. Priced, which is not funded."""
    return _price(
        space, COST, price_per_hour=price_per_hour,
        authorized_usd=authorized_usd, session_phases=SESSION_PHASE_MINUTES,
        setup_phase="setup_and_asset_staging", transfer_phase="bundle_transfer",
        search_phase_name="beam_search_depth_early", statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=SCHEDULE_V1.warmup_levels)


#: Phase B's path labels name KINDS and profiles, not implementations, and the
#: mapping back is unambiguous in the registry Phase B searched: the
#: no-calibration sentinel identifies the `CalibrationNeed.NONE` implementation
#: of that kind, anything else identifies the calibrated one.
_PHASE_B_IMPL_OF = {
    ("DEPTH", "calib.none@v1"): "depth.positional_v0",
    ("DEPTH", "*"): "depth.causal_kl_greedy_v1",
    ("FFN", "*"): "ffn.activation_importance_v0",
    ("RESIDUAL_WIDTH", "*"): "width.global_pca_v0",
    ("ATTENTION", "calib.none@v1"): "attention.weight_proxy_v0",
    ("COMPOSITE_STAGE1", "*"): "composite.stage1_sandwich_v0",
}


def _class_of_path(label: str) -> frozenset[str]:
    """`FFN(calib.domain_balanced@v1)->DEPTH(...)` -> the applied impl ids."""
    applied = set()
    for step in label.split("->"):
        kind, _, rest = step.partition("(")
        profile = rest.rstrip(")")
        applied.add(_PHASE_B_IMPL_OF.get((kind, profile))
                    or _PHASE_B_IMPL_OF[(kind, "*")])
    return frozenset(applied)


def replay_phase_b(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Feed Phase-B attempt 5's ACTUAL beams through the branching model.

    The real back-test, and much stronger than checking that the observed run
    lands inside a range. A range can contain a run while being wrong about
    every level; this replays the beams the Pareto policy really returned and
    asks the model to predict each level's expansion count exactly, and its
    minutes to within the spread of the same telemetry the cost table came from.

    A model that cannot reproduce the past cannot bound the future.
    """
    result = json.loads((Path(repo_root) / PHASE_B_RESULT).read_text())
    space = phase_b_space()

    beam: tuple[tuple[frozenset[str], int], ...] = ((frozenset(), 1),)
    levels = []
    for record in result["levels"]:
        minutes, _partial, n = level_children(space, beam, statistic="mean")
        levels.append({
            "level": record["level"],
            "observed_generated": len(record["generated"]),
            "predicted_generated": n,
            "observed_minutes": round(record["seconds"] / 60, 2),
            "predicted_minutes": round(minutes, 2),
        })
        #: `ranking.selected` is a list of state ids; the path label lives on the
        #: matching `decisions` entry, which also carries the `selected` flag.
        ranking = record.get("ranking") or {}
        chosen = set(ranking.get("selected") or ())
        kept: dict[frozenset[str], int] = {}
        for decision in ranking.get("decisions", []):
            if decision["state_id"] not in chosen and not decision.get("selected"):
                continue
            cls = _class_of_path(decision["path"])
            kept[cls] = kept.get(cls, 0) + 1
        beam = tuple(sorted(kept.items(), key=lambda kv: tuple(sorted(kv[0]))))

    return {
        "levels": levels,
        "observed_expansions": sum(L["observed_generated"] for L in levels),
        "predicted_expansions": sum(L["predicted_generated"] for L in levels),
        "observed_minutes": round(
            sum(L["observed_minutes"] for L in levels), 2),
        "predicted_minutes": round(
            sum(L["predicted_minutes"] for L in levels), 2),
        "every_level_exact": all(L["observed_generated"]
                                 == L["predicted_generated"] for L in levels),
    }


def register_c2_operators() -> None:
    """Everything this space needs resolved, registered explicitly.

    `attention.activation_importance_v1` is NOT a shipped default:
    `operators/register.py` omits it so a search enumerating the registry cannot
    pick it up by accident. A C2 search names it in `allowed_impls`, and
    `_allowed_impl_ids` validates that list against the registry — so it must be
    registered first or the search refuses.

    The adapter bootstrap is here for the same reason and was forgotten once
    already: `get_adapter("qwen3")` raises on an empty registry, and C1 attempt
    16 died at stage D on exactly that, on a paid pod. Registration being
    explicit is correct; assuming somebody else did it is not.
    """
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators import attention_activation
    from aadistill.initialization.operators.register import (
        register_builtin_operators)

    register_builtin_adapters()
    register_builtin_operators()
    attention_activation.register()


def main() -> int:
    register_c2_operators()
    space = c2_search1_space()
    print("Phase-C2 Search-1 — the configured space\n")
    print(f"  allowed implementations : {len(C2_ALLOWED_IMPLS)}")
    for impl_id in C2_ALLOWED_IMPLS:
        profiles = C2_IMPL_PROFILES.get(impl_id, C2_PROFILE_IDS)
        print(f"     {impl_id:38} {', '.join(profiles)}")
    print(f"  order                   : free ({len(space.teacher.diff(space.target))} "
          "differing fields, one per kind)")
    print(f"  beam                    : width {SCHEDULE_V1.width}, "
          f"{SCHEDULE_V1.warmup_levels} warmup level(s)")

    for statistic in ("mean", "max"):
        b = bound(space, statistic=statistic)
        early = trajectory(space, prefer_depth=True, statistic=statistic)
        print(f"\n  bound on {statistic:4} per-expansion minutes")
        print(f"     expansions   : {b.min_expansions} .. {b.max_expansions}")
        print(f"     minutes      : {b.min_minutes:.1f} .. {b.max_minutes:.1f}"
              f"   ({b.min_minutes / 60:.2f} .. {b.max_minutes / 60:.2f} h)")
        print(f"     DEPTH-early  : {early['minutes']:.1f} min "
              f"({early['hours']:.2f} h) — the trajectory with evidence, "
              "not a bound")
        if b.unmeasured:
            print(f"     UNMEASURED   : {', '.join(b.unmeasured)} — priced at "
                  f"{ATTENTION_ACTIVATION_PROXY_FACTOR}x "
                  f"{ATTENTION_ACTIVATION_PROXY_IMPL}")

    print(f"\n  beam-width menu at ${PRICE_PER_HOUR_LAST_QUOTED}/h "
          "(max statistic)")
    print(f"     {'width':>5} {'expansions':>12} {'DEPTH-early h':>14} "
          f"{'worst h':>9} {'expected $':>11} {'ceiling $':>10}")
    for width in (3, 4, 5, 6):
        b = bound(space, beam_width=width, statistic="max")
        early = trajectory(space, prefer_depth=True, beam_width=width,
                           statistic="max")
        plan = price(space, price_per_hour=PRICE_PER_HOUR_LAST_QUOTED,
                     authorized_usd=10_000.0, beam_width=width)
        mark = " <- SCHEDULE_V1" if width == SCHEDULE_V1.width else ""
        print(f"     {width:>5} {b.min_expansions:>5}..{b.max_expansions:<5} "
              f"{early['hours']:>14.2f} {b.max_minutes / 60:>9.2f} "
              f"{plan.expected_usd:>11.4f} {plan.hard_terminate_usd:>10.4f}"
              f"{mark}")

    back = replay_phase_b()
    print("\n  back-test: Phase-B attempt 5's real beams through this model")
    print(f"     {'level':>5} {'generated':>22} {'minutes':>20}")
    for L in back["levels"]:
        print(f"     {L['level']:>5} "
              f"{L['observed_generated']:>10} obs /{L['predicted_generated']:>4} pred "
              f"{L['observed_minutes']:>8.1f} /{L['predicted_minutes']:>8.1f}")
    print(f"     total {back['observed_expansions']:>10} obs /"
          f"{back['predicted_expansions']:>4} pred "
          f"{back['observed_minutes']:>8.1f} /{back['predicted_minutes']:>8.1f}")
    print(f"     every level exact: {back['every_level_exact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
