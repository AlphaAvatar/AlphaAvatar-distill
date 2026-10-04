"""The D-series behavioural battery family — six disjoint roles, one frozen rule.

**What this replaces.** A3 reused `c1_confirmation_v1`, and the D-series cannot:
B was PROMOTED on that battery and C2's and C3's negative results were measured
on it, so it must stay held out. The pool under the frozen C1 mixture is also
exhausted — `math_verified` holds 70 eligible items of the 150 one battery needs
(`logs/stages/stage-1/phase_d1/analyses/d1_evidence_capacity.json`). The
alternative to a family is a one-off workaround per experiment, decided after
each result is known, which is how a held-out battery stops being held out.

So the allocation is frozen **now**, before any D1 outcome exists.

**The six roles**, in a fixed build order, every one disjoint from every other
and from every historical role:

    d1_screening   d1_confirmation
    d2_screening   d2_confirmation
    d3_screening   d3_confirmation

**This is a NEW behavioural distribution, not a continuation of C1's.** The
stratum names, domains and per-battery counts are inherited unchanged from the
C1 mixture (imported below, never restated), so the high-level balance and the
950/850 denominators are preserved. But the POPULATION three strata are drawn
from has to grow — `math_verified`, `code` and `gsm8k` cannot fund six batteries
from their currently pinned sources — and a different population is a different
distribution. It therefore carries its own identity, `d_series_behavioural_v1`,
and must not be described as, or compared against, the `c1_confirmation_v1`
distribution. Concretely: B's historical C1 score is NOT imported, and the C0
SESOI's transfer to this population is an explicit recorded assumption rather
than an inherited fact.

**Why that is scientifically acceptable.** Every D-series comparison re-measures
the challenger AND the incoming incumbent on the same fresh battery, under the
same recovery budget and the same protocol, in one session. No D-series decision
reads a historical score, so no D-series decision needs comparability with the
C1 population.

**The allocation mechanism is the one that already exists.** `battery_render`'s
`rank_take` orders an eligible pool by
`SHA256(base_digest : rank_domain : stratum : stable_id)` and takes the first
`n`. It is content-derived — no seed, no date, nothing an agent could choose
after seeing a result — and a distinct `rank_domain` per role gives each role an
independent sample rather than the next slice of another role's ordering, which
is exactly how `c2_screening_v1` was drawn beside `c1_confirmation_v1`. This
module adds no selection code; it declares the six rank domains, the build
order, and the exclusion chain.

**Two identities, because they are frozen at different times.**

``allocation_rule_id``
    the rule: role order, rank domains, strata and counts, the exclusion chain,
    the determinism contract. Computable TODAY, with no pool and no source
    decision, which is what makes "frozen before any outcome" checkable rather
    than asserted.
``family_content_id``
    the realized batteries: source pins plus each role's item-id and
    normalized-content digests. Computable only at materialization. Absent here
    by construction.

**This module materializes nothing and authorizes nothing.** It derives what the
family requires and reports the gap. Pinning new sources is a maintainer
decision — it changes the evaluation population, needs a licence and
contamination record, and the directive for this round forbids spending.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/data"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

SCHEMA = "aadistill.autoinit.d_series_battery_family/v1"

#: The distribution's own identity. NOT `c1_confirmation`: the stratum balance is
#: inherited, the population is not.
FAMILY_ID = "d_series_behavioural_v1"

#: The committed capacity measurement this module reads rather than recomputing.
#: Recomputing would need the HF snapshots and would silently disagree with the
#: record the D1 blocker is stated from.
CAPACITY_RECORD = "logs/stages/stage-1/phase_d1/analyses/d1_evidence_capacity.json"

#: THE SIX ROLES, in build order. Each entry is
#: `(role_name, rank_domain, experiment, purpose)`.
#:
#: The `rank_domain` strings are frozen: `rank_key` hashes them, so a typo would
#: silently produce a different sample of the same pool rather than an error.
#: They are pinned by a test for that reason.
#:
#: The ORDER is part of the rule because the exclusion chain depends on it (see
#: `exclusion_chain`), and because a reader has to be able to check that it was
#: fixed before any result existed.
ROLES: tuple[tuple[str, str, str, str], ...] = (
    ("d1_screening", "d-series-v1-d1-screening", "D1",
     "rank the Top-K candidates of the D1 target-aware search"),
    ("d1_confirmation", "d-series-v1-d1-confirmation", "D1",
     "confirm the advanced D1 candidate against the incoming incumbent"),
    ("d2_screening", "d-series-v1-d2-screening", "D2",
     "rank the Top-K candidates of the D2 confidence-weighted search"),
    ("d2_confirmation", "d-series-v1-d2-confirmation", "D2",
     "confirm the advanced D2 candidate against the incoming incumbent"),
    ("d3_screening", "d-series-v1-d3-screening", "D3",
     "rank the Top-K candidates of the D3 search"),
    ("d3_confirmation", "d-series-v1-d3-confirmation", "D3",
     "confirm the advanced D3 candidate against the incoming incumbent"),
)

#: The historical roles every D-series battery is isolated from, in addition to
#: the five baseline isolation roles the C1 builder already excludes
#: (`final_promotion`, `recovery_search`, `recovery_training`,
#: `initializer_state_eval`, `operator_calibration`).
#:
#: Both behavioural batteries that have been drawn are here. `c1_confirmation_v1`
#: must remain held out because B was promoted on it and C2's, C3's and A3's
#: results were measured on it; `c2_screening_v1` because it was drawn from the
#: same pools and reusing it would re-measure prompts a selection already saw.
HELD_OUT_BATTERIES: tuple[str, ...] = (
    "artifacts/stage3/c1_confirmation_v1",
    "artifacts/stage3/c2_screening_v1",
)

#: How many roles the family has to fund. Named rather than `len(ROLES)` at the
#: call sites so the arithmetic below reads as a requirement.
N_ROLES = len(ROLES)


def strata() -> dict[str, tuple[str, int, bool]]:
    """The C1 mixture, imported rather than restated.

    `stratum -> (domain, per_battery, scorable)`. Restating it is how two copies
    of a mixture drift apart; `build_c2_screening_battery` imports it for the
    same reason.
    """
    import build_c1_confirmation_battery as c1

    return dict(c1.SETS)


def _capacity() -> dict[str, Any]:
    return json.loads((REPO_ROOT / CAPACITY_RECORD).read_text())


def requirement() -> dict[str, Any]:
    """What six roles need per stratum, against what is eligible today.

    **The number no existing record computes.** `d1_evidence_capacity` asks how
    many whole batteries remain under the frozen mixture and answers zero,
    binding on `math_verified`. That is the right question for D1 alone and the
    wrong one for the family: a six-role allocation needs `6 x per_battery` from
    every stratum at once, and at that scale three strata are short, not one.
    """
    capacity = _capacity()
    eligible = {name: int(row["eligible"])
                for name, row in capacity["strata"].items()}
    out: dict[str, Any] = {}
    for stratum, (domain, take, scorable) in sorted(strata().items()):
        needed = take * N_ROLES
        have = eligible[stratum]
        out[stratum] = {
            "domain": domain,
            "scorable": bool(scorable),
            "per_battery": take,
            "needed_for_six_roles": needed,
            "eligible_today": have,
            "short_by": max(0, needed - have),
            "roles_fundable_today": have // take,
            "source_today": capacity["strata"][stratum]["source"],
        }
    return out


#: Candidate resolutions for the short strata. **OPTIONS, not pins.** No
#: revision, size or digest is asserted here, because asserting one without
#: fetching and hashing it would be inventing provenance — and this round
#: neither fetches nor spends. Each entry says what must be measured before it
#: can become a pin.
#:
#: `same_repo` matters: an option that only reads a different FILE of an
#: already-pinned repository inherits its licence and its renderer, so it is a
#: much smaller decision than a new dataset.
SOURCE_OPTIONS: dict[str, tuple[dict[str, Any], ...]] = {
    "math_verified": (
        {"option": "full Hendrycks MATH rather than the MATH-500 subset",
         "same_repo": False,
         "candidate": "the upstream Hendrycks MATH release (~12.5k problems)",
         "licence": "MIT per the upstream repository — TO BE VERIFIED at pinning",
         "must_be_measured": (
             "a revision pin and per-file digest; the eligible count after the "
             "full exclusion chain; renderer parity against the current "
             "`make_math_verified` so a D-series math item is rendered exactly "
             "as a C1 one was; and whether MATH-500's 500 items are a subset of "
             "it, which they are expected to be and which the exclusion chain "
             "handles either way"),
         "note": ("MATH-500 is a 500-item subset of the MATH test split, so the "
                  "larger source is the SAME population sampled less sparsely. "
                  "That is the mildest of the three changes and is still a "
                  "population change: the subset was itself selected.")},
    ),
    "code": (
        {"option": "the remaining splits of the already-pinned MBPP repository",
         "same_repo": True,
         "candidate": "google-research-datasets/mbpp, splits other than full/test",
         "licence": "inherited from the existing pin",
         "must_be_measured": (
             "the eligible count after exclusions. MBPP holds ~974 problems in "
             "total against the 500 of `full/test`, so the remainder may NOT "
             "cover the shortfall; if it does not, either a second code source "
             "or a reduced behaviour-only code component is a maintainer "
             "decision, and the second changes the 950 denominator of "
             "`usable_rollout_rate`")},
        {"option": "a second verified code source",
         "same_repo": False,
         "candidate": "undecided",
         "licence": "TO BE VERIFIED",
         "must_be_measured": (
             "everything a new source needs: licence, revision, digests, "
             "renderer, contamination against the training corpus, and an "
             "argument that it is the same KIND of task as MBPP")},
    ),
    "gsm8k": (
        {"option": "the train split of the already-pinned GSM8K repository",
         "same_repo": True,
         "candidate": "openai/gsm8k, main/train (~7.5k rows)",
         "licence": "inherited from the existing pin",
         "must_be_measured": (
             "the eligible count after exclusions — and this one matters, "
             "because the recovery TRAINING corpus may draw from the same "
             "split. The exclusion chain removes any overlap automatically, so "
             "the risk is not contamination but capacity: the surviving count "
             "has to be derived, not assumed from the row count"),
         "note": ("the shortfall is 11 items, the smallest of the three, and "
                  "this option needs no new dataset — only a second file of a "
                  "repository already pinned at a frozen revision")},
    ),
}


def exclusion_chain() -> dict[str, Any]:
    """What each role excludes — the part of the rule that makes six disjoint.

    Distinct rank domains give independent samples, not disjoint ones: two
    independent draws from one pool overlap. Disjointness comes from building in
    a fixed order with each role excluding every role before it, which is
    exactly how `c2_screening_v1` was drawn against `c1_confirmation_v1`.

    Both identity kinds are excluded at every link, because either alone is
    insufficient: a stable id misses the same question reaching the pool twice
    under two ids, and a normalized-content hash misses a paraphrase carrying one
    id. `rank_take` enforces both, and `verify_c1_battery_isolation` re-derives
    them afterwards against the roles as they exist then.
    """
    chain = []
    for index, (role, domain, _experiment, _purpose) in enumerate(ROLES):
        chain.append({
            "role": role,
            "rank_domain": domain,
            "excludes": ["the five baseline isolation roles",
                         *HELD_OUT_BATTERIES,
                         *[prior[0] for prior in ROLES[:index]]],
        })
    return {
        "order_is_part_of_the_rule": True,
        #: THREE coordinates in v2. The first two are the historical chain's; the
        #: third is what the 2026-10-03 source round established as necessary.
        "identity_kinds": [
            "historical native/source identity (stable id, including source_key)",
            "historical rendered-prompt exact identity "
            "(sha256 of norm(prompt_text))",
            "canonical problem-content identity "
            "(sha256 of norm(<problem>), experiments.phase_d_series.identity)",
        ],
        "all_three_required": (
            "each misses what the others catch. An id misses one problem "
            "entering the pool under two ids. A rendered-prompt hash misses the "
            "same problem rendered differently -- MBPP full/train task 602 and "
            "consumed full/test task 217 are the same problem and differ in both "
            "task_id and rendered tests. And neither sees a problem that reached "
            "the recovery-training corpus: the historical chain hashes a "
            "session's non-assistant messages JOINED, which can never equal a "
            "bare problem, so it caught 0 of the 1,708 GSM8K train rows that are "
            "in the corpus."),
        "problem_content_payload": {
            "code": "the bare problem text", "gsm8k": "the question",
            "math_verified": "the problem statement",
            "_owner": "experiments.phase_d_series.identity.PROBLEM_FIELD",
        },
        "reserved_populations_for_problem_content": [
            "the five baseline isolation roles",
            *HELD_OUT_BATTERIES,
            "the recovery-training corpus, by FIRST USER PROBLEM -- not the "
            "historical joined system+user hash",
            "every prior D-series role",
        ],
        "historical_chain_is_unmodified": (
            "`build_c1_confirmation_battery.excluded_identities` keeps its exact "
            "semantics, because frozen C1/C2/C3 battery membership is "
            "reproducible only from it. The third coordinate is a D-series layer "
            "ON TOP, not a repair of the historical function."),
        "chain": chain,
    }


def _exclusion_chain_v1() -> dict[str, Any]:
    """v1's exclusion block verbatim, so v1's id stays reproducible."""
    chain = []
    for index, (role, domain, _experiment, _purpose) in enumerate(ROLES):
        chain.append({
            "role": role,
            "rank_domain": domain,
            "excludes": ["the five baseline isolation roles",
                         *HELD_OUT_BATTERIES,
                         *[prior[0] for prior in ROLES[:index]]],
        })
    return {
        "order_is_part_of_the_rule": True,
        "identity_kinds": ["stable id (including source_key)",
                           "normalized prompt content (sha256 of norm(text))"],
        "both_required": (
            "either kind alone is insufficient: ids miss one question entering "
            "the pool twice, content hashes miss a paraphrase under one id"),
        "chain": chain,
    }


def allocation_rule_id_of(rule: dict[str, Any]) -> str:
    """`H(rule)[:32]` — one hashing function for every rule version."""
    from aadistill.infrastructure.manifest import sha256_json

    return sha256_json(rule)[:32]


#: ALLOCATION RULE VERSION. v1 froze a rule with two isolation coordinates and a
#: ranking keyed on the HISTORICAL stable id. The 2026-10-03 source round
#: established that contract is insufficient and added canonical
#: `problem_content_id` as a third load-bearing coordinate -- and new D-series
#: rows use a split-aware item identity, which is what the ranking hashes. Both
#: changes decide WHICH candidate wins the ranking and WHETHER a candidate may
#: enter it, so neither belongs in `family_content_id` at materialization: they
#: belong in the prospectively frozen rule.
#:
#: v1 is NOT mutated. Its shape is preserved by `allocation_rule_v1` so its id
#: stays reproducible, and it is recorded as superseded with the reason. Silently
#: changing what a committed hash meant is the one thing a prospective freeze
#: cannot survive.
ALLOCATION_RULE_VERSION = 2
#: v1's id as it was COMMITTED, so `allocation_rule_v1` is checked against the
#: value a reader may already hold rather than against itself. Factoring v1's
#: `distribution_identity` into a shared helper left a trailing comma, which made
#: the return value a one-element tuple and moved the hash to 8d266c70 -- caught
#: only because this value existed to compare against. A preserved-for-
#: reproducibility function with nothing pinning it is not preserved.
V1_RULE_ID_AS_COMMITTED = "ced017a1f3f155ba5aaf383e61156c12"


def _frozen_review() -> dict[str, Any]:
    """The duplicate review, bound into the rule because it gates entry.

    Imported from `identity.py` rather than restated: the decision has one owner
    and a second copy here would be the duplicate-source-of-truth problem the
    source round just finished removing.
    """
    from experiments.phase_d_series.identity import (
        REVIEW_PROVENANCE,
        SEMANTIC_DUPLICATE_EXCLUSIONS,
    )

    return {
        "_why_in_the_rule": (
            "a reviewed exclusion decides WHETHER a candidate may enter the "
            "ranking, so it is part of the prospective rule and not of the "
            "realized content."),
        "provenance": dict(REVIEW_PROVENANCE),
        "exclusions": {group: [dict(d) for d in rows]
                       for group, rows in sorted(
                           SEMANTIC_DUPLICATE_EXCLUSIONS.items()) if rows},
        "retained_explicitly": (
            "the other 43 reviewed pairs are RETAINED by decision, not by "
            "default. A similarity cutoff would have dropped all 44."),
        "threshold_role": (
            "the 0.8 Jaccard trigger selected which pairs a human read and "
            "excluded nothing by itself."),
    }


def _source_policy() -> dict[str, Any]:
    """Which population each short stratum draws from. Decided 2026-10-03."""
    from experiments.phase_d_series import math_source as ms

    return {
        "_why_in_the_rule": (
            "the population a stratum draws from determines the candidate pool, "
            "so it decides which rows can win the ranking at all. The per-file "
            "digests stay with `family_content_id`; the POLICY is here."),
        "code": ("google-research-datasets/mbpp, the already-pinned revision, "
                 "full/train + full/validation + full/prompt"),
        "gsm8k": "openai/gsm8k, the already-pinned revision, main/train",
        "math_verified": (f"{ms.REPO_ID} @ {ms.REVISION}, TEST splits only, "
                          f"licence {ms.LICENCE}; the canonical MATH test "
                          "population"),
        "excluded_populations": [
            "gsm8k `socratic` -- the same problems re-rendered, not capacity",
            "mbpp `sanitized` -- a verified SUBSET of `full`",
            "MATH train -- not used for this behavioural stratum",
            "any second code dataset -- MBPP's own splits suffice",
        ],
        "status": "DECIDED and IMPLEMENTED; see "
                  "logs/shared/analyses/autoinit_d_series_source_evidence.json",
    }


def _distribution_identity() -> dict[str, Any]:
    return {
        "id": FAMILY_ID,
        "is_not": "c1_confirmation",
        "why": ("three strata must draw from larger populations, and a "
                "different population is a different behavioural "
                "distribution. The stratum names, domains and counts are "
                "inherited unchanged, so the high-level balance and the "
                "950/850 denominators are preserved — but a score on this "
                "family is NOT comparable with a score on "
                "c1_confirmation_v1, and B's historical C1 number is not "
                "imported."),
        "sesoi_transfer": (
            "the C0 SESOI of 0.010 is CARRIED FORWARD AS AN ASSUMPTION, not "
            "as an inherited measurement. It was characterized on the C1 "
            "population. Every D-series comparison re-measures both arms on "
            "this family under one protocol, so the SESOI is used as a "
            "decision boundary for a within-family paired difference — "
            "which is the use it can support — and not as a claim that a "
            "difference measured here equals one measured on C1."),
    }


def allocation_rule_v1() -> dict[str, Any]:
    """The v1 rule verbatim, so its committed id remains reproducible.

    Kept as a function rather than a recorded constant: a hand-copied hash would
    be a second source of truth for what v1 meant, and the point of keeping v1 is
    that a reader can recompute it.
    """
    return {
        "schema": SCHEMA,
        "family_id": FAMILY_ID,
        "n_roles": N_ROLES,
        "roles": [{"role": r, "rank_domain": d, "experiment": e, "purpose": p}
                  for r, d, e, p in ROLES],
        "mixture": {name: take for name, (_d, take, _s) in sorted(
            strata().items())},
        "mixture_source": ("imported from build_c1_confirmation_battery.SETS; "
                           "never restated"),
        "n_prompts_per_role": sum(take for _d, take, _s in strata().values()),
        "n_scorable_prompts_per_role": sum(
            take for _d, take, scorable in strata().values() if scorable),
        "selection": {
            "mechanism": "battery_render.rank_take",
            "order": "ascending SHA256(base_digest : rank_domain : stratum : "
                     "stable_id), ties broken by str(id)",
            "base_digest": "the C0 digest the C1 and C2 batteries were ordered "
                           "from, so the family's ordering is derived from a "
                           "constant that predates every D-series result",
            "outcome_dependence": "NONE. The order is a function of the pool and "
                                  "the frozen domain strings; there is no seed, "
                                  "no date and no choice available after a "
                                  "result is seen",
            "deterministic": True,
        },
        "exclusion": _exclusion_chain_v1(),
        "what_is_NOT_in_this_rule": (
            "the source pins and the realized item lists. The sources for the "
            "three short strata are an open maintainer decision, and the items "
            "do not exist. Binding them here would make the rule unfreezable "
            "until materialization, which would defeat the point of freezing "
            "it before any D1 outcome. They are bound by `family_content_id` at "
            "materialization instead."),
        "distribution_identity": _distribution_identity(),
    }


def allocation_rule() -> dict[str, Any]:
    """The rule, frozen prospectively — everything except the realized items."""
    return {
        "allocation_rule_version": ALLOCATION_RULE_VERSION,
        "_supersedes": {
            "version": 1,
            "id": allocation_rule_id_of(allocation_rule_v1()),
            "why": (
                "v1 bound two isolation coordinates and ranked on the HISTORICAL "
                "stable id. The 2026-10-03 source round established that "
                "contract is insufficient -- MBPP full/train task 602 is the "
                "same problem as consumed full/test task 217 and passes both of "
                "v1's coordinates, and the historical chain's training-corpus "
                "content exclusion catches 0 of the 1,708 GSM8K train rows that "
                "are in the corpus. v1 is not mutated; it is superseded, and its "
                "id above is recomputed from its preserved shape rather than "
                "transcribed."),
        },
        "schema": SCHEMA,
        "family_id": FAMILY_ID,
        "n_roles": N_ROLES,
        "roles": [{"role": r, "rank_domain": d, "experiment": e, "purpose": p}
                  for r, d, e, p in ROLES],
        "mixture": {name: take for name, (_d, take, _s) in sorted(
            strata().items())},
        "mixture_source": ("imported from build_c1_confirmation_battery.SETS; "
                           "never restated"),
        "n_prompts_per_role": sum(take for _d, take, _s in strata().values()),
        "n_scorable_prompts_per_role": sum(
            take for _d, take, scorable in strata().values() if scorable),
        "selection": {
            "mechanism": "battery_render.rank_take",
            "order": "ascending SHA256(base_digest : rank_domain : stratum : "
                     "stable_id), ties broken by str(id)",
            "base_digest": "the C0 digest the C1 and C2 batteries were ordered "
                           "from, so the family's ordering is derived from a "
                           "constant that predates every D-series result",
            "rank_domains": {role: domain for role, domain, _e, _p in ROLES},
            #: THE SEMANTICS OF `stable_id`, bound because it decides the order.
            "ranking_stable_id_semantics": {
                "for_new_d_series_rows": (
                    "the D-SERIES SPLIT-AWARE ITEM IDENTITY -- "
                    "`experiments.phase_d_series.identity.d_series_item_id`. "
                    "mbpp-<config>-<split>-<task_id>, "
                    "gsm8k-<config>-<split>-<NNNNN>, "
                    "math-<config>-<split>-<NNNNN>"),
                "for_historical_rows": (
                    "unchanged. No frozen battery's item is renamed, and the "
                    "historical renderers keep their ids exactly."),
                "_why_this_is_in_the_RULE": (
                    "the ranking hashes the stable id, so changing the id scheme "
                    "changes the order and therefore WHICH rows are selected. "
                    "That cannot be deferred to `family_content_id` at "
                    "materialization: by then the selection has happened."),
                "_why_the_historical_scheme_cannot_be_reused": (
                    "`make_gsm8k` builds its id from the row's POSITION in the "
                    "file, so applied to main/train it emits 1,319 ids that "
                    "collide with the consumed test split and name different "
                    "problems. `mbpp-test-<task_id>` names the wrong split for "
                    "three of MBPP's four files."),
            },
            "outcome_dependence": "NONE. The order is a function of the pool and "
                                  "the frozen domain strings; there is no seed, "
                                  "no date and no choice available after a "
                                  "result is seen",
            "deterministic": True,
        },
        "exclusion": exclusion_chain(),
        "review": _frozen_review(),
        "source_policy": _source_policy(),
        "what_is_NOT_in_this_rule": (
            "the per-file source DIGESTS and the realized item lists. Those are "
            "bound by `family_content_id` at materialization, because they "
            "describe what was drawn rather than how. Everything that decides "
            "WHICH candidate wins the ranking, or WHETHER a candidate may enter "
            "it, is above -- which is the correction v2 makes to v1."),
        "distribution_identity": _distribution_identity(),
    }


def allocation_rule_id() -> str:
    """`H(the rule)`, computable today.

    This is what "frozen before any D1 outcome is observed" means operationally:
    the hash can be committed now, and a materialization that produced different
    roles, a different order, a different exclusion chain, a different ranking
    identity, a different review decision or a different source policy could not
    reproduce it.

    **v2.** The id below is NOT v1's. v1 is superseded for a stated reason and its
    own id remains recomputable from `allocation_rule_v1`, so nothing that was
    committed under v1 is silently reinterpreted.
    """
    return allocation_rule_id_of(allocation_rule())


def report() -> dict[str, Any]:
    need = requirement()
    short = {name: row["short_by"] for name, row in need.items()
             if row["short_by"] > 0}
    capacity = _capacity()
    return {
        "schema": SCHEMA,
        "family_id": FAMILY_ID,
        "status": "DESIGNED / NOT MATERIALIZED / BLOCKED ON SOURCES",
        "allocation_rule": allocation_rule(),
        "allocation_rule_id": allocation_rule_id(),
        "family_content_id": None,
        "_family_content_id_is_null": (
            "it binds the source pins and the realized item digests, neither of "
            "which exists. A non-null value here would be fabricated."),
        "requirement": need,
        "short_strata": short,
        "roles_fundable_today": min(row["roles_fundable_today"]
                                    for row in need.values()),
        "binding_stratum": min(need, key=lambda s: need[s]["roles_fundable_today"]),
        "capacity_record": {
            "path": CAPACITY_RECORD,
            "binding_stratum_for_one_battery": capacity["binding_stratum"],
            "batteries_remaining_for_one": capacity["batteries_remaining"],
            "_why_the_two_disagree": (
                "the record asks how many whole batteries remain and answers "
                "zero, binding on math_verified. At six roles the question "
                "changes: gsm8k funds five roles and code two, so extending "
                "math alone would unblock D1 and still leave the family short."),
        },
        "source_options": SOURCE_OPTIONS,
        "shortfall_against_the_original_pins": {
            "_what": ("how far each stratum fell short of six roles under the "
                      "ORIGINAL pinned files. Kept because it is what the "
                      "source decision was taken to resolve."),
            **{k: v for k, v in sorted(short.items())},
        },
        "blocker": (
            "CAPACITY IS CLOSED. The source decision of 2026-10-03 extended all "
            "three short strata and the strengthened chain's measured eligible "
            "counts clear every shortfall -- see "
            "logs/shared/analyses/autoinit_d_series_source_evidence.json, which "
            "owns those figures. The remaining blockers are NOT capacity: the "
            "GSM8K split-aware RENDERER does not exist, so its realized "
            "membership is not final, and the MATERIALIZATION decision itself "
            "has not been taken. No row is drawn."),
        "_authorizes": "nothing",
        "_materializes": "nothing",
        "what_this_may_not_be_used_to_claim": [
            "that the six batteries exist. None is built.",
            "that a D-series score is comparable with a C1 score. The "
            "population differs and the family carries its own identity.",
            "that the SESOI has been characterized on this population. It is "
            "carried forward as a recorded assumption.",
            "that any source is pinned. The options above are candidates with "
            "unverified licences and no revisions.",
        ],
    }


#: Where the committed record lives. `logs/shared/analyses/` is the declared
#: home for AutoInit program material spanning the Stage-1 experiments, which a
#: family owned by three of them is; the `autoinit_` prefix is that directory's
#: convention.
RECORD = "logs/shared/analyses/autoinit_d_series_battery_family.json"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"regenerate {RECORD}")
    args = parser.parse_args(argv)

    doc = report()
    if args.write:
        out = REPO_ROOT / RECORD
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {RECORD}\n")
    print(f"D-series behavioural battery family — {doc['family_id']}\n")
    print(f"  status               : {doc['status']}")
    print(f"  allocation rule id   : {doc['allocation_rule_id']}")
    print(f"  roles                : {doc['allocation_rule']['n_roles']}")
    print(f"  prompts per role     : "
          f"{doc['allocation_rule']['n_prompts_per_role']} "
          f"({doc['allocation_rule']['n_scorable_prompts_per_role']} scorable)")
    print("\n  roles, in build order:")
    for role in doc["allocation_rule"]["roles"]:
        print(f"    {role['role']:18} {role['rank_domain']}")
    print(f"\n  {'stratum':16} {'per role':>8} {'x6':>6} {'eligible':>9} "
          f"{'short':>7} {'roles':>6}")
    for name, row in doc["requirement"].items():
        flag = "  SHORT" if row["short_by"] else ""
        print(f"  {name:16} {row['per_battery']:>8} "
              f"{row['needed_for_six_roles']:>6} {row['eligible_today']:>9} "
              f"{row['short_by']:>7} {row['roles_fundable_today']:>6}{flag}")
    print(f"\n  roles fundable today : {doc['roles_fundable_today']} of "
          f"{doc['allocation_rule']['n_roles']} "
          f"(binding: {doc['binding_stratum']})")
    print(f"\n  BLOCKER: {doc['blocker']}")
    print("\n  MATERIALIZES NOTHING. AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
