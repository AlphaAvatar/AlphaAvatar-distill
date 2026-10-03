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

REPO_ROOT = Path(__file__).resolve().parents[3]
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
        "identity_kinds": ["stable id (including source_key)",
                           "normalized prompt content (sha256 of norm(text))"],
        "both_required": (
            "either kind alone is insufficient: ids miss one question entering "
            "the pool twice, content hashes miss a paraphrase under one id"),
        "chain": chain,
    }


def allocation_rule() -> dict[str, Any]:
    """The rule, frozen prospectively — everything except the realized items."""
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
        "exclusion": exclusion_chain(),
        "what_is_NOT_in_this_rule": (
            "the source pins and the realized item lists. The sources for the "
            "three short strata are an open maintainer decision, and the items "
            "do not exist. Binding them here would make the rule unfreezable "
            "until materialization, which would defeat the point of freezing "
            "it before any D1 outcome. They are bound by `family_content_id` at "
            "materialization instead."),
        "distribution_identity": {
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
        },
    }


def allocation_rule_id() -> str:
    """`H(the rule)`, computable today.

    This is what "frozen before any D1 outcome is observed" means operationally:
    the hash can be committed now, and a materialization that produced different
    roles, a different order or a different exclusion chain could not reproduce
    it.
    """
    from aadistill.infrastructure.manifest import sha256_json

    return sha256_json(allocation_rule())[:32]


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
        "blocker": (
            "THREE strata cannot fund six roles from their currently pinned "
            f"sources: {', '.join(f'{k} short by {v}' for k, v in sorted(short.items()))}. "
            "Extending a source changes the evaluation population, so it is a "
            "maintainer decision with a licence and contamination record "
            "attached, and it is not taken here."),
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
