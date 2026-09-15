#!/usr/bin/env python3
"""Price the Phase-C2 Search-1 session. PRICING ONLY — this authorizes nothing.

    PYTHONPATH=src:scripts .venv/bin/python scripts/autoinit/price_c2.py

Five envelopes, separately named, because each answers a different question and
collapsing any two of them loses the answer to one:

* **the expected path** — the session phases plus the search on the trajectory
  the Phase-B ranking was actually observed to take (DEPTH-early);
* **contingency**, 10%, on the expected path only;
* **`beam_composition_risk`** — the difference between DEPTH-early and the
  structural worst case. Not a contingency and not a surprise: it is the exact
  arithmetic of a beam that defers DEPTH, which the ranking policy is permitted
  to return and which no measurement bounds away;
* **`baseline_rebuild_reserve`** — CONDITIONAL. It is spent only if the beam
  does not re-derive the frozen C1 baseline B, and it is named rather than
  folded into the artifact-recovery reserve because those protect different
  things: one buys a baseline the experiment needs, the other buys the chance to
  bring evidence home after something has already gone wrong. Paying for a
  rebuild out of the teardown reserve is how a session ends with a result it
  cannot collect;
* **`artifact_recovery_reserve`** — teardown and collection, untouched.

Every minute figure is DERIVED at generation time: the search envelope from
`experiments.phase_c2.search_space`, which is itself back-tested against
Phase-B attempt 5's real beams, and the baseline rebuild from the stage
timestamps C1 attempt 18 recorded. Nothing here is transcribed, so nothing here
can drift from the model that produced it.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from experiments.phase_c2.search_space import (  # noqa: E402
    ATTENTION_ACTIVATION_PROXY_FACTOR,
    ATTENTION_ACTIVATION_PROXY_IMPL,
    PRICE_PER_HOUR_LAST_QUOTED,
    SESSION_PHASE_MINUTES,
    bound,
    c2_search1_space,
    register_c2_operators,
    trajectory,
)

OUT = REPO / "logs/stages/stage-1/phase_c2/plans/phase_c2_pricing.json"

#: C1 attempt 18's evidence, and the arm identities beside it. The rebuild's
#: cost is read from what the same work actually took, on the same card, under
#: the same operators.
C1_EVIDENCE = ("logs/stages/stage-1/phase_c1/runs/attempt18/evidence/"
               "c1_evidence.json")
C1_ARM_IDENTITIES = ("logs/stages/stage-1/phase_c1/runs/attempt18/evidence/"
                     "c1_arm_identities.json")

#: The per-expansion overhead a candidate pays after its operator: canonical
#: reload, identify, validate and `state_eval`. MAXIMUM observed in Phase-B
#: attempt 5, because a reserve that uses a mean is not a reserve.
STATE_EVAL_OVERHEAD_MAX_MIN = 3.24

CONTINGENCY_FRACTION = 0.10
ARTIFACT_RECOVERY_RESERVE_MIN = 30.0


def minutes_to_usd(minutes: float, rate: float) -> float:
    return minutes / 60.0 * rate


def baseline_rebuild_minutes() -> dict[str, object]:
    """What rebuilding B costs, from C1's own stage timestamps.

    Stage D replayed the shared `DEPTH→FFN→RESIDUAL_WIDTH` parent from the
    teacher; stage F executed the treatment's ATTENTION step and materialized
    it. Together they ARE the rebuild — the same `materialize_fixed_path` call
    over the same frozen spec — so the cost is measured rather than modelled.
    Stage E is excluded on purpose: it is the incumbent's ATTENTION replay,
    which the C2 baseline does not need.
    """
    ev = json.loads((REPO / C1_EVIDENCE).read_text())
    finished = {k: datetime.fromisoformat(v["finished_utc"])
                for k, v in ev["stages"].items() if v.get("finished_utc")}
    parent_min = (finished["D"] - finished["C"]).total_seconds() / 60.0
    treatment_min = (finished["F"] - finished["E"]).total_seconds() / 60.0

    arms = json.loads((REPO / C1_ARM_IDENTITIES).read_text())
    return {
        "parent_replay_minutes": round(parent_min, 3),
        "treatment_step_minutes": round(treatment_min, 3),
        "state_eval_minutes": STATE_EVAL_OVERHEAD_MAX_MIN,
        "minutes": round(parent_min + treatment_min
                         + STATE_EVAL_OVERHEAD_MAX_MIN, 3),
        "basis": "measured",
        "source": (
            f"C1 attempt 18, stage C->D ({parent_min:.3f} min, the shared "
            f"DEPTH->FFN->RESIDUAL_WIDTH parent from the teacher) plus stage "
            f"E->F ({treatment_min:.3f} min, the treatment ATTENTION step and "
            f"its materialization), plus the maximum per-candidate "
            f"canonical-reload/identify/validate/state_eval overhead observed "
            f"in Phase-B attempt 5 ({STATE_EVAL_OVERHEAD_MAX_MIN} min). Stage E "
            "is excluded: it replays the INCUMBENT's ATTENTION, which the C2 "
            "baseline does not need"),
        "conditional_on": (
            "spent ONLY if the Search-1 beam does not re-derive B. B's order "
            "and ATTENTION mixture are both inside the searched space and the "
            "seed is the frozen one, so the expected case is that the search "
            "produces B and this reserve is not touched"),
        "attention_step_now_measured": {
            "seconds": arms["treatment"]["seconds"],
            "what_it_tells_us": (
                "attention.activation_importance_v1 took "
                f"{arms['treatment']['seconds']:.2f} s on the narrow "
                "post-WIDTH parent. This is the operator's FIRST measured "
                "timing and it confirms the search cost model's "
                f"{ATTENTION_ACTIVATION_PROXY_FACTOR}x "
                f"{ATTENTION_ACTIVATION_PROXY_IMPL} proxy is conservative for "
                "deeper parents by roughly 3x. The proxy is NOT sharpened here: "
                "the search may apply this operator at the ROOT, on the full "
                "teacher, which remains unmeasured, and re-pricing an accepted "
                "envelope downward is a separate decision from adding a reserve "
                "the envelope was missing"),
        },
    }


def main() -> int:
    register_c2_operators()
    space = c2_search1_space()
    rate = PRICE_PER_HOUR_LAST_QUOTED

    #: The two search envelopes, from the model that back-tests against
    #: Phase-B attempt 5's real beams (10/46/19/7 exactly, minutes to 0.2%).
    early = trajectory(space, prefer_depth=True, statistic="max")
    limit = bound(space, statistic="max")
    rebuild = baseline_rebuild_minutes()

    phases = dict(SESSION_PHASE_MINUTES)
    items = [
        {"item": "session setup and asset staging",
         "minutes": phases["setup_and_asset_staging"], "basis": "planned",
         "source": ("the figure this repository plans with "
                    "(autoinit_preflight_launch.py setup_minutes=45.0), NOT C1 "
                    "attempt 18's warm-image 6 minutes. The same script on the "
                    "same image and card has taken 5, 8.5 and over 150")},
        {"item": "bundle transfer", "minutes": phases["bundle_transfer"],
         "basis": "planned", "source": "preflight plan transfer_minutes=6.0"},
        {"item": "teacher fetch and verify",
         "minutes": phases["teacher_fetch_and_verify"], "basis": "measured",
         "source": "C1 attempt 18 stage B, ~6 min, rounded up"},
        {"item": "machine gates", "minutes": phases["machine_gates"],
         "basis": "planned", "source": "preflight plan stage1_machine_gates=22.0"},
        {"item": "beam search, DEPTH-early trajectory",
         "minutes": early["minutes"], "basis": "derived",
         "source": (
             f"{early['expansions']} expansions priced at the MAXIMUM "
             "per-expansion minutes measured in Phase-B attempt 5's telemetry, "
             "on the trajectory that ranking was observed to take: at Phase-B "
             "level 1 all six retained states already contained DEPTH and the "
             "epsilon-Pareto front put FFN->DEPTH in front 0. The same model "
             "replays that run's real beams and predicts every level's "
             "expansion count exactly and its minutes to 0.2%")},
        {"item": "selection commit and artifact manifest",
         "minutes": phases["selection_commit_and_artifact_manifest"],
         "basis": "planned", "source": "preflight plan artifact_manifest=8.0"},
        {"item": "artifact synchronization",
         "minutes": phases["artifact_synchronization"], "basis": "planned",
         "source": "preflight plan artifact_synchronization=6.0"},
    ]

    reserves = [
        {"reserve": "beam_composition_risk",
         "minutes": round(limit.max_minutes - early["minutes"], 3),
         "conditional": False, "basis": "derived",
         "source": (
             f"the exact structural worst case ({limit.max_minutes:.1f} min) "
             f"minus the DEPTH-early trajectory ({early['minutes']:.1f} min). "
             "`bound()` enumerates every beam composition the ranking policy "
             "could return, so the maximum is attained by some ranking and no "
             "ranking exceeds it. NOT a contingency: DEPTH costs 26-34 min "
             "wherever it runs, and what this prices is how many beam members "
             "still owe it. `children_max x mean` is what authorized Phase-B "
             "attempt 3 at 1.91-7.51 h for a run that took 9.08 and did not "
             "finish")},
        {"reserve": "baseline_rebuild_reserve",
         "minutes": rebuild["minutes"], "conditional": True,
         "basis": rebuild["basis"], "source": rebuild["source"],
         "detail": rebuild},
    ]

    expected_min = sum(i["minutes"] for i in items)
    reserve_min = sum(r["minutes"] for r in reserves)
    soft_stop_min = expected_min * (1.0 + CONTINGENCY_FRACTION) + reserve_min
    hard_min = soft_stop_min + ARTIFACT_RECOVERY_RESERVE_MIN

    doc = {
        "schema": "aadistill.autoinit.c2_pricing/v1",
        "generated_utc": "2026-09-15T00:00:00Z",
        "_contract": (
            "A structural cost bound for the Phase-C2 Search-1 session. PRICING "
            "ONLY: not an authorization, not a grant, and not permission to "
            "launch. Nothing here approves spending, and neither remaining "
            "project headroom nor C1's unused formal allowance is permission."),
        "hardware": {
            "name": "L40S", "price_per_hour_usd": rate,
            "field": "securePrice",
            "basis": ("quoted live 2026-09-15, stock Medium. `securePrice` is "
                      "the field session_runner prices on and refuses above "
                      "--max-price; communityPrice was 0.79 at the same moment "
                      "and is a different product this launcher never "
                      "provisions"),
            "a_launch_must_requote": (
                "this quote is planning evidence with a date on it. "
                "check_gpu_offered re-queries before any pod exists, and an "
                "hour-old price is not a price"),
        },
        "session_shape": {
            "kind": "search only — no training, no probes, no battery",
            "operator_kinds": 4, "implementations": 4,
            "order": "free; DEPTH is not pinned",
            "beam_width": limit.beam_width,
            "warmup_levels": limit.warmup_levels,
            "expansions_min": limit.min_expansions,
            "expansions_max": limit.max_expansions,
            "search_minutes_min": round(limit.min_minutes, 2),
            "search_minutes_max": round(limit.max_minutes, 2),
        },
        "line_items": [dict(i, usd=round(minutes_to_usd(i["minutes"], rate), 4))
                       for i in items],
        "reserves": [dict(r, usd=round(minutes_to_usd(r["minutes"], rate), 4))
                     for r in reserves],
        "totals": {
            "expected_minutes": round(expected_min, 3),
            "expected_usd": round(minutes_to_usd(expected_min, rate), 4),
            "contingency_fraction": CONTINGENCY_FRACTION,
            "soft_stop_minutes": round(soft_stop_min, 3),
            "soft_stop_usd": round(minutes_to_usd(soft_stop_min, rate), 4),
            "artifact_recovery_reserve_minutes": ARTIFACT_RECOVERY_RESERVE_MIN,
            "hard_ceiling_minutes": round(hard_min, 3),
            # ROUNDED UP. A ceiling rounds up or it under-authorizes the plan it
            # prices: C1's exact 13.757733 truncated to 13.7577 is $0.000033
            # below its own plan, and a grant at that value fails closed.
            "hard_ceiling_usd": math.ceil(
                minutes_to_usd(hard_min, rate) * 10000) / 10000,
            "hard_ceiling_usd_exact": minutes_to_usd(hard_min, rate),
            "hard_ceiling_rounding": ("ceil to 4 dp; a ceiling rounds UP or it "
                                      "under-authorizes the plan it prices"),
            "_five_envelopes_stay_separate": (
                "expected path, contingency on it, beam_composition_risk, "
                "baseline_rebuild_reserve and artifact_recovery_reserve each "
                "answer a different question. One number for two of them is one "
                "number nobody can interpret"),
        },
        "the_unmeasured_input": {
            "impl_id": "attention.activation_importance_v1",
            "priced_at": (f"{ATTENTION_ACTIVATION_PROXY_FACTOR}x the maximum "
                          f"measured {ATTENTION_ACTIVATION_PROXY_IMPL} "
                          "expansion"),
            "why": (
                "it has never run inside a BeamSearch. C1 drove it through "
                "fixed_path only. Per token its attention second moment "
                "accumulates 36 x 32 x 128^2 = 18.9M MAC against the residual "
                "covariance's 36 x 2560^2 = 236M, so the factor is generous in "
                "the direction the arithmetic can afford"),
            "now_partly_measured": rebuild["attention_step_now_measured"],
        },
        "what_is_not_priced_here": [
            "any authorization, which is a separate maintainer decision",
            "a second attempt after a failure",
            "Search-2, which is conditional on Search-1's evidence",
            "behavioural confirmation of any finalist, which is a separate "
            "paid experiment in roughly C1's cost class",
            "durable large-checkpoint storage, which Search-1 does not need "
            "because it trains no probes",
        ],
        "authorizes": "nothing",
    }
    doc["pricing_sha256"] = sha256_json(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"wrote {OUT.relative_to(REPO)}\n")
    print(f"{'line item':50s} {'basis':>9s} {'min':>8s} {'USD':>8s}")
    for i in doc["line_items"]:
        print(f"{i['item'][:50]:50s} {i['basis']:>9s} {i['minutes']:8.2f} "
              f"{i['usd']:8.4f}")
    print(f"\n{'reserve':50s} {'cond':>9s} {'min':>8s} {'USD':>8s}")
    for r in doc["reserves"]:
        print(f"{r['reserve'][:50]:50s} "
              f"{'YES' if r['conditional'] else 'no':>9s} "
              f"{r['minutes']:8.2f} {r['usd']:8.4f}")
    t = doc["totals"]
    print(f"\n  expected      {t['expected_minutes']:8.1f} min   "
          f"${t['expected_usd']:.4f}")
    print(f"  soft stop     {t['soft_stop_minutes']:8.1f} min   "
          f"${t['soft_stop_usd']:.4f}   (+10% contingency + both reserves)")
    print(f"  HARD CEILING  {t['hard_ceiling_minutes']:8.1f} min   "
          f"${t['hard_ceiling_usd']:.4f}   "
          f"(+{ARTIFACT_RECOVERY_RESERVE_MIN:.0f} min artifact recovery)")
    print(f"\n  pricing_sha256 {doc['pricing_sha256']}")
    print("  AUTHORIZES NOTHING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
