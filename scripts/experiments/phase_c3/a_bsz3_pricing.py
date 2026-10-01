"""What the shortened A-bsz3 validation costs, and out of which book.

Three session shapes, priced from components this programme has MEASURED --
most of them from attempt75's own stage markers, on the device and the
geometry A-bsz3 would run on:

    step1_structural      the same-device structural/runtime comparison
    step2_fail_fast       step 1's prefix + ONE A-bsz3 recovery probe
    step2_three_seeds     the same session carrying all three

`step2_fail_fast` and `step2_three_seeds` are not two sessions. They are the
cheap and dear ENDS OF ONE session: the stop decision happens on the pod after
seed 1, and splitting it would pay a second setup and a second 21.78-minute
parent replay for nothing.

**The books do not transfer, and this study straddles them.** Step 1 trains no
probe and consumes no battery, so it is GPU engineering validation and the
`$10.00` engineering allowance is its book. Step 2 trains recovery probes and
scores them on the frozen confirmation battery, which the execution package's
own terms place OUTSIDE that allowance:

    engineering_allowance_does_not_authorize:
      - complete formal probes
      - consuming the frozen confirmation battery for an endpoint
      - formal execution -- that comes from the $55.0000 formal allowance

Calling the study "engineering" in its SCIENTIFIC claim does not move it
between books; what decides the book is what the session does. So this module
reports step 2 against the formal allowance, where it currently does not fit,
rather than quietly charging it to the allowance that has money left.

Nothing here authorizes anything.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3.formal_pricing import (  # noqa: E402
    DISK_USD_PER_GB_MONTH, C3PricingError, billed_rate, live_envelopes,
)

OUT = "logs/stages/stage-1/phase_c3/plans/a_bsz3_pricing.json"

#: The container disk, REUSED from the C3 provision rather than re-derived.
#: At $0.10/GB/month a 120 GB provision costs $0.0167/h, so the entire disk
#: term on a 1.4-hour step-1 session is about two cents and a tighter
#: derivation would move the price by less than one. Deriving a second
#: provision would create a second number to defend for no decision it could
#: change. A session that trains probes needs the C3 figure anyway.
CONTAINER_DISK_GB = 120

#: Every component in minutes, with the session that measured it. attempt75's
#: driver log is the owner of the stage timings; its markers are quoted in the
#: provenance string so a reader can re-derive them from the log rather than
#: from this table.
COMPONENTS: dict[str, dict[str, Any]] = {
    "setup": {
        "minutes": 7.70,
        "measured": ("attempt75: session start 17:58:39Z to STAGE_START:B at "
                     "18:06:21Z. THE SAME IMAGE HAS TAKEN 5, 8.5 AND 150+ "
                     "MINUTES, which is why the hard ceiling uses a worst "
                     "case instead of this figure."),
    },
    "pre_provider_gates": {
        "minutes": 3.0,
        "measured": "c3_pricing_9probe.json: attempt18 ran 14 gates in-window",
    },
    "teacher_fetch_verify": {
        "minutes": 0.22,
        "measured": "attempt75 stage B: 18:06:21Z to 18:06:34Z, 3 shards "
                    "bound at 768f209d9ea8",
    },
    "register_operator": {
        "minutes": 0.03,
        "measured": "attempt75 stage C: 18:06:34Z to 18:06:36Z",
    },
    "parent_replay": {
        "minutes": 21.78,
        "measured": ("attempt75 stage D: 18:06:36Z to 18:28:23Z, parent "
                     "eea90c91346a reproduced. THE DOMINANT TERM of a step-1 "
                     "session, which is why the overrun factor reaches it."),
    },
    "attention_round": {
        "minutes": 0.25,
        "measured": ("attempt75 stage E: 18:28:23Z to 18:28:38Z -- one "
                     "ATTENTION suffix end to end, including the checkpoint "
                     "write. The operator's own statistics pass is 11.2732 s "
                     "(C1 attempt18, same parent geometry)."),
    },
    "recovery_probe": {
        "minutes": 61.76,
        "measured": "attempt75 stage G: 555.85 min over 9 probes",
    },
    "evaluation_probe": {
        "minutes": 26.85,
        "measured": "attempt75 stage H: 241.63 min over 9 probes",
    },
    "collect_and_closeout": {
        "minutes": 5.0,
        "measured": ("attempt75 collected 235 files across 19 classes inside "
                     "its closeout window"),
    },
    "decide_and_closeout": {
        "minutes": 15.0,
        "measured": ("c3_pricing_9probe.json, for a session that also runs "
                     "the paired bootstrap on the pod"),
    },
}

#: The two hard-ceiling allowances, and the one place they differ from C3's.
SETUP_WORST_CASE_MINUTES = 45.0
OVERRUN_FACTOR = 1.3

#: C3 applied the overrun factor to training and evaluation ONLY, because on a
#: nine-probe session everything else was rounding error. On a step-1 session
#: the parent replay is more than half the work, so a ceiling that gave it no
#: allowance would not be a ceiling. The factor therefore reaches every
#: measured GPU component; `setup` is excluded because it has its own worst
#: case, and the pre-provider gates because they run before the meter starts.
_NO_OVERRUN = ("setup", "pre_provider_gates")

#: How many interleaved rounds the structural comparison runs, per protocol.
#: One warm-up plus three timed; see `compare_a_bsz3`'s module docstring for
#: why an 11-second workload is not measured once.
STRUCTURAL_ROUNDS = 4
N_PROTOCOLS = 2

SHAPES: dict[str, dict[str, Any]] = {
    "step1_structural": {
        "_what": ("both protocols from the same verified frozen parent, in "
                  "one session: digests, kept heads, per-head scores and rank "
                  "correlation, selection margins, observed position "
                  "counters, runtime over interleaved rounds, peak VRAM"),
        "probes": 0,
        "attention_rounds": STRUCTURAL_ROUNDS * N_PROTOCOLS,
        "closeout": "collect_and_closeout",
        "book": "gpu_engineering_allowance",
        "_book_because": ("trains no probe, consumes no battery, produces no "
                          "endpoint -- it is GPU engineering validation"),
    },
    "step2_fail_fast": {
        "_what": ("step 1's prefix, plus the A-bsz3 initialization and ONE "
                  "recovery probe at seed 217230555, trained and scored"),
        "probes": 1,
        "attention_rounds": 1,
        "closeout": "decide_and_closeout",
        "book": "formal_allowance",
        "_book_because": ("trains a recovery probe and scores it on "
                          "c1_confirmation_v1; the package's terms place both "
                          "outside the engineering allowance"),
    },
    "step2_three_seeds": {
        "_what": ("the same session carried to all three frozen C3 recovery "
                  "seeds, if seed 1 shows no material regression"),
        "probes": 3,
        "attention_rounds": 1,
        "closeout": "decide_and_closeout",
        "book": "formal_allowance",
        "_book_because": "as above",
    },
}


def _minutes(shape: str, *, worst: bool) -> dict[str, float]:
    """Per-component minutes for one shape. Pure, and itemised on purpose."""
    s = SHAPES[shape]
    per: dict[str, float] = {}
    counts = {
        "setup": 1.0, "pre_provider_gates": 1.0, "teacher_fetch_verify": 1.0,
        "register_operator": 1.0, "parent_replay": 1.0,
        "attention_round": float(s["attention_rounds"]),
        "recovery_probe": float(s["probes"]),
        "evaluation_probe": float(s["probes"]),
        s["closeout"]: 1.0,
    }
    for name, count in counts.items():
        if count == 0:
            continue
        minutes = COMPONENTS[name]["minutes"]
        if worst and name == "setup":
            minutes = SETUP_WORST_CASE_MINUTES
        elif worst and name not in _NO_OVERRUN:
            minutes *= OVERRUN_FACTOR
        per[name] = round(minutes * count, 4)
    return per


@dataclass(frozen=True)
class ShapePrice:
    shape: str
    gpu_rate_usd_per_hour: float
    billed_rate_usd_per_hour: float
    expected_minutes: float
    expected_usd: float
    hard_minutes: float
    hard_usd: float
    expected_components: dict[str, float]
    hard_components: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        s = SHAPES[self.shape]
        return {
            "_what": s["_what"], "probes": s["probes"],
            "book": s["book"], "_book_because": s["_book_because"],
            "expected": {"minutes": round(self.expected_minutes, 2),
                         "hours": round(self.expected_minutes / 60.0, 4),
                         "usd": self.expected_usd,
                         "components_minutes": self.expected_components},
            "hard_ceiling": {"minutes": round(self.hard_minutes, 2),
                             "hours": round(self.hard_minutes / 60.0, 4),
                             "usd": self.hard_usd,
                             "components_minutes": self.hard_components},
        }


def price(shape: str, gpu_rate_usd_per_hour: float) -> ShapePrice:
    """Expected and hard-ceiling cost for one shape at a live rate. Pure."""
    if shape not in SHAPES:
        raise C3PricingError(f"unknown A-bsz3 session shape {shape!r}")
    rate = float(gpu_rate_usd_per_hour)
    if not (0.0 < rate < 100.0):
        raise C3PricingError(f"implausible live rate ${rate}/h; refusing to price")
    billed = billed_rate(rate, disk_gb=CONTAINER_DISK_GB)
    exp, hard = _minutes(shape, worst=False), _minutes(shape, worst=True)
    exp_m, hard_m = sum(exp.values()), sum(hard.values())
    return ShapePrice(
        shape=shape, gpu_rate_usd_per_hour=rate,
        billed_rate_usd_per_hour=billed,
        expected_minutes=exp_m,
        expected_usd=round(exp_m / 60.0 * billed, 4),
        hard_minutes=hard_m,
        #: CEILED. A limit rounds down; a ceiling rounds up. A budget planner
        #: already refused once to build a plan that terminated $0.00003 above
        #: what `round()` had authorized.
        hard_usd=math.ceil(hard_m / 60.0 * billed * 10_000) / 10_000,
        expected_components=exp, hard_components=hard)


def fundability(shape: str, gpu_rate_usd_per_hour: float,
                envelopes: dict[str, float] | None = None) -> dict[str, Any]:
    """Whether this shape fits, against EVERY book that binds it.

    Four conditions, and the fourth is the one C3's pricer does not ask
    because C3 never spends the engineering allowance: a shape booked to the
    engineering allowance must fit THAT allowance, and an unspent formal
    balance would not help it.
    """
    env = dict(envelopes if envelopes is not None else live_envelopes())
    if "engineering_remaining_usd" not in env:
        env["engineering_remaining_usd"] = _engineering_remaining()
    p = price(shape, gpu_rate_usd_per_hour)
    book = SHAPES[shape]["book"]
    book_remaining = (env["engineering_remaining_usd"]
                      if book == "gpu_engineering_allowance"
                      else env["formal_remaining_usd"])
    conditions = {
        "derived_ceiling_within_session_envelope":
            p.hard_usd <= env["per_session_envelope_usd"],
        "cumulative_plus_derived_within_project_cap":
            round(env["cumulative_spend_usd"] + p.hard_usd, 4)
            <= env["project_cap_usd"],
        "booked_allowance_covers_derived": book_remaining >= p.hard_usd,
    }
    shortfalls: dict[str, float] = {}
    if not conditions["derived_ceiling_within_session_envelope"]:
        shortfalls["per_session_envelope"] = round(
            p.hard_usd - env["per_session_envelope_usd"], 4)
    if not conditions["cumulative_plus_derived_within_project_cap"]:
        shortfalls["project_cap"] = round(
            env["cumulative_spend_usd"] + p.hard_usd - env["project_cap_usd"], 4)
    if not conditions["booked_allowance_covers_derived"]:
        shortfalls[book] = round(p.hard_usd - book_remaining, 4)
    return {
        "price": p.as_dict(),
        "book": book,
        "book_remaining_usd": round(book_remaining, 4),
        "envelopes": {k: round(v, 4) for k, v in env.items()},
        "conditions": conditions,
        "FUNDABLE": all(conditions.values()),
        "shortfalls_usd": shortfalls,
        "_fitting_is_not_permission": (
            "a shape that fits every book is still unfunded until a "
            "maintainer grants it"),
    }


def _engineering_remaining() -> float:
    """The engineering allowance's remaining balance, from its owner."""
    import subprocess

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/consolidate/derive_budget.py"),
         "--json"], capture_output=True, text=True, cwd=str(REPO))
    if out.returncode != 0:
        raise C3PricingError(f"derive_budget failed: {out.stderr[-400:]}")
    return float(json.loads(out.stdout)["engineering"]["remaining_usd"])


def assess_all(gpu_rate_usd_per_hour: float,
               envelopes: dict[str, float] | None = None) -> dict[str, Any]:
    """Every shape, priced and judged, plus the two-step totals."""
    env = dict(envelopes if envelopes is not None else live_envelopes())
    if "engineering_remaining_usd" not in env:
        env["engineering_remaining_usd"] = _engineering_remaining()
    shapes = {name: fundability(name, gpu_rate_usd_per_hour, env)
              for name in SHAPES}
    s1 = shapes["step1_structural"]["price"]["hard_ceiling"]["usd"]
    s2 = shapes["step2_three_seeds"]["price"]["hard_ceiling"]["usd"]
    return {
        "shapes": shapes,
        "worst_case_both_steps": {
            "usd": round(s1 + s2, 4),
            "_is_not_one_authorization": (
                "two sessions, each authorized separately, and step 2 only "
                "exists if step 1's digests differ. The sum is what the "
                "programme could spend, not what anything is authorized for."),
            "project_headroom_after_usd": round(
                env["project_cap_usd"] - env["cumulative_spend_usd"] - s1 - s2,
                4),
        },
        "_rate_must_be_requoted": (
            "securePrice, live, immediately before any authorization. The "
            "launcher prices on securePrice and a communityPrice figure is "
            "one it cannot act on."),
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate", type=float, default=1.09,
                    help="GPU securePrice in USD/h (default: the historical "
                         "L40S observation; re-quote live before authorizing)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    doc = {
        "schema": "aadistill.phase_c3.a_bsz3_pricing/v1",
        "_contract": (
            "The shortened A-bsz3 validation, priced from measured "
            "components. Supersedes the `~$30` 16-probe estimate the "
            "withdrawn non-inferiority design carried. AUTHORIZES NOTHING."),
        "gpu": "NVIDIA L40S",
        "gpu_rate_usd_per_hour": args.rate,
        "billed_rate_usd_per_hour": round(
            billed_rate(args.rate, disk_gb=CONTAINER_DISK_GB), 6),
        "container_disk_gb": CONTAINER_DISK_GB,
        "_disk_is_billed_separately": (
            f"{CONTAINER_DISK_GB} GB x ${DISK_USD_PER_GB_MONTH}/GB/month"),
        "components": COMPONENTS,
        "allowances": {
            "setup_worst_case_minutes": SETUP_WORST_CASE_MINUTES,
            "overrun_factor": OVERRUN_FACTOR,
            "_overrun_reaches": (
                "every measured GPU component except setup, which has its own "
                "worst case, and the pre-provider gates, which run before the "
                "meter. C3 applied it to training and evaluation only; on a "
                "step-1 session the parent replay is the dominant term and a "
                "ceiling that excluded it would not bound the work."),
        },
        **assess_all(args.rate),
    }

    for name, row in doc["shapes"].items():
        p = row["price"]
        print(f"{name:20s} expected {p['expected']['minutes']:7.2f} min "
              f"${p['expected']['usd']:7.4f}   hard "
              f"{p['hard_ceiling']['minutes']:7.2f} min "
              f"${p['hard_ceiling']['usd']:7.4f}   book {row['book']:26s} "
              f"remaining ${row['book_remaining_usd']:8.4f}   "
              f"{'FUNDABLE' if row['FUNDABLE'] else 'NOT FUNDABLE ' + str(row['shortfalls_usd'])}")
    print(f"\nworst case across both steps: "
          f"${doc['worst_case_both_steps']['usd']:.4f}")

    if args.write:
        (REPO / args.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
