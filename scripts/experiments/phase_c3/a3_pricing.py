"""What the COMPLETE A3 chain costs, and whether every limit funds it.

ONE experiment, one session, one price:

    setup -> parent replay -> both initialization protocols + diagnostics
          -> 3 recovery probes -> 3 evaluations -> aggregation
          -> evidence collection -> teardown

**This replaces the step-1 / step-2 split.** The maintainer merged them on
2026-10-01: A3 answers the practical question once, without stopping for
intermediate approvals, so there is one ceiling to authorize rather than two.
The three-shape pricer that preceded this is deleted rather than kept beside
it — a redundant mechanism left in place is the complexity ratchet AGENTS.md
P8.2.1 forbids.

**Storage is DERIVED from a MEASUREMENT, not from an estimate.** A3 holds
one four-step path's intermediates, two initialization leaves and ONE training
working set at a time, because a probe is released after its bytes are durable
off-pod -- so the SCIENCE peak is modest. What dominates is the floor, and the
floor was estimated at 18.61 GiB until a pod reported 58 GB of 60 GB used
before stage D could write its first shard. `a3_attempt31` died there with
`No space left on device`, the floor term is now the measured 50.3 GiB, and
the derivation says 67.75 GiB at the parent replay. The whole disk term is
cents either way, which is exactly why it should be derived rather than
guessed -- in either direction.

**Every applicable limit is checked**, including the package total, through
`formal_pricing.evaluate_limits`. That function is the single owner of which
limits bind; this module owns only what A3 costs.

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
sys.path.insert(0, str(REPO / "src"))

from aadistill.runtime import cost as COST  # noqa: E402
from experiments.phase_c3.formal_pricing import (  # noqa: E402
    DISK_USD_PER_GB_MONTH, HOURS_PER_MONTH, C3PricingError, Fundability,
    evaluate_limits, live_envelopes, price_c3,
)

OUT = "logs/stages/stage-1/phase_c3/plans/a3_pricing.json"

#: A3's book. It trains recovery probes and scores them on the frozen
#: confirmation battery, and the execution package states that the GPU
#: engineering allowance authorizes neither. What decides the book is what the
#: session DOES, not what its scientific claim is called.
BOOK = "formal_allowance"

#: --- geometry and recipe, read or measured, never assumed ----------------

#: The A arm and the shared parent, from attempt18's recorded arm identities.
PARAMS_A = 596_049_920
PARAMS_PARENT = 713_490_432

#: The frozen recovery recipe A3 reuses unchanged from the attempt75 controls.
FROZEN_RECIPE_REL = "configs/stage3/e1/e1_r0860k_sa_pca.json"

#: Measured in `scripts/experiments/phase_c2/behavioural.py`, which owns both:
#: the teacher's on-disk size and the worst single four-step path's
#: intermediates as observed during the C2 replay.
TEACHER_GIB = 7.51
PATH_INTERMEDIATES_GIB = 16.12

#: Tokenizer, config, generation config and per-sample evidence beside the
#: weights. Small and not zero.
CHECKPOINT_EXTRA_BYTES = 800_000

#: The floor the residency model does not otherwise see: everything an A3 pod
#: holds before A3 writes a byte of science -- the container image and its
#: python, BOTH session venvs, the wheelhouses that built them, the HF teacher
#: cache, the checked-out repository and the staged assets.
FLOOR_SUPPLEMENT_GIB = {
    #: MEASURED ON A POD, not estimated. The previous terms were
    #: `container_image_and_python: 8.0` and `venv_and_wheel_cache: 2.0`,
    #: described as "the generous side of what a runpod/pytorch image plus
    #: this repository's install occupies". They were not generous: an A3 pod
    #: reports 58 GB of 60 GB used BEFORE stage D can write its shards, and
    #: a3_attempt31 died there with `No space left on device` -- the third
    #: time this repository has run a pod out of disk, and the second at
    #: `Writing model shards` specifically.
    #:
    #: The anchor is `df` on the overlay, which cannot double-count: 58 GB
    #: used, of which roughly 4 GB was the replay's own partial output. The
    #: floor is therefore ~54 GB = 50.3 GiB, against the 18.61 GiB the
    #: estimate produced. Attribution from `du`, recorded in
    #: logs/stages/stage-1/phase_a3/runs/a3_attempt31/evidence/
    #: container_disk_measurement.json: image 14 GB, both session venvs 16 GB,
    #: FOUR wheelhouses 15.2 GB, HF teacher cache 22 GB, repository and
    #: staged assets the remainder. Those terms sum above the anchor because
    #: `du` double-counts hardlinked and bind-mounted paths, which is exactly
    #: why the anchor is `df` and the itemisation is commentary.
    "measured_pod_floor": 50.3,
    "_basis": ("df on overlay: 60 GB provisioned, 58 GB used at the moment "
               "stage D failed, minus ~4 GB of partial replay output. One "
               "MEASURED term rather than four guessed ones, because four "
               "terms that sum to a number the filesystem contradicts are "
               "worse than one term with a provenance."),
    "_the_teacher_is_inside_it": (
        "TEACHER_GIB is NOT added on top of this floor any more. The 22 GB "
        "HF cache the measurement attributes to /root is the teacher, and "
        "adding a separate 7.51 GiB term would charge the same bytes twice -- "
        "the mistake this module's own docstring warns about in the other "
        "direction, where charging container and durable storage for the "
        "same bytes once produced a 140 GB request."),
    "_what_would_reduce_it": (
        "the four wheelhouses are 15.2 GB of install-time archives, a quarter "
        "of the disk, needed only while the two venvs are built. Deleting "
        "them after ASSETS_READY would free more than the whole science peak. "
        "That is a change to the SHARED setup script every session runs, so "
        "it is recorded here as the cheaper fix and not taken under an "
        "executing experiment: container disk is $0.10/GB/month, and 60 extra "
        "GB over a 7.5-hour chain is about six cents."),
}

#: Margin on the derived total. Not "to be safe": this repository has run a
#: pod out of disk THREE times -- attempt5 lost probe 11 of 12 and the
#: campaign's verdict, a C2 replay died at `Writing model shards`, and
#: a3_attempt31 died at the same call with a floor that had been estimated at
#: a third of its measured size. The marginal cost of the margin is under a
#: cent an hour; the marginal cost of being wrong is the whole chain.
PROVISION_MARGIN = 1.5

#: --- the measured component table ----------------------------------------

COMPONENTS: dict[str, dict[str, Any]] = {
    "setup": {
        "minutes": 7.70,
        "measured": ("attempt75: session start 17:58:39Z to STAGE_START:B at "
                     "18:06:21Z. THE SAME IMAGE HAS TAKEN 5, 8.5 AND 150+ "
                     "MINUTES, which is why the hard ceiling substitutes a "
                     "worst case rather than scaling this."),
    },
    "pre_provider_gates": {
        "minutes": 3.0,
        "measured": "c3_pricing_9probe.json: attempt18 ran 14 gates in-window",
    },
    "teacher_fetch_verify": {
        "minutes": 0.22,
        "measured": "attempt75 stage B: 18:06:21Z to 18:06:34Z",
    },
    "register_operator": {
        "minutes": 0.03,
        "measured": "attempt75 stage C: 18:06:34Z to 18:06:36Z",
    },
    "parent_replay": {
        "minutes": 21.78,
        "measured": ("attempt75 stage D: 18:06:36Z to 18:28:23Z, parent "
                     "eea90c91346a reproduced under its frozen digest gate"),
    },
    "initialization_rounds": {
        "minutes": 2.00,
        "measured": ("8 suffix materializations at 0.25 min each -- 4 "
                     "interleaved rounds per protocol, attempt75 stage E "
                     "18:28:23Z to 18:28:38Z for one. The operator's own "
                     "statistics pass is 11.2732 s (C1 attempt18, same parent "
                     "geometry). Round 0 of A_bsz3 IS the initialization the "
                     "probes train from, so no separate materialization is "
                     "priced."),
    },
    "recovery_probes": {
        "minutes": 185.28,
        "measured": "attempt75 stage G: 555.85 min over 9 probes -> 61.76 x 3",
    },
    "evaluations": {
        "minutes": 80.55,
        "measured": "attempt75 stage H: 241.63 min over 9 probes -> 26.85 x 3",
    },
    "aggregation": {
        "minutes": 15.0,
        "measured": ("c3_pricing_9probe.json `decide_bootstrap_closeout`, for "
                     "a session that runs the paired bootstrap on the pod. A3 "
                     "computes one contrast rather than three and more "
                     "breakdowns, so the figure is carried rather than scaled "
                     "down."),
    },
    "collect_and_teardown": {
        "minutes": 5.0,
        "measured": ("attempt75 collected 235 files across 19 classes inside "
                     "its closeout window"),
    },
}

SETUP_WORST_CASE_MINUTES = 45.0
OVERRUN_FACTOR = 1.3

#: The overrun factor reaches every measured GPU component. `setup` is
#: excluded because it has its own worst case, and the pre-provider gates
#: because they run before the meter starts. C3 scaled only training and
#: evaluation, which was right when everything else was rounding error at nine
#: probes; on a chain whose parent replay is 22 of 320 expected minutes it is
#: not.
_NO_OVERRUN = ("setup", "pre_provider_gates")


class A3PricingError(C3PricingError):
    """A3 cannot be priced as declared."""


# --- storage --------------------------------------------------------------


def training_dtypes(repo_root: str | Path = REPO) -> dict[str, Any]:
    """What a probe trains and SAVES in, from the frozen recipe.

    READ, never assumed. A storage model that charged a trained probe at the
    bf16 size of the initialization leaf it started from -- while the recipe
    declares `float32` -- is exactly how a pod with "room to spare" ran out of
    disk, wrong by precisely the ratio of the two dtypes.
    """
    recipe_path = Path(repo_root) / FROZEN_RECIPE_REL
    if not recipe_path.is_file():
        raise A3PricingError(
            f"the frozen recovery recipe is not at {FROZEN_RECIPE_REL}; the "
            "storage bound reads its dtypes and will not guess them")
    recipe = json.loads(recipe_path.read_text())
    declared = str(recipe.get("dtype") or "").lower()
    if declared not in COST.BYTES_PER_PARAM:
        raise A3PricingError(
            f"{FROZEN_RECIPE_REL} declares dtype={recipe.get('dtype')!r}, "
            f"which is not a per-parameter byte count this repository knows "
            f"({sorted(COST.BYTES_PER_PARAM)})")
    betas = (recipe.get("optim") or {}).get("betas")
    if not isinstance(betas, list) or not betas:
        raise A3PricingError(
            f"{FROZEN_RECIPE_REL} states no optim.betas, so the optimizer's "
            "moment count cannot be derived")
    return {"save_dtype": declared, "weight_dtype": declared,
            "grad_dtype": declared, "moment_dtype": declared,
            "n_moments": len(betas), "_source": FROZEN_RECIPE_REL}


def storage_requirement(n_probes: int = 3,
                        repo_root: str | Path = REPO) -> dict[str, Any]:
    """Peak LOCAL residency and the DURABLE requirement. Two resources.

    Container storage dies with the pod; the durable requirement must survive
    teardown and is bounded against the backend that holds it. Charging the
    same bytes to both produced a 140 GB provision request once.
    """
    tr = training_dtypes(repo_root)
    leaf = COST.CheckpointFootprint(PARAMS_A, "bfloat16",
                                    extra_bytes=CHECKPOINT_EXTRA_BYTES)
    parent = COST.CheckpointFootprint(PARAMS_PARENT, "bfloat16",
                                      extra_bytes=CHECKPOINT_EXTRA_BYTES)
    probe = COST.CheckpointFootprint(PARAMS_A, tr["save_dtype"],
                                     extra_bytes=CHECKPOINT_EXTRA_BYTES)
    working = COST.training_working_set_bytes(
        PARAMS_A, weight_dtype=tr["weight_dtype"],
        grad_dtype=tr["grad_dtype"], moment_dtype=tr["moment_dtype"],
        n_moments=tr["n_moments"])

    #: The teacher is INSIDE the measured floor; see
    #: `FLOOR_SUPPLEMENT_GIB._the_teacher_is_inside_it`. Adding `TEACHER_GIB`
    #: here would charge the HF cache twice.
    floor_gib = sum(v for k, v in FLOOR_SUPPLEMENT_GIB.items()
                    if not k.startswith("_"))

    U = COST.ResidencyUnit
    units = [
        U(label="parent_replay", durable_bytes=0, retained_bytes=parent.bytes,
          transient_bytes=int(PATH_INTERMEDIATES_GIB * 2 ** 30),
          released_on_completion=False),
        #: Round 0 of each protocol is retained as the canonical artifact;
        #: later rounds are deleted as soon as their record is read, so they
        #: cost one leaf of transient rather than one each.
        U(label="A_bsz1_diagnostic_rounds", durable_bytes=0,
          retained_bytes=leaf.bytes, transient_bytes=leaf.bytes,
          released_on_completion=False),
        U(label="A_bsz3_initialization", durable_bytes=0,
          retained_bytes=leaf.bytes, transient_bytes=leaf.bytes,
          released_on_completion=False),
    ]
    for i in range(n_probes):
        #: `released_on_completion=True` is a statement about the CODE: the
        #: driver releases a probe only after its bytes arrived off-pod AND
        #: re-identified there, and refuses to continue if a release fails.
        units.append(U(label=f"probe_{i}", durable_bytes=probe.bytes,
                       retained_bytes=probe.bytes, transient_bytes=working,
                       materializing_bytes=probe.bytes,
                       released_on_completion=True))

    local = COST.peak_local_residency_bytes(
        units, fixed_bytes=int(floor_gib * 2 ** 30))
    durable = COST.durable_backend_bytes(units)
    #: GB -> GiB through the repository's recorded basis. Treating the
    #: provider's GB flag as GiB overstates capacity by 7%, in the direction
    #: that hurts.
    gb_per_gib = 1.073741824
    needed_gb = local["peak_gib"] * PROVISION_MARGIN * gb_per_gib
    provision_gb = int(math.ceil(needed_gb / 10.0) * 10)
    return {
        "footprints_gib": {
            "initialization_leaf_bf16": round(leaf.gib, 3),
            "shared_parent_bf16": round(parent.gib, 3),
            "trained_probe": round(probe.gib, 3),
            "one_training_working_set": round(working / 2 ** 30, 3),
        },
        "training_dtypes": tr,
        "fixed_floor_gib": round(floor_gib, 3),
        "fixed_floor_components_gib": FLOOR_SUPPLEMENT_GIB,
        "container_residency": {
            "peak_gib": local["peak_gib"],
            "peak_at_unit": local["peak_at_unit"],
            "n_units": local["n_units"],
        },
        "durable_requirement": {
            "gib": durable["gib"], "n_probes": n_probes,
            "_bounded_against": ("the durable backend, never container "
                                 "storage"),
        },
        "provision": {
            "margin": PROVISION_MARGIN,
            "container_disk_gb": provision_gb,
            "available_gib": round(provision_gb / gb_per_gib, 3),
            "headroom_gib": round(provision_gb / gb_per_gib
                                  - local["peak_gib"], 3),
            "_why_not_c3s_120_gb": (
                "C3 provisioned for nine probes. A3 holds one training "
                "working set at a time and three probes' worth of retained "
                "bytes at most, and the difference between this provision and "
                "120 GB is a few cents over the whole chain -- which is the "
                "reason to derive it rather than to round it up."),
        },
    }


# --- price ----------------------------------------------------------------


def component_minutes(worst: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, entry in COMPONENTS.items():
        minutes = float(entry["minutes"])
        if worst and name == "setup":
            minutes = SETUP_WORST_CASE_MINUTES
        elif worst and name not in _NO_OVERRUN:
            minutes *= OVERRUN_FACTOR
        out[name] = round(minutes, 4)
    return out


def billed_rate(gpu_rate_usd_per_hour: float, disk_gb: int) -> float:
    """GPU price plus container disk, per hour. A GPU rate is not the bill."""
    return gpu_rate_usd_per_hour + disk_gb * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH


@dataclass(frozen=True)
class A3Price:
    gpu_rate_usd_per_hour: float
    billed_rate_usd_per_hour: float
    container_disk_gb: int
    expected_minutes: float
    expected_usd: float
    hard_minutes: float
    hard_usd: float
    expected_components: dict[str, float]
    hard_components: dict[str, float]
    storage: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_rate_usd_per_hour": self.gpu_rate_usd_per_hour,
            "billed_rate_usd_per_hour": round(self.billed_rate_usd_per_hour, 6),
            "container_disk_gb": self.container_disk_gb,
            "expected": {"minutes": round(self.expected_minutes, 2),
                         "hours": round(self.expected_minutes / 60.0, 4),
                         "usd": self.expected_usd,
                         "components_minutes": self.expected_components},
            "hard_ceiling": {"minutes": round(self.hard_minutes, 2),
                             "hours": round(self.hard_minutes / 60.0, 4),
                             "usd": self.hard_usd,
                             "components_minutes": self.hard_components,
                             "_allowances": {
                                 "setup_worst_case_minutes":
                                     SETUP_WORST_CASE_MINUTES,
                                 "overrun_factor": OVERRUN_FACTOR,
                                 "_overrun_reaches": (
                                     "every measured GPU component except "
                                     "setup, which has its own worst case, "
                                     "and the pre-provider gates, which run "
                                     "before the meter")}},
            "storage": self.storage,
        }


def price_a3(gpu_rate_usd_per_hour: float, n_probes: int = 3) -> A3Price:
    """The complete chain at a given live rate. Pure given the tree."""
    rate = float(gpu_rate_usd_per_hour)
    if not (0.0 < rate < 100.0):
        raise A3PricingError(f"implausible live rate ${rate}/h; refusing to price")
    storage = storage_requirement(n_probes)
    disk_gb = int(storage["provision"]["container_disk_gb"])
    billed = billed_rate(rate, disk_gb)
    exp, hard = component_minutes(False), component_minutes(True)
    exp_m, hard_m = sum(exp.values()), sum(hard.values())
    return A3Price(
        gpu_rate_usd_per_hour=rate, billed_rate_usd_per_hour=billed,
        container_disk_gb=disk_gb,
        expected_minutes=exp_m,
        expected_usd=round(exp_m / 60.0 * billed, 4),
        hard_minutes=hard_m,
        #: CEILED. A limit rounds down; a ceiling rounds up. A planner already
        #: refused once to build a plan terminating $0.00003 above its grant.
        hard_usd=math.ceil(hard_m / 60.0 * billed * 10_000) / 10_000,
        expected_components=exp, hard_components=hard, storage=storage)


def assess(gpu_rate_usd_per_hour: float,
           envelopes: dict[str, float] | None = None,
           n_probes: int = 3) -> dict[str, Any]:
    """The price, and every limit that binds it."""
    env = dict(envelopes if envelopes is not None else live_envelopes())
    p = price_a3(gpu_rate_usd_per_hour, n_probes)
    conditions, shortfalls, book_remaining = evaluate_limits(
        p.hard_usd, env, book=BOOK)
    return {
        "price": p.as_dict(),
        "book": BOOK,
        "_book_because": (
            "A3 trains recovery probes and scores them on c1_confirmation_v1; "
            "the execution package states that the GPU engineering allowance "
            "authorizes neither. What decides the book is what the session "
            "does."),
        "book_remaining_usd": round(book_remaining, 4),
        "envelopes": {k: round(float(v), 4) for k, v in env.items()},
        "conditions": conditions,
        "FUNDABLE": all(conditions.values()),
        "shortfalls_usd": shortfalls,
        "_every_applicable_limit_is_checked": sorted(conditions),
        "_the_envelope_is_not_the_grant": (
            "the authorization receives hard_ceiling.usd, never an envelope"),
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate", type=float, default=None,
                    help="GPU securePrice $/h; omit to select live")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    gpu = "(rate supplied)"
    rate = args.rate
    if rate is None:
        from experiments.phase_c3.hardware import query_offers, select
        offers = query_offers()
        chosen = select(offers)
        if chosen is None or not chosen.usable:
            raise A3PricingError("the approved tier has no usable capacity")
        gpu, rate = chosen.gpu_type_id, chosen.secure_price_usd_per_hour

    f = assess(rate)
    p = f["price"]
    st = p["storage"]
    print(f"A3 complete chain on {gpu} at ${rate}/h")
    print(f"  container disk     {p['container_disk_gb']} GB  "
          f"(peak {st['container_residency']['peak_gib']} GiB at "
          f"{st['container_residency']['peak_at_unit']}, headroom "
          f"{st['provision']['headroom_gib']} GiB)")
    print(f"  durable            {st['durable_requirement']['gib']} GiB")
    print(f"  billed rate        ${p['billed_rate_usd_per_hour']:.6f}/h")
    print(f"  expected           {p['expected']['minutes']:8.2f} min  "
          f"${p['expected']['usd']:8.4f}")
    print(f"  HARD               {p['hard_ceiling']['minutes']:8.2f} min  "
          f"${p['hard_ceiling']['usd']:8.4f}")
    print()
    for name, ok in f["conditions"].items():
        print(f"  [{'OK ' if ok else 'NO '}] {name}")
    print(f"\n  FUNDABLE: {f['FUNDABLE']}")
    for k, v in f["shortfalls_usd"].items():
        print(f"    short on {k}: ${v:.4f}")

    if args.write:
        doc = {
            "schema": "aadistill.phase_c3.a3_pricing/v1",
            "_contract": (
                "The COMPLETE A3 chain, priced from measured components at a "
                "live rate, with every applicable limit checked. Supersedes "
                "the three-shape step-1/step-2 pricing. AUTHORIZES NOTHING."),
            "gpu": gpu, "queried_rate_usd_per_hour": rate,
            "components": COMPONENTS,
            **f,
        }
        (REPO / args.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0 if f["FUNDABLE"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
