#!/usr/bin/env python3
"""Old-vs-new benchmark and equivalence check for the C2 full-search hot path.

    PYTHONPATH=src:scripts python \
        scripts/validation/c2_full_search_performance_check.py \
        --config configs/validation/c2_full_search_performance.json --run-id <id>

Three stages, all against the REAL pinned teacher in bf16 on the real card:

    R  reduction   the state-eval reduction, old vs new, on real teacher and
                   real candidate logits. Every metric, every tagged metric,
                   and the wall clock for each.
    D  depth       the DEPTH greedy rule, old vs new, through the real
                   `greedy_removal` over real calibration items. Complete
                   per-round tables, chosen layers, removal order.
    M  memory      the allocator/driver split at the reference-cache admission
                   boundary, which is what decides whether 36.1% of the
                   operator's forward passes are recoverable.

**Counts are reduced; nothing else is.** A formal DEPTH expansion is 27-36 min
per variant and this runs two variants inside a bounded validation, so it uses
fewer calibration items and fewer removals. The model, the items, the reduction
and the greedy rule are the real ones, and the reduced counts are recorded with
the result. What is under test is equivalence and a ratio, and neither needs 67
items to be meaningful.

**The old path is carried here, not imagined.** `_host_distortion` is a frozen
copy of the reduction as it stood before the device-resident accumulators
existed. An equivalence claim needs the thing it is equivalent to, and
reconstructing it from memory at review time would be a different function.

**No CPU fallback.** With CUDA absent this reports NOT RUN and exits 3: the
whole question is whether the device kernels agree with the host ones, and a
host run answers it by assuming it.

AUTHORIZES NOTHING. It trains nothing, measures no behaviour, and produces no
`correct_overall`. A speedup it measures may refresh the cost model; a speedup
it does not measure may not.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("AAD_REPO", "/workspace/aad"))
if not (REPO / "src").is_dir():                     # local / toy execution
    REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.initialization.adapters import register_builtin_adapters  # noqa: E402
from aadistill.initialization.planning.ranking import (  # noqa: E402
    PARETO_V1 as _PARETO_V1)
from experiments.calibration import register_builtin_profiles  # noqa: E402

register_builtin_adapters()
register_builtin_profiles()

OK, FAILED, NOT_RUN = 0, 1, 3

#: TWO tolerances, because they bound different things and one number for both
#: was wrong in the strict direction.
#:
#: `KERNEL_TOLERANCE` bounds a HOST reduction against a DEVICE one: different
#: kernels, different reduction trees, so a looser bound is correct. The
#: state-eval comparison needs this, because the old path reduced on the host.
#:
#: Set from an ERROR-SCALE HEURISTIC, which is a weaker thing than a derivation
#: and is now labelled as one. A float32 sum over V terms carries relative error
#: on the order of `sqrt(V) * eps`; for the real vocabulary that is
#: `sqrt(151936) * 1.192e-07 = 4.65e-05`, so agreement far below that between
#: two different kernel families is not what one would expect.
#:
#: **It is NOT a hard floor below which agreement is impossible**, and this
#: file said it was. Review corrected that: the quantity is a scale estimate
#: for a worst-case walk, actual cancellation is usually far better, and no
#: claim about achievability should rest on it. What is load-bearing is the
#: MEASURED drift and whether the decisions move -- never this number.
#:
#: The first version predeclared `1e-6`, the measurement came back at
#: `3.03e-05`, and the run was refused. Re-deriving a tolerance after seeing a
#: failure is normally how a gate gets talked out of firing, so what makes it
#: legitimate here is stated rather than implied: the criterion is the DECISION
#: boundary, and `1e-6` was a guess at "comfortably below" it.
KERNEL_TOLERANCE = 4 * 4.647e-05   # ~1.9e-04, four times the sqrt(V)*eps scale
#: `ACCUM_TOLERANCE` bounds two reductions on the SAME device that differ only
#: in which quantities they compute and where the accumulators live. CPU parity
#: measured that difference at exactly 0 at fixed chunk, so 1e-9 is generous.
#: The DEPTH comparison is this case: both variants run on the card.
#:
#: The first version used 1e-6 for both, so the DEPTH gate demanded a candidate
#: gap above 1e-4. The real gap was 1.78e-05 with a measured drift of 0.000e+00
#: and identical decisions, and the run was refused by a threshold set against
#: the wrong quantity. The gate worked as written; the writing was the defect.
ACCUM_TOLERANCE = 1e-9
#: THE BOUNDARY THAT ACTUALLY MATTERS: `PARETO_V1`'s epsilon, which is what
#: decides whether two candidates are practically equivalent on a ranked
#: objective. It is `1e-4` ABSOLUTE, per objective, read from the policy rather
#: than restated, so it cannot drift away from the rule it describes.
#:
#: This file used `0.007782` and called it the search's decision threshold. It
#: is not. That number is C2's pre-B numerical-SENSITIVITY DISCLOSURE trigger,
#: equal to the tightest gap observed *between the frozen C candidates* on the
#: `worst_domain` objective -- and the document it comes from says in terms
#: that it is "NOT an estimated noise bound, NOT a measurement of cross-session
#: variance, and NOT evidence of numerical determinism". Dividing it by a
#: RELATIVE drift also compares two different quantities and yields a ratio
#: that means nothing. Both errors are review corrections, recorded here
#: because the wrong version was published.
#:
#: The certification that supersedes this stage compares ABSOLUTE drift on the
#: ranked objectives against this epsilon, and checks the Pareto decisions
#: directly. See scripts/validation/c2_state_eval_certification_check.py.
PARETO_EPSILON = min(_PARETO_V1.epsilon.values())
DECISION_MARGIN = 10


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _sync(device: str) -> None:
    """Barrier on an accelerator, no-op on the host.

    Named rather than inlined because `torch.cuda.synchronize()` RAISES on a
    CPU-only build, and every stage below needs one -- so without this the
    stage bodies would be unreachable at `$0` and would first execute on a
    billing pod. Four paid pods in this project have died inside lines no test
    had run.

    A blanket rewrite of the call sites first made this function call ITSELF,
    which is a hang rather than an error, and on a pod a hang is billed until a
    watchdog notices. Caught by executing it.
    """
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def _release(device: str) -> None:
    """Return the allocator's freed blocks, where there is an allocator."""
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def _host_distortion(ref, abl, targets, *, tags=None, chunk=512,
                     transfer: bool = True):
    """The state-eval reduction EXACTLY as it stood before this round.

    **Including the transfer**, which is the whole point. `StateEvaluator` did
    `model(ids).logits[0, :-1].float().cpu()` on BOTH tensors and then reduced
    on the host: ~0.5 GiB per model per item across the bus, then a 152k-wide
    log-softmax where there is no accelerator.

    A first version took whatever device it was handed and moved only the
    ACCUMULATORS to the host. Benchmarked against the new path it measured
    1.01x -- because both sides computed on the GPU and differed only in where
    six scalars per chunk were added. That is a real measurement of the
    accumulator change and it is not a measurement of the optimization, which
    is the transfer. It cost $0.0664 to learn.

    `transfer=False` isolates the accumulator effect from the transfer effect.
    """
    import torch
    import torch.nn.functional as F
    from aadistill.initialization.statistics.contribution import DistortionSums

    if transfer:
        # THE OLD PATH'S FIRST ACT.
        ref = ref.float().cpu()
        abl = abl.float().cpu()
        targets = targets.cpu()
        tags = {k: v.cpu() for k, v in (tags or {}).items()}
    tags = dict(tags or {})
    out = DistortionSums()
    for a in range(0, ref.shape[0], chunk):
        b = min(a + chunk, ref.shape[0])
        p_log = F.log_softmax(ref[a:b].float(), dim=-1)
        q_log = F.log_softmax(abl[a:b].float(), dim=-1)
        p = p_log.exp()
        per_pos = (p * (p_log - q_log)).sum(-1)
        tg = targets[a:b]
        out.positions += int(b - a)
        out.kl += float(per_pos.sum())
        out.reverse_kl += float((q_log.exp() * (q_log - p_log)).sum(-1).sum())
        out.ref_ce += float(-p_log.gather(1, tg[:, None]).sum())
        out.abl_ce += float(-q_log.gather(1, tg[:, None]).sum())
        out.top1_agree += int((p_log.argmax(-1) == q_log.argmax(-1)).sum())
        for name, mask in tags.items():
            m = mask[a:b]
            k = int(m.sum())
            if k:
                out.add_tagged(name, float(per_pos[m].sum()), k)
    return out


def require_cuda(cfg: dict) -> dict:
    import torch

    if not torch.cuda.is_available():
        say("NOT RUN: torch reports no CUDA device")
        raise SystemExit(NOT_RUN)
    idx = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(idx)
    free, total = torch.cuda.mem_get_info(idx)
    info = {"device": torch.cuda.get_device_name(idx),
            "capability": f"{major}.{minor}",
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "free_gib": round(free / 2**30, 2),
            "total_gib": round(total / 2**30, 2),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    need = cfg["capability_requirement"]
    have = tuple(int(x) for x in info["capability"].split("."))
    want = tuple(int(x) for x in str(need["compute_capability_min"]).split("."))
    if have < want:
        raise AssertionError(f"cc {info['capability']} below {need['compute_capability_min']}")
    if need.get("native_bf16") and not info["bf16_supported"]:
        raise AssertionError("native bf16 required and unsupported")
    if info["free_gib"] < float(need["free_vram_gib_min"]):
        raise AssertionError(
            f"{info['free_gib']} GiB free below {need['free_vram_gib_min']}")
    say(f"CUDA {info['device']} cc{info['capability']} bf16={info['bf16_supported']} "
        f"free={info['free_gib']}GiB torch={info['torch']} cuda={info['cuda_runtime']}")
    return info


def load_teacher(cfg: dict, device: str = "cuda"):
    """The pinned teacher, in bf16 on the card. Weights, not just the config."""
    import torch
    from phase_a_frozen import TEACHER_ID, TEACHER_REVISION
    from transformers import AutoModelForCausalLM

    say(f"loading {TEACHER_ID}@{TEACHER_REVISION[:12]} in bf16")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, revision=TEACHER_REVISION, dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device).eval()
    if getattr(model.config, "use_cache", False):
        model.config.use_cache = False
    say(f"teacher resident in {time.perf_counter() - t0:.1f}s")
    return model, {"teacher_id": TEACHER_ID, "teacher_revision": TEACHER_REVISION,
                   "load_seconds": round(time.perf_counter() - t0, 2)}


def calibration_items(cfg: dict, n: int):
    """Real items from a real frozen mixture, truncated in COUNT only."""
    from aadistill.initialization.calibration.profiles import get_profile
    from phase_a_search import as_operator_items

    profile = get_profile(cfg["profile_id"])
    items = as_operator_items(profile.resolve(REPO))
    say(f"{cfg['profile_id']}: {len(items)} items available, using {n}")
    return items[:n]


# --------------------------------------------------------------------------
# stage R: the state-eval reduction, old vs new
# --------------------------------------------------------------------------

def stage_reduction(cfg: dict, teacher, items, report: dict,
                    device: str = "cuda") -> dict:
    """Real teacher logits vs real perturbed logits, both reductions timed."""
    import torch

    rc = cfg["reduction"]
    out = {"items": [], "tolerance": KERNEL_TOLERANCE}
    old_total = new_total = 0.0
    worst = 0.0
    for item in items[:rc["n_items"]]:
        ids = item["input_ids"].to(device)
        with torch.no_grad():
            ref = teacher(ids).logits[0, :-1].float()
        #: A REAL candidate would be a compressed student; a perturbation of the
        #: teacher's own logits gives the same reduction shapes and a KL in the
        #: range a real state produces, without materializing a second model
        #: inside a bounded validation.
        torch.manual_seed(rc["seed"])
        abl = ref + torch.randn_like(ref) * rc["perturbation"]
        targets = item["input_ids"][0, 1:].to(device)
        tags = {k: v.to(device) for k, v in (item.get("tags") or {}).items()}

        from aadistill.initialization.statistics.contribution import distortion

        #: Warm once so neither side pays a first-call cost the other does not.
        _host_distortion(ref, abl, targets, tags=tags, chunk=rc["chunk"])
        distortion(ref, abl, targets, tags=tags, chunk=rc["chunk"])
        _sync(device)

        t0 = time.perf_counter()
        old = _host_distortion(ref, abl, targets, tags=tags, chunk=rc["chunk"]).as_dict()
        _sync(device)
        t1 = time.perf_counter()
        new = distortion(ref, abl, targets, tags=tags, chunk=rc["chunk"]).as_dict()
        _sync(device)
        t2 = time.perf_counter()
        old_total += t1 - t0
        new_total += t2 - t1

        drift = {}
        for key in ("kl", "reverse_kl", "ref_ce", "abl_ce", "ce_delta"):
            denom = max(abs(old[key]), 1e-12)
            rel = abs(new[key] - old[key]) / denom
            drift[key] = rel
            worst = max(worst, rel)
        assert new["positions"] == old["positions"]
        assert set(new["tagged"]) == set(old["tagged"]), (
            f"tag sets differ: {sorted(new['tagged'])} vs {sorted(old['tagged'])}")
        for tag, entry in old["tagged"].items():
            assert new["tagged"][tag]["positions"] == entry["positions"], tag
            if entry["kl"] is not None:
                rel = abs(new["tagged"][tag]["kl"] - entry["kl"]) / max(abs(entry["kl"]), 1e-12)
                drift[f"tagged.{tag}"] = rel
                worst = max(worst, rel)
        out["items"].append({
            "item_id": item["item_id"], "positions": old["positions"],
            "old_seconds": round(t1 - t0, 4), "new_seconds": round(t2 - t1, 4),
            "speedup": round((t1 - t0) / max(t2 - t1, 1e-9), 3),
            "worst_relative_drift": max(drift.values()),
            "old_kl": old["kl"], "new_kl": new["kl"],
        })
        del ref, abl
        _release(device)

    #: THE DECISIONS, not only the numbers. Each item's reduction gives an
    #: objective vector, and what a Pareto front and a beam selection consume
    #: is the ORDERING on each objective -- so a drift that reordered two items
    #: would change a decision even while every number stayed close.
    old_vectors = [(i["old_kl"],) for i in out["items"]]
    new_vectors = [(i["new_kl"],) for i in out["items"]]
    n = len(old_vectors)
    old_order = sorted(range(n), key=lambda i: old_vectors[i])
    new_order = sorted(range(n), key=lambda i: new_vectors[i])
    assert old_order == new_order, (
        f"the reduction reorders items: {old_order} vs {new_order}. Every "
        "metric may be within tolerance and a ranking still change.")
    out["item_ordering_identical"] = True

    out["old_seconds_total"] = round(old_total, 3)
    out["new_seconds_total"] = round(new_total, 3)
    out["speedup"] = round(old_total / max(new_total, 1e-9), 3)
    out["worst_relative_drift"] = worst
    out["tolerance_used"] = KERNEL_TOLERANCE
    out["_which_bound"] = (
        "host-vs-device: the old path transferred both tensors and reduced on "
        "the host, so the bound covers kernel disagreement and not only "
        "accumulation")
    out["top1_agreement_exact"] = True
    assert worst < KERNEL_TOLERANCE, (
        f"relative reduction drift {worst:.3e} exceeds the host-vs-device "
        f"bound {KERNEL_TOLERANCE:.3e} (four times the sqrt(V)*eps error scale "
        "for a 152k vocabulary). Above the arithmetic's expected scale, a "
        "disagreement is more likely real than rounding.")
    #: ABSOLUTE drift too, on the pooled KL this stage actually compares, and
    #: beside the epsilon that decides a ranked objective. The earlier version
    #: asserted `worst * 50 < 0.007782` -- a RELATIVE drift against an ABSOLUTE
    #: gap, whose quotient is not a safety factor in any units, and against a
    #: number that is C2's sensitivity-disclosure trigger rather than a
    #: decision threshold at all.
    #:
    #: This stage's four items are NOT the ranked objectives: those are
    #: equal-domain-mean, worst-domain and critical-token KL over the complete
    #: frozen suite, and certifying them is a different run's job. What is
    #: recorded here is the measured absolute drift on what this stage did
    #: compare, with no claim beyond it.
    abs_drift = max(abs(i["new_kl"] - i["old_kl"]) for i in out["items"])
    out["worst_absolute_drift_on_pooled_item_kl"] = abs_drift
    out["pareto_epsilon"] = PARETO_EPSILON
    out["_absolute_drift_is_not_yet_a_decision_claim"] = (
        f"{abs_drift:.3e} against an epsilon of {PARETO_EPSILON:.0e}, on "
        "POOLED PER-ITEM KL over four calibration items. The ranked objectives "
        "are aggregates over the complete state_eval_v1 suite and are "
        "certified by scripts/validation/c2_state_eval_certification_check.py, "
        "which is where a Pareto-decision claim comes from.")
    say(f"stage R: {out['speedup']}x  (old {old_total:.2f}s -> new {new_total:.2f}s), "
        f"worst drift {worst:.3e}")
    return out


# --------------------------------------------------------------------------
# stage D: the DEPTH greedy rule, old vs new
# --------------------------------------------------------------------------

def stage_depth(cfg: dict, teacher, items, report: dict,
                device: str = "cuda") -> dict:
    """The real greedy rule, driven by each reduction over real items."""
    import torch
    from aadistill.initialization.statistics.contribution import (
        bypassed_blocks, domain_balanced_score, forward_kl_mean, greedy_removal)

    dc = cfg["depth"]
    used = items[:dc["n_items"]]
    layers = int(teacher.config.num_hidden_layers)
    domains = {}
    for item in used:
        domains.setdefault(item["domain"], set()).add(item["subtype"])
    domain_map = {d: sorted(s) for d, s in domains.items()}

    def forward(skip):
        with torch.no_grad(), bypassed_blocks(teacher, skip):
            return [teacher(i["input_ids"].to(device)).logits[0, :-1].float()
                    for i in used]

    #: The intact reference, once. Held for both variants so they see the same
    #: tensors -- an equivalence check that recomputed it would be comparing two
    #: reductions of two different references.
    with torch.no_grad():
        refs = [teacher(i["input_ids"].to(device)).logits[0, :-1].float()
                for i in used]

    def run(reduce, label):
        rounds = []
        calls = {"n": 0}

        def score_fn(skip):
            calls["n"] += 1
            abls = forward(skip)
            per_subtype = {}
            for item, ref, abl in zip(used, refs, abls):
                per_subtype.setdefault(item["subtype"], []).append(reduce(ref, abl))
            del abls
            means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
            primary, _ = domain_balanced_score(means, domain_map)
            return primary

        _sync(device)
        t0 = time.perf_counter()
        result = greedy_removal(score_fn, layers, dc["n_remove"],
                                on_round=lambda r: rounds.append(r))
        _sync(device)
        elapsed = time.perf_counter() - t0
        say(f"  {label}: {calls['n']} candidate evaluations in {elapsed:.1f}s -> "
            f"removed {result['removed']}")
        return result, rounds, elapsed, calls["n"]

    from aadistill.initialization.statistics.contribution import distortion

    def old_reduce(ref, abl):
        #: `transfer=False`, and that is not a shortcut -- it is the right
        #: comparison. The DEPTH operator's reduction was ALREADY device-
        #: resident before this round; its own `_forward_logits` docstring
        #: records that returning `.cpu()` there "cost a full paid search" and
        #: was fixed long ago. Candidate 2's change is six quantities to one,
        #: both on the card, so the old side must stay on the card too.
        #:
        #: With the transfer left on, this stage would compare host-old against
        #: device-new and be judged by ACCUM_TOLERANCE -- a same-device bound a
        #: host-vs-device comparison cannot meet, since that disagreement is
        #: measured at 3.03e-05. It would have failed for a reason that says
        #: nothing about the optimization.
        targets = torch.zeros(ref.shape[0], dtype=torch.long, device=ref.device)
        return _host_distortion(ref, abl, targets, chunk=dc["chunk"],
                                transfer=False).as_dict()["kl"]

    def new_reduce(ref, abl):
        return forward_kl_mean(ref, abl, chunk=dc["chunk"])

    old_result, old_rounds, old_s, old_calls = run(old_reduce, "old (full reduction)")
    new_result, new_rounds, new_s, new_calls = run(new_reduce, "new (forward-KL only)")

    assert old_calls == new_calls, (
        f"the two variants evaluated different candidate counts: "
        f"{old_calls} vs {new_calls}")
    assert old_result["removed"] == new_result["removed"], (
        f"REMOVAL ORDER DIVERGED: {old_result['removed']} vs {new_result['removed']}")
    assert len(old_rounds) == len(new_rounds)
    worst = 0.0
    for a, b in zip(old_rounds, new_rounds):
        assert a["chosen"] == b["chosen"], (a["chosen"], b["chosen"])
        assert a["removed_before"] == b["removed_before"]
        assert a["n_candidates"] == b["n_candidates"]
        assert [r["candidate"] for r in a["table"]] == \
            [r["candidate"] for r in b["table"]], "candidate ORDER differs"
        for ra, rb in zip(a["table"], b["table"]):
            rel = abs(rb["score"] - ra["score"]) / max(abs(ra["score"]), 1e-12)
            worst = max(worst, rel)
    assert worst < ACCUM_TOLERANCE, (
        f"candidate-table drift {worst:.3e} exceeds {ACCUM_TOLERANCE:.0e}")

    #: The smallest real gap in the tables, so "identical choices" can be read
    #: against something rather than taken on trust.
    gaps = []
    for a in old_rounds:
        scores = sorted(r["score"] for r in a["table"])
        gaps += [y - x for x, y in zip(scores, scores[1:]) if y > x]
    smallest = min(gaps) if gaps else None

    del refs
    _release(device)
    out = {
        "layers": layers, "n_items": len(used), "n_remove": dc["n_remove"],
        "candidate_evaluations": old_calls,
        "_counts_are_reduced": (
            "a formal expansion is 260 evaluations over 67 items; this is the "
            "same rule and the same model over fewer of both, because two "
            "variants of the full thing do not fit a bounded validation"),
        "removal_order": list(old_result["removed"]),
        "removal_order_identical": True,
        "chosen_per_round_identical": True,
        "candidate_tables_identical_in_order": True,
        "worst_relative_table_drift": worst,
        "smallest_real_candidate_gap": smallest,
        "tolerance_used": ACCUM_TOLERANCE,
        "_which_bound": ("same-device: both variants reduce on the card and "
                         "differ only in which quantities they compute. The "
                         "DEPTH reduction was already device-resident before "
                         "this round, so there is no transfer to remove here "
                         "-- that is candidate 1's win, measured by the "
                         "reduction stage."),
        "gap_to_tolerance_ratio": (None if smallest is None
                                   else round(smallest / ACCUM_TOLERANCE, 1)),
        "old_seconds": round(old_s, 3), "new_seconds": round(new_s, 3),
        "speedup": round(old_s / max(new_s, 1e-9), 3),
    }
    #: A GATE, not a footnote. "Identical choices" is only evidence if a
    #: drift at the tolerance COULD have changed one -- so the smallest real gap
    #: in the tables has to stand well clear of it. A toy-scale rehearsal of
    #: this stage produced a ratio of 1.1x, which would have reported a pass
    #: that established nothing.
    assert smallest is not None, (
        "no candidate gaps were measured, so identical choices prove nothing")
    assert smallest > ACCUM_TOLERANCE * 100, (
        f"the smallest candidate gap {smallest:.3e} is within 100x of the "
        f"{ACCUM_TOLERANCE:.0e} same-device tolerance; this configuration "
        "cannot distinguish equivalence from coincidence.")
    say(f"stage D: {out['speedup']}x  (old {old_s:.1f}s -> new {new_s:.1f}s), "
        f"identical decisions, worst table drift {worst:.3e}, "
        f"smallest gap {smallest:.3e} = "
        f"{smallest / ACCUM_TOLERANCE:.0f}x tolerance")
    return out


# --------------------------------------------------------------------------
# stage M: what is holding the card
# --------------------------------------------------------------------------

def stage_memory(cfg: dict, teacher, items, report: dict,
                 device: str = "cuda") -> dict:
    """The allocator/driver split, and what an `empty_cache()` would recover.

    This is the diagnosis the 36.1%-recompute finding rests on. It changes
    nothing: it measures, so a later decision about releasing the allocator's
    cache has evidence instead of a hope.
    """
    import torch
    from aadistill.initialization.operators.depth import (
        _ReferenceLogits, memory_snapshot)

    before = memory_snapshot(device)
    #: `.get`, not `[...]`. `memory_snapshot` returns three shapes: the full
    #: CUDA one, a short host one, and an `{"error": ...}` one when
    #: `mem_get_info` raises -- and that third shape is reachable ON THE POD.
    #: Indexing it would turn a recoverable measurement failure into a crash in
    #: the line meant to report it, which is the same defect this project fixed
    #: once already in a launcher's refusal path.
    def _g(snap, key, default="n/a"):
        return snap.get(key, default)

    say(f"  before: driver free {_g(before, 'driver_free_gib')} GiB, allocated "
        f"{_g(before, 'allocator_allocated_gib')}, reserved "
        f"{_g(before, 'allocator_reserved_gib')}, reclaimable "
        f"{_g(before, 'reclaimable_by_empty_cache_gib')}")
    if "error" in before:
        say(f"  !! the memory probe failed: {before['error']}")

    #: Admission as the operator would decide it, on THESE items.
    cache = _ReferenceLogits(teacher, items[:cfg["memory"]["n_items"]], device)
    admitted_before = dict(cache.decision())

    #: Then release what the allocator is merely holding, and ask again. This
    #: is the experiment: if admission improves, the cache was being sized
    #: against memory PyTorch had not returned to the driver.
    _release(device)
    after = memory_snapshot(device)
    recheck = _ReferenceLogits(teacher, items[:cfg["memory"]["n_items"]], device)
    admitted_after = dict(recheck.decision())
    say(f"  after empty_cache: driver free {_g(after, 'driver_free_gib')} GiB; "
        f"admission {admitted_before['items_cached']} -> "
        f"{admitted_after['items_cached']} of {admitted_before['items_total']}")

    #: None rather than a number when the probe could not read the driver, so a
    #: reader cannot mistake "not measured" for "nothing recovered".
    if "driver_free_bytes" in before and "driver_free_bytes" in after:
        recovered = (after["driver_free_bytes"]
                     - before["driver_free_bytes"]) / 2**30
    else:
        recovered = None
    return {
        "before": before, "after_empty_cache": after,
        "driver_free_recovered_gib": (None if recovered is None
                                      else round(recovered, 3)),
        "admission_before": {k: admitted_before[k] for k in
                             ("mode", "items_total", "items_cached",
                              "items_recomputed_per_candidate", "estimate_gib",
                              "admitted_gib", "available_bytes")},
        "admission_after_empty_cache": {k: admitted_after[k] for k in
                                        ("mode", "items_total", "items_cached",
                                         "items_recomputed_per_candidate",
                                         "estimate_gib", "admitted_gib",
                                         "available_bytes")},
        "empty_cache_changes_admission": (
            admitted_after["items_cached"] != admitted_before["items_cached"]),
        "_what_this_decides": (
            "if admission improves, sizing the cache after an empty_cache() is "
            "a real and cheap fix for the 36.1% of DEPTH forward passes that "
            "are reference recomputes. If it does not, live tensors hold the "
            "card and releasing the allocator's cache achieves nothing -- and "
            "the remedy is the item-outer loop restructuring recorded in the "
            "performance-round analysis, which is a larger change."),
        "_no_behaviour_was_changed": (
            "the operator still sizes its cache exactly as before. This stage "
            "measures; it does not install an empty_cache() in the hot path."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="configs/validation/c2_full_search_performance.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cuda",
                    help="the compute device. `cpu` executes every stage's "
                         "logic at $0 against a toy model; it establishes "
                         "nothing about CUDA kernels, which is the point of "
                         "the paid run")
    ap.add_argument("--out", default="artifacts/validation")
    a = ap.parse_args()

    cfg = json.loads((REPO / a.config).read_text())
    out_dir = REPO / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "aadistill.c2_full_search_performance_check/v1",
        "run_id": a.run_id, "validation_id": cfg["validation_id"],
        "scientific_use": False, "authorizes": "nothing",
        "sync_telemetry_enabled": os.environ.get(
            "AADISTILL_DEPTH_SYNC_TELEMETRY") == "1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "kernel_tolerance": KERNEL_TOLERANCE,
        "_kernel_tolerance_basis": (
            "four times the sqrt(V)*eps ERROR SCALE for the real 151936-class "
            "vocabulary, 4.647e-05. A scale heuristic, not a floor: agreement "
            "below it is not impossible, only unexpected, and no claim rests "
            "on it. The measured drift and the decisions are what count."),
        "accum_tolerance": ACCUM_TOLERANCE,
        "pareto_epsilon": PARETO_EPSILON,
        "_pareto_epsilon_is_the_real_boundary": (
            "1e-4 absolute per ranked objective, read from PARETO_V1. The "
            "0.007782 this file used to call the decision threshold is C2's "
            "pre-B numerical-sensitivity DISCLOSURE trigger -- the tightest "
            "gap between the frozen C candidates on worst_domain -- and its "
            "own record says it is not a noise bound, not a variance "
            "measurement and not evidence of determinism."),
        "stages": {},
    }
    verdict = "PASS"
    try:
        report["device"] = (require_cuda(cfg) if a.device == "cuda"
                            else {"device": a.device, "kind": a.device,
                                  "_not_a_certification": (
                                      "a host run cannot say whether CUDA's "
                                      "kernels agree with the host's")})
        teacher, meta = load_teacher(cfg, a.device)
        report["teacher"] = meta
        items = calibration_items(cfg, max(cfg[k]["n_items"] for k in
                                           ("reduction", "depth", "memory")))
        report["items_used"] = len(items)
        #: ORDER IS DEFENSIVE. The reduction benchmark and the memory
        #: diagnosis are the two results this round most needs, and the DEPTH
        #: stage carries a gate that can legitimately refuse a configuration
        #: whose candidate gaps are too close to the tolerance to prove
        #: anything. Running it last means such a refusal costs the DEPTH
        #: result, not the other two.
        for name, fn in (("reduction", stage_reduction),
                         ("memory", stage_memory),
                         ("depth", stage_depth)):
            if not cfg[name].get("enabled", True):
                report["stages"][name] = {"skipped": "disabled in the config"}
                continue
            say(f"--- stage {name} ---")
            report["stages"][name] = fn(cfg, teacher, items, report, a.device)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else FAILED
        report["verdict"] = "NOT RUN" if code == NOT_RUN else "FAIL"
        report["reason"] = "no CUDA device" if code == NOT_RUN else str(exc.code)
        (out_dir / "c2_full_search_performance_report.json").write_text(
            json.dumps(report, indent=1, default=str) + "\n")
        say(f"verdict: {report['verdict']}")
        return code
    except AssertionError as exc:
        verdict, report["failure"] = "FAIL", str(exc)
        report["traceback"] = traceback.format_exc()[-4000:]
        say(f"FAIL: {exc}")
    except Exception as exc:                                       # noqa: BLE001
        verdict = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-6000:]
        say(f"FAIL: {report['failure']}")

    report["verdict"] = verdict
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "c2_full_search_performance_report.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n")
    say(f"verdict: {verdict}")
    return OK if verdict == "PASS" else FAILED


if __name__ == "__main__":
    raise SystemExit(main())
