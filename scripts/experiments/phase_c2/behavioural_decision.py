"""The frozen Phase-C decision, applied to C2's confirmation evidence.

The inference is C1's and is reused unchanged: `paired_differences`,
`stratified_cluster_bootstrap` and `decide` are imported, not reimplemented.
What differs is the *view* they are given — C2's three confirmation seeds, C2's
own bootstrap seed, and C2's arms — and that view is derived from the frozen C2
protocol rather than borrowed from a C1 experiment that does not describe this
one.

Two things this file exists to prevent.

**A fake C1 isolation plan.** `decide` takes a plan object, and the obvious
shortcut is to build a `C1IsolationPlan` to satisfy it. That class enforces
invariants about an ATTENTION-isolation experiment — two arms differing in their
ATTENTION implementation, "otherwise nothing is being isolated" — which are not
true of C2, where a candidate and the incumbent differ in a whole construction.
Constructing one anyway would mean lying to a validator to get past it. `decide`
reads six fields; this supplies exactly those six, from the protocol that froze
them.

**The wrong bootstrap seed.** `stratified_cluster_bootstrap` defaults to
`bootstrap_seed()`, which domain-separates under `:phase-c1:bootstrap` and
returns 816109261. C2 froze its own under `:phase-c2:bootstrap`, 834816710.
They are different numbers, so a silent fallback would compute a C2 verdict on
C1's resampling draw and nothing in the output would say so. The seed is passed
explicitly, and derived from the protocol's stated rule rather than trusted as a
transcribed constant.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from aadistill.evaluation import measurement_field as MF

REPO_ROOT = Path(__file__).resolve().parents[3]

#: B is the incumbent and the advanced candidate is the treatment. The scorer's
#: `--arm` takes exactly these two, and the estimand is defined in that
#: direction: Delta = correct_candidate - correct_B.
INCUMBENT_ARM = "incumbent"
TREATMENT_ARM = "treatment"

#: `decide` emits "NO-GO"; the C2 protocol's terminal vocabulary spells it
#: "NO_GO". Same terminal state, two spellings, and the implementation's is
#: frozen C1 machinery this stage reuses unchanged — so the spelling is mapped
#: here rather than by editing either side. A consumer matching on the
#: protocol's spelling would otherwise silently never match.
VERDICT_SPELLING = {"GO": "GO", "NO-GO": "NO_GO", "INCONCLUSIVE": "INCONCLUSIVE"}
TERMINAL_STATES = ("GO", "NO_GO", "INCONCLUSIVE")


class BehaviouralDecisionError(RuntimeError):
    """The confirmation evidence cannot support the frozen rule."""


@dataclass(frozen=True)
class C2DecisionRule:
    """Exactly what `decide` reads, derived from the frozen C2 protocol.

    Not a `C1IsolationPlan`. It carries no arms, no ATTENTION implementations
    and no isolation semantics, because C2 is a selection between whole
    constructions and those invariants would be false here.
    """

    plan_hash: str
    seeds: tuple[int, int, int]
    sesoi: float
    seed_robustness_min_positive: int
    usable_pooled_min_delta: float
    usable_per_seed_min_delta: float
    bootstrap_seed: int
    bootstrap_iterations: int
    alpha: float
    catastrophic_candidate_max: float
    catastrophic_control_min: float
    battery_asset_id: str
    battery_content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash, "seeds": list(self.seeds),
            "sesoi": self.sesoi,
            "seed_robustness_min_positive": self.seed_robustness_min_positive,
            "usable_pooled_min_delta": self.usable_pooled_min_delta,
            "usable_per_seed_min_delta": self.usable_per_seed_min_delta,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_iterations": self.bootstrap_iterations,
            "alpha": self.alpha,
            "catastrophic_candidate_max": self.catastrophic_candidate_max,
            "catastrophic_control_min": self.catastrophic_control_min,
            "battery_asset_id": self.battery_asset_id,
            "battery_content_sha256": self.battery_content_sha256,
        }


def decision_rule(repo_root: str | Path = REPO_ROOT) -> C2DecisionRule:
    """Read the rule out of the frozen protocol. Nothing here is a default."""
    from experiments.phase_c2 import behavioural as BH

    proto = BH.protocol(repo_root)
    beh = proto["behavioural_selection"]
    fs = beh["frozen_science"]
    guard = fs["behavioural_guardrails"]["usable_rollout_veto"]

    def _threshold(text: str) -> float:
        """`"Delta_usable > -0.05 absolute"` -> -0.05.

        Requires EXACTLY one numeric token. Taking the first would silently
        pick a threshold out of a sentence that had come to contain two
        numbers, and a guardrail read from the wrong half of its own sentence
        fails in the direction that promotes.
        """
        found = []
        for token in text.replace(",", " ").split():
            try:
                found.append(float(token))
            except ValueError:
                continue
        if len(found) != 1:
            raise BehaviouralDecisionError(
                f"expected exactly one numeric threshold in {text!r}, found "
                f"{found}. Which one governs is not for this reader to guess.")
        return found[0]

    seeds = tuple(int(s) for s in beh["seeds"]["confirmation"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise BehaviouralDecisionError(
            f"the frozen protocol registers {seeds} as confirmation seeds; the "
            "rule needs exactly three distinct ones")

    #: Re-derive the bootstrap seed from the rule the protocol states, rather
    #: than trusting the transcribed integer beside it.
    stated = int(beh["seeds"]["bootstrap_seed"])
    digest = hashlib.sha256(
        f"{beh['seeds']['base_digest']}:phase-c2:bootstrap".encode()).digest()
    derived = int.from_bytes(digest[:4], "big") % (2 ** 31)
    if derived != stated:
        raise BehaviouralDecisionError(
            f"the protocol states bootstrap_seed={stated} but its own stated "
            f"derivation yields {derived}")

    return C2DecisionRule(
        plan_hash=proto["protocol_sha256"],
        seeds=seeds,
        sesoi=float(fs["effect_sizes"]["sesoi"]),
        #: ">=2 of the 3 seed-specific deltas > 0", from the frozen rule text.
        seed_robustness_min_positive=2,
        usable_pooled_min_delta=_threshold(guard["pooled"]),
        usable_per_seed_min_delta=_threshold(guard["per_seed"]),
        bootstrap_seed=int(beh["seeds"]["bootstrap_seed"]),
        bootstrap_iterations=int(fs["bootstrap"]["iterations"]),
        alpha=0.05,
        #: READ, not repeated. The protocol records the veto verbatim from the
        #: implementation it preserves; hardcoding 0.10/0.40 here would make
        #: this a second owner of a frozen threshold, and two owners of one
        #: number disagree the first time either moves.
        catastrophic_candidate_max=float(
            fs["behavioural_guardrails"]["catastrophic_capability_veto"]
            ["candidate_max"]),
        catastrophic_control_min=float(
            fs["behavioural_guardrails"]["catastrophic_capability_veto"]
            ["control_min"]),
        battery_asset_id=beh["batteries"]["confirmation"]["asset_id"],
        battery_content_sha256=beh["batteries"]["confirmation"]["content_sha256"],
    )


#: The admission rule lives in the CORE, beside the paired arithmetic it
#: guards: `aadistill.evaluation.measurement_field`. Nothing in it names C2, so
#: C3 and C4 inherit it by importing core rather than by importing a C2 script
#: -- placement is what makes a rule inheritable, not genericity alone.
#:
#: C2 is why it exists. Its confirmation field carried a uniform battery,
#: scoring contract and metric contract, and THREE distinct generation protocol
#: fingerprints: the four attempt5 probes on one, incumbent B's third seed on a
#: second, and the candidate's third seed on a third. The paired difference at
#: that seed was therefore computed across two protocols -- a confound inside
#: the pair, on the seed with the largest magnitude. Nothing refused it.
PROTOCOL_IDENTITY_FIELDS = MF.PROTOCOL_IDENTITY_FIELDS
GENERATION_IDENTITY_FIELD = MF.GENERATION_IDENTITY_FIELD


def assert_one_measurement_protocol(
        protocols: Mapping[Any, Mapping[str, Any]], *,
        context: str = "confirmation field") -> dict[str, Any]:
    """Refuse unless every probe establishes ONE compatible measurement protocol.

    The core rule, with this module's error type: every caller and test here
    handles `BehaviouralDecisionError`, and a refusal that escaped as a
    different class would pass straight through the recompute's clean-refusal
    path and surface as a traceback on a paid pod instead of exit 3.
    """
    try:
        return MF.assert_one_measurement_protocol(protocols, context=context)
    except MF.MeasurementFieldError as exc:
        raise BehaviouralDecisionError(str(exc)) from exc


def confirm(per_sample, *, rule: C2DecisionRule,
            protocols: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    """The frozen three-way verdict, from the six confirmation probes' rows.

    `per_sample` maps `(arm, seed) -> that probe's per-prompt rows`, exactly the
    shape C1's stage I assembles, with `arm` in {`incumbent`, `treatment`} — B
    and the advanced candidate respectively.

    The alignment is `decision_inputs`', not a second implementation of it. That
    function refuses a duplicate prompt id, a prompt set that differs between
    probes, a scorable count that is not 850 and a total that is not 950 — every
    one of which would otherwise still produce a number. An earlier draft of this
    function aligned the rows by hand and guessed the schema wrong, reading
    `prompt_id`/`usable_rollout` for what the scorer writes as `id`/`usable`,
    which is the class of error that having one owner prevents.
    """
    from experiments.phase_c1.isolation import (
        bootstrap_seed, decide, paired_differences,
        stratified_cluster_bootstrap,
    )
    from experiments.phase_c1.probe_results import decision_inputs

    #: The fallback this function must not take. Asserted rather than assumed:
    #: if the two ever coincided, passing the seed explicitly would stop being a
    #: check on anything and the guard below would pass vacuously.
    c1_fallback = bootstrap_seed()
    if rule.bootstrap_seed == c1_fallback:
        raise BehaviouralDecisionError(
            f"C2's bootstrap seed equals C1's default ({c1_fallback}); they are "
            "meant to be domain-separated, and a collision would make passing "
            "it explicitly a check on nothing")

    if len(per_sample) != 6:
        raise BehaviouralDecisionError(
            f"the confirmation rung is six probes; got {len(per_sample)}. A "
            "verdict from a partial field is not the preregistered experiment.")

    #: BEFORE any arithmetic. A paired interval over probes that were not
    #: measured alike has no estimand, and C2's field -- three distinct
    #: generation protocol fingerprints across six probes, two of them inside
    #: one seed's pair -- reached a verdict because nothing asked. `protocols`
    #: is required rather than optional so a caller cannot omit it and get a
    #: number anyway.
    protocol_report = assert_one_measurement_protocol(
        protocols, context="confirmation field")
    if set(map(str, protocols)) != set(map(str, per_sample)):
        raise BehaviouralDecisionError(
            "the protocols declared and the probes measured are different "
            f"sets: {sorted(map(str, protocols))} vs "
            f"{sorted(map(str, per_sample))}. A protocol record that does not "
            "name the probe it describes certifies nothing about it.")
    seeds = sorted(rule.seeds)
    seen = sorted({int(s) for _, s in per_sample})
    if seen != seeds:
        raise BehaviouralDecisionError(
            f"the probes cover seeds {seen}; the frozen rule registers {seeds}")
    arms = {a for a, _ in per_sample}
    if arms != {INCUMBENT_ARM, TREATMENT_ARM}:
        raise BehaviouralDecisionError(
            f"the confirmation arms are {sorted(arms)}; the frozen binding is "
            f"{INCUMBENT_ARM} = the C1 treatment B and {TREATMENT_ARM} = the "
            "candidate screening advanced")

    #: The veto operands are NAMED, not inherited. The protocol requires the
    #: control operand to be stated explicitly and never silently re-pointed;
    #: for C2 it is the incumbent B. The thresholds come from the frozen rule.
    inputs = decision_inputs(
        per_sample, seeds=seeds,
        candidate_operand=TREATMENT_ARM, control_operand=INCUMBENT_ARM,
        catastrophic_candidate_max=rule.catastrophic_candidate_max,
        catastrophic_control_min=rule.catastrophic_control_min)

    d = paired_differences(inputs.arm(INCUMBENT_ARM), inputs.arm(TREATMENT_ARM))

    #: EXPLICIT. `stratified_cluster_bootstrap` falls back to `bootstrap_seed()`
    #: — C1's, 816109261 — and a C2 verdict computed on C1's resampling draw
    #: would be a different number with nothing in the output to say so.
    boot = stratified_cluster_bootstrap(
        d, inputs.strata, iterations=rule.bootstrap_iterations,
        seed=rule.bootstrap_seed, alpha=rule.alpha)

    per_seed_delta = [
        sum(bool(inputs.correct[TREATMENT_ARM][s][j])
            - bool(inputs.correct[INCUMBENT_ARM][s][j]) for j in d) / len(d)
        for s in seeds]

    outcome = decide(rule, boot=boot, per_seed_delta=per_seed_delta,
                     usable_pooled_delta=inputs.usable_pooled_delta,
                     usable_per_seed_delta=list(inputs.usable_per_seed_delta),
                     catastrophic_violations=inputs.catastrophic_violations)
    terminal = VERDICT_SPELLING.get(outcome["verdict"])
    if terminal is None:
        raise BehaviouralDecisionError(
            f"the frozen rule returned {outcome['verdict']!r}, which is not one "
            f"of C2's terminal states {TERMINAL_STATES}")
    return {
        "rule": rule.as_dict(),
        "measurement_protocol": protocol_report,
        "n_prompts": len(d),
        "bootstrap": boot,
        "bootstrap_seed_used": rule.bootstrap_seed,
        "c1_default_bootstrap_seed_not_used": c1_fallback,
        "per_seed_delta_correct": per_seed_delta,
        "usable_pooled_delta": inputs.usable_pooled_delta,
        "usable_per_seed_delta": list(inputs.usable_per_seed_delta),
        "catastrophic_violations": [dict(v)
                                    for v in inputs.catastrophic_violations],
        "audit": dict(inputs.audit),
        "decision": outcome,
        "terminal_state": terminal,
        "_arms": {INCUMBENT_ARM: "frozen_c1_treatment_b",
                  TREATMENT_ARM: "the candidate screening advanced"},
        "_spelling": (
            f"the frozen rule emits {outcome['verdict']!r}; C2's terminal "
            f"vocabulary spells it {terminal!r}. Same state."),
    }
