"""The A3 comparison, driven on a cheaply-built replica of a finished run.

**Why this test exists at all.** attempt75 trained, preserved and scored nine
probes over 919 minutes for `$16.71`, and then its aggregation raised on a
one-line vocabulary default and wrote no decision. The code path that only a
completed experiment reaches was the only code path nothing had executed. So
this builds the SHAPE of three finished treatment probes from the controls'
own evidence — real per-sample rows, real scored-record schema — and drives the
real `build()` over it.

Nothing here claims to predict A3's result. It claims that when A3's evidence
exists, the thing that reads it works, refuses what it should refuse, and
reports every quantity the design promises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "autoinit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import aggregate_a3 as AG  # noqa: E402

BOOTSTRAP_SEED = 654678655          # phase-c3's own, domain-separated
#: Small, because an interval's SHAPE is what is under test here and 20_000
#: resamples of 850 prompts three times over is a minute of CPU for nothing.
ITERATIONS = 200


#: THE SIGNAL THIS SKIP READS, named inline so the skip-predicate audit can
#: classify it instead of reporting it `unrecognised`. A
#: `devbox_only_artifact`: attempt75's durable evidence lives under the host's
#: home and no manifest stages it onto a pod, so these tests skip there by
#: design — which is correct rather than a defect, because A3's comparison runs
#: OFF POD and belongs where the evidence is.
#:
#: `Path.home()`, NOT a hardcoded `/home/ecs-user`. That class's parity holds
#: because the contract points HOME at a fresh empty directory on BOTH
#: machines, so both decide absent; a literal would sit outside the declared
#: isolation and the two would disagree. The repository replaced its hardcoded
#: literals for exactly this reason.
CONTROL_EVIDENCE_ROOT = str(Path.home() / "aad-artifacts/phase_c3/attempt75")


def _controls_present(root: str = CONTROL_EVIDENCE_ROOT) -> bool:
    if not Path(root).is_dir():
        return False
    try:
        probes = AG.load_controls()
    except AG.A3AggregationError:
        return False
    return all(Path(p["per_sample_path"]).is_file()
               and Path(p["result_path"]).is_file() for p in probes.values())


needs_controls = pytest.mark.skipif(
    not _controls_present(CONTROL_EVIDENCE_ROOT),
    reason=(f"attempt75's durable control evidence is not mounted at "
            f"{CONTROL_EVIDENCE_ROOT}; the comparison reads per-sample rows "
            "and scored records and will not fabricate them"))


def _admission(label: str, **overrides) -> dict:
    """The record the DRIVER writes beside each probe, in its real shape.

    The rule id is read from `generation_compat` rather than spelled here, so a
    renamed or revised rule turns these tests red instead of leaving a fixture
    asserting a string the production code stopped using.
    """
    from aadistill.initialization.planning.generation_compat import (
        GENERATION_RUNTIME_COMPARABILITY_V2,
    )

    same = "f9d5bc543e49c4513f1bc700d914801ad3c8a6417ad0c75522b27622211ec020"
    cmp = {
        "rule": GENERATION_RUNTIME_COMPARABILITY_V2.as_dict()["qualified_id"],
        "live_identity": same,
        "historical_identity": same,
        "identities_equal": True,
        #: A PATCH apart within one branch, which is the real attempt38 case
        #: and the one the v2 rule exists to admit.
        "live_driver": "580.126.09",
        "historical_driver": "580.159.03",
        "driver_branch_equal": True,
        "driver_patch_differs": True,
    }
    cmp.update(overrides.pop("comparability", {}))
    rec = {"probe_id": label, "comparability": cmp, "comparable": True}
    rec.update(overrides)
    return rec


def _write_treatment(root: Path, *, flip: int = 0, field: dict | None = None,
                     seeds: list[int] | None = None,
                     drop_rows_for: int | None = None,
                     admission: dict | None = None,
                     drop_admission_for: int | None = None) -> Path:
    """Three finished A_bsz3 probes, built from the controls' real evidence.

    `flip` turns that many scorable prompts from incorrect to correct in the
    treatment, which is how a known non-zero delta gets into the fixture: a
    comparison tested only on identical inputs cannot show that it measures
    anything.
    """
    controls = AG.load_controls()
    full = AG.load_control_results()
    audit = root / "audit" / "autoinit_a3"
    audit.mkdir(parents=True, exist_ok=True)
    use = seeds if seeds is not None else sorted(controls)

    for seed in use:
        probe = controls[seed]
        label = f"autoinit.v1.phase_a3.A_bsz3.{seed}"
        rows = [json.loads(line) for line
                in Path(probe["per_sample_path"]).read_text().splitlines()
                if line.strip()]
        flipped = 0
        out_rows = []
        for row in rows:
            row = dict(row)
            row["label"] = label
            if (flipped < flip and row["scorable"] and not row["correct"]
                    and row["usable"]):
                row["correct"] = True
                row["scorer_correct"] = True
                flipped += 1
            out_rows.append(row)

        if drop_rows_for != seed:
            (audit / f"{label}_per_sample.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in out_rows))

        scored = dict(full[seed])
        scored["label"] = label
        scored["seed"] = seed
        #: PERTURB THE FIELD THE CONSUMER READS, which is nested for two of
        #: the three. Writing `scoring_contract_digest` at the top level left
        #: `scoring_contract.digest` untouched, so the mismatch test passed
        #: against an unperturbed record -- the fixture's bug, and exactly the
        #: shape of "read the field the consumer reads".
        nested = {"battery_manifest_sha256": ("battery", "manifest_sha256"),
                  "battery_content_sha256": ("battery", "content_sha256"),
                  "scoring_contract_digest": ("scoring_contract", "digest")}
        for key, value in (field or {}).items():
            if key in nested:
                block, leaf = nested[key]
                scored[block] = {**scored[block], leaf: value}
            else:
                scored[key] = value
        n_scorable = sum(1 for r in out_rows if r["scorable"])
        correct = sum(1 for r in out_rows if r["correct"])
        scored["correct"] = correct
        scored["n_scorable"] = n_scorable
        scored["correct_overall"] = round(correct / n_scorable, 4)
        (audit / f"{label}_c1_confirmation.json").write_text(
            json.dumps(scored, indent=1))

        #: The comparability premise, written where the driver writes it. A
        #: replica that omits it would exercise a comparison the real path
        #: cannot reach, which is the whole point of building the shape of a
        #: finished run.
        if drop_admission_for != seed:
            (audit / f"{label}_generation_admission.json").write_text(
                json.dumps(_admission(label, **(admission or {})), indent=1))
    return root


# --- it runs, end to end, on a finished-shaped field ----------------------


@needs_controls
def test_the_comparison_runs_on_a_finished_field_and_reports_everything(tmp_path):
    """The path only a completed experiment reaches, executed at $0."""
    _write_treatment(tmp_path)
    doc = AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED,
                   iterations=ITERATIONS)

    #: Every quantity the design promises to report.
    c = doc["correctness"]
    assert len(c["per_seed"]) == 3 and len(c["mcnemar"]) == 3
    assert c["n_scorable"] == 850
    assert "pooled_delta" in c
    for key in ("lcb_one_sided", "ucb_one_sided", "ci_two_sided_low",
                "ci_two_sided_high", "claim_boundary"):
        assert key in doc["interval"]
    assert doc["interval"]["_descriptive_only"]
    beh = doc["behaviour"]
    assert len(beh["per_seed"]) == 3
    for comp in ("non_empty", "natural_termination", "no_severe_repetition",
                 "no_context_limit", "protocol_valid"):
        assert f"{comp}_delta" in beh["per_seed"][0], comp
    assert set(doc["breakdowns"]) == {"per_capability", "per_domain", "per_set"}
    assert doc["guardrails"]["any_fired"] is False
    assert doc["controls_retrained"] is False
    assert doc["sesoi"] == 0.010
    assert doc["comparison_sha256"]
    #: And it names every byte it read.
    consumed = doc["consumed_inputs_sha256"]
    assert "a3_design.json" in consumed
    assert sum(1 for k in consumed if k.startswith("control_per_sample_")) == 3
    assert sum(1 for k in consumed if k.startswith("treatment_per_sample_")) == 3
    assert sum(1 for k in consumed if k.startswith("control_scored_")) == 3


@needs_controls
def test_an_identical_field_reports_exactly_zero(tmp_path):
    """The control against itself. A comparison that cannot return zero is
    not measuring a difference."""
    _write_treatment(tmp_path, flip=0)
    doc = AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED,
                   iterations=ITERATIONS)
    assert doc["correctness"]["pooled_delta"] == 0.0
    assert all(r["delta"] == 0.0 for r in doc["correctness"]["per_seed"])
    #: McNemar's discordant pairs are the sharper statement: none at all.
    assert all(m["discordant"] == 0 for m in doc["correctness"]["mcnemar"])
    assert doc["interval"]["ci_two_sided_low"] == 0.0
    assert doc["interval"]["ci_two_sided_high"] == 0.0


@needs_controls
def test_a_known_improvement_shows_up_with_the_right_sign_and_size(tmp_path):
    """Flip N prompts to correct in every seed; the delta must be N/850."""
    flip = 17
    _write_treatment(tmp_path, flip=flip)
    doc = AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED,
                   iterations=ITERATIONS)
    expected = flip / 850
    assert doc["correctness"]["pooled_delta"] == pytest.approx(expected, abs=1e-6)
    for row, m in zip(doc["correctness"]["per_seed"],
                      doc["correctness"]["mcnemar"]):
        assert row["delta"] == pytest.approx(expected, abs=1e-6)
        #: Directional: gained by the treatment, none lost.
        assert m["treatment_only_correct"] == flip
        assert m["control_only_correct"] == 0
    #: A one-sided lower bound on a uniformly positive effect is positive.
    assert doc["interval"]["lcb_one_sided"] > 0


# --- and it refuses what it must refuse ----------------------------------


@needs_controls
def test_a_differing_fingerprint_alone_is_NOT_an_integrity_failure(tmp_path):
    """THIS TEST WAS INVERTED, deliberately, and the old version encoded a bug.

    It used to require that a treatment fingerprint differing from the
    controls' refuse the whole comparison. That looked like the C2 lesson —
    three fingerprints in one confirmation set and no verdict — but it asked
    the comparability question a FOURTH time and under the rule the repository
    had already replaced: `generation_protocol_fingerprint` transitively
    contains `runtime_digest`, which fuses the image tag with the host NVIDIA
    driver patch, and `generation_compat.NON_MATERIAL_PROTOCOL_FIELDS` names it
    provenance precisely so the demoted patch cannot be smuggled back into an
    identity. The pod's own admission had already compared both runtimes under
    v2 and admitted the probe; this check then refused a complete three-probe
    measurement over a patch number the provider chose.

    So the fingerprint is RECORDED and no longer compared, and the decision has
    one owner — the admission record, whose refusals are the four tests below.
    """
    doc = AG.build(
        _write_treatment(tmp_path,
                         field={"generation_protocol_fingerprint": "f" * 64}),
        bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)
    assert doc["correctness"]["pooled_delta"] == 0.0
    #: And the premise it DOES rest on is in the artifact, per seed.
    assert len(doc["admitted_against_the_controls"]) == 3


@needs_controls
def test_a_probe_with_no_admission_record_is_refused(tmp_path):
    """Scored outside the gate. "I cannot tell whether it was comparable" is
    not evidence that it was."""
    seeds = sorted(AG.load_controls())
    _write_treatment(tmp_path, drop_admission_for=seeds[1])
    with pytest.raises(AG.A3AggregationError,
                       match="no generation-admission record"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


@needs_controls
def test_an_admission_that_says_not_comparable_is_refused(tmp_path):
    _write_treatment(tmp_path, admission={"comparable": False,
                                          "reason": "runtime_stack differs"})
    with pytest.raises(AG.A3AggregationError, match="comparable=False"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


@needs_controls
def test_an_admission_under_the_SUPERSEDED_v1_rule_is_refused(tmp_path):
    """attempt34 was refused by v1's exact-hash rule and the repository
    replaced it. A record admitted under v1 does not answer the v2 question,
    so accepting it would silently reinstate the rule that was withdrawn."""
    _write_treatment(tmp_path, admission={
        "comparability": {"rule": "generation_runtime_comparability@v1"}})
    with pytest.raises(AG.A3AggregationError, match="expects"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


@needs_controls
@pytest.mark.parametrize("bad, match", [
    ({"identities_equal": False,
      "differing_material_keys": ["attention_backend"]}, "not equal"),
    #: A patch within a branch is provenance; a BRANCH change is a real
    #: runtime event, and it is what refused attempt35 after three trainings.
    ({"driver_branch_equal": False, "live_driver": "595.91.07"}, "branch"),
])
def test_an_admission_whose_own_block_contradicts_it_is_refused(
        tmp_path, bad, match):
    _write_treatment(tmp_path, admission={"comparability": bad})
    with pytest.raises(AG.A3AggregationError, match=match):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


def test_the_artifact_binds_what_computed_it_and_not_only_what_it_read():
    """AGENTS.md P4. The first version bound 20+ input files by `sha256` and
    nothing about the code that read them, so two aggregators disagreeing about
    (say) which balance a bootstrap resamples would produce two artifacts that
    looked equally authoritative. That is not hypothetical: C3's records
    asserted bootstrap seed `654678655` while the resampler used C1's
    `816109261`, and no artifact field could have revealed it.
    """
    impl = AG.implementation_identity()
    assert len(impl["commit"]) == 40
    assert impl["aggregator_path"] == "scripts/autoinit/aggregate_a3.py"
    #: Named modules, not a glob: these two DECIDE the numbers, and a self-hash
    #: alone would not move when the strata or the comparability rule changed.
    for key in ("aggregator_sha256", "probe_results_module_sha256",
                "generation_compat_module_sha256"):
        assert len(impl[key]) == 64, key
    assert len({impl[k] for k in
                ("aggregator_sha256", "probe_results_module_sha256",
                 "generation_compat_module_sha256")}) == 3, (
        "two of the hashed modules are the same file")

    #: And the committed artifact carries it. A field that exists only in a
    #: function's return value binds nothing.
    doc = json.loads((REPO / "logs/stages/stage-1/phase_a3/analyses"
                      / "a3_comparison.json").read_text())
    assert doc["canonical"] is True, (
        "the committed comparison is not canonical; it was built from a dirty "
        "tree and names a commit whose content did not run")
    assert doc["implementation"]["aggregator_sha256"] == impl[
        "aggregator_sha256"], (
        "the committed comparison was produced by a different aggregator than "
        "the one in this tree; re-run it")


def test_a_build_with_no_implementation_says_so_rather_than_omitting_it(
        tmp_path):
    """The absence must be LOUD. A missing key reads as an older schema; a
    `canonical: false` with a reason reads as what it is."""
    doc = AG.build(_write_treatment(tmp_path), bootstrap_seed=BOOTSTRAP_SEED,
                   iterations=50)
    assert doc["canonical"] is False
    assert "_not_recorded" in doc["implementation"]


def test_the_cli_refuses_a_dirty_tree_unless_told_otherwise():
    """Read from the source, because the alternative is to dirty the tree."""
    src = (REPO / "scripts/autoinit/aggregate_a3.py").read_text()
    block = src[src.index("impl = implementation_identity()"):]
    block = block[:block.index("doc = build(")]
    assert 'if impl["tree_is_dirty"] and not args.allow_dirty' in block
    assert "A3AggregationError" in block
    assert "--allow-dirty" in src


def test_the_real_committed_admission_records_satisfy_the_verifier():
    """Not the fixture — the three records `a3_attempt38` actually wrote.

    Checked here because the fixture is built by this file and could agree with
    the verifier while both disagreed with the pod.
    """
    #: NOT SKIPPED IF ABSENT. The first version guarded this with
    #: `pytest.skip` when the directory was missing, which added an
    #: unresolvable skip predicate to the audit: the records are TRACKED, so a
    #: tracked logs/ path travels in the bundle and its absence is a defect in
    #: the checkout, not a property of the machine. A skip would turn a lost
    #: artifact into a silent pass.
    run = REPO / "logs/stages/stage-1/phase_a3/runs/a3_attempt38/evidence"
    assert run.is_dir(), (
        f"{run} is committed evidence and is missing from this checkout")
    recs = sorted(run.glob("*_generation_admission.json"))
    assert len(recs) == 3, f"expected three, found {[p.name for p in recs]}"
    for p in recs:
        doc = json.loads(p.read_text())
        cmp = doc["comparability"]
        assert doc["comparable"] is True
        assert cmp["identities_equal"] is True
        assert cmp["live_identity"] == cmp["historical_identity"]
        assert cmp["driver_branch_equal"] is True
        from aadistill.initialization.planning.generation_compat import (
            GENERATION_RUNTIME_COMPARABILITY_V2 as V2)
        assert cmp["rule"] == V2.as_dict()["qualified_id"]


@needs_controls
def test_a_scoring_contract_mismatch_is_refused(tmp_path):
    _write_treatment(tmp_path, field={"scoring_contract_digest": "a" * 64})
    with pytest.raises(AG.A3AggregationError, match="does not share attempt75"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


@needs_controls
def test_a_battery_mismatch_is_refused(tmp_path):
    _write_treatment(tmp_path, field={"battery_manifest_sha256": "b" * 64})
    with pytest.raises(AG.A3AggregationError, match="does not share attempt75"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


@needs_controls
def test_two_seeds_are_not_the_estimand(tmp_path):
    """A paired difference over three seeds is not computable from two, and
    averaging what exists is exactly what R5 refused for C2's attempt5."""
    controls = sorted(AG.load_controls())
    _write_treatment(tmp_path, seeds=controls[:2])
    with pytest.raises(AG.A3AggregationError, match="missing seeds"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


@needs_controls
def test_a_scored_aggregate_without_its_rows_is_refused(tmp_path):
    """The paired comparison reads the ROWS; an aggregate alone cannot pair."""
    seeds = sorted(AG.load_controls())
    _write_treatment(tmp_path, drop_rows_for=seeds[1])
    with pytest.raises(AG.A3AggregationError, match="per-sample rows"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


@needs_controls
def test_an_empty_evidence_directory_refuses_rather_than_returning_nothing(
        tmp_path):
    (tmp_path / "audit" / "autoinit_a3").mkdir(parents=True)
    with pytest.raises(AG.A3AggregationError, match="no scored A3 records"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)
    with pytest.raises(AG.A3AggregationError, match="no treatment evidence"):
        AG.build(tmp_path / "nowhere", bootstrap_seed=BOOTSTRAP_SEED,
                 iterations=ITERATIONS)


@needs_controls
def test_a_tampered_control_row_file_is_caught_by_its_recorded_hash(tmp_path):
    """The controls are evidence; evidence that changed since it was scored is
    not evidence. Driven by pointing the record at altered bytes."""
    probes = AG.load_controls()
    seed = sorted(probes)[0]
    original = Path(probes[seed]["per_sample_path"])
    altered = tmp_path / "altered.jsonl"
    altered.write_text(original.read_text() + "\n")

    import unittest.mock as mock
    patched = {s: dict(p) for s, p in probes.items()}
    patched[seed]["per_sample_path"] = str(altered)
    with mock.patch.object(AG, "load_controls", return_value=patched):
        with pytest.raises(AG.A3AggregationError, match="changed since"):
            AG._control_rows()


# --- the bootstrap's own properties --------------------------------------


@needs_controls
def test_the_bootstrap_is_deterministic_in_its_seed(tmp_path):
    """A seed that does not reach the resampler is the C3 defect: every record
    asserted 654678655 and the computation used C1's 816109261."""
    _write_treatment(tmp_path, flip=11)
    a = AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)
    b = AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)
    assert a["interval"] == b["interval"]
    c = AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED + 1,
                 iterations=ITERATIONS)
    assert c["interval"]["seed"] != a["interval"]["seed"]
    assert c["interval"] != a["interval"], (
        "the interval did not move with the seed, so the seed is not reaching "
        "the resampler")


@needs_controls
def test_seeds_are_fixed_blocks_and_the_claim_boundary_says_so(tmp_path):
    _write_treatment(tmp_path)
    doc = AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED,
                   iterations=ITERATIONS)
    i = doc["interval"]
    assert "FIXED BLOCKS" in i["resamples"]
    assert "NOT an interval over hypothetical" in i["claim_boundary"]
    assert i["n_strata"] == 6 and i["n_prompts"] == 850
