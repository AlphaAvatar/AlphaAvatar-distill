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

REPO = Path(__file__).resolve().parents[2]
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


def _write_treatment(root: Path, *, flip: int = 0, field: dict | None = None,
                     seeds: list[int] | None = None,
                     drop_rows_for: int | None = None) -> Path:
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
def test_a_generation_fingerprint_mismatch_is_an_integrity_failure(tmp_path):
    """C2's confirmation set carried three fingerprints and produced no
    verdict. A3 raises instead of reporting a pooled number."""
    _write_treatment(tmp_path,
                     field={"generation_protocol_fingerprint": "f" * 64})
    with pytest.raises(AG.A3AggregationError, match="does not share attempt75"):
        AG.build(tmp_path, bootstrap_seed=BOOTSTRAP_SEED, iterations=ITERATIONS)


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
