"""Restore a durable probe and resume at scoring, instead of retraining it.

AGENTS.md P8.4's probe-state machine, state 2:

    Trained + durable + not validly scored. Required: checkpoint weights,
    descriptor, identity -- because scoring genuinely consumes the model.
    Restore the existing checkpoint and resume at scoring. Do not retrain it.

`a3_attempt35` left exactly that: three probes trained for 61.5-62.0 minutes
each, preserved at the moment each finished with per-file digests, and never
scored because the protocol admission refused the host's driver branch.
`a3_attempt36` then retrained all three -- `$3.39`, 29% of the formal balance
-- because this path did not exist. It exists now.

What it must not become is a way to pool sessions. The three probes have to
come from ONE preserving attempt, carry ONE initialization digest, be the
three frozen seeds, be complete, and not already be evaluated. Each of those
is a refusal with its own case below.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "pod",
          REPO / "scripts" / "autoinit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stages.phase_a3 import autoinit_a3_driver as D  # noqa: E402

from stages.phase_a3 import a3_session as A3S  # noqa: E402

SEEDS = list(A3S.recovery_seeds())
SOURCE = "a3_attempt35"
RECORDS = (REPO / "logs/stages/stage-1/phase_a3/runs" / SOURCE
           / "evidence" / "probes")

needs_records = pytest.mark.skipif(
    not RECORDS.is_dir(),
    reason=("the preserving attempt's probe records are not in this "
            "checkout, so there is nothing to resume from"))


def _drv(resume_from=SOURCE):
    drv = D.A3Driver.__new__(D.A3Driver)
    drv.a = types.SimpleNamespace(resume_scoring_from=resume_from,
                                  probe_train_minutes=61.76)
    drv.seeds = list(SEEDS)
    drv.training = {}
    drv.ev = {}
    drv.a3_init_digest = None
    drv.a3_init_dir = None
    return drv


# --- the ladder -----------------------------------------------------------

def test_resuming_skips_the_stages_whose_only_consumer_was_training():
    """D replays the parent and E rebuilds both initializations. With the
    probes already built, nothing downstream reads either -- which is the
    whole saving: 21 + 2.5 + 185 minutes."""
    #: From the DECLARED ladder, not from a source scan: the resume sequence
    #: is a constant in the session module precisely so it is reviewable.
    assert A3S.RESUME_STAGE_LETTERS == ("B", "C", "F", "G", "H"), (
        f"the declared resume ladder is {A3S.RESUME_STAGE_LETTERS}")
    full = set(A3S.STAGE_LETTERS)
    resumed = set(A3S.RESUME_STAGE_LETTERS)
    assert full - resumed == {"D", "E"}, (
        "only the parent replay and the initialization rounds may be absent: "
        "their single consumer is the training a resume session does not do")
    #: Scoring and packaging are the POINT, so they may never be absent.
    for keep in ("B", "C", "F", "G", "H"):
        assert keep in resumed, f"stage {keep} must not be skipped when resuming"

    #: And the driver selects that ladder rather than computing its own.
    src = (REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_driver.py").read_text()
    block = src[src.index("if self.a.resume_scoring_from:"):
                src.index("        try:\n            for letter in letters:")]
    assert 'stages["F"] = self.stage_f_resume' in block
    assert "A3S.RESUME_STAGE_LETTERS" in block

    #: The order guard accepts BOTH declared ladders and refuses anything else.
    A3S.assert_stage_order(["B", "C", "F"], A3S.RESUME_STAGE_LETTERS)
    A3S.assert_stage_order(["B", "C", "D"], A3S.STAGE_LETTERS)
    with pytest.raises(Exception, match="not a declared A3 ladder"):
        A3S.assert_stage_order(["B"], ("B", "G"))
    with pytest.raises(Exception, match="out of order"):
        A3S.assert_stage_order(["B", "C", "G"], A3S.RESUME_STAGE_LETTERS)


def test_the_default_is_a_full_chain():
    """A flag that defaults to resuming would silently stop training."""
    args = D.build_parser().parse_args(
        ["--image-digest", "x", "--rate", "1.09",
         "--soft-stop-usd", "1", "--authorized-usd", "2"])
    assert args.resume_scoring_from is None


def test_the_resume_path_records_what_it_did_not_do():
    """A session that cites another's diagnostics must not read as having
    produced them."""
    src = (REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_driver.py").read_text()
    fn = src[src.index("def stage_f_resume"):src.index("def stage_f(self)")]
    assert '"training_started"] = False' in fn
    assert "_what_this_session_did_not_do" in fn
    assert "did not replay the parent" in fn


# --- the five refusals ----------------------------------------------------

@needs_records
def test_it_loads_the_real_preserved_set():
    drv = _drv()
    records = D.A3Driver.preserved_records(drv)
    assert [r["seed"] for r in records] == SEEDS, (
        "the records must come back in the frozen seed order, because the "
        "estimand is a paired difference over those seeds")
    assert drv.a3_init_digest, "no initialization digest was resolved"
    assert len({r["initialization_artifact_digest"] for r in records}) == 1


def test_an_unknown_attempt_is_refused():
    with pytest.raises(D.A3DriverError, match="names no probe records"):
        D.A3Driver.preserved_records(_drv("a3_attempt_does_not_exist"))


@needs_records
@pytest.mark.parametrize("mutate,expected", [
    ({"complete": False}, "INCOMPLETE training"),
    ({"evaluated": True}, "already recorded as evaluated"),
    ({"arm": "A_incumbent"}, "not 'A_bsz3'"),
    ({"initialization_artifact_digest": "f" * 64}, "different initialization"),
])
def test_each_refusal_fires(tmp_path, monkeypatch, mutate, expected):
    """Mutate ONE field of ONE preserved record and require the refusal.

    Against a copy of the real records, so the shapes are the production
    shapes rather than a fixture's idea of them.
    """
    dest = tmp_path / "logs/stages/stage-1/phase_a3/runs/mutated/evidence/probes"
    dest.mkdir(parents=True)
    for i, src in enumerate(sorted(RECORDS.glob("*.training.json"))):
        doc = json.loads(src.read_text())
        if i == 0:
            doc.update(mutate)
        (dest / src.name).write_text(json.dumps(doc) + "\n")
    monkeypatch.setattr(
        D, "RUNS_ROOT",
        tmp_path / "logs/stages/stage-1/phase_a3/runs")
    with pytest.raises(D.A3DriverError, match=expected):
        D.A3Driver.preserved_records(_drv("mutated"))


@needs_records
def test_a_missing_seed_is_refused(tmp_path, monkeypatch):
    """Two probes are not the estimand."""
    dest = tmp_path / "logs/stages/stage-1/phase_a3/runs/short/evidence/probes"
    dest.mkdir(parents=True)
    for src in sorted(RECORDS.glob("*.training.json"))[:2]:
        (dest / src.name).write_text(src.read_text())
    monkeypatch.setattr(
        D, "RUNS_ROOT",
        tmp_path / "logs/stages/stage-1/phase_a3/runs")
    with pytest.raises(D.A3DriverError, match="preserved no probe for seed"):
        D.A3Driver.preserved_records(_drv("short"))


# --- the restore itself ---------------------------------------------------

def test_a_restore_that_cannot_be_re_identified_is_refused():
    """A restore with no per-file digests is not a restore."""
    drv = _drv()
    with pytest.raises(D.A3DriverError, match="nothing to restore"):
        D.A3Driver.restore_one(drv, {"probe_id": "p", "preserved": {}})
    with pytest.raises(D.A3DriverError, match="no per-file digests"):
        D.A3Driver.restore_one(
            drv, {"probe_id": "p",
                  "preserved": {"preserved": True, "files": {}}})


def test_a_digest_mismatch_is_an_integrity_stop(tmp_path, monkeypatch):
    """The one thing worse than retraining is scoring something that is not
    the probe the record describes."""
    blob = tmp_path / "model.safetensors"
    blob.write_bytes(b"not the probe")
    monkeypatch.setattr(D, "TRAIN", tmp_path / "train")
    monkeypatch.setattr(D, "TOKEN_FILE", tmp_path / "absent-token")
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: str(blob))
    drv = _drv()
    with pytest.raises(D.A3DriverError, match="integrity stop"):
        D.A3Driver.restore_one(drv, {
            "probe_id": "autoinit.v1.phase_a3.A_bsz3.217230555",
            "preserved": {"preserved": True,
                          "relay_prefix": "a3_preserved_probes/x/y",
                          "files": {"model.safetensors": "a" * 64}}})


@needs_records
def test_the_cited_evidence_the_resume_run_carries_exists():
    """A resume package must be self-contained AND honestly attributed.

    The success spec requires `a3_diagnostics.json`, `a3_arm_identities.json`
    and `a3_replay.json`, every one written by the stages a resume session
    skips. They are copied from the preserving attempt with a sidecar that
    says so; absent, the manifest gate reports them MISSING at teardown,
    which is where C3 learned that a required artifact nobody writes is a
    session that completes and comes home incomplete.
    """
    import json as _json

    spec = _json.loads(
        (REPO / "configs/autoinit/a3_artifacts.json").read_text())
    required = {e["pattern"].rsplit("/", 1)[-1] for e in spec["entries"]
                if e.get("required") and "*" not in e["pattern"]
                and "/audit/" in f"/{e['pattern']}"}
    produced_by_skipped_stages = {"a3_diagnostics.json",
                                  "a3_arm_identities.json", "a3_replay.json"}
    assert produced_by_skipped_stages <= required, (
        "the spec no longer requires what stages D and E write; this check "
        "exists because a resume run does not run them")

    src = (REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_driver.py").read_text()
    fn = src[src.index("def stage_f_resume"):src.index("def stage_f(self)")]
    for name in produced_by_skipped_stages:
        assert name in fn, f"the resume path does not carry {name}"
    assert "a3_cited_evidence.json" in fn, "nothing records the attribution"

    #: And the preserving attempt really has them, so the copy is not a
    #: promise about a file that does not exist.
    have = {p.name for p in RECORDS.parent.glob("*.json")}
    missing = sorted(produced_by_skipped_stages - have)
    assert not missing, (
        f"{SOURCE} carries no {missing}; a resume run from it would come "
        "home without evidence its own spec requires")


A75 = REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i"
GEN35 = (REPO / "logs/stages/stage-1/phase_a3/runs/a3_attempt35"
         / "evidence" / "gen_summaries")

needs_dry_run = pytest.mark.skipif(
    not (GEN35.is_dir() and (A75 / "c3_engine_probe.json").is_file()),
    reason=("needs a real generation summary set and attempt75's engine "
            "probe; both are committed evidence, absent only in a partial "
            "checkout"))


@needs_dry_run
def test_the_admission_is_satisfiable_on_the_host_class_admission_guarantees():
    """The `$0` proof that stage G can pass, before a pod is paid for.

    Two wrong historical sources cost two paid sessions. The first compared
    against this session's own attestation under the v1 exact rule and
    refused on `runtime_digest`. The second compared against attempt75's
    ATTESTATION runtime block -- which is the TRAIN environment, `cuda_runtime
    12.8`, torch 2.11+cu128 -- while generation runs under the vLLM venv at
    `cuda_runtime 13.0`, torch 2.13+cu130, and refused on `runtime_stack`
    with all five material fields "differing", three of them only because the
    older block does not record them.

    The historical side is attempt75's ENGINE PROBE: same kind of
    environment as the live one. With it, a real generation summary set and a
    580-branch host, the admission ADMITS.
    """
    import glob
    import json as _json

    from aadistill.initialization.planning.generation import (
        RecoveryEvaluationProtocol, observe_generation_protocol,
    )
    from aadistill.initialization.planning.generation_compat import (
        comparable_generation_identity, require_comparable,
    )
    from stages.phase_c1.scoring import c1_scoring_contract

    summaries = [_json.loads(Path(f).read_text())
                 for f in sorted(glob.glob(str(GEN35 / "*.json")))]
    assert summaries, "no generation summaries to observe a protocol from"
    manifest = _json.loads(
        (REPO / "artifacts/stages/stage-1/phase_c1/batteries/c1_confirmation_v1/manifest.json").read_text())
    contract = c1_scoring_contract(REPO)
    observed = RecoveryEvaluationProtocol(
        generation=observe_generation_protocol(summaries).protocol,
        scoring_contract=contract["contract"],
        scoring_digest=contract["digest"],
        battery_artifact=manifest["artifact"],
        battery_manifest_sha256=manifest["manifest_sha256"],
        battery_content_sha256=manifest["content_sha256"])

    att = _json.loads((A75 / "c3_attested_evaluation_protocol.json").read_text())
    probe75 = _json.loads((A75 / "c3_engine_probe.json").read_text())
    hist = comparable_generation_identity(
        protocol=att["evaluation_protocol"],
        runtime=probe75.get("runtime") or probe75)

    #: A 580-branch host, which is what `host_admission` admits.
    live_runtime = dict(probe75.get("runtime") or probe75)
    live = comparable_generation_identity(
        protocol=observed.as_dict(), runtime=live_runtime,
        host_provenance={"image_digest_arg": "x"})
    cmp = require_comparable(live, hist, context="dry run")
    assert cmp["identities_equal"] is True
    assert cmp["driver_branch_equal"] is True

    #: And the driver reads the engine probe, not the attestation's runtime.
    src = (REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_driver.py").read_text()
    fn = src[src.index("def admit_generation"):src.index("def stage_h")]
    assert "CONTROL_ENGINE_PROBE" in fn, (
        "the historical runtime no longer comes from attempt75's engine "
        "probe; the attestation's block is the TRAIN environment")
    assert 'runtime=control_att["runtime"]' not in fn, (
        "the driver is back to comparing against the train runtime")


@needs_dry_run
def test_the_controls_two_runtime_blocks_really_do_differ():
    """The premise of the fix above, asserted rather than asserted-about.

    If attempt75's attestation runtime and its engine probe ever agreed, the
    check above would pass for the wrong reason and the comment explaining it
    would be wrong.
    """
    import json as _json

    att = _json.loads(
        (A75 / "c3_attested_evaluation_protocol.json").read_text())["runtime"]
    probe = _json.loads((A75 / "c3_engine_probe.json").read_text())
    probe = probe.get("runtime") or probe
    assert att.get("cuda_runtime") != probe.get("cuda_runtime"), (
        "the two runtime blocks agree on cuda_runtime; the train/vLLM "
        "distinction this fix rests on would no longer exist")
    assert att.get("cuda_runtime") == "12.8" and probe.get("cuda_runtime") == "13.0", (
        f"train {att.get('cuda_runtime')} / vLLM {probe.get('cuda_runtime')}")
