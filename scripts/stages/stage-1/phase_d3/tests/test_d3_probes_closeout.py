"""The D3 probes closeout, end to end on KNOWN HISTORICAL EVIDENCE.

TEST EVIDENCE, NOT A MEASUREMENT. No D3 probe has run when these tests are
written. The simulation takes the D2 shared confirmation's three c4 units
(`d2_confirmation_20261016_105802`, from its durable store) and files them as
a D3 probes run: the closeout's contract is the real D3 contract with only the
probes and the initialization replaced by c4's, its authorization is a real
`D3ProbesAuthorization` issued against that contract, and the units are
re-scored by the real pinned confirmation scorer. A closeout that reproduces
D2's recorded comparisons on those bytes is a closeout whose verification,
pairing and exploratory analysis are wired end to end.

Each negative removes ONE proof and must yield INCOMPLETE -- money booked,
no scientific result -- or, for a run this closeout may not record at all, a
refusal.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1", "tests"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from stages.phase_d3 import closeout_d3_probes as C  # noqa: E402
from stages.phase_d3 import d3_design as D  # noqa: E402
from stages.phase_d3 import d3_probes as P  # noqa: E402

D2_RUN = "d2_confirmation_20261016_105802"
D2_SCR = Path(f"/home/ecs-user/aad-artifacts/phase_d2/{D2_RUN}")
D2_RUN_DIR = REPO / "logs/stages/stage-1/phase_d2/runs" / D2_RUN
RUN = "d3_probes_simulated_from_d2_c4"

pytestmark = pytest.mark.skipif(
    not (D2_SCR / "store" / "probes").is_dir(),
    reason="the D2 confirmation's durable store is not on this host")


def _sha(path: Path) -> str:
    return C._sha(path)


def _fake_contract() -> dict:
    """The real D3 contract, with the probes and initialization c4's."""
    from stages.phase_d2 import d2_confirmation as CF

    real = P.session_contract(REPO, verify_bytes=False)
    arm = CF.session_contract(REPO, verify_bytes=False)["field"]["arms"][0]
    record = json.loads((D2_SCR / "store" / "d2_confirmation.json").read_text())
    fake = copy.deepcopy(real)
    fake["probes"] = [{"probe_id": p["probe_id"], "arm": p["arm"],
                       "seed": int(p["seed"])} for p in record["probes"]]
    fake["initialization"] = {
        **real["initialization"], "identities": dict(arm["identities"]),
        "state_id": arm["state_id"],
        "_simulation": "TEST EVIDENCE: c4's D2 confirmation units"}
    return fake


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """scr (D2's store under D3's names) + a run directory, outside the tree."""
    mp = pytest.MonkeyPatch()
    fake = _fake_contract()
    mp.setattr(P, "session_contract", lambda *a, **k: copy.deepcopy(fake))
    root = tmp_path_factory.mktemp("d3p_closeout")
    scr, run_dir = root / "scr", root / "run"
    store = scr / "store"
    store.mkdir(parents=True)
    lay = C.names()
    for src, dest in (("probes", "probes"), ("extracted", "extracted"),
                      ("d2_confirmation_attested_protocol.json",
                       lay["attested"]),
                      ("engine_probe.json", "engine_probe.json"),
                      ("d2_confirmation_artifacts.tar.gz",
                       "d3_probes_artifacts.tar.gz")):
        (store / dest).symlink_to(D2_SCR / "store" / src)
    shutil.copy(D2_SCR / "store" / "manifest.json", store / "manifest.json")
    record = json.loads((D2_SCR / "store" / "d2_confirmation.json").read_text())
    record["contract_hash"] = P.contract_hash(fake)
    record["run_id"] = RUN
    (store / lay["record"]).write_text(json.dumps(record))

    from stages.phase_d1.d1_authorization import live_money
    from stages.phase_d3 import d3_authorization as A

    money = dict(live_money(REPO))
    money["funds_formal_sessions_of"] = sorted(
        {*money["funds_formal_sessions_of"], D.EXPERIMENT_ID})
    payload = A.build_payload(
        grant={"rung": "probes",
               "approves_registration_hash": D.registration_hash(REPO),
               "cumulative_envelope_usd": 12.0, "granted_by": "test",
               "covers": "TEST EVIDENCE"},
        session_commit="d" * 40, granted_utc="2026-10-17T00:00:00+00:00",
        run_id=RUN, rung="probes", repo_root=REPO, live_rate=1.09,
        consumed_usd=0.0, money=money)
    (run_dir / "governance").mkdir(parents=True)
    (run_dir / "governance" / "authorization.json").write_text(
        json.dumps(payload))
    (run_dir / "runtime").mkdir()
    shutil.copy(D2_RUN_DIR / "runtime" / "session.json",
                run_dir / "runtime" / "session.json")
    (run_dir / "evidence").mkdir()
    (run_dir / C.RELAY_COPY).write_text(json.dumps({
        "matched": True, "_test_evidence": "no upload was made",
        "sha256": _sha(store / "d3_probes_artifacts.tar.gz")}))
    yield {"scr": scr, "run_dir": run_dir, "root": root, "fake": fake,
           "record": record}
    mp.undo()


def _build(world, *, run=RUN, run_dir=None, rescore=None):
    return C.build(run, world["scr"], balance_after=None, disk_gb=120,
                   repo_root=REPO, rescore=rescore,
                   run_dir=run_dir or world["run_dir"])


def _stored(scorer_rung, generations, *, label, seed, screening_arm,
            repo_root, extra_args):
    """A fast stand-in for the pinned scorer: the stored result itself. Used
    only where the proof under test is something other than the re-score."""
    probe_dir = Path(generations).parent
    result = json.loads((probe_dir / "result.json").read_text())
    return {"rows_sha256": result["per_sample_sha256"], "result": result}


@pytest.fixture(scope="module")
def complete(world):
    """THE END-TO-END RUN: the real pinned scorer on every unit."""
    return _build(world)


class TestTheSimulatedCompleteCloseout:

    def test_every_proof_holds_and_it_is_complete(self, complete):
        doc, _plan, _diag, comparison = complete
        assert doc["missing_proofs"] == [] and doc["outcome"] == C.COMPLETE
        assert comparison is not None
        assert all(p["verified"] for p in doc["per_probe"].values())

    def test_it_reproduces_d2s_recorded_comparisons(self, complete):
        """The pinned scorer, the pairing and the registered analysis,
        through the closeout, give D2's recorded deltas on these bytes."""
        doc, *_ = complete
        verdict = json.loads((D2_RUN_DIR / "evidence" /
                              "d2_confirmation_verdict.json").read_text())
        q2 = verdict["verdict"]["secondary_vs_q2_descriptive"]["decision"]
        b = verdict["verdict"]["primary_vs_B"]["decision"]
        got = doc["exploratory_result"]
        assert got["vs_q2"]["delta"] == pytest.approx(q2["delta"], abs=1e-12)
        assert got["vs_q2"]["per_seed_delta"] == pytest.approx(
            q2["per_seed_delta"], abs=1e-12)
        assert got["vs_B"]["delta"] == pytest.approx(b["delta"], abs=1e-12)
        assert got["vs_B"]["per_seed_delta"] == pytest.approx(
            b["per_seed_delta"], abs=1e-12)

    def test_the_labels_are_the_registered_descriptive_ones(self, complete):
        doc, *_ = complete
        got = doc["exploratory_result"]
        for key in ("primary_label_vs_q2", "anchor_label_vs_B"):
            assert got[key] in ("BEHAVIOURAL_IMPROVEMENT",
                                "BEHAVIOURAL_REGRESSION", "INCONCLUSIVE")
        assert "SHARED_GO" not in json.dumps(got)

    def test_per_domain_behaviour_and_lengths_are_reported(self, complete):
        _doc, _plan, _diag, comparison = complete
        for arm in ("d3", "q2", "B"):
            for seed_profile in comparison["profiles"][arm].values():
                assert set(seed_profile["per_domain"]) >= {
                    "reasoning_math", "code", "tool", "general",
                    "rag_multihop"}
                for block in seed_profile["per_domain"].values():
                    assert {"correct_overall", "usable_rollout_rate",
                            "correct_given_usable", "protocol_valid",
                            "no_severe_repetition"} <= set(block)
        d3 = next(iter(comparison["profiles"]["d3"].values()))
        assert d3["overall"]["lengths"]["generated_tokens_mean"] > 0

    def test_it_books_money_and_the_envelope(self, complete):
        doc, *_ = complete
        assert doc["money"]["all_in_usd"] > 0
        env = doc["envelope"]
        assert env["cumulative_envelope_usd"] == 12.0
        assert env["remaining_usd"] == pytest.approx(
            12.0 - env["consumed_usd"], abs=1e-9)
        assert doc["limits_after"]["already_booked_by_the_deriver"] is False

    def test_the_closeout_names_its_identities(self, complete, world):
        doc, *_ = complete
        assert doc["schema"] == "aadistill.phase_d3.probes_closeout/v1"
        assert doc["contract_hash"] == P.contract_hash(world["fake"])
        assert doc["seeds"] == [396880939, 766527341, 1004399678]
        assert doc["durable_store"]["token_id_sidecars"]
        assert doc["archive"]["verified"] is True


def _variant(world, tmp_path, **edits):
    """A copy of the simulated run directory and scr with ONE edit."""
    scr = tmp_path / "scr"
    shutil.copytree(world["scr"], scr, symlinks=True)
    run_dir = tmp_path / "run"
    shutil.copytree(world["run_dir"], run_dir)
    store = scr / "store"
    record = json.loads((store / "d3_probes.json").read_text())
    if "record" in edits:
        edits["record"](record)
    (store / "d3_probes.json").write_text(json.dumps(record))
    if "session" in edits:
        path = run_dir / "runtime" / "session.json"
        doc = json.loads(path.read_text())
        edits["session"](doc)
        path.write_text(json.dumps(doc))
    if edits.get("no_relay_copy"):
        (run_dir / C.RELAY_COPY).unlink()
    return {**world, "scr": scr, "run_dir": run_dir}


class TestEachMissingProofIsIncomplete:

    @pytest.mark.parametrize("edits,proof", [
        ({"no_relay_copy": True}, "second_durable_copy"),
        ({"record": lambda r: r.update(status="FAILED")}, "driver_complete"),
        ({"record": lambda r: r["probes"][0].update(scored=False)},
         "all_probes_scored"),
        ({"record": lambda r: r.update(contract_hash="0" * 64)},
         "contract_bound"),
        ({"session": lambda s: s.update(provider_confirms_gone=False)},
         "teardown_verified"),
        ({"session": lambda s: s.update(required_products_secured={
            "ok": False})}, "evidence_secured"),
    ])
    def test_one_proof_missing(self, world, tmp_path, edits, proof):
        doc, _plan, _diag, comparison = _build(
            _variant(world, tmp_path, **edits), rescore=_stored)
        assert doc["outcome"] == C.INCOMPLETE and proof in doc["missing_proofs"]
        assert doc["exploratory_result"] is None and comparison is None
        assert doc["money"]["all_in_usd"] > 0

    def test_a_rescore_that_disagrees_is_not_a_result(self, world, tmp_path):
        def wrong(*a, **k):
            out = _stored(*a, **k)
            return {**out, "rows_sha256": "0" * 64}

        doc, *_ = _build(_variant(world, tmp_path), rescore=wrong)
        assert "units_verified_and_paired" in doc["missing_proofs"]
        assert "rescore_reproduces_per_sample" in doc["comparison_refused"]

    def test_an_authorization_for_another_run_is_incomplete(self, world,
                                                             tmp_path):
        doc, *_ = _build(_variant(world, tmp_path),
                         run="d3_probes_another", rescore=_stored)
        assert doc["authorization_checks"]["names_this_run"] is False
        assert "authorization_valid" in doc["missing_proofs"]


class TestRunsItMayNotRecord:

    def test_a_run_outside_the_prefix_is_refused(self, world):
        with pytest.raises(C.CloseoutRefused, match="not a d3_probes_"):
            _build(world, run="d2_confirmation_x")

    def test_an_edited_authorization_is_refused(self, world, tmp_path):
        v = _variant(world, tmp_path)
        path = v["run_dir"] / "governance" / "authorization.json"
        doc = json.loads(path.read_text())
        doc["hard_cap_usd"] = 29.0
        path.write_text(json.dumps(doc))
        with pytest.raises(C.CloseoutRefused, match="does not load"):
            _build(v, rescore=_stored)


class TestTheDeriverBooksD3ProbesRuns:
    """Complete and incomplete closeouts alike carry money.all_in_usd, the
    field the budget deriver books first."""

    def test_both_outcomes_carry_the_booked_field(self, world, tmp_path,
                                                  complete):
        incomplete, *_ = _build(_variant(world, tmp_path, no_relay_copy=True),
                                rescore=_stored)
        for doc in (complete[0], incomplete):
            assert isinstance(doc["money"]["all_in_usd"], float)
            assert "budget" not in doc and "cost" not in doc
