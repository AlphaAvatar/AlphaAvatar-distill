"""A corpus candidate is its own identity, and its feasibility is the formal ladder's verdict."""
from __future__ import annotations

import json
import random

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_arm_b as AB
from stages.reasoning_data import corpus_feasibility as CF
from stages.reasoning_data import source_plan as SP

from aadistill.data.nested_ladder import LadderError, LadderUnit, build_ladder

V2 = A.CONFIG_DIR / "corpus_v2_candidate.json"
V3 = A.CONFIG_DIR / "corpus_v3.json"
TARGETS = [8_600, 16_000, 29_600]


def test_v2_is_the_maintainers_split_under_its_own_id_and_never_the_chosen_one():
    spec = SP.candidate_spec(V2)
    v1 = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    assert spec["corpus_id"] == "stage2.reasoning_master@v2" != v1["corpus_id"]
    assert spec["chosen_variant"]["id"] == "stage2.reasoning_master@v1"
    assert spec["salt"] != v1["salt"]
    cells = AB.cell_shares(spec, spec["corpus_id"])
    assert cells == pytest.approx({
        "primary.task_contained_deduction/s0_generators_v2_calsplit": 0.095,
        "primary.task_contained_deduction/prontoqa": 0.03,
        "primary.task_contained_deduction/reasoning_core": 0.03,
        "primary.task_contained_deduction/reasoning_gym": 0.025,
        "primary.inductive_rule_discovery/s0_generators_v2_calsplit": 0.17,
        "primary.compositional_generalization/s0_generators_v2_calsplit": 0.12,
        "primary.compositional_generalization/reasoning_core": 0.03,
        "primary.compositional_generalization/reasoning_gym": 0.02,
        "primary.abstract_analogical/s0_generators_v2_calsplit": 0.078,
        "primary.abstract_analogical/reasoning_core": 0.002,
        "primary.interactive_adaptation/s0_generators_v2_calsplit": 0.10,
        "supporting.math_reused_gsm8k/corpus_v2_gsm8k_verified": 0.06,
        "supporting.math_new/math_train": 0.03,
        "supporting.math_new/mathematics_dataset": 0.02,
        "supporting.math_new/reasoning_gym": 0.01,
        "supporting.code_algorithmic/mbpp_train": 0.055,
        "supporting.code_algorithmic/reasoning_core": 0.015,
        "supporting.reading_verified/corpus_v2_rag_verified": 0.06,
        "supporting.reading_verified/corpus_v2_multihop_verified": 0.02,
        "supporting.answerability/squad_v2_train_unanswerable": 0.03})
    reasoning = sum(v for k, v in spec["shares"].items() if k.startswith("primary."))
    assert reasoning == pytest.approx(0.70) and sum(spec["shares"].values()) == pytest.approx(1.0)
    # everything the candidate does not declare is @v1's
    for k in ("block_len", "rung_targets", "reserve_tokens", "acceptance", "size_measure"):
        assert spec[k] == v1[k]
    # @v1 itself is untouched: its own variant still resolves to its own split
    assert AB.cell_shares(v1, "stage2.reasoning_master@v1")["primary.abstract_analogical/reasoning_core"] == 0.03


def test_plan_2_keeps_v1s_capability_shares():
    spec, v1 = SP.candidate_spec(V3), A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    assert spec["shares"] == v1["shares"] and spec["corpus_id"] not in (v1["corpus_id"], "stage2.reasoning_master@v2")


def test_neither_candidate_can_be_built_through_the_arm_b_path():
    for path in (V2, V3):
        with pytest.raises(SystemExit):
            AB.main(["--variant", A.read_json(path)["corpus_id"], "--out", "/nonexistent/never-written"])


def _candidate(tmp_path, **edit):
    doc = A.read_json(V2)
    for k, v in edit.items():
        doc[k] = v
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc))
    return p


def test_a_candidate_whose_base_changed_or_that_reuses_an_id_or_breaks_a_sum_refuses(tmp_path):
    base = dict(A.read_json(V2)["derived_from"])
    with pytest.raises(SystemExit, match="not the bytes"):
        SP.candidate_spec(_candidate(tmp_path, derived_from={**base, "spec_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="id of its own"):
        SP.candidate_spec(_candidate(tmp_path, corpus_id="stage2.reasoning_master@v1"))
    shares = {**A.read_json(V2)["shares"], "primary.abstract_analogical": 0.09}
    with pytest.raises(SystemExit):
        SP.candidate_spec(_candidate(tmp_path, shares=shares))
    alloc = json.loads(json.dumps(A.read_json(V2)["source_allocation"]))
    alloc["primary.abstract_analogical"]["reasoning_core"] = 0.003
    with pytest.raises(SystemExit, match="sum"):
        SP.candidate_spec(_candidate(tmp_path, source_allocation=alloc))


def test_the_ladder_salt_comes_from_the_spec_and_v1s_is_unchanged(monkeypatch):
    v1 = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    assert v1["salt"] == AB.SALT
    units = [{"id": "u#t0", "unit_id": "u", "unit_sessions": 1, "stratum": "s", "attributes": {"source": "x"},
              "messages": []}]
    seen = {}
    monkeypatch.setattr(AB.B, "build", lambda sessions, spec, tok: seen.setdefault("spec", spec))
    AB.build(units, {**v1, "salt": "other/v9"}, {"s/x": 1.0}, [10], None)
    assert seen["spec"]["salt"] == "other/v9" and seen["spec"]["allow_composition_drift"] is False
    AB.build(units, v1, {"s/x": 1.0}, [10], None)
    assert seen["spec"]["salt"] == "other/v9"  # setdefault kept the first; v1's own salt is checked above


def _corpus(n_big=40, one_unit_cell=True, seed=0, lo=80):
    rng = random.Random(seed)
    units = [LadderUnit(f"a{i}", "A", rng.randint(lo, 120), family=f"fa{i % 4}") for i in range(n_big * 3)]
    units += [LadderUnit(f"b{i}", "B", rng.randint(lo, 120), family=f"fb{i % 3}") for i in range(n_big)]
    if one_unit_cell:
        units.append(LadderUnit("c0", "C", 100, family="fc"))
    shares = {"A": 0.6, "B": 0.38, "C": 0.02} if one_unit_cell else {"A": 0.6, "B": 0.4}
    return units, shares


def test_a_one_unit_cell_blocks_every_rung_and_is_absent_from_the_validation_tail():
    units, shares = _corpus(n_big=100)
    con, _ = CF.contract(units, shares, salt="t", reserve=1_000, targets=TARGETS)
    c = con["cells"]["C"]
    assert c["runs_dry_at_prefix"] == con["composition_capacity"] < TARGETS[0]
    assert c["blocks_rungs"] == TARGETS and c["reserve_units"] == 0
    assert "C" in con["validation_tail"]["cells_absent"]
    with pytest.raises(LadderError, match="composition capacity"):
        build_ladder(units, shares, TARGETS, salt="t", reserve=1_000)
    found = CF.unsatisfied(con)
    assert any(x.startswith("C: 1 unit(s)") for x in found) and any("ONE unit" in x for x in found)


@pytest.mark.parametrize("seed", range(40))
def test_the_contract_predicts_exactly_what_the_ladder_refuses(seed):
    """Unit sizes from 1 token up, so a cell can run dry a few tokens below a rung."""
    rng = random.Random(seed)
    units, shares = _corpus(n_big=rng.randint(60, 110), one_unit_cell=rng.random() < 0.3, seed=seed, lo=1)
    con, _ = CF.contract(units, shares, salt=f"s{seed}", reserve=1_000, targets=TARGETS)
    blocked = any(v["blocks_rungs"] for v in con["cells"].values())
    reachable = all(v is not None for v in con["rung_prefix_sizes"].values())
    assert CF.builds(units, shares, salt=f"s{seed}", reserve=1_000, targets=TARGETS) == (reachable and not blocked)
    # a rung one token past the capacity: its prefix ends a unit or more beyond, so it is refused
    edge = [con["composition_capacity"] + 1]
    at_edge, _ = CF.contract(units, shares, salt=f"s{seed}", reserve=1_000, targets=edge)
    assert any(v["blocks_rungs"] == edge for v in at_edge["cells"].values())
    assert not CF.builds(units, shares, salt=f"s{seed}", reserve=1_000, targets=edge)


def test_the_minimum_is_minimal_and_the_planning_count_is_at_least_it():
    units, _ = _corpus(n_big=100, one_unit_cell=False)
    shares = {"A": 0.6, "B": 0.4}  # B holds ~10k of the ~12.2k its share of the top rung + reserve needs
    k = CF.minimum_new_units(units, shares, ["B"], salt="m", reserve=1_000, targets=TARGETS, order="ascending")
    assert k["B"] > 0
    ok = lambda n: CF.builds(CF.with_stand_ins(units, {"B": n}, order="ascending"), shares, salt="m",
                             reserve=1_000, targets=TARGETS)
    assert ok(k["B"]) and not ok(k["B"] - 1)
    con, _ = CF.contract(CF.with_stand_ins(units, k, order="ascending"), shares, salt="m", reserve=1_000,
                         targets=TARGETS)
    plan = CF.planning_units(units, shares, k, prefix_top=con["rung_prefix_sizes"][str(TARGETS[-1])],
                             reserve=1_000)
    assert plan["B"] >= k["B"]


def test_tasks_cover_both_the_oversupply_and_the_binomial_chance():
    t = CF.tasks_for(3, 0.15)
    assert t["by_oversupply"] == 30 and t["tasks"] == max(30, t["by_binomial_p90"])
    assert t["chance_of_enough_units"] >= 0.9
    assert CF.tasks_for(11, 0.6721)["tasks"] == 25


def test_independence_finds_a_shared_prompt_and_a_validation_prompt_in_training():
    s = lambda uid, q, t: {"unit_id": uid, "messages": [{"role": "user", "content": q}], "target": t}
    rows = [s("u1", "same", {"r": 1}), s("u2", "same", {"r": 2}), s("u3", "other", {"r": 1})]
    got = CF.independence(rows, reserved={"u2"})
    assert got["units_sharing_a_prompt"] == 2 and got["units_sharing_a_target"] == 2
    assert got["validation_prompts_also_in_training"] == 1
    clean = CF.independence([s("u1", "a", {"r": 1}), s("u2", "b", {"r": 2})], reserved={"u2"})
    assert clean["units_sharing_a_prompt"] == clean["validation_prompts_also_in_training"] == 0


def test_attempt_never_allows_drift_and_reports_the_refusal(monkeypatch):
    seen = {}

    def fake(sessions, spec, shares, targets, tok, **kw):
        seen.update(kw)
        raise LadderError("rung 1: refused")
    monkeypatch.setattr(AB, "build", fake)
    got = CF.attempt([], {"rung_targets": [1]}, {}, None)
    assert got == {"built": False, "refusal": "rung 1: refused"} and not seen.get("drift")


# --- two generation sessions, one corpus ---------------------------------------

def _rec(tid, lid="a" * 64, proto="p" * 64):
    return {"task_id": tid, "generation_pass": {"launch_identity_sha256": lid, "protocol_sha256": proto,
                                                "runner_sha256": "r" * 64}}


def _run(run_id, phase, tids, **kw):
    sess = [{"id": f"{t}#t0", "unit_id": t, "stratum": "primary.abstract_analogical",
             "attributes": {"source": "s0_generators_v2_calsplit"}, "messages": [],
             "target": {"record_sha256": f"sha-{t}"}} for t in tids]
    ident = {"phase": phase, "run_id": run_id, "authorization_id": f"auth.{run_id}",
             "launch_identity_sha256": kw.get("lid", run_id * 4)}
    return {"identity": ident, "records": [_rec(t, **kw) for t in tids], "ins": {"records": len(tids), "shards": 1},
            "got": {"sessions": sess, "selected": {t: {} for t in tids}}}


def test_two_sessions_merge_and_every_unit_keeps_the_session_that_generated_it():
    got, prov = CF.merge([_run("p1", "P1", ["a", "b"]), _run("p1t", "P1t", ["c"])], tag=True)
    assert {s["unit_id"]: s["target"]["generation"]["run_id"] for s in got} == {"a": "p1", "b": "p1", "c": "p1t"}
    assert got[0]["target"]["record_sha256"] == "sha-a"            # the record identity is kept beside it
    assert [s["phase"] for s in prov["sessions"]] == ["P1", "P1t"] and prov["tasks_in_two_sessions"] == 0
    untagged, _ = CF.merge([_run("p1", "P1", ["a"])], tag=False)    # @v2's one-run shape is left as it was
    assert "generation" not in untagged[0]["target"]


@pytest.mark.parametrize("second,match", [
    (lambda: _run("p1t", "P1t", ["b"]), "tasks generated in two sessions"),
    (lambda: _run("p1t", "P1t", ["c"], proto="q" * 64), "protocol_sha256"),
])
def test_sessions_that_share_a_task_or_differ_in_protocol_do_not_merge(second, match):
    with pytest.raises(SystemExit, match=match):
        CF.merge([_run("p1", "P1", ["a", "b"]), second()], tag=True)


def test_two_runners_refuse_unless_that_exact_pair_was_shown_equivalent():
    """H1-PA's PA1 ran under run_battery's bytes after the opt-in --decoding; only
    the proven pair (audit/h1pa_runner_equivalence.json) folds, never another."""
    other = _run("pa1", "PA1", ["c"])
    for r in other["records"]:
        r["generation_pass"]["runner_sha256"] = "n" * 64
    with pytest.raises(SystemExit, match="runner_sha256"):
        CF.merge([_run("p1", "P1", ["a"]), other], tag=True)
    with pytest.raises(SystemExit, match="runner_sha256"):
        CF.merge([_run("p1", "P1", ["a"]), other], tag=True, equivalent_runners={"x" * 64: "r" * 64})
    got, _ = CF.merge([_run("p1", "P1", ["a"]), other], tag=True, equivalent_runners={"n" * 64: "r" * 64})
    assert {s["unit_id"] for s in got} == {"a", "c"}


def test_a_deallocated_cell_is_set_aside_by_name_and_an_undeclared_one_refuses():
    spec = SP.candidate_spec(V3)
    rc = {"id": "u#t0", "unit_id": "u", "stratum": "primary.abstract_analogical",
          "attributes": {"source": "reasoning_core"}, "messages": [], "target": {"record_sha256": "x"}}
    kept = {**rc, "id": "k#t0", "unit_id": "k", "attributes": {"source": "s0_generators_v2_calsplit"}}
    keep, aside = CF.set_aside([rc, kept], spec)
    assert keep == [kept] and aside == [{"unit_id": "u", "cell": "primary.abstract_analogical/reasoning_core",
                                         "target": {"record_sha256": "x"}}]
    nowhere = {**kept, "attributes": {"source": "not_declared"}}
    assert CF.set_aside([nowhere], spec)[0] == [nowhere]          # not 0.0: left for place() to refuse
    with pytest.raises(SystemExit, match="does not allocate"):
        AB.place([nowhere], AB.cell_shares(spec, spec["corpus_id"]))
    # @v2 keeps analogy's Reasoning Core cell: nothing of it is set aside there
    assert CF.set_aside([rc], SP.candidate_spec(V2))[1] == []


# --- a session is admitted only as the complete run its own records say ---------

G0, IDENT = "generation_input_g0", {"authorization_id": "auth.r", "launch_identity_sha256": "l" * 64,
                                    "harness_digest": "h" * 64}


def _complete_run():
    """Four tasks, two of whose first candidates the run's driver sent to a second."""
    recs = [{"task_id": t, "candidate_index": k, "generation_pass": {"group_input": f"{G0}.jsonl"}}
            for k, ts in ((0, "abcd"), (1, "bd")) for t in ts]
    return {"planned": {G0: set("abcd")}, "records": recs,
            "ins": {"shards": 2, "records": 6, "groups": {f"pass0_{G0}": {"tasks": 4, "complete_and_matching": True},
                                                          f"pass1_{G0}": {"tasks": 2, "complete_and_matching": True}}},
            "driver": {"terminal": "ALL_DONE", "contract": {"status": "ok", "hard_cap_usd": 2.6, **IDENT},
                       "groups": [{"label": f"pass0_{G0}", "tasks": 4, "sealed": 4, "complete": True,
                                   "second_candidate_tasks": 2},
                                  {"label": f"pass1_{G0}", "tasks": 2, "sealed": 2, "complete": True}]},
            "closeout": {"status": "COMPLETE", "complete": True, "terminal": "ALL_DONE", "session_passed": True,
                         "shards": {"secured": 2, "records": 6, "unverified": []}},
            "identity": dict(IDENT)}


def _without_pass1(kw, *, keep_driver_entry=False):
    """The first pass complete and the second, due, absent -- store and closeout agreeing with each other."""
    kw["records"] = [r for r in kw["records"] if r["candidate_index"] == 0]
    del kw["ins"]["groups"][f"pass1_{G0}"]
    kw["ins"].update(shards=1, records=4)
    kw["closeout"]["shards"].update(secured=1, records=4)
    if keep_driver_entry:
        kw["driver"]["groups"][1].update(sealed=0, complete=False)
    else:
        kw["driver"]["groups"].pop()


def test_a_complete_run_is_admitted():
    assert CF.completeness(**_complete_run()) == []


@pytest.mark.parametrize("mutate,match", [
    # the case the old check let through: every first candidate sealed, the due second pass never ran
    (_without_pass1, f"pass1_{G0}: 2 tasks due, the driver's entry is None"),
    (lambda kw: _without_pass1(kw, keep_driver_entry=True), f"pass1_{G0}: 2 tasks due, its .complete is None"),
    (lambda kw: kw["records"].pop(), f"pass1_{G0}: 2 tasks due, 1 distinct sealed"),
    (lambda kw: kw["driver"].update(terminal="RUNS_INCOMPLETE"), "terminal is 'RUNS_INCOMPLETE'"),
    (lambda kw: kw.update(driver=None), "no driver record"),
    (lambda kw: kw.update(closeout=None), "no closeout"),
    (lambda kw: kw["closeout"].update(status="INCOMPLETE", complete=False), "the closeout is"),
    (lambda kw: kw["ins"].update(records=7), "not the one the closeout closed"),
    (lambda kw: kw["driver"]["contract"].update(harness_digest="x" * 64), "is not the authorized"),
    (lambda kw: kw["driver"]["groups"][0].pop("second_candidate_tasks"), "no second-candidate count"),
    (lambda kw: kw["driver"]["groups"][0].update(second_candidate_tasks=0), "work outside the due groups"),
    (lambda kw: kw["planned"][G0].add("e"), f"pass0_{G0}: 5 tasks due"),
])
def test_a_run_that_is_not_complete_is_refused(mutate, match):
    kw = _complete_run()
    mutate(kw)
    problems = CF.completeness(**kw)
    assert any(match in p for p in problems), problems


def test_p1_is_admitted_on_its_frozen_records_and_execution_identity():
    """P1's second pass is the 577 its own driver, under harness d6affe40, sent:
    the admission reads it from the run, not from today's repaired adapter."""
    from stages.reasoning_data import governance as GOV
    from stages.reasoning_data import p1v
    run = "h1gen_20261022_190653"
    store = A.REPO / "artifacts/stages/stage-2/reasoning_data/generation" / run
    if not store.is_dir():
        pytest.skip("P1's secured records are not on this machine")
    auth = GOV.load(A.REPO / GOV.authorization_path(run))
    plan = A.read_json(A.REPO / GOV.plan_of(auth))
    recs, ins = p1v.inputs(store, auth.plan_hash)
    driver = A.read_json(store / "generation_session.json")
    assert CF.completeness(
        planned={f["file"].removesuffix(".jsonl"): {r["task_id"] for r in A.jsonl(A.REPO / GOV.inputs_dir(plan) / f["file"])}
                 for f in plan["inputs"]},
        records=recs, ins=ins, driver=driver, closeout=A.read_json(A.REPO / GOV.run_dir(run) / "closeout/outcome.json"),
        identity={"authorization_id": auth.authorization_id, "launch_identity_sha256": auth.plan_hash,
                  "harness_digest": auth.harness_source_digest}) == []
    assert driver["contract"]["harness_digest"] == auth.harness_source_digest and \
        auth.harness_source_digest.startswith("d6affe40")                 # P1's, not today's harness
    assert driver["groups"][0]["second_candidate_tasks"] == 577 == sum(r["candidate_index"] == 1 for r in recs)


def test_the_corpus_declares_both_sessions_and_a_pack_needs_every_one(tmp_path):
    v3 = CF.declared_sessions(A.read_json(V3))
    assert [(d["phase"], d["run_id"]) for d in v3] == [("P1", "h1gen_20261022_190653"),
                                                       ("P1t", "h1p1t_20261023_094007")]
    assert CF.declared_sessions(A.read_json(V2)) == [{"phase": "P1", "run_id": "h1gen_20261022_190653",
                                                      "launch_identity_sha256": v3[0]["launch_identity_sha256"]}]
    with pytest.raises(SystemExit, match="no secured store given"):
        CF.verify_sessions(v3, {"h1p1t_20261023_094007": tmp_path})
    with pytest.raises(SystemExit, match="no authorization"):          # a declared session never granted
        CF.verify_sessions([{**v3[1], "run_id": "h1p1t_never_granted"}], {"h1p1t_never_granted": tmp_path})
    with pytest.raises(SystemExit, match="not the complete run"):     # P1t ran: an empty store is not it
        CF.verify_sessions(v3[1:], {"h1p1t_20261023_094007": tmp_path})
    with pytest.raises(SystemExit, match="not the complete run: .*not the one the closeout closed"):  # empty
        CF.main(["--candidate", str(V3), "--store", f"h1gen_20261022_190653={tmp_path}",
                 "--store", f"h1p1t_20261023_094007={tmp_path}", "--pack-out", str(tmp_path / "never")])
    assert not (tmp_path / "never").exists()
