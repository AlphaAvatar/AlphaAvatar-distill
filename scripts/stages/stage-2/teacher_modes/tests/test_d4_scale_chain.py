"""Research A's D4-scale chain at $0: the frozen order, the teacher session's waves and bounds, resume, the dev box's
exact admission, the packs the recovery pod rebuilds, the recovery plan, the decision rule.

The teacher is replaced by a runner that writes token-native records and the verifier by a seeded verdict: what is
under test is the orchestration and the registered rules, not the teacher or H1's verifiers (their own tests)."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-2"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import generation_adapter as GA  # noqa: E402
from stages.teacher_modes import admission as ADM  # noqa: E402
from stages.teacher_modes import teacher_session as GEN  # noqa: E402

PROTO = json.loads((REPO / GEN.PROTOCOL).read_text())
ORDER = REPO / PROTO["inputs"]["order"]["path"]
IDENT = "f" * 64


def test_the_order_is_frozen_isolated_and_holds_t0s_proportions():
    from stages.reasoning_data import assets as A
    assert hashlib.sha256(ORDER.read_bytes()).hexdigest() == PROTO["inputs"]["order"]["sha256"]
    man = json.loads(ORDER.with_name("order_v1.manifest.json").read_text())
    iso = json.loads(ORDER.with_name("order_v1.isolation.json").read_text())
    assert man["sha256"] == PROTO["inputs"]["order"]["sha256"] and iso["pass"] is True
    assert {s: v["positions"] for s, v in man["slices"].items()} == {"V": [0, 240], "E": [240, 3762], "X": [3762, 17762]}
    assert all(abs(v) < 0.5 for s in ("E", "X") for v in man["stratum_share_shift_points_vs_t0"][s].values())
    t0 = {json.loads(l)["task_id"] for lst in PROTO["inputs"]["t0"] for l in (REPO / lst["path"]).read_text().splitlines()
          if l.strip()}
    rows = list(A.jsonl(ORDER))
    assert len({r["task_id"] for r in rows}) == len(rows) == 17762 and not t0 & {r["task_id"] for r in rows}
    assert len({r["content_sha256"] for r in rows}) == len(rows)


# ---- a tiny phase: real tasks, a fake teacher ----------------------------------------------------------------------

def _tiny_repo(tmp: Path) -> Path:
    from stages.reasoning_data import assets as A
    root = tmp / "repo"
    for rel in ("data/stages/stage-2/reasoning_data/tasks",):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).symlink_to(REPO / rel)
    t0_rows = [json.loads(l) for l in (REPO / PROTO["inputs"]["t0"][0]["path"]).read_text().splitlines() if l.strip()]
    by = {t["task_id"]: t for n in GEN.POOLS for t in A.jsonl(REPO / f"data/stages/stage-2/reasoning_data/tasks/{n}.jsonl")}
    mbpp = [r for r in t0_rows if by[r["task_id"]].get("verifier") == "code_asserts"][:2]
    t0 = [r for r in t0_rows if not r.get("interactive") and r not in mbpp][:22] + mbpp
    order = [json.loads(l) for l in ORDER.read_text().splitlines()[:6 + 24 + 24]]
    for i, r in enumerate(order):
        r["slice"], r["wave"] = (("V", None) if i < 6 else ("E", (i - 6) // 8 + 1) if i < 30 else ("X", (i - 30) // 8 + 1))
    cal = [json.loads(l) for l in (REPO / PROTO["inputs"]["calibration"]["path"]).read_text().splitlines()[:4]]
    files = {"t0.jsonl": t0, "order.jsonl": order, "cal.jsonl": cal}
    proto = json.loads(json.dumps(PROTO))
    for name, rows in files.items():
        p = root / "inputs" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        files[name] = {"path": f"inputs/{name}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "tasks": len(rows)}
    proto["inputs"].update(t0=[files["t0.jsonl"]], order=files["order.jsonl"], calibration=files["cal.jsonl"])
    proto["stopping"].update(target_sequences=11, pod_margin=0.0)
    (root / GEN.PROTOCOL).parent.mkdir(parents=True, exist_ok=True)
    (root / GEN.PROTOCOL).write_text(json.dumps(proto))
    return root


def _verdict(task, rec, index):
    """A seeded stand-in for H1's verifiers: ~70% of candidates accepted, per (task, mode, candidate)."""
    from aadistill.data.target_quality import CandidateRecord
    st = rec["turns"][0].get("think_state")
    ok = int(hashlib.sha256(f"{task['task_id']}|{st}|{index}".encode()).hexdigest(), 16) % 10 < 7
    limited = any(t.get("stop_reason") == "length_limited" for t in rec["turns"])
    return CandidateRecord(unit_id=task["task_id"], candidate_index=index, naturally_terminated=not limited,
                           length_limited=limited, degenerated=False, protocol_valid=True,
                           answer="correct" if ok else "incorrect", answer_scope="seeded", steps="not_assessed",
                           reasons=[] if ok else ["answer:incorrect:wrong answer"], accepted_for_training=ok)


class FakeTeacher:
    """run_battery's contract: one record per input task streamed to --stream-completed."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, log, tick):
        arg = lambda k: cmd[cmd.index(k) + 1]  # noqa: E731
        thinking = json.loads(arg("--chat-template-kwargs"))["enable_thinking"]
        mode = "H-T" if thinking else "H-N"
        suffix = PROTO["modes"][mode]["generation_suffix_ids"]
        self.calls.append((Path(arg("--tasks")).name, mode, arg("--seed-base")))
        with open(arg("--stream-completed"), "a") as out:
            for line in Path(arg("--tasks")).read_text().splitlines():
                t = json.loads(line)
                rng = random.Random(f"{t['task_id']}|{mode}|{arg('--seed-base')}")
                n = rng.randint(2300, 2700) if thinking else rng.randint(1600, 1800)
                limited = thinking and rng.random() < 0.1
                p = [11] * 40 + suffix
                o = [7] * n + ([] if limited else [151645])
                turn = {"turn": 0, "prompt_tokens": len(p), "generated_tokens": len(o), "generation_allowance": 8184 - len(p),
                        "stop_reason": "length_limited" if limited else "eos", "think_state": PROTO["modes"][mode]["think_state"],
                        "raw": "x", "answer": "x", "prompt_token_ids": p, "output_token_ids": o}
                out.write(json.dumps({"task_id": t["task_id"], "turns": [turn], "end": "answered",
                                      "messages": [{"role": "system", "content": "You are a helpful Assistant."},
                                                   *t["messages"]]}) + "\n")
        return 0


def _initialized(c, root, cal, status, inits=None):
    """initialization.py's durable product, stood in (its own tests are test_l40s_first's)."""
    (root / "initialization").mkdir(parents=True, exist_ok=True)
    (root / "initialization" / "initialization.json").write_text("{}")
    return {"status": "persisted"}


def _session(root: Path, durable: Path, fake, monkeypatch, phases="calibration,initialization,waves,x",
             initialize=_initialized):
    monkeypatch.setattr(GA, "candidate", _verdict)
    authorize = lambda path, repo: {"authorization_id": "a", "identity": IDENT, "harness_digest": "h"}  # noqa: E731
    return GEN.main(["--authorization", "a.json", "--durable-root", str(durable), "--status", str(durable / "status"),
                     "--phases", phases], run=fake, authorize=authorize, repo=root, initialize=initialize)


def test_the_teacher_session_runs_its_waves_bounds_the_rule_and_resumes_without_regenerating(tmp_path, monkeypatch):
    root, durable, fake = _tiny_repo(tmp_path), tmp_path / "vol", FakeTeacher()
    assert _session(root, durable, fake, monkeypatch) == 0
    rec = json.loads((durable / "teacher_session.json").read_text())
    assert rec["terminal"] == "ALL_DONE" and "MARKER:ALL_DONE" in (durable / "status").read_text()
    # both modes on T0/V/E with the same seed base per candidate; X in H-N only; calibration one candidate
    by_stage = {}
    for f, mode, seed in fake.calls:
        by_stage.setdefault(f.split("_")[0], set()).add((mode, seed))
    assert by_stage["cal"] == {("H-T", "20261029"), ("H-N", "20261029")}
    assert {m for m, _ in by_stage["w00"]} == {"H-T", "H-N"} and all(m == "H-N" for s, v in by_stage.items()
                                                                       if s.startswith("x") for m, _ in v)
    waves = rec["phases"]["waves"]
    stopped = waves["stopped"]["after"]
    lower = [s["H-T-S_lower"] for s in waves["bound"]["stages"]]
    assert lower[-1] >= 11 and all(x < 11 for x in lower[:-1])               # E stopped at its first reaching wave
    assert stopped.startswith("w") and stopped != "w00"                       # ... inside E, after T0
    assert stopped == waves["bound"]["stages"][-1]["stage"]
    # resume: a second session on the same volume generates nothing and reaches the same decisions
    again = FakeTeacher()
    assert _session(root, durable, again, monkeypatch) == 0
    assert again.calls == []
    rec2 = json.loads((durable / "teacher_session.json").read_text())
    assert rec2["phases"]["waves"]["stopped"]["after"] == stopped

    # the dev box's exact rule on the same shards
    got = ADM.decide(durable / "generation", IDENT, repo=root)
    first = next(w["wave"] for w in got["waves"] if w["H-T-S_sequences"] >= 11)
    assert got["stop"] == first and got["e_state"] == "REACHED"
    pod = {s["stage"]: s for s in waves["bound"]["stages"]}
    assert all(pod[w["wave"]]["H-T-S_lower"] <= w["H-T-S_sequences"] <= pod[w["wave"]]["H-T-S_upper"]
               for w in got["waves"])
    assert all(v["admitted"] is False for (t, m), v in got["verdicts"].items()
               if v["candidates"] and all(c["length_limited"] for c in v["candidates"]))
    hts, hns = got["arms"]["H-T-S"], got["arms"]["H-N-S"]
    assert hts["sequences"] >= 11 and hts["optimizer_steps"] == -(-3 * hts["sequences"] // 2)
    assert set(t.rsplit("|", 2)[0] for t in hts["doc_ids"]) == set(got["I"]) == set(t.rsplit("|", 2)[0] for t in hns["doc_ids"])
    if "H-N-C" in got["arms"]:
        assert got["arms"]["H-N-C"]["sequences"] >= hts["sequences"]
    else:
        assert got["c_state"].startswith("FALLBACK") and got["arms"]["H-T-C"]["sequences"] >= hns["sequences"] - 1
    # the packs: written from the selection alone, identical when rebuilt (what the recovery pod checks)
    sel = ADM.selection(got)
    a = ADM.build_packs(sel, durable / "generation", IDENT, tmp_path / "packs_a", repo=root)
    b = ADM.build_packs(json.loads(json.dumps(sel)), durable / "generation", IDENT, tmp_path / "packs_b", repo=root)
    assert {k: v["files_sha256"] for k, v in a.items()} == {k: v["files_sha256"] for k, v in b.items()}
    from aadistill.data.ladder import ladder_blocks
    lad = a["H-T-S"]
    train, val, _ = ladder_blocks(tmp_path / "packs_a" / "H-T-S", lad["rungs"][0]["target_supervised_tokens"],
                                  n_val=lad["n_val_blocks"])
    assert len(train) == 4 and train[0].shape[0] == hts["sequences"] and len(val) == 4
    # C's feasible branch (here the tiny X shifts a stratum by more than 2 points, so the rule falls back): the
    # smallest whole-task prefix of X's stratified interleave that brings H-N-C to H-T-S's count
    monkeypatch.setattr(ADM, "MIXTURE_POINTS", 100.0)
    wide = ADM.decide(durable / "generation", IDENT, repo=root)
    hnc, x = wide["arms"]["H-N-C"], wide["x_ids"]
    assert wide["c_state"] == "FEASIBLE" and hnc["sequences"] >= wide["arms"]["H-T-S"]["sequences"] and x
    base = ADM.docs_of(wide["I"], "H-N", wide["verdicts"])
    assert ADM.count({**base, **ADM.docs_of(x[:-1], "H-N", wide["verdicts"])}, PROTO) < wide["arms"]["H-T-S"]["sequences"]
    assert ADM.build_packs(ADM.selection(wide), durable / "generation", IDENT, tmp_path / "packs_c", repo=root)["H-N-C"]
    cen = ADM.censoring(got)
    assert cen["rows"] and all(0 <= r["first_length_limited_share"] <= 1 for r in cen["rows"])
    # each candidate's correctness is the verifier's answer state (CandidateRecord.answer is that string)
    firsts = [v["candidates"][0] for v in got["verdicts"].values() if v["candidates"]]
    assert sum(r["first_correct"] for r in cen["rows"]) == sum(c["correct"] for c in firsts) > 0


def test_split_sessions_generate_the_one_sessions_tasks_and_x_stops_on_the_waves_sealed_before(tmp_path, monkeypatch):
    root = _tiny_repo(tmp_path)
    one, a = tmp_path / "one", FakeTeacher()
    assert _session(root, one, a, monkeypatch) == 0
    r1 = json.loads((one / "teacher_session.json").read_text())
    assert "reason" not in r1["phases"]["x"]["stopped"]         # X reached its target before X ran out
    # the same work in three sessions -- bounded after W00, the rest of the waves, then X alone: X's target is
    # rebuilt from the sealed shards, never from an earlier process
    two, w, b, c = tmp_path / "two", FakeTeacher(), FakeTeacher(), FakeTeacher()
    assert _session(root, two, w, monkeypatch, phases="calibration,initialization,waves@w00") == 0
    r0 = json.loads((two / "teacher_session.json").read_text())["phases"]["waves"]
    assert r0["stopped"]["after"] == "w00" and "bounded" in r0["stopped"]["reason"] and "x" not in json.loads(
        (two / "teacher_session.json").read_text())["phases"]
    assert r0["stages"][0]["generated_here"] and r0["stages"][0]["generated_tokens"] > 0
    assert {f.split("_")[0] for f, _, _ in w.calls} == {"cal", "w00"}
    assert _session(root, two, b, monkeypatch, phases="waves") == 0
    assert _session(root, two, c, monkeypatch, phases="x") == 0
    r2 = json.loads((two / "teacher_session.json").read_text())
    assert c.calls and all(f.startswith("x") for f, _, _ in c.calls)
    assert sorted(w.calls + b.calls + c.calls) == sorted(a.calls)
    assert sorted(p.name for p in (two / "generation").glob("*.complete")) == sorted(
        p.name for p in (one / "generation").glob("*.complete"))
    assert r2["phases"]["waves"]["replayed_from_sealed_shards"] and r2["phases"]["x"]["stopped"] == r1["phases"]["x"]["stopped"]
    # X does not start on waves that are not all sealed, nor anything after an initialization that is not persisted
    part = tmp_path / "part"
    assert _session(root, part, FakeTeacher(), monkeypatch, phases="x") == 3
    assert "initialization is not persisted" in json.loads((part / "teacher_session.json").read_text())["phases"]["x"]["reason"]
    _initialized(None, part, None, None)
    x = FakeTeacher()
    assert _session(root, part, x, monkeypatch, phases="x") == 3 and x.calls == []
    assert "not all sealed" in json.loads((part / "teacher_session.json").read_text())["phases"]["x"]["reason"]


def test_a_failed_calibration_or_initialization_stops_every_phase_that_depends_on_it(tmp_path, monkeypatch):
    root = _tiny_repo(tmp_path)

    class NoCalibration(FakeTeacher):
        def __call__(self, cmd, log, tick):
            if Path(cmd[cmd.index("--tasks") + 1]).name.startswith("cal_"):
                self.calls.append(("cal", "", ""))
                return 1
            return super().__call__(cmd, log, tick)
    inits, fake = [], NoCalibration()
    track = lambda *a: inits.append(a) or _initialized(*a)  # noqa: E731
    assert _session(root, tmp_path / "a", fake, monkeypatch, initialize=track) == 3
    assert inits == [] and {f for f, _, _ in fake.calls} == {"cal"}
    assert _session(root, tmp_path / "a", FakeTeacher(), monkeypatch, phases="initialization", initialize=track) == 3
    assert inits == [] and "calibration is not complete" in json.loads(
        (tmp_path / "a" / "teacher_session.json").read_text())["phases"]["initialization"]["reason"]
    failed, gen = lambda *a: {"status": "failed", "returncode": 1}, FakeTeacher()  # noqa: E731
    assert _session(root, tmp_path / "b", gen, monkeypatch, initialize=failed) == 3
    assert {f.split("_")[0] for f, _, _ in gen.calls} == {"cal"}            # no wave, no X after it


def test_a_hybrid_preclosed_turn_is_split_by_its_state_and_h1s_turns_are_unchanged():
    ht = {"raw": "<think>\nr\n</think>\n\nA<|im_end|>", "think_state": "self_opening"}
    hn = {"raw": "A <|im_end|>", "think_state": "preclosed"}
    assert GA.split_turn(ht) == ("r", "A") and GA.split_turn(hn) == ("", "A")
    assert GA.split_turn({"raw": "<think>x</think> A", "think_state": "preclosed"}) is None
    from stages.reasoning_data import assets as A
    for raw in ("r\n</think>\n\nA<|im_end|>", "no close", "a</think>b</think>c"):
        assert GA.split_turn({"raw": raw}) == A.split_raw(raw) == GA.split_turn({"raw": raw, "think_state": "preopened"})


def test_calibration_items_are_mode_balanced_whole_trajectories():
    from stages.teacher_modes import hybrid_calibration as CAL
    tasks = {f"t{i}": {"calibration": {"profile": "calib.s0v_base_d4@v1", "item_id": f"i{i}", "domain": "d",
                                       "subtype": "s"}} for i in range(6)}
    recs = {m: [{"task_id": f"t{i}", "turns": [{"prompt_token_ids": [1] * 5,
                                                 "output_token_ids": [2] * (300 if m == "H-T" else 50) + [151645]}]}
                for i in range(6)] for m in ("H-T", "H-N")}
    got = CAL.items_from_records(recs, tasks, [151645])["calib.hybrid_modes_base_d4@v1"]
    used = got["scored_positions_used"]
    assert got["items_used"]["H-N"] == 6 and used["H-T"] >= used["H-N"] and used["H-T"] - used["H-N"] < 302
    assert all(i["tags"]["assistant"] == list(range(4, len(i["ids"]) - 1)) for i in got["items"])
    assert {i["subtype"] for i in got["items"]} == {"s|H-T", "s|H-N"}


def test_the_decision_rule_labels_a_thinking_gain_and_is_bound():
    from stages.teacher_modes import mode_statistics as ST
    pre = json.loads((REPO / "logs/stages/stage-2/teacher_modes/plans/d4_scale_preregistration.json").read_text())
    assert pre["decision_rule"]["bound_sha256"] == ST.bound_identity()
    dec = pre["evaluation"]["decoding"]["sha256"]
    rng, rows = random.Random(3), []
    for i in range(240):
        for seed in (1, 2):
            for arm, p in (("H-T-S", 0.6), ("H-N-S", 0.3), ("H-N-C", 0.58)):
                rows.append({"arm": arm, "seed": seed, "item_id": f"i{i}", "cluster": f"c{i % 30}",
                             "axis": ("task_contained_deduction", "inductive_rule_discovery")[i % 2],
                             "correct": rng.random() < p, "decoding_sha256": dec})
    out = ST.decide(rows, prereg=pre, c_fallback=False)
    assert out["comparisons"]["S"]["primary"]["label"] == "GO_THINKING"
    assert out["comparisons"]["C"]["primary"]["label"] in ("NO_GO", "INCONCLUSIVE")
    with pytest.raises(ValueError, match="decoding"):
        ST.decide([{**rows[0], "decoding_sha256": "x"}] + rows[1:], prereg=pre, c_fallback=False)


def test_a_recovery_artifact_cannot_be_issued_before_the_admission_and_initialization_records():
    from stages.teacher_modes import phase as PH
    if (REPO / PH.ADMISSION).is_file() and (REPO / PH.INIT_RECORD).is_file():
        pytest.skip("the records exist: the phase is past this point")
    with pytest.raises(PH.SE.AuthorizationError, match="teacher session"):
        PH.kind_conditions("recovery")
    assert PH.kind_conditions("check") == {}
    assert PH.STUDY.formal and set(PH.STUDY.kinds) == {"check", "teacher", "recovery"}
    assert set(PH.STUDY.identity_files) >= {PH.DESIGN, PH.PREREG, PH.PROTOCOL, PH.ORDER}


def _records(tmp: Path, arms=("H-T-S", "H-N-S", "H-N-C")) -> tuple[Path, Path]:
    pack = {"files_sha256": {"blocks.npz": "1" * 64, "ladder.json": "2" * 64, "audit.jsonl": "3" * 64},
            "rungs": [{"target_supervised_tokens": 4_000_000, "n_blocks": 2500}], "n_val_blocks": 16}
    adm = {"arms": {a: {"mode": "H-T" if a.startswith("H-T") else "H-N", "sequences": 2500 if a != "H-N-S" else 800,
                        "pack": pack} for a in arms}, "C": {"state": "FEASIBLE"}, "selection": {}}
    init = {"materialization_identity": {"artifact_digest": "a" * 64, "weights_digest": "b" * 64,
                                         "arch_signature": "c" * 64, "num_parameters": 596049920}}
    (tmp / "adm.json").write_text(json.dumps(adm))
    (tmp / "init.json").write_text(json.dumps(init))
    return tmp / "adm.json", tmp / "init.json"


def test_the_recovery_recipe_differs_from_e1_only_by_the_declared_protocol(tmp_path):
    from stages.reasoning_data import h1pa_driver as HD
    from stages.reasoning_data import unit_driver as UD
    from stages.teacher_modes import recovery_units as RD
    adm, init = _records(tmp_path)
    plan = RD.load_plan(admission=adm, init_record=init)
    assert [u["unit"] for u in plan["units"]] == [f"{a}-s{s}" for a in ("H-N-C", "H-N-S", "H-T-S")
                                                   for s in (20260726, 20260801)]
    assert plan["study"]["frozen"] is False                        # set only after the pod's authorization check
    u = plan["units"][-1]
    sched = {"rung": 4_000_000, **HD.e1_schedule(2500, epochs=3, batch_size=2)}
    got = HD.derive_recipe(plan, u, sched, run_name="r", purpose="p", out_dir="o", data_dir="d", student_path="s")
    e1 = plan["recipes"][u["seed"]]
    a, b = UD._flat(e1), UD._flat(got)
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    declared = {"loss.kd_scope", "loss.kd_support.support_id", "loss.kd_support.top_k", "batch.context_strategy",
                "checkpoint.bind_identity", "teacher.model_id", "teacher.revision", "val_blocks", "intervals.eval_blocks",
                "rung", "schedule.total_steps", "schedule.warmup_steps", "checkpoint.save_every", "intervals.eval_every",
                *HD.WHERE, *HD.PINS}
    assert diff <= declared and {"loss.kd_support.top_k", "batch.context_strategy", "teacher.model_id"} <= diff
    assert got["loss"]["kd_support"] == {"support_id": "reference_topk_tail_v1", "top_k": 200}
    assert got["teacher"] == {"model_id": "Qwen/Qwen3-4B", "revision": PROTO["teacher"]["revision"], "dtype": "bfloat16"}
    assert got["schedule"]["total_steps"] == 3750 and got["batch"]["context_strategy"] == "document_isolated"
    groups = RD.partition(plan["units"], plan, 2)
    assert sorted(sum(groups, [])) == sorted(x["unit"] for x in plan["units"]) and len(groups) == 2
    with pytest.raises(HD.DriverRefusal, match="not committed"):
        RD.load_plan(admission=tmp_path / "none.json", init_record=init)


def test_the_initialization_registers_the_hybrid_profiles_and_d4_top1s_four_operators(tmp_path):
    """In a child (the operator and profile registries are process-global): the two hybrid profiles validate
    under the profile rules, D4 Top-1's four implementations are registered, and the fixed path builds."""
    import subprocess
    code = f"""
import json, sys
sys.path[:0] = {[str(REPO / p) for p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-2")]}
from pathlib import Path
from stages.teacher_modes import hybrid_calibration as CAL, initialization as INI
from aadistill.initialization.calibration.profiles import load_profiles, get_profile
from aadistill.initialization.operators.base import get_implementation
from shared import forward_baseline as FB
from stages.phase_d1 import d1_session as D1S
tasks = {{f"t{{i}}|{{p}}": {{"calibration": {{"profile": p, "item_id": f"i{{i}}", "domain": "inductive_rule_discovery",
          "subtype": "g"}}}} for i in range(4) for p in CAL.PROFILES}}
recs = {{m: [{{"task_id": t, "turns": [{{"prompt_token_ids": [1] * 9, "output_token_ids": [2] * 40 + [151645]}}]}}
             for t in tasks] for m in ("H-T", "H-N")}}
built = CAL.items_from_records(recs, tasks, [151645])
doc = INI.profiles_doc(built, Path({str(tmp_path)!r}))
D1S._register_frozen_operators(); FB.register_operators(FB.selected())
load_profiles(doc)
for impl, prof in INI.PATH:
    get_implementation(impl); get_profile(CAL.HYBRID[prof])
spec = INI.fixed_path_spec("cpu")
print(json.dumps({{"steps": [s.profile_id for s in spec.steps], "root": [spec.root_repo_id, spec.root_revision],
                   "hash": spec.spec_hash, "seed": spec.seed}}))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert res.returncode == 0, res.stderr[-3000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["steps"] == ["calib.hybrid_modes_rh_d4@v1"] + ["calib.hybrid_modes_base_d4@v1"] * 3
    assert out["root"] == ["Qwen/Qwen3-4B", PROTO["teacher"]["revision"]] and out["seed"] == 20260815
