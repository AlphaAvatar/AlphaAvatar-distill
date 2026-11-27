"""DMR1 variant 1 (the alpha merge at SOB1's depth structure) and the round's shared chain, at $0.

* the design's pins: the spans are SOB1's committed mapping's, the protocol and the registered items are SOB1's, the
  base digests are the base's, the proposals reproduce;
* the alpha merge on the toy teacher with a 4-block span: each merged slot is sum_i alpha_i A_i(theta_i) with alpha the
  importance normalized over the span, recomputed here from the definitions; an all-zero slot is uniform; single
  spans and every other tensor are the base's;
* the alignments are durable pair by pair and reused only under the same inputs (the FFN correlation AND the
  attention Grams): an unchanged correlation with changed head statistics is recomputed;
* the build's products are durable only when the relay verifies them AND their handoff does, are never sent twice,
  and each is bound to its own scientific identity; a takeover makes only what is missing (weights verified, no init
  record, pod lost -> nothing is built; only the checkpoint's identity changed -> the statistics are reused and only
  the weights made again; a failed handoff -> only the handoff is retried; products published before their build
  record came back -> restored from their relay records); a takeover asks the relay only for what the chain still
  adds; the span diagnostic moves only the kept blocks to the device;
* the plan is SOB1's depth unit outside the declared fields;
* the registered comparison's family rule (thinking at 0.05, non-thinking only after it rejects; formal only when
  complete) and its per-seed reading;
* the launcher: its kinds and grants, an artifact that does not fund the arm is refused at $0.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stages.depth_merge import common as DC
from stages.depth_merge import merge as M

REPO = Path(__file__).resolve().parents[5]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- the design's pins --------------------------------------------------------------------------------------------

def test_the_spans_are_the_committed_mappings_and_the_protocol_is_sob1s():
    d = DC.design()
    spans = M.merge_spans(d)
    assert [len(s) for s in spans] == [3, 3, 2, 6, 3, 5, 2, 3, 2] and len(M.pairs(spans)) == 20
    assert sum(len(s) for s in M.all_spans(d)) == 36 and len(M.all_spans(d)) == 16
    sob = json.loads((REPO / d["baseline"]["sob1_design"]["path"]).read_text())
    assert d["evaluation"]["protocol"] == sob["evaluation"]["protocol"]
    assert d["analysis"]["primary_items"] == sob["analysis"]["primary_items"]
    assert d["analysis"]["units_of_independence"] == sob["analysis"]["units_of_independence"]
    assert d["teacher"] == sob["teacher"]
    for k in ("sob1_design", "init_record", "training_plan", "study_completion"):
        assert sha(REPO / d["baseline"][k]["path"]) == d["baseline"][k]["sha256"], k
    dig = json.loads(M.DIGESTS.read_text())
    assert dig["checkpoint"]["weights_sha256"] == d["baseline"]["init_record"]["model_safetensors_sha256"]
    assert dig["n_tensors"] == 16 * 11 + 2


def test_the_proposals_and_estimates_reproduce():
    from stages.depth_merge import proposal as P
    for p, doc in P.documents().items():
        assert p.read_text() == P.render(doc), p
    t = json.loads((DC.LOGS / "plans/proposal_dmr1_merge_train.json").read_text())
    b = json.loads((DC.LOGS / "plans/proposal_dmr1_merge_build.json").read_text())
    assert t["status"] == b["status"] == "proposed" and t["authorizes"].startswith("nothing")
    total = t["per_session_hard_cap_usd"] + b["per_session_hard_cap_usd"] + 2 * P.TEARDOWN_RESERVE
    assert total == pytest.approx(t["variant_1_total_with_reserves_usd"])
    assert total <= P.POSITION["project_remaining_usd"]          #: fits the cap's remainder as read 2026-11-26


# ---- the alpha merge on the toy teacher ---------------------------------------------------------------------------

def _batches(n=3, t=40, vocab=128):
    g = torch.Generator().manual_seed(5)
    seqs = [torch.randint(0, vocab, (1, t), generator=g) for _ in range(n)]

    def make():
        for s in seqs:
            yield SimpleNamespace(batch=SimpleNamespace(input_ids=s, attention_mask=torch.ones_like(s)),
                                  original_indices=(0,))
    return make


TOY_SEL = {"n_root_blocks": 6, "kept_blocks": [0, 1, 5], "source": {"seed": 0}}      #: spans [0], [1..4], [5]


def _toy(teacher):
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.planning.layer_merge import spans_from_kept_blocks
    from aadistill.initialization.specs.arch import get_adapter
    adapter = get_adapter("qwen3")
    spans = [s for s in spans_from_kept_blocks(6, TOY_SEL["kept_blocks"]) if len(s) > 1]
    base = FM.keep_blocks(adapter, teacher, TOY_SEL["kept_blocks"])
    digests = {n: FM.tensor_sha256(t) for n, t in base.state_dict().items() if n != "lm_head.weight"}
    return adapter, spans, base, digests


@torch.no_grad()
def test_each_merged_slot_is_the_importance_normalized_sum_of_the_aligned_blocks(teacher, tmp_path):
    LMX = M.lmx()
    adapter, spans, base, digests = _toy(teacher)
    assert spans == [(1, 2, 3, 4)]
    n_q, n_kv, hd = adapter.head_groups(adapter.spec_of(teacher))
    st = LMX.collect_statistics(adapter, teacher, spans, _batches(), None, chunk=16)
    written = {}
    al, report = M.pair_alignments(st, n_q, n_kv, on_pair=lambda tag, x: written.__setitem__(tag, x))
    assert sorted(written) == ["1->2", "1->3", "1->4"] and set(report.values()) == {"computed"}
    stats = LMX.statistics_tensors(st, al)
    model, rec = LMX.build_candidate(adapter, teacher, TOY_SEL, spans, M.METHOD, stats, digests, seed=0)
    #: every tensor outside the merged block is the base's, the merged block is not
    assert rec["checks"]["tensors_outside_spans_differ"] == [] and all(rec["checks"]["span_blocks_differ_from_D"].values())
    #: the gate rows, recomputed from the definitions: fold, align, alpha = I / sum I per neuron
    blocks = adapter.blocks(teacher)
    span = spans[0]
    g_al, s_al = [], []
    for k, b in enumerate(span):
        gate = blocks[b].mlp.gate_proj.weight.double() * blocks[b].post_attention_layernorm.weight.double()[None, :]
        s = stats[f"block{b}.ffn_mean_abs"].double() * blocks[b].mlp.down_proj.weight.double().norm(dim=0)
        if k:
            perm = stats[f"pair{span[0]}_{b}.ffn_perm"]
            gate, s = gate[perm], s[perm]
        g_al.append(gate)
        s_al.append(s)
    alpha = torch.stack(s_al) / torch.stack(s_al).sum(0)
    assert torch.allclose(alpha.sum(0), torch.ones_like(alpha[0]))
    want = sum(alpha[k][:, None] * g_al[k] for k in range(len(span)))
    got = adapter.blocks(model)[1].mlp.gate_proj.weight.double()
    assert torch.allclose(got, want.float().double(), atol=1e-6)
    merged = adapter.blocks(model)[1]
    assert torch.equal(merged.post_attention_layernorm.weight, torch.ones_like(merged.post_attention_layernorm.weight))
    #: the single spans are the base's, bit for bit
    for i in (0, 2):
        for (n, a), (_, b_) in zip(adapter.blocks(model)[i].state_dict().items(), adapter.blocks(base)[i].state_dict().items()):
            assert torch.equal(a, b_), n


@torch.no_grad()
def test_an_all_zero_slot_takes_uniform_weights_and_is_counted(teacher):
    LMX = M.lmx()
    adapter, spans, _base, _dig = _toy(teacher)
    n_q, n_kv, _ = adapter.head_groups(adapter.spec_of(teacher))
    st = LMX.collect_statistics(adapter, teacher, spans, _batches(), None, chunk=16)
    al, _ = M.pair_alignments(st, n_q, n_kv)
    stats = LMX.statistics_tensors(st, al)
    for b in spans[0]:
        stats[f"block{b}.ffn_mean_abs"][3] = 0.0              #: neuron slot 3's importance is zero in every block
        perm = stats.get(f"pair1_{b}.ffn_perm")
        if perm is not None:                                  #: the absorbed block's neuron that lands in slot 3
            stats[f"block{b}.ffn_mean_abs"][int(perm[3])] = 0.0
    _t, how = LMX.merged_tensors(adapter, teacher, spans[0], M.METHOD, stats)
    assert how["uniform_fallback_slots"]["ffn"] >= 1


def test_alignments_are_reused_only_under_the_same_inputs(teacher):
    LMX = M.lmx()
    adapter, spans, _b, _d = _toy(teacher)
    n_q, n_kv, _ = adapter.head_groups(adapter.spec_of(teacher))
    st = LMX.collect_statistics(adapter, teacher, spans, _batches(), None, chunk=16)
    written = {}
    al1, _ = M.pair_alignments(st, n_q, n_kv, on_pair=lambda tag, x: written.__setitem__(tag, x))
    again = {}
    al2, rep = M.pair_alignments(st, n_q, n_kv, persisted=written, on_pair=lambda t, x: again.__setitem__(t, x))
    assert set(rep.values()) == {"reused (same inputs)"} and not again
    for k in al1:
        assert torch.equal(al1[k]["ffn_perm"], al2[k]["ffn_perm"])
    stale = {k: {**v, "inputs": {**v["inputs"], "code_sha256": "0" * 64}} for k, v in written.items()}
    _al3, rep3 = M.pair_alignments(st, n_q, n_kv, persisted=stale, on_pair=lambda t, x: again.__setitem__(t, x))
    assert set(rep3.values()) == {"recomputed (code_sha256 differ)"} and sorted(again) == sorted(written)


def test_an_unchanged_ffn_correlation_does_not_carry_a_stale_attention_matching(teacher):
    """The counter-example: the absorbed block's head statistics change (its heads in reverse order), its FFN Pearson
    moments do not. The persisted pair's FFN correlation hash still matches; its attention matching must not be reused."""
    LMX = M.lmx()
    adapter, spans, _b, _d = _toy(teacher)
    n_q, n_kv, _ = adapter.head_groups(adapter.spec_of(teacher))
    st = LMX.collect_statistics(adapter, teacher, spans, _batches(), None, chunk=16)
    written = {}
    M.pair_alignments(st, n_q, n_kv, on_pair=lambda tag, x: written.__setitem__(tag, x))
    key = (1, 2)
    st2 = copy.deepcopy(st)
    rev = torch.arange(n_q - 1, -1, -1)
    hm = st2.heads[key]
    hm.cross, hm.energy_b = hm.cross[:, rev], hm.energy_b[rev]
    st2.self_heads[2].cross = st2.self_heads[2].cross[rev][:, rev]
    assert torch.equal(st2.pearson[key].correlation(), st.pearson[key].correlation())
    fresh = LMX.alignments(st2, n_q, n_kv)[key]
    old = written["1->2"]
    assert (fresh["kv_perm"].tolist(), fresh["q_perm_within"].tolist()) != (old["kv_perm"], old["q_perm_within"])
    got, rep = M.pair_alignments(st2, n_q, n_kv, persisted=written)
    assert rep["1->2"] == "recomputed (attention_sha256 differ)"
    assert rep["1->3"] == rep["1->4"] == "reused (same inputs)"
    assert torch.equal(got[key]["kv_perm"], fresh["kv_perm"]) and torch.equal(got[key]["q_perm_within"],
                                                                             fresh["q_perm_within"])
    assert torch.equal(got[key]["ffn_perm"], fresh["ffn_perm"])


def test_the_base_check_refuses_a_rebuilt_base_that_is_not_the_bases(teacher):
    from stages.depth_merge import build_driver as BD
    adapter, _spans, _base, digests = _toy(teacher)
    BD.check_base(adapter, teacher, TOY_SEL, digests)
    bad = dict(digests)
    bad["model.norm.weight"] = "0" * 64
    with pytest.raises(BD.DriverError):
        BD.check_base(adapter, teacher, TOY_SEL, bad)


# ---- the build's durability and takeover -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def contract(tmp_path_factory):
    """The real $0 contract of this tree (the packed search's session on the CPU, about 3 s)."""
    from stages.depth_merge import build_driver as BD
    return BD.contract(tmp_path_factory.mktemp("contract"), "cpu")


W_BYTES, CFG_BYTES, STATS_BYTES = b"weights", b"{}", b"stats"


def _checkpoint(stats_sha: str, ident: dict | None, *, weights: bytes = W_BYTES, digest: str = "a" * 64) -> dict:
    from stages.depth_merge import build_driver as BD
    w = hashlib.sha256(weights).hexdigest()
    ck = {"artifact_digest": digest, "reload_forward_finite": True, "statistics_sha256": stats_sha,
          "shards": [{"filename": "model.safetensors", "sha256": w, "size_bytes": len(weights)}],
          "files_sha256": {"model.safetensors": w, "config.json": hashlib.sha256(CFG_BYTES).hexdigest()}}
    return {**ck, "identity": BD.stamp(ident, "checkpoint")} if ident else ck


def _secured(ev: Path, ident: dict, *, finished: bool = True, relay: dict | None = None) -> dict:
    """A previous attempt's secured evidence (what `take_over` copies into ``out``): its build record with statistics,
    checkpoint and (``finished``) diagnostics, each stamped with ``ident``, the statistics file, and ``relay`` as
    relay.json."""
    from stages.depth_merge import build_driver as BD
    rec_dir = BD.records(ev)
    rec_dir.mkdir(parents=True)
    (rec_dir / f"{M.STATS}.safetensors").write_bytes(STATS_BYTES)
    st = sha(rec_dir / f"{M.STATS}.safetensors")
    rec = {"run_id": "r1", "arm": BD.ARM, "method": M.METHOD, "method_identity": {},
           "root": {"root_teacher_id": "t", "root_teacher_sha256": "s"},
           "statistics": {"tensors_sha256": st, "saved_utc": "u", "identity": BD.stamp(ident, "statistics")},
           "checks": {"equivalence": {"passed": True, "worst": 1e-6}}, "candidate": {},
           "checkpoint": _checkpoint(st, ident), "attempts": [{"run_id": "r1"}]}
    if finished:
        rec.update(diagnostics={"spans": {}}, state_evaluation={"values": {}}, finished_utc="u")
    (rec_dir / "build.json").write_text(json.dumps(rec))
    if relay is not None:
        (rec_dir / "relay.json").write_text(json.dumps(relay))
    return rec


def _verified(name: str, rec: dict, ident: dict, *, handed_off: bool = True) -> dict:
    """The group's relay record as `secure` writes it: verified, under ``ident``, with the product's description."""
    from stages.depth_merge import build_driver as BD
    return {"relay": "r", "prefix": BD.group_prefix(name, rec), "files_sha256": BD.group_files(name, rec),
            "bytes": {k: 7 for k in BD.group_files(name, rec)}, "verified": True, "group": name,
            "identity_sha256": ident[BD.WHAT[name]], "run_id": "r1", "product": BD.product_view(name, rec),
            "handoff": {"verified": handed_off}}


class Relay:
    """The relay as the driver sees it (`_put`, `_get`, `_listing`, `_handoff`), every call recorded. A put stores the
    files; a verified handoff publishes the record beside them (what `relay.records` then lists: the record without
    its handoff, `train_driver._pointer_body`); ``handoffs`` scripts their answers (True when exhausted)."""

    def __init__(self, listed=(), handoffs=()):
        self.calls, self.listed, self.handoffs, self.files = [], list(listed), list(handoffs), {}

    def put(self, payload, timeout_s):
        self.calls.append(("put", payload["prefix"].split("/")[-2]))
        files = {k: sha(v) for k, v in payload["files"].items()}
        self.files.update({f"{payload['prefix']}/{k}": Path(v).read_bytes() for k, v in payload["files"].items()})
        return {"relay": "r", "prefix": payload["prefix"], "verified": True, "files_sha256": files,
                "bytes": {k: Path(v).stat().st_size for k, v in payload["files"].items()}}

    def get(self, record, dest, timeout_s):
        self.calls.append(("get", record["prefix"].split("/")[-2]))
        for k in record["files_sha256"]:
            t = Path(dest) / k
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(self.files.get(f"{record['prefix']}/{k}") or
                          (W_BYTES if k.endswith("model.safetensors") else STATS_BYTES if "statistics" in k
                           else CFG_BYTES))
        return {"verified": True}

    def listing(self, prefix):
        self.calls.append(("list", prefix.split("/")[-1]))
        return {"records": [{"present": True, "record": r} for r in self.listed if r["prefix"].startswith(prefix)]}

    def handoff(self, rec, scratch, listing, message):
        from stages.teacher_parent_initialization import train_driver as TD
        self.calls.append(("handoff", rec["group"]))
        ok = self.handoffs.pop(0) if self.handoffs else True
        if ok:
            self.listed = [r for r in self.listed if r["prefix"] != rec["prefix"]] + [TD._pointer_body(rec)]
        return {"verified": ok, "why": None if ok else "the listing does not name this record with its files"}

    def install(self, monkeypatch, BD):
        monkeypatch.setattr(BD, "_put", self.put)
        monkeypatch.setattr(BD, "_get", self.get)
        monkeypatch.setattr(BD, "_listing", self.listing)
        monkeypatch.setattr(BD, "_handoff", self.handoff)

    def moved(self) -> list:
        return [c for c in self.calls if c[0] in ("put", "get")]


def _no_gpu(*_a, **_k):
    raise AssertionError("a takeover of durable products opened the teacher")


def _gpu(work, device, rec):
    """`open_gpu` at the GPU boundary: what the real one records (the root, the base check), nothing loaded."""
    rec["root"] = {"root_teacher_id": "t", "root_teacher_sha256": "s"}
    rec.setdefault("checks", {})["base"] = {"tensors": 178, "equal_to_base": 178}
    return SimpleNamespace(kind="gpu")


class Steps:
    """The three GPU steps at their boundary: each makes the product the real one makes (the files, the record) and
    is counted. Identity, stamping, takeover and completion stay the driver's own."""

    def __init__(self, *, weights: bytes = W_BYTES, digest: str = "a" * 64, lost_in_diagnostics: bool = False):
        self.ran, self.w_bytes, self.digest, self.lost = [], weights, digest, lost_in_diagnostics

    def statistics(self, g, rec, out, status, run_id):
        from stages.depth_merge import build_driver as BD
        self.ran.append("statistics")
        p = BD.records(out) / f"{M.STATS}.safetensors"
        p.write_bytes(STATS_BYTES)
        return {"sequences": 1, "positions": 1, "tensors_sha256": sha(p), "saved_utc": "now"}

    def weights(self, g, rec, out, work, status):
        self.ran.append("weights")
        d = work / "products/init"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_bytes(self.w_bytes)
        (d / "config.json").write_bytes(CFG_BYTES)
        rec["checks"]["equivalence"] = {"passed": True, "worst": 1e-6}
        rec["candidate"] = {"method": M.METHOD}
        return None, _checkpoint(rec["statistics"]["tensors_sha256"], None, weights=self.w_bytes, digest=self.digest)

    def diagnostics(self, g, rec, out, work, m=None):
        self.ran.append(("diagnostics", sha(work / "products/init/model.safetensors")))
        if self.lost:
            raise SystemExit("the pod is lost")
        rec.update(diagnostics={"spans": {}}, state_evaluation={"values": {}}, finished_utc="now")

    def install(self, monkeypatch, BD):
        monkeypatch.setattr(BD, "open_gpu", _gpu)
        monkeypatch.setattr(BD, "statistics_step", self.statistics)
        monkeypatch.setattr(BD, "weights_step", self.weights)
        monkeypatch.setattr(BD, "diagnostics_step", self.diagnostics)


def _main(BD, tmp_path, contract, monkeypatch, resume: Path, tag: str = "2") -> tuple[int, dict, Path]:
    monkeypatch.setattr(BD, "contract", lambda work, device: contract)
    out = tmp_path / f"out{tag}"
    rc = BD.main(["--run-id", f"r{tag}", "--out", str(out), "--work", str(tmp_path / f"work{tag}"), "--status",
                  str(tmp_path / f"st{tag}"), "--device", "cpu", "--resume", str(resume)])
    return rc, json.loads((out / "report.json").read_text()), out


def test_weights_uploaded_and_verified_but_no_init_record_is_taken_over_without_building(tmp_path, contract,
                                                                                       monkeypatch):
    """The real interruption: the weights' relay copy verified (relay.json secured), the statistics' upload not yet,
    the init record never written, the pod lost with its files. The entry point writes the init record from the
    secured evidence: no teacher, no statistics forward, no weights build, the weights never sent again."""
    from stages.depth_merge import build_driver as BD
    ident = BD.identity(contract)
    ev = tmp_path / "secured"
    rec = _secured(ev, ident)
    (BD.records(ev) / "relay.json").write_text(json.dumps({"init": _verified("init", rec, ident)}))
    relay = Relay()
    relay.install(monkeypatch, BD)
    monkeypatch.setattr(BD, "open_gpu", _no_gpu)
    for step in ("statistics_step", "weights_step", "diagnostics_step"):
        monkeypatch.setattr(BD, step, _no_gpu)
    rc, rep, out = _main(BD, tmp_path, contract, monkeypatch, ev)
    assert rc == 0, rep.get("error")
    assert relay.moved() == [("put", "statistics")]
    doc = BD.durable_init(out, ident)
    assert doc is not None and doc["relay"]["init"]["files_sha256"] == BD.group_files("init", rec)
    assert "weights: taken over (the verified relay copy)" in rep["actions"]
    assert "diagnostics: taken over" in rep["actions"]
    rc2, rep2, _ = _main(BD, tmp_path, contract, monkeypatch, ev)          #: once more: the record itself
    assert rc2 == 0 and rep2["actions"] == ["init record: taken over (durable under this identity)"]


def test_a_verified_copy_the_dev_box_never_pulled_is_found_by_the_relay_listing(tmp_path, contract, monkeypatch):
    from stages.depth_merge import build_driver as BD
    from stages.teacher_parent_initialization import train_driver as TD
    ident = BD.identity(contract)
    ev = tmp_path / "secured"
    rec = _secured(ev, ident)                                        #: no relay.json: lost with the pod
    relay = Relay(listed=[TD._pointer_body(_verified(g, rec, ident)) for g in ("init", "statistics")])
    relay.install(monkeypatch, BD)
    monkeypatch.setattr(BD, "open_gpu", _no_gpu)
    rc, rep, out = _main(BD, tmp_path, contract, monkeypatch, ev)
    assert rc == 0, rep.get("error")
    assert not relay.moved() and not [c for c in relay.calls if c[0] == "handoff"]
    assert BD.durable_init(out, ident)["relay"]["statistics"]["found_by"].startswith("the relay listing")


def test_missing_diagnostics_restore_the_weights_from_their_verified_copy_and_rebuild_nothing(tmp_path, contract,
                                                                                           monkeypatch):
    from stages.depth_merge import build_driver as BD
    ident = BD.identity(contract)
    ev = tmp_path / "secured"
    rec = _secured(ev, ident, finished=False)
    (BD.records(ev) / "relay.json").write_text(json.dumps({"init": _verified("init", rec, ident),
                                                           "statistics": _verified("statistics", rec, ident)}))
    relay = Relay()
    relay.install(monkeypatch, BD)
    steps = Steps()
    steps.install(monkeypatch, BD)
    for step in ("statistics_step", "weights_step"):
        monkeypatch.setattr(BD, step, _no_gpu)
    rc, rep, out = _main(BD, tmp_path, contract, monkeypatch, ev)
    assert rc == 0, rep.get("error")
    assert relay.moved() == [("get", "init")]
    assert steps.ran == [("diagnostics", rec["checkpoint"]["files_sha256"]["model.safetensors"])]
    assert BD.durable_init(out, ident)


def test_products_made_under_another_identity_are_set_aside_and_never_adopted(tmp_path, contract, monkeypatch):
    from stages.depth_merge import build_driver as BD
    ident = BD.identity(contract)
    other = copy.deepcopy(ident)
    other["parts"]["calibration"]["profiles"] = {"calib.other@v1": "0" * 64}
    other["statistics"] = other["checkpoint"] = "f" * 64
    ev = tmp_path / "secured"
    rec = _secured(ev, other)
    (BD.records(ev) / "relay.json").write_text(json.dumps({"init": _verified("init", rec, other),
                                                           "statistics": _verified("statistics", rec, other)}))
    Relay().install(monkeypatch, BD)

    class Opened(Exception):
        pass

    def opened(*_a, **_k):
        raise Opened("the teacher is opened: the statistics are made again")
    monkeypatch.setattr(BD, "open_gpu", opened)
    rc, rep, out = _main(BD, tmp_path, contract, monkeypatch, ev)
    assert rc == 3 and "Opened" in rep["error"]["error"]
    assert list(BD.records(out).glob("build.superseded_*.json"))
    doc = BD.init_record(rec, {"init": _verified("init", rec, other), "statistics": _verified("statistics", rec, other)},
                         other)
    assert BD.product_problems(doc, other) == []
    bad = BD.product_problems(doc, ident)
    assert any("calibration" in b for b in bad), bad
    assert BD.durable_init(out, ident) is None


@pytest.mark.parametrize("change", ["candidate_code", "base", "method"])
def test_a_checkpoint_identity_change_reuses_the_statistics_and_rebuilds_only_the_weights(tmp_path, contract,
                                                                                          monkeypatch, change):
    """The statistics and the weights were made and secured, the init record never written; then only what the
    CHECKPOINT is a function of changes (the candidate code, the base, the merge method). Through the real entry
    point: the statistics are taken over (no forward; fetched from the relay only for the build when the pod's file is
    gone), the old weights are refused and kept as evidence under THEIR identity, only the weights, their upload and
    the diagnostics are made again, and the init record is under the new identity. A prose-only change (the design
    file's bytes) moves no identity."""
    from stages.depth_merge import build_driver as BD
    ident = BD.identity(contract)
    assert BD.identity({**contract, "design_sha256": "e" * 64}) == ident           #: prose is not identity
    ev = tmp_path / "secured"
    rec = _secured(ev, ident)
    (BD.records(ev) / "relay.json").write_text(json.dumps({"init": _verified("init", rec, ident),
                                                           "statistics": _verified("statistics", rec, ident)}))
    c2 = copy.deepcopy(contract)
    if change == "candidate_code":
        c2["method_identity"]["candidate"]["sha256"] = "c" * 64
    elif change == "base":
        d2 = copy.deepcopy(DC.design())
        d2["baseline"]["init_record"]["artifact_digest"] = "b" * 64
        monkeypatch.setattr(DC, "design", lambda: d2)
        (BD.records(ev) / f"{M.STATS}.safetensors").unlink()        #: and the pod's statistics file is gone
    else:
        monkeypatch.setattr(M, "METHOD", "M-other")
    new = BD.identity(c2)
    assert new["statistics"] == ident["statistics"] and new["checkpoint"] != ident["checkpoint"]
    relay = Relay()
    relay.files[f"{BD.group_prefix('statistics', rec)}/statistics/{M.STATS}.safetensors"] = STATS_BYTES
    relay.install(monkeypatch, BD)
    steps = Steps(weights=b"weights-2", digest="e" * 64)
    steps.install(monkeypatch, BD)
    rc, rep, out = _main(BD, tmp_path, contract=c2, monkeypatch=monkeypatch, resume=ev)
    assert rc == 0, rep.get("error")
    assert [r for r in steps.ran if r == "statistics"] == [] and steps.ran.count("weights") == 1
    assert steps.ran[-1] == ("diagnostics", hashlib.sha256(b"weights-2").hexdigest())
    assert relay.moved() == ([("get", "statistics")] if change == "base" else []) + [("put", "init")]
    assert any(a.startswith("weights: set aside (checkpoint: made under another identity") for a in rep["actions"])
    doc = BD.durable_init(out, new)
    assert doc is not None and BD.durable_init(out, ident) is None
    assert doc["checkpoint"]["identity"]["sha256"] == new["checkpoint"]
    assert doc["statistics"]["identity"]["sha256"] == ident["statistics"]
    assert doc["relay"]["statistics"]["identity_sha256"] == ident["statistics"]
    old = doc["superseded"][0]
    assert old["product"] == "checkpoint" and old["record"]["checkpoint"]["identity"]["sha256"] == ident["checkpoint"]
    assert old["record"]["checkpoint"]["files_sha256"] == rec["checkpoint"]["files_sha256"]
    assert old["relay"]["identity_sha256"] == ident["checkpoint"]
    assert json.loads((BD.records(out) / "relay.json").read_text())["superseded"][0]["identity_sha256"] == \
        ident["checkpoint"]
    #: the record's method is the one its weights were built with, never the earlier build record's
    assert doc["method"] == doc["checkpoint"]["identity"]["parts"]["merge"]["method"] == new["parts"]["merge"]["method"]
    assert json.loads((BD.records(out) / "build.json").read_text())["method"] == doc["method"]
    if change == "method":
        assert doc["method"] == "M-other"
        assert old["record"]["checkpoint"]["identity"]["parts"]["merge"]["method"] == ident["parts"]["merge"]["method"]
        stale = {**doc, "method": ident["parts"]["merge"]["method"]}         #: a record contradicting its weights
        assert any("method" in b for b in BD.product_problems(stale, new)), BD.product_problems(stale, new)


def test_a_handoff_that_does_not_verify_keeps_the_files_and_the_next_attempt_completes_only_it(tmp_path, contract,
                                                                                             monkeypatch):
    """Both groups' files uploaded and verified, but neither handoff verifies (`hand_off_with_retries` answers
    ``verified: false``): the verified records are kept in relay.json, no init record is written and the session's
    completion is INCOMPLETE. The next attempt (the pod lost; its evidence resumed) takes the verified copies over
    and retries ONLY the handoffs: nothing is uploaded, fetched or computed again."""
    from stages.depth_merge import build_driver as BD
    from stages.depth_merge import launch as L
    from stages.depth_merge import session as S
    from stages.reasoning_inheritance.driver import Status
    ident = BD.identity(contract)
    out, work = tmp_path / "out", tmp_path / "work"
    relay = Relay(handoffs=[False] * 4)
    relay.install(monkeypatch, BD)
    steps = Steps()
    steps.install(monkeypatch, BD)
    BD.records(out).mkdir(parents=True)
    with pytest.raises(BD.DriverError, match="verified but not handed off"):
        BD.run(out, work, "cpu", Status(tmp_path / "st"), "r1", ident)
    assert steps.ran[:2] == ["statistics", "weights"]
    assert relay.moved() == [("put", "statistics"), ("put", "init")]
    kept = json.loads((BD.records(out) / "relay.json").read_text())
    assert all(kept[g]["verified"] is True and kept[g]["handoff"]["verified"] is False for g in BD.GROUPS)
    assert BD.durable_init(out, ident) is None and not (out / BD.ARM / "init_record.json").exists()
    assert "RELAY_HANDOFF_PENDING" in (tmp_path / "st").read_text()
    scr = tmp_path / "scr"
    (scr / "store").mkdir(parents=True)
    (scr / "store/out").symlink_to(out)
    monkeypatch.setattr(BD, "current_identity", lambda work: ident)
    assert S.completion(L.VARIANT, L.BUILD_KIND, None, "r1", scr, root=tmp_path)["status"] == "INCOMPLETE"
    #: a record with a pending handoff is not durable, even where everything else is
    doc = BD.init_record(json.loads((BD.records(out) / "build.json").read_text()), {g: kept[g] for g in BD.GROUPS},
                         ident)
    assert any("not handed off" in b for b in BD.product_problems(doc, ident))
    relay.calls.clear()
    steps.ran.clear()
    for step in ("open_gpu", "statistics_step", "weights_step", "diagnostics_step"):
        monkeypatch.setattr(BD, step, _no_gpu)
    rc, rep, out2 = _main(BD, tmp_path, contract, monkeypatch, out)
    assert rc == 0, rep.get("error")
    assert not relay.moved()
    assert [c for c in relay.calls if c[0] == "handoff"] == [("handoff", "statistics"), ("handoff", "init")]
    doc = BD.durable_init(out2, ident)
    assert doc is not None and all(BD.handed_off(doc["relay"][g]) for g in BD.GROUPS)


@pytest.mark.parametrize("left", ["no build record", "the statistics only"])
def test_products_published_before_their_latest_build_record_came_back_are_restored_from_the_relay(
        tmp_path, contract, monkeypatch, left):
    """The real sync window: on the pod the statistics and the weights were uploaded, verified and their records
    published beside them, then the pod was lost in the diagnostics -- before the dev box received the build record
    that names them (it has none, or the one from before the weights). The next attempt restores each product's
    record from its relay record (under this identity, its files exactly the description's), fetches the weights
    only for the diagnostics, and builds and uploads nothing."""
    from stages.depth_merge import build_driver as BD
    from stages.reasoning_inheritance.driver import Status
    ident = BD.identity(contract)
    out, work = tmp_path / "pod_out", tmp_path / "pod_work"
    relay = Relay()
    relay.install(monkeypatch, BD)
    Steps(lost_in_diagnostics=True).install(monkeypatch, BD)
    BD.records(out).mkdir(parents=True)
    with pytest.raises(SystemExit, match="lost"):
        BD.run(out, work, "cpu", Status(tmp_path / "st"), "r1", ident)
    on_pod = json.loads((BD.records(out) / "build.json").read_text())
    assert {r["group"] for r in relay.listed} == set(BD.GROUPS)            #: both published on the relay
    ev = tmp_path / "secured"
    BD.records(ev).mkdir(parents=True)
    if left == "the statistics only":                  #: pulled after the statistics, before the weights
        early = {k: on_pod[k] for k in ("started_utc", "run_id", "arm", "method", "method_identity", "statistics",
                                        "attempts", "root")}
        (BD.records(ev) / "build.json").write_text(json.dumps({**early, "checks": {"base": on_pod["checks"]["base"]}}))
        (BD.records(ev) / f"{M.STATS}.safetensors").write_bytes(STATS_BYTES)
    else:
        (ev / "report.json").write_text("{}")
    relay.calls.clear()
    steps = Steps()
    steps.install(monkeypatch, BD)
    for step in ("statistics_step", "weights_step"):
        monkeypatch.setattr(BD, step, _no_gpu)
    rc, rep, out2 = _main(BD, tmp_path, contract, monkeypatch, ev)
    assert rc == 0, rep.get("error")
    assert relay.moved() == [("get", "init")]                    #: the weights, for the diagnostics only
    assert steps.ran == [("diagnostics", on_pod["checkpoint"]["files_sha256"]["model.safetensors"])]
    doc = BD.durable_init(out2, ident)
    assert doc is not None and doc["checkpoint"] == on_pod["checkpoint"]
    assert doc["statistics"]["tensors_sha256"] == on_pod["statistics"]["tensors_sha256"]
    assert any("restored from the record beside their verified relay copy" in a for a in rep["actions"])


def test_a_fresh_build_puts_each_group_once_and_a_refused_copy_writes_no_record(tmp_path, contract, monkeypatch):
    from stages.depth_merge import build_driver as BD
    from stages.reasoning_inheritance.driver import Status
    ident = BD.identity(contract)
    out, work = tmp_path / "out", tmp_path / "work"
    _secured(out, ident)                                  #: this attempt's own records, its files on the pod
    (work / "products/init").mkdir(parents=True)
    (work / "products/init/model.safetensors").write_bytes(W_BYTES)
    (work / "products/init/config.json").write_bytes(CFG_BYTES)
    relay = Relay()
    relay.install(monkeypatch, BD)
    monkeypatch.setattr(BD, "open_gpu", _no_gpu)
    monkeypatch.setattr(BD, "_put", lambda p, t: {"verified": False, "refused": "quota"})
    with pytest.raises(BD.DriverError):
        BD.run(out, work, "cpu", Status(tmp_path / "st"), "r1", ident)
    assert BD.durable_init(out, ident) is None and not (out / BD.ARM / "init_record.json").exists()
    monkeypatch.setattr(BD, "_put", relay.put)
    BD.run(out, work, "cpu", Status(tmp_path / "st"), "r1", ident)
    assert [c for c in relay.calls if c[0] == "put"] == [("put", "statistics"), ("put", "init")]
    assert BD.durable_init(out, ident)
    (out / BD.ARM / "init_record.json").unlink()                          #: a takeover: verified copies stay
    relay.calls.clear()
    BD.run(out, work, "cpu", Status(tmp_path / "st"), "r1", ident)
    assert not [c for c in relay.calls if c[0] == "put"]
    from stages.depth_merge import launch as L
    from stages.depth_merge import session as S
    scr = tmp_path / "scr"
    (scr / "store").mkdir(parents=True)
    (scr / "store/out").symlink_to(out)
    monkeypatch.setattr(BD, "current_identity", lambda work: ident)
    res = S.completion(L.VARIANT, L.BUILD_KIND, None, "r1", scr, root=tmp_path)
    assert res["status"] == "COMPLETE"
    monkeypatch.setattr(BD, "current_identity", lambda work: {**ident, "checkpoint": "0" * 64})
    assert S.completion(L.VARIANT, L.BUILD_KIND, None, "r1", scr, root=tmp_path)["status"] == "INCOMPLETE"


def test_a_build_takeover_asks_the_relay_only_for_what_the_chain_still_adds(contract, monkeypatch):
    """The launch gate's room for a build (`session.build_need`, the real `build_driver.held` and
    `storage.remaining`) against a simulated relay whose live headroom already excludes what it holds: a first run
    asks for the whole chain; an attempt holding the statistics, or both products (handed off and listed, or verified
    with the handoff pending in the resumed relay.json), asks for the rest at the chain's own count, with the same
    margin. Products under another identity, or whose files are gone, are not held."""
    from stages.depth_merge import build_driver as BD
    from stages.depth_merge import session as S
    from stages.depth_merge import storage as STO
    from stages.teacher_parent_initialization import train_driver as TD
    ident = BD.identity(contract)
    seq = STO.sequence()["relay"]
    count = STO.counted(("statistics", "init"))
    rec = {"statistics": {"tensors_sha256": "d" * 64, "saved_utc": "u", "identity": BD.stamp(ident, "statistics")},
           "checks": {"equivalence": {"passed": True, "worst": 0.0}}, "candidate": {},
           "checkpoint": _checkpoint("d" * 64, ident)}
    stats_only = {"statistics": rec["statistics"]}
    pub = {g: TD._pointer_body(_verified(g, rec, ident)) for g in BD.GROUPS}

    def need(listed=(), build=None, prev=None, present=lambda r: True):
        monkeypatch.setattr(BD, "_listing", Relay(listed=listed).listing)
        return S.build_need(ident, build or {}, prev or {}, present)
    first = need()
    assert first["need_bytes"] == seq["need_bytes"] and first["held"] == []
    capacity = first["need_bytes"]                                     #: the first run fits exactly
    assert capacity >= first["need_bytes"] and not capacity - 1 >= first["need_bytes"]
    s = need(listed=[pub["statistics"]])
    assert s["held"] == ["statistics"] and s["need_bytes"] == seq["need_bytes"] - count["statistics"]
    both = need(listed=list(pub.values()))
    assert both["held"] == ["init", "statistics"]
    assert both["need_bytes"] == seq["need_bytes"] - count["statistics"] - count["init"]
    assert both["margin_bytes"] == first["margin_bytes"]
    #: the brief's case: the first run's headroom was exactly the need; the build's products went in since
    assert capacity - count["statistics"] - count["init"] >= both["need_bytes"]
    assert need(build=stats_only, listed=[pub["statistics"]])["held"] == ["statistics"]
    pending = {g: _verified(g, rec, ident, handed_off=False) for g in BD.GROUPS}
    assert need(build=rec, prev=pending)["held"] == ["init", "statistics"]                 #: handoff pending
    assert need(build=rec, prev=pending, present=lambda r: False)["held"] == []           #: its files gone
    other = copy.deepcopy(ident)
    other["statistics"] = other["checkpoint"] = "f" * 64
    alien = {g: {**p, "identity_sha256": other[BD.WHAT[g]]} for g, p in pub.items()}
    assert need(listed=list(alien.values()))["held"] == []
    monkeypatch.setattr(BD, "_listing", lambda prefix: (_ for _ in ()).throw(BD.DriverError("unreadable")))
    with pytest.raises(BD.DriverError):                       #: unknown is not empty: the gate refuses
        S.build_need(ident, {}, {}, lambda r: True)


def test_the_span_diagnostic_moves_only_the_kept_blocks_to_the_teachers_device(teacher):
    from stages.depth_merge import build_driver as BD
    adapter, spans, base, _digests = _toy(teacher)
    merged = copy.deepcopy(base)
    cands = BD.span_candidates(adapter, base, merged, TOY_SEL, spans, "meta")
    assert set(cands) == {"kept", "merged"} and set(cands["kept"]) == {1}
    assert all(p.device.type == "meta" for p in cands["kept"][1].parameters())
    blocks = adapter.blocks(base)
    assert all(p.device.type == "cpu" for i in (0, 2) for p in blocks[i].parameters())
    assert all(p.device.type == "cpu" for p in adapter.embedding(base).parameters())
    assert cands["merged"][1] is adapter.blocks(merged)[1]


# ---- the comparison's rule ----------------------------------------------------------------------------------------

def _y(acc: dict, items, seeds):
    """``{mode: {seed: {item: 0/1}}}`` with the first round(acc * n) items correct."""
    out = {}
    for mode, a in acc.items():
        k = round(a * len(items))
        out[mode] = {s: {i: int(j < k) for j, i in enumerate(items)} for s in seeds}
    return out


def test_each_comparison_is_its_own_fixed_sequence_family_and_formal_only_when_complete():
    from stages.depth_merge import analysis as A
    items = [f"i{j}" for j in range(80)]
    groups = {f"g{j}": [items[2 * j], items[2 * j + 1]] for j in range(40)}
    seeds = [1, 2]
    better = _y({"H-T": 0.6, "H-N": 0.6}, items, seeds)
    worse = _y({"H-T": 0.2, "H-N": 0.2}, items, seeds)
    res = A.compare(better, worse, groups, seeds, ("a", "b"))
    ht, hn = res["modes"]["H-T"], res["modes"]["H-N"]
    assert ht["status"] == hn["status"] == "formal" and ht["verdict"] == "a_BETTER" and hn["verdict"] == "a_BETTER"
    assert ht["seeds_agree_in_sign"] and set(ht["per_seed"]) == {"1", "2"}
    #: the thinking test does not reject: non-thinking is descriptive, never tested
    same = A.compare(worse, worse, groups, seeds, ("a", "b"))
    assert same["modes"]["H-T"]["verdict"] == "NOT_DISTINGUISHED" and same["modes"]["H-N"]["status"] == "descriptive_only"
    assert "p_two_sided" not in same["modes"]["H-N"]
    #: a seed missing: descriptive in every mode
    part = copy.deepcopy(better)
    del part["H-T"][2]
    miss = A.compare(part, worse, groups, seeds, ("a", "b"))
    assert miss["modes"]["H-T"]["status"] == miss["modes"]["H-N"]["status"] == "descriptive_only"


def test_the_comparison_registry_names_the_three_direct_questions():
    from stages.depth_merge import analysis as A
    c = A.comparisons()
    assert c["C1"]["pair"] == ["depth_merge", "depth_only"] and c["C3"]["pair"] == ["depth_loop_hidden", "depth_loop"]
    assert A.result_path("C1").parent == DC.LOGS / "results"


# ---- the plan -----------------------------------------------------------------------------------------------------

def merge_init_record(root: Path) -> Path:
    """A COMPLETE variant-1 init record as the build writes it (`build_driver.init_record`, under this tree's identity
    of the build's products), at a committed-like path under ``root``."""
    import tempfile
    from stages.depth_merge import build_driver as BD
    base = json.loads((REPO / DC.design()["baseline"]["init_record"]["path"]).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        ident = BD.current_identity(Path(tmp))
    w, stats = "b" * 64, "d" * 64
    cfg = DC.design()["baseline"]["init_record"]["config_json_sha256"]
    rec = {"run_id": "dmr1_merge_build_x", "root": base["root_teacher"], "method": M.METHOD,
           "method_identity": {}, "statistics": {"tensors_sha256": stats, "saved_utc": "u",
                                                 "identity": BD.stamp(ident, "statistics")},
           "checks": {"equivalence": {"passed": True, "worst": 1e-6}}, "candidate": {},
           "checkpoint": {"artifact_digest": "c" * 64, "statistics_sha256": stats, "reload_forward_finite": True,
                          "shards": [{"filename": "model.safetensors", "sha256": w, "size_bytes": 4007724000}],
                          "files_sha256": {"model.safetensors": w, "config.json": cfg},
                          "num_parameters_counted": 2003851776, "num_parameters": 2003851776,
                          "identity": BD.stamp(ident, "checkpoint")}}
    relay = {"init": {"relay": "AlphaAvatar/aadistill-artifacts", "prefix": "durable/stage1_depth_merge/depth_merge/init/cccc",
                      "files_sha256": {"init/model.safetensors": w, "init/config.json": cfg},
                      "bytes": {"init/model.safetensors": 4007724000, "init/config.json": 1154}, "verified": True,
                      "identity_sha256": ident["checkpoint"], "handoff": {"verified": True}},
             "statistics": {"relay": "AlphaAvatar/aadistill-artifacts",
                            "prefix": "durable/stage1_depth_merge/depth_merge/statistics/dddd",
                            "files_sha256": {f"statistics/{M.STATS}.safetensors": stats}, "verified": True,
                            "identity_sha256": ident["statistics"], "handoff": {"verified": True}}}
    doc = BD.init_record(rec, relay, ident)
    assert BD.product_problems(doc, ident) == []
    p = root / "logs/stages/stage-1/depth_merge/runs/dmr1_merge_build_x/evidence/depth_merge/init_record.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    return p


def link_into(root: Path, rel: str) -> None:
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.symlink_to(REPO / rel)


def test_the_merge_arms_plan_is_sob1s_depth_unit_outside_the_declared_fields(tmp_path):
    from stages.depth_merge import plan as PL
    link_into(tmp_path, DC.design()["baseline"]["training_plan"]["path"])
    doc = PL.derive("depth_merge", merge_init_record(tmp_path), root=tmp_path)
    assert [u["unit"] for u in doc["units"]] == ["depth_merge-s20260726", "depth_merge-s20260801"]
    for u in doc["units"]:
        c = u["config"]
        assert c["stage"] == "stage3_recovery" and "hidden_state" not in c["loss"]
        assert c["lineage"]["initialization"]["model_safetensors_sha256"] == "b" * 64
        assert set(doc["derivation_checks"][u["unit"]]["differs_from_sob1_depth_unit"]) <= {
            "run_name", "student_path", "out_dir", "lineage.initialization.model_safetensors_sha256",
            "lineage.initialization.provenance.record", "lineage.initialization.provenance.record_sha256"}
        assert c["schedule"]["total_steps"] == 9701 and c["block_len"] == 8192
    assert doc["relay_prefix_root"] == "durable/stage1_depth_merge"
    bad = json.loads(merge_init_record(tmp_path).read_text())
    bad["relay"]["init"]["verified"] = False
    p = tmp_path / "logs/bad_init.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(PL.PlanError):
        PL.derive("depth_merge", p, root=tmp_path)


# ---- the launcher -------------------------------------------------------------------------------------------------

def test_the_launchers_kinds_grants_and_an_artifact_for_another_arm_is_refused(tmp_path, monkeypatch, capsys):
    from stages.depth_merge import launch as L
    from stages.depth_merge import session as S
    from stages.depth_recurrence import launch as L6
    assert L.STUDY.kinds == ("dmr1_merge_build", "dmr1_merge_train") and L.STUDY.formal
    assert L6.STUDY.kinds == ("dmr1_loop_train", "dmr1_loop_hidden_train")
    assert {k: str(g) for k, g, _ in L.STUDY.kind_grants} == {
        "dmr1_merge_build": "logs/budget/approvals/dmr1_dmr1_merge_build_grant.json",
        "dmr1_merge_train": "logs/budget/approvals/dmr1_dmr1_merge_train_grant.json"}
    cond = L.kind_conditions(L.BUILD_KIND)
    assert cond["digests_sha256"] == sha(M.DIGESTS) and "identity:configs/stages/stage-1/depth_merge/design.json" in cond
    #: variant 1's two kinds are granted (maintainer, 2026-11-27) exactly as proposed: the caps and the reserves
    for k, cap in ((L.BUILD_KIND, 5.43), (L.TRAIN_KIND, 64.46)):
        g = S.kind_grant(L.VARIANT, k)
        prop = json.loads((REPO / L.VARIANT.proposal_path(k)).read_text())
        assert g["status"] == "granted" and g["ceiling_usd"] == g["per_session_hard_cap_usd"] == cap
        assert g["teardown_reserve_usd"] == 1.0 and g["project_cumulative_cap_usd"] == 1161.06
        assert g["kinds"] == prop["kinds"] and g["gpus"] == prop["gpus"] == {"NVIDIA L40S": 1.11}
        assert g["proposal"]["path"] == str(L.VARIANT.proposal_path(k))
    #: the loop kinds have no proposal this round: budget refuses
    with pytest.raises(S.SE.AuthorizationError):
        S.kind_grant(L6.VARIANT, L6.LOOP_KIND, dry_run=True)
    with pytest.raises(S.SE.AuthorizationError, match="no grant and no proposal"):
        L6.launch_spec(L6.LOOP_KIND)
    assert L.launch_spec(L.BUILD_KIND).driver.endswith("build_driver.py")
    #: an artifact naming another arm's plan is refused before anything is read
    sc = {"kind": L.TRAIN_KIND, "arm": "depth_merge", "plan": "x", "plan_sha256": "y", "init_record_sha256": "z"}
    monkeypatch.setattr(S, "kind_conditions", lambda v, kind, root=REPO, arm=None: {**sc, "plan_sha256": "other"})
    assert S.arm_binding_problems(L.VARIANT, sc, "depth_merge")
