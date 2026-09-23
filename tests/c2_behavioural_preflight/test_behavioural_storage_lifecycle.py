"""The storage derivation and the probe-local lifecycle, after attempt5's ENOSPC.

attempt5 trained ten of twelve probes and then died in the trainer's
`save_checkpoint` with `No space left on device`, losing probe 11's compute and
the campaign's verdict. Three separate defects, each with its own tests here:

1. **the model charged a trained probe at the size of the leaf it started
   from.** `total_probes * leaf_gib` -- and the frozen leaves are bf16 while
   the recipe declares `dtype: "float32"`, so every retained probe was charged
   at exactly half its real size. The storage model contradicted the config it
   was running under.
2. **nothing released a probe's workdir.** `announce_durable` only announces,
   and it runs BEFORE the transfer, so it cannot authorize a release; there was
   no later point that could. Twelve workdirs accumulated.
3. **container storage and durable capacity were one quantity.** The bytes that
   must merely survive teardown were charged to the pod's own disk.

The lesson underneath all three is that a derivation can be wrong in the one
direction that costs a session, so the driver also MEASURES before each probe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.runtime import cost as COST  # noqa: E402

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_continuation as BC  # noqa: E402

import autoinit_c2_behavioural_launch as L  # noqa: E402
import autoinit_c2_behavioural_driver as D  # noqa: E402


# --- 1. the generic byte model takes every dtype as an argument -------------

class TestTheGenericByteModel:
    """It lives in `aadistill.runtime.cost` and knows nothing about C2."""

    #: The maintainer's hard rule, as a pattern table rather than a word list:
    #: generic core carries no experiment-instance or model-instance fact. A
    #: current experiment may PASS all of these in from its config, protocol
    #: or application layer; none of them may be written down here.
    INSTANCE_FACTS = {
        "experiment names": r"\b(phase_c[0-9]|c2_behavioural|attempt[0-9]|behavioural)\b",
        "model family": r"\b([Qq]wen|llama|mistral)\b",
        "a parameter count": r"\b\d{3}[,_]\d{3}[,_]\d{3}\b",
        "config or recipe paths": r"configs/[a-z0-9_/]+\.json",
        "candidate ids": r"\b[0-9a-f]{32}\b",
        "seeds or batteries": r"\b(616738081|1936324010|1916380711|1523147638"
                              r"|834816710|screening_v1|confirmation_v1)\b",
        "budgets": r"\$\d+\.\d{2}",
        "provider paths": r"(/workspace|runpod|aad-artifacts)",
        "CUDA ordinals": r"cuda:\d",
        "compression ratios": r"\b0\.86M\b",
    }

    def test_no_experiment_or_model_facts_are_encoded(self):
        """Including in COMMENTS. Provenance belongs to the experiment's logs.

        A comment naming the run that motivated a line is tempting and is
        still an instance fact in generic code -- it dates the module to one
        experiment and invites the next one to add its own.
        """
        import re

        src = (REPO / "src/aadistill/runtime/cost.py").read_text()
        found = {label: sorted({m.group(0) for m in re.finditer(pat, src)})
                 for label, pat in self.INSTANCE_FACTS.items()}
        found = {k: v for k, v in found.items() if v}
        assert not found, (
            f"generic runtime code encodes instance facts: {found}. A current "
            "experiment may pass these in from its config, protocol or "
            "application layer; it may not write them down here.")

    def test_every_new_symbol_takes_its_facts_as_arguments(self):
        """Generic means parameterized, not merely free of names.

        Each of the seven added symbols must accept the model's size, its
        dtypes and its policies rather than reading them from anywhere.
        """
        import inspect

        for name in ("bytes_per_param", "training_working_set_bytes",
                     "peak_local_residency_bytes", "durable_backend_bytes"):
            fn = getattr(COST, name)
            assert inspect.signature(fn).parameters, name
        for cls, fields in (
                (COST.CheckpointFootprint,
                 {"num_parameters", "save_dtype", "extra_bytes"}),
                (COST.TrainerCheckpointFootprint,
                 {"num_parameters", "trainable_parameters", "save_dtype",
                  "moment_dtype", "n_moments", "extra_bytes"}),
                (COST.ResidencyUnit,
                 {"label", "durable_bytes", "retained_bytes",
                  "transient_bytes", "released_on_completion",
                  "materializing_bytes"})):
            have = set(inspect.signature(cls).parameters)
            assert fields <= have, (cls.__name__, sorted(fields - have))

    @pytest.mark.parametrize("dtype,per_param", [
        ("bf16", 2), ("bfloat16", 2), ("fp32", 4), ("float32", 4),
        ("fp8", 1), ("int4", 0.5),
    ])
    def test_bytes_per_param_covers_both_spellings(self, dtype, per_param):
        assert COST.bytes_per_param(dtype) == per_param

    def test_an_unknown_dtype_raises_rather_than_defaulting(self):
        """A default here is a silent assumption about someone else's
        checkpoint, and the ratio between two dtypes is exactly the factor by
        which the bound would then be wrong."""
        with pytest.raises(ValueError, match="unknown dtype"):
            COST.bytes_per_param("float3")

    def test_the_footprint_reproduces_a_real_probe_on_disk(self):
        """The arithmetic, checked against bytes that actually exist.

        A storage model whose only evidence is itself is what this replaces.
        """
        store = Path("/home/ecs-user/aad-artifacts/phase_c2_behavioural/"
                     "c2-behavioural-12probe-v1/attempt5")
        probes = sorted(store.glob("*/model.safetensors")) if store.is_dir() else []
        if not probes:
            pytest.skip("no real probe on this host to check the model against")
        params = BH.candidate_manifest(REPO)[0]["num_parameters"]
        tr = BH.training_dtypes(REPO)
        want = COST.CheckpointFootprint(params, tr["save_dtype"]).bytes
        got = probes[0].stat().st_size
        assert abs(got - want) < 1 << 20, (
            f"a real probe is {got:,} bytes and the model derives {want:,} "
            f"from {tr['save_dtype']}; the save dtype is not what is declared")

    def test_release_decides_whether_retention_accumulates(self):
        """`released_on_completion` is a statement about the CODE.

        With it True nothing accumulates and the peak is one transient above
        the floor; with it False every unit's retention stacks. A bound derived
        with True while no call site frees anything describes a program that
        does not exist -- which is the bound attempt5 ran under.
        """
        ck = COST.CheckpointFootprint(600_000_000, "float32")
        mk = lambda rel: [COST.ResidencyUnit(f"u{i}", ck.bytes, 2 * ck.bytes,
                                             ck.bytes, rel) for i in range(12)]
        freed = COST.peak_local_residency_bytes(mk(True), fixed_bytes=0)
        kept = COST.peak_local_residency_bytes(mk(False), fixed_bytes=0)
        assert kept["peak_bytes"] > freed["peak_bytes"]
        assert freed["peak_at_unit"] == "u0"          # nothing accumulates
        assert kept["peak_at_unit"] == "u11"          # the last one overflows
        assert freed["accumulated_retained_bytes"] == 0

    def test_durable_is_not_the_container_bound(self):
        ck = COST.CheckpointFootprint(600_000_000, "float32")
        units = [COST.ResidencyUnit(f"u{i}", ck.bytes, 2 * ck.bytes, ck.bytes,
                                    True) for i in range(12)]
        assert COST.durable_backend_bytes(units)["bytes"] == 12 * ck.bytes
        #: and the container peak does not include them
        assert COST.peak_local_residency_bytes(
            units, fixed_bytes=0)["peak_bytes"] < 12 * ck.bytes


# --- 2. the experiment layer READS its dtypes ------------------------------

class TestTheDtypesAreRead:
    def test_they_come_from_the_recipe_the_driver_derives_probes_from(self):
        tr = BH.training_dtypes(REPO)
        recipe = json.loads((REPO / BH.FROZEN_RECIPE_REL).read_text())
        assert tr["save_dtype"] == str(recipe["dtype"]).lower()
        assert tr["n_moments"] == len(recipe["optim"]["betas"])
        assert tr["keep_last"] == recipe["checkpoint"]["keep_last"]
        #: and it is the SAME file the driver builds probe configs from
        import autoinit_c1_driver as C1D
        assert Path(C1D.FROZEN_RECIPE).name == Path(BH.FROZEN_RECIPE_REL).name

    def test_a_recipe_with_an_unreadable_dtype_refuses(self, tmp_path):
        recipe = json.loads((REPO / BH.FROZEN_RECIPE_REL).read_text())
        recipe["dtype"] = "float3"
        p = tmp_path / BH.FROZEN_RECIPE_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(recipe))
        with pytest.raises(BH.BehaviouralProposalError, match="not a per-parameter"):
            BH.training_dtypes(tmp_path)

    def test_a_recipe_with_no_betas_refuses(self, tmp_path):
        recipe = json.loads((REPO / BH.FROZEN_RECIPE_REL).read_text())
        recipe["optim"].pop("betas")
        p = tmp_path / BH.FROZEN_RECIPE_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(recipe))
        with pytest.raises(BH.BehaviouralProposalError, match="optim.betas"):
            BH.training_dtypes(tmp_path)


# --- 3. the storage requirement charges the save footprint -----------------

def test_the_probe_is_charged_at_its_save_size_not_its_leaf_size():
    sched = BH.schedule(REPO)
    req = BH.storage_requirement(BH.candidate_manifest(REPO), sched, REPO)
    fp = req["components_gib"]["_probe_checkpoint_footprint"]
    leaf_gib = BH.candidate_manifest(REPO)[0]["bytes"] / 2**30
    assert fp["gib"] > leaf_gib * 1.9, (
        "the trained probe is charged at about the leaf size again; the leaves "
        "are bf16 and the recipe saves float32")
    assert req["components_gib"]["trained_probe_checkpoints"] == pytest.approx(
        int(sched["total_probes"]) * fp["gib"], abs=0.01)


def test_container_and_durable_are_separate_bounds():
    req = BH.storage_requirement(BH.candidate_manifest(REPO),
                                 BH.schedule(REPO), REPO)
    c, d = req["container_residency"], req["durable_backend"]
    assert c["peak_gib"] > 0 and d["gib"] > 0
    assert c["released_on_completion"] is True
    #: the durable bytes are NOT inside the container peak
    assert c["peak_gib"] < req["subtotal_gib"]
    assert d["gib"] == pytest.approx(
        req["components_gib"]["trained_probe_checkpoints"], abs=0.01)


# --- 4. both gates charge the right resource -------------------------------

def test_the_destination_gate_charges_the_derived_durable_requirement():
    src = (REPO / "scripts/pod/autoinit_c2_behavioural_launch.py").read_text()
    body = src.split("def destination_gate(", 1)[1].split("\ndef ")[0]
    assert "durable_backend" in body
    #: The old EXPRESSION, not the string "1.11" -- that appears in the comment
    #: explaining what was removed, and a test that cannot tell code from the
    #: prose describing it fails on a correct repair.
    import re
    code = "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#:"))
    assert not re.search(r"int\(\s*1\.11\s*\*", code), (
        "the hardcoded 1.11 GiB expression is back; it is the bf16 LEAF size "
        "and the probes are saved in float32")


class TestTheContainerGate:
    class _Ctx:
        def __init__(self, gb):
            self.args = type("A", (), {"disk_gb": gb})()
            self.evidence = {}

    def test_it_passes_the_authorized_provision(self):
        ok, why = L.container_gate(self._Ctx(120))
        assert ok, why

    def test_it_refuses_a_provision_that_cannot_hold_the_work(self):
        ctx = self._Ctx(48)
        ok, why = L.container_gate(ctx)
        assert not ok and "peak local residency" in why
        assert ctx.evidence["container_storage"]["headroom_gib"] < 0

    def test_it_reads_the_flag_rather_than_the_constant(self):
        """A session that overrode the flag must be checked against what it
        asked for, not against what the module's default says."""
        big = L.container_gate(self._Ctx(400))[0]
        small = L.container_gate(self._Ctx(8))[0]
        assert big and not small

    def test_it_is_wired_into_the_prechecks(self):
        src = (REPO / "scripts/pod/autoinit_c2_behavioural_launch.py").read_text()
        assert "container_gate," in src.split("prechecks", 1)[-1][:900] or \
               "container_gate," in src
        #: and it sits beside the destination gate it complements
        i, j = src.index("destination_gate,\n"), src.index("container_gate,\n")
        assert 0 < j - i < 40


# --- 5. the probe-local lifecycle has a real acknowledgement boundary ------

class TestTheReleaseBoundary:
    """`announce_durable` cannot authorize a release: it runs BEFORE the
    transfer, so at announcement time the pod's copy is the only one."""

    def test_the_ack_path_is_one_shared_constant(self):
        assert L.RELEASE_ACK_DIR.endswith(BC.RELEASE_ACK_REL), (
            "the launcher writes the ack somewhere the driver does not read; "
            "two spellings of one path is how a checker verifies nothing")
        dsrc = (REPO / "scripts/pod/autoinit_c2_behavioural_driver.py").read_text()
        assert "BC.RELEASE_ACK_REL" in dsrc

    def test_the_launcher_acks_only_after_destination_re_identification(self):
        src = (REPO / "scripts/pod/autoinit_c2_behavioural_launch.py").read_text()
        block = src.split("if matched:", 1)[1].split("\n    return fetched")[0]
        assert "durable_ack.json" in block and "release_ack(ctx" in block, (
            "the release ack is not inside the matched branch; it would permit "
            "a release for bytes that did not verify")

    def _driver(self, tmp_path, training, acked):
        drv = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
        drv.ev = {}
        drv.training = {}
        for name in training:
            d = tmp_path / "work" / name
            (d / "model").mkdir(parents=True, exist_ok=True)
            (d / "model" / "w.bin").write_bytes(b"x" * 4096)
            drv.training[name] = {"out_dir": str(d)}
        ack = tmp_path / BC.RELEASE_ACK_REL
        ack.mkdir(parents=True, exist_ok=True)
        for name in acked:
            (ack / f"{name}.json").write_text("{}")
        return drv

    def test_only_acked_probes_are_released(self, tmp_path, monkeypatch):
        monkeypatch.setattr(D, "REPO", tmp_path)
        drv = self._driver(tmp_path, ["p1", "p2", "p3"], ["p1", "p3"])
        out = drv.release_acked_probe_workdirs()
        assert out["released"] == ["p1", "p3"]
        assert out["kept"] == ["p2"]
        assert not (tmp_path / "work" / "p1").exists()
        assert (tmp_path / "work" / "p2" / "model" / "w.bin").is_file(), (
            "an unacked probe was released; its bytes may be the only copy")

    def test_a_path_outside_the_pods_workspace_is_never_deleted(
            self, tmp_path, monkeypatch):
        """A pod mounts a volume holding this campaign's pre-staged probes.

        They are shared with every future attempt of the campaign and, for
        those bytes, the only copy that is not on the launcher host. This loop
        deletes directories named by journal entries, and a RESTORED probe's
        entry points at the volume: it carries no `out_dir` today, so nothing
        reaches it — but that is a property of one dictionary literal in
        `restore_campaign`, which is too thin a thing to stand between an
        `rmtree` and every later attempt's probes.

        Acked, present, and outside. It must be refused, and refused LOUDLY:
        `failed` is what makes the caller stop, and a silent skip would leave
        the storage bound looking satisfied.
        """
        monkeypatch.setattr(D, "REPO", tmp_path)
        drv = self._driver(tmp_path, ["p1"], ["p1", "restored"])
        volume = tmp_path.parent / "durable" / "restored"
        volume.mkdir(parents=True, exist_ok=True)
        (volume / "model.safetensors").write_bytes(b"y" * 4096)
        drv.training["restored"] = {"out_dir": str(volume)}

        out = drv.release_acked_probe_workdirs()
        assert (volume / "model.safetensors").is_file(), (
            "the shared volume's copy was deleted by a pod")
        assert out["released"] == ["p1"]
        #: REPORTED, not FAILED: `failed` is what stops the session, and it
        #: means the LOCAL storage bound was falsified. An external path was
        #: never part of that bound, so refusing it must not end a run.
        assert out["failed"] == []
        assert any("outside this session's workspace" in f
                   for f in out["refused_outside_workspace"]), out

    def test_an_unacked_probe_is_not_an_error(self, tmp_path, monkeypatch):
        """The launcher pulls asynchronously, so a probe that finished moments
        ago legitimately has no ack yet."""
        monkeypatch.setattr(D, "REPO", tmp_path)
        drv = self._driver(tmp_path, ["p1"], [])
        out = drv.release_acked_probe_workdirs()
        assert out == {"released": [], "failed": [], "freed_gib": 0.0,
                       "kept": ["p1"], "refused_outside_workspace": []}

    def test_a_failed_release_fails_closed(self, tmp_path, monkeypatch):
        """The storage bound assumes the release happened; if it did not, no
        further probe may be trained under a bound that no longer holds."""
        import shutil
        monkeypatch.setattr(D, "REPO", tmp_path)
        drv = self._driver(tmp_path, ["p1"], ["p1"])
        monkeypatch.setattr(
            shutil, "rmtree",
            lambda *a, **k: (_ for _ in ()).throw(OSError("device busy")))
        with pytest.raises(D.C2DriverError, match="NO FURTHER PROBE MAY BE TRAINED"):
            drv.release_acked_probe_workdirs()
        #: the bytes survive the refusal, and the failure is recorded
        assert (tmp_path / "work" / "p1" / "model" / "w.bin").is_file()
        assert drv.ev["probe_workdirs_released"][-1]["failed"]

    def test_the_release_runs_before_each_probe_is_trained(self):
        src = (REPO / "scripts/pod/autoinit_c2_behavioural_driver.py").read_text()
        body = src.split("def run_rung(", 1)[1].split("\n    def ")[0]
        i = body.index("release_acked_probe_workdirs()")
        j = body.index("self.train_one(")
        assert i < j, "the release happens after training; it buys nothing there"


# --- 6. and the driver MEASURES, because a derivation can be wrong --------

class TestTheRuntimeHeadroomRefusal:
    def _drv(self, tmp_path):
        drv = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
        drv.ev = {}
        drv.a = type("A", (), {"b_workdir": str(tmp_path)})()
        return drv

    def test_the_need_is_derived_from_the_recipe(self, tmp_path):
        need = self._drv(tmp_path).probe_local_need_bytes()
        tr = BH.training_dtypes(REPO)
        assert need["keep_last"] == tr["keep_last"]
        assert need["checkpoint"]["save_dtype"] == tr["save_dtype"]
        assert need["need_bytes"] == (need["transient_bytes"]
                                      + need["retained_bytes"])

    def test_it_refuses_when_the_disk_cannot_hold_the_next_probe(
            self, tmp_path, monkeypatch):
        from aadistill.runtime import leaf_durability as LD
        monkeypatch.setattr(LD, "free_bytes_at", lambda p: 1 << 20)
        drv = self._drv(tmp_path)
        with pytest.raises(D.C2DriverError, match="REFUSING BEFORE training"):
            drv.require_probe_headroom("probe_x")
        assert drv.ev["probe_headroom"][-1]["free_gib"] < 1

    def test_it_passes_and_records_when_there_is_room(self, tmp_path,
                                                      monkeypatch):
        from aadistill.runtime import leaf_durability as LD
        monkeypatch.setattr(LD, "free_bytes_at", lambda p: 400 << 30)
        rec = self._drv(tmp_path).require_probe_headroom("probe_x")
        assert rec["free_gib"] > rec["need_gib"]

    def test_it_runs_before_training_and_after_the_release(self):
        src = (REPO / "scripts/pod/autoinit_c2_behavioural_driver.py").read_text()
        body = src.split("def run_rung(", 1)[1].split("\n    def ")[0]
        assert (body.index("release_acked_probe_workdirs()")
                < body.index("require_probe_headroom(")
                < body.index("self.train_one(")), (
            "measure after releasing and before training, or it measures the "
            "wrong moment")
