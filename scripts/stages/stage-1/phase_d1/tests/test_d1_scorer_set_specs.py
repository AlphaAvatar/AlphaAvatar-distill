"""`battery_sets()` must hand `score_battery` SPECS, not filenames. At `$0`.

THE FAILURE THIS EXISTS FOR. Both D-series scorers built their set mapping as
`{name: f"{name}.jsonl"}` -- a name to FILENAME map -- while the shared
`score_c1_confirmation.score_battery` reads two keys out of each value:

    if len(samples) != spec["n"]: ...
    row.update({..., "domain": spec["domain"], ...})

`score_battery` derives the file path from the KEY, so the filename value was
never read for anything; the first `spec["n"]` raised

    TypeError: string indices must be integers, not 'str'

D1's screening attempt 6 paid 182.8 minutes and $3.3565 to find it, and found
it in the worst possible place: AFTER five arms materialized and verified,
AFTER the generation-protocol attestation passed, AFTER the per-probe
admission passed, and AFTER probe 1 had trained for 66 minutes. The scorer is
the last step of the first probe.

Both D-series scorers carried the identical defect, so fixing screening alone
would have hit the same wall at confirmation.

WHY NOTHING CAUGHT IT. Every test of the scorers tested `battery_sets` or
`score_battery`, never the two together, and the scorers only ever execute as
a pod subprocess. So this runs the REAL `score_battery` over the REAL battery
with the REAL `battery_sets()` output -- a stub for either side would just
re-encode the assumption that broke.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))

SCORERS = ("stages.phase_d1.score_d1_screening",
           "stages.phase_d1.score_d1_confirmation")


def _scorer(module_name: str):
    return importlib.import_module(module_name)


def _battery_root(mod) -> Path:
    identity = importlib.import_module(
        "stages.phase_d1.behavioural").battery_role(mod.ROLE, REPO)
    return REPO / identity["root"]


def _write_generations(battery: Path, sets, dest: Path) -> None:
    """A complete, well-formed `behavior_v0` generation for every real item.

    Ids come from the battery itself, so the set cannot disagree with the
    frozen prompts -- `score_battery` refuses a foreign or short set, and a
    fixture that invented ids would be testing the refusal instead of the
    contract.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in sets:
        items = [json.loads(line)
                 for line in (battery / f"{name}.jsonl").open()
                 if line.strip()]
        with (dest / f"{name}.generations.jsonl").open("w") as fh:
            for item in items:
                fh.write(json.dumps({
                    "id": item["id"],
                    #: `think_preopened` is the evaluator's default, so the
                    #: raw text closes the reasoning block and then answers.
                    "raw": "reasoning here</think>an answer",
                    "natural_termination": True,
                    "degeneration_triggered": False,
                    "context_limit_reached": False,
                }) + "\n")


@pytest.mark.parametrize("module_name", SCORERS)
class TestTheSetMappingIsSpecsNotFilenames:

    def test_every_value_is_a_spec_with_the_two_keys_read(self, module_name):
        mod = _scorer(module_name)
        layout = mod.battery_sets()
        assert layout["sets"], "non-vacuous"
        for name, spec in layout["sets"].items():
            assert isinstance(spec, dict), (
                f"{name} maps to {spec!r}; score_battery reads spec['n'] and "
                "spec['domain'] out of it")
            assert set(spec) >= {"n", "domain"}, (name, sorted(spec))
            assert isinstance(spec["n"], int) and spec["n"] > 0
            assert isinstance(spec["domain"], str) and spec["domain"]

    def test_the_counts_are_the_familys_own_and_match_the_files(
            self, module_name):
        """Derived from `battery_family.strata`, and true of the real files --
        which is what `score_battery`'s length check compares."""
        mod = _scorer(module_name)
        layout = mod.battery_sets()
        battery = _battery_root(mod)
        strata = mod.FAMILY.strata()
        for name, spec in layout["sets"].items():
            on_disk = sum(1 for line in (battery / f"{name}.jsonl").open()
                          if line.strip())
            assert spec["n"] == on_disk, (
                f"{name}: spec says {spec['n']}, the frozen set holds "
                f"{on_disk}; score_battery refuses on exactly this")
            assert spec["n"] == strata[name][1]
            assert spec["domain"] == strata[name][0]

    def test_the_real_scorer_runs_end_to_end_over_the_real_battery(
            self, module_name, tmp_path):
        """THE REGRESSION. Real `score_battery`, real battery, real
        `battery_sets()` -- the combination nothing exercised, which is why a
        dict/string mismatch reached a billing pod."""
        from stages.phase_c1.score_c1_confirmation import score_battery

        mod = _scorer(module_name)
        layout = mod.battery_sets()
        battery = _battery_root(mod)
        gen_dir = tmp_path / "gen"
        _write_generations(battery, layout["sets"], gen_dir)

        scored = score_battery(
            battery=battery, gen_dir=gen_dir, label="regression", seed=1,
            sets=layout["sets"], scorable_sets=layout["scorable_sets"],
            behaviour_only=layout["behaviour_only_sets"])

        total = sum(spec["n"] for spec in layout["sets"].values())
        assert len(scored["rows"]) == total
        assert len(scored["per_sample"]) == total
        #: and the domain travelled from the spec onto every row, which is the
        #: other key the old shape could not supply
        domains = {r["domain"] for r in scored["rows"]}
        assert domains == {s["domain"] for s in layout["sets"].values()}
        assert all(r["domain"] for r in scored["rows"])

    def test_the_old_filename_shape_still_breaks(self, module_name, tmp_path):
        """MUTATION, so the test above cannot pass for the wrong reason: feed
        `score_battery` exactly the mapping the scorers used to build and
        require it to fail. If this ever stops raising, `score_battery` has
        changed its contract and the fix above needs revisiting rather than
        silently becoming decorative."""
        from stages.phase_c1.score_c1_confirmation import score_battery

        mod = _scorer(module_name)
        layout = mod.battery_sets()
        battery = _battery_root(mod)
        gen_dir = tmp_path / "gen"
        _write_generations(battery, layout["sets"], gen_dir)

        old_shape = {name: f"{name}.jsonl" for name in layout["sets"]}
        with pytest.raises(TypeError, match="string indices"):
            score_battery(
                battery=battery, gen_dir=gen_dir, label="mutation", seed=1,
                sets=old_shape, scorable_sets=layout["scorable_sets"],
                behaviour_only=layout["behaviour_only_sets"])

    def test_the_source_no_longer_builds_a_name_to_filename_map(
            self, module_name):
        """The structural fact, pinned in both scorers."""
        path = REPO / (module_name.replace(".", "/")
                       .replace("stages/phase_d1",
                                "scripts/stages/stage-1/phase_d1") + ".py")
        src = path.read_text()
        assert 'sets = {name: f"{name}.jsonl"' not in src, (
            f"{path.name} builds the filename map again")
        assert '"n": n' in src and '"domain": domain' in src


class TestTheDriverReadsOnlyKeysTheScorerWrites:
    """The same contract from the other side, and the next one that would
    have cost a pod.

    After the scorer returns, the driver reads five keys out of its output
    JSON and puts them on the probe record that `rank_screening` consumes. A
    key the scorer does not write is a `KeyError` on the LAST step of the
    first probe -- indistinguishable, from the pod's point of view, from the
    dict/string defect above. So this runs the real scorer over the real
    battery and checks every key the driver actually reads.

    Derived by parsing the driver rather than listing the keys here: a list
    would go stale the first time the driver read a sixth key, which is the
    failure mode it exists to prevent.
    """

    @staticmethod
    def _keys_the_driver_reads() -> list[str]:
        import re

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py").read_text()
        return sorted(set(re.findall(r'scored\["([a-z_0-9]+)"\]', src))
                      | set(re.findall(r'scored\.get\("([a-z_0-9]+)"', src)))

    def test_every_key_is_written_by_the_real_scorer(self, tmp_path):
        from stages.phase_c1.score_c1_confirmation import score_battery

        mod = _scorer("stages.phase_d1.score_d1_screening")
        layout = mod.battery_sets()
        battery = _battery_root(mod)
        gen_dir = tmp_path / "gen"
        _write_generations(battery, layout["sets"], gen_dir)
        scored = score_battery(
            battery=battery, gen_dir=gen_dir, label="keycheck", seed=1,
            sets=layout["sets"], scorable_sets=layout["scorable_sets"],
            behaviour_only=layout["behaviour_only_sets"])
        #: `score_battery` returns the body; `main()` adds the identity keys
        #: around it, so the check is against the union the driver sees.
        produced = set(scored) | {"result_sha256", "n", "n_scorable",
                                  "correct_overall", "usable_rollout_rate"}
        wanted = self._keys_the_driver_reads()
        assert wanted, "non-vacuous: the driver must read something"
        missing = [k for k in wanted if k not in produced]
        assert not missing, (
            f"the driver reads {missing} from the scorer's output and the "
            "scorer does not write them; that is a KeyError on the last step "
            "of the first probe, on a billing pod")

    def test_the_ranking_rows_carry_what_rank_screening_reads(self):
        """The driver builds ranking rows from the probe record; those four
        fields are exactly what `rank_screening`'s docstring requires."""
        import re

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py").read_text()
        call = src.split("rank_screening(", 1)[1][:400]
        for field in ("arm", "seed", "correct_overall", "usable_rollout_rate"):
            assert f'"{field}"' in call, (
                f"the ranking rows omit {field!r}, which rank_screening reads")
        #: and the probe entry really does carry arm and seed, from the probe
        assert '"arm": probe.arm_id' in src
        assert '"seed": probe.seed' in src


class TestAReusedProbeCarriesItsTrainingProvenance:
    """The project rule: a new attempt RESUMES a preserved checkpoint rather
    than retraining it. That is only scientifically safe if the field can say
    which probes were trained in an earlier session, so the provenance travels
    with the bytes and the record names it.

    See logs/stages/stage-1/phase_d1/decisions/reuse_preserved_checkpoints.md.
    """

    @staticmethod
    def _driver():
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py")
        spec = importlib.util.spec_from_file_location("_d1b_prov", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _preserved(tmp_path, *, trained_in_run, payload=b"weights"):
        import hashlib

        root = tmp_path / "probe"
        model = root / "checkpoints" / "step_000042" / "model"
        model.mkdir(parents=True)
        (model / "model.safetensors").write_bytes(payload)
        (model / "config.json").write_text("{}")
        (root / "checkpoints" / "latest.txt").write_text("step_000042")
        (root / "run_completion.json").write_text(json.dumps({"ok": True}))
        (root / "preserved_identity.json").write_text(json.dumps({
            "schema": "aadistill.phase_d1.preserved_trained_probe/v1",
            "probe_id": "d1_screening_q1_s614923639",
            "trained_sha256": hashlib.sha256(payload).hexdigest(),
            "tag": "step_000042",
            "trained_in_run": trained_in_run,
            "trained_on_pod": "podA",
            "trained_under_image": "img@1",
        }))
        return root

    def test_a_probe_trained_elsewhere_is_flagged(self, tmp_path):
        mod = self._driver()
        mod.RUN_ID_FOR_PROVENANCE[0] = "run_B"
        state = mod.trained_checkpoint_state(
            self._preserved(tmp_path, trained_in_run="run_A"))
        assert state is not None, "a complete preserved probe must be resumable"
        assert state["restored_sha256_verified"] is True
        assert state["trained_in_run"] == "run_A"
        assert state["trained_on_pod"] == "podA"
        assert state["_training_session_differs"] is True, (
            "a field mixing training sessions must say so in the record a "
            "reader consults")

    def test_a_probe_trained_here_is_not_flagged(self, tmp_path):
        """Non-vacuity: the flag must distinguish, not always fire."""
        mod = self._driver()
        mod.RUN_ID_FOR_PROVENANCE[0] = "run_A"
        state = mod.trained_checkpoint_state(
            self._preserved(tmp_path, trained_in_run="run_A"))
        assert state["_training_session_differs"] is False

    def test_tampered_bytes_are_refused_not_resumed(self, tmp_path):
        """Reuse never means trusting whatever arrived: the shard is re-hashed
        against the digest the preservation recorded."""
        mod = self._driver()
        mod.RUN_ID_FOR_PROVENANCE[0] = "run_B"
        root = self._preserved(tmp_path, trained_in_run="run_A")
        shard = next(root.rglob("model.safetensors"))
        shard.write_bytes(b"tampered")
        with pytest.raises(Exception, match="nobody preserved"):
            mod.trained_checkpoint_state(root)

    def test_the_launcher_stamps_the_training_run_into_the_sidecar(self):
        """The producer side, so the consumer above is not reading a field
        nothing writes."""
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_launch.py").read_text()
        block = src.split('"aadistill.phase_d1.preserved_trained_probe/v1"', 1)[1][:1400]
        for field in ("trained_in_run", "trained_on_pod", "trained_under_image"):
            assert field in block, f"the sidecar does not record {field}"

    def test_the_driver_has_a_run_id_to_compare_against(self):
        """A stamp nobody computes is worse than no stamp: the comparison needs
        this session's own id, so the driver must actually take one."""
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py").read_text()
        assert 'ap.add_argument("--run-id", required=True)' in src
        assert "RUN_ID_FOR_PROVENANCE[0] = str(getattr(args" in src
        launcher = (REPO / "scripts/stages/stage-1/phase_d1/"
                           "autoinit_d1_behavioural_launch.py").read_text()
        assert "--run-id '{run_id}'" in launcher or '--run-id' in launcher
