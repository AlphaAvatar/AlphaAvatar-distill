"""The D1 launch chain, against the REAL runner contract rather than a described one.

Every class here closes a blocker that was invisible because the thing it
depended on was never executed. The prepared chain's `$0` evidence was correct
about everything it checked and checked none of these:

* **the evidence layout.** The driver wrote `artifacts/stage1/d1/<run_id>/`
  while `ArtifactPolicy.audit_dirname` sent the relay and the report fetch to
  `artifacts/audit/autoinit_d1/`, and three of the five artifact-spec patterns
  began `artifacts/` again under a collector root that is already
  `<checkout>/artifacts`. A successful beam would have reached teardown with
  every required artifact reported missing.
* **the terminal markers.** `SessionRunner` polls `STATUS_PATH` for
  `MARKER:ALL_DONE`; the driver wrote only `status` into its JSON. A completed
  search would have been observed as `DRIVER_EXITED`, collected under the
  REDUCED spec and reported as a failure.
* **the bound identities.** `config_hash` includes `run_id` AND `device`, and
  the grant was built with `run_id="authorization-probe"` on `cpu` against a
  driver using the real run id on `cuda`. Stage A's equality check would have
  refused EVERY launch, after setup had been paid for.
* **the science inputs.** All three are untracked in git, so a pod that clones
  the bundle has none of them, and `relay_inputs`/`local_assets` were both `()`.
* **the operational arguments.** `--max-price`, `--disk-gb` and `--out` all
  defaulted to `None` and nothing resolved them, so `make_plan` aborted on a
  `TypeError` before any gate ran -- and `--dry-run` returned before
  `SessionRunner` was constructed, which is why nobody saw it.

These run on CPU and create nothing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit", "tests"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.phase_d1 import d1_authorization as A  # noqa: E402
from experiments.phase_d1 import d1_session as D1S  # noqa: E402

RUN_ID = "d1_search_contract_probe"


def _launcher():
    import autoinit_d1_launch as L

    return L


def _driver():
    import autoinit_d1_driver as D

    return D


def _args(**over):
    L = _launcher()
    argv = ["--scr", over.pop("scr", "/tmp/d1-contract"),
            "--session-commit", over.pop("session_commit", "d" * 40),
            "--bundle", over.pop("bundle", "aad_autoinit_dddddddd.bundle"),
            "--run-id", over.pop("run_id", RUN_ID)]
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    args = L.build_parser().parse_args(argv)
    L.resolve_operational_defaults(args)
    return args


# ---------------------------------------------------------------------------
# one authoritative evidence layout
# ---------------------------------------------------------------------------

class TestOneAuthoritativeEvidenceLayout:
    """The driver's `--out`, the relay, the report fetch and both artifact specs
    must name ONE directory."""

    @staticmethod
    def _driver_out(args) -> str:
        import shlex
        import types

        L = _launcher()
        cmd = L.driver_command(types.SimpleNamespace(args=args),
                              types.SimpleNamespace(soft_stop_seconds=600.0))
        toks = shlex.split(cmd)
        return toks[toks.index("--out") + 1]

    def test_the_driver_writes_into_the_declared_audit_directory(self):
        L = _launcher()
        args = _args()
        out = self._driver_out(args)
        spec = L.spec(args)
        #: EXACTLY what the runner builds its relay path and collection root
        #: from. Derived from the spec, not from a literal, so a change to
        #: `audit_dirname` moves both sides or fails here.
        assert out == f"{L.REPO}/artifacts/audit/{spec.artifacts.audit_dirname}"

    def test_the_relay_evidence_path_is_the_file_the_driver_writes(self):
        """`session_runner` relays
        `{checkout}/artifacts/audit/{audit_dirname}/{evidence_filename}`."""
        L = _launcher()
        args = _args()
        spec = L.spec(args)
        relayed = (f"{L.REPO}/artifacts/audit/{spec.artifacts.audit_dirname}/"
                   f"{spec.artifacts.evidence_filename}")
        assert relayed == f"{self._driver_out(args)}/d1_search.json"

    def test_every_report_name_is_a_file_the_driver_writes_into_out(self):
        """The runner fetches each one from `{audit}/{name}`; a name the driver
        does not write there comes home absent, and `fetch_products` reads two
        of these to decide which checkpoints are products."""
        L = _launcher()
        spec = L.spec(_args())
        assert set(spec.artifacts.report_names) == {
            "d1_search.json", "journal.jsonl", "stage1_selection.json"}

    def test_no_artifact_spec_pattern_repeats_the_collector_root(self):
        """THE REGRESSION. `build_manifest` globs `root.glob(pattern)` with
        `root = <checkout>/artifacts`, so a pattern beginning `artifacts/`
        resolves to `<checkout>/artifacts/artifacts/...` and can never match.
        Three of the five entries did."""
        for name in ("d1_search_artifacts.json",
                     "d1_search_artifacts_failed.json"):
            doc = json.loads((REPO / "configs/autoinit" / name).read_text())
            for entry in doc["entries"]:
                assert not entry["pattern"].startswith("artifacts/"), (
                    name, entry["artifact_class"], entry["pattern"])

    def test_every_pattern_lives_under_the_declared_audit_directory(self):
        L = _launcher()
        spec = L.spec(_args())
        prefix = f"audit/{spec.artifacts.audit_dirname}/"
        for name in (spec.artifacts.spec_success, spec.artifacts.spec_failed):
            doc = json.loads((REPO / name).read_text())
            for entry in doc["entries"]:
                assert entry["pattern"].startswith(prefix), (
                    name, entry["artifact_class"], entry["pattern"])

    def test_the_real_collector_finds_everything_the_driver_writes(self, tmp_path):
        """THE WHOLE CONTRACT, through the real `build_manifest`.

        A simulated pod checkout, the five files the driver actually produces
        written where it writes them, and the SUCCESS spec expanded against the
        root `SessionRunner` passes (`<checkout>/artifacts`). `manifest.ok` is
        what the teardown gate's `required_files_present` reads.
        """
        from aadistill.infrastructure.artifact_gate import build_manifest
        from collect_artifacts import load_specs

        L = _launcher()
        spec = L.spec(_args())
        checkout = tmp_path / "aad"
        out = checkout / "artifacts" / "audit" / spec.artifacts.audit_dirname
        (out / "search").mkdir(parents=True)
        #: Exactly the driver's own filenames. `min_bytes` is 512 on the record
        #: and 256 on the selection, so the payloads are realistic sizes.
        (out / "d1_search.json").write_text(json.dumps(
            {"status": "COMPLETE", "pad": "x" * 600}))
        (out / "stage1_selection.json").write_text(json.dumps(
            {"selected": [], "pad": "x" * 300}))
        (out / "journal.jsonl").write_text('{"stage":"A"}\n')
        (out / "search" / "states.jsonl").write_text('{"state_id":"s"}\n')
        (out / "search" / "telemetry.jsonl").write_text('{"t":0}\n')

        manifest = build_manifest(
            checkout / "artifacts",
            load_specs(str(REPO / spec.artifacts.spec_success)),
            settle_seconds=0.0)
        assert manifest.ok, manifest.missing
        assert len(manifest.entries) == 5
        assert {Path(e.path).name for e in manifest.entries} == {
            "d1_search.json", "stage1_selection.json", "journal.jsonl",
            "states.jsonl", "telemetry.jsonl"}

    def test_the_success_spec_is_unsatisfied_by_the_old_layout(self, tmp_path):
        """The same files under `artifacts/stage1/d1/<run_id>/` must NOT satisfy
        the spec -- otherwise this suite would pass for either layout and settle
        nothing."""
        from aadistill.infrastructure.artifact_gate import build_manifest
        from collect_artifacts import load_specs

        L = _launcher()
        spec = L.spec(_args())
        checkout = tmp_path / "aad"
        old = checkout / "artifacts" / "stage1" / "d1" / RUN_ID
        (old / "search").mkdir(parents=True)
        for rel, body in (("d1_search.json", {"status": "COMPLETE", "pad": "x" * 600}),
                          ("stage1_selection.json", {"selected": [], "pad": "x" * 300})):
            (old / rel).write_text(json.dumps(body))
        (old / "journal.jsonl").write_text('{"stage":"A"}\n')
        (old / "search" / "states.jsonl").write_text('{"state_id":"s"}\n')
        (old / "search" / "telemetry.jsonl").write_text('{"t":0}\n')
        manifest = build_manifest(
            checkout / "artifacts",
            load_specs(str(REPO / spec.artifacts.spec_success)),
            settle_seconds=0.0)
        assert not manifest.ok
        assert len(manifest.missing) == 5


# ---------------------------------------------------------------------------
# the terminal markers the runner consumes
# ---------------------------------------------------------------------------

class TestTheTerminalMarkerContract:
    """`SessionRunner` ends a session on `MARKER:<success>` / `MARKER:<failure>`
    in the status file. The driver wrote neither."""

    def test_the_driver_and_the_launcher_name_the_same_markers(self):
        L, D = _launcher(), _driver()
        spec = L.spec(_args())
        assert D.SUCCESS_MARKER == spec.markers.success
        assert D.FAILURE_MARKER in spec.markers.failure

    @pytest.mark.parametrize("check_only,status,written,expect", [
        (False, "COMPLETE", True, "ALL_DONE"),
        #: EVIDENCE IS A PRECONDITION of the success terminal: the launcher
        #: collects under the SUCCESS spec when it sees `ALL_DONE`, and that
        #: spec REQUIRES `d1_search.json`.
        (False, "COMPLETE", False, "RUN_FAILED"),
        (False, "FAILED", True, "RUN_FAILED"),
        (False, "STARTED", True, "RUN_FAILED"),
        #: A $0 CONTRACT CHECK IS NOT A COMPLETED FORMAL SEARCH.
        (True, "CHECK_ONLY_OK", True, "CHECK_ONLY_OK"),
        (True, "COMPLETE", True, "CHECK_ONLY_OK"),
        (True, "FAILED", True, "CHECK_ONLY_OK"),
    ])
    def test_the_marker_decision(self, check_only, status, written, expect):
        """The predicate itself, tabled. Mutating any branch fails a row."""
        D = _driver()
        assert D.terminal_marker(check_only=check_only, status=status,
                                 evidence_written=written) == expect

    def test_mark_writes_the_substring_the_poller_greps_for(self, tmp_path):
        """The runner reads `tail -1 <status>` and tests
        `f"MARKER:{success}" in st`. A marker the poller cannot see is not a
        marker."""
        L, D = _launcher(), _driver()
        status = tmp_path / "autoinit_d1.status"
        D.mark(status, D.SUCCESS_MARKER)
        last = status.read_text().splitlines()[-1]
        spec = L.spec(_args())
        assert f"MARKER:{spec.markers.success}" in last
        #: And the failure marker must not match the success test.
        D.mark(status, D.FAILURE_MARKER)
        last = status.read_text().splitlines()[-1]
        assert f"MARKER:{spec.markers.success}" not in last
        assert any(f"MARKER:{m}" in last for m in spec.markers.failure)

    def test_mark_appends_rather_than_truncating(self, tmp_path):
        """The status file is append-only; the runner also greps it for setup
        markers and `_collect_setup_failure_evidence` reads the whole thing."""
        D = _driver()
        status = tmp_path / "s"
        D.mark(status, "DRIVER_START")
        D.mark(status, "RUN_FAILED")
        assert len(status.read_text().splitlines()) == 2

    def test_an_unwritable_status_path_never_kills_the_driver(self, tmp_path):
        """A driver that died writing a marker would lose the evidence it was
        about to announce."""
        D = _driver()
        blocked = tmp_path / "file" / "s"
        (tmp_path / "file").write_text("not a directory")
        D.mark(blocked, "ALL_DONE")      # must not raise

    def test_the_polled_path_is_the_appended_path(self):
        """One declaration, two readers. A driver appending to a file the runner
        does not poll is the same defect as writing no marker at all.

        `d1_session.STATUS_PATH` is the owner; the launcher's
        `SessionSpec.status_path` and the `--status` the driver command carries
        are both reads of it.
        """
        import shlex
        import types

        L = _launcher()
        spec = L.spec(_args())
        assert spec.status_path == D1S.STATUS_PATH
        cmd = L.driver_command(types.SimpleNamespace(args=_args()),
                               types.SimpleNamespace(soft_stop_seconds=600.0))
        toks = shlex.split(cmd)
        assert toks[toks.index("--status") + 1] == spec.status_path

    def test_check_only_really_runs_and_marks_check_only_ok(self, tmp_path):
        """THE REAL DRIVER, on this CPU box, through stage A for real: the four
        registries, the frozen state-eval asset, the session and its contract.

        It must reach `CHECK_ONLY_OK` and must NOT emit the paid success
        terminal. Stage A's CUDA probe is the one thing a CPU box cannot
        satisfy, so this asserts on whichever terminal the run earns -- never
        `ALL_DONE`, which is the property under test.
        """
        D = _driver()
        out = tmp_path / "out"
        status = tmp_path / "s"
        rc = D.main(["--out", str(out), "--run-id", RUN_ID,
                     "--status", str(status), "--device", "cpu",
                     "--check-only"])
        record = json.loads((out / "d1_search.json").read_text())
        last = status.read_text().splitlines()[-1]
        assert "MARKER:ALL_DONE" not in status.read_text(), (
            "a --check-only run must never emit the paid success terminal")
        if rc == 0:
            assert record["status"] == "CHECK_ONLY_OK"
            assert "MARKER:CHECK_ONLY_OK" in last
        else:
            #: A CPU box without CUDA: stage A refuses, which is correct.
            assert record["status"] == "FAILED"
            assert "CUDA" in json.dumps(record["failure"])
        #: EVIDENCE ON EVERY PATH OUT, and the run id in it.
        assert record["run_id"] == RUN_ID
        assert (out / "journal.jsonl").is_file()

    def test_a_failing_run_marks_run_failed_and_still_writes_evidence(self,
                                                                     tmp_path):
        """A real refusal: no `--authorization` and no `--check-only` is
        unauthorized work, and the driver says so."""
        D = _driver()
        out = tmp_path / "out"
        status = tmp_path / "s"
        rc = D.main(["--out", str(out), "--run-id", RUN_ID,
                     "--status", str(status), "--device", "cpu"])
        assert rc == 1
        assert "MARKER:RUN_FAILED" in status.read_text().splitlines()[-1]
        assert json.loads((out / "d1_search.json").read_text())["status"] == \
            "FAILED"

    def test_the_driver_marks_a_start_before_anything_can_fail(self, tmp_path):
        """So a pod whose driver died in stage A is distinguishable from one
        whose driver never started."""
        D = _driver()
        status = tmp_path / "s"
        D.main(["--out", str(tmp_path / "o"), "--run-id", RUN_ID,
                "--status", str(status), "--device", "cpu"])
        assert "MARKER:DRIVER_START" in status.read_text().splitlines()[0]


# ---------------------------------------------------------------------------
# the identities the money is authorized against
# ---------------------------------------------------------------------------

class TestTheBoundIdentityIsReachable:
    """`config_hash` carries `run_id` and `device`. The grant must be issued
    with both, or stage A refuses every launch after setup."""

    def _session(self, run_id: str, device: str):
        import tempfile

        D1S._register_frozen_operators()
        with tempfile.TemporaryDirectory() as tmp:
            return D1S.build_session(arm=A.FORMAL_ARM, workdir=Path(tmp),
                                     run_id=run_id, device=device,
                                     repo_root=REPO).config

    def test_run_id_moves_the_config_hash(self):
        """So a grant naming the wrong run id binds an unmatchable identity."""
        a = self._session("run_one", A.FORMAL_DEVICE).config_hash
        b = self._session("run_two", A.FORMAL_DEVICE).config_hash
        assert a != b

    def test_device_moves_the_config_hash(self):
        """The second half of the same defect: the grant was built on `cpu`."""
        cpu = self._session(RUN_ID, "cpu").config_hash
        cuda = self._session(RUN_ID, "cuda").config_hash
        assert cpu != cuda

    def test_the_formal_device_is_what_the_driver_is_told_to_use(self):
        import shlex
        import types

        L = _launcher()
        cmd = L.driver_command(types.SimpleNamespace(args=_args()),
                               types.SimpleNamespace(soft_stop_seconds=600.0))
        toks = shlex.split(cmd)
        assert toks[toks.index("--device") + 1] == A.FORMAL_DEVICE

    def test_an_issued_payload_binds_the_hash_the_driver_computes(self):
        """END TO END on the identity that could not be matched: the payload the
        issuer builds for a run id must equal the session the tree builds for
        the same run id on the formal device."""
        payload = A.build_payload(
            grant={"granted_by": "test", "covers": "identity probe"},
            session_commit="a" * 40, granted_utc="2026-10-05T18:00:00Z",
            run_id=RUN_ID, live_rate=1.09, repo_root=REPO)
        config = self._session(RUN_ID, A.FORMAL_DEVICE)
        assert payload["run_id"] == RUN_ID
        assert payload["config_hash"] == config.config_hash
        assert payload["measurement_protocol_id"] == \
            config.measurement_protocol_id

    def test_a_payload_for_another_run_does_not_bind_this_one(self):
        payload = A.build_payload(
            grant={"granted_by": "test", "covers": "identity probe"},
            session_commit="a" * 40, granted_utc="2026-10-05T18:00:00Z",
            run_id="some_other_run", live_rate=1.09, repo_root=REPO)
        assert payload["config_hash"] != \
            self._session(RUN_ID, A.FORMAL_DEVICE).config_hash

    def test_an_unnamed_run_cannot_be_authorized(self):
        with pytest.raises(A.D1AuthorizationRefused, match="run_id"):
            A.build_payload(
                grant={"granted_by": "test"}, session_commit="a" * 40,
                granted_utc="2026-10-05T18:00:00Z", run_id="  ",
                live_rate=1.09, repo_root=REPO)

    def test_an_artifact_without_a_run_id_is_refused_on_load(self, tmp_path):
        """A grant binding a config hash without saying which run it belongs to
        binds a number nothing can be compared against."""
        from aadistill.infrastructure.manifest import sha256_json

        payload = A.build_payload(
            grant={"granted_by": "test", "covers": "x"},
            session_commit="a" * 40, granted_utc="2026-10-05T18:00:00Z",
            run_id=RUN_ID, live_rate=1.09, repo_root=REPO)
        payload.pop("run_id")
        payload.pop("authorization_sha256")
        payload["authorization_sha256"] = sha256_json(payload)
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(A.D1AuthorizationRefused, match="run_id"):
            A.D1Authorization.load(path)

    def test_require_run_id_refuses_a_mismatch(self, tmp_path):
        payload = A.build_payload(
            grant={"granted_by": "test", "covers": "x"},
            session_commit="a" * 40, granted_utc="2026-10-05T18:00:00Z",
            run_id=RUN_ID, live_rate=1.09, repo_root=REPO)
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload))
        auth = A.D1Authorization.load(path)
        auth.require_run_id(RUN_ID)
        with pytest.raises(A.D1AuthorizationRefused, match="issued for run"):
            auth.require_run_id("a_different_run")


# ---------------------------------------------------------------------------
# every science input, materialized by setup
# ---------------------------------------------------------------------------

class TestEveryScienceInputIsStaged:
    """The driver's non-source inputs are not in the bundle, so setup must put
    them there. `relay_inputs` and `local_assets` were both `()`."""

    def test_none_of_the_three_is_in_git(self):
        """The premise. If one of these is ever committed, its `LocalAsset`
        declaration becomes redundant and this test is where that is noticed."""
        L = _launcher()
        for asset in L.SCIENCE_ASSETS:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", asset.repo_path],
                capture_output=True, cwd=REPO)
            assert tracked.returncode != 0, (
                f"{asset.repo_path} is tracked; a bundle-only checkout would "
                "already have it and the local asset is redundant")

    def test_the_session_declares_all_three(self):
        L = _launcher()
        spec = L.spec(_args())
        declared = {a.repo_path for a in spec.setup.local_assets}
        assert declared == {
            D1S.STATE_EVAL_ROOT,
            "artifacts/stage1/e8_calibration_v1",
            "artifacts/stage1/reasoning_heavy_v2"}

    def test_an_empty_pod_filesystem_cannot_satisfy_the_driver(self, tmp_path):
        """A bundle-only checkout has none of them, which is why they must be
        declared. If this ever passes, the staging below is proving nothing."""
        empty = tmp_path / "aad"
        (empty / "artifacts" / "stage1").mkdir(parents=True)
        with pytest.raises(D1S.D1SessionError):
            D1S.verify_state_eval_bytes(empty)

    def test_the_declared_staging_materializes_every_input(self, tmp_path):
        """THE SIMULATION. Start from an empty pod filesystem, run exactly what
        the launcher and the setup script do with `local_assets`, then resolve
        each input through its REAL consumer against that tree.

        The launcher scp's `repo_root/<repo_path>` to `$WS/assets/<dest_name>`;
        setup does `mkdir -p $REPO/<install_to> && cp -r $WS/assets/<dest_name>
        $REPO/<install_to>/`. Both steps are reproduced rather than described.
        """
        from aadistill.initialization.calibration.profiles import get_profile
        from experiments.calibration import register_builtin_profiles

        L = _launcher()
        spec = L.spec(_args())
        ws = tmp_path / "workspace"
        checkout = ws / "aad"
        checkout.mkdir(parents=True)
        (ws / "assets").mkdir()

        for asset in spec.setup.local_assets:
            #: the launcher's scp -r
            shutil.copytree(REPO / asset.repo_path,
                            ws / "assets" / asset.dest_name)
            #: setup's mkdir -p + cp -r
            into = checkout / asset.install_to
            into.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ws / "assets" / asset.dest_name,
                            into / asset.dest_name)

        #: THE REAL CONSUMERS, against the simulated checkout.
        verified = D1S.verify_state_eval_bytes(checkout)
        assert verified["items_sha256"]
        suite, items, content = D1S.load_state_eval_suite(checkout)
        assert len(items) == verified["n_items"]
        assert content == verified["content_sha256"]

        register_builtin_profiles()
        for pid in sorted(D1S.design(REPO)["search_stage"]["profiles"]):
            #: `resolve` checks `items_file_sha256` and recomputes the mixture
            #: content hash -- the same call the search makes on the pod.
            assert get_profile(pid).resolve(checkout)

    def test_a_truncated_items_file_is_refused_rather_than_measured(self,
                                                                   tmp_path):
        """`LocalAsset` carries no digest, so this is the check that makes the
        scp safe: a short `items.jsonl` would otherwise yield a SMALLER suite
        while the session still bound the full `content_sha256`."""
        checkout = tmp_path / "aad"
        root = checkout / D1S.STATE_EVAL_ROOT
        root.mkdir(parents=True)
        shutil.copy(REPO / D1S.STATE_EVAL_ROOT / "manifest.json",
                    root / "manifest.json")
        full = (REPO / D1S.STATE_EVAL_ROOT / "items.jsonl").read_text()
        (root / "items.jsonl").write_text(
            "\n".join(full.splitlines()[:10]) + "\n")
        with pytest.raises(D1S.D1SessionError, match="hashes to"):
            D1S.verify_state_eval_bytes(checkout)

    def test_an_edited_manifest_is_refused(self, tmp_path):
        """The pins are only trustworthy if the record carrying them is."""
        checkout = tmp_path / "aad"
        root = checkout / D1S.STATE_EVAL_ROOT
        root.mkdir(parents=True)
        doc = json.loads(
            (REPO / D1S.STATE_EVAL_ROOT / "manifest.json").read_text())
        doc["counts"]["n_items"] = 1
        (root / "manifest.json").write_text(json.dumps(doc))
        shutil.copy(REPO / D1S.STATE_EVAL_ROOT / "items.jsonl",
                    root / "items.jsonl")
        with pytest.raises(D1S.D1SessionError, match="manifest_sha256"):
            D1S.verify_state_eval_bytes(checkout)

    def test_the_teacher_step_is_declared_with_its_pinned_revision(self):
        """Stage B materializes a 4B-class teacher. Nothing downloaded it and
        nothing set a path to it."""
        L = _launcher()
        spec = L.spec(_args())
        binding = json.loads((REPO / D1S.TEACHER_BINDING).read_text())
        assert "TEACHER_READY" in spec.setup.setup_markers
        assert spec.setup.teacher_revision == binding["revision"]
        assert "TEACHER_REVISION" in spec.setup.required_env
        #: And it reaches setup's environment, which is what the step reads.
        env = spec.setup_environment(session_commit="a" * 40, bundle="b.bundle")
        assert env["TEACHER_REVISION"] == binding["revision"]

    def test_the_driver_asks_the_hub_for_a_lookup_not_a_download(self,
                                                                 monkeypatch):
        """`local_files_only=True`, and a cache miss becomes a refusal that
        names the setup step rather than an 8 GB transfer on the search's clock.

        The hub call is SPIED rather than redirected: `huggingface_hub` resolves
        its cache directory at IMPORT time, so `HF_HOME` only takes effect for a
        process that has not imported it yet -- which made an env-based version
        of this test pass alone and fail inside the suite. The contract under
        test is how this driver calls the library, so the call is what is
        observed, and the exception raised is the real `LocalEntryNotFoundError`
        the library raises on a cache miss.
        """
        import huggingface_hub
        from huggingface_hub.errors import LocalEntryNotFoundError

        D = _driver()
        D1S._register_frozen_operators()
        monkeypatch.delenv("D1_TEACHER_PATH", raising=False)
        monkeypatch.delenv("TEACHER_PATH", raising=False)

        seen: dict = {}

        def spy(repo_id, **kwargs):
            seen.update({"repo_id": repo_id, **kwargs})
            raise LocalEntryNotFoundError("nothing in the cache")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", spy)
        with pytest.raises(D.D1DriverError, match="Hugging Face cache"):
            D.resolve_root_teacher_path()
        #: THE MECHANISM. Without this the call is an ordinary download.
        assert seen["local_files_only"] is True
        binding = json.loads((REPO / D1S.TEACHER_BINDING).read_text())
        assert seen["repo_id"] == binding["repo_id"]
        assert seen["revision"] == binding["revision"]

    def test_the_refusal_names_how_the_teacher_is_staged(self, monkeypatch):
        """A refusal that does not say what stages the bytes is a stop, not a
        diagnosis."""
        import huggingface_hub
        from huggingface_hub.errors import LocalEntryNotFoundError

        D = _driver()
        D1S._register_frozen_operators()
        monkeypatch.delenv("D1_TEACHER_PATH", raising=False)
        monkeypatch.delenv("TEACHER_PATH", raising=False)
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download",
            lambda *a, **k: (_ for _ in ()).throw(
                LocalEntryNotFoundError("miss")))
        with pytest.raises(D.D1DriverError) as exc:
            D.resolve_root_teacher_path()
        assert "TEACHER_READY" in str(exc.value)
        assert "D1_TEACHER_PATH" in str(exc.value)

    def test_an_explicit_teacher_path_is_honoured(self, tmp_path, monkeypatch):
        """How a $0 test and a local reproduction point at a staged copy."""
        D = _driver()
        D1S._register_frozen_operators()
        staged = tmp_path / "teacher"
        staged.mkdir()
        monkeypatch.setenv("D1_TEACHER_PATH", str(staged))
        out = D.resolve_root_teacher_path()
        assert out["path"] == str(staged)
        assert out["_resolved_by"] == "environment"

    def test_the_undeclared_steps_are_the_ones_that_would_break_or_waste(self):
        """`ROPE_OK` exits non-zero with no staged checkpoint, which D1 has;
        `VLLM_READY` is the most expensive step and D1 generates nothing."""
        L = _launcher()
        declared = set(L.spec(_args()).setup.setup_markers)
        assert "ROPE_OK" not in declared
        assert "VLLM_READY" not in declared
        #: And the shell really does refuse without a staged checkpoint, so the
        #: omission is load-bearing rather than a preference.
        shell = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
        assert "no staged checkpoint to check" in shell


# ---------------------------------------------------------------------------
# the operational arguments the runner reads
# ---------------------------------------------------------------------------

class TestTheOperationalArgumentsResolve:
    """`--max-price`, `--disk-gb` and `--out` defaulted to `None` and nothing
    filled them, so `make_plan` aborted on a TypeError before any gate ran."""

    def test_all_three_are_resolved(self):
        args = _args()
        assert args.max_price is not None
        assert args.disk_gb is not None
        assert args.out is not None

    def test_the_price_is_the_rate_the_ceiling_was_derived_at(self):
        """A launch priced above it would terminate at a dollar figure nothing
        authorized; the help text promised exactly this and nothing did it."""
        args = _args()
        assert args.max_price == A.session_ceiling(REPO)["price_per_hour"]

    def test_the_disk_is_the_size_the_cost_model_charged_for(self):
        args = _args()
        assert args.disk_gb == A.session_ceiling(REPO)["container_disk_gb"]

    def test_the_record_lands_in_this_runs_directory(self):
        L = _launcher()
        args = _args()
        assert args.out.startswith(L.run_dir_for(RUN_ID))
        assert args.out.endswith(".json")

    def test_an_explicit_value_is_respected(self):
        args = _args(max_price=0.99, disk_gb=123)
        assert args.max_price == 0.99 and args.disk_gb == 123

    def test_the_runner_argument_contract_is_satisfied(self):
        """`missing_arguments` is what aborts construction; it is free here."""
        from aadistill.infrastructure.session import missing_arguments

        assert missing_arguments(_args()) == []

    def test_the_budget_plan_lands_on_the_accepted_bound(self):
        """The plan's hard threshold must not exceed the derived ceiling: two
        models of one session is what produced a $22.19 plan against a $21.4897
        authorization."""
        L = _launcher()
        priced = A.session_ceiling(REPO)
        plan = L.budget_spec(REPO).plan(
            price_per_hour=priced["price_per_hour"],
            authorized_usd=priced["hard_ceiling_usd"])
        assert plan.hard_terminate_minutes == pytest.approx(
            priced["hard_ceiling_minutes"], abs=0.01)
        assert plan.hard_terminate_usd <= priced["hard_ceiling_usd"] + 5e-4

    def test_the_poll_limit_outlasts_the_hard_threshold(self):
        """Or the launcher stops watching a pod that is still billing."""
        args = _args()
        priced = A.session_ceiling(REPO)
        assert args.poll_limit_min > priced["hard_ceiling_minutes"]


# ---------------------------------------------------------------------------
# the gates, and that they are wired
# ---------------------------------------------------------------------------

def _auth_for(tmp_path, run_id=RUN_ID, commit="a" * 40):
    payload = A.build_payload(
        grant={"granted_by": "test", "covers": "gate probe"},
        session_commit=commit, granted_utc="2026-10-05T18:00:00Z",
        run_id=run_id, live_rate=1.09, repo_root=REPO)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload))
    return A.D1Authorization.load(path), payload


def _gate_ctx(args, auth):
    import types

    return types.SimpleNamespace(args=args, auth=auth, evidence={},
                                 say=lambda *a, **k: None)


class TestThePreProviderGates:
    """`SessionSpec.precheck` was `()`, and the runner never calls
    `require_session_commit`."""

    def test_the_spec_declares_all_six_in_order(self):
        L = _launcher()
        names = [getattr(c, "__name__", str(c))
                 for c in L.spec(_args()).precheck]
        assert names == ["run_identity_gate", "session_contract_gate",
                         "staged_science_inputs_gate", "launch_readiness_gate",
                         "session_commit_and_lineage", "bundle_staged_gate"]

    def test_the_commit_gate_is_the_shared_one_with_lineage(self):
        """Not a private copy: the shared gate is what checks the harness digest
        at the commit AND that nothing but the authorization differs from the
        authorized base."""
        L = _launcher()
        gate = [c for c in L.spec(_args()).precheck
                if getattr(c, "__name__", "") == "session_commit_and_lineage"]
        assert len(gate) == 1

    def test_run_identity_refuses_a_different_run(self, tmp_path):
        L = _launcher()
        auth, _ = _auth_for(tmp_path, run_id="the_authorized_run")
        ok, why = L.run_identity_gate(_gate_ctx(_args(run_id=RUN_ID), auth))
        assert ok is False and "the_authorized_run" in why

    def test_run_identity_accepts_the_authorized_run(self, tmp_path):
        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        ctx = _gate_ctx(_args(), auth)
        ok, why = L.run_identity_gate(ctx)
        assert ok is True, why
        assert ctx.evidence["d1_run_identity"]["run_id"] == RUN_ID

    def test_run_identity_refuses_the_control_arm(self, tmp_path):
        """A FORMAL D1 search is treatment-only; the control is a $0 check."""
        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        ok, why = L.run_identity_gate(
            _gate_ctx(_args(arm=D1S.CONTROL_ARM), auth))
        assert ok is False

    def test_the_contract_gate_accepts_this_tree(self, tmp_path):
        """And therefore proves the bound identities are reachable at $0 -- the
        equality stage A asserts on the pod."""
        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        ctx = _gate_ctx(_args(), auth)
        ok, why = L.session_contract_gate(ctx)
        assert ok is True, why
        record = ctx.evidence["d1_session_contract_check"]
        assert record["problems"] == []
        assert record["built_with_device"] == A.FORMAL_DEVICE
        assert record["config_hash"] == auth.config_hash

    def test_the_contract_gate_refuses_a_hash_the_tree_cannot_produce(self,
                                                                     tmp_path):
        """THE EXPENSIVE ONE MADE CHEAP. This is the mismatch that would have
        landed in stage A on a billing pod, and it is the shape the committed
        grant really had: a `config_hash` no session this tree builds can
        reproduce.

        Reproduced by replacing the bound hash rather than by issuing for
        another run -- an authorization issued for run X and launched as run X
        agrees by construction, and `require_run_id` already refuses the
        cross-run case.
        """
        import dataclasses

        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        wrong = dataclasses.replace(auth, config_hash="0" * 64)
        ok, why = L.session_contract_gate(_gate_ctx(_args(), wrong))
        assert ok is False and "config_hash" in why

    def test_the_contract_gate_refuses_a_cpu_built_hash(self, tmp_path):
        """EXACTLY the committed grant's second defect: a `config_hash` computed
        on `cpu` for a session the driver runs on `cuda`."""
        import dataclasses
        import tempfile

        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        D1S._register_frozen_operators()
        with tempfile.TemporaryDirectory() as tmp:
            on_cpu = D1S.build_session(
                arm=A.FORMAL_ARM, workdir=Path(tmp), run_id=RUN_ID,
                device="cpu", repo_root=REPO).config.config_hash
        assert on_cpu != auth.config_hash
        wrong = dataclasses.replace(auth, config_hash=on_cpu)
        ok, why = L.session_contract_gate(_gate_ctx(_args(), wrong))
        assert ok is False and "config_hash" in why

    def test_the_staged_inputs_gate_passes_on_this_tree(self, tmp_path):
        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        ctx = _gate_ctx(_args(), auth)
        ok, why = L.staged_science_inputs_gate(ctx)
        assert ok is True, why
        checked = ctx.evidence["d1_staged_science_inputs"]["verified"]
        assert "state_eval_v1" in checked
        assert len(checked) == 3

    def test_the_readiness_gate_refuses_an_absent_record(self, tmp_path):
        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        ok, why = L.launch_readiness_gate(
            _gate_ctx(_args(run_id="no_such_run"), auth))
        assert ok is False and "launch_readiness" in why

    def test_the_bundle_gate_refuses_a_non_canonical_name(self, tmp_path):
        """`--bundle c1` was an alias for nothing and cost a paid pod."""
        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        ok, why = L.bundle_staged_gate(
            _gate_ctx(_args(bundle="d1.bundle"), auth))
        assert ok is False and "canonical" in why

    def test_the_bundle_gate_refuses_an_unstaged_bundle(self, tmp_path):
        """And names the command that stages it. The prepared chain's bundle did
        not exist on the relay at all."""
        L = _launcher()
        auth, _ = _auth_for(tmp_path)
        ok, why = L.bundle_staged_gate(
            _gate_ctx(_args(run_id="no_such_run"), auth))
        assert ok is False and "stage_d1_bundle.py" in why

    def test_the_canonical_bundle_name_is_derived_from_the_commit(self):
        commit = "6c1dd8d9e795004aa7bf1423070682e3c4ac636b"
        assert D1S.canonical_bundle_name(commit) == \
            "aad_autoinit_6c1dd8d9.bundle"
        assert D1S.canonical_bundle_path(commit) == \
            "transfer/aad_autoinit_6c1dd8d9.bundle"


class TestTheGovernanceGeneratorsExist:
    """Every governance artifact needs a command that produces it (AGENTS.md
    P4). The readiness record had no generator at all, and there was no issuer."""

    @pytest.mark.parametrize("script", [
        "scripts/autoinit/issue_d1_authorization.py",
        "scripts/autoinit/write_d1_launch_readiness.py",
        "scripts/autoinit/stage_d1_bundle.py",
    ])
    def test_it_exists_and_parses(self, script):
        path = REPO / script
        assert path.is_file(), script
        compile(path.read_text(), str(path), "exec")

    @pytest.mark.parametrize("script", [
        "scripts/autoinit/issue_d1_authorization.py",
        "scripts/autoinit/write_d1_launch_readiness.py",
        "scripts/autoinit/stage_d1_bundle.py",
    ])
    def test_it_requires_a_run_id(self, script):
        """All three artifacts are per-run, and the authorization's bound
        `config_hash` is a function of the run id."""
        import importlib.util

        name = Path(script).stem
        spec = importlib.util.spec_from_file_location(name, REPO / script)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        actions = {a.dest: a for a in mod.build_parser()._actions}
        assert "run_id" in actions and actions["run_id"].required

    def test_the_paths_agree_with_the_launchers(self):
        """One owner for the run directory: a readiness record the launcher
        cannot find refuses every launch."""
        import importlib.util

        L = _launcher()
        for script, fn, name in (
                ("issue_d1_authorization", "authorization_path_for",
                 "authorization.json"),
                ("write_d1_launch_readiness", None, None),
                ("stage_d1_bundle", None, None)):
            spec = importlib.util.spec_from_file_location(
                script, REPO / f"scripts/autoinit/{script}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[script] = mod
            spec.loader.exec_module(mod)
            if fn:
                assert getattr(mod, fn)(RUN_ID) == L.auth_path_for(RUN_ID)
            else:
                assert mod.governance_path(RUN_ID, "authorization.json") == \
                    L.auth_path_for(RUN_ID)
                assert mod.governance_path(RUN_ID, "launch_readiness.json") == \
                    L.readiness_path_for(RUN_ID)
                assert mod.governance_path(RUN_ID, "bundle.json") == \
                    L.bundle_record_for(RUN_ID)

    def test_the_run_directory_comes_from_the_shared_layout_helper(self):
        from experiments.run_layout import rel_run_dir

        L = _launcher()
        assert L.run_dir_for(RUN_ID) == rel_run_dir(
            D1S.EXPERIMENT_ID, RUN_ID, D1S.STAGE_ID)


class TestTheLaunchCommitIsCheckedByLineageNotEquality:
    """`require_session_commit` asserts equality, and equality can only hold at
    ISSUANCE.

    The authorization is granted against the clean PRE-authorization HEAD --
    the artifact cannot be committed before it exists -- so the commit a pod
    checks out is always a later one. The readiness writer called
    `require_session_commit` with the LAUNCH commit and was refused by it, which
    is the trap this class exists to keep closed.
    """

    def test_equality_cannot_hold_for_an_authorization_carrying_commit(self,
                                                                       tmp_path):
        """The method is correct and the CALLER was wrong: asked about a commit
        that carries the artifact, it must refuse."""
        auth, _ = _auth_for(tmp_path, commit="a" * 40)
        auth.require_session_commit("a" * 40)        # the base: holds
        with pytest.raises(A.D1AuthorizationRefused, match="binds commit"):
            auth.require_session_commit("b" * 40)    # a later commit: refused

    def test_the_readiness_writer_uses_the_shared_lineage_helper(self):
        """And does not CALL the equality check.

        Asserted over the call sites rather than over the text: the file
        explains in a comment why equality is wrong, and a substring check
        cannot tell an explanation from a call -- it flagged the comment.
        """
        import ast

        src = (REPO / "scripts/autoinit/write_d1_launch_readiness.py").read_text()
        called = {
            node.func.attr for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        } | {
            node.func.id for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "lineage_from_authorized_base" in called
        assert "require_session_commit" not in called, (
            "the readiness writer asks for commit EQUALITY against the "
            "authorized base, which no launch commit can satisfy")

    def test_the_launcher_uses_the_same_relation(self):
        """One relation, one implementation. The launcher's gate and the
        readiness writer must not disagree about what a valid launch commit is."""
        L = _launcher()
        names = [getattr(c, "__name__", "") for c in L.spec(_args()).precheck]
        assert "session_commit_and_lineage" in names

    def test_the_helper_distinguishes_the_three_outcomes(self, tmp_path):
        """Descends + only the authorization differs -> ok; not a descendant ->
        refused; an extra path -> refused. Driven on a real throwaway git
        repository rather than on this branch's history, so the test does not
        pin this round's commits."""
        from aadistill.infrastructure.session_prechecks import (
            lineage_from_authorized_base,
        )

        repo = tmp_path / "r"
        repo.mkdir()

        def git(*a, cwd=repo):
            out = subprocess.run(["git", *a], capture_output=True, text=True,
                                 cwd=cwd)
            assert out.returncode == 0, (a, out.stderr)
            return out.stdout.strip()

        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        auth_rel = "governance/authorization.json"
        (repo / "code.py").write_text("x = 1\n")
        git("add", "-A")
        git("commit", "-qm", "base")
        base = git("rev-parse", "HEAD")

        (repo / "governance").mkdir()
        (repo / auth_rel).write_text("{}\n")
        git("add", "-A")
        git("commit", "-qm", "authorization")
        good = git("rev-parse", "HEAD")

        ok = lineage_from_authorized_base(repo, base, good, auth_rel)
        assert ok["ok"] is True, ok["reason"]
        assert ok["changed_paths"] == [auth_rel]

        #: An extra path alongside the authorization.
        (repo / "code.py").write_text("x = 2\n")
        git("add", "-A")
        git("commit", "-qm", "and something else")
        extra = lineage_from_authorized_base(
            repo, base, git("rev-parse", "HEAD"), auth_rel)
        assert extra["ok"] is False
        assert "code.py" in extra["unexpected_paths"]

        #: Backwards: the base does not descend from the later commit.
        assert lineage_from_authorized_base(repo, good, base, auth_rel)["ok"] \
            is False


class TestTheProviderAccountRequirement:
    """D1 declares what the RUNPOD ACCOUNT must hold, and derives the amount.

    The 2026-10-06 search passed all six $0 gates and was stopped by RunPod at
    449.8 of 1125.55 authorized minutes, mid-beam, for an exhausted account
    balance: $8.1716 and 39 of 92 expansions, no endpoint. The gate is generic
    (`BudgetSpec.account_balance_required_usd`); the AMOUNT is this campaign's
    and belongs here.
    """

    def test_the_session_declares_a_requirement(self):
        L = _launcher()
        spec = L.budget_spec(L.REPO_ROOT)
        assert spec.account_balance_required_usd is not None, (
            "a formal paid session that does not declare what the provider "
            "account must hold is the 2026-10-06 configuration exactly")

    def test_it_is_at_least_the_per_attempt_envelope(self):
        """An account holding less than one full envelope cannot fund an attempt
        this package permits, whatever this session's own ceiling is."""
        L = _launcher()
        from experiments.phase_d1 import d1_authorization as A

        envelope = float(
            A.live_money(L.REPO_ROOT)["per_session_envelope_usd"])
        assert float(L.budget_spec(L.REPO_ROOT).account_balance_required_usd) \
            >= envelope

    def test_it_covers_the_ceiling_and_a_reserve(self):
        L = _launcher()
        from experiments.phase_d1 import d1_authorization as A

        priced = A.session_ceiling(L.REPO_ROOT)
        floor = (float(priced["hard_ceiling_usd"])
                 + L.ACCOUNT_OPERATIONAL_RESERVE_USD)
        assert float(L.budget_spec(L.REPO_ROOT).account_balance_required_usd) \
            >= floor, (
            "the requirement must cover the authorized ceiling and an "
            "operational reserve")

    def test_the_container_disk_is_not_counted_twice(self):
        """THE ACCOUNTING BUG. The first derivation was

            hard_ceiling 21.4897 + container_disk 1.0422 + reserve 5.0 = 27.5319

        and `hard_ceiling_usd` ALREADY CONTAINS the disk: gpu_usd 20.4475 plus
        container_disk_usd 1.0422 is 21.4897 exactly. A session ceiling owns
        which priced components it contains, and a caller that re-adds one is
        asserting a cost model it does not own.

        It did not change D1's answer -- the $30 envelope floor dominates
        either way -- which is exactly why it needed a test rather than a
        passing run to find it.
        """
        L = _launcher()
        from experiments.phase_d1 import d1_authorization as A

        priced = A.session_ceiling(L.REPO_ROOT)
        #: `gpu_usd` is a term of the DESIGN's cost cell, which is where the
        #: decomposition lives; `session_ceiling()` returns the bound and the
        #: disk but not the GPU half.
        cost = D1S.design(L.REPO_ROOT)["search_stage"]["cost"]
        gpu = float(cost["gpu_usd"])
        disk = float(cost["container_disk_usd"])
        ceiling = float(priced["hard_ceiling_usd"])
        assert gpu + disk == pytest.approx(ceiling, abs=5e-4), (
            "this test's premise is that the ceiling contains the disk; the "
            "pricing record no longer decomposes that way, so re-derive the "
            "rule rather than trusting this assertion")

        derived = float(L.budget_spec(L.REPO_ROOT).account_balance_required_usd)
        envelope = float(A.live_money(L.REPO_ROOT)["per_session_envelope_usd"])
        correct = max(envelope, ceiling + L.ACCOUNT_OPERATIONAL_RESERVE_USD)
        double = max(envelope, ceiling + disk + L.ACCOUNT_OPERATIONAL_RESERVE_USD)
        assert derived == pytest.approx(correct, abs=5e-4)
        #: Non-vacuity: the probe only bites while the two differ. Today the
        #: envelope floor hides the difference, so assert on the SUM the rule
        #: computes rather than only on its clamped result.
        assert ceiling + L.ACCOUNT_OPERATIONAL_RESERVE_USD < \
            ceiling + disk + L.ACCOUNT_OPERATIONAL_RESERVE_USD
        src = (L.REPO_ROOT / "scripts/pod/autoinit_d1_launch.py").read_text()
        i = src.index("account_balance_required_usd=round(max(")
        rule = src[i:src.index("), 4),", i)]
        assert "container_disk_usd" not in rule, (
            "the derivation re-adds a component the ceiling already contains")
        assert double >= correct       # the bug was always >=, never <

    def test_the_amount_is_derived_and_not_typed(self):
        """No dollar figure for the requirement is written into the launcher:
        the envelope is read from the authorization config and the ceiling from
        the cost model. Only the small reserve is a declared constant, and it is
        named as one."""
        L = _launcher()
        src = (L.REPO_ROOT / "scripts/pod/autoinit_d1_launch.py").read_text()
        assert "account_balance_required_usd=round(max(" in src
        assert 'live_money(repo_root)["per_session_envelope_usd"]' in src
        for typed in ("= 30.0", "= 30\n", "30.0)"):
            assert f"account_balance_required_usd{typed}" not in src


class TestTheReplayFillsEveryRegistryItNeeds:
    """A replay must reproduce the SEARCH's process-global registry state.

    The replay's first paid attempt died five seconds into step 0 with

        KeyError: no calibration profile 'calib.domain_balanced@v1';
                  registered: []

    after setup, the teacher download and a 266-second pod test gate had all
    been paid for. It had registered adapters and the C2 operators and not the
    calibration PROFILES, which are a separate registry. The GPU qualification's
    first subrun failed the same way -- three of four registries filled -- so
    this is the second time one partial registration has cost a pod.

    `d1_session._register_frozen_operators` is the single owner of that list,
    in the order the search fills it. These assert the replay goes through it
    rather than assembling its own subset.
    """

    def test_the_spec_builder_calls_the_single_registry_owner(self):
        import ast

        L = _launcher()
        src = (L.REPO_ROOT /
               "scripts/experiments/stage-1/phase_d1/replay_specs.py").read_text()
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_register_frozen_operators" in calls, (
            "replay_specs builds its own registry subset; a replay must "
            "reproduce the search's registry state through its one owner")
        assert "register_c2_operators" not in calls, (
            "a partial registration is back alongside the owner, which is how "
            "the two can disagree again")

    def test_the_driver_calls_it_too(self):
        import ast

        L = _launcher()
        src = (L.REPO_ROOT /
               "scripts/pod/autoinit_d1_replay_driver.py").read_text()
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_register_frozen_operators" in calls

    def test_every_frozen_profile_and_operator_resolves(self):
        """The check the pod made at $1.09/h, made here for nothing.

        Reads the frozen space and the design -- NOT the replay plan. The first
        version called `replay_plan()`, which reads the 67 MB state journal from
        an out-of-tree dev-box path, and the pod test gate ran it where that
        path does not exist:

            ReplaySourceError: the state journal is not at
            /home/ecs-user/aad-scratch/.../states.jsonl

        A third pod, $0.1849, for a test of mine that could not run on a pod.
        The journal belongs to plan RESOLUTION, which happens on the dev box;
        what a pod needs is that the registries fill, and that is answerable
        from the committed tree alone.
        """
        from aadistill.initialization.calibration.profiles import get_profile
        from aadistill.initialization.operators.base import get_implementation
        from aadistill.initialization.specs.arch import get_adapter
        from experiments.phase_d1 import d1_session as D1S
        from experiments.phase_d1 import replay_specs as R

        D1S._register_frozen_operators()
        stage = D1S.design()["search_stage"]
        #: KIND -> impl_id, so the VALUES are the implementations. Iterating the
        #: mapping yields four plausible-looking kind strings that resolve to
        #: nothing -- a mistake this repository has already made once.
        impls = tuple(stage["frozen_implementations"].values())
        profiles = tuple(stage["profiles"])
        assert len(impls) == 4, impls
        assert len(profiles) >= 2, profiles
        for impl_id in impls:
            assert get_implementation(impl_id).impl_id == impl_id
        for qualified in profiles:
            assert get_profile(qualified).qualified_id == qualified
        assert get_adapter(R.FAMILY).family == R.FAMILY
