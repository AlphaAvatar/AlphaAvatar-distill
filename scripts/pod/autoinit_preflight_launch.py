#!/usr/bin/env python3
"""Dev-box orchestrator for the AutoInitializer micro-preflight.

    PYTHONPATH=src setsid nohup python -u scripts/pod/autoinit_preflight_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

Built on the modules the 2026-08-09 canary verified live and on
`scripts/pod/AGENTS.md`: detached start with a durable descriptor, a
provider-only watchdog that terminates and confirms disappearance, continuous log
relay, and manifest-driven collection behind the teardown gate.

Four budget layers, in the order they are trusted:

1. the **authorization artifact** — loaded before a pod exists, bound to the
   preflight plan hash, and consulted before every priced decision;
2. the **soft stop** inside the driver — it refuses to start a stage it cannot
   finish before the artifact reserve;
3. this launcher's teardown at `ALL_DONE` or at any blocking stage failure,
   through the artifact gate;
4. the **independent watchdog**, which polls the provider, terminates at the hard
   threshold and verifies the pod is gone.

`--terminate-after` is set and never trusted; it has never been observed to fire.

Phase A is not reachable from this launcher. It starts one driver, that driver
has no stage 4, and the authorization refuses one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.authorization import (  # noqa: E402
    AuthorizationError, SpendAuthorization,
)
from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1  # noqa: E402
from aadistill.infrastructure.artifact_gate import (  # noqa: E402
    ArtifactManifest, evaluate_teardown, verify_extracted,
)
from aadistill.infrastructure.budget import Phase, StepTime, plan_session  # noqa: E402
from aadistill.infrastructure.log_relay import LogRelay, RelaySpec  # noqa: E402
from aadistill.infrastructure.provider import RunPodProvider, read_api_key  # noqa: E402
from aadistill.infrastructure.remote import (  # noqa: E402
    JobSpec, SSHTarget, probe, start_detached,
)

WS = "/workspace"
REPO = f"{WS}/aad"
STATUS = f"{WS}/autoinit_preflight.status"
RUN_LOG = f"{WS}/autoinit_preflight_run.log"
AUTH_PATH = "logs/autoinit_micro_preflight_authorization.json"
#: Dev-box-only artifacts the pod cannot fetch from the relay (~1.6 MB).
LOCAL_ASSETS = ("artifacts/stage1/state_eval_v1",
                "artifacts/stage3/recovery_search_v2")
CONTROLS = ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb")

PROBE_COMMAND = (
    "echo \"SETUP_DONE=$(grep -c 'MARKER:SETUP_DONE' {status} 2>/dev/null | tail -1)\"; "
    "echo \"HOST_COLD=$(grep -c 'HOST_COLD' {status} 2>/dev/null | tail -1)\"; "
    "echo \"SETUP_RC=$(grep -o 'SETUP_RC=[0-9]*' {log} 2>/dev/null | tail -1 | cut -d= -f2)\"; "
    "echo \"TAIL=$(tail -1 {log} 2>/dev/null)\""
)


def parse_setup_probe(stdout: str) -> dict:
    """Read the probe by LABEL, never by line position (see e8b: a $0.19 misread)."""
    out = {"setup_done": "0", "host_cold": "0", "setup_rc": "", "tail": ""}
    for line in stdout.splitlines():
        key, _, value = line.partition("=")
        if key.strip().lower() in out:
            out[key.strip().lower()] = value.strip()
    return out


class Preflight:
    #: Everything below names something a *session* owns rather than something
    #: the orchestration does: where the driver writes, which markers it can end
    #: on, which artifacts must survive, and what it produced that has to be
    #: fetched. A subclass that runs a different driver overrides these; the
    #: mechanism around them — detached start, watchdog, relay, budget,
    #: artifact gate, provider-confirmed teardown — is inherited untouched.
    #: The defaults are the preflight's own values, so its behaviour is
    #: unchanged by their existence.
    audit_dirname = "autoinit_preflight"
    evidence_filename = "preflight_evidence.json"
    archive_basename = "preflight_artifacts.tar.gz"
    spec_success = "configs/autoinit/preflight_artifacts.json"
    spec_failed = "configs/autoinit/preflight_artifacts_failed.json"
    #: Terminal markers other than ALL_DONE. `incomplete_markers` are the ones
    #: that mean the blocking stages passed, so whatever they produced exists
    #: and must be fetched (2026-08-13: `if terminal == "ALL_DONE"` deleted
    #: $2.82 of verified checkpoints).
    failure_markers = ("PREFLIGHT_FAILED", "PREFLIGHT_INCOMPLETE")
    incomplete_markers = ("PREFLIGHT_INCOMPLETE",)
    failure_note = ("a blocking stage failed — collecting evidence, then "
                    "tearing down. Permanent controls were not trained under "
                    "a configuration that has to change.")
    report_names = ("preflight_evidence.json", "attested_protocol.json",
                    "materialized_thresholds.json")

    def event_streams(self) -> tuple[str, ...]:
        """Append-only streams a torn-down session may have left mid-write."""
        return tuple(f"artifacts/stage3/{c}/train_log.jsonl" for c in CONTROLS)

    def fetch_products(self, host: str, target, stage2_passed: bool) -> list:
        """Fetch what this session PRODUCED and cannot regenerate for free.

        The preflight trains two permanent controls; they are the only such
        artifact, and they are fetched whenever they exist rather than only on
        a fully successful session.
        """
        fetched: list = []
        if not stage2_passed:
            return fetched
        for name in CONTROLS:
            dest = Path(self.a.ckpt_store) / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(
                ["timeout", f"{self.a.ckpt_fetch_limit_min}m", "scp", "-r",
                 "-P", str(target.port), "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 f"root@{host}:{REPO}/artifacts/stage3/{name}/checkpoints",
                 str(dest)], capture_output=True, timeout=None)
            size = sum(f.stat().st_size for f in dest.rglob("*")
                       if f.is_file()) if dest.exists() else 0
            fetched.append({"control": name, "rc": rc.returncode,
                            "bytes": size, "dest": str(dest)})
            self.say(f"  checkpoint {name}: rc={rc.returncode}, "
                     f"{size / 2**30:.2f} GiB -> {dest}")
        return fetched

    def __init__(self, a):
        self.a = a
        self.scr = Path(a.scr)
        self.scr.mkdir(parents=True, exist_ok=True)
        self.auth = SpendAuthorization.load(REPO_ROOT / AUTH_PATH)
        self.auth.require_plan(PREFLIGHT_PLAN_V1.plan_hash)
        # Before a pod can exist: the harness on disk must be the one this
        # authorization was granted against. An edited launcher, driver, setup
        # script or watchdog is an unrehearsed harness, and a paid run that
        # produces permanent artifacts must not be executed by one.
        self.harness = self.auth.require_harness(REPO_ROOT)
        self.key = os.environ.get("RUNPOD_API_KEY") or read_api_key(a.runpod_config)
        self.provider = RunPodProvider(self.key)
        self.cli = shutil.which("runpodctl") or os.path.expanduser(
            "~/.local/bin/runpodctl")
        if not Path(self.cli).is_file():
            raise SystemExit("runpodctl not found")
        self.ev: dict = {"schema": "aadistill.autoinit.preflight_session/v1",
                         "timeline": [], "stages": {},
                         "authorization": self.auth.as_dict(),
                         "preflight_plan_hash": PREFLIGHT_PLAN_V1.plan_hash,
                         "harness_source_digest": self.harness["digest"],
                         "harness_source_files": [f["path"] for f in
                                                  self.harness["files"]],
                         "phase_a_launched": False,
                         "phase_a_reachable_from_this_launcher": False}
        self.pod_id = ""
        self.start_epoch = 0.0
        self.price = None
        self.plan = None
        self.endpoint = ("", "")
        self.image_digest = ""

    # -- helpers ----------------------------------------------------------
    def say(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.scr / "launch.log", "a") as f:
            f.write(line + "\n")
        self.ev["timeline"].append(
            {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "elapsed_min": round(self.elapsed(), 2), "msg": msg})

    def elapsed(self) -> float:
        return (time.time() - self.start_epoch) / 60 if self.start_epoch else 0.0

    def usd(self) -> float:
        return self.elapsed() / 60 * (self.price or self.a.max_price)

    def save(self) -> None:
        (REPO_ROOT / self.a.out).write_text(
            json.dumps(self.ev, indent=2, default=str) + "\n")

    # -- 1. plan and price -------------------------------------------------
    def make_plan(self) -> bool:
        phases = (Phase("stage0_attestation_and_engine_probe", 8.0),
                  Phase("stage1_machine_gates", 22.0),
                  Phase("stage3_characterization_two_seeds", 36.0),
                  Phase("artifact_manifest_and_verify", 8.0),
                  Phase("artifact_synchronization", 6.0))
        self.plan = plan_session(
            price_per_hour=self.a.max_price,
            authorized_usd=self.auth.hard_cap_usd,
            arms=2, steps_per_arm=1023,
            step_time=StepTime(4.15, "E6b measured 4.15 s/step for this exact "
                                     "model, rung and card"),
            setup_minutes=45.0, other_phases=phases,
            eval_minutes_per_arm=0.0, transfer_minutes=6.0,
            contingency_fraction=0.10, artifact_recovery_reserve_minutes=30.0)
        self.ev["budget_plan"] = self.plan.as_dict()
        try:
            self.auth.require_within_cap(self.plan.hard_terminate_usd,
                                         what="planned hard threshold")
        except AuthorizationError as exc:
            self.say(f"ABORT: {exc}")
            return False
        self.say(f"budget: expected {self.plan.expected_minutes:.0f} min "
                 f"${self.plan.expected_usd:.2f} · soft "
                 f"${self.plan.soft_stop_usd:.2f} · hard "
                 f"${self.plan.hard_terminate_usd:.2f} "
                 f"(authorized ${self.auth.hard_cap_usd:.2f})")

        return self.check_gpu_offered()

    def check_gpu_offered(self) -> bool:
        """Is the GPU offered at or below the priced rate? Shared with subclasses."""
        d = self.provider._gql(
            'query { gpuTypes(input:{id:"%s"}) { id securePrice '
            'lowestPrice(input:{gpuCount:1}) { stockStatus } } }' % self.a.gpu)
        rows = (d.get("data") or {}).get("gpuTypes") or []
        if not rows:
            self.say(f"ABORT: {self.a.gpu} not offered")
            return False
        self.price = rows[0].get("securePrice")
        self.ev["quoted_price_per_hour"] = self.price
        self.ev["stock_status"] = (rows[0].get("lowestPrice") or {}).get("stockStatus")
        self.say(f"{self.a.gpu} ${self.price}/h, stock {self.ev['stock_status']}")
        if self.price is None or self.price > self.a.max_price:
            self.say(f"ABORT: ${self.price}/h above the priced ${self.a.max_price}/h")
            return False
        return True

    #: The detached job's id, and the command it runs. Subclasses override.
    job_id = "autoinit_preflight_driver"

    def driver_command(self) -> str:
        return (f"/opt/train/bin/python scripts/pod/autoinit_preflight_driver.py "
                f"--stage all --image-digest '{self.image_digest}' "
                f"--rate {self.price} --spent-usd {self.usd():.3f} "
                f"--soft-stop-usd {self.plan.soft_stop_usd:.2f} "
                f"--authorized-usd {self.plan.hard_terminate_usd:.2f}")

    def materialize_inputs(self, target, host: str, scp: list) -> bool:
        """Inputs the driver consumes but does not fetch. Nothing, by default."""
        return True

    def relay_precheck(self) -> bool:
        """Fail at $0 rather than after a 45-minute setup."""
        need = ["stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors",
                "stage3_recovery_corpus_v2/ladder_uniform/blocks.npz"]
        try:
            from huggingface_hub import HfApi
            present = set(HfApi().list_repo_files(
                "AlphaAvatar/aadistill-artifacts", repo_type="model"))
        except Exception as exc:                                  # noqa: BLE001
            self.say(f"ABORT: cannot list the relay: {exc!r}"[:200])
            return False
        missing = [f for f in need if f not in present]
        local_missing = [p for p in LOCAL_ASSETS if not (REPO_ROOT / p).is_dir()]
        self.ev["precheck"] = {"relay_needed": need, "relay_missing": missing,
                               "local_assets": list(LOCAL_ASSETS),
                               "local_missing": local_missing}
        if missing or local_missing:
            self.say(f"ABORT: relay missing {missing}, local missing {local_missing}")
            return False
        self.say(f"precheck OK: {len(need)} relay inputs, "
                 f"{len(LOCAL_ASSETS)} local assets")
        return True

    # -- 2. create ---------------------------------------------------------
    def create(self) -> bool:
        deadline = (datetime.now(timezone.utc)
                    + timedelta(minutes=self.plan.hard_terminate_minutes))
        for attempt in range(1, self.a.create_attempts + 1):
            raw = subprocess.run(
                [self.cli, "pod", "create", "--image", self.a.image,
                 "--gpu-id", self.a.gpu, "--gpu-count", "1",
                 "--container-disk-in-gb", str(self.a.disk_gb), "--volume-in-gb", "0",
                 "--min-cuda-version", "13.0", "--ports", "22/tcp",
                 "--name", "aadistill-autoinit-preflight",
                 "--terminate-after", deadline.strftime("%Y-%m-%dT%H:%M:%SZ")],
                capture_output=True, text=True, timeout=300)
            (self.scr / f"create_raw_{attempt}.txt").write_text(raw.stdout + raw.stderr)
            try:
                pid = json.loads(raw.stdout).get("id", "")
            except Exception:
                m = re.search(r'"id"\s*:\s*"([^"]+)"', raw.stdout + raw.stderr)
                pid = m.group(1) if m else ""
            if pid:
                try:
                    actual = json.loads(raw.stdout).get("costPerHr")
                    if actual is not None:
                        self.price = float(actual)
                except Exception:
                    pass
                if self.price > self.a.max_price:
                    self.say(f"ABORT: provisioned at ${self.price}/h — deleting")
                    subprocess.run([self.cli, "remove", "pod", pid],
                                   capture_output=True, timeout=120)
                    return False
                if not self.start_epoch:
                    self.start_epoch = time.time()
                    (self.scr / "pod_start_epoch").write_text(str(self.start_epoch))
                self.pod_id = pid
                (self.scr / "pod_id").write_text(pid)
                self.ev["pod_id"] = pid
                self.ev["actual_price_per_hour"] = self.price
                self.ev["terminate_after_utc"] = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
                self.say(f"created {pid} at ${self.price}/h")
                return True
            self.say(f"attempt {attempt}: create failed — "
                     f"{(raw.stdout + raw.stderr).strip()[:200]}")
            if attempt < self.a.create_attempts:
                time.sleep(self.a.create_retry_seconds)
        self.say("ABORT: could not create a pod")
        return False

    def launch_watchdog(self) -> Path:
        """Independent of the driver and of this launcher, by construction."""
        journal = self.scr / "watchdog.jsonl"
        cmd = [sys.executable, str(REPO_ROOT / "scripts/pod/watchdog.py"),
               "--pod-id", self.pod_id,
               "--session-start-epoch", str(self.start_epoch),
               "--price-per-hour", str(self.price),
               "--hard-minutes", str(self.plan.hard_terminate_minutes),
               "--authorized-usd", str(self.auth.hard_cap_usd),
               "--journal", str(journal), "--poll-seconds", "60"]
        out = open(self.scr / "watchdog.out", "w")
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=REPO_ROOT,
                         env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                         start_new_session=True)
        self.say(f"watchdog detached — hard {self.plan.hard_terminate_minutes:.0f} min "
                 f"= ${self.plan.hard_terminate_usd:.2f}")
        return journal

    def wait_endpoint(self):
        deadline = time.time() + self.a.startup_limit_min * 60
        i = 0
        while time.time() < deadline:
            d = self.provider._gql(
                'query { pod(input:{podId:"%s"}) { runtime { ports '
                '{ ip publicPort privatePort type } } } }' % self.pod_id)
            rt = ((d.get("data") or {}).get("pod") or {}).get("runtime")
            for p in (rt or {}).get("ports") or []:
                if p.get("privatePort") == 22 and p.get("type") == "tcp":
                    self.say(f"TCP 22 at {p['ip']}:{p['publicPort']} after "
                             f"{self.elapsed():.1f} min")
                    return str(p["ip"]), str(p["publicPort"])
            i += 1
            if i % 6 == 0:
                self.say(f"  starting ({i * 10}s) — ${self.usd():.2f}")
            time.sleep(10)
        return None

    def read_image_digest(self, target: SSHTarget) -> str:
        """The image the pod is really running, from the provider, not the arg."""
        d = self.provider._gql(
            'query { pod(input:{podId:"%s"}) { imageName machine { podHostId } } }'
            % self.pod_id)
        pod = ((d.get("data") or {}).get("pod") or {})
        name = pod.get("imageName") or self.a.image
        # Prefer a real digest when the host exposes one; otherwise the fully
        # qualified image name plus the driver/CUDA it presents. Either way it is
        # an observation, and Stage 0 refuses an empty one.
        rc = target.run(
            "cat /etc/podinfo/image_digest 2>/dev/null || "
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader", timeout=60)
        extra = (rc.stdout or "").strip().splitlines()[:1]
        digest = f"{name}@{extra[0]}" if extra else name
        self.ev["image_identity"] = {"image_name": name, "observed": extra,
                                     "digest": digest}
        return digest

    # -- 3. setup ----------------------------------------------------------
    def setup_on_draw(self, draw: int) -> str:
        ep = self.wait_endpoint()
        if not ep:
            return "no_endpoint"
        host, port = ep
        self.endpoint = (host, port)
        target = SSHTarget(host, port)
        for _ in range(30):
            if target.run("true", timeout=30).returncode == 0:
                break
            time.sleep(10)
        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null"]
        self.say(f"draw {draw}: ssh reachable — ${self.usd():.2f}")

        token = Path(self.a.token_src)
        if not token.is_file() or token.stat().st_size == 0:
            return "no_hf_token"
        target.run(f"mkdir -p {WS}/hf {WS}/assets && chmod 700 {WS}/hf", timeout=60)
        subprocess.run(scp + [str(token), f"root@{host}:{WS}/hf/token"],
                       capture_output=True, timeout=180)
        if target.run(f"test -s {WS}/hf/token", timeout=60).returncode != 0:
            return "empty_hf_token"
        # The two frozen search assets are dev-box-only and small; the uplink is
        # 0.72 MB/s, so 1.6 MB is ~3 s and needs no relay round trip.
        for asset in LOCAL_ASSETS:
            subprocess.run(scp + ["-r", str(REPO_ROOT / asset),
                                  f"root@{host}:{WS}/assets/"],
                           capture_output=True, timeout=600)
        subprocess.run(scp + [str(REPO_ROOT / "scripts/pod/autoinit_preflight_setup.sh"),
                              f"root@{host}:{WS}/"], capture_output=True, timeout=180)

        self.image_digest = self.read_image_digest(target)
        self.say(f"draw {draw}: image identity {self.image_digest}")
        self.say(f"draw {draw}: running setup")
        target.run(
            f"cd {WS} && SESSION_COMMIT={self.a.session_commit} "
            f"BUNDLE_NAME={self.a.bundle} "
            f"TEACHER_REVISION={self.a.teacher_revision} "
            f"UV_MAX_S={self.a.uv_max_s} TESTS_MAX_S={self.a.tests_max_s} "
            f"bash {WS}/autoinit_preflight_setup.sh > {WS}/setup.log 2>&1; "
            f"echo SETUP_RC=$? >> {WS}/setup.log",
            timeout=self.a.setup_timeout_s)
        result = parse_setup_probe(target.run(
            PROBE_COMMAND.format(status=STATUS, log=f"{WS}/setup.log"),
            timeout=120).stdout)
        self.ev["stages"].setdefault("setup", []).append({"draw": draw, **result})
        if result["host_cold"] not in ("", "0") or result["setup_rc"] == "90":
            return "cold"
        if result["setup_done"] in ("", "0"):
            tail = target.run(f"tail -40 {WS}/setup.log", timeout=120).stdout
            self.say(f"setup did not reach SETUP_DONE:\n{tail[-2000:]}")
            return "setup_failed"
        self.say(f"draw {draw}: setup complete — ${self.usd():.2f}")
        return "ok"

    # -- 4. run ------------------------------------------------------------
    def run(self) -> bool:
        if not self.make_plan() or not self.relay_precheck():
            return False
        for draw in range(1, self.a.host_draws + 1):
            if not self.create():
                return False
            self.ev.setdefault("watchdog_journals", []).append(
                str(self.launch_watchdog()))
            self.save()
            outcome = self.setup_on_draw(draw)
            if outcome == "ok":
                break
            if outcome in ("cold", "no_endpoint") and draw < self.a.host_draws:
                self.say(f"{outcome.upper()} on draw {draw} — abandoning "
                         f"{self.pod_id} and redrawing")
                subprocess.run([self.cli, "remove", "pod", self.pod_id],
                               capture_output=True, timeout=180)
                self.pod_id = ""
                continue
            self.say(f"ABORT after draw {draw}: {outcome}")
            if self.pod_id:
                self.teardown_now(f"setup {outcome}")
            return False
        else:
            self.say(f"ABORT: {self.a.host_draws} consecutive unusable draws")
            return False

        host, port = self.endpoint
        target = SSHTarget(host, port)
        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null"]

        # Inputs the driver needs but does not fetch itself. Empty for the
        # preflight, which trains its own; the continuation materializes two
        # existing controls here, by whichever transport it was told to use.
        if not self.materialize_inputs(target, host, scp):
            self.teardown_now("inputs did not materialize")
            return False

        job = start_detached(target, JobSpec(
            job_id=self.job_id, workdir=REPO, command=self.driver_command(),
            job_dir=f"{WS}/jobs", log_path=RUN_LOG, status_path=STATUS,
            env={"PYTHONPATH": f"{REPO}/src"}),
            start_timeout=120, verify_timeout=60)
        self.ev["driver_job"] = job.as_dict()
        self.say(f"driver detached, pid {job.pid}, confirmed by {job.confirmed_by} "
                 f"— ${self.usd():.2f}")
        self.save()

        relay = LogRelay(target, (
            RelaySpec(RUN_LOG, "preflight_run.log", required=False),
            RelaySpec(STATUS, "preflight.status", required=False),
            RelaySpec(f"{REPO}/artifacts/audit/{self.audit_dirname}/"
                      f"{self.evidence_filename}", self.evidence_filename,
                      required=False),
        ) + tuple(
            RelaySpec(f"{REPO}/artifacts/stage3/{c}/train_log.jsonl",
                      f"{c}.train_log.jsonl", required=False) for c in CONTROLS
        ), self.scr / "relay")

        last, terminal = "", ""
        deadline = time.time() + self.a.poll_limit_min * 60
        while time.time() < deadline:
            time.sleep(self.a.poll_seconds)
            r = relay.sync_once()
            if r.errors:
                self.say(f"  relay errors: {list(r.errors)[:2]}")
            st = target.run(f"tail -1 {STATUS} 2>/dev/null", timeout=60).stdout.strip()
            if st and st != last:
                last = st
                self.say(f"  {st} — ${self.usd():.2f}")
            if "MARKER:ALL_DONE" in st:
                terminal = "ALL_DONE"
                break
            hit = [m for m in self.failure_markers if f"MARKER:{m}" in st]
            if hit:
                terminal = hit[0]
                self.say(self.failure_note)
                break
            state = self.provider.get(self.pod_id)
            if not state.billing:
                terminal = "POD_GONE"
                self.say("the pod is gone — the watchdog acted")
                break
            live, _ = probe(target, job)
            if live != "ALIVE":
                terminal = f"DRIVER_{live}"
                self.say(f"driver no longer running: {live}")
                break
        self.ev["terminal"] = terminal or "POLL_LIMIT"
        self.say(f"polling ended: {self.ev['terminal']} — ${self.usd():.2f}")
        relay.sync_once()
        self.save()

        if terminal == "POD_GONE":
            self.finish_emergency()
            return False
        return self.collect_and_teardown(target, host, scp, terminal)

    # -- 5. collect --------------------------------------------------------
    def collect_and_teardown(self, target, host, scp, terminal: str) -> bool:
        cc = (f"cd {REPO} && PYTHONPATH={REPO}/src /opt/train/bin/python "
              "scripts/pod/collect_artifacts.py")
        audit = f"{REPO}/artifacts/audit/{self.audit_dirname}"
        target.run(f"mkdir -p {audit}/session && "
                   f"cp {RUN_LOG} {STATUS} {WS}/setup.log "
                   f"{audit}/session/ 2>/dev/null "
                   "|| true", timeout=120)
        # A blocking failure has a smaller required set: the controls do not
        # exist, and demanding them would block teardown on artifacts the run
        # correctly refused to produce.
        spec = self.spec_failed if terminal != "ALL_DONE" else self.spec_success
        man, arc = f"{WS}/manifest.json", f"{WS}/{self.archive_basename}"
        r_man = target.run(
            f"{cc} manifest --root {REPO}/artifacts --spec {REPO}/{spec} "
            f"--out {man} --settle-seconds {self.a.settle_seconds}", timeout=900)
        self.say(f"  manifest rc={r_man.returncode}\n{r_man.stdout.strip()[-900:]}")
        r_arc = target.run(f"{cc} archive --manifest {man} --out {arc}", timeout=1800)
        r_ver = target.run(f"{cc} verify-archive --manifest {man} --archive {arc}",
                           timeout=900)
        store = self.scr / "store"
        store.mkdir(exist_ok=True)
        for remote, local in ((man, store / "manifest.json"),
                              (arc, store / self.archive_basename)):
            subprocess.run(scp + [f"root@{host}:{remote}", str(local)],
                           capture_output=True, timeout=1800)

        local_ok, manifest = False, None
        if (store / "manifest.json").is_file():
            import tarfile
            extract = store / "extracted"
            extract.mkdir(exist_ok=True)
            try:
                with tarfile.open(store / self.archive_basename) as tar:
                    tar.extractall(extract, filter="data")
                manifest = ArtifactManifest.load(store / "manifest.json")
                problems = verify_extracted(extract, manifest)
                local_ok = not problems
                self.ev["local_hash_problems"] = problems
            except Exception as exc:                              # noqa: BLE001
                self.ev["local_verify_error"] = f"{type(exc).__name__}: {exc}"

        # The permanent controls are the only artifacts that cannot be
        # regenerated without paying again — so they are fetched whenever they
        # EXIST, which is whenever Stage 2 passed, not only when the whole
        # session succeeded.
        #
        # This was `if terminal == "ALL_DONE"` on 2026-08-13, and it destroyed
        # both controls of a $2.82 session: Stage 3's generation failed, which
        # is non-blocking by design precisely because "the controls still exist
        # and are kept" — and then the launcher skipped the fetch and deleted
        # the pod. `PREFLIGHT_INCOMPLETE` means Stages 0-2 passed. The comment
        # above was already right; the condition did not implement it.
        stage2_passed = bool(
            (self.ev.get("driver_stages") or {}).get("2")
            or terminal in ("ALL_DONE", *self.incomplete_markers))
        fetched = self.fetch_products(host, target, stage2_passed)
        self.ev["checkpoints_fetched"] = fetched
        for name in self.report_names:
            subprocess.run(scp + [
                f"root@{host}:{audit}/{name}",
                str(store / name)], capture_output=True, timeout=600)
            if (store / name).is_file():
                self.ev.setdefault("fetched_reports", []).append(name)

        done = terminal == "ALL_DONE"
        state = {
            "training_complete": done,
            "evaluation_complete": done,
            "artifact_manifest_created": (store / "manifest.json").is_file(),
            "required_files_present": bool(manifest and manifest.ok),
            "final_streams_quiescent": bool(manifest and manifest.final_streams_quiescent),
            "archive_created": r_arc.returncode == 0,
            "archive_contents_verified": r_ver.returncode == 0,
            "transfer_complete": (store / self.archive_basename).is_file(),
            "local_hashes_verified": local_ok,
            "checkpoint_hashes_matched": all(f["rc"] == 0 for f in fetched),
            "report_inputs_verified": local_ok,
        }
        decision = evaluate_teardown(
            state,
            emergency_budget=not done,
            emergency_reason=(
                "" if done else
                f"a blocking stage failed ({terminal}); the controls do not exist "
                "and must not be demanded. Evidence is collected under the "
                "reduced spec and the pod is torn down."),
            incomplete_event_streams=() if done else self.event_streams())
        self.ev["teardown_gate"] = decision.as_dict()
        self.ev["manifest_summary"] = (
            {"ok": manifest.ok, "entries": len(manifest.entries),
             "missing": manifest.missing,
             "still_being_written": manifest.still_being_written}
            if manifest else None)
        self.say(f"teardown gate: allowed={decision.allowed} "
                 f"failed={decision.failed_check}")
        self.save()
        if not decision.allowed:
            self.say("GATE BLOCKED — the launcher is NOT deleting the pod; the "
                     "watchdog remains the backstop")
            return False
        self.teardown_now("gate passed" if done else f"gate passed after {terminal}")
        return done

    def finish_emergency(self) -> None:
        events = {}
        for name in CONTROLS:
            p = self.scr / "relay" / f"{name}.train_log.jsonl"
            events[name] = sum(1 for _ in p.open()) if p.is_file() else 0
        decision = evaluate_teardown(
            {"training_complete": False, "evaluation_complete": False,
             "artifact_manifest_created": False, "required_files_present": False,
             "final_streams_quiescent": False},
            emergency_budget=True,
            emergency_reason=(f"the watchdog terminated the pod at the hard "
                              f"threshold ({self.plan.hard_terminate_minutes:.0f} "
                              f"min / ${self.plan.hard_terminate_usd:.2f})"),
            incomplete_event_streams=self.event_streams())
        self.ev["teardown_gate"] = decision.as_dict()
        self.ev["relayed_events"] = events
        self.say(f"EMERGENCY: only relayed snapshots survive — {events}")
        self.save()

    def teardown_now(self, why: str) -> None:
        self.say(f"deleting pod ({why})")
        subprocess.run([self.cli, "remove", "pod", self.pod_id],
                       capture_output=True, timeout=180)
        st = None
        for _ in range(18):
            time.sleep(10)
            st = self.provider.get(self.pod_id)
            if not st.billing:
                break
        self.ev["final_pod_state"] = {
            "exists": st.exists if st else None,
            "desired_status": st.desired_status if st else None,
            "billing": st.billing if st else None}
        self.ev["provider_confirms_gone"] = bool(st and not st.billing)
        self.ev["cost"] = {"price_per_hour": self.price,
                           "elapsed_minutes": round(self.elapsed(), 2),
                           "actual_usd": round(self.usd(), 4),
                           "authorized_usd": self.auth.hard_cap_usd,
                           "within_authorization": self.usd() <= self.auth.hard_cap_usd}
        self.say(f"pod deleted — {self.elapsed():.1f} min, ${self.usd():.2f}; "
                 f"provider confirms gone: {self.ev['provider_confirms_gone']}")
        self.save()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=0.99)
    ap.add_argument("--disk-gb", type=int, default=150)
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--ckpt-store", default="/home/ecs-user/aad-artifacts/autoinit")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=25)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=420.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/autoinit_preflight_session.json")
    args = ap.parse_args()

    session = Preflight(args)
    ok = False
    try:
        ok = session.run()
    except Exception as exc:                                      # noqa: BLE001
        session.ev["launcher_error"] = f"{type(exc).__name__}: {exc}"
        session.say(f"LAUNCHER ERROR: {type(exc).__name__}: {exc}")
        if session.pod_id:
            session.teardown_now("launcher error")
    session.ev["passed"] = bool(ok)
    session.ev["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session.save()
    print(f"\nmicro-preflight {'COMPLETE' if ok else 'INCOMPLETE'} — "
          f"{REPO_ROOT / args.out}. Phase A NOT launched.")
    return 0 if ok else 11


if __name__ == "__main__":
    raise SystemExit(main())
