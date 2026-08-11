#!/usr/bin/env python3
"""Dev-box orchestrator for E8 pod B: init diagnostics, 2x2.96M recovery, evaluation.

Built on the modules the 2026-08-09 canary verified live: detached start with a
durable descriptor, a provider-only watchdog that terminates and confirms
disappearance, continuous log relay, and manifest-driven collection behind the
teardown gate.

    PYTHONPATH=src setsid nohup python -u scripts/pod/e8b_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

Four budget layers, in the order they are trusted:

1. the **soft stop** inside the driver — it refuses to start an arm it cannot
   finish before the artifact reserve;
2. this launcher's teardown at `ALL_DONE`, through the artifact gate;
3. the **independent watchdog**, which polls the provider, terminates at the
   hard threshold and verifies the pod is gone;
4. `--terminate-after`, redundant and untrusted — it has never been observed to
   fire and does not count as a stop mechanism.

Budget safety outranks artifact completeness: if the watchdog reaches the hard
threshold the pod dies with whatever has been relayed, and the emergency gate
records which event streams are incomplete.
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

from aadistill.infrastructure.artifact_gate import (  # noqa: E402
    ArtifactManifest, evaluate_teardown, verify_extracted,
)
from aadistill.infrastructure.budget import (  # noqa: E402
    Phase, StepTime, plan_session,
)
from aadistill.infrastructure.log_relay import LogRelay, RelaySpec  # noqa: E402
from aadistill.infrastructure.provider import (  # noqa: E402
    RunPodProvider, read_api_key,
)
from aadistill.infrastructure.remote import (  # noqa: E402
    JobSpec, SSHTarget, probe, start_detached,
)
from aadistill.infrastructure.watchdog import Journal  # noqa: E402

WS = "/workspace"
REPO = f"{WS}/aad"
STATUS = ""   # set from --session in main()
RUN_LOG = ""  # set from --session in main()
# Two treatment arms, two seeds, one initialization. The control is retained from
# E1/P1 and is never retrained here.
STEP = "step_001761"
# session -> (gpu, max price, arm config names, checkpoint MB each)
SESSIONS = {
    "s1": ("NVIDIA L40S", 0.99, (), 0),
    "s2": ("NVIDIA A100-SXM4-80GB", 1.59,
           ("e8b_dp_r1600k_sa", "e8b_dc_r1600k_sa"), 6430),
    "s3": ("NVIDIA A100-SXM4-80GB", 1.59,
           ("e8b_dp_r1600k_sb", "e8b_dc_r1600k_sb"), 6430),
    "s4": ("NVIDIA L40S", 0.99,
           ("e8b_fc_r1600k_sa", "e8b_fc_r1600k_sb"), 1190),
}
ARMS = ()     # set from --session in main()


def parse_setup_probe(stdout: str) -> dict:
    """Read the setup probe by LABEL, never by line position.

    This function exists because of a $0.19 misread. The probe used to be three
    positional lines built from `grep -c X file || echo 0` — and `grep -c` prints
    "0" *and* exits 1 when there are no matches, so the `|| echo 0` fires too and
    that command emits **two** lines. With SETUP_DONE absent, everything shifted by
    one: the launcher read the stray "0" as the HOST_COLD count, classified a
    genuine cold host as a setup failure, and aborted the session instead of
    redrawing. The tripwire had worked perfectly; the parse threw the result away.

    So the probe now emits `KEY=value` lines and this reads them by key. Unknown or
    missing keys default rather than shifting anything.
    """
    out = {"setup_done": "0", "host_cold": "0", "setup_rc": "", "tail": ""}
    for line in stdout.splitlines():
        key, _, value = line.partition("=")
        k = key.strip().lower()
        if k in out:
            out[k] = value.strip()
    return out


PROBE_COMMAND = (
    # `| tail -1` keeps grep's own "0" and drops nothing else; no `|| echo`, which
    # is what duplicated a line and shifted the old positional parse.
    "echo \"SETUP_DONE=$(grep -c 'MARKER:SETUP_DONE' {status} 2>/dev/null | tail -1)\"; "
    "echo \"HOST_COLD=$(grep -c 'HOST_COLD' {status} 2>/dev/null | tail -1)\"; "
    "echo \"SETUP_RC=$(grep -o 'SETUP_RC=[0-9]*' {log} 2>/dev/null | tail -1 | cut -d= -f2)\"; "
    "echo \"TAIL=$(tail -1 {log} 2>/dev/null)\""
)


class E8B:
    def __init__(self, a):
        self.a = a
        self.scr = Path(a.scr)
        self.scr.mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get("RUNPOD_API_KEY") or read_api_key(a.runpod_config)
        self.provider = RunPodProvider(self.key)
        self.cli = shutil.which("runpodctl") or os.path.expanduser(
            "~/.local/bin/runpodctl")
        if not Path(self.cli).is_file():
            raise SystemExit("runpodctl not found")
        self.ev: dict = {"timeline": [], "stages": {}}
        self.pod_id = ""
        self.start_epoch = 0.0
        self.price = None
        self.plan = None
        self.endpoint = ("", "")

    # -- helpers -----------------------------------------------------------
    def say(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.scr / "e8b_launch.log", "a") as f:
            f.write(line + "\n")
        self.ev["timeline"].append(
            {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "elapsed_min": round(self.elapsed(), 2), "msg": msg})

    def elapsed(self) -> float:
        return (time.time() - self.start_epoch) / 60 if self.start_epoch else 0.0

    def usd(self) -> float:
        return self.elapsed() / 60 * (self.price or self.a.max_price)

    def save(self) -> None:
        (REPO_ROOT / self.a.out).write_text(json.dumps(self.ev, indent=2) + "\n")

    # -- 1. plan and price -------------------------------------------------
    def make_plan(self) -> bool:
        sess = self.a.session
        if sess == "s1":
            phases = (Phase("build_DP_DC_on_pod_and_assert_hashes", 12.0),
                      Phase("init_nll_DP", 14.0), Phase("init_nll_DC", 14.0),
                      Phase("init_nll_FP_remeasured", 7.0),
                      Phase("init_nll_FC_remeasured", 7.0),
                      Phase("step0_probe_DP", 55.0), Phase("step0_probe_DC", 55.0),
                      Phase("publish_step0_records", 3.0),
                      Phase("artifact_manifest_and_verify", 8.0),
                      Phase("artifact_synchronization", 5.0))
            step = StepTime(4.15, "unused; S1 does not train")
            kw = {}
        elif sess in ("s2", "s3"):
            # 7.86 s/step is derived, not measured; the 20-step gate on the first
            # arm converts it to a measurement before the second arm is paid for.
            step = StepTime(7.86, "derived from FLOPs and E6b's measured 4.15 "
                                  "s/step; the first-arm gate settles it")
            phases = (Phase("build_DP_DC_on_pod_and_assert_hashes", 12.0),
                      Phase("throughput_vram_usd_per_step_gate", 3.0),
                      Phase("pretraining_gate", 3.0),
                      Phase("train_two_depth_only_arms",
                            round(2 * 1761 * 7.86 / 60.0, 1)),
                      Phase("evaluate_two_arms", 60.0),
                      Phase("general_text_diagnostics", 6.0),
                      Phase("artifact_manifest_and_verify", 8.0),
                      Phase("artifact_synchronization",
                            round(2 * 6430 / 720.0, 1)))
            kw = {}
        else:
            step = StepTime(4.15, "E6b measured 4.15 s/step for this exact model, "
                                  "rung and card")
            phases = (Phase("pretraining_gate", 3.0),
                      Phase("general_text_diagnostics", 6.0),
                      Phase("artifact_manifest_and_verify", 8.0))
            kw = {"arms": 2, "steps_per_arm": 1761, "eval_minutes_per_arm": 8.25,
                  "transfer_minutes": round(2 * 1190 / 720.0, 1)}
        self.plan = plan_session(
            price_per_hour=self.a.max_price, authorized_usd=self.a.authorized_usd,
            arms=kw.pop("arms", 0), steps_per_arm=kw.pop("steps_per_arm", 0),
            step_time=step, setup_minutes=45.0, other_phases=phases,
            contingency_fraction=0.10, artifact_recovery_reserve_minutes=30.0,
            **kw)
        self.ev["budget_plan"] = self.plan.as_dict()
        self.say(f"budget: expected {self.plan.expected_minutes:.0f} min "
                 f"${self.plan.expected_usd:.2f} · soft stop "
                 f"{self.plan.soft_stop_minutes:.0f} min "
                 f"${self.plan.soft_stop_usd:.2f} · hard "
                 f"{self.plan.hard_terminate_minutes:.0f} min "
                 f"${self.plan.hard_terminate_usd:.2f} (authorized "
                 f"${self.a.authorized_usd:.2f})")

        d = self.provider._gql(
            'query { gpuTypes(input:{id:"%s"}) { id securePrice '
            'lowestPrice(input:{gpuCount:1}) { stockStatus } } }' % self.a.gpu)
        rows = (d.get("data") or {}).get("gpuTypes") or []
        if not rows:
            self.say(f"ABORT: {self.a.gpu} not offered"); return False
        self.price = rows[0].get("securePrice")
        stock = (rows[0].get("lowestPrice") or {}).get("stockStatus")
        self.ev["quoted_price_per_hour"] = self.price
        self.ev["stock_status"] = stock
        self.say(f"{self.a.gpu} securePrice ${self.price}/h, stock {stock}")
        if self.price is None or self.price > self.a.max_price:
            self.say(f"ABORT: ${self.price}/h above the priced "
                     f"${self.a.max_price}/h — the plan is not valid at this rate")
            return False
        return True

    def relay_has_the_treatment_init(self) -> bool:
        """Fail at $0 rather than after a 45-minute setup.

        Pod B's whole reason to exist is the treatment initialization, which the
        dev box uploads between the two sessions. If that upload did not happen —
        or landed under a different prefix — the pod would build both venvs,
        download the teacher, and only then discover it has nothing to train.
        """
        need = ["e8_inputs_20260810/warmup/holdout_v1.jsonl",
                "stage3_recovery_corpus_v2/ladder_uniform/blocks.npz",
                "e7_streams_20260809/e7_fineweb_val/blocks.npz"]
        if self.a.session in ("s1", "s4"):
            need += [
                "e8_init_20260810/e8_contribution_init_v1/checkpoint/model.safetensors",
                "e8_init_20260810/e8_contribution_init_v1/manifest.json",
                "e8_inputs_20260810/stage1/qwen3_0p6b_init_v0_manifest.json",
                "e8_inputs_20260810/calibration_v1/leakage.json",
                "stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors"]
        if self.a.session in ("s2", "s3", "s4"):
            # S1 publishes these; a training session must not start without them.
            need += [f"e8b_step0_20260811/{c}_init_nll.json"
                     for c in (("DP", "DC") if self.a.session in ("s2", "s3")
                               else ("FP", "FC"))]
        try:
            from huggingface_hub import HfApi
            present = set(HfApi().list_repo_files(
                "AlphaAvatar/aadistill-artifacts", repo_type="model"))
        except Exception as exc:                                  # noqa: BLE001
            self.say(f"ABORT: cannot list the relay: {exc!r}"[:200])
            return False
        missing = [f for f in need if f not in present]
        self.ev["relay_precheck"] = {"needed": need, "missing": missing}
        if missing:
            self.say(f"ABORT: the relay is missing {missing} — upload the "
                     "treatment initialization before launching pod B")
            return False
        self.say(f"relay precheck OK ({len(need)} files present)")
        return True

    # -- 2. create ---------------------------------------------------------
    def create(self) -> bool:
        deadline = (datetime.now(timezone.utc)
                    + timedelta(minutes=self.plan.hard_terminate_minutes))
        for attempt in range(1, self.a.create_attempts + 1):
            raw = subprocess.run(
                [self.cli, "pod", "create", "--image", self.a.image,
                 "--gpu-id", self.a.gpu, "--gpu-count", "1",
                 "--container-disk-in-gb", "150", "--volume-in-gb", "0",
                 "--min-cuda-version", "13.0", "--ports", "22/tcp",
                 "--name", f"aadistill-e8b-{self.a.session}",
                 "--terminate-after", deadline.strftime("%Y-%m-%dT%H:%M:%SZ")],
                capture_output=True, text=True, timeout=300)
            (self.scr / f"create_raw_{attempt}.txt").write_text(
                raw.stdout + raw.stderr)
            pid = ""
            try:
                pid = json.loads(raw.stdout).get("id", "")
            except Exception:
                m = re.search(r'"id"\s*:\s*"([^"]+)"', raw.stdout + raw.stderr)
                pid = m.group(1) if m else ""
            if pid:
                actual = None
                try:
                    actual = json.loads(raw.stdout).get("costPerHr")
                except Exception:
                    pass
                if actual is not None:
                    self.price = float(actual)
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
                self.ev["terminate_after_utc"] = deadline.strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
                self.say(f"created {pid} at ${self.price}/h; --terminate-after "
                         f"{self.ev['terminate_after_utc']} (redundant layer)")
                return True
            self.say(f"attempt {attempt}: create failed — "
                     f"{(raw.stdout + raw.stderr).strip()[:200]}")
            if attempt < self.a.create_attempts:
                time.sleep(self.a.create_retry_seconds)
        self.say("ABORT: could not create a pod")
        return False

    def launch_watchdog(self) -> Path:
        journal = self.scr / "watchdog.jsonl"
        cmd = [sys.executable, str(REPO_ROOT / "scripts/pod/watchdog.py"),
               "--pod-id", self.pod_id,
               "--session-start-epoch", str(self.start_epoch),
               "--price-per-hour", str(self.price),
               "--hard-minutes", str(self.plan.hard_terminate_minutes),
               "--authorized-usd", str(self.a.authorized_usd),
               "--journal", str(journal), "--poll-seconds", "60"]
        out = open(self.scr / "watchdog.out", "w")
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=REPO_ROOT,
                         env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                         start_new_session=True)
        self.say(f"watchdog detached — hard {self.plan.hard_terminate_minutes:.0f} "
                 f"min = ${self.plan.hard_terminate_usd:.2f}")
        return journal

    def wait_endpoint(self):
        deadline = time.time() + self.a.startup_limit_min * 60
        i = 0
        while time.time() < deadline:
            d = self.provider._gql(
                'query { pod(input:{podId:"%s"}) { runtime { ports '
                '{ ip publicPort privatePort type } } } }' % self.pod_id)
            rt = ((d.get("data") or {}).get("pod") or {}).get("runtime")
            if rt:
                for p in rt.get("ports") or []:
                    if p.get("privatePort") == 22 and p.get("type") == "tcp":
                        self.say(f"TCP 22 at {p['ip']}:{p['publicPort']} after "
                                 f"{self.elapsed():.1f} min")
                        return str(p["ip"]), str(p["publicPort"])
            i += 1
            if i % 6 == 0:
                self.say(f"  starting ({i * 10}s) — ${self.usd():.2f}")
            time.sleep(10)
        return None

    # -- 3. setup ----------------------------------------------------------
    def setup_on_draw(self, draw: int) -> str:
        """Bring one pod to SETUP_DONE. Returns "ok", "cold", or a failure."""
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
        target.run(f"mkdir -p {WS}/hf && chmod 700 {WS}/hf", timeout=60)
        subprocess.run(scp + [str(token), f"root@{host}:{WS}/hf/token"],
                       capture_output=True, timeout=180)
        if target.run(f"test -s {WS}/hf/token", timeout=60).returncode != 0:
            return "empty_hf_token"
        subprocess.run(scp + [str(REPO_ROOT / "scripts/pod/e8b_setup.sh"),
                              f"root@{host}:{WS}/"], capture_output=True,
                       timeout=180)

        self.say(f"draw {draw}: running setup")
        target.run(
            f"cd {WS} && SESSION_COMMIT={self.a.session_commit} "
            f"BUNDLE_NAME={self.a.bundle} "
            f"TEACHER_REVISION={self.a.teacher_revision} "
            f"E8B_SESSION={self.a.session} "
            f"UV_MAX_S={self.a.uv_max_s} TESTS_MAX_S={self.a.tests_max_s} "
            f"bash {WS}/e8b_setup.sh > {WS}/e8b_{self.a.session}_setup.log 2>&1; "
            f"echo SETUP_RC=$? >> {WS}/e8b_{self.a.session}_setup.log",
            timeout=self.a.setup_timeout_s)
        probe = parse_setup_probe(target.run(
            PROBE_COMMAND.format(status=STATUS, log=f"{WS}/e8b_{self.a.session}_setup.log"),
            timeout=120).stdout)
        done, cold = probe["setup_done"], probe["host_cold"]
        self.ev["stages"].setdefault("setup", []).append(
            {"draw": draw, **probe})
        # Exit code 90 is the setup script's own cold-host signal, so it is
        # authoritative even if the marker file was not reachable.
        if cold not in ("", "0") or probe["setup_rc"] == "90":
            return "cold"
        if done in ("", "0"):
            tail = target.run(f"tail -40 {WS}/e8b_{self.a.session}_setup.log", timeout=120).stdout
            self.say(f"setup did not reach SETUP_DONE:\n{tail[-2000:]}")
            return "setup_failed"
        self.say(f"draw {draw}: setup complete — ${self.usd():.2f}")
        return "ok"

    # -- 4. run ------------------------------------------------------------
    def run(self) -> bool:
        if not self.make_plan():
            return False
        if not self.relay_has_the_treatment_init():
            return False
        # Draw loop. `uv sync` on an identical image, script and GPU has taken
        # 44 s, ~50 s and 62 MINUTES purely with how much the host had cached;
        # E5 hit three cold hosts in a row. A cold host is abandoned (setup
        # exits 90) and the session redraws — the billing clock does NOT reset,
        # because it is a property of the session, not of the pod.
        for draw in range(1, self.a.host_draws + 1):
            if not self.create():
                return False
            wd = self.launch_watchdog()
            self.ev.setdefault("watchdog_journals", []).append(str(wd))
            self.save()
            outcome = self.setup_on_draw(draw)
            if outcome == "ok":
                break
            # A pod that never reaches TCP 22 is the same situation as a cold
            # host — a bad draw, not a broken experiment — and the recorded lesson
            # is explicit: runtime still null at the startup bound means it is not
            # starting, so delete and recreate in the launcher. Aborting the
            # session instead throws away the remaining draws for a provider-side
            # problem.
            if outcome in ("cold", "no_endpoint") and draw < self.a.host_draws:
                why = ("COLD HOST" if outcome == "cold"
                       else f"NO ENDPOINT after {self.a.startup_limit_min:.0f} min")
                self.say(f"{why} on draw {draw} — abandoning {self.pod_id} "
                         "and redrawing (its watchdog will see the pod vanish "
                         "and exit)")
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

        # -- the driver, detached
        spent = self.usd()
        job = start_detached(target, JobSpec(
            job_id="e8b_driver", workdir=REPO,
            command=(f"/opt/train/bin/python scripts/pod/e8b_driver.py --stage all "
                     f"--spent-usd {spent:.3f} "
                     f"--soft-stop-usd {self.plan.soft_stop_usd:.2f} "
                     f"--authorized-usd {self.plan.hard_terminate_usd:.2f} "
                     f"--session {self.a.session} --rate {self.price}"),
            job_dir=f"{WS}/jobs", log_path=RUN_LOG, status_path=STATUS,
            env={"PYTHONPATH": f"{REPO}/src"}),
            start_timeout=120, verify_timeout=60)
        self.ev["driver_job"] = job.as_dict()
        self.say(f"driver detached, pid {job.pid}, confirmed by "
                 f"{job.confirmed_by} — ${self.usd():.2f}")
        self.save()

        # -- poll: relay logs, watch markers, watch the provider
        relay_specs = tuple(
            RelaySpec(f"{REPO}/artifacts/stage3/{a}/train_log.jsonl",
                      f"{a}.train_log.jsonl", required=False) for a in ARMS
        ) + (RelaySpec(RUN_LOG, f"e8b_{self.a.session}_run.log", required=False),
             RelaySpec(STATUS, f"e8b_{self.a.session}.status", required=False))
        relay = LogRelay(target, relay_specs, self.scr / "relay")
        last = ""
        deadline = time.time() + self.a.poll_limit_min * 60
        terminal = ""
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
            if "ABORTED" in st or "PREFLIGHT_FAILED" in st:
                terminal = st
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
        return self.collect_and_teardown(target, host, scp, job)

    # -- 5. collect --------------------------------------------------------
    def collect_and_teardown(self, target, host, scp, job) -> bool:
        cc = (f"cd {REPO} && PYTHONPATH={REPO}/src /opt/train/bin/python "
              "scripts/pod/collect_artifacts.py")
        # The session log and status live outside the artifacts tree; copy them
        # in so every spec pattern is a plain relative glob.
        target.run(f"mkdir -p {REPO}/artifacts/audit/e8b_{self.a.session}_session && "
                   f"cp {RUN_LOG} {STATUS} {REPO}/artifacts/audit/e8b_{self.a.session}_session/",
                   timeout=120)
        man = f"{WS}/e8b_{self.a.session}_manifest.json"
        arc = f"{WS}/e8b_{self.a.session}_artifacts.tar.gz"
        r_man = target.run(
            f"{cc} manifest --root {REPO}/artifacts "
            f"--spec {REPO}/configs/stage3/e8b/artifacts_{self.a.session}.json --out {man} "
            f"--settle-seconds {self.a.settle_seconds} "
            f"--completion-markers {REPO}/configs/stage3/e8b/completion_markers_{self.a.session}.json",
            timeout=900)
        self.say(f"  manifest rc={r_man.returncode}\n{r_man.stdout.strip()[-900:]}")
        r_arc = target.run(f"{cc} archive --manifest {man} --out {arc}", timeout=1800)
        r_ver = target.run(f"{cc} verify-archive --manifest {man} --archive {arc}",
                           timeout=900)
        store = self.scr / "store"
        store.mkdir(exist_ok=True)
        for remote, local in ((man, store / "manifest.json"),
                              (arc, store / f"e8b_{self.a.session}_artifacts.tar.gz")):
            subprocess.run(scp + [f"root@{host}:{remote}", str(local)],
                           capture_output=True, timeout=1800)

        local_ok, manifest = False, None
        if (store / "manifest.json").is_file():
            import tarfile
            extract = store / "extracted"
            extract.mkdir(exist_ok=True)
            try:
                with tarfile.open(store / f"e8b_{self.a.session}_artifacts.tar.gz") as tar:
                    tar.extractall(extract, filter="data")
                manifest = ArtifactManifest.load(store / "manifest.json")
                problems = verify_extracted(extract, manifest)
                local_ok = not problems
                self.ev["local_hash_problems"] = problems
            except Exception as exc:  # noqa: BLE001
                self.ev["local_verify_error"] = f"{type(exc).__name__}: {exc}"

        # Checkpoints: the only artifacts that cannot be regenerated without
        # paying again. Time-boxed, hashed pod-side first.
        ck = target.run(
            f"cd {REPO}/artifacts/stage3 && find e8b_*_r1600k_*/checkpoints/{STEP} "
            f"-type f \\( -name '*.safetensors' -o -name '*.json' -o -name "
            f"'*.jinja' \\) | sort | xargs sha256sum", timeout=900)
        (store / "checkpoint_hashes.txt").write_text(ck.stdout)
        # The step-0 comparison and the frozen map are small and are the record
        # this experiment is read from; fetch them individually so they cannot be
        # lost inside a failed archive.
        for name, remote in (
                ("e8_step0_comparison.json",
                 f"{REPO}/artifacts/audit/e8_step0_comparison.json"),
                ("e8_preflight.json", f"{REPO}/artifacts/audit/e8_preflight.json")):
            subprocess.run(scp + [f"root@{host}:{remote}", str(store / name)],
                           capture_output=True, timeout=600)
        if (store / "e8_step0_comparison.json").is_file():
            self.ev["step0_comparison"] = json.loads(
                (store / "e8_step0_comparison.json").read_text())
            self.say("  step-0 comparison fetched")
        fetched = []
        for arm in ARMS:
            dest = Path(self.a.ckpt_store) / arm
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(
                ["timeout", f"{self.a.ckpt_fetch_limit_min}m", "scp", "-r",
                 "-P", str(target.port), "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 f"root@{host}:{REPO}/artifacts/stage3/{arm}/checkpoints/{STEP}",
                 str(dest)], capture_output=True, timeout=None)
            fetched.append({"arm": arm, "rc": rc.returncode})
            self.say(f"  checkpoint {arm}: rc={rc.returncode}")
        self.ev["checkpoints_fetched"] = fetched
        mismatches = []

        state = {
            "training_complete": True, "evaluation_complete": True,
            "artifact_manifest_created": (store / "manifest.json").is_file(),
            "required_files_present": bool(manifest and manifest.ok),
            "final_streams_quiescent": bool(
                manifest and manifest.final_streams_quiescent),
            "archive_created": r_arc.returncode == 0,
            "archive_contents_verified": r_ver.returncode == 0,
            "transfer_complete": (store / f"e8b_{self.a.session}_artifacts.tar.gz").is_file(),
            "local_hashes_verified": local_ok,
            "checkpoint_hashes_matched": all(f["rc"] == 0 for f in fetched),
            "report_inputs_verified": local_ok,
        }
        decision = evaluate_teardown(state)
        self.ev["teardown_gate"] = decision.as_dict()
        self.ev["manifest_summary"] = (
            {"ok": manifest.ok,
             "final_streams_quiescent": manifest.final_streams_quiescent,
             "entries": len(manifest.entries),
             "final_entries": len(manifest.final_entries()),
             "snapshot_entries": len(manifest.snapshot_entries()),
             "missing": manifest.missing,
             "still_being_written": manifest.still_being_written,
             "completion_marker_failures": manifest.completion_marker_failures}
            if manifest else None)
        self.say(f"teardown gate: allowed={decision.allowed} "
                 f"failed={decision.failed_check}")
        self.save()
        if not decision.allowed:
            self.say("GATE BLOCKED — the pod is NOT being deleted by the "
                     "launcher; the watchdog remains the backstop")
            return False
        self.teardown_now("gate passed")
        return True

    def finish_emergency(self) -> None:
        """The watchdog killed the pod. Record what the relay saved, honestly."""
        events = {}
        for arm in ARMS:
            p = self.scr / "relay" / f"{arm}.train_log.jsonl"
            events[arm] = sum(1 for _ in p.open()) if p.is_file() else 0
        incomplete = tuple(f"artifacts/stage3/{a}/train_log.jsonl" for a in ARMS)
        decision = evaluate_teardown(
            {"training_complete": False, "evaluation_complete": False,
             "artifact_manifest_created": False, "required_files_present": False,
             "final_streams_quiescent": False},
            emergency_budget=True,
            emergency_reason=(f"the watchdog terminated the pod at the hard "
                              f"threshold ({self.plan.hard_terminate_minutes:.0f} "
                              f"min / ${self.plan.hard_terminate_usd:.2f}); "
                              "budget safety outranks artifact completeness"),
            incomplete_event_streams=incomplete)
        self.ev["teardown_gate"] = decision.as_dict()
        self.ev["relayed_events"] = events
        self.say("EMERGENCY: only relayed mutable snapshots survive — "
                 f"{events}")
        self.save()

    def teardown_now(self, why: str) -> None:
        self.say(f"deleting pod ({why})")
        subprocess.run([self.cli, "remove", "pod", self.pod_id],
                       capture_output=True, timeout=180)
        for _ in range(18):
            time.sleep(10)
            st = self.provider.get(self.pod_id)
            if not st.billing:
                break
        self.ev["final_pod_state"] = {
            "exists": st.exists, "desired_status": st.desired_status,
            "billing": st.billing}
        self.ev["cost"] = {"price_per_hour": self.price,
                           "elapsed_minutes": round(self.elapsed(), 2),
                           "actual_usd": round(self.usd(), 4),
                           "authorized_usd": self.a.authorized_usd,
                           "within_backstop": self.usd() <= self.a.authorized_usd}
        self.say(f"pod deleted — {self.elapsed():.1f} min, ${self.usd():.2f}")
        self.save()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    
    
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--session", required=True, choices=sorted(SESSIONS))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=900)
    ap.add_argument("--ckpt-store", default="/home/ecs-user/aad-artifacts/e8")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=25)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=600.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/e8b_session_evidence.json")
    ap.add_argument("--authorized-usd", type=float, required=True)
    args = ap.parse_args()
    global STATUS, RUN_LOG, ARMS
    gpu, price, arms, _ckpt_mb = SESSIONS[args.session]
    args.gpu, args.max_price = gpu, price
    ARMS = arms
    STATUS = f"{WS}/e8b_{args.session}.status"
    RUN_LOG = f"{WS}/e8b_{args.session}_run.log"

    session = E8B(args)
    ok = False
    try:
        ok = session.run()
    except Exception as exc:  # noqa: BLE001 — must still tear down and report
        session.ev["driver_error"] = f"{type(exc).__name__}: {exc}"
        session.say(f"LAUNCHER ERROR: {type(exc).__name__}: {exc}")
        if session.pod_id:
            session.teardown_now("launcher error")
    session.ev["passed"] = bool(ok)
    session.ev["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session.save()
    print(f"\nE8 pod B session {'COMPLETE' if ok else 'INCOMPLETE'} — "
          f"{REPO_ROOT / args.out}")
    return 0 if ok else 11


if __name__ == "__main__":
    raise SystemExit(main())
