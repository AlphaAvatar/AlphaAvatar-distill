#!/usr/bin/env python3
"""Dev-box orchestrator for E8 pod A: the contribution-guided depth search.

Built on the modules the 2026-08-09 canary verified live: detached start with a
durable descriptor, a provider-only watchdog that terminates and confirms
disappearance, continuous log relay, and manifest-driven collection behind the
teardown gate.

    PYTHONPATH=src setsid nohup python -u scripts/pod/e8a_launch.py \
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
STATUS = f"{WS}/e8a.status"
RUN_LOG = f"{WS}/e8a_run.log"
# Pod A trains nothing. Its whole product is a 28-entry layer list plus the
# 260-candidate trace that produced it, which is why there is no arm list here and
# no checkpoint fetch below.
SEARCH_OUT = "artifacts/stage1/e8_depth_search"


class E8A:
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
        with open(self.scr / "e8a_launch.log", "a") as f:
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
        self.plan = plan_session(
            price_per_hour=self.a.max_price, authorized_usd=self.a.authorized_usd,
            arms=0, steps_per_arm=0,
            step_time=StepTime(4.15, "unused; pod A does not train"),
            setup_minutes=45.0, transfer_minutes=5.0,
            other_phases=(
                Phase("contribution_search_260_evaluations",
                      self.a.search_minutes),
                Phase("self_consistency_and_positional_baseline", 4.0),
                Phase("artifact_manifest_and_verify", 8.0)),
            contingency_fraction=0.10, artifact_recovery_reserve_minutes=30.0)
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
                 "--name", "aadistill-e8a",
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
        subprocess.run(scp + [str(REPO_ROOT / "scripts/pod/e8a_setup.sh"),
                              f"root@{host}:{WS}/"], capture_output=True,
                       timeout=180)

        self.say(f"draw {draw}: running setup")
        target.run(
            f"cd {WS} && SESSION_COMMIT={self.a.session_commit} "
            f"BUNDLE_NAME={self.a.bundle} "
            f"TEACHER_REVISION={self.a.teacher_revision} "
            f"bash {WS}/e8a_setup.sh > {WS}/e8a_setup.log 2>&1; "
            f"echo SETUP_RC=$? >> {WS}/e8a_setup.log",
            timeout=self.a.setup_timeout_s)
        probe_out = target.run(
            f"grep -c 'MARKER:SETUP_DONE' {STATUS} 2>/dev/null || echo 0; "
            f"grep -c 'HOST_COLD' {STATUS} 2>/dev/null || echo 0; "
            f"tail -1 {WS}/e8a_setup.log", timeout=120).stdout.splitlines()
        done = probe_out[0].strip() if probe_out else "0"
        cold = probe_out[1].strip() if len(probe_out) > 1 else "0"
        self.ev["stages"].setdefault("setup", []).append(
            {"draw": draw, "setup_done": done, "host_cold": cold,
             "tail": probe_out[-1] if probe_out else ""})
        if cold not in ("", "0"):
            return "cold"
        if done in ("", "0"):
            tail = target.run(f"tail -40 {WS}/e8a_setup.log", timeout=120).stdout
            self.say(f"setup did not reach SETUP_DONE:\n{tail[-2000:]}")
            return "setup_failed"
        self.say(f"draw {draw}: setup complete — ${self.usd():.2f}")
        return "ok"

    # -- 4. run ------------------------------------------------------------
    def run(self) -> bool:
        if not self.make_plan():
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
            if outcome == "cold" and draw < self.a.host_draws:
                self.say(f"COLD HOST on draw {draw} — abandoning {self.pod_id} "
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
            self.say(f"ABORT: {self.a.host_draws} consecutive cold hosts")
            return False
        host, port = self.endpoint
        target = SSHTarget(host, port)
        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null"]

        # -- the driver, detached
        spent = self.usd()
        job = start_detached(target, JobSpec(
            job_id="e8a_driver", workdir=REPO,
            command=(f"/opt/train/bin/python scripts/pod/e8a_driver.py --stage all "
                     f"--spent-usd {spent:.3f} "
                     f"--soft-stop-usd {self.plan.soft_stop_usd:.2f} "
                     f"--authorized-usd {self.plan.hard_terminate_usd:.2f} "
                     f"--rate {self.price} "
                     f"--search-minutes {self.a.search_minutes}"),
            job_dir=f"{WS}/jobs", log_path=RUN_LOG, status_path=STATUS,
            env={"PYTHONPATH": f"{REPO}/src"}),
            start_timeout=120, verify_timeout=60)
        self.ev["driver_job"] = job.as_dict()
        self.say(f"driver detached, pid {job.pid}, confirmed by "
                 f"{job.confirmed_by} — ${self.usd():.2f}")
        self.save()

        # -- poll: relay logs, watch markers, watch the provider
        # `rounds.jsonl` is the search's own event stream: one line per committed
        # removal. Relaying it continuously means a lost pod still leaves the
        # rounds it finished, and the search resumes from them.
        relay_specs = (
            RelaySpec(f"{REPO}/{SEARCH_OUT}/rounds.jsonl",
                      "e8a_rounds.jsonl", required=False),
            RelaySpec(RUN_LOG, "e8a_run.log", required=False),
            RelaySpec(STATUS, "e8a.status", required=False))
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
        target.run(f"mkdir -p {REPO}/artifacts/audit/e8a_session && "
                   f"cp {RUN_LOG} {STATUS} {REPO}/artifacts/audit/e8a_session/",
                   timeout=120)
        man = f"{WS}/e8a_manifest.json"
        arc = f"{WS}/e8a_artifacts.tar.gz"
        r_man = target.run(
            f"{cc} manifest --root {REPO}/artifacts "
            f"--spec {REPO}/configs/stage3/e8/artifacts_a.json --out {man} "
            f"--settle-seconds {self.a.settle_seconds} "
            f"--completion-markers {REPO}/configs/stage3/e8/completion_markers_a.json",
            timeout=900)
        self.say(f"  manifest rc={r_man.returncode}\n{r_man.stdout.strip()[-900:]}")
        r_arc = target.run(f"{cc} archive --manifest {man} --out {arc}", timeout=1800)
        r_ver = target.run(f"{cc} verify-archive --manifest {man} --archive {arc}",
                           timeout=900)
        store = self.scr / "store"
        store.mkdir(exist_ok=True)
        for remote, local in ((man, store / "manifest.json"),
                              (arc, store / "e8a_artifacts.tar.gz")):
            subprocess.run(scp + [f"root@{host}:{remote}", str(local)],
                           capture_output=True, timeout=1800)

        local_ok, manifest = False, None
        if (store / "manifest.json").is_file():
            import tarfile
            extract = store / "extracted"
            extract.mkdir(exist_ok=True)
            try:
                with tarfile.open(store / "e8a_artifacts.tar.gz") as tar:
                    tar.extractall(extract, filter="data")
                manifest = ArtifactManifest.load(store / "manifest.json")
                problems = verify_extracted(extract, manifest)
                local_ok = not problems
                self.ev["local_hash_problems"] = problems
            except Exception as exc:  # noqa: BLE001
                self.ev["local_verify_error"] = f"{type(exc).__name__}: {exc}"

        # No checkpoints here — pod A produces a 28-entry layer list. What must
        # survive is the map and its trace, so they are fetched individually and
        # hashed on both sides rather than trusted to the archive alone.
        small = {"depth_map.json": f"{REPO}/{SEARCH_OUT}/depth_map.json",
                 "depth_search.json": f"{REPO}/{SEARCH_OUT}/depth_search.json",
                 "rounds.jsonl": f"{REPO}/{SEARCH_OUT}/rounds.jsonl",
                 "e8_frozen_depth_map.json":
                     f"{REPO}/artifacts/audit/e8_frozen_depth_map.json"}
        pod_hashes = target.run(
            "sha256sum " + " ".join(small.values()) + " 2>/dev/null", timeout=300)
        (store / "search_hashes.txt").write_text(pod_hashes.stdout)
        fetched = []
        for name, remote in small.items():
            dest = store / name
            rc = subprocess.run(
                scp + [f"root@{host}:{remote}", str(dest)],
                capture_output=True, timeout=600)
            ok = dest.is_file() and dest.stat().st_size > 0
            fetched.append({"file": name, "rc": rc.returncode, "present": ok})
            self.say(f"  fetched {name}: rc={rc.returncode} present={ok}")
        self.ev["search_artifacts_fetched"] = fetched
        # Cross-check the pod-side hash against the local copy for each file.
        import hashlib
        mismatches = []
        for line in pod_hashes.stdout.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            sha, remote = parts
            name = Path(remote).name
            local = store / name
            if local.is_file():
                got = hashlib.sha256(local.read_bytes()).hexdigest()
                if got != sha:
                    mismatches.append({"file": name, "pod": sha, "local": got})
        self.ev["search_artifact_hash_mismatches"] = mismatches
        if mismatches:
            self.say(f"  HASH MISMATCH on {[m['file'] for m in mismatches]}")
        if (store / "e8_frozen_depth_map.json").is_file():
            frozen = json.loads((store / "e8_frozen_depth_map.json").read_text())
            self.ev["frozen_depth_map"] = frozen
            self.say(f"  frozen map: kept {frozen['kept_teacher_layers']}")
            self.say(f"  removed {frozen['removed_teacher_layers']} vs positional "
                     f"{frozen['positional_removed']}; primary KL "
                     f"{frozen['primary_kl']:.6f} vs "
                     f"{frozen['positional_baseline_primary_kl']:.6f}")

        state = {
            "training_complete": True, "evaluation_complete": True,
            "artifact_manifest_created": (store / "manifest.json").is_file(),
            "required_files_present": bool(manifest and manifest.ok),
            "final_streams_quiescent": bool(
                manifest and manifest.final_streams_quiescent),
            "archive_created": r_arc.returncode == 0,
            "archive_contents_verified": r_ver.returncode == 0,
            "transfer_complete": (store / "e8a_artifacts.tar.gz").is_file(),
            "local_hashes_verified": local_ok,
            "checkpoint_hashes_matched": (
                all(f["present"] for f in fetched) and not mismatches),
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
        p = self.scr / "relay" / "e8a_rounds.jsonl"
        rounds = sum(1 for _ in p.open()) if p.is_file() else 0
        events = {"rounds_committed": rounds, "rounds_needed": 8}
        incomplete = (f"{SEARCH_OUT}/rounds.jsonl",)
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
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--max-price", type=float, default=0.99)
    ap.add_argument("--authorized-usd", type=float, default=2.71)
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    # 59.5, not 60: this is the figure `plan_e8_budget.py` derived from the
    # forward-pass arithmetic, and the planner refuses a $2.70 authorization at 60
    # by exactly one cent. The launcher must price what was authorized.
    ap.add_argument("--search-minutes", type=float, default=59.5,
                    help="priced search duration; the driver refuses to start a "
                         "search it cannot finish before the soft stop")
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=160.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/e8a_session_evidence.json")
    args = ap.parse_args()

    session = E8A(args)
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
    print(f"\nE8 pod A session {'COMPLETE' if ok else 'INCOMPLETE'} — "
          f"{REPO_ROOT / args.out}")
    return 0 if ok else 11


if __name__ == "__main__":
    raise SystemExit(main())
