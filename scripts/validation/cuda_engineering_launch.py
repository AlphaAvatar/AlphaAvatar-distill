#!/usr/bin/env python3
"""Run the CUDA stage-F engineering validation on one authorized GPU.

    PYTHONPATH=src:scripts python scripts/validation/cuda_engineering_launch.py \
        --run-id <unique-engineering-run-id> --scr <scratch-dir>

A THIN entry point. It builds no provider controller and no authorization
framework of its own: `RunPodProvider`, `scripts/pod/watchdog.py`, `SSHTarget`
and the artifact/teardown helpers are the same ones every paid session uses.
What it adds is the one thing those sessions cannot express -- a run that needs
no bundle, no teacher weights, no venv build and no formal authorization, and
that must fit inside `$0.40`.

**It is not a C1 session.** It reads
`logs/validations/cuda-stage-f/v1/authorization.json`, an engineering
authorization recorded under this validation's own governance evidence. It
cannot read, and does not accept, a formal C1 grant.

The spend contract, enforced here rather than hoped for:

* **exactly one** provider-create attempt. Not a loop over candidates -- the
  canary loops, and a loop is a second create;
* the watchdog is detached the instant a pod id exists, before anything else;
* the accepted rate decides the minute budgets. There is no fixed allowance:
  `$0.25 / rate` is the work stop and `$0.40 / rate` the hard limit, so a
  dearer GPU buys proportionally less time;
* a pod that provisions ABOVE the accepted rate is registered, torn down and
  never used. It is not replaced;
* on success or on the first substantive exception, work stops immediately and
  teardown begins. The remaining allowance is not spent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aadistill.infrastructure.provider import (  # noqa: E402
    RunPodProvider, read_api_key)
from aadistill.infrastructure.remote import SSHTarget  # noqa: E402
from aadistill.runtime.run_layout import ArtifactSpec, RunLayout  # noqa: E402

from experiments.deployment import (  # noqa: E402
    POD_IMAGE, provider_cli_candidates)

AUTHORIZATION = REPO_ROOT / "logs/validations/cuda-stage-f/v1/authorization.json"
VALIDATION_DIR = REPO_ROOT / "logs/validations/cuda-stage-f/v1"
#: THIS run's image, not the formal C1 one. `configs/infrastructure/pod_image.json`
#: describes an image whose `/opt/train/bin/python` is built by a long setup this
#: run does not perform.
DEPLOYMENT_CONFIG = REPO_ROOT / "configs/validation/cuda_engineering_deployment.json"

#: Candidates that can satisfy cc >= 8.0 with native BF16 and >= 2 GiB. The
#: cheapest AVAILABLE one is chosen from a single bounded quote pass -- not a
#: predetermined SKU, and not a stock-polling loop.
CANDIDATES = (
    "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX A4000", "NVIDIA RTX A4500",
    "NVIDIA RTX A5000", "NVIDIA L4", "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA L40S",
)

#: What the pod needs. Everything else -- torch, CUDA -- is in the image.
SHIP = ("src", "scripts/validation", "scripts/experiments", "configs/validation")


class Stop(RuntimeError):
    """Stop work now, preserve what exists, and tear down."""


class Engineering:
    def __init__(self, args):
        self.a = args
        self.scr = Path(args.scr)
        self.scr.mkdir(parents=True, exist_ok=True)
        self.auth = json.loads(AUTHORIZATION.read_text())
        rc = self.auth["resource_contract"]
        self.soft_usd = float(rc["engineering_soft_cap_usd"])
        self.hard_usd = float(rc["total_resource_cost_ceiling_usd"])
        self.reserve_usd = float(rc["teardown_reserve_usd"])
        if rc["provider_create_attempts_max"] != 1 or rc["retries_or_replacement_pods"] != 0:
            raise Stop("this entry point implements exactly one create and no retry")
        if self.reserve_usd >= self.hard_usd:
            raise Stop("the teardown reserve must sit INSIDE the ceiling")

        self.deploy = json.loads(DEPLOYMENT_CONFIG.read_text())
        self.ev_deploy = {k: v for k, v in self.deploy.items()
                          if not k.startswith("_") and k != "schema"}

        self.key = os.environ.get("RUNPOD_API_KEY") or read_api_key(
            os.path.expanduser("~/.runpod/config.toml"))
        self.provider = RunPodProvider(self.key)
        self.cli = next((c for c in provider_cli_candidates()
                         if c and Path(c).is_file()), None)
        if not self.cli:
            raise Stop("no provider CLI among the deployment's candidates")

        self.pod_id = ""
        self.rate = None
        self.start_epoch = None
        self.created = 0
        self.watchdogs = 0
        self.ev: dict = {
            "schema": "aadistill.cuda_engineering_run/v1",
            "run_id": args.run_id,
            "scientific_use": False,
            "is_formal_c1": False,
            "authorizes": "nothing",
            "authorization": self.auth["authorization_id"],
            "authorization_sha256": self.auth["authorization_sha256"],
            "execution_sha": args.execution_sha,
            "deployment": None,
            "timeline": [],
        }
        self.ev["deployment"] = self.ev_deploy

    # -- reporting ---------------------------------------------------------
    def say(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}", flush=True)
        self.ev["timeline"].append({"utc": datetime.now(timezone.utc).isoformat(),
                                    "msg": msg})

    def spent(self) -> float:
        if self.start_epoch is None or self.rate is None:
            return 0.0
        return (time.time() - self.start_epoch) / 3600.0 * self.rate

    def check_soft_cap(self, what: str) -> None:
        s = self.spent()
        if s >= self.soft_usd:
            raise Stop(f"soft cap reached before {what}: ${s:.4f} >= "
                       f"${self.soft_usd:.2f}")

    # -- 1. one bounded quote pass ----------------------------------------
    def quote(self) -> tuple[str, float]:
        """One read-only pass. The cheapest available candidate wins."""
        quotes = []
        for gpu in CANDIDATES:
            try:
                d = self.provider._gql(
                    'query { gpuTypes(input:{id:"%s"}) { id securePrice '
                    'memoryInGb lowestPrice(input:{gpuCount:1}) { stockStatus } } }'
                    % gpu)
            except Exception as exc:                              # noqa: BLE001
                self.say(f"  {gpu}: quote failed ({type(exc).__name__})")
                continue
            rows = (d.get("data") or {}).get("gpuTypes") or []
            if not rows:
                continue
            sp = rows[0].get("securePrice")
            stock = (rows[0].get("lowestPrice") or {}).get("stockStatus")
            mem = rows[0].get("memoryInGb")
            self.say(f"  {gpu}: securePrice ${sp}/h, {mem}GB, stock {stock}")
            #: securePrice, never communityPrice: `pod create` provisions
            #: secure, and reading the community floor has under-reported two
            #: runs. A quote with no stock is not a candidate.
            if sp is not None and stock:
                quotes.append((float(sp), gpu, stock, mem))
        self.ev["quotes"] = [{"gpu": g, "secure_price_per_hour": s,
                              "stock": st, "vram_gb": m}
                             for s, g, st, m in sorted(quotes)]
        if not quotes:
            raise Stop("no candidate GPU is both quoted and in stock")
        quotes.sort()
        rate, gpu, stock, mem = quotes[0]
        #: The ceiling must buy enough time to be worth starting. Below this
        #: the run cannot finish setup, so creating a pod would only spend.
        if self.hard_usd / rate * 60 < self.a.min_minutes:
            raise Stop(
                f"cheapest available is {gpu} at ${rate}/h, which buys only "
                f"{self.hard_usd / rate * 60:.1f} min inside the ${self.hard_usd} "
                f"ceiling; {self.a.min_minutes} min is the declared minimum")
        self.ev["accepted_quote"] = {"gpu": gpu, "secure_price_per_hour": rate,
                                     "stock": stock, "vram_gb": mem}
        self.say(f"accepted {gpu} at ${rate}/h "
                 f"(soft ${self.soft_usd} = {self.soft_usd / rate * 60:.1f} min, "
                 f"hard ${self.hard_usd} = {self.hard_usd / rate * 60:.1f} min)")
        return gpu, rate

    # -- 2. exactly one create --------------------------------------------
    def create(self, gpu: str, quoted: float) -> None:
        if self.created:
            raise Stop("a create has already been attempted; there is no second")
        hard_minutes = self.hard_usd / quoted * 60
        deadline = datetime.now(timezone.utc) + timedelta(minutes=hard_minutes)
        argv = [self.cli, "pod", "create", "--image", self.deploy["image"],
                "--gpu-id", gpu, "--gpu-count", "1",
                "--container-disk-in-gb", str(self.deploy["container_disk_gb"]),
                "--volume-in-gb", "0", "--ports", "22/tcp",
                "--name", f"aad-cuda-eng-{self.a.run_id}",
                "--terminate-after", deadline.strftime("%Y-%m-%dT%H:%M:%SZ")]
        if self.deploy.get("min_cuda_version"):
            argv[argv.index("--ports"):argv.index("--ports")] = [
                "--min-cuda-version", self.deploy["min_cuda_version"]]
        self.ev["create_command"] = argv
        self.created = 1
        self.start_epoch = time.time()
        self.rate = quoted
        self.say(f"creating ONE pod (attempt 1 of 1) — {' '.join(argv[2:8])}")
        raw = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        (self.scr / "create_raw.txt").write_text(raw.stdout + raw.stderr)
        self.ev["create_stdout_tail"] = (raw.stdout + raw.stderr)[-2000:]

        pid = ""
        try:
            pid = json.loads(raw.stdout).get("id", "")
        except Exception:                                          # noqa: BLE001
            m = re.search(r'"id"\s*:\s*"([^"]+)"', raw.stdout + raw.stderr)
            pid = m.group(1) if m else ""
        if not pid:
            #: An empty response is NOT proof that nothing was created. Reconcile
            #: read-only against the unique run name before claiming $0.
            pid = self.reconcile()
        if not pid:
            raise Stop("create returned no id and no pod matches this run name; "
                       "no resource was created and none will be retried")
        self.register(pid)

    def reconcile(self) -> str:
        """Read-only: does a pod carrying THIS run's unique name exist?"""
        self.say("create returned no id — reconciling read-only by run name")
        try:
            d = self.provider._gql(
                "query { myself { pods { id name desiredStatus costPerHr } } }")
        except Exception as exc:                                   # noqa: BLE001
            self.ev["reconcile_error"] = f"{type(exc).__name__}: {exc}"
            self.say("  reconcile FAILED; treat the resource state as UNKNOWN")
            return ""
        pods = ((d.get("data") or {}).get("myself") or {}).get("pods") or []
        self.ev["reconcile_inventory"] = pods
        want = f"aad-cuda-eng-{self.a.run_id}"
        for p in pods:
            if p.get("name") == want:
                self.say(f"  reconciled: {p['id']} carries this run's name")
                return p["id"]
        return ""

    def register(self, pid: str) -> None:
        """Own the resource before anything else can go wrong."""
        self.pod_id = pid
        (self.scr / "pod_id").write_text(pid)
        (self.scr / "pod_start_epoch").write_text(str(self.start_epoch))
        self.ev["provider_resource_created"] = True
        self.ev["pod_id"] = pid
        self.say(f"created {pid} — registering and starting the watchdog NOW")
        self.launch_watchdog()

    def launch_watchdog(self) -> None:
        journal = self.scr / "watchdog.jsonl"
        hard_minutes = self.hard_usd / self.rate * 60
        cmd = [sys.executable, str(REPO_ROOT / "scripts/pod/watchdog.py"),
               "--pod-id", self.pod_id,
               "--session-start-epoch", str(self.start_epoch),
               "--price-per-hour", str(self.rate),
               "--hard-minutes", f"{hard_minutes:.4f}",
               "--authorized-usd", str(self.hard_usd),
               "--journal", str(journal),
               "--poll-seconds", "20", "--verify-delay-seconds", "10",
               "--terminate-rounds", "3", "--verify-polls", "3"]
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        out = open(self.scr / "watchdog.out", "w")
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=REPO_ROOT, env=env,
                         start_new_session=True)
        self.watchdogs += 1
        self.ev["watchdog_launches"] = self.watchdogs
        self.ev["watchdog_journal"] = str(journal)
        self.say(f"watchdog detached (launch #{self.watchdogs}) — hard "
                 f"{hard_minutes:.1f} min / ${self.hard_usd}")

    def verify_rate(self) -> None:
        """The rate it ACTUALLY provisioned at, not the quote."""
        try:
            d = self.provider._gql(
                'query { pod(input:{podId:"%s"}) { costPerHr } }' % self.pod_id)
            actual = ((d.get("data") or {}).get("pod") or {}).get("costPerHr")
        except Exception as exc:                                   # noqa: BLE001
            self.ev["actual_rate_error"] = f"{type(exc).__name__}: {exc}"
            return
        if actual is None:
            return
        actual = float(actual)
        self.ev["actual_price_per_hour"] = actual
        if actual > self.rate + 1e-9:
            raise Stop(
                f"provisioned at ${actual}/h, above the accepted ${self.rate}/h. "
                "The resource is registered and will be torn down; it is not "
                "replaced.")
        self.say(f"actual rate ${actual}/h confirmed at or below the quote")

    # -- 3. endpoint, ship, run -------------------------------------------
    def endpoint(self) -> tuple[str, str]:
        deadline = time.time() + self.a.startup_limit_min * 60
        while time.time() < deadline:
            self.check_soft_cap("the endpoint appeared")
            d = self.provider._gql(
                'query { pod(input:{podId:"%s"}) { runtime { ports '
                "{ ip publicPort privatePort type } } } }" % self.pod_id)
            rt = ((d.get("data") or {}).get("pod") or {}).get("runtime")
            for p in ((rt or {}).get("ports") or []):
                if p.get("privatePort") == 22 and p.get("ip"):
                    self.say(f"endpoint {p['ip']}:{p['publicPort']}")
                    return p["ip"], str(p["publicPort"])
            time.sleep(15)
        raise Stop(f"no SSH endpoint within {self.a.startup_limit_min} min")

    def ship(self, target: SSHTarget, host: str, port: str) -> None:
        tar = self.scr / "payload.tar.gz"
        with tarfile.open(tar, "w:gz") as t:
            for rel in SHIP:
                t.add(REPO_ROOT / rel, arcname=rel)
        size = tar.stat().st_size
        self.ev["payload_bytes"] = size
        self.say(f"shipping {size/1048576:.2f} MB of source")
        ws = POD_IMAGE["workspace_root"]
        repo = POD_IMAGE["checkout_root"]
        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null"]
        target.run(f"mkdir -p {repo}", timeout=60)
        subprocess.run(scp + [str(tar), f"root@{host}:{ws}/payload.tar.gz"],
                       capture_output=True, timeout=600)
        r = target.run(f"cd {repo} && tar xzf {ws}/payload.tar.gz && ls src scripts",
                       timeout=300)
        self.ev["unpack_rc"] = r.returncode
        if r.returncode != 0:
            raise Stop(f"unpack failed: {r.stdout[-400:]}{r.stderr[-400:]}")
        self.check_soft_cap("dependency install")
        deps = " ".join(self.deploy["pip_deps"])
        r = target.run(f"pip install --no-input -q {deps} 2>&1 | tail -5",
                       timeout=900)
        self.ev["pip_rc"] = r.returncode
        self.ev["pip_tail"] = (r.stdout + r.stderr)[-1500:]
        if r.returncode != 0:
            raise Stop(f"dependency install failed: {r.stdout[-400:]}")
        self.say("dependencies ready")

    def validate(self, target: SSHTarget) -> dict:
        """The one validation invocation. There is no second."""
        self.check_soft_cap("the validation run")
        repo = POD_IMAGE["checkout_root"]
        py = self.deploy["remote_python"]
        cmd = (f"cd {repo} && PYTHONPATH=src:scripts {py} "
               f"scripts/validation/cuda_engineering_check.py "
               f"--config configs/validation/cuda_engineering.json "
               f"--run-id {self.a.run_id}")
        self.ev["validation_command"] = cmd
        self.say(f"running: {cmd}")
        started = datetime.now(timezone.utc).isoformat()
        r = target.run(cmd, timeout=self.a.run_limit_min * 60)
        self.ev["validation"] = {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": r.returncode,
            "stdout": r.stdout[-20000:],
            "stderr": r.stderr[-8000:],
        }
        (self.scr / "validation_stdout.txt").write_text(r.stdout)
        (self.scr / "validation_stderr.txt").write_text(r.stderr)
        self.say(f"validation exited {r.returncode}")
        return self.ev["validation"]

    def collect(self, target: SSHTarget, host: str, port: str) -> None:
        repo = POD_IMAGE["checkout_root"]
        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null", "-r"]
        remote = f"{repo}/artifacts/validation"
        subprocess.run(scp + [f"root@{host}:{remote}", str(self.scr / "artifacts")],
                       capture_output=True, timeout=600)
        got = sorted(str(p.relative_to(self.scr))
                     for p in (self.scr / "artifacts").rglob("*") if p.is_file())
        self.ev["collected_files"] = got
        self.say(f"collected {len(got)} artifact file(s)")

    # -- 4. teardown, verified --------------------------------------------
    def teardown(self) -> None:
        if not self.pod_id:
            self.ev["teardown"] = {"needed": False}
            return
        self.say(f"tearing down {self.pod_id}")
        rm = subprocess.run([self.cli, "remove", "pod", self.pod_id],
                            capture_output=True, text=True, timeout=180)
        self.ev.setdefault("teardown", {})["remove_rc"] = rm.returncode
        self.ev["teardown"]["remove_output"] = (rm.stdout + rm.stderr)[-600:]
        #: A successful remove REQUEST is not confirmation. Poll the provider
        #: until the pod is gone from the inventory.
        gone, polls = False, []
        for _ in range(12):
            time.sleep(10)
            try:
                d = self.provider._gql(
                    "query { myself { pods { id desiredStatus } } }")
                pods = ((d.get("data") or {}).get("myself") or {}).get("pods") or []
            except Exception as exc:                               # noqa: BLE001
                polls.append({"error": f"{type(exc).__name__}: {exc}"})
                continue
            ids = [p["id"] for p in pods]
            polls.append({"pods": ids})
            if self.pod_id not in ids:
                gone = True
                break
        self.ev["teardown"]["provider_confirms_gone"] = gone
        self.ev["teardown"]["polls"] = polls
        self.ev["final_pod_inventory"] = polls[-1] if polls else None
        if gone:
            self.say("provider confirms the pod is gone and not billing")
        else:
            self.say("!! TEARDOWN NOT CONFIRMED — the pod id is retained and the "
                     "watchdog continues; treat this as an open billing risk")

    def finish(self, verdict: str, reason: str = "") -> None:
        self.ev["verdict"] = verdict
        if reason:
            self.ev["verdict_reason"] = reason
        elapsed = (time.time() - self.start_epoch) / 60 if self.start_epoch else 0.0
        self.ev["elapsed_minutes"] = round(elapsed, 3)
        self.ev["estimated_cost_usd"] = round(self.spent(), 4)
        self.ev["_cost_basis"] = (
            "elapsed wall clock x the provider-confirmed hourly rate. An "
            "ESTIMATE: the provider bills per second from provisioning, and "
            "this is the evidence available at teardown.")
        self.ev["provider_create_attempts"] = self.created
        self.ev["watchdog_launches"] = self.watchdogs

    def run(self) -> int:
        verdict, reason = "FAIL", ""
        try:
            gpu, rate = self.quote()
            if self.a.dry_run:
                self.say("DRY RUN: stopping before provider creation")
                #: Through the same `finally` as every other path, so the
                #: verdict cannot be overwritten by the initial FAIL.
                verdict, reason = "NOT RUN", "dry run: stopped before creation"
                return 0
            self.create(gpu, rate)
            self.verify_rate()
            host, port = self.endpoint()
            target = SSHTarget(host=host, port=int(port), user="root")
            self.ship(target, host, port)
            out = self.validate(target)
            self.collect(target, host, port)
            verdict = "CUDA ENGINEERING VALIDATION PASS" if out["returncode"] == 0 \
                else ("NOT RUN" if out["returncode"] == 3 else "FAIL")
            reason = "" if out["returncode"] == 0 else \
                f"validation exited {out['returncode']}"
        except Stop as exc:
            verdict, reason = "STOPPED", str(exc)
            self.say(f"STOP: {exc}")
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            verdict, reason = "FAIL", f"{type(exc).__name__}: {exc}"
            self.ev["traceback"] = traceback.format_exc()[-4000:]
            self.say(f"EXCEPTION: {reason}")
        finally:
            self.teardown()
            self.finish(verdict, reason)
            self.write_evidence()
        return 0 if verdict.endswith("PASS") else 1

    def write_evidence(self) -> None:
        """Through RunLayout, under one engineering run root."""
        #: `cuda_stage_f`, not `cuda-stage-f`: RunLayout refuses a hyphen in an
        #: id, and refusing is right -- it validates ids so a path separator or
        #: `..` cannot resolve outside the run root. The dry run found this at
        #: $0, in a line only a completed run reaches.
        layout = RunLayout(run_root=REPO_ROOT / "logs/runs",
                           experiment_id="cuda_stage_f",
                           run_id=self.a.run_id).create({
            "evidence": "evidence.json", "validation_stdout": "validation_stdout.txt",
            "watchdog": "watchdog.jsonl", "artifacts": "artifacts/"})
        spec = ArtifactSpec(spec_id="cuda_engineering_run_v1",
                            required=("evidence",),
                            optional=("validation_stdout", "watchdog", "artifacts"))
        layout.path("evidence.json").write_text(
            json.dumps(self.ev, indent=1) + "\n")
        for src_name, role in (("validation_stdout.txt", "validation_stdout.txt"),
                               ("watchdog.jsonl", "watchdog.jsonl")):
            src = self.scr / src_name
            if src.is_file():
                shutil.copy2(src, layout.path(role))
        arts = self.scr / "artifacts"
        if arts.is_dir():
            shutil.copytree(arts, layout.path("artifacts/"), dirs_exist_ok=True)
        ok, why = spec.check(r for r, rel in {"evidence": "evidence.json"}.items()
                             if layout.path(rel).exists())
        print(f"\nverdict: {self.ev.get('verdict')}")
        print(f"evidence: {layout.rel_root}  (artifacts {'complete' if ok else why})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--execution-sha", default="")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--remote-python", default="python")
    ap.add_argument("--disk-gb", type=int, default=20)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--run-limit-min", type=float, default=20.0)
    ap.add_argument("--min-minutes", type=float, default=25.0,
                    help="refuse to create if the ceiling buys less than this")
    ap.add_argument("--dry-run", action="store_true",
                    help="quote and stop before provider creation")
    return Engineering(ap.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
