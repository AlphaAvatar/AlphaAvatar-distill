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
import base64
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
from aadistill.infrastructure.session_runner import (  # noqa: E402
    watchdog_journal_name)
from experiments.deployment import (  # noqa: E402
    POD_IMAGE, provider_cli_candidates)
from experiments.run_layout import (  # noqa: E402
    ArtifactSpec, claim_output_root, open_run, present_roles,
    rel_run_dir,
    record_run, require_output_claim, write_run_readmes,
)

#: This validation's key in `logs/runs/` and in the run index.
#:
#: `cuda_stage_f`, not `cuda-stage-f`: a run id validates as a single path
#: segment, and refusing a hyphen is how a separator or `..` cannot resolve
#: outside the run root. The dry run found this at `$0`, in a line only a
#: completed run reaches.
RUN_EXPERIMENT_ID = "cuda_stage_f"

#: NOT a pipeline stage. This validates that a CUDA device executes an operator
#: correctly; it trains nothing, evaluates nothing and produces no stage
#: artifact. Filing it under whichever stage the operator happens to belong to
#: would make "show me Stage 3 runs" return an engineering probe.
RUN_STAGE_ID = "shared"

#: role -> path inside this run. NOT C1's vocabulary: this validation has no
#: authorization snapshot to keep, no bundle, no driver evidence and no probe
#: results — it has a stdout transcript and whatever the suffix wrote. The areas
#: are shared; what lives in them is per-experiment, which is the property that
#: makes the same mechanism carry both.
RUN_ROLES: dict[str, str] = {
    "evidence": "evidence/evidence.json",
    "validation_stdout": "runtime/validation_stdout.txt",
    #: A directory: one journal per resource, named by pod id.
    "watchdog": "runtime/watchdog/",
    "artifacts": "artifacts/",
}

RUN_SPEC = ArtifactSpec(
    spec_id="cuda_engineering_run_v1",
    required=("evidence",),
    optional=("validation_stdout", "watchdog", "artifacts"))

#: The scratch-relative paths this run writes and later collects. Same rule as
#: the C1 session's, with this validation's own vocabulary: `--scr` is a second
#: writable output location, independent of `logs/runs/`, and a fresh run id
#: aimed at a previous subrun's scratch would otherwise collect that subrun's
#: stdout and artifacts as its own.
RUN_OUTPUTS: tuple[str, ...] = (
    "validation_stdout.txt", "artifacts", "watchdog_*.jsonl")

AUTHORIZATION = REPO_ROOT / "logs/validations/cuda-stage-f/v1/authorization.json"
VALIDATION_DIR = REPO_ROOT / "logs/validations/cuda-stage-f/v1"
#: THIS run's image, not the formal C1 one. `configs/infrastructure/pod_image.json`
#: describes an image whose `/opt/train/bin/python` is built by a long setup this
#: run does not perform.
DEPLOYMENT_CONFIG = REPO_ROOT / "configs/validation/cuda_engineering_deployment.json"
#: Cumulative cost across every resource and subrun of this task. A rerun does
#: NOT reset it and a replacement resource does NOT get a fresh allocation.
CAMPAIGN = REPO_ROOT / "logs/validations/cuda-stage-f/v1/campaign.json"

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
        #: Before the budget arithmetic and long before `create()`, so a scratch
        #: belonging to another subrun refuses at `$0` rather than after a pod
        #: exists.
        claim_output_root(self.scr, RUN_EXPERIMENT_ID, args.run_id,
                          outputs=RUN_OUTPUTS)
        self.auth = json.loads(AUTHORIZATION.read_text())
        rc = self.auth["resource_contract"]
        #: TASK-CUMULATIVE caps, not per-invocation allocations.
        self.soft_usd = float(rc["engineering_soft_cap_usd"])
        self.hard_usd = float(rc["total_resource_cost_ceiling_usd"])
        self.reserve_usd = float(rc["teardown_reserve_usd"])

        self.campaign = json.loads(CAMPAIGN.read_text())
        #: What earlier subruns already spent. Booked, never refunded by a
        #: rerun policy -- the first subrun's $0.0073 reduces what this one may
        #: use, which is the whole point of a cumulative cap.
        self.booked_usd = float(self.campaign["booked_usd"])
        self.remaining_total = round(self.hard_usd - self.booked_usd, 6)
        self.remaining_soft = round(self.soft_usd - self.booked_usd, 6)
        if self.remaining_total <= self.reserve_usd:
            raise Stop(
                f"${self.booked_usd:.4f} already booked leaves "
                f"${self.remaining_total:.4f} of the ${self.hard_usd:.2f} "
                f"ceiling, which does not clear the ${self.reserve_usd:.2f} "
                "teardown reserve. Stop.")
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
            "campaign_id": self.campaign["campaign_id"],
            "subrun_id": args.run_id,
            "campaign_booked_before_usd": self.booked_usd,
            "campaign_remaining_total_usd": self.remaining_total,
            "campaign_remaining_to_soft_usd": self.remaining_soft,
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

    def subrun_spent(self) -> float:
        """What THIS subrun has accrued."""
        if self.start_epoch is None or self.rate is None:
            return 0.0
        return (time.time() - self.start_epoch) / 3600.0 * self.rate

    def spent(self) -> float:
        """CAMPAIGN spend: what earlier subruns booked, plus this one."""
        return self.booked_usd + self.subrun_spent()

    def check_soft_cap(self, what: str) -> None:
        s = self.spent()
        if s >= self.soft_usd:
            raise Stop(f"cumulative soft cap reached before {what}: "
                       f"${s:.4f} >= ${self.soft_usd:.2f} "
                       f"(${self.booked_usd:.4f} booked by earlier subruns "
                       f"+ ${self.subrun_spent():.4f} here)")

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
        #: Against what is LEFT, minus the reserve -- not against the original
        #: ceiling. A useful attempt must fit and the teardown reserve must
        #: survive it.
        usable = self.remaining_total - self.reserve_usd
        if usable / rate * 60 < self.a.min_minutes:
            raise Stop(
                f"cheapest available is {gpu} at ${rate}/h. ${usable:.4f} is "
                f"usable (${self.remaining_total:.4f} left of the "
                f"${self.hard_usd:.2f} ceiling, minus the "
                f"${self.reserve_usd:.2f} reserve), which buys "
                f"{usable / rate * 60:.1f} min; {self.a.min_minutes} min is the "
                "declared minimum for a useful attempt")
        self.ev["accepted_quote"] = {"gpu": gpu, "secure_price_per_hour": rate,
                                     "stock": stock, "vram_gb": mem}
        self.say(f"accepted {gpu} at ${rate}/h — ${self.booked_usd:.4f} booked, "
                 f"${self.remaining_soft:.4f} to the soft stop "
                 f"({self.remaining_soft / rate * 60:.1f} min), "
                 f"${self.remaining_total:.4f} to the ceiling "
                 f"({self.remaining_total / rate * 60:.1f} min)")
        return gpu, rate

    # -- 2. exactly one create --------------------------------------------
    def create(self, gpu: str, quoted: float) -> None:
        if self.created:
            raise Stop("a create has already been attempted; there is no second")
        hard_minutes = self.remaining_total / quoted * 60
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
        #: Named after the resource it watches, the same convention the shared
        #: runner uses. This launcher creates exactly one pod, so the fixed name
        #: was not yet a collision -- but the collector globs for the shared
        #: convention, and a second spelling here would mean a journal nothing
        #: collects.
        journal = self.scr / watchdog_journal_name(self.pod_id)
        #: The watchdog terminates at what is LEFT of the CAMPAIGN ceiling,
        #: not at a fresh $0.40. It measures this pod's own clock, so the
        #: budget it is given is the remaining one.
        hard_minutes = self.remaining_total / self.rate * 60
        cmd = [sys.executable, str(REPO_ROOT / "scripts/pod/watchdog.py"),
               "--pod-id", self.pod_id,
               "--session-start-epoch", str(self.start_epoch),
               "--price-per-hour", str(self.rate),
               "--hard-minutes", f"{hard_minutes:.4f}",
               "--authorized-usd", f"{self.remaining_total:.4f}",
               "--journal", str(journal),
               "--poll-seconds", "20", "--verify-delay-seconds", "10",
               "--terminate-rounds", "3", "--verify-polls", "3"]
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        out = open(self.scr / watchdog_journal_name(self.pod_id, "out"),
                   "w")
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=REPO_ROOT, env=env,
                         start_new_session=True)
        self.watchdogs += 1
        self.ev["watchdog_launches"] = self.watchdogs
        self.ev["watchdog_journal"] = str(journal)
        self.say(f"watchdog detached (launch #{self.watchdogs}) — hard "
                 f"{hard_minutes:.1f} min / ${self.remaining_total:.4f} remaining")

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
        self.prepare_environment(target, repo)

    # -- setup: ONE interpreter, real exit codes, real readiness -----------
    def prepare_environment(self, target: SSHTarget, repo: str) -> None:
        """Build the environment the validation will actually run in.

        The 2026-09-09 failure was three separate mistakes stacked:
        installation used a bare `pip` while validation used a different
        `python`, so nothing tied them together; the install was piped through
        `tail`, so the shell reported tail's status and a refusal read as
        success; and the image's interpreter is PEP 668 externally managed, so
        the install had in fact been refused.

        Each is closed here. `py` is the ONE interpreter used to install, to
        probe and to validate. `-m pip` is that interpreter's own pip. The
        return code is pip's, captured through a marker rather than inferred
        from a pipeline. And the venv inherits system site-packages so the
        image's CUDA-enabled torch survives -- a clean venv would hide it, and
        resolving torch afresh can land a CPU build, which would turn a GPU
        validation into a CPU one without saying so.
        """
        venv = self.deploy["venv"]
        py = self.deploy["remote_python"]
        flags = "--system-site-packages" if venv["system_site_packages"] else ""
        r = target.run(
            f"python3 -m venv {flags} {venv['path']} && {py} -V; echo RC=$?",
            timeout=600)
        out = r.stdout + r.stderr
        self.ev["venv_setup"] = out[-1500:]
        if not re.search(r"RC=0\b", out):
            raise Stop(f"could not build the validation environment: {out[-500:]}")
        self.say(f"environment {venv['path']} ready ({r.stdout.strip().splitlines()[-2] if len(r.stdout.strip().splitlines())>1 else r.stdout.strip()})")

        deps = " ".join(self.deploy["pip_deps"])
        override = ("--break-system-packages "
                    if self.deploy.get("allow_break_system_packages") else "")
        #: THE interpreter's own pip, and PIP's OWN return code. `| tail`
        #: reports tail's status, which is always 0.
        r = target.run(f"{py} -m pip install --no-input {override}-q {deps}; "
                       f"echo PIP_RC=$?", timeout=1200)
        out = r.stdout + r.stderr
        m = re.search(r"PIP_RC=(\d+)", out)
        pip_rc = int(m.group(1)) if m else -1
        self.ev["pip_rc"] = pip_rc
        self.ev["pip_output_tail"] = out[-3000:]
        if pip_rc != 0:
            raise Stop(f"dependency install failed (pip rc={pip_rc}): {out[-800:]}")

        #: What actually got installed, recorded rather than assumed.
        r = target.run(f"{py} -m pip list --format=freeze 2>/dev/null | "
                       f"grep -Ei '^(torch|numpy|transformers|safetensors)=='",
                       timeout=300)
        self.ev["installed_versions"] = sorted(
            l.strip() for l in r.stdout.splitlines() if "==" in l)
        self.ev["validation_interpreter"] = py
        self.say(f"installed: {', '.join(self.ev['installed_versions']) or '(none reported)'}")
        self.readiness_probe(target, repo, py)

    def readiness_probe(self, target: SSHTarget, repo: str, py: str) -> None:
        """Prove the GPU is usable BEFORE spending on the validation.

        Not a print. The probe exits non-zero on every failure, and it checks
        the things that decide whether the validation can mean anything:
        imports resolve, CUDA is present, the capability floor is met, the
        declared dtype is supported, and a real allocation and matmul complete
        ON the device. `CUDA=False` beside an `IMPORTS_OK` marker is not
        readiness.
        """
        cap = self.deploy["capability"]
        major, minor = cap["compute_capability_min"]
        probe = f"""
import json, sys
import numpy, safetensors, transformers, torch
out = {{"torch": torch.__version__, "numpy": numpy.__version__,
       "transformers": transformers.__version__,
       "cuda_available": torch.cuda.is_available(),
       "cuda_runtime": torch.version.cuda}}
if not out["cuda_available"]:
    print(json.dumps(out)); print("PROBE_FAIL cuda_not_available"); sys.exit(2)
cc = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info(0)
out.update(device=torch.cuda.get_device_name(0), capability=list(cc),
           free_gib=round(free/2**30, 3), total_gib=round(total/2**30, 3),
           bf16_supported=torch.cuda.is_bf16_supported())
if cc < ({major}, {minor}):
    print(json.dumps(out)); print("PROBE_FAIL capability"); sys.exit(3)
if not out["bf16_supported"]:
    print(json.dumps(out)); print("PROBE_FAIL dtype"); sys.exit(4)
if out["free_gib"] < {cap["min_free_vram_gib"]}:
    print(json.dumps(out)); print("PROBE_FAIL vram"); sys.exit(5)
a = torch.randn(64, 64, device="cuda", dtype=torch.{cap["dtype"]})
b = (a @ a).float().sum().item()
out["device_matmul_finite"] = bool(b == b)
if not out["device_matmul_finite"]:
    print(json.dumps(out)); print("PROBE_FAIL matmul"); sys.exit(6)
print(json.dumps(out)); print("PROBE_OK")
"""
        (self.scr / "probe.py").write_text(probe)
        #: base64, not a heredoc. The probe is Python source full of quotes,
        #: braces and newlines, and it has to survive an ssh command string and
        #: a remote shell. base64 has no shell metacharacters, so there is
        #: nothing to quote wrongly -- and a quoting failure here would look
        #: like a device failure.
        blob = base64.b64encode(probe.encode()).decode()
        r = target.run(
            f"echo {blob} | base64 -d > /tmp/readiness_probe.py && "
            f"cd {repo} && PYTHONPATH=src:scripts {py} /tmp/readiness_probe.py; "
            f"echo PROBE_RC=$?", timeout=600)
        out = r.stdout + r.stderr
        m = re.search(r"PROBE_RC=(\d+)", out)
        probe_rc = int(m.group(1)) if m else -1
        self.ev["readiness_probe"] = {"rc": probe_rc, "output": out[-3000:]}
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.ev["readiness_probe"]["report"] = json.loads(line)
                except Exception:                                  # noqa: BLE001
                    pass
        if probe_rc != 0 or "PROBE_OK" not in r.stdout:
            raise Stop(
                f"GPU readiness probe failed (rc={probe_rc}); the validation "
                f"would fail for an environment reason: {out[-700:]}")
        rep = self.ev["readiness_probe"].get("report", {})
        self.say(f"GPU ready — {rep.get('device')} cc{rep.get('capability')} "
                 f"bf16={rep.get('bf16_supported')} free={rep.get('free_gib')}GiB "
                 f"torch={rep.get('torch')}")

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
        self.ev["subrun_cost_usd"] = round(self.subrun_spent(), 4)
        self.ev["campaign_cost_after_usd"] = round(self.spent(), 4)
        self.ev["provider_create_attempts"] = self.created
        self.ev["watchdog_launches"] = self.watchdogs

    def book_subrun(self, verdict: str, failure_class: str = "") -> None:
        """Append this subrun's cost to the CAMPAIGN ledger.

        Written whether it passed or failed. A failed subrun that did not book
        its spend would hand the next one a budget that does not exist.
        """
        doc = json.loads(CAMPAIGN.read_text())
        if any(x["subrun_id"] == self.a.run_id for x in doc["subruns"]):
            return                       # already booked; never double-count
        doc["subruns"].append({
            "subrun_id": self.a.run_id,
            "pod_id": self.pod_id or None,
            "gpu": (self.ev.get("accepted_quote") or {}).get("gpu"),
            "rate_usd_per_hour": self.rate,
            "elapsed_minutes": self.ev.get("elapsed_minutes"),
            "cost_usd": round(self.subrun_spent(), 4),
            "cost_basis": "elapsed x confirmed rate",
            "verdict": verdict,
            "failure_class": failure_class or None,
            "teardown_confirmed": bool(
                (self.ev.get("teardown") or {}).get("provider_confirms_gone")),
            "execution_sha": self.a.execution_sha or None,
            "record": f"logs/runs/cuda_stage_f/{self.a.run_id}/",
        })
        doc["booked_usd"] = round(sum(x["cost_usd"] for x in doc["subruns"]), 4)
        CAMPAIGN.write_text(json.dumps(doc, indent=1) + "\n")
        self.say(f"campaign ledger: {len(doc['subruns'])} subrun(s), "
                 f"${doc['booked_usd']:.4f} booked of ${self.hard_usd:.2f}")

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
            if self.created:
                self.book_subrun(verdict, self.classify(verdict, reason))
            self.write_evidence()
        return 0 if verdict.endswith("PASS") else 1

    @staticmethod
    def classify(verdict: str, reason: str) -> str:
        """Which KIND of failure this was. Recorded, not inferred later."""
        if verdict.endswith("PASS"):
            return ""
        r = reason.lower()
        for needle, kind in (("readiness probe", "setup"),
                             ("dependency install", "setup"),
                             ("environment", "setup"),
                             ("unpack", "setup"),
                             ("endpoint", "infrastructure"),
                             ("create", "infrastructure"),
                             ("soft cap", "budget"),
                             ("ceiling", "budget"),
                             ("validation exited", "operator")):
            if needle in r:
                return kind
        return "unknown"

    def write_evidence(self, repo_root: Path | None = None) -> None:
        """Into this run's own directory, then record its manifest.

        The three 2026-09-10 subruns wrote `evidence.json` and friends straight
        into the run root and recorded no manifest, so `record_run_index.py`
        could not see them: real runs existed on disk that the index reported as
        zero. This writes the same evidence into the shared areas and finishes
        by recording the run, which is what makes it discoverable.
        """
        repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
        #: Asked again at the collecting end, for the same reason the C1
        #: session asks: the open and the closeout are far apart, and what is
        #: recorded has to be what THIS subrun produced.
        require_output_claim(self.scr, RUN_EXPERIMENT_ID, self.a.run_id)
        layout = open_run(repo_root, RUN_EXPERIMENT_ID, self.a.run_id,
                          stage_id=RUN_STAGE_ID,
                          roles=RUN_ROLES)
        write_run_readmes(layout, experiment_id=RUN_EXPERIMENT_ID,
                          run_id=self.a.run_id,
                          stage_id=RUN_STAGE_ID, roles=RUN_ROLES)
        layout.path(RUN_ROLES["evidence"]).write_text(
            json.dumps(self.ev, indent=1) + "\n")
        src = self.scr / "validation_stdout.txt"
        if src.is_file():
            shutil.copy2(src, layout.path(RUN_ROLES["validation_stdout"]))
        #: Every journal and console this subrun's backstops left, by pod id.
        wd = layout.path(RUN_ROLES["watchdog"])
        wd.mkdir(parents=True, exist_ok=True)
        for j in sorted(self.scr.glob("watchdog_*.jsonl")) + sorted(
                self.scr.glob("watchdog_*.out")):
            shutil.copy2(j, wd / j.name)
        arts = self.scr / "artifacts"
        if arts.is_dir():
            shutil.copytree(arts, layout.path(RUN_ROLES["artifacts"]),
                            dirs_exist_ok=True)
        doc = record_run(
            layout, spec=RUN_SPEC,
            plan={"validation": "cuda-stage-f", "execution_sha":
                  self.a.execution_sha, "image": self.a.image},
            implementation={"launcher":
                            "scripts/validation/cuda_engineering_launch.py"},
            status={"verdict": self.ev.get("verdict"),
                    "pod_id": self.ev.get("pod_id"),
                    #: The keys `finish` actually writes, not the session
                    #: runner's `cost` block — this launcher has none of that.
                    "subrun_cost_usd": self.ev.get("subrun_cost_usd"),
                    "campaign_cost_after_usd": self.ev.get(
                        "campaign_cost_after_usd"),
                    "authorizes": "nothing"},
            roles=present_roles(layout, RUN_ROLES))
        print(f"\nverdict: {self.ev.get('verdict')}")
        print("evidence: "
              f"{rel_run_dir(doc['experiment_id'], doc['run_id'], RUN_STAGE_ID)}  "
              f"({len(doc['roles'])} role(s) recorded)")


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
