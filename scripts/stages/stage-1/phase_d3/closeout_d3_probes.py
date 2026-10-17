#!/usr/bin/env python3
"""Close out ONE D3 probes session, off-pod, at `$0`.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/closeout_d3_probes.py \
        [record|archive] --run d3_probes_<utc> --scr <scr> \
        [--balance-after <usd>] [--write]

The session is D2's shared-confirmation session under D3's layout, so the
SMALL-evidence plan, the rescoring diagnostics and the money model are D2's
closeout helpers (`closeout_d2_screening`, given this rung's names). What is
D3's own is the scientific reading -- `analyze_d3_probes.analyze_run`, the
registered EXPLORATORY comparison, never D2's verdict contract -- and the list
of proofs a complete closeout requires:

1. the run is a `d3_probes_` run and its one-use `D3ProbesAuthorization`
   loads, names this run, the registration and the contract re-derived on
   this tree from the recorded structural product;
2. the driver completed and recorded that same contract;
3. every registered probe was scored, and every unit VERIFIES: identity,
   admission, rows and generations, the pinned scorer re-run on the stored
   generations reproducing every row and figure, training records and
   identity, the session's arm verification of the D3 initialization -- and
   is aligned id-for-id with BOTH q2's and B's imported rows at its seed;
4. the session's evidence archive verifies against its manifest, as an
   archive and as the extracted tree;
5. a SECOND durable copy of that archive is on the relay, downloaded and
   re-hashed (`archive`, which writes `evidence/archive_relay_copy.json`);
6. the runner secured the required products, and the provider confirms the
   pod gone.

All present: COMPLETE, and the comparison is written with the closeout. Any
missing: INCOMPLETE, naming what is missing -- the money is booked, the
recoverable evidence is copied, and no scientific result is stated.

Without `--write` it prints what it would record and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_REL = "logs/stages/stage-1/phase_d3/runs"
SCHEMA = "aadistill.phase_d3.probes_closeout/v1"
COMPARISON = "evidence/d3_probes_comparison.json"
RELAY_COPY = "evidence/archive_relay_copy.json"
COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
#: The proofs a COMPLETE closeout requires, in the order they are reported.
REQUIRED_PROOFS = ("authorization_valid", "driver_complete", "contract_bound",
                   "all_probes_scored", "units_verified_and_paired",
                   "archive_verified", "second_durable_copy",
                   "evidence_secured", "teardown_verified")


class CloseoutRefused(RuntimeError):
    """Not a D3 probes run this closeout may record at all."""


def names() -> dict[str, str]:
    """This rung's layout, in the shape D2's closeout helpers take."""
    from stages.phase_d3 import d3_probes as P

    layout = P.RUNG_LAYOUT
    return {"audit": layout["audit_dirname"],
            "record": layout["evidence_filename"],
            "attested": layout["attested_name"],
            "diagnostics": "d3_probes_diagnostics.json",
            "diagnostics_schema": "aadistill.phase_d3.probes_diagnostics/v1",
            "shared_rung": P.SHARED_RUNG}


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_path(scr: Path) -> Path:
    from stages.phase_d3.autoinit_d3_probes_launch import ARCHIVE_BASENAME

    return scr / "store" / ARCHIVE_BASENAME


def sidecars(scr: Path, record: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Per probe, the token-id sidecars' sha256: they stay in the durable
    store and the archive, named here."""
    out = {}
    for probe in record.get("probes", []):
        gen = scr / "store" / "probes" / probe["probe_id"] / "generations"
        out[probe["probe_id"]] = {p.name: _sha(p) for p in
                                  sorted(gen.glob("*.token_ids.jsonl"))}
    return out


# --- the authorization -------------------------------------------------------------

def authorization(run: str, run_dir: Path,
                  repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The run's one-use grant, loaded through its type and checked against
    this run and the contract re-derived here. A run that is not a D3 probes
    run, or whose grant does not load, is REFUSED outright."""
    from stages.phase_d3 import d3_authorization as A
    from stages.phase_d3 import d3_design as D
    from stages.phase_d3 import d3_probes as P

    prefix = A.governance("probes").run_prefix
    if not run.startswith(prefix):
        raise CloseoutRefused(f"{run!r} is not a {prefix}* run")
    path = run_dir / "governance" / "authorization.json"
    if not path.is_file():
        raise CloseoutRefused(f"{run}: no authorization at {path}")
    try:
        auth = A.D3ProbesAuthorization.load(path)
    except Exception as exc:                                  # noqa: BLE001
        raise CloseoutRefused(f"{run}: the authorization does not load: "
                              f"{exc}") from exc
    contract = P.session_contract(repo_root, verify_bytes=False)
    derived = P.contract_hash(contract)
    checks = {"names_this_run": auth.run_id == run,
              "binds_the_registration": auth.plan_hash
              == D.registration_hash(repo_root),
              "binds_the_contract_derived_here": auth.contract_hash == derived,
              "authorizes_the_probes": auth.authorizes_d3_probes is True}
    raw = json.loads(path.read_text())
    return {"checks": checks, "valid": all(checks.values()),
            "authorization_sha256": raw.get("authorization_sha256"),
            "authorization_file_sha256": _sha(path),
            "authorized_session_commit": auth.authorized_session_commit,
            "contract_hash": auth.contract_hash, "derived_contract": derived,
            "hard_cap_usd": auth.hard_cap_usd,
            "cumulative_envelope_usd": (raw.get("money") or {}).get(
                "cumulative_envelope_usd"),
            "contract": contract}


# --- the archive -------------------------------------------------------------------

def archive_integrity(scr: Path) -> dict[str, Any]:
    """The session archive against its manifest: as an archive, and as the
    tree extracted from it on this host."""
    from aadistill.infrastructure.artifact_gate import (
        ArtifactManifest, verify_archive, verify_extracted,
    )

    manifest_path = scr / "store" / "manifest.json"
    archive = archive_path(scr)
    if not (manifest_path.is_file() and archive.is_file()):
        return {"verified": False, "archive": str(archive),
                "problems": ["no manifest or no archive came home"]}
    manifest = ArtifactManifest.load(manifest_path)
    problems = [*(f"archive: {p}" for p in verify_archive(archive, manifest)),
                *(f"extracted: {p}" for p in verify_extracted(
                    scr / "store" / "extracted", manifest))]
    if not manifest.ok:
        problems.append("the manifest itself is incomplete")
    return {"verified": not problems, "archive": str(archive),
            "archive_sha256": _sha(archive),
            "archive_bytes": archive.stat().st_size,
            "manifest_sha256": _sha(manifest_path),
            "n_files": len(manifest.entries), "problems": problems}


def relay_prefix(run: str) -> str:
    return f"stage1/phase_d3/{run}"


def copy_archive_to_relay(run: str, scr: Path) -> dict[str, Any]:
    """The second durable copy: upload once, download elsewhere, re-hash."""
    from huggingface_hub import HfApi, hf_hub_download

    from shared.deployment import MAIN_RELAY

    token = Path("~/.cache/huggingface/token").expanduser().read_text().strip()
    archive = archive_path(scr)
    sha = _sha(archive)
    if sha is None:
        raise CloseoutRefused(f"no archive at {archive}")
    remote = f"{relay_prefix(run)}/{archive.name}"
    HfApi(token=token).upload_file(
        path_or_fileobj=str(archive), path_in_repo=remote, repo_id=MAIN_RELAY,
        repo_type="model", commit_message=f"{run}: D3 probes evidence archive")
    with tempfile.TemporaryDirectory(prefix="d3p-archive-") as tmp:
        got = hf_hub_download(MAIN_RELAY, remote, repo_type="model",
                              token=token, cache_dir=tmp)
        downloaded = _sha(Path(got))
    return {"schema": "aadistill.phase_d3.evidence_archive_relay_copy/v1",
            "run_id": run, "local": str(archive), "relay": MAIN_RELAY,
            "remote": remote, "bytes": archive.stat().st_size, "sha256": sha,
            "downloaded_sha256": downloaded, "matched": downloaded == sha,
            "checked_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "_what": ("the session's manifest-verified evidence archive "
                      "(results, per-sample rows, raw generations, token-id "
                      "sidecars, admissions, attestation, configs, training "
                      "records, rescoring) copied off the dev host; restored "
                      "and re-hashed")}


# --- money -------------------------------------------------------------------------

def envelope(run: str, this_usd: float, envelope_usd: float | None,
             repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """This rung's cumulative envelope: every d3_probes_ subrun's booked
    closeout, with this run counted once."""
    from stages.phase_d3 import d3_authorization as A

    prior = A.consumed_by_prior_subruns(repo_root, "probes",
                                        exclude=run)["subruns"]
    consumed = round(sum(float(s["all_in_usd"]) for s in prior) + this_usd, 4)
    return {"cumulative_envelope_usd": envelope_usd,
            "prior_attempts": prior, "this_attempt_usd": this_usd,
            "consumed_usd": consumed,
            "remaining_usd": (None if envelope_usd is None
                              else round(float(envelope_usd) - consumed, 4))}


def limits_after(run: str, this_usd: float,
                 repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The four limits with this session booked exactly once."""
    from stages.phase_d3 import d3_budget as BUD

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "derive_budget", repo_root / "scripts/maintenance/consolidation/"
                                     "derive_budget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    booked = any(s.get("run_id") == run
                 for s in mod.derive(repo_root).get("sessions", []))
    live = BUD.live_limits()
    less = 0.0 if booked else this_usd
    return {"already_booked_by_the_deriver": booked,
            **{k: (round(v - less, 4) if k.endswith("remaining_usd")
                   and not k.startswith("engineering") else v)
               for k, v in live.items()},
            "_basis": ("derive_budget's four limits; this session's all-in "
                       "subtracted from the project, formal and package "
                       "remainders unless the deriver already books it "
                       "(after the run index records the run). The "
                       "engineering allowance is not spent by a formal "
                       "session")}


# --- the closeout ------------------------------------------------------------------

def build(run: str, scr: Path, *, balance_after: float | None,
          disk_gb: int, repo_root: Path = REPO_ROOT,
          rescore: Any = None, evidence_root: Path | None = None,
          run_dir: Path | None = None
          ) -> tuple[dict[str, Any], list[tuple[Path, Path]], dict[str, Any],
                     dict[str, Any] | None]:
    """(closeout, evidence copy plan, diagnostics, comparison-or-None).

    `run_dir` defaults to the run's directory in `repo_root`; a test passes
    its own so a simulated run never lands in the tree."""
    from stages.phase_d2 import closeout_d2_screening as C2
    from stages.phase_d3 import analyze_d3_probes as AN

    lay = names()
    run_dir = run_dir or (repo_root / RUNS_REL / run)
    auth = authorization(run, run_dir, repo_root)
    session_path = run_dir / "runtime" / "session.json"
    session = (json.loads(session_path.read_text())
               if session_path.is_file() else {})
    try:
        record = C2._session_record(scr, names=lay)
    except SystemExit:
        record = {}
    plan = (C2.evidence_plan(run_dir, scr, record, names=lay)
            if record else
            [(scr / "launch.log", run_dir / "runtime" / "launcher.log"),
             *((p, run_dir / "runtime" / p.name)
               for p in sorted(scr.glob("watchdog_*.jsonl")))])
    diag = C2.diagnostics(scr, record, names=lay) if record else {}
    contract = auth["contract"]
    registered = [p["probe_id"] for p in contract["probes"]]
    scored = {p["probe_id"] for p in record.get("probes", [])
              if p.get("scored")}

    comparison, refused = None, None
    if record.get("status") == "COMPLETE":
        try:
            comparison = AN.analyze_run(
                {**record, "run_id": run}, scr / "store",
                repo_root=repo_root, evidence_root=evidence_root,
                rescore=rescore)
        except AN.AnalysisRefused as exc:
            refused = str(exc)
    archive = archive_integrity(scr)
    relay_copy = (json.loads((run_dir / RELAY_COPY).read_text())
                  if (run_dir / RELAY_COPY).is_file() else None)
    gone = bool(session.get("provider_confirms_gone"))
    state = session.get("final_pod_state") or {}
    teardown = bool(gone and state.get("exists") is False
                    and state.get("billing") is False)
    secured = session.get("required_products_secured") or {}
    proofs = {
        "authorization_valid": auth["valid"],
        "driver_complete": record.get("status") == "COMPLETE",
        "contract_bound": bool(record) and record.get("contract_hash")
        == auth["contract_hash"] == auth["derived_contract"],
        "all_probes_scored": scored == set(registered),
        "units_verified_and_paired": comparison is not None,
        "archive_verified": archive["verified"],
        "second_durable_copy": bool(
            relay_copy and relay_copy.get("matched") is True
            and relay_copy.get("sha256") == archive.get("archive_sha256")),
        "evidence_secured": secured.get("ok") is True,
        "teardown_verified": teardown}
    missing = [k for k in REQUIRED_PROOFS if proofs[k] is not True]
    complete = not missing
    cash = C2.money(session, disk_gb=disk_gb, balance_after=balance_after)
    this_usd = float(cash["all_in_usd"])
    result = None
    if complete:
        cq, cb = (comparison["comparisons"][a]["correct_overall"]
                  for a in ("q2", "B"))
        uq, ub = (comparison["comparisons"][a]["usable_rollout_descriptive"]
                  for a in ("q2", "B"))
        result = {
            "primary_label_vs_q2": cq["label"],
            "anchor_label_vs_B": cb["label"],
            "vs_q2": {k: cq[k] for k in ("delta", "ci_two_sided_95",
                                         "per_seed_delta", "seeds_positive",
                                         "seeds_negative", "candidate_mean",
                                         "comparator_mean")},
            "vs_B": {k: cb[k] for k in ("delta", "ci_two_sided_95",
                                        "per_seed_delta", "seeds_positive",
                                        "seeds_negative", "candidate_mean",
                                        "comparator_mean")},
            "usable_vs_q2_descriptive": {k: uq[k] for k in (
                "delta", "ci_two_sided_95", "per_seed_delta")},
            "usable_vs_B_descriptive": {k: ub[k] for k in (
                "delta", "ci_two_sided_95", "per_seed_delta")},
            "comparison": COMPARISON}
    per_probe = {}
    for probe in record.get("probes", []):
        pid = probe["probe_id"]
        verified = ((comparison or {}).get("units_verified") or {}).get(pid)
        per_probe[pid] = {
            "seed": probe.get("seed"), "scored": bool(probe.get("scored")),
            "correct_overall": probe.get("correct_overall"),
            "usable_rollout_rate": probe.get("usable_rollout_rate"),
            "trained_sha256": probe.get("trained_sha256"),
            "result_sha256": probe.get("result_sha256"),
            "token_id_sidecar": probe.get("token_id_sidecar"),
            "verified": None if verified is None else verified["verified"],
            "training": None if verified is None else verified["training"]}
    doc = {
        "schema": SCHEMA, "run_id": run, "stage_id": "1",
        "experiment_id": "phase_d3", "rung": "d3_probes",
        "outcome": COMPLETE if complete else INCOMPLETE,
        "classification": (
            "COMPLETE: three registered probes trained from the D3 "
            "initialization, generated, scored, verified off-pod and paired "
            "with q2's and B's imported units; the registered EXPLORATORY "
            "comparison applied" if complete else
            "INCOMPLETE: no scientific result is stated; missing "
            + ", ".join(missing)),
        "missing_proofs": missing, "proofs": proofs,
        "comparison_refused": refused,
        "recorded_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "session_commit": (session.get("session_commit_check") or {}).get(
            "session_commit"),
        "authorized_base_commit": auth["authorized_session_commit"],
        "authorization_sha256": auth["authorization_sha256"],
        "authorization_file_sha256": auth["authorization_file_sha256"],
        "authorization_checks": auth["checks"],
        "contract_hash": auth["contract_hash"],
        "registration_hash": contract["preregistration_hash"],
        "initialization": contract["initialization"],
        "seeds": contract["seeds"],
        "recovery": {k: contract["recovery"][k] for k in (
            "recipe_id", "recipe_digest", "data_digest")},
        "evaluation": contract["evaluation"],
        "battery": {k: contract["battery"][k] for k in (
            "role", "item_ids_sha256", "n_prompts", "n_scorable")},
        "comparators": {arm: [u["probe_id"] for u in units]
                        for arm, units in contract["comparator"].items()},
        "driver_status": record.get("status"),
        "runner_terminal": session.get("terminal"),
        "exploratory_result": result,
        "per_probe": per_probe,
        "runtime_comparability": {
            "training_runtime": record.get("training_runtime"),
            "attested_evaluation_protocol_hash":
                record.get("attested_evaluation_protocol_hash"),
            "attested_matches_the_shared_rung":
                record.get("attested_matches_the_shared_rung"),
            "scoring_contract": record.get("scoring_contract"),
            "arm_verification": record.get("arm_verification")},
        "durable_store": ({**C2.durable_store(scr, record),
                           "token_id_sidecars": sidecars(scr, record)}
                          if record else None),
        "archive": archive, "archive_relay_copy": relay_copy,
        "required_products_secured": secured or None,
        "trained_checkpoints": session.get("trained_unscored_preserved"),
        "artifact_loss_risk": session.get("artifact_loss_risk"),
        "provider_resource_created": bool(
            session.get("provider_resource_created")),
        "provider_teardown_verified": teardown,
        "teardown_evidence": {"provider_confirms_gone": gone,
                              "final_pod_state": state,
                              "teardown_gate": session.get("teardown_gate")},
        "pod": {"pod_id": session.get("pod_id"),
                "image_identity": session.get("image_identity"),
                "draws": session.get("draws")},
        "money": cash,
        "envelope": envelope(run, this_usd, auth["cumulative_envelope_usd"],
                             repo_root),
        "limits_after": limits_after(run, this_usd, repo_root),
        "diagnostics": f"evidence/{lay['diagnostics']}" if diag else None,
        "_claim_boundary": (
            "EXPLORATORY on an observed battery: descriptive two-sided 95% "
            "intervals conditional on the three seeds, no alpha, no "
            "SHARED_GO, no promotion. The mixed-domain battery does not "
            "measure fluid reasoning (AGENTS.md P10.2): no claim about rule "
            "induction, compositional generalization or interactive "
            "reasoning follows from it."),
        "what_this_authorizes": "nothing",
    }
    return doc, plan, diag, comparison if complete else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", nargs="?", default="record",
                    choices=("record", "archive"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--balance-after", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    scr = Path(args.scr)
    run_dir = REPO_ROOT / RUNS_REL / args.run
    if args.action == "archive":
        if not archive_integrity(scr)["verified"]:
            raise SystemExit("the archive does not verify against its "
                             "manifest; nothing is copied")
        out = copy_archive_to_relay(args.run, scr)
        print(json.dumps(out, indent=1))
        if args.write:
            (run_dir / RELAY_COPY).parent.mkdir(parents=True, exist_ok=True)
            (run_dir / RELAY_COPY).write_text(json.dumps(out, indent=1) + "\n")
        return 0 if out["matched"] else 1
    doc, plan, diag, comparison = build(
        args.run, scr, balance_after=args.balance_after, disk_gb=args.disk_gb)
    present = [(s, d) for s, d in plan if s.is_file()]
    print(json.dumps({"outcome": doc["outcome"],
                      "missing_proofs": doc["missing_proofs"],
                      "comparison_refused": doc["comparison_refused"],
                      "exploratory_result": doc["exploratory_result"],
                      "copy": len(present),
                      "money": doc["money"], "envelope": doc["envelope"]},
                     indent=1, default=str))
    if not args.write:
        return 0
    from stages.phase_d2 import closeout_d2_screening as C2

    for src, dest in present:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    lay = names()
    try:
        record = C2._session_record(scr, names=lay)
        (run_dir / "evidence" / lay["record"]).write_text(
            json.dumps(record, indent=2, default=str) + "\n")
    except SystemExit:
        pass
    if diag:
        (run_dir / "evidence" / lay["diagnostics"]).write_text(
            json.dumps(diag, indent=1) + "\n")
    if comparison is not None:
        (run_dir / COMPARISON).write_text(json.dumps(comparison, indent=1)
                                          + "\n")
    (run_dir / "closeout").mkdir(parents=True, exist_ok=True)
    (run_dir / "closeout" / "outcome.json").write_text(
        json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {run_dir.relative_to(REPO_ROOT)}/closeout/outcome.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
