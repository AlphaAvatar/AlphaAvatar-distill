#!/usr/bin/env python3
"""The FORMAL admission of tool_pool@v1's two task sources (maintainer 2026-11-04), bound to the reviewed content. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_source_admission \\
        --pool data/stages/stage-2/teacher_modes/tool_pool_v1 --write logs/stages/stage-2/teacher_modes/audit/tool_source_admission.json

The maintainer's review of a143b7ed, section 1: Agent-Ark/Toucan-1.5M (SFT) and
nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 are formal tool-calling task sources, under the existing
audit, licence, revision, attribution, T1-T5 isolation and verification standards; the pilot's import record is reused
only when the pool's content is the reviewed ``be4652d7...`` (`REVIEWED_CONTENT`); no other source -- Pivot
Function-Calling, Nemotron-SFT-Agentic-v4 or anything not listed -- is admitted; every training target and every online
KD signal comes natively from the student's own teacher, and another model's output in a source record is task
material or registered reference verification only. The admission authorizes no other stage and no spend.

`record` writes the record from the pool's manifest; `problems` is the one check the issue (on the dev box, against the
pool: `token_matched_phase.kind_conditions`) and the pod (against the plan's content and its bound task file:
`tool_session.checked_plan`) apply. The signature follows the pilot admission's contract (`tool_pilot.
admission_problems`): every field filled, signed by the maintainer, approving exactly `APPROVAL`, the decision quoted,
an ISO date between the record's writing and today, and the content and sources it signs equal to the record's. The
agent fills it only on the maintainer's explicit written instruction to register the decision, as here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

SCHEMA = "aadistill.stage2.teacher_modes.tool_source_admission/v1"
RECORD = REPO / "logs/stages/stage-2/teacher_modes/audit/tool_source_admission.json"
REVIEWED_CONTENT = "be4652d77873b629197ca22e84d302d35d21d995a0684a0db02041f307906eff"
SIGNER = "maintainer"
APPROVAL = "formal tool-calling task sources"
SCOPE = ("tool_pool@v1's two sources as FORMAL tool-calling task sources: every split the pool assigns them (training, "
         "validation, calibration, state evaluation, reserve), for each lineage's teacher-native generation and the "
         "episodes admitted from it under the pool's verification")
CONDITIONS = [
    "every training target and every online KD signal comes natively from the student's own teacher (P17, P17.1); "
    "another model's output in a source record -- its assistant turns, gold calls, recorded tool responses -- is task "
    "material (the prompt, the tool schema, the replay environment) or registered reference verification only, never "
    "a student target",
    "the pool's audit, licence, revision, attribution, per-atom isolation, decontamination (T1-T5) and verification "
    "levels unchanged; the pilot's import record is reused only while the pool's content is REVIEWED_CONTENT",
    "each source's licence and attribution kept with every derived record (Apache-2.0 notice for Toucan; CC-BY-4.0 "
    "attribution for Pivot Conversational)",
    "third-party MCP and tool content may carry residual PII: generations stay outside git (P15); only counts, hashes "
    "and measurements enter the repository",
]
NOT_ADMITTED = ["nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1", "nvidia/Nemotron-SFT-Agentic-v4",
                "any dataset or revision not listed under sources"]
DOES_NOT = ["authorize a GPU, a pod or any spend",
            "authorize any experiment stage: a search, a training, a calibration, a state evaluation or a formal "
            "evaluation",
            "approve any SHORTFALL, proportion exception or approved_exceptions entry",
            "admit any source or revision not listed under sources"]
SIGNATURE_FIELDS = ("signed_by", "approves", "decision", "date", "pool_content_sha256", "sources")
LICENCES = {"Agent-Ark/Toucan-1.5M": "Apache-2.0",
            "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1": "CC-BY-4.0"}


def _sources(man: dict) -> list[str]:
    return sorted(f"{d}@{r}" for d, r in man.get("source_revisions") or [])


def record(man: dict, pool_json_sha256: str, *, decision: str, date: str, provenance: dict | None = None) -> dict:
    """The signed record for a pool manifest ``man`` (`tool_pool.load_pool`'s); ``decision`` and ``date`` are the
    maintainer's written decision, quoted; ``provenance`` how the pool's bytes were built and compared."""
    sources = _sources(man)
    return {"schema": SCHEMA, "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": f"SIGNED by the maintainer's written decision of {date} (signature below)",
            "scope": SCOPE, "pool_id": man.get("pool_id"),
            "pool_content_sha256": man["content_sha256"], "pool_json_sha256": pool_json_sha256,
            "reviewed_content_sha256": REVIEWED_CONTENT, "provenance": provenance,
            "sources": sources,
            "per_source": {name: {"datasets": v.get("datasets"), "converted_sha256": v.get("converted_sha256"),
                                  "tasks": v.get("tasks"),
                                  "licence": sorted({LICENCES.get(d.split("@")[0]) for d in v.get("datasets") or []},
                                                    key=str)}
                           for name, v in sorted((man.get("sources") or {}).items())},
            "tasks_by_split": {s: {"tasks": f.get("tasks"), "by_source": f.get("by_source"), "sha256": f.get("sha256")}
                               for s, f in sorted((man.get("files") or {}).items())},
            "conditions": CONDITIONS, "not_admitted": NOT_ADMITTED, "does_not": DOES_NOT,
            "signature": {"signed_by": SIGNER, "approves": APPROVAL, "decision": decision, "date": date,
                          "pool_content_sha256": man["content_sha256"], "sources": sources},
            "_signature": f"the maintainer's fields: signed_by {SIGNER!r}, approves exactly {APPROVAL!r}, the decision "
                          "quoted, the date (YYYY-MM-DD), and the pool content and sources copied from this record "
                          "(problems checks all of them); filled by the agent only on the maintainer's explicit written "
                          "instruction to register the decision",
            "authorizes": "nothing by itself: each kind's grant authorizes its sessions"}


def problems(rec: dict, *, pool_content_sha256: str, tasks=(), today: str | None = None) -> list[str]:
    """Why ``rec`` does not admit a generation reading a pool of ``pool_content_sha256`` and the ``tasks`` (dicts with
    ``source.dataset`` / ``source.revision``) -- empty when it does. Fail-closed on every field."""
    bad = []
    if rec.get("schema") != SCHEMA:
        return ["not a formal tool source admission record"]
    if (rec.get("scope"), rec.get("conditions"), rec.get("not_admitted"), rec.get("does_not")) != (
            SCOPE, CONDITIONS, NOT_ADMITTED, DOES_NOT):
        bad.append("its scope, conditions or exclusions are not exactly the formal admission's")
    if rec.get("pool_content_sha256") != pool_content_sha256:
        bad.append(f"it admits pool content {str(rec.get('pool_content_sha256'))[:12]}, not "
                   f"{str(pool_content_sha256)[:12]}")
    if rec.get("pool_content_sha256") != REVIEWED_CONTENT:
        bad.append("it admits a pool content other than the reviewed one")
    admitted = set(rec.get("sources") or [])
    stray = sorted({f"{(t.get('source') or {}).get('dataset')}@{(t.get('source') or {}).get('revision')}"
                    for t in tasks} - admitted)
    if stray:
        bad.append(f"tasks from sources it does not admit: {stray[:4]}")
    sig = rec.get("signature") or {}
    if not any(sig.get(k) for k in SIGNATURE_FIELDS):
        return bad + ["not signed by the maintainer"]
    if sig.get("signed_by") != SIGNER:
        bad.append(f"signed_by is {sig.get('signed_by')!r}, not the {SIGNER}")
    if sig.get("approves") != APPROVAL:
        bad.append(f"the signature approves {sig.get('approves')!r}, not exactly {APPROVAL!r}")
    if not isinstance(sig.get("decision"), str) or not sig["decision"].strip():
        bad.append("the signature carries no decision")
    try:
        day = dt.date.fromisoformat(str(sig.get("date")))
        start = dt.date.fromisoformat(str(rec.get("written_utc", ""))[:10])
        now = dt.date.fromisoformat(today) if today else dt.datetime.now(dt.timezone.utc).date()
        if not start <= day <= now:
            bad.append(f"the signature's date {day} is outside {start} .. {now}")
    except ValueError:
        bad.append(f"the signature's date {sig.get('date')!r} is not an ISO date")
    for k in ("pool_content_sha256", "sources"):
        if sig.get(k) != rec.get(k):
            bad.append(f"the signature's {k} is not the record's")
    return bad


def load(path: Path = RECORD) -> dict:
    return json.loads(Path(path).read_text())


DECISION = ("maintainer 2026-11-04 (review of a143b7ed), section 1, '正式数据源 GO': '批准 Toucan-1.5M SFT 与 "
            "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 作为正式 tool-calling 任务来源。按现有审计、license、"
            "revision、attribution、T1–T5 隔离与验证标准构建正式池。内容身份必须与已审查的 be4652d7… 一致，才能直接复用 "
            "Pilot 的导入记录。本消息构成上述两个数据源的正式准入批准；可以依照既有签名契约登记本次书面决定。来源准入不自动"
            "授权其他实验阶段。不得引入 Pivot Function-Calling、Nemotron-SFT-Agentic-v4 或其他尚未准入的来源。所有训练目标和"
            "在线 KD 必须由对应 Teacher 原生提供。来源记录中的其他模型输出只能作为任务材料或已登记的参考验证信息，不得进入 "
            "Student 训练目标。' -- the signature fields filled by the agent on that explicit written instruction")


def main(argv=None) -> int:
    from stages.teacher_modes import tool_generation_plan as TGP
    from stages.teacher_modes import tool_pool as TP
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--write", default=str(RECORD))
    ap.add_argument("--date", default="2026-11-04")
    ap.add_argument("--reviewed-copy", default=None, help="the reviewed pool directory the bytes are compared with")
    a = ap.parse_args(argv)
    man, _ = TP.load_pool(Path(a.pool))        #: every split verified against pool.json
    prov = {"builder": "scripts/stages/stage-2/teacher_modes/tool_pool.py",
            "builder_sha256": TGP.sha_file(REPO / "scripts/stages/stage-2/teacher_modes/tool_pool.py"),
            "converted_sources_sha256": {k: v.get("converted_sha256") for k, v in sorted(man["sources"].items())}}
    if a.reviewed_copy:
        rc = Path(a.reviewed_copy).resolve()
        other = json.loads((rc / "pool.json").read_text())
        prov["reviewed_copy"] = {
            "path": str(rc.relative_to(REPO)) if REPO in rc.resolve().parents else str(rc),
            "pool_json_sha256": TGP.sha_file(rc / "pool.json"),
            "split_files_identical": all(TGP.sha_file(rc / f["file"]) == f["sha256"] for f in man["files"].values()),
            "content_sha256_equal": other.get("content_sha256") == man["content_sha256"],
            "pool_json_keys_that_differ": sorted(k for k in set(other) | set(man) if other.get(k) != man.get(k))}
    rec = record(man, TGP.sha_file(Path(a.pool) / "pool.json"), decision=DECISION, date=a.date, provenance=prov)
    bad = problems(rec, pool_content_sha256=man["content_sha256"])
    if bad:
        raise SystemExit("the record refuses: " + "; ".join(bad))
    Path(a.write).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("pool_content_sha256", "pool_json_sha256", "sources")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
