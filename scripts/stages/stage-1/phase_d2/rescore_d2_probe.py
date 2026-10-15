#!/usr/bin/env python3
"""Rescore ONE scored D2 probe's own rollouts against the teacher. DIAGNOSTIC.

    /opt/train/bin/python scripts/stages/stage-1/phase_d2/rescore_d2_probe.py \
        --student <trained model dir> --package <evaluation package> \
        --generations <probe generations dir> --label <probe id> \
        --generation-fingerprint <admitted fingerprint> --out <json>

Per sample of the probe's own generations: student confidence, the student's
log-probability of its own tokens, teacher surprisal and the teacher Top-K +
tail KL at the student's prefixes (`shared.evaluation.rollout_rescoring`),
summarized by reasoning and answer span, with the high-confidence/high-KL
quadrant at the thresholds the reference calibration froze before any D2
candidate existed.

WHICH TRAJECTORY: the engine's exact token ids from the read-only sidecar when
its stratum header says `exact` and the row aligns; otherwise the record's text
retokenized under the generator's own rendering rule, labelled
`retokenized_text` per sample -- an APPROXIMATION, never presented as exact.

ISOLATION: this runs after the probe is scored, writes only `--out`, and the
driver treats any failure here as a diagnostic failure: no score, row or
generation of the probe is read back, changed or re-run because of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def strata(gen_dir: Path) -> list[Path]:
    """The per-stratum summaries the generator wrote (`<stem>.json`)."""
    return sorted(p for p in gen_dir.glob("*.json")
                  if not p.name.endswith((".generations.jsonl",
                                          ".token_ids.jsonl")))


def retokenizer(tok, summary: dict[str, Any], battery_file: Path):
    """The generator's rendering rule, for records with no exact engine ids.

    Mirrors `uncapped_eval.py`: the sample's turns without the assistant's, the
    run's system message injected only where the record says it was, the
    sample's tools parsed by the generator's own `tool_schemas`, the run's
    chat-template kwargs, `add_special_tokens=False`.
    """
    import torch

    from shared.evaluation.uncapped_eval import tool_schemas

    samples = {}
    for line in battery_file.read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            samples[str(s["id"])] = s
    kwargs = dict(summary.get("chat_template_kwargs") or {})
    system = summary.get("system_message")

    def retokenize(record: dict[str, Any]):
        s = samples[str(record["id"])]
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        if record.get("system_source") == "default" and system:
            turns = [{"role": "system", "content": system}] + turns
        prompt = tok.apply_chat_template(turns, tools=tool_schemas(s),
                                         tokenize=False,
                                         add_generation_prompt=True, **kwargs)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        c_ids = tok(record["raw"], add_special_tokens=False).input_ids
        return torch.tensor([p_ids]), torch.tensor([c_ids])

    return retokenize


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--package", required=True,
                    help="the probe's evaluation package (its tokenizer)")
    ap.add_argument("--generations", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--generation-fingerprint", required=True)
    ap.add_argument("--battery-root", default=None,
                    help="where the battery strata are on this host; "
                         "defaults to the role named by the contract")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter
    from shared.evaluation.rollout_rescoring import (
        rescore_unit, trajectory_sample,
    )
    from shared.evaluation.uncapped_eval_token_ids import load_sidecar
    from stages.d_series import objectives as O
    from stages.d_series import scoring_protocol as SP
    from stages.phase_d1 import d1_session as D1S
    from stages.phase_d2 import d2_screening as SC

    started = time.time()
    out: dict[str, Any] = {
        "schema": "aadistill.phase_d2.probe_rescoring/v1",
        "probe_id": args.label, "diagnostic_only": True,
        "generation_fingerprint": args.generation_fingerprint}
    diag = SC.diagnostics(REPO_ROOT)
    thresholds = diag["thresholds"]
    out["thresholds"] = thresholds
    out["thresholds_from"] = diag["calibration_record_sha256"]

    register_builtin_adapters()
    adapter = get_adapter("qwen3")
    teacher_id = D1S.root_teacher_identity(REPO_ROOT)
    repo_id, _, revision = teacher_id["root_teacher_id"].partition("@")
    teacher_path = snapshot_download(repo_id, revision=revision)
    out["teacher"] = D1S.verify_staged_teacher(teacher_path, REPO_ROOT)
    teacher = adapter.load(teacher_path, dtype="bfloat16", device=args.device)
    student = adapter.load(args.student, dtype="bfloat16", device=args.device)
    for model in (teacher, student):
        model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(args.package)
    tok_files = sorted(p for p in Path(args.package).glob("tokenizer*")
                       if p.is_file())
    tok_sha = hashlib.sha256(b"".join(p.read_bytes() for p in tok_files)
                             ).hexdigest()
    think_close = tok.convert_tokens_to_ids("</think>")
    battery_root = Path(args.battery_root) if args.battery_root else (
        REPO_ROOT / SC.battery(REPO_ROOT)["root"])

    per_stratum, rows, failures = [], [], []
    for summary_path in strata(Path(args.generations)):
        summary = json.loads(summary_path.read_text())
        gens = summary_path.with_suffix(".generations.jsonl")
        records = [json.loads(line) for line in gens.read_text().splitlines()
                   if line.strip()]
        sidecar = load_sidecar(summary_path)
        header = sidecar[0] if sidecar else None
        retok = retokenizer(tok, summary,
                            battery_root / f"{summary_path.stem}.jsonl")
        samples = []
        for record in records:
            try:
                samples.append(trajectory_sample(record, sidecar, retok))
            except Exception as exc:                          # noqa: BLE001
                failures.append({"id": record.get("id"),
                                 "error": f"{type(exc).__name__}: {exc}"})
        unit = rescore_unit(
            student, teacher, samples, tokenizer=tok,
            think_close_id=think_close if isinstance(think_close, int) else None,
            tokenizer_sha256=tok_sha,
            generation_fingerprint=args.generation_fingerprint,
            support_top_k=SP.D_SERIES_TOP_K, entropy_top_k=O.ENTROPY_TOP_K,
            thresholds=thresholds)
        for row in unit["rows"]:
            row["stratum"] = summary_path.stem
        rows.extend(unit["rows"])
        failures.extend(unit["failures"])
        per_stratum.append({
            "stratum": summary_path.stem, "records": len(records),
            "rescored": len(unit["rows"]), "n_exact": unit["n_exact"],
            "sidecar": None if header is None else {
                k: header.get(k) for k in ("available", "exact",
                                           "reason_unavailable",
                                           "verification", "hook_errors")}})
        torch.cuda.empty_cache()
    out.update({
        "strata": per_stratum, "rows": rows, "failures": failures,
        "n_samples": sum(s["records"] for s in per_stratum),
        "n_rescored": len(rows),
        "n_exact_engine_ids": sum(s["n_exact"] for s in per_stratum),
        "n_retokenized_approximation": sum(
            1 for r in rows if not r["alignment"]["exact"]),
        "complete": not failures,
        "seconds": round(time.time() - started, 1)})
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out) + "\n")
    print(json.dumps({k: out[k] for k in ("probe_id", "n_samples",
                                          "n_rescored", "n_exact_engine_ids",
                                          "n_retokenized_approximation",
                                          "complete", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
