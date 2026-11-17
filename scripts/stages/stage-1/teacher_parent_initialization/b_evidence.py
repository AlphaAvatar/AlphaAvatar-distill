#!/usr/bin/env python3
"""B_packed_top1's sealed evidence, audited before it is paired with anything ($0, CPU, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/teacher_parent_initialization/b_evidence.py --write   # or --check

B is the packed Top-1 as A-MIX-6cat-sr trained it (`design.json :: arms.B_packed_top1`): two seeds, each evaluated in
both modes on eval-set-1. It is never retrained; its per-item generations are what A's are paired with. This reads,
for each seed and mode, ``<extracted>/A-MIX-6cat-sr-s<seed>/evaluation_<mode>/generations/`` and records:

* completeness: every registered unit of eval-set-1 (the manifest's units, the withheld TriviaQA copy included) has its
  generations file, and its records hold every registered item id exactly once -- the expectations are the set's own
  item files (`shared.eval_sets.items`), never typed counts;
* the generation protocol each unit's summary records (`protocol_fields`): decoding protocol and its sampling and seed
  rule, the resolved context and its rule, the system message, the chat template and its kwargs, the tokenizer, the
  degeneration check interval, the fluid unit's action protocol, seed namespace and task file; compared with the
  design. A field a summary does not carry is ``"not recorded"``, with what it can be derived from -- never a pass;
* the evaluation package's tokenizer files against the teacher identity's sha256 (the package's weights were not
  archived), B's final weights sha256 per seed as the units record states it, the training plan's and the evidence
  archive's sha256.

``protocol_to_match`` is the per-mode protocol A's summaries must carry (analysis.py compares A against it, field by
field); ``implementation_to_compare`` (runner and generation-source digests, engine and library versions) is reported
beside it but does not decide comparability: the runners gained a streaming option after A-MIX, so their source
digests differ by construction while the protocol does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json"
OUT = REPO / "logs/stages/stage-1/teacher_parent_initialization/audit/b_evidence.json"
SCHEMA = "aadistill.stage1.teacher_parent_initialization.b_evidence/v1"
SET_ID = "eval-set-1"
NOT_RECORDED = "not recorded"
FLUID = "fluid_reasoning_720__dev_v1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def design() -> dict:
    return json.loads(DESIGN.read_text())


def seeds(d: dict) -> list[int]:
    return [int(s) for s in d["arms"]["B_packed_top1"]["trained"]["final_weights_sha256"]]


def b_root(d: dict) -> Path:
    return Path(d["arms"]["B_packed_top1"]["evidence"]["dev_box_extracted"]).expanduser()


def b_gens(d: dict, seed: int, mode: str) -> Path:
    return b_root(d) / f"A-MIX-6cat-sr-s{seed}" / d["evaluation"]["modes"][mode]["dir"] / "generations"


def registered_units() -> dict[str, tuple[str, list[str]]]:
    """``<component>__<unit>`` -> (the records' id field, the registered item ids in file order), for every unit of
    the set, from the set's own files."""
    from shared import eval_sets as ES
    m = ES.manifest(SET_ID)
    out = {}
    for comp, c in m["components"].items():
        for unit, u in c["units"].items():
            field = u["id_field"]
            out[f"{comp}__{unit}"] = (field, [str(it[field]) for it in ES.items(SET_ID, comp, unit)])
    return out


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()]


def completeness(gens: Path, units: dict) -> dict:
    """Per registered unit: present, records, unique ids, missing / extra / duplicated ids (first few)."""
    out = {}
    for key, (field, ids) in units.items():
        f = gens / f"{key}.jsonl"
        if not f.is_file():
            out[key] = {"present": False, "complete": False, "registered": len(ids)}
            continue
        got = [str(r.get(field) if field in r else r.get("id")) for r in jl(f)]
        seen, dup = set(), []
        for i in got:
            if i in seen:
                dup.append(i)
            seen.add(i)
        missing, extra = sorted(set(ids) - seen), sorted(seen - set(ids))
        out[key] = {"present": True, "registered": len(ids), "records": len(got), "unique": len(seen),
                    "missing": missing[:5], "n_missing": len(missing), "extra": extra[:5], "n_extra": len(extra),
                    "duplicated": dup[:5], "n_duplicated": len(dup),
                    "complete": not missing and not extra and not dup,
                    "generations_sha256": sha256_file(f),
                    "summary_present": (gens / f"{key}.summary.json").is_file()}
    return out


def _get(d: dict, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return NOT_RECORDED
        d = d[k]
    return d


def protocol_fields(summary: dict, key: str) -> dict:
    """The generation protocol a unit's summary records, in one shape for both runners (run_battery's summary nests
    it under ``identity``; uncapped_eval's carries it at the top). Absent fields are ``NOT_RECORDED``."""
    fluid = key == FLUID
    s = summary.get("identity", {}) if fluid else summary
    dp = s.get("decoding_protocol") if isinstance(s.get("decoding_protocol"), dict) else {}
    out = {
        "decoding_protocol_id": _get(dp, "protocol_id") if dp else NOT_RECORDED,
        "decoding_sha256": _get(dp, "sha256") if dp else NOT_RECORDED,
        "sampling": _get(dp, "sampling") if dp else NOT_RECORDED,
        "samples_per_item": _get(dp, "samples_per_item") if dp else NOT_RECORDED,
        "seed_rule_base": _get(dp, "seed_rule", "base") if dp else NOT_RECORDED,
        "seed_rule_key": _get(dp, "seed_rule", "key") if dp else NOT_RECORDED,
        "seed_rule_derivation": _get(dp, "seed_rule", "derivation") if dp else NOT_RECORDED,
        "resolved_context": _get(s, "context_resolution", "resolved_context"),
        "context_source": _get(s, "context_resolution", "context_source"),
        "trained_context": _get(s, "context_resolution", "trained_context"),
        "architectural_context": _get(s, "context_resolution", "architectural_context"),
        "system_message": _get(s, "system_message"),
        "chat_template_sha256": _get(s, "chat_template_sha256"),
        "chat_template_kwargs": _get(s, "chat_template_kwargs"),
        "tokenizer_sha256": _get(s, "tokenizer_sha256"),
        "stop_ids": _get(s, "stop_ids"),
        "degeneration_check_every": (_get(s, "degeneration_check_every") if fluid
                                     else _get(s, "sampling", "degeneration_check_every")),
    }
    if fluid:
        out.update(action_protocol=_get(s, "action_protocol"), seed_namespace=_get(s, "seed_namespace"),
                   tasks_sha256=_get(s, "tasks_sha256"))
    else:
        out.update(protocol=_get(s, "protocol"), degeneration_stop=_get(s, "degeneration_stop"),
                   max_tokens_rule=_get(s, "sampling", "max_tokens_rule"),
                   #: uncapped_eval derives the namespace from the prompt file's stem, which the evaluation stages
                   #: as ``<unit key>.jsonl``; the summary does not repeat it
                   seed_namespace={"recorded": NOT_RECORDED, "derived_from": "the prompt file stem (uncapped_eval), "
                                   "which the evaluation stages as the unit key", "derived": key})
    return out


def implementation_fields(summary: dict, key: str) -> dict:
    """Code and runtime identity: compared and reported, never a comparability condition."""
    fluid = key == FLUID
    s = summary.get("identity", {}) if fluid else summary
    out = {"engine_max_model_len": _get(s, "engine", "max_model_len"), "vllm_version": _get(s, "engine", "vllm_version"),
           "libraries": _get(s, "libraries")}
    if fluid:
        out["runner_sha256"] = _get(s, "runner_sha256")
    else:
        out.update(generation_source_digest=_get(s, "identity", "generation_source_digest"),
                   degeneration_source_digest=_get(s, "identity", "degeneration_source_digest"),
                   runtime_digest=_get(s, "identity", "runtime_digest"))
    return out


def expected_protocol(d: dict, mode: str) -> dict:
    """What the design fixes, in `protocol_fields`' names (the rest is taken from B's own records)."""
    ev = d["evaluation"]
    ident = json.loads((REPO / d["teacher"]["identity"]["path"]).read_text())
    return {"decoding_sha256": ev["decoding"]["sha256"], "resolved_context": int(ev["context"]["tokens"]),
            "context_source": "trained_block_len", "trained_context": int(ev["context"]["tokens"]),
            "system_message": ev["system_message"], "chat_template_sha256": ident["chat_template_sha256"],
            "chat_template_kwargs": ev["modes"][mode]["template_kwargs"],
            "degeneration_check_every": int(ev["degeneration_check_every"])}


def compare(fields: dict, want: dict) -> list[str]:
    """The fields of ``want`` that ``fields`` does not equal (a NOT_RECORDED field is a difference)."""
    return [f"{k}: {fields.get(k, NOT_RECORDED)!r} != {v!r}" for k, v in want.items() if fields.get(k, NOT_RECORDED) != v]


def mode_protocol(per_unit: dict) -> dict:
    """One mode's protocol to match: per unit kind (fluid / capability), the fields every unit of that kind records
    identically; a field that differs between units of a kind is kept per unit."""
    out = {"fluid": dict(per_unit[FLUID]) if FLUID in per_unit else {}, "capability": {}, "capability_per_unit": {}}
    caps = {k: v for k, v in per_unit.items() if k != FLUID}
    if caps:
        names = sorted(set().union(*(v.keys() for v in caps.values())) - {"seed_namespace"})
        for n in names:
            vals = {json.dumps(v.get(n, NOT_RECORDED), sort_keys=True) for v in caps.values()}
            if len(vals) == 1:
                out["capability"][n] = next(iter(caps.values())).get(n, NOT_RECORDED)
            else:
                for k, v in caps.items():
                    out["capability_per_unit"].setdefault(k, {})[n] = v.get(n, NOT_RECORDED)
    return out


def runner_summaries(summary: dict) -> list[dict]:
    """The runner summaries a sealed unit summary stands on. B's are the runners' own files. A's are written by the
    streaming evaluation (`reasoning_inheritance.driver.fluid` / `capability`): the fluid unit's is a composition whose
    ``runs[*].summary`` are the runner's, one per attempt that generated; a capability unit's is the runner's own when
    generated in one run, else a composition with that attempt's ``run_summary``."""
    if "runs" in summary and "composition" in summary:
        return [r["summary"] for r in summary["runs"] if isinstance(r.get("summary"), dict)]
    if "composition" in summary:
        return [summary["run_summary"]] if isinstance(summary.get("run_summary"), dict) else []
    return [summary]


def protocol_problems(a_summaries: dict[str, dict], want: dict) -> list[str]:
    """A's sealed unit summaries (key -> summary) against one mode's ``protocol_to_match``: every runner summary each
    stands on must carry B's protocol. The fluid unit's task file is compared on the sealed summary (a resumed run
    submits only the missing tasks, so its own file is a subset); a capability unit's seed namespace is derived from
    its prompt file's stem, as B's. A unit with no runner summary is a problem: an unrecorded protocol is not B's."""
    bad = []
    for key, summary in sorted(a_summaries.items()):
        if key == FLUID:
            ref = dict(want.get("fluid") or {})
            tasks = ref.pop("tasks_sha256", None)
            got_tasks = summary.get("tasks_sha256") if "composition" in summary else \
                protocol_fields(summary, key).get("tasks_sha256")
            if tasks is not None and got_tasks != tasks:
                bad.append(f"{key}: tasks_sha256 {got_tasks!r} != {tasks!r}")
        else:
            ref = {**(want.get("capability") or {}), **(want.get("capability_per_unit", {}).get(key) or {})}
        runs = runner_summaries(summary)
        if not runs:
            bad.append(f"{key}: no runner summary recorded")
        for i, run in enumerate(runs):
            got = protocol_fields(run, key)
            bad += [f"{key}[run {i}]: {p}" for p in compare(got, ref)]
    return bad


def audit(d: dict | None = None) -> dict:
    d = d or design()
    b = d["arms"]["B_packed_top1"]
    ident = json.loads((REPO / d["teacher"]["identity"]["path"]).read_text())
    units = registered_units()
    plan_path = REPO / b["trained"]["plan"]["path"]
    urec_path = REPO / b["trained"]["units_record"]["path"]
    urec = json.loads(urec_path.read_text())
    archive = Path(d["arms"]["B_packed_top1"]["evidence"]["dev_box_extracted"]).expanduser().parent \
        / Path(b["evidence"]["archive"]["relay_path"]).name
    out: dict = {
        "schema": SCHEMA, "design_sha256": sha256_file(DESIGN), "eval_set": SET_ID,
        "eval_set_manifest_sha256": sha256_file(REPO / "data/eval-set" / SET_ID / "manifest.json"),
        "eval_set_manifest_sha256_matches_design": sha256_file(REPO / "data/eval-set" / SET_ID / "manifest.json")
        == d["evaluation"]["manifest_sha256"],
        "registered_units": {k: len(v[1]) for k, v in units.items()},
        "training_plan": {"path": b["trained"]["plan"]["path"], "sha256": sha256_file(plan_path),
                          "matches_design": sha256_file(plan_path) == b["trained"]["plan"]["sha256"]},
        "units_record": {"path": b["trained"]["units_record"]["path"], "sha256": sha256_file(urec_path),
                         "matches_design": sha256_file(urec_path) == b["trained"]["units_record"]["sha256"],
                         "terminal": urec.get("terminal")},
        "evidence_archive": ({"dev_box_copy": str(archive).replace(str(Path.home()), "~"),
                              "sha256": sha256_file(archive),
                              "matches_design": sha256_file(archive) == b["evidence"]["archive"]["sha256"]}
                             if archive.is_file() else {"dev_box_copy": str(archive), "present": False}),
        "seeds": {}, "protocol_to_match": {}, "implementation_to_compare": {}, "problems": []}
    for seed in seeds(d):
        unit = f"A-MIX-6cat-sr-s{seed}"
        urow = urec["units"].get(unit, {})
        final = urow.get("final_model_safetensors_sha256", NOT_RECORDED)
        srec = {"unit": unit, "final_weights_sha256": final,
                "final_weights_match_design": final == b["trained"]["final_weights_sha256"][str(seed)],
                "modes": {}}
        for mode, m in d["evaluation"]["modes"].items():
            gens = b_gens(d, seed, mode)
            ev_dir = gens.parent
            comp = completeness(gens, units)
            pkg = ev_dir / "package"
            tok = {n: (sha256_file(pkg / n) if (pkg / n).is_file() else NOT_RECORDED)
                   for n in ("tokenizer.json", "tokenizer_config.json")}
            per_unit, impl = {}, {}
            for key in units:
                sp = gens / f"{key}.summary.json"
                if sp.is_file():
                    s = json.loads(sp.read_text())
                    per_unit[key], impl[key] = protocol_fields(s, key), implementation_fields(s, key)
            want = expected_protocol(d, mode)
            design_problems = {k: compare(v, want) for k, v in per_unit.items()}
            mode_name = {True: "thinking", False: "non_thinking"}[bool(m["template_kwargs"]["enable_thinking"])]
            evr = (urow.get("evaluations") or {}).get(mode_name, {})
            mrec = {"generations_dir": str(gens).replace(str(Path.home()), "~"),
                    "complete": all(v["complete"] for v in comp.values()),
                    "units": comp,
                    "package_tokenizer_sha256": tok,
                    "package_tokenizer_matches_teacher": all(tok[n] == ident["files_sha256"][n] for n in tok),
                    "formal_report_sha256": (sha256_file(ev_dir / "formal_report.json")
                                             if (ev_dir / "formal_report.json").is_file() else NOT_RECORDED),
                    "units_record_template": evr.get("template", NOT_RECORDED),
                    "protocol_against_design": {k: v for k, v in design_problems.items() if v},
                    "not_recorded_fields": sorted({f"{k}.{n}" for k, v in per_unit.items() for n, x in v.items()
                                                   if x == NOT_RECORDED})}
            srec["modes"][mode] = mrec
            mp = mode_protocol(per_unit)
            prev = out["protocol_to_match"].get(mode)
            if prev is not None and prev != mp:
                out["problems"].append(f"{mode}: B's two seeds record different protocols")
            out["protocol_to_match"].setdefault(mode, mp)
            out["implementation_to_compare"].setdefault(mode, {})[str(seed)] = impl
            if not mrec["complete"]:
                out["problems"].append(f"s{seed} {mode}: incomplete units "
                                       f"{sorted(k for k, v in comp.items() if not v['complete'])}")
            if mrec["protocol_against_design"]:
                out["problems"].append(f"s{seed} {mode}: protocol differs from the design: "
                                       f"{mrec['protocol_against_design']}")
            if not mrec["package_tokenizer_matches_teacher"]:
                out["problems"].append(f"s{seed} {mode}: the package's tokenizer files are not the teacher's")
        if not srec["final_weights_match_design"]:
            out["problems"].append(f"s{seed}: the units record's final weights are not the design's")
        out["seeds"][str(seed)] = srec
    for k in ("training_plan", "units_record", "evidence_archive"):
        if out[k].get("matches_design") is not True:
            out["problems"].append(f"{k}: not the design's sha256 (or absent)")
    if not out["eval_set_manifest_sha256_matches_design"]:
        out["problems"].append("eval-set-1's manifest is not the design's")
    out["verdict"] = ("B_EVIDENCE_COMPLETE_AND_BOUND" if not out["problems"] else "B_EVIDENCE_PROBLEMS")
    out["reading"] = ("B's per-item records are complete for both seeds and modes and carry the design's generation "
                      "protocol; fields marked 'not recorded' are derived as stated, never counted as checks"
                      if not out["problems"] else "see problems")
    return out


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(audit())
    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO)}: {json.loads(text)['verdict']}")
        return 0
    same = OUT.is_file() and OUT.read_text() == text
    print("b_evidence.json reproduces" if same else "b_evidence.json does NOT reproduce")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
