#!/usr/bin/env python3
"""DMR1 variant 1: the alpha (importance-normalized) merge at SOB1's depth-only structure -- the instance.

    # $0, dev box: the base's per-tensor digests (what the build checks the unmerged tensors against), the spans
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/merge.py digests \\
        --checkpoint ~/aad-artifacts/stage1_single_operator_budget/depth_only/init/36888440ef83bfd6/init
    ... merge.py spans

What is this round's and what is reused (``configs/stages/stage-1/depth_merge/design.json :: merge, reuse``):

* **this round's** -- the spans (from SOB1's committed depth mapping, `spans`), the base (SOB1's depth init, checked
  through its per-tensor digests), the calibration call (`calibration`: the design's profile and positions under the
  packed search's own session), the per-pair persistence of the alignments (`pair_alignments`) and the build call;
* **RI-LM1's, by identity** -- the M-act method: `reasoning_inheritance.layer_merge`'s ``collect_statistics``,
  ``alignments``, ``statistics_tensors``, ``merged_tensors`` (method ``M-act``), ``build_candidate``,
  ``equivalence_checks`` and ``diagnostics``, on top of the core arithmetic (`aadistill.initialization.planning.
  layer_merge`). They are called with THIS design's spans, kept blocks and base digests; nothing of RI-LM1's
  statistics or merged tensors is an input (`statistics_reproduction` compares the pairs the two rounds share, as a
  diagnostic only).

The alpha of the brief is M-act's convex weight: per slot (FFN neuron, query head, KV group, block-level tensors)
``alpha_{j,i} = I_{j,i} / sum_k I_{j,k}`` over the span ``S_j``, uniform where every ``I`` of the slot is zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402

METHOD = "M-act"
DIGESTS = REPO / "configs/stages/stage-1/depth_merge/depth_init_tensor_digests.json"
STATS = "merge_statistics"


class MergeError(RuntimeError):
    pass


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def lmx():
    """RI-LM1's instance module (the method's code), imported where it is used."""
    from stages.reasoning_inheritance import layer_merge as LMX
    return LMX


# ---- the spans and the base --------------------------------------------------------------------------------------

def kept_blocks(d: dict | None = None) -> list[int]:
    """SOB1's committed depth selection, read from its init record (whose sha256 the design pins) and checked
    against the design's copy."""
    d = d or DC.design()
    b = d["baseline"]
    rec_p = REPO / b["init_record"]["path"]
    if sha256_file(rec_p) != b["init_record"]["sha256"]:
        raise MergeError(f"{rec_p} is not the init record the design pins")
    kept = [int(x) for x in read_json(rec_p)["depth"]["kept_layers"]]
    if kept != [int(x) for x in b["mapping"]["kept_blocks"]]:
        raise MergeError(f"the init record keeps {kept}, the design {b['mapping']['kept_blocks']}")
    return kept


def all_spans(d: dict | None = None) -> list[tuple[int, ...]]:
    from aadistill.initialization.planning.layer_merge import spans_from_kept_blocks
    d = d or DC.design()
    return spans_from_kept_blocks(int(d["baseline"]["mapping"]["n_root_blocks"]), kept_blocks(d))


def merge_spans(d: dict | None = None) -> list[tuple[int, ...]]:
    """The spans of more than one block (the ones a merge rewrites), checked against the design's table."""
    d = d or DC.design()
    spans = [s for s in all_spans(d) if len(s) > 1]
    want = [tuple(s) for s in d["merge"]["spans"]["merged"]]
    single = [s[0] for s in all_spans(d) if len(s) == 1]
    if spans != want or single != [int(x) for x in d["merge"]["spans"]["single"]]:
        raise MergeError(f"the mapping gives merged spans {spans} / single {single}, the design registers "
                         f"{want} / {d['merge']['spans']['single']}")
    return spans


def pairs(spans) -> list[tuple[int, int]]:
    """Every (kept block, absorbed block) of the merged spans: what the statistics and alignments are computed for."""
    return [(s[0], b) for s in spans for b in s[1:]]


def selection_view(d: dict | None = None) -> dict:
    """The shape RI-LM1's builders read (``kept_blocks``, ``source.seed``): the kept blocks and SOB1's selection seed
    (the one SOB1 materialized its arms with; a pure deletion draws nothing from it)."""
    sob_sel = read_json(REPO / "configs/stages/stage-1/single_operator_budget/selection.json")
    return {"kept_blocks": kept_blocks(d), "n_root_blocks": 36, "source": {"seed": int(sob_sel["source"]["seed"])}}


def digests_command(a) -> int:
    """$0 on the dev box: every tensor of the base's ``model.safetensors`` by name (`frozen_mapping.tensor_sha256`),
    from the dev box's copy, which must hash to the design's base."""
    d = DC.design()
    ckpt = Path(a.checkpoint).expanduser()
    want = d["baseline"]["init_record"]
    got = sha256_file(ckpt / "model.safetensors")
    if got != want["model_safetensors_sha256"]:
        raise MergeError(f"{ckpt} hashes to {got[:12]}, the base is {want['model_safetensors_sha256'][:12]}")
    if sha256_file(ckpt / "config.json") != want["config_json_sha256"]:
        raise MergeError(f"{ckpt}/config.json is not the base's")
    t0 = time.time()
    dig = lmx().tensor_digests(ckpt)
    doc = {"schema": "aadistill.stage1.depth_merge.base_tensor_digests/v1",
           "checkpoint": {"weights_sha256": got, "config_sha256": want["config_json_sha256"],
                          "artifact_digest": want["artifact_digest"]},
           "rule": "frozen_mapping.tensor_sha256 of every tensor of the base's model.safetensors (dtype, shape, raw "
                   "bytes); the build compares every tensor outside the merged blocks with these, bit for bit",
           "n_tensors": len(dig), "seconds": round(time.time() - t0, 1), "tensors": dig}
    DIGESTS.write_text(json.dumps(doc, indent=1) + "\n")
    print(DIGESTS.relative_to(REPO), len(dig))
    return 0


def base_digests(d: dict | None = None) -> dict:
    d = d or DC.design()
    doc = read_json(DIGESTS)
    if doc["checkpoint"]["weights_sha256"] != d["baseline"]["init_record"]["model_safetensors_sha256"]:
        raise MergeError(f"{DIGESTS.name} is not the base's digests")
    return doc["tensors"]


# ---- the calibration the statistics are read on ------------------------------------------------------------------

def calibration(s, model, device: str, d: dict | None = None):
    """The design's profile at its operator's positions under the session's packing (RI-LM1's call path, the FFN
    operator's statistics positions): (items, active positions, a factory of the packed batches)."""
    from aadistill.initialization.calibration.batching import resolve_pad_id
    from aadistill.initialization.calibration.items import prepare_calibration_items
    from aadistill.initialization.calibration.packing import packed_batches
    from aadistill.initialization.calibration.profiles import get_profile, profile_for
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.scoring.batches import active_positions
    st = (d or DC.design())["statistics"]
    impl = get_implementation(st["positions_of_operator"])
    prof = profile_for(impl, get_profile(st["profile_id"]))
    items = prepare_calibration_items(prof.resolve(str(REPO)), profile_id=prof.qualified_id)
    active = active_positions(items, s.config.position_policy)
    pad = resolve_pad_id(model)

    def batches():
        return packed_batches(items, 1, packing=s.execution.calibration_batch_packing, pad_id=pad, device=device,
                              forward_context=s.config.forward_context)
    return items, active, batches


# ---- statistics and alignments, each pair durable the moment it exists -------------------------------------------

def correlation_sha256(c) -> str:
    """The identity of one pair's Pearson correlation matrix: its float64 bytes."""
    import torch
    return hashlib.sha256(c.detach().to("cpu", torch.float64).contiguous().numpy().tobytes()).hexdigest()


def pair_alignments(st, n_q: int, n_kv: int, *, persisted: dict | None = None, on_pair=None) -> tuple[dict, dict]:
    """RI-LM1's `alignments`, one pair at a time: a pair persisted by an earlier attempt is reused only when its
    record's correlation sha256 is this statistics pass's (the permutations are then the same function of the same
    numbers); otherwise it is computed and handed to ``on_pair(key, record)`` at once. Returns (alignments,
    reuse report)."""
    import torch
    LMX = lmx()
    out, report = {}, {}
    for key, pm in st.pearson.items():
        csha = correlation_sha256(pm.correlation())
        tag = f"{key[0]}->{key[1]}"
        prev = (persisted or {}).get(tag)
        if prev is not None and prev.get("correlation_sha256") == csha:
            out[key] = {"ffn_perm": torch.as_tensor(prev["ffn_perm"], dtype=torch.long),
                        "kv_perm": torch.as_tensor(prev["kv_perm"], dtype=torch.long),
                        "q_perm_within": torch.as_tensor(prev["q_perm_within"], dtype=torch.long),
                        "summary": prev["summary"]}
            report[tag] = "reused (same correlation)"
            continue
        one = type(st)(positions=st.positions, sequences=st.sequences, abs_sum=st.abs_sum, self_heads=st.self_heads,
                       pearson={key: pm}, heads={key: st.heads[key]})
        t0 = time.time()
        al = LMX.alignments(one, n_q, n_kv)[key]
        out[key] = al
        report[tag] = "computed" if prev is None else "recomputed (another correlation)"
        if on_pair is not None:
            on_pair(tag, {"correlation_sha256": csha, "ffn_perm": al["ffn_perm"].tolist(),
                          "kv_perm": al["kv_perm"].tolist(), "q_perm_within": al["q_perm_within"].tolist(),
                          "summary": al["summary"], "seconds": round(time.time() - t0, 1)})
    return out, report


def statistics_reproduction(stats: dict, ri_lm1_stats: dict | None) -> dict:
    """Diagnostic only: for the pairs RI-LM1 and this round share, whether the permutations are the same and how
    close the per-block statistics are. Nothing here is an input to the merge."""
    if ri_lm1_stats is None:
        return {"compared": False, "why": "RI-LM1's statistics file is not here"}
    out = {"role": "diagnostic only: RI-LM1's statistics are never an input", "pairs": {}, "blocks": {}}
    for k in sorted(stats):
        if k in ri_lm1_stats and k.startswith("pair"):
            out["pairs"][k] = bool((stats[k] == ri_lm1_stats[k]).all())
        elif k in ri_lm1_stats and k.startswith("block"):
            a, b = stats[k].double(), ri_lm1_stats[k].double()
            out["blocks"][k] = float(((a - b).norm() / b.norm().clamp_min(1e-30)))
    return out


def method_identity() -> dict:
    """The code that computes the statistics and the candidate (RI-LM1's `code_identity` over its own lists, plus this
    module's calibration and per-pair persistence)."""
    LMX = lmx()
    return {"statistics": LMX.code_identity(*LMX.statistics_code(), calibration, pair_alignments, correlation_sha256),
            "candidate": LMX.code_identity(*LMX.candidate_code())}


# ---- CLI ($0) ------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("digests", help="the base's per-tensor digests from its dev-box copy ($0)")
    g.add_argument("--checkpoint", required=True)
    sub.add_parser("spans", help="the merged spans and the pairs, from SOB1's mapping, checked against the design")
    a = ap.parse_args(argv)
    if a.cmd == "digests":
        return digests_command(a)
    spans = merge_spans()
    print(json.dumps({"merged_spans": [list(s) for s in spans], "pairs": [list(p) for p in pairs(spans)],
                      "kept_blocks": kept_blocks()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
