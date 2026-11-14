#!/usr/bin/env python3
"""Layer merging at D's structure: the experiment's instance of `aadistill.initialization.planning.layer_merge`.

    # $0, dev box: D's per-tensor digests (what the pod checks the 22 unchanged blocks against), the spans
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge.py digests \\
        --checkpoint ~/aad-artifacts/stage1_reasoning_inheritance/913433664fce0fed/groups/D
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge.py spans

Everything instance-specific lives here and in ``configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json``:
which blocks form the spans (the frozen selection's kept blocks), the calibration profile, positions and packing the
statistics are read at (the anchor search's own session, `driver.session`), the two candidates, the tolerances, and
D's identity. The arithmetic -- the exact assignment, the streaming moments, folding, permuting, merging, the update
moments -- is the core module's and names nothing of this.

The pod driver (`layer_merge_driver.py`) calls, in order: `collect_statistics` (one teacher forward over the
calibration, at the statistics positions), `alignments` (the permutations), `equivalence_checks` (folding alone and
each permutation alone reproduce the block's forward in float32), `build_candidate` (D's 28 blocks with the six span
blocks rewritten; every other tensor checked bit for bit against D's digests) and `diagnostics` (each candidate block's
residual update against the teacher span's, on the same positions).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import torch  # noqa: E402

from aadistill.initialization.planning import layer_merge as LM  # noqa: E402

DESIGN = REPO / "configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json"
DIGESTS = REPO / "configs/stages/stage-1/reasoning_inheritance/layer_merge/d_tensor_digests.json"
SELECTION = REPO / "configs/stages/stage-1/reasoning_inheritance/frozen_selection.json"
CANDIDATES = ("M-avg", "M-act")


class MergeError(RuntimeError):
    pass


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def design() -> dict:
    return read_json(DESIGN)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- spans ---------------------------------------------------------------------------------------------------------

def merge_spans(d: dict | None = None, sel: dict | None = None) -> list[tuple[int, ...]]:
    """The spans of more than one block, from the frozen selection's kept blocks (each removed block joins the nearest
    kept block before it), checked against the design's table."""
    d = d or design()
    sel = sel or read_json(SELECTION)
    spans = [s for s in LM.spans_from_kept_blocks(int(sel["n_root_blocks"]), sel["kept_blocks"]) if len(s) > 1]
    want = [tuple(s) for s in d["spans"]["table"]]
    if spans != want:
        raise MergeError(f"the frozen selection gives spans {spans}, the design registers {want}")
    return spans


def child_index(sel: dict, root_block: int) -> int:
    """Where a kept root block sits in D (D's blocks are the kept blocks in order)."""
    return [int(b) for b in sel["kept_blocks"]].index(int(root_block))


# ---- D's tensor digests ($0, dev box) ------------------------------------------------------------------------------

def tensor_digests(checkpoint: Path) -> dict:
    """Every tensor of a safetensors checkpoint, by name: `frozen_mapping.tensor_sha256` (dtype, shape, raw bytes)."""
    from safetensors import safe_open
    from aadistill.initialization.planning.frozen_mapping import tensor_sha256
    files = sorted(Path(checkpoint).glob("*.safetensors"))
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for k in sorted(h.keys()):
                out[k] = tensor_sha256(h.get_tensor(k))
    return out


def digests_command(a) -> int:
    ckpt = Path(a.checkpoint).expanduser()
    d = design()
    want = d["base"]["d_checkpoint"]
    got_sha = sha256_file(ckpt / "model.safetensors")
    if got_sha != want["weights_sha256"]:
        raise MergeError(f"{ckpt} hashes to {got_sha[:12]}, D is {want['weights_sha256'][:12]}")
    t0 = time.time()
    dig = tensor_digests(ckpt)
    doc = {"schema": "aadistill.stage1.reasoning_inheritance.layer_merge.d_tensor_digests/v1",
           "checkpoint": {"weights_sha256": got_sha, "config_sha256": sha256_file(ckpt / "config.json"),
                          "artifact_digest": want["artifact_digest"]},
           "rule": "frozen_mapping.tensor_sha256 of every tensor of D's model.safetensors (dtype, shape, raw bytes); "
                   "the pod compares each candidate's tensors outside the six span blocks with these, bit for bit",
           "n_tensors": len(dig), "seconds": round(time.time() - t0, 1), "tensors": dig}
    DIGESTS.parent.mkdir(parents=True, exist_ok=True)
    DIGESTS.write_text(json.dumps(doc, indent=1) + "\n")
    print(DIGESTS, len(dig))
    return 0


# ---- the calibration the statistics are read on ---------------------------------------------------------------------

def calibration(s, model, device: str) -> tuple[list, object, callable]:
    """The design's calibration profile under the anchor session's packing and position policy: (items, the active
    positions, a factory of the packed batches). The same call path as the FFN operator's statistics
    (`operators._common.collect_activation_stats`), so the positions are that operator's."""
    from aadistill.initialization.calibration.batching import resolve_pad_id
    from aadistill.initialization.calibration.items import prepare_calibration_items
    from aadistill.initialization.calibration.packing import packed_batches
    from aadistill.initialization.calibration.profiles import get_profile, profile_for
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.scoring.batches import active_positions
    st = design()["statistics"]
    impl = get_implementation(st["positions_of_operator"])
    prof = profile_for(impl, get_profile(st["profile_id"]))
    items = prepare_calibration_items(prof.resolve(str(REPO)), profile_id=prof.qualified_id)
    active = active_positions(items, s.config.position_policy)
    pad = resolve_pad_id(model)

    def batches():
        return packed_batches(items, 1, packing=s.execution.calibration_batch_packing, pad_id=pad, device=device,
                              forward_context=s.config.forward_context)
    return items, active, batches


def valid_rows(packed, active) -> torch.Tensor:
    """Flat bool over the batch's positions: real (not padding) and admitted by the position policy."""
    b = packed.batch
    valid = b.attention_mask.reshape(-1).bool()
    m = active.token_mask_for(b, packed.original_indices) if active is not None else None
    if m is not None:
        valid = valid & m.reshape(-1).to(valid.device)
    return valid


def forward(model, packed) -> None:
    b = packed.batch
    padded = bool((b.attention_mask == 0).any())
    model(b.input_ids, use_cache=False, **({"attention_mask": b.attention_mask} if padded else {}))


# ---- statistics ----------------------------------------------------------------------------------------------------

@dataclass
class Statistics:
    """Per involved root block: ``abs_sum`` (sum |a| of each FFN neuron) and ``head_cross`` (its heads' write Gram,
    for group energies); per (anchor, absorbed) pair: ``pearson`` (FFN features) and ``heads`` (head writes)."""
    positions: int = 0
    sequences: int = 0
    abs_sum: dict = field(default_factory=dict)
    self_heads: dict = field(default_factory=dict)
    pearson: dict = field(default_factory=dict)
    heads: dict = field(default_factory=dict)


def head_writes(a: torch.Tensor, o_weight: torch.Tensor, n_q: int, head_dim: int) -> torch.Tensor:
    """``[T, n_q, hidden]``: each head's write into the residual stream, ``W_o,h a_h(t)`` (float32)."""
    t = a.shape[0]
    w = o_weight.to(torch.float32).reshape(o_weight.shape[0], n_q, head_dim)
    return torch.einsum("thd,ehd->the", a.to(torch.float32).reshape(t, n_q, head_dim), w)


@torch.no_grad()
def collect_statistics(adapter, teacher, spans, batches, active, *, chunk: int = 1024) -> Statistics:
    """One teacher forward over the calibration (module docstring): FFN features (the down projection's input) and
    head writes (the output projection's input, per head, through its column block), at the statistics positions, for
    every block of every span. Per packed sequence the cross products are float32 of bf16 activations; every sum
    across sequences is float64 (`PearsonMoments`, `GramMoments`)."""
    n_q, n_kv, hd = adapter.head_groups(adapter.spec_of(teacher))
    blocks = adapter.blocks(teacher)
    dev = next(teacher.parameters()).device
    involved = sorted({b for s in spans for b in s})
    pairs = [(s[0], b) for s in spans for b in s[1:]]
    width_ffn = adapter.stream_out_projections(blocks[involved[0]])["ffn_out"].weight.shape[1]
    st = Statistics()
    for b in involved:
        st.abs_sum[b] = torch.zeros(width_ffn, dtype=torch.float64, device=dev)
        st.self_heads[b] = LM.GramMoments(n_q, n_q, device=dev)
    for p in pairs:
        st.pearson[p] = LM.PearsonMoments(width_ffn, width_ffn, device=dev)
        st.heads[p] = LM.GramMoments(n_q, n_q, device=dev)
    captured: dict = {}
    handles = []
    for b in involved:
        outs = adapter.stream_out_projections(blocks[b])
        for role in ("ffn_out", "attn_out"):
            def hook(_m, args, _b=b, _r=role):
                captured[(_b, _r)] = args[0].reshape(-1, args[0].shape[-1]).detach()
            handles.append(outs[role].register_forward_pre_hook(hook))
    try:
        for packed in batches():
            captured.clear()
            forward(teacher, packed)
            valid = valid_rows(packed, active).to(dev)
            n = int(valid.sum())
            if n == 0:
                continue
            feats = {b: captured[(b, "ffn_out")][valid] for b in involved}
            attn = {b: captured[(b, "attn_out")][valid] for b in involved}
            for b in involved:
                st.abs_sum[b] += feats[b].to(torch.float64).abs().sum(0)
            for p in pairs:
                st.pearson[p].add(feats[p[0]], feats[p[1]])
            for lo in range(0, n, chunk):
                w = {b: head_writes(attn[b][lo:lo + chunk], adapter.stream_out_projections(blocks[b])["attn_out"].weight,
                                    n_q, hd) for b in involved}
                for b in involved:
                    st.self_heads[b].add(w[b], w[b])
                for p in pairs:
                    st.heads[p].add(w[p[0]], w[p[1]])
                del w
            st.positions += n
            st.sequences += 1
    finally:
        for h in handles:
            h.remove()
    return st


def alignments(st: Statistics, n_q: int, n_kv: int) -> dict:
    """Per (anchor, absorbed) pair: the FFN permutation (linear assignment on the Pearson correlations), the KV-group
    permutation (on the cosine of the groups' summed writes) and, within each matched group, the query-head
    permutation (on the heads' write cosine); with what each matching achieved."""
    g = n_q // n_kv
    out = {}
    for (a, b), pm in st.pearson.items():
        c = pm.correlation().cpu()
        zx, zy = pm.zero_variance()
        perm = LM.linear_assignment(c)
        hm = st.heads[(a, b)]
        head_cos = hm.cosine().cpu()
        ea_g = torch.diagonal(LM.group_sum(st.self_heads[a].cross, g, (0, 1))).cpu()
        eb_g = torch.diagonal(LM.group_sum(st.self_heads[b].cross, g, (0, 1))).cpu()
        group_cos = hm.cosine(LM.group_sum(hm.cross, g, (0, 1)).cpu(), ea_g, eb_g)
        kv = LM.linear_assignment(group_cos)
        within = [LM.linear_assignment(head_cos[j * g:(j + 1) * g, int(kv[j]) * g:(int(kv[j]) + 1) * g])
                  for j in range(n_kv)]
        idx = torch.arange(c.shape[0])
        out[(a, b)] = {
            "ffn_perm": perm, "kv_perm": kv, "q_perm_within": torch.stack(within),
            "summary": {"ffn_matched_correlation_mean": float(c[idx, perm].mean()),
                        "ffn_identity_correlation_mean": float(c[idx, idx].mean()),
                        "ffn_zero_variance": [int(zx.sum()), int(zy.sum())],
                        "ffn_moved": int((perm != idx).sum()),
                        "kv_matched_cosine_mean": float(group_cos[torch.arange(n_kv), kv].mean()),
                        "kv_identity_cosine_mean": float(torch.diagonal(group_cos).mean()),
                        "kv_moved": int((kv != torch.arange(n_kv)).sum()),
                        "head_matched_cosine_mean": float(torch.stack([
                            head_cos[j * g + r, int(kv[j]) * g + int(within[j][r])] for j in range(n_kv)
                            for r in range(g)]).mean()),
                        "head_identity_cosine_mean": float(torch.diagonal(head_cos).mean())}}
    return out


def statistics_tensors(st: Statistics, al: dict) -> dict[str, torch.Tensor]:
    """What the merge consumes, as one flat tensor dict (saved with the run's records; a later attempt reuses it)."""
    out = {}
    for b in st.abs_sum:
        out[f"block{b}.ffn_mean_abs"] = (st.abs_sum[b] / st.positions).cpu()
        out[f"block{b}.head_energy"] = (torch.diagonal(st.self_heads[b].cross) / st.positions).cpu()
    for (a, b), x in al.items():
        out[f"pair{a}_{b}.ffn_perm"] = x["ffn_perm"].to(torch.int64)
        out[f"pair{a}_{b}.kv_perm"] = x["kv_perm"].to(torch.int64)
        out[f"pair{a}_{b}.q_perm_within"] = x["q_perm_within"].to(torch.int64)
    return out


# ---- the candidates ------------------------------------------------------------------------------------------------

def merged_tensors(adapter, teacher, span, method: str, stats: dict[str, torch.Tensor] | None) -> tuple[dict, dict]:
    """The span's one block (role -> float64 tensor, on the host) and a record of how it was weighted."""
    n_q, n_kv, hd = adapter.head_groups(adapter.spec_of(teacher))
    blocks = adapter.blocks(teacher)
    t = [{k: v.cpu() for k, v in LM.block_tensors(adapter, blocks[b]).items()} for b in span]
    n_ffn = t[0]["gate"].shape[0]
    if method == "M-avg":
        return LM.merge_blocks(t, LM.MergeWeights.uniform(len(span), n_ffn, n_q, n_kv), head_dim=hd), {
            "weights": "uniform 1/|span| on every tensor, the norm gains included; no folding, no alignment"}
    if method != "M-act":
        raise MergeError(f"unknown candidate {method!r}")
    g = n_q // n_kv
    folded = [LM.fold_norm_gains(adapter, x) for x in t]
    aligned, ffn_s, head_s = [folded[0]], [], []
    for k, b in enumerate(span):
        mean_abs = stats[f"block{b}.ffn_mean_abs"].to(torch.float64)
        energy = stats[f"block{b}.head_energy"].to(torch.float64)
        s = mean_abs * t[k]["ffn_out"].norm(dim=0)                       #: E|a| * ||down column||_2
        if k == 0:
            ffn_s.append(s)
            head_s.append(energy)
            continue
        key = f"pair{span[0]}_{b}"
        perm, kv, within = (stats[f"{key}.ffn_perm"], stats[f"{key}.kv_perm"], stats[f"{key}.q_perm_within"])
        aligned.append(LM.permute_block(folded[k], ffn_perm=perm.tolist(), kv_perm=kv.tolist(),
                                        q_perm_within=within.tolist(), n_q=n_q, n_kv=n_kv, head_dim=hd))
        ffn_s.append(s[perm])
        head_s.append(energy[torch.as_tensor(LM.query_head_order(kv.tolist(), within.tolist(), g))])
    ffn_w, ffn_zero = LM.convex_weights(torch.stack(ffn_s))
    head_w, head_zero = LM.convex_weights(torch.stack(head_s))
    kv_w, kv_zero = LM.convex_weights(LM.group_sum(torch.stack(head_s), g, (1,)))
    block_w, _ = LM.convex_weights(torch.stack(head_s).sum(1))
    w = LM.MergeWeights(ffn=ffn_w, query_head=head_w, kv_group=kv_w, block=block_w)
    return LM.merge_blocks(aligned, w, head_dim=hd), {
        "weights": "per FFN neuron E|a|*||down col||, per query head and KV group the write energy, per block its "
                   "share of head energy; uniform where every score of a slot is zero",
        "uniform_fallback_slots": {"ffn": ffn_zero, "query_head": head_zero, "kv_group": kv_zero},
        "ffn_weight_of_anchor_mean": float(ffn_w[0].mean()), "head_weight_of_anchor_mean": float(head_w[0].mean()),
        "block_weights": [round(float(x), 6) for x in block_w]}


@torch.no_grad()
def build_candidate(adapter, teacher, sel: dict, spans, method: str, stats, digests: dict, *, seed: int):
    """D's 28 blocks (`frozen_mapping.keep_blocks`) with each span's block rewritten; every tensor outside the span
    blocks compared bit for bit with D's digests, every span block required to differ. Returns (model, record)."""
    from aadistill.initialization.planning import frozen_mapping as FM
    kept = [int(b) for b in sel["kept_blocks"]]
    model = FM.keep_blocks(adapter, teacher, kept, seed=seed)
    rec = {"method": method, "spans": {}}
    for span in spans:
        tensors, how = merged_tensors(adapter, teacher, span, method, stats)
        LM.write_block(adapter, adapter.blocks(model)[child_index(sel, span[0])], tensors)
        rec["spans"][str(list(span))] = how
    changed = {child_index(sel, s[0]) for s in spans}
    same, differ, missing = 0, [], []
    sd = model.state_dict()
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    for name, t in sd.items():
        want = digests.get(name)
        if want is None and tied and name == "lm_head.weight":
            continue          #: the tied head IS the embedding; save_pretrained writes it once, as the embedding
        if want is None:
            missing.append(name)
            continue
        block = int(name.split(".")[2]) if name.startswith("model.layers.") else None
        got = FM.tensor_sha256(t)
        if block in changed:
            continue
        if got == want:
            same += 1
        else:
            differ.append(name)
    unchanged_d = [n for n in digests if n not in sd and not n.startswith("lm_head")]
    spans_differ = {str(list(s)): any(FM.tensor_sha256(sd[n]) != digests[n] for n in sd
                                      if n.startswith(f"model.layers.{child_index(sel, s[0])}."))
                    for s in spans}
    rec["checks"] = {"tensors_outside_spans_equal_D": same, "tensors_outside_spans_differ": differ,
                     "names_not_in_D": missing, "D_names_absent": unchanged_d, "span_blocks_differ_from_D": spans_differ,
                     "num_parameters": int(sum(p.numel() for p in model.parameters()))}
    if differ or missing or unchanged_d or not all(spans_differ.values()):
        raise MergeError(f"{method}: the candidate is not D outside the spans or a span block is D's: {rec['checks']}")
    return model, rec


# ---- numerical checks: folding alone, each permutation alone -------------------------------------------------------

def _cast_kwargs(kwargs: dict, dtype) -> dict:
    def cast(x):
        if torch.is_tensor(x) and x.is_floating_point():
            return x.to(dtype)
        if isinstance(x, tuple):
            return tuple(cast(y) for y in x)
        return x
    return {k: cast(v) for k, v in kwargs.items()}


def _out(x):
    return x[0] if isinstance(x, tuple) else x


@torch.no_grad()
def capture_block_inputs(model, adapter, blocks_idx, packed) -> dict:
    """Run one forward; return ``{block: (hidden input, kwargs, output)}`` for the given block indices."""
    blocks = adapter.blocks(model)
    store, handles = {}, []
    for b in blocks_idx:
        def pre(_m, args, kwargs, _b=b):
            h = args[0] if args else kwargs["hidden_states"]
            kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            store[_b] = [h.detach(), kw, None]

        def post(_m, _args, _kwargs, out, _b=b):
            store[_b][2] = _out(out).detach()
        handles.append(blocks[b].register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(blocks[b].register_forward_hook(post, with_kwargs=True))
    try:
        forward(model, packed)
    finally:
        for h in handles:
            h.remove()
    return store


@torch.no_grad()
def equivalence_checks(adapter, teacher, spans, stats, packed, tolerance: float) -> dict:
    """For every span block, in float32 on the first packed sequence: the block with its gains folded, and (absorbed
    blocks) the block with its alignment permutation alone, against the unmodified block; the error is
    ``||y' - y||_F / ||y - x||_F`` (relative to the block's own update). A failure is a defect of the mechanism."""
    n_q, n_kv, hd = adapter.head_groups(adapter.spec_of(teacher))
    blocks = adapter.blocks(teacher)
    involved = sorted({b for s in spans for b in s})
    cap = capture_block_inputs(teacher, adapter, involved, packed)
    out, worst = {}, 0.0
    for s in spans:
        for k, b in enumerate(s):
            h, kw, _ = cap[b]
            x32, kw32 = h.to(torch.float32), _cast_kwargs(kw, torch.float32)
            t = LM.block_tensors(adapter, blocks[b])
            ref_blk = LM.block_like(adapter, blocks[b], t, dtype=torch.float32)
            y = _out(ref_blk(x32, **kw32))
            update = (y - x32).norm()
            rec = {}
            fold = LM.block_like(adapter, blocks[b], LM.fold_norm_gains(adapter, t), dtype=torch.float32)
            rec["fold"] = float((_out(fold(x32, **kw32)) - y).norm() / update)
            del fold
            if k > 0:
                key = f"pair{s[0]}_{b}"
                perm = LM.permute_block(t, ffn_perm=stats[f"{key}.ffn_perm"].tolist(),
                                        kv_perm=stats[f"{key}.kv_perm"].tolist(),
                                        q_perm_within=stats[f"{key}.q_perm_within"].tolist(),
                                        n_q=n_q, n_kv=n_kv, head_dim=hd)
                pb = LM.block_like(adapter, blocks[b], perm, dtype=torch.float32)
                rec["permutation"] = float((_out(pb(x32, **kw32)) - y).norm() / update)
                del pb
            del ref_blk
            out[str(b)] = rec
            worst = max(worst, *rec.values())
    return {"tolerance": tolerance, "worst": worst, "passed": worst <= tolerance, "blocks": out,
            "positions": int(cap[involved[0]][0].shape[1]), "precision": "float32 copies of the bf16 teacher blocks",
            "rule": "||y' - y||_F / ||y - x||_F over every position of the first packed sequence"}


# ---- diagnostics: residual updates against the teacher span's -------------------------------------------------------

@torch.no_grad()
def diagnostics(adapter, teacher, spans, candidates: dict, batches, active) -> dict:
    """Over the calibration at the statistics positions: for each span, the teacher span's attention, FFN and whole
    residual update (the sums over its blocks), and each candidate block's on the span's teacher input (``x``):
    norms, norm ratio, cosine and normalized error (`UpdateMoments`). ``candidates``: name -> {span start: block}.
    Diagnostics only: they select, gate and judge nothing."""
    blocks = adapter.blocks(teacher)
    dev = next(teacher.parameters()).device
    moments = {(str(list(s)), name, c): LM.UpdateMoments() for s in spans for name in candidates
               for c in ("attention", "ffn", "block")}
    span_of = {b: s for s in spans for b in s}
    store, sub, handles = {}, {}, []
    for s in spans:
        def pre(_m, args, kwargs, _s=s):
            h = args[0] if args else kwargs["hidden_states"]
            store[_s] = [h.detach(), {k: v for k, v in kwargs.items() if k != "hidden_states"}, None]

        def post(_m, _a, _k, out, _s=s):
            store[_s][2] = _out(out).detach()
        handles.append(blocks[s[0]].register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(blocks[s[-1]].register_forward_hook(post, with_kwargs=True))
    for b, s in span_of.items():
        outs = adapter.stream_out_projections(blocks[b])
        for role, comp in (("attn_out", "attention"), ("ffn_out", "ffn")):
            def acc(_m, _a, out, _s=s, _c=comp):
                key = (_s, _c)
                sub[key] = out.detach() if key not in sub else sub[key] + out.detach()
            handles.append(outs[role].register_forward_hook(acc))
    cand_sub: dict = {}

    def cand_hook(name, comp):
        def h(_m, _a, out):
            cand_sub[(name, comp)] = out.detach()
        return h
    cand_handles = []
    for name, by_span in candidates.items():
        for blk in by_span.values():
            outs = adapter.stream_out_projections(blk)
            cand_handles.append(outs["attn_out"].register_forward_hook(cand_hook(name, "attention")))
            cand_handles.append(outs["ffn_out"].register_forward_hook(cand_hook(name, "ffn")))
    n_pos = 0
    try:
        for packed in batches():
            store.clear()
            sub.clear()
            forward(teacher, packed)
            valid = valid_rows(packed, active).to(dev)
            if not bool(valid.any()):
                continue
            n_pos += int(valid.sum())
            for s in spans:
                x, kw, y = store[s]
                width = x.shape[-1]
                ref = {"block": (y - x).reshape(-1, width)[valid],
                       "attention": sub[(s, "attention")].reshape(-1, width)[valid],
                       "ffn": sub[(s, "ffn")].reshape(-1, width)[valid]}
                for name, by_span in candidates.items():
                    cand_sub.clear()
                    yc = _out(by_span[s[0]](x, **kw))
                    got = {"block": (yc - x).reshape(-1, width)[valid],
                           "attention": cand_sub[(name, "attention")].reshape(-1, width)[valid],
                           "ffn": cand_sub[(name, "ffn")].reshape(-1, width)[valid]}
                    for c in ("attention", "ffn", "block"):
                        moments[(str(list(s)), name, c)].add(got[c], ref[c])
    finally:
        for h in handles + cand_handles:
            h.remove()
    out = {}
    for (s, name, c), m in moments.items():
        out.setdefault(s, {}).setdefault(name, {})[c] = m.summary()
    return {"positions": n_pos, "spans": out,
            "reference": "the teacher's own span on the same input: attention = the sum of its blocks' attention "
                         "updates, ffn = the sum of their FFN updates, block = the span's whole residual update",
            "role": "diagnostic only (design.json :: diagnostics): it selects, gates and judges nothing"}


# ---- CLI ($0) ------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digests", help="D's per-tensor digests from its checkpoint ($0)")
    d.add_argument("--checkpoint", required=True)
    sub.add_parser("spans", help="the spans, from the frozen selection, checked against the design")
    a = ap.parse_args(argv)
    if a.cmd == "digests":
        return digests_command(a)
    print(json.dumps(merge_spans()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
