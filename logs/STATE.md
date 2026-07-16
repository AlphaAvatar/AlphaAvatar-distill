# Current project state

Updated: 2026-07-16 (UTC+8 dev box) — Stage 1 gate-passed; RunPod control plane
verified read-only (see Environment).

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV).

User decisions (2026-07-13, see decisions.md): student target **0.6B-class**
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb — Qwen3-0.6B geometry);
**BF16 training, INT8 deployment target**; warm-up v1 public-corpus download approved.

Pipeline position: **Stage 0 passed (v1) → Stage 1 passed → next is Stage 2**
(offline warm-up data collection), then Stage 3 recovery.

Verified state of the initialized student (real teacher, real stats, CPU):

- checkpoint `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint` (596.0M params, bf16);
- held-out eval (21,080 tokens): teacher NLL 2.63 / init 11.75 / random-init 12.13;
- init beats random but width (2560→1024) is the dominant zero-shot bottleneck —
  expected pre-recovery; see the Stage 1 experiment log's ablation table.

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
- `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1, safetensors 0.8.0,
  **datasets** (added 2026-07-13 for warm-up v1), pytest.
- Known nondeterminism (logged per P5): two model instances with bitwise-identical
  bf16 weights can differ by a few ULPs in logits (oneDNN/AMX alignment-dependent);
  each instance is self-deterministic. Gate checks compare weights bitwise and
  logits with tolerance 0.5.
- RunPod control plane verified read-only 2026-07-16: `runpodctl` 2.7.1 installed
  (`~/.local/bin`), authenticated (`~/.runpod/config.toml`, not in repo), balance
  $250 / $0 per hr spend / $80 spend limit, no pods or volumes exist, 21 GPU types
  listable, 2 account SSH keys with local private keys present (`~/.ssh/id_ed25519`
  ed25519 + `~/.runpod/ssh/runpodctl-ssh-key` RSA). Skill at
  `.agents/skills/runpodctl` (pinned in `skills-lock.json`). No paid resource was
  created. Ready to rent a GPU worker for Stage 3 pending user approval (P12).

## What exists and why

- `src/aadistill/` — `env.py`, `manifest.py`, `teacher.py`, `collect.py` (Stage 0
  streaming sufficient statistics), plus Stage 1 core added this session:
  - `project.py` — weighted global stream projection from stats (uncentered,
    trace-normalized, end-points upweighted 9/8), FFN neuron importance,
    final-norm least-squares diagonal;
  - `sandwich.py` — middle-band depth span map (first-of-span representative),
    GQA-preserving Q-head selection, norm-folding sandwich init, `init_student`;
  - `student.py` — Qwen3 student config/model builder (teacher-inherited keys).
- `scripts/` — `collect_stage0.py`; new: `build_warmup_v1.py`, `build_holdout_v1.py`
  (revision-pinned dataset builders), `init_stage1.py` (init + gate checks +
  manifest), `eval_ppl.py` (deterministic NLL/ppl eval).
- `configs/` — `stage0_qwen3_4b_thinking.json` (v0), `stage0_qwen3_4b_thinking_v1.json`,
  `stage1_qwen3_0p6b_from_4b_thinking.json`.
- `data/warmup/` — `warmup_v0.jsonl` (committed), `warmup_v1.jsonl` + `holdout_v1.jsonl`
  (gitignored, rebuildable; committed `.manifest.json` files pin source revisions,
  licenses, hashes). v1: 3,202 samples / 949,859 tokens (fineweb-edu ODC-By,
  dolly CC-BY-SA, gsm8k MIT, mbpp CC-BY-4.0, v0 handcrafted).
- `tests/` — `test_collect_toy.py` (8) + `test_stage1_toy.py` (6, incl.
  identity-projection exactness for the full sandwich algebra). All 14 pass.
- `artifacts/` (gitignored) — `stage0/qwen3_4b_thinking_v1/` (1.95 GB stats cache +
  manifest), `stage1/qwen3_0p6b_init_v0/` (checkpoint, random_baseline, manifest,
  eval report, run logs).
- `logs/` — decisions (4 records), experiments (3), supported_models, this file.

## Latest known working commands

```
uv run pytest tests/ -q                                                # 14 passed
uv run python scripts/build_warmup_v1.py                               # rebuild v1 data
uv run python scripts/collect_stage0.py --config configs/stage0_qwen3_4b_thinking_v1.json
uv run python scripts/init_stage1.py --config configs/stage1_qwen3_0p6b_from_4b_thinking.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl --model <dir-or-hf-id> ...
```

## Latest verification

- Stage 0 v1 gate: passed 2026-07-13 (62 min, bitwise-deterministic teacher,
  cache 1.95 GB < 2.5 GB budget, full-rank mid-layer covariance).
- Stage 1 gate: passed 2026-07-14 (loads, finite forward, 596.0M params, bitwise
  weight round-trip, bitwise-reproducible init, eval report + random baseline).
- First-attempt init failed usefully (worse than uniform); root causes isolated by
  single-axis ablation and fixed (middle-band depth merge, end-weighted P). Full
  history in the Stage 1 experiment log.

## Not done yet (next, in order)

1. **Stage 2 — offline warm-up data collection**: data groups per AGENTS.md 4.4
   (instruction, RAG, multi-hop, tool, refusal, code/math, realtime, long-context,
   quant calibration), manifests, loader dry run. Large downloads / teacher-generated
   corpora need user approval.
2. Stage 3 recovery design (FFN/norm → block → span → full offline KD) — this is the
   designed repair for the width bottleneck. Serious training runs need user approval
   (and likely GPU rental).
3. Stage 1 ablation backlog (optional): function-aware width subspace, per-group P
   with Procrustes chaining, activation-based head importance.

## Open decisions for the user

- None blocking Stage 2 scaffolding. Blocking later: approval for large Stage 2
  downloads / teacher-generation runs; approval + hardware plan for Stage 3 training.

## Links

- `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`
- `logs/decisions.md` (2026-07-13 ×3, 2026-07-14 recipe fixes)
- `logs/supported_models.md`
