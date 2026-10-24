# H1-PA — the PCA-aligned redesign of H1

**Status (2026-10-24): DESIGN, NOT FROZEN. NO-GO for formal training** until
the maintainer decides how D4-Reasoning is scheduled and packed (section 6).
Nothing is authorized; no GPU has been used.
Study identity: `stage2.reasoning_data.h1pa`, owned by
[`configs/stages/stage-2/reasoning_data/h1pa.json`](../../../../configs/stages/stage-2/reasoning_data/h1pa.json).
Every number below is computed by
[`h1pa_audit.py`](../../../../scripts/stages/stage-2/reasoning_data/h1pa_audit.py)
and owned by
[`audit/h1pa_audit.json`](../../../../logs/stages/stage-2/reasoning_data/audit/h1pa_audit.json).

## 1. Why H1 was annulled

H1 is annulled for promotion
([`plans/h1_annulment.json`](../../../../logs/stages/stage-2/reasoning_data/plans/h1_annulment.json)).
It did not run the PCA baseline the maintainer required:

- arm A started from D4 Top-1 instead of E1's PCA initialization;
- it trained 614 optimizer steps, where E1's 2.96M run trained 2,916;
- it used its own data selection and best-fit-decreasing packing instead of
  E1's pack;
- it evaluated with greedy decoding.

Its results, runs, money, evidence and checkpoints are kept unchanged. Its
grant is closed, and the $28.16 remainder funds nothing. The P1/P1t
teacher data is not annulled.

## 2. Three arms, three comparisons

| arm | initialization | training data | recipe | execution |
| --- | --- | --- | --- | --- |
| PCA-Original | E1's PCA init | E1's 2.96M pack | E1 | reuse E1's two checkpoints, re-evaluate |
| D4-Original | D4 Top-1 | E1's 2.96M pack, the same bytes | E1 | train, seeds 20260726 and 20260801 |
| D4-Reasoning | D4 Top-1 | `stage2.reasoning_master@v3`, 70/30, packed by E1's sequential packer | E1 | train, same two seeds |

- **A — initialization effect** (Q1): D4-Original − PCA-Original. Only the
  initialization differs (section 4).
- **B — data effect under D4** (Q2): D4-Reasoning − D4-Original. Its training
  budget cannot match on every quantity at once (section 5). The conclusion
  is limited to whatever the chosen schedule controls.
- **C — end to end**: D4-Reasoning − PCA-Original. Descriptive only, because
  initialization and data change together. No PCA-Reasoning arm is run, so
  this is not a 2×2 and no interaction is estimated.

## 3. PCA-Original is E1 as it ran (28/28 identity checks pass)

**Training data.**
- E1 trained from `artifacts/stage3/ladder_uniform`. Its run manifests record
  that pack's hashes, and `artifacts/shared/instruments/ladder_uniform_probe`
  carries the same files: `blocks.npz` `6f324cb0…`, `ladder.json` `d4941722…`,
  `audit.jsonl` `15f16b7b…`, built from corpus_v2 `sessions.jsonl` `2b4edc2e…`.
- E1's packer (`scripts/shared/data/build_token_ladder.py`) rebuilds that pack
  **byte for byte** from those sessions: same `blocks.npz`, same audit, same
  arrays.

**Configs.** Canonical hashes match: seed 20260726 `c2e46e61…` (E1 sa's run
manifest) and seed 20260801 `8b385927…` (`configs/stages/stage-3/e6b/provenance.json`;
sb's own run manifest was not preserved).

**Initialization.** The PCA init weights are `86fbba78…`.

**Checkpoints.**
- Seed 20260726, `3f08482c…`, is on the relay; its pod-side hash file is there
  too.
- Seed 20260801, `b658fe39…`, exists **only on the dev box**
  (`artifacts/stages/stage-3/rescued/`). All 6 of its files match the pod-side
  hashes. It must reach durable storage before any session reads it.

**Recipe as E1 ran it** (L40S, 10,232 s for 2,916 steps, logged peak 39.96 GB):

| setting | value |
| --- | --- |
| batch size | 2 sequences |
| micro-batch size | 1 |
| gradient accumulation steps | 2 |
| sequence length | 8,192 |
| epochs / optimizer steps | 3 / 2,916 |
| warmup | 146 steps, then cosine to 0.1 of peak |
| optimizer | AdamW, lr 5e-5, betas (0.9, 0.95), eps 1e-8, weight decay 0.01, gradient clipping 1.0 |
| loss | CE 0.25 + KD 1.0 (T 1, every non-padding position) |
| numerics | fp32 weights, bf16 autocast, gradient checkpointing |
| teacher | Qwen3-4B-Thinking-2507 @ `768f209d`, bf16 |

## 4. D4-Original is E1 with one factor changed

- **Same data, same order.** It trains on E1's pack, the same bytes. Today's
  trainer reproduces E1's training order: at **all 291 logged steps of both
  seeds**, the per-step supervised-token and KD-position counts and the
  learning rate equal E1's own logs.
  - Seed 20260801 is checked through `e1_r2960k_sb_rand`, which ran the same
    pack and seed.
  - The validation set is identical: 16 sequences starting at sequence 2941.
- **Same architecture.** D4 Top-1 (`fc28e0dd…`) has the same 310 tensors,
  names and shapes as the PCA student: 596,049,920 parameters, bf16 stored,
  fp32 trained.
- **Harmless config differences.** Only `max_window_layers` (unused without
  sliding window) and `use_cache` (inference only) differ. D4 ships no
  tokenizer, so it uses the PCA directory's tokenizer, the files E1 loaded.
- **Derived config.** Each seed's E1 config, with only `student_path`,
  `tokenizer_source`, `data_dir`, `run_name`, `out_dir` and `_purpose`
  replaced.

**Comparison A is a controlled initialization comparison.** What remains to
show on a GPU: D4-Original's first logged steps match E1's counts, and GPU
non-determinism is bounded by the floor already measured
(H1.md §16: two identical runs differ from step 2).

## 5. D4-Reasoning under E1's sequential packing

**The pool.** D4-Reasoning's sessions are re-derived from the P1 and P1t stores
by the feasibility tool's own functions, and hash to `6458f2a8…`, the sessions
the @v3 pack was built from:
- 7,492 sessions in 7,377 units;
- no teacher generation.

**How the packer receives them.** E1's packer gets them unchanged except for
two mapped fields:
- `data_type` is the mixture cell (`stratum/source`), the key @v3 declares its
  shares in;
- `source_id` is the unit.

**The packs compared.** The second column is E1's own pack, used by
PCA-Original and D4-Original:

| at the 2.96M cut | E1's pack | D4-Reasoning, E1 packer as is | same packer, terminal truncation off (not adopted) |
| --- | ---: | ---: | ---: |
| packed sequences | 1,944 | 440 | 568 |
| source sessions / units | 4,524 / 4,354 | 1,860 / 1,766 | — |
| supervised tokens | 2,960,507 | 2,964,159 | 2,962,543 |
| non-padding tokens | 4,730,748 | 3,567,472 | 3,437,587 |
| padding ratio | 70.3% | 1.0% | 26.1% |
| supervised tokens per sequence (p10 / p50 / p90) | 177 / 296 / 5,709 | 5,291 / 7,237 / 7,804 | — |
| sequences under 15% full | 1,373 | 0 | — |
| truncated sessions | 352 | 372 | 0 |
| supervised tokens discarded by truncation | 629,315 (500,065 reasoning, 129,250 answer) | 814,745 (679,996 reasoning, 134,749 answer) | 0 |
| truncated targets with no answer left | 307 (+43 answer cut) | **331** (+41 answer cut) | 0 |
| largest drift from the declared mixture | 0.04 pp | **−3.42 pp** compositional, +3.88 pp supporting | 0.13 pp |
| units split by the cut | 133 | 15 | — |
| validation (16 sequences) | from sequence 2941; 1 unit and 5 prompts also in training | from sequence 944; **supporting data only**, 0 overlap | — |

All 65 rule families are present in D4-Reasoning's 2.96M cut.

**Why E1's packer distorts this data.** It packs sessions in order and cuts the
last session of a sequence at the boundary. For long verified reasoning traces
the cut keeps only 69–76% of the supervision drawn, against 79–96% for the short
supporting sessions. To retain 2.96M:
- the analogical cell is drawn to 100% of its supply and compositional to 98%;
- supporting data fills the rest;
- 331 verified trajectories are trained without their verified answer.

E1's own pack loses 307 answers the same way.

## 6. The training-budget conflict (needs the maintainer)

At E1's batch size 2 and 3 epochs:

| | optimizer steps | epochs | supervised-token exposure | non-padding-token exposure | positions computed | training FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA-Original, D4-Original | 2,916 | 3 | 8,881,521 | 14,192,244 | 47,775,744 | 9.07e17 |
| D4-Reasoning, epoch-matched | 660 | 3 | 8,892,477 | 10,702,416 | 10,813,440 | 2.05e17 |
| D4-Reasoning, step- and FLOPs-matched | 2,916 | 13.26 | 39,288,580 | 47,285,220 | 47,775,744 | 9.07e17 |

**How FLOPs are counted.**
- E1's config does not set `batch.truncate_padding`, so student and teacher
  forward every one of the 8,192 positions, padding included.
- Per position, about 19.0 GFLOP:
  - **student:** 4 forwards, i.e. a forward, a backward counted as 2 forwards,
    and the gradient-checkpointing recompute;
  - **teacher:** 1 forward.
- Each forward follows Kaplan et al. 2020: 2N + 2·n_layer·n_ctx·d_attn, plus the
  unembedding matmul, which that convention omits but which matters with a
  151,936-token vocabulary.

**No schedule matches epochs, optimizer steps, supervised exposure and FLOPs
together.** The two packs hold the same supervised tokens in 1,944 and 440
sequences: 1,523 against 6,737 supervised tokens per sequence. 1,373 of E1's
1,944 sequences (70.6%) are under 15% full, every one of them tool calling
(1,347 hold a single session), because the system prompt is a hard packing
boundary. A step-matched D4-Reasoning sees its data
13.3 times; an epoch-matched one gets 4.4× fewer updates. Changing the batch
size, the learning-rate schedule, repeating samples or padding to match is
excluded.

**Options** (none chosen). They combine a schedule with a packing:

| schedule | what Comparison B then controls | cost of the whole study (L40S, $1.09/h, incl. $7 storage) |
| --- | --- | --- |
| A — epoch-matched (660 steps; 852 with truncation off) | same epochs and supervised exposure; 4.4× fewer optimizer steps and FLOPs | $24.6–33.3 |
| B — step- and FLOPs-matched (2,916 steps) | same optimizer steps, LR schedule and FLOPs; 13.3 epochs (10.3 with truncation off) | $29.4–38.1 |
| C — both schedules, two extra runs | brackets the effect between the two controls | $34.0–45.6 |

The packing choice is either **(i)** E1's packer as is, with terminal
truncation (exactly E1's behaviour), or **(ii)** the same packer with its own
built-in switch `allow_terminal_truncation=False`. (ii) keeps E1's order,
interleave and system-prompt grouping, removes the 331 lost answers and the
drift, and costs 128 more sequences.

**Recommendation: (ii) with C.**
- **Why C:** Q2 asks for a controlled compute budget, which is B, and B's
  residual confound is repetition. A gives the exposure-matched reading. The
  claim is supported only if both agree.
- **Why (ii):** under (i), a fifth of the reasoning arm's verified answers are
  never trained on.
- **What stays exact:** PCA-Original and D4-Original keep E1's pack exactly
  either way.

## 7. Independence and validity

**E1's training data against eval-set-1**
([`audit/asset_audit.json`](../../../../logs/stages/stage-2/reasoning_data/audit/asset_audit.json)):
- **Fluid items:** no shared prompt key, and no 13-gram near-copy of any
  `fluid_reasoning_720` item. The primary fluid comparisons are isolated.
- **Near-copies:** 9 of the 28 flagged corpus_v2 near-copies are in E1's 2.96M
  cut, 8 of `eval_behavior_v0` prompts and 1 of a `multihop` item. Those units
  are reported with this caveat.
- **`stage3_legacy_150`:** made of E1's own training prompts; in-sample only.
- **Nothing is deleted** from E1's data to "pass" isolation.

**D4-Reasoning's pool:** R0–R9 pass for every generated unit, and the reuse
selection passes its own record (`audit/corpus_v3_feasibility.json`). Packing
changes no content.

**D4's calibration sets:** built under `require_disjoint`
(`scripts/stages/stage-0/calibration_volume/build_task_sets.py`).

## 8. Evaluation: nucleus sampling (`nucleus_v1`)

[`configs/shared/evaluation/decoding/nucleus_v1.json`](../../../../configs/shared/evaluation/decoding/nucleus_v1.json),
read through `scripts/shared/evaluation/decoding.py`.

**Sampling.**
- Pure top-p: temperature 0.6, top-p 0.95, top-k and min-p disabled, no
  penalties.
- These are the teacher's model-card thinking-mode values without its
  TopK=20.
- One sample per item.

**Seeds.**
- Each request's seed is sha256(base | unit | item | turn | sample).
- It never depends on the model, so all six checkpoints draw the same seed for
  the same item and turn.

**Unchanged, and what each sample records.**
- Context (P18) and the EOS and degeneration stops are unchanged.
- Every sample records its seed, output token ids and `stop_reason`
  (eos / degeneration / context_limit).
- Each run records the protocol's path and sha256.

**Where it is wired.**
- `uncapped_eval.py --decoding` and `run_battery.py --decoding
  --seed-namespace`.
- `dose_driver.evaluate_all` passes it on when a preregistration names
  `evaluation.decoding`.
- Without the flag both runners make their frozen greedy call unchanged.

**Boundaries.**
- eval-set-1's items, validity mask and scoring are untouched.
- No greedy result is compared with a nucleus one.

Tested with stub engines. A real vLLM run belongs to the GPU validation
session.

## 9. Hardware, storage, budget

**GPU: L40S (48 GB).** E1 ran exactly this recipe on L40S with a logged 39.96 GB
peak. A live query (2026-10-24) found L40S at $1.09/h. EU-NL-1, EUR-IS-2 and
US-MO-1 each have L40S stock ("Low") and network-volume support. Choose the
datacenter first, then create one volume there.

**Storage.** About 50 GB:
- the D4 init;
- PCA-Original seed 20260801 (dev-box-only today; it goes up first, about
  55 min at the dev box's uplink);
- the packs;
- the four new final checkpoints and their latest optimizer, RNG and
  save-point state;
- the evaluation progress.

PCA-Original seed 20260726 comes from the relay.

**Old volume.** H1's volume `1980hnh75i` (EUR-IS-1, no L40S) has no consumer
(the annulment record's consumer audit). Retiring it is the maintainer's
decision.

**Budget.** See section 6, from measured rates only:
- **Training:** 3.509 s per optimizer step (E1 sa, L40S).
- **Evaluation:** 4,763–9,526 s per checkpoint (the H1 forecast's L40S maximum,
  ×2 contingency for students off the floor).
- **Overhead:** 536 s per session.
- **GPU validation:** one 0.5 h session.

Suggested hard cap: **$50** for C. That needs a new grant; H1's remainder does
not transfer.

## 10. Risks

- **Floor.** Under greedy decoding, E1's PCA 2.96M student answered 10 of the
  570 formal static fluid items (1.75%), below the 2% floor H1's rule uses.
  Comparison A may resolve only on deduction, or not at all. Nucleus sampling
  changes the levels, and they are unmeasured.
- **Run-to-run variance.** Two seeds per arm. GPU training is not bitwise
  deterministic (H1.md §16).
- **Durability.** PCA-Original seed 20260801 has one copy until it is uploaded.

## 11. GO/NO-GO

**NO-GO for formal training now.**
- **Comparison A: scientifically ready.** It needs a grant, the study driver
  (train D4-Original from the derived config, evaluate six checkpoints under
  `nucleus_v1`) and one GPU validation session.
- **Comparison B: blocked.** It needs the maintainer's choice of schedule (A, B
  or C) and packing ((i) or (ii)). The choice freezes `h1pa.json`.

## Reproduce

```bash
PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_audit sessions \
  --store h1gen_20261022_190653=<P1 scr>/store --store h1p1t_20261023_094007=<P1t scr>/store \
  --out artifacts/stages/stage-2/reasoning_data/packs/D4R_seq_sessions.jsonl
# then the builder command it prints, --out artifacts/stages/stage-2/reasoning_data/packs/D4R_seq
PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_audit audit \
  --reasoning-sessions artifacts/stages/stage-2/reasoning_data/packs/D4R_seq_sessions.jsonl \
  --relay-sa <dir with E1 sa's relay run manifest and pod hashes> --e1-regen <E1 pack rebuilt> --price <live L40S $/h>
```
