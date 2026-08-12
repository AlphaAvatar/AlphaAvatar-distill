# Micro-preflight plan — measure five things, rerun the control, then delete the pod

**Status: PROPOSAL, NOT AUTHORIZED. Nothing has been launched.**

This is **not** Phase A. It is a short single-GPU session that replaces five
estimates with numbers and produces the canonical control probes, so the Phase A
authorization rests on measurement rather than on a range. **Phase A must not
start automatically when it finishes.**

---

## 1. What changed, and why it now costs more

The recipe-fingerprint audit answered a question that had only been half-asked.
The historical `e1_r0860k_sa_pca` / `e1_r0860k_sb_pca` checkpoints **exist, hash-
verify, and are not recipe-matched to the Phase-A probes.**

37 of 43 recipe fields match exactly. What does not:

| field | historical | Phase A today | verdict |
| --- | --- | --- | --- |
| `trainer_git_commit` | `69c3fe1f`, **dirty** | current HEAD | **material** |
| `torch_version` | 2.11.0+cu128 | 2.13.0 | **material** |
| `trainer_uncommitted_sha256` | `2e04f683…` | — | **unreconstructable** |
| `kd_chunk` | not recorded | 512 | unverifiable |
| `pack` | `ladder_uniform` | `ladder_uniform_probe` | benign — *same bytes* |
| `resume_semantics` | no consumed-block accounting | present | benign — neither run resumed |

`src/aadistill/training/train.py` is **+528 / −30** lines since that commit. The
added machinery (extra stream, LoRA, gradient-share diagnostics) is inactive for
this recipe, and the ladder loader is untouched — but "probably inactive" is not
"identical", and the historical tree was dirty in a way that cannot be
reconstructed. A control trained by a different trainer build on a different torch
confounds the initialization with the trainer.

**So the canonical control is rerun under the current frozen trainer**, and those
two runs become the **permanent Phase-A control probes** — retained, reused at
rung 2, and never repeated.

The pack mismatch is worth recording as resolved rather than assumed: the relay
holds the historical pack under its historical name, and
`stage3_recovery_corpus_v2/ladder_uniform/blocks.npz` hashes to `6f324cb0…` —
exactly what the frozen recipe pins for `ladder_uniform_probe`. A rename, proved
by hash.

## 2. What is measured or produced

| # | item | why it cannot be done at $0 |
| --- | --- | --- |
| 1 | activation-statistics wall clock: GPU forward vs float64 CPU accumulation, real 4B teacher, real 59,763-position mixture | needs the teacher on a GPU |
| 2 | state-evaluator repeatability: one checkpoint scored 10× on the real suite, per-objective range | GPU reduction non-determinism is invisible on CPU |
| 3 | peak GPU resident memory on the widest operator (DEPTH at full width) | the 14.3 GiB figure is arithmetic |
| 4 | **canonical control rerun**: `qwen3_0p6b_init_v0` → 0.86M, seeds sa and sb, current frozen trainer | recipe-matched controls do not exist |
| 5 | control characterization on `recovery_search_v1`, both seeds: pooled and per-seed `usable_rollout_rate` / `correct_overall`, per-capability breakdown | needs rollouts from the reruns |
| 6 | checkpoint write/read throughput for a 5.99 GiB intermediate | the 106 GiB working-set plan assumes disk is not the bottleneck |

Both rerun inputs are on the relay and hash-verified: the canonical init
(`86fbba78…`, 1.11 GiB) and the pack (`6f324cb0…`, 13.7 MiB). Nothing needs to be
uploaded from the dev box, whose uplink is 0.72 MB/s.

## 3. What is explicitly **not** done

* no beam search, no operator search, no Phase A;
* no searched candidate is created, scored or ranked;
* no threshold is chosen with a candidate in view — items 1–3 and 6 are properties
  of the machinery, and 4–5 are properties of the incumbent.

## 4. Hardware

| | |
| --- | --- |
| GPU | 1 × L40S 48 GB, $0.99/h — the card the 88.83 TFLOP/s anchor was measured on |
| container disk | ≥ 60 GiB |
| expected wall clock | ~3.5 h including setup and both reruns |

## 5. Cost

| item | expected | hard |
| --- | ---: | ---: |
| setup, teacher/init/pack stage-in | $0.25 | $0.50 |
| (1) statistics-pass split, 3 repeats | $0.15 | $0.30 |
| (2) evaluator repeatability, 10 scorings | $0.20 | $0.40 |
| (3) peak-memory probe | $0.15 | $0.30 |
| (4) **two 0.86M control reruns** (1,023 steps × 4.15 s/step × 1.20, each) | $2.80 | $4.06 |
| (5) control characterization, 2 × 190 prompts | $0.60 | $1.10 |
| (6) disk throughput | $0.05 | $0.10 |
| setup/redraw reserve | — | $1.80 |

**Expected $4.20. Maximum authorized: $8.60.**

Up from $1.55/$4.50 entirely because of item 4. That $2.80 is **not** disposable
setup: it produces the two permanent Phase-A control probes, so Phase A afterwards
needs **7** probes rather than 9 and its probe cost drops by the same $2.80. Net
project cost is unchanged; what changes is that the control is actually matched.

### Budget position

```
actual cumulative spend                $180.7033
authorized cumulative cap              $211.07
unused, uncommitted                     $30.3667
this micro-preflight, maximum            $8.60
remaining for Phase A afterwards        $21.7667
```

## 6. Stop rules

The session **stops and reports** rather than continuing if:

* setup exceeds 40 minutes — redraw once, then stop;
* a control rerun diverges from its expected step time by more than 25% — the
  trainer is not behaving as priced;
* peak resident memory exceeds 40 GiB — the hardware plan is wrong and Phase A
  needs re-pricing before it is booked;
* spend reaches $8.60.

**Measurement 2 has a frozen response rule, not a judgement call.**
`EpsilonResponseRule` (`conservative_review_gate@v1`) is fixed before the
measurement: if the measured per-objective range is **below** 1e-4, epsilon stands
unchanged; if it **reaches or exceeds** 1e-4, no new epsilon is derived
automatically — the preflight is marked as requiring review and **Phase A is
blocked**. The rejected alternative, `epsilon_final = max(1e-4, 2 × measured)`, is
recorded in the rule: it would derive a scientific beam tolerance from a single
profiling run.

## 7. Outputs, and what they change

1. **Statistics-pass split** → collapses the Phase A search cost from $0.93–3.57
   to a point estimate.
2. **Evaluator repeatability** → the frozen response rule decides: epsilon stands,
   or Phase A is blocked pending review.
3. **Peak memory** → confirms or replaces the 14.3 GiB arithmetic and the "L40S is
   sufficient" claim.
4. **Two recipe-matched control checkpoints** → permanent Phase-A control probes,
   retained with hashes and a run manifest carrying a complete
   `RecoveryRecipeFingerprint` (so this audit never has to be reconstructed again).
5. **Control characterization, pooled and per-seed** → materializes three frozen
   rules: the seed-aware equivalence interval
   `2 · max(binomial_se, |p_sa − p_sb|/2)`, the seed-aware feasibility floor
   `max(0.30, u_pool − 3 · max(binomial_se, |u_sa − u_sb|/2))`, and the
   catastrophic rule's per-capability reference values.
6. **Disk throughput** → confirms or corrects the 106 GiB / 135 GiB plan.

Then: **delete the pod, verify it is gone, and stop for review.**

## 8. What the preflight does not decide

It measures the machinery and characterizes the incumbent. It produces no searched
candidate, so no threshold it materializes can have been chosen with a candidate in
view. That ordering is the entire reason it is a separate session.

## 9. Session contract

[`scripts/pod/AGENTS.md`](../scripts/pod/AGENTS.md) applies unchanged: detached
start via `start_job.py`, `watchdog.py` beside the launcher from creation,
`LogRelay` mirroring continuously, `collect_artifacts.py` gating teardown, and
termination confirmed by polling the control plane rather than by a return code.

Before launch, the suite runs the two ways the E8b failures did not:

```
OMP_NUM_THREADS=8 taskset -c 0-12 python -m pytest tests -q
HIDDEN_PATHS="..." OMP_NUM_THREADS=8 taskset -c 0-12 bash scripts/pod/simulate_pod_env.sh
```

The control reruns additionally record a full `RecoveryRecipeFingerprint` in their
run manifests, including the pack content hash, `kd_chunk`, the trainer commit and
dirty state, and the torch version — the four things the historical runs did not,
which is why this audit was necessary at all.
