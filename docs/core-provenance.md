# Core provenance — where `src/aadistill`'s guards came from

`src/aadistill` is the reusable algorithm core. It must read as mechanism:
mathematical definitions, interfaces, invariants, capability requirements and
failure semantics (AGENTS.md P3, 2.3). It must not document itself by one
campaign's run history — which experiment measured what, at which probe size, on
which seeds, for how many dollars — because a core that explains itself through
`E7`, `Phase-A attempt 5` and `$0.6426` is owned by that campaign.

Most of those guards are real, though, and each was paid for. Deleting the
reason would make the next agent remove the guard. So the reasons live **here**,
outside the core, keyed by the module and the mechanism they justify.

**This file authorizes nothing.** It is a register of why code exists. The
authoritative accounts stay where they already are: `logs/archive/repository/indexes/EXPERIMENTS.md` for
runs, `logs/budget/ledger.md` for money, `logs/state/phase_index.md` for phase
history, and the per-validation directories under `logs/validations/`.

`scripts/architecture/core_ownership.py` rule `instance_prose` is what keeps
core clean; this file is where what it removes is supposed to land.

---

## Register

Captured 2026-09-10 from the prose that was removed from core, verbatim, so
nothing was lost in the move. Each entry names the module whose mechanism the
incident justifies.


### `src/aadistill/data/e5_pack.py`

```text
    E5 examples are not sessions to be rendered: they are token streams that already
    source trajectory are never co-packed — for E5 that matters as much as it did for
    Production packing may cut the last session at a block boundary; E5 may not,
    """Adapt one E5 example record to the packer's input type.
    """Pack E5 examples with truncation forbidden.
    `ladder.json` declares ONE rung, covering the training blocks only. An E5
```

### `src/aadistill/data/extra_stream.py`

```text
    E7 asks whether adding general-text teacher KD restores general language
```

### `src/aadistill/data/paired_corpus.py`

```text
    # Attempt 4 measured what makes this necessary: R's supervised continuation runs
```

### `src/aadistill/data/prefix_split.py`

```text
    E5 asks whether training on **student-visited** prefix states beats training on
```

### `src/aadistill/data/sessions.py`

```text
    # E5 forbids cutting any sample: a cut prefix changes the state
```

### `src/aadistill/evaluation/general_text.py`

```text
    E6b showed two objectives improving validation CE by the same amount while only
    one moved autonomous behaviour, and the FineWeb NLL of the E1 lineage actually
```

### `src/aadistill/governance/authorization.py`

```text
    lived in prose: E6b overran by $0.56 with the number in a plan document, and a
    finished corpus build idled ~$8.70 because teardown was tied to a generous
    schema and stated "no code path to Phase A" for every artifact any
```

### `src/aadistill/governance/grant.py`

```text
    identity nobody computed. Phase A, Phase B, the continuation and C1 each carry
```

### `src/aadistill/governance/post_freeze.py`

```text
    frozen source sets — Phase A's harness, Phase B's executable, the recovery
    #: Phase-B values now live in `scripts/experiments/phase_b/post_freeze.py`.
    # `accounted_for` above answers one question: may a Phase-B launch run against
    # But "Phase B may not launch against this tree" and "nobody ever explained why
```

### `src/aadistill/infrastructure/artifact_gate.py`

```text
    E6b lost its structured training logs to a bundling step that listed
    `artifacts/audit/three_mode` and two JSON files — the set E6 needed — and never
    from E6, which did not train. `tar` bundled what it was given, exited 0, the
    that construct (E3, E4, E5 — frozen records of completed runs). It is banned
    # required because it is the one E6b lost, and because it is the only artifact
    tarball — that difference is the entire E6b artifact loss.
    # the products this session OWES off-pod actually secured? Phase-A attempt 11
```

### `src/aadistill/infrastructure/budget.py`

```text
    E6b billed $7.68 against a $7.12 authorization. The proximate cause was a step
    run actually sustained — a 14% miss, ~$0.81 of unbudgeted time. The structural
    hashing, transfer and verification have somewhere to happen. E6b had no such
    # The step time E6b actually sustained: L40S, Stage 3 ladder arms, 2916 steps,
    # E6b — is superseded. It is recorded in `SUPERSEDED_STEP_SECONDS` so that a
    This is the gate E6b did not have. The driver re-priced before each arm
    `step_time_floor` defaults to the E6b measurement because that is the only
    # materializes EARLY in a session — Phase A's reference-cache fallback is
```

### `src/aadistill/infrastructure/provider.py`

```text
    * **Polling** goes over the RunPod GraphQL API. Every launcher since E2 reads
```

### `src/aadistill/infrastructure/remote.py`

```text
    had returned in 74 seconds. So the lesson is not "use setsid": E6b already did.
    # killed the E6b setup at INIT_READY.
```

### `src/aadistill/infrastructure/session.py`

```text
    Phase-A attempt 1             $0.1075   ``SESSION_KIND`` leaked between two sessions
    device canary attempt 1       $0.0603   the base read ``self.a.teacher_revision``; the
    device canary retry           $0.0637   the shared setup copies two assets out of
    been paid for. Phase-A attempt 5 died at $0.6426 on a calibration file that
    #: Added 2026-08-22 because the five Attempt-12 leaves cannot travel by the
    retry, which had honestly declared it wanted none, died at $0.0637 when that
    #: deleted $2.82 of verified checkpoints on 2026-08-13 for want of this.
    controls of a $2.82 session.
    #: callable rather than a list because Phase A derives its nine probe streams
    #: while having secured nothing at all. Phase-A attempt 11 staged five
    #: log. C1 attempt 5's complete skip list was therefore ~100 lines that could
    between two sessions sharing one setup script ($0.1075) because it was a
    them anyway ($0.0637); here it reads `SESSION_ASSETS`.
    Device-canary attempt 1 was lost at $0.0603 because the base read three
```

### `src/aadistill/infrastructure/session_prechecks.py`

```text
    Phase A each had their own copy of one, and nobody could see the asymmetry
    checks out this commit. Continuation attempt 5 died at $0.1369 on a stale
    `check_lineage` adds Phase A's stronger third question: is everything else in
```

### `src/aadistill/infrastructure/session_runner.py`

```text
    """Read the probe by LABEL, never by line position (see e8b: a $0.19 misread)."""
    # $0.0603 because the machinery read three attributes its parser had
    # both controls of a $2.82 session.
    # The reports are fetched BEFORE the products: Phase A's
```

### `src/aadistill/infrastructure/source_identity.py`

```text
    inside a frozen source set, and rewriting them would move Phase-A and Phase-B
```

### `src/aadistill/infrastructure/watchdog.py`

```text
    Two E6b failures live here.
    has been the documented last-resort cost layer since E4 and has never once been
    the pod and must not hand the replacement a fresh meter (E6b launcher,
    the E6b inference — quiet log, therefore quiet session — is not expressible.
```

### `src/aadistill/initialization/calibration/datasets.py`

```text
    is no longer an out-of-sample number. E8a already hit a near-miss here — its
    existing E8 proof mean the same thing by the same rule.
```

### `src/aadistill/initialization/calibration/items.py`

```text
    it and :mod:`aadistill.autoinit.fixed_path` did not. C1 attempt 8 paid for that
    defect appeared once before, on Phase-A attempt 5, and for the same reason.
```

### `src/aadistill/initialization/calibration/profiles.py`

```text
    E8a measured its depth objective on one frozen 67-item domain-balanced mixture.
    without a pinned revision cannot support P4 reproduction, and leaving the
```

### `src/aadistill/initialization/device.py`

```text
    """The device contract for the Phase-A Stage-1 search path.
    false on a GPU. Attempt 6 died in the search's reload validation; attempt 7 died
    ``scripts/training/search_depth_map.py`` inserted ``.cpu()``, and E8a runs that
    reduction on the accelerator. Attempt 10 spent $11.43 discovering it. The
    checked it. Attempt 9 died at $0.34 on ``project.py``'s ``avg``, allocated with a
    # CPU budget. Added 2026-08-19 after Phase-A attempt 10.
    `autoinit_preflight_setup.sh` has computed this correctly since E8b and
```

### `src/aadistill/initialization/operators/attention_activation.py`

```text
    executable source set that Phase B's closed preregistration binds to digest
    not in the declared set, so the Phase-A/B executable identity is untouched. This
    #: THE TRANSFER BOUNDARY, and the defect C1 attempt 9 died on.
    #: evidence/cache form, and it is what gets hashed and kept. Attempt 9
```

### `src/aadistill/initialization/operators/base.py`

```text
    different algorithms for deciding it, and E8a showed they disagree: the
```

### `src/aadistill/initialization/operators/depth.py`

```text
    Two algorithms, deliberately kept as separate immutable ids because E8a showed
    **This returned ``.cpu()`` until 2026-08-19, and that cost $11.43.** E8a —
    copied off the device. Attempt 10 ran 10 h 47 m inside one expansion, GPU at
    where E8a left them.
    it against the real mixture: Phase-A attempt 5 died earlier, at the
    ``scripts/training/search_depth_map.py``, the E8a script whose algorithm this
    # PARTIAL CACHING. Until 2026-08-27 this was all-or-nothing, and Phase-B
    "E8a kept its cache on the accelerator and therefore checked
    it, which is the $11.43 failure.
```

### `src/aadistill/initialization/operators/width.py`

```text
    sees the teacher's. Those are different projections, and E8a is the evidence that
```

### `src/aadistill/initialization/planning/fixed_path.py`

```text
    C1 session owns those constants and hands them in, exactly as it hands in
```

### `src/aadistill/initialization/planning/generation_compat.py`

```text
    Phase-A attempt 4 was refused at $0.2052 by a binding that was working exactly as
```

### `src/aadistill/initialization/planning/ranking.py`

```text
    preference. E7 moved held-out FineWeb NLL by −5.22 nats and autonomous behaviour
    discarding one on a single step-0 measurement is exactly the mistake E8a
    # Not a style rule: a single-objective beam is the failure mode E7
    structural hypothesis on a step-0 measurement. E8a is the standing
    #: **NLL is not an objective.** E7 is the reason, and it is a direct measurement
    #: rather than a worry: a −5.22 nat swing in held-out NLL moved autonomous
    #: a single step-0 measurement is the mistake E8a documented: a proxy that looked
    should. Instead the preflight is marked as requiring review and Phase A does
    #: flag appears under. These were literals naming Phase A, so a generic
```

### `src/aadistill/initialization/planning/recovery.py`

```text
    ->  rung 1: identical 0.86M recovery on seed sa, all of them
    ->  rung 2: the control (unconditionally) + the best S searched leaves, seed sb
    ->  optional seed sc, for tied candidates only
    * **Selection is on autonomous behaviour, not state NLL** (E7: a −5.22 nat NLL
    0.1290 on seed alone.
    #: E1/P1 at the 0.86M probe rung. Frozen: AutoInitializer v1 changes the
    never computed correctness that way, and cannot: of the battery's 190 prompts
    only **170 are correctness-scorable**, because the 20 `code` items have no
    measured. The behaviour metric moves **0.1290 on training seed alone**, and
    #: and the Phase-A plan always declares one.
    #: one study -- 0.1290 for behavior_v0 -- and it belongs to that
    """Which **searched** leaves advance to seed sb.
    ``tie_pending``            finalists are equivalent after sa+sb; seed sc is
    fully compressed leaf — E8b measured exactly that reversal, DC beating DP by
```

### `src/aadistill/initialization/planning/search.py`

```text
    have continued to the watchdog's $23.05 ceiling.
    # inside the beam. A correct Phase-B loader delegating to
    # `config` entirely. So no state id, and no recorded Phase-A state, moves.
    # Phase-A attempt 6 died here. `_validate` forwards both models through
```

### `src/aadistill/initialization/planning/stage1_import.py`

```text
    """Import a *completed, verified* Phase-A Stage-1 result. Nothing weaker.
```

### `src/aadistill/initialization/planning/stage1_selection.py`

```text
    Phase-B attempt 4 paid for and **completed** an eight-hour joint P=2 search. It
    # because no file carries them. Attempt 5 omitted `arch_signature`,
```

### `src/aadistill/initialization/specs/identity_collapse.py`

```text
    Phase-B attempt 5 completed its joint P=2 search and then died in Stage 2 on
    retained Phase-A finalists: same content-derived state id, same re-derived
```

### `src/aadistill/initialization/specs/metrics.py`

```text
    longest — the same rule E8a used (``domain_balanced_score``).
```

### `src/aadistill/initialization/specs/state.py`

```text
    E8, generalized to every node of the search. The binding is to the *artifact*
```

### `src/aadistill/initialization/statistics/attention.py`

```text
    and that is what killed Phase-A attempt 7. The same rule applies here, and
    # FAIL CLOSED, and do not repair it here. C1 attempt 9 died on this exact
```

### `src/aadistill/initialization/statistics/collect.py`

```text
    # is what killed Phase-A attempt 7 in `ffn_abs_sum[idx] += ...`, and it
```

### `src/aadistill/initialization/transforms/nll_gate.py`

```text
    initialization NLL must not cancel or promote E8 — the endpoint is autonomous
    # The three general-language / teacher-native series E8 requires per checkpoint.
```

### `src/aadistill/initialization/transforms/project.py`

```text
    # what killed Phase-A attempt 9 at $0.34, on the `+=` below, and no CPU
```

### `src/aadistill/models/student.py`

```text
    dies inside `Tensor.item()`. That is exactly what happened on E8 pod A: setup
    failed after TEACHER_READY and the session self-terminated at $0.08.
```

### `src/aadistill/models/tokenizer_contract.py`

```text
    Phase-A attempt 11 lost a Stage-2 probe to this, after a 180-minute search had
```

### `src/aadistill/runtime/cost.py`

```text
    Attempt 3 makes the size of the gap concrete: 544.7 min of Stage 1, 388.2 min
    Phase-B attempt 3 actually ran — 16.9 GiB of reference against a 13.4 GiB
```

### `src/aadistill/runtime/cpu_test_env.py`

```text
    C1 attempt 5's `--strict` skip-set comparison was correct machinery pointed at
```

### `src/aadistill/runtime/device_handoff.py`

```text
    Phase-A attempt 12 died six seconds after Stage 1 succeeded:
    nothing ever acted on it. Attempt 4's handoff said, in as many words,
```

### `src/aadistill/runtime/leaf_durability.py`

```text
    Phase-A attempt 11 spent **180.3 minutes** producing five valid, measured,
    * **Headroom is measured, and refusal is the answer.** Attempt 11's five leaves
```

### `src/aadistill/runtime/pod_environment.py`

```text
    C1 attempt 3R reached `VLLM_READY → TEACHER_READY → ROPE_OK` and then died at the
    setup test gate: `14 failed, 2650 passed`, `$0.3482`, no scientific stage. Seven
    went by without anyone finding out, because no C1 attempt had ever reached
    # WHY it failed. Attempt 6 named all 18 failing nodeids and
    Two sweeps with the same digest skipped exactly the same tests. Attempt 5
    Phase-A cases did not skip under SESSION_KIND=c1" is not something a generic
    `session_prechecks.py` is a member of Phase B's and continuation B's frozen
    # Attempt 4's sweep used simulate_pod_env.sh's GENERIC default HIDDEN_PATHS,
```

### `src/aadistill/runtime/staging_contract.py`

```text
    C1 attempt 4 died at the pod CPU test gate for `$0.6986` with six failures, and
    # C1 attempt 5 died at the pod test gate for `$0.3150` because two tests about
```

### `src/aadistill/runtime/telemetry.py`

```text
    Phase-B attempt 3 spent 544.7 min in Stage 1 and stopped on its deadline. What
```

### `src/aadistill/training/train.py`

```text
    config hash. `kind` in particular is not decoration: the E7 comparison is
    E7 preregisters one `lambda_extra` and runs no sweep. That is only safe if
    # logged config computes, which is exactly what P4 forbids. Opting in per
```
