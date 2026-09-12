# Phase-A attempt 8 — 2026-08-18, $0.19, failed closed at the setup test gate

**No stage ran. Nothing was trained, measured, or written to the relay. The pod
was deleted by the launcher and the provider confirmed it gone.**

| | |
| --- | --- |
| authorization | `autoinit.phase_a.2026-08-18T1244Z`, sha256 `25c85fbaeb3ae57c6ad6c401a2e11736f627975fc6ee029ade1b5d113f5fa173` |
| grant | [`../autoinit_phase_a_attempt8_grant.json`](../autoinit_phase_a_attempt8_grant.json), sha256 `09541ef547c6a3c9b9210b136171eba79f4c43330af2b123ee1f956206f1fa43` |
| authorized base | `0b03d366c252ef0f47ef1af5451463d218c98b32` |
| session commit | `4563d10644591f8030d0a43ddcd4dbce95e12031` |
| harness digest | `24d89b9ffe8e41f2712d2f105ca2076fd0552ccea7a089aeb5f2a844c39e1f42` (16 files) |
| bundle | `transfer/aad_autoinit_4563d106.bundle`, sha256 `509b29abb227e15ac3d86af75b08c4cd2f4e9add37d7d2530b97d2facadaa19b` |
| pod | `2maapdxqg566r5`, 1×L40S @ $0.99/h, stock Low, image `…cu1300-torch291-ubuntu2404@580.178.04` |
| lifetime | 12:47:40 → 12:58:57 UTC, 11.28 min |
| cost | **$0.19** (watchdog accrued $0.1833 at its last tick; the launcher's rounded figure is carried) |
| terminal | `setup_failed` on draw 1 — no driver started, no `terminal` marker exists |

## What passed, and it matters

Setup reached **`ROPE_OK`**, which is eight markers in:

```
ENV_READY → REPO_READY → ASSETS_STAGED → TRAIN_ENV → ASSETS_READY
          → VLLM_READY → TEACHER_READY → ROPE_OK → [TESTS_OK] ✗
```

So on real hardware, for the first time:

* the **manifest-driven relay staging worked** — all 10 declared science inputs
  fetched from `SESSION_RELAY_INPUTS`, every declared digest verified at every
  destination, `ASSETS_STAGED` reached;
* `ASSETS_READY` — the frozen-asset gate passed against the preregistered
  constants, run pod-side under `/opt/train`;
* `ROPE_OK` — the staged checkpoint loaded via `AutoConfig.from_pretrained` in
  **both** venvs (transformers 5.13.1 and 5.15.0), each reading stored RoPE base
  5,000,000. That is direct evidence the five companion files staged correctly:
  a checkpoint missing `config.json` cannot be read at all.

The $0 precheck also reported **10 relay inputs** where attempt 7 reported 3.

## What failed

The blocking CPU test gate, on **two tests, neither of which can pass in a
container**: `2 failed, 1789 passed, 63 skipped in 277.29s`.

### 1. `test_every_path_named_in_the_repo_layout_exists`

`docs/REPO_LAYOUT.md` names two **absolute dev-box paths** —
`/home/ecs-user/aad-artifacts/` and `/home/ecs-user/aad-scratch/` — and the test
checks `(REPO / ref).exists()`. `pathlib` discards the base when the right
operand is absolute, so the assertion reads the literal host filesystem. A pod
has neither path. Introduced by `69b2e74`, the inventory commit that made the
out-of-tree store visible in the layout.

### 2. `test_no_tombstoned_path_is_still_on_disk`

The tombstone `stage3_ladder_uniform_local_cache` records
`artifacts/stage3/ladder_uniform` as a deleted verified-stale cache. **The pod's
setup stages the recovery pack into exactly that directory** — it always has; it
is the mirror the recovery-corpus loader reads, beside the
`ladder_uniform_probe` copy `p2_driver.py` reads. So on a pod the path exists
and the assertion fires. Introduced by `dded03e`.

This is the *same shape* as the `podsim_quarantine_residue` tombstone that
`dded03e` itself withdrew: a tombstone asserting that something stopped
existing, pointed at a destination a routine command recreates by design.
Whether this tombstone should likewise be withdrawn is a maintainer decision and
is **not** taken here.

## Why no $0 path caught it

The pod simulator hides gitignored artifacts **inside the repo** and runs on the
dev box. It cannot reproduce either condition:

* it cannot make `/home/ecs-user/aad-artifacts` stop existing — that path is
  outside the repository, and hiding it would break the dev box;
* for the tombstone test it produced the **opposite** of the pod's state. On the
  dev box the simulator *hides* `artifacts/stage3/ladder_uniform` because it is
  gitignored, so the assertion passes — for the wrong reason. On a pod, setup
  *creates* it, so the assertion fails.

Both failing tests were added by the 2026-08-18 inventory and cleanup work.
Attempt 7 ran on 2026-08-17, before both commits, which is why attempt 8 is the
first paid session to execute them. Neither failure comes from the relay-staging
fix; that fix is the part of this run that worked.

## Disposition

Failed closed as instructed: evidence recorded, pod terminated with provider
confirmation, **stopped for review**. No repair was attempted on the live pod,
no retry was launched, and no subsequent attempt is authorized. The grant covers
one launch and is spent.

Raw evidence here: [`launcher.out`](launcher.out), [`session.json`](session.json),
[`watchdog.jsonl`](watchdog.jsonl), [`poll.log`](poll.log).
