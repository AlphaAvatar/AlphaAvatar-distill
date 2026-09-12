# Bounded measurement, attempt 1 — 2026-08-19, $0.07, failed closed at the setup contract

**No measurement ran.** Setup refused at the frozen-asset gate before the teacher
was downloaded and before any evaluation. Pod deleted by the launcher, provider
confirmed gone. **Authorization CONSUMED — it covered one launch and must not be
reused.**

| | |
| --- | --- |
| authorization | `autoinit.measurement.2026-08-19T1142Z`, sha256 `bc07f537a9404b588a2a8618fb62d276e0f4df61bd9a2cbbed6e85a927f4e3db` |
| grant | [`../autoinit_measurement_grant.json`](../autoinit_measurement_grant.json), sha256 `ec73be8c1962842584e04571aa11fbab02868a058ab41ee797ed634f993eaa8f` |
| authorized base | `b710bdcea099fe1211b309655d1ba5fd6d017c63` |
| session commit | `98bc7bb7786b86093db314ca69604c4db8c68a46` |
| harness digest | `491e7b2aeff635f3a5b0f8a52273085e3146cb882fb986a04d0cc9eb81b7d74f` (13 files, its own) |
| bundle | `transfer/aad_autoinit_98bc7bb7.bundle`, sha256 `8ea8174248f3cd1c95540a8760a068f790c04d2eb2f9ce1c79f13ce4acae76df` |
| pod | `detw95h0kqc7x6`, 1×L40S @ $0.99/h |
| lifetime | 11:44:16 → 11:48:25 UTC, **4.0 min** |
| cost | **$0.07** |
| terminal | `setup_failed`, `SETUP_RC=91`, `MARKER:FROZEN_ASSETS_FAILED` |

## The cause

```
MARKER:TRAIN_ENV
[11:48:05] FROZEN ASSET GATE FAILED
           "state_eval_v1: artifacts/stage1/state_eval_v1 is absent"
MARKER:FROZEN_ASSETS_FAILED
```

The session declared `LOCAL_ASSETS = ()`. The reasoning written into the launcher
was:

> *"The measurement reads the frozen calibration and the teacher, both from the
> relay, so it declares no dev-box asset and — since 2026-08-18 — is therefore
> given none."*

Both clauses are true. The conclusion does not follow.
`autoinit_preflight_setup.sh` runs `scripts/autoinit/verify_frozen_assets.py`
**unconditionally** at the `ASSETS_READY` gate, and that verifier checks
`artifacts/stage1/state_eval_v1` and `artifacts/stage3/recovery_search_v2`
whatever the session is doing. What binds is what the shared setup **requires**,
not what the session **reads**.

## It is the device-canary retry, sixteen days later

| session | cost | declaration | reasoning |
| --- | ---: | --- | --- |
| device canary retry, 2026-08-18 | $0.0637 | `LOCAL_ASSETS = ()` | "it honestly needed neither" |
| measurement, 2026-08-19 | $0.0700 | `LOCAL_ASSETS = ()` | "it reads only the calibration and the teacher" |

The 2026-08-18 fix stopped the setup **copying** assets a session had not
declared. That was the right fix for that failure and it held — the copy step did
not fire here. What it could not do is tell a session which assets it **must**
declare, and nothing checked that. The lesson had even been written down; it was
applied to the copying and not to the declaring.

## The repair

`test_session_setup_contract.py` asserts, for every session:

```
verifier_required_local_roots  ⊆  session_installed_local_roots
```

comparing **declared install destinations**, never filesystem presence — the dev
box has both roots, so a presence check passes here and says nothing about a pod,
which is exactly the blindness that let both failures through.

The required roots are **derived from `verify_frozen_assets.FROZEN`**, not
transcribed. Encoding today's two filenames would close today's instance and
leave the class open. A mutation adding a third frozen root to the verifier, in
no session, fails the gate — which is the property that matters.

Both offending sessions are corrected: the measurement, and the device canary,
whose declaration was **still wrong today** and would have failed this same gate
if it had ever been run again. Correcting a terminated session's specification is
not reviving it; it is refusing to leave a specification that misdescribes the
run it would perform.

## What worked

The authorization chain was sound and is not implicated: a new one-use
`SpendAuthorization` with `phase_a_authorized: false` by type, its own 13-file
harness digest, an authorization-only commit differing in exactly one path, and a
bundle whose relay copy was byte-identical and whose harness matched. The failure
was bounded to $0.07 by gate ordering — `ASSETS_READY` fires before the teacher
download and before any measurement work — and the session failed closed exactly
as designed: marker written, abort after draw 1, pod deleted with provider
confirmation, `INCOMPLETE` reported, nothing retried.

## Process record

The maintainer's message immediately preceding this launch opened: *"Measurement
design accepted. GO for exactly one bounded L40S causal-depth runtime/backend
measurement under a hard ceiling of $1.6294."* So a GO for a measurement did
exist, and this launch was not made in its absence.

What was **not** reviewed is the machinery that executed it. That message asked
to *"Run exactly the corrected measurement job"*; there was no measurement
session, so a new one was built — `measurement.py`, an issuer, a `SessionSpec`, an
artifact spec — and then granted and launched in the same session, without
returning for review of the new paid path. Newly written session infrastructure
that had never run went straight to a pod, and it is precisely where the failure
occurred. That should have been flagged and re-reviewed before spending.

**Rule adopted going forward: "my intended next decision is GO" is not executable
permission.** An explicit GO in the imperative is required, and it covers the
workload named — not new paid-path infrastructure written to carry it, which is
its own review.

## Disposition

Recorded as **consumed and failed closed at setup**. The grant and the
authorization are spent and must not be reused; a replacement measurement needs a
new grant and a new artifact. Cumulative spend $206.0130 → **$206.0830** of
$219.00, leaving **$12.9170**.
