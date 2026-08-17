# Session architecture — the problem, and the specified replacement

**Status: SPECIFIED, NOT IMPLEMENTED.** The live launchers still use the
inheritance design described in "What exists today". This document is the
replacement, written after building it far enough to validate the design and
then deliberately reverting it — see "What was built, and why it was not landed"
at the end. It is the starting point for that work, not a description of it.

## The problem, stated in money

Three paid pods have been lost to one shape of defect, and none of them was a
science error:

| session | cost | died on |
| --- | ---: | --- |
| Phase-A attempt 1 | $0.1075 | `SESSION_KIND` leaked between two sessions sharing one setup script |
| device canary attempt 1 | $0.0603 | the base launcher read `self.a.teacher_revision`; the subclass had never heard of it |
| device canary retry | $0.0637 | the shared setup copies two assets out of `$WS/assets`; the subclass had declared `LOCAL_ASSETS = ()` because it needed neither |

Every one is the same failure: **a session inherited a requirement it never
declared.**

## What exists today, and why it produces that failure

A session is a subclass of `Preflight` in
`scripts/pod/autoinit_preflight_launch.py`. To retarget it, each launcher
**mutates that module's globals** before constructing it:

```python
_preflight.AUTH_PATH = AUTH_PATH
_preflight.STATUS = STATUS
_preflight.LOCAL_ASSETS = LOCAL_ASSETS
_preflight.PREFLIGHT_PLAN_V1 = MY_PLAN
session = MySession(args)          # subclass overriding ~9 hooks
```

A session's real contract is therefore the union of three things no one of which
is readable from the others:

1. its own class attributes and overridden hooks;
2. every `self.a.<name>` the base reads — 21 of them, spread over 11 methods;
3. every line a shared shell script executes unconditionally.

Nothing checks that a session satisfies all three, which is why two sessions
paid to discover that they did not.

## The replacement: explicit composition

**One immutable specification per session, one shared runner, no inheritance.**

```
scripts/pod/<session>_launch.py     spec(args) -> SessionSpec        (thin)
src/aadistill/infrastructure/session.py        the typed spec        (new)
src/aadistill/infrastructure/session_runner.py the one runner        (new)
```

The runner consumes a complete specification and nothing else. It never mutates
a module global and is never subclassed. `SessionSpec.validate()` refuses an
incomplete declaration before anything is priced.

### The types

```python
@dataclass(frozen=True)
class RelayInput:      path: str; dest: str; sha256: str | None
class LocalAsset:      repo_path: str; dest_name: str; install_to: str
class SetupManifest:   env, required_env, relay_inputs, local_assets,
                       setup_markers, uv_max_seconds, tests_max_seconds,
                       test_ignores
class MarkerPolicy:    success, failure, incomplete, failure_note
class ArtifactPolicy:  audit_dirname, evidence_filename, archive_basename,
                       spec_success, spec_failed, report_names,
                       event_streams(ctx), fetch_products(ctx)
class TeardownPolicy:  always, require_provider_confirmation, note
class BudgetSpec:      price_per_hour, setup/transfer minutes, arms,
                       steps_per_arm, step_seconds + source, below_floor_reason,
                       eval_minutes_per_arm, other_phases, contingency_fraction,
                       soft_stop_reserves, artifact_recovery_reserve_minutes
class SessionContext:  scr, host, target, evidence, stage2_passed,
                       auth, args, say          # what a spec callable may see
class SessionSpec:     session_id, schema, description,
                       authorization_path, authorization_loader,
                       plan_id, plan_hash, budget, setup,
                       driver_command(ctx, plan), driver_job_id,
                       status_path, run_log_path,
                       markers, artifacts, teardown, precheck, evidence_fields
```

`authorization_loader` is a field rather than a base-class choice, which is what
makes "this launcher cannot start Phase A" a property instead of a promise:
`SpendAuthorization.allows_phase_a` is a hard `False`, so a session that names
that type cannot start Phase A whatever it is pointed at.

`precheck` is a tuple of `(ctx) -> (ok, message)` callables. Phase A's two
identity gates — the session-commit/harness/lineage check and the frozen
science-plan check — become entries in it rather than overridden methods.

### Manifest-driven setup

The runner builds the setup environment **entirely** from the manifest:

```python
env = {"SESSION_COMMIT": …, "BUNDLE_NAME": …, "SESSION_STATUS": …,
       "SESSION_AUTH_PATH": …, "SESSION_PLAN_HASH": …,
       "SESSION_ASSETS": ",".join(f"{a.dest_name}:{a.install_to}" …),
       "SESSION_TEST_IGNORES": " ".join(f"--ignore={p}" …),
       "UV_MAX_S": …, "TESTS_MAX_S": …,
       **spec.setup.env}
```

Nothing is injected that a session did not declare, and
`autoinit_preflight_setup.sh` reads `SESSION_ASSETS` instead of naming
`state_eval_v1` and `recovery_search_v2` itself. That single change is what makes
the canary retry's failure impossible rather than merely unlikely.

### Structural checks the replacement must ship with

* every attribute the runner reads off the args namespace is present in each
  launcher's **real** parser (this exists today for the canary only:
  `tests/pod/test_device_canary_argument_contract.py`);
* every variable the setup script reads is declared in each session's manifest;
* every asset the setup script installs is declared as a `LocalAsset`;
* `SessionSpec.validate()` refuses a spec with no success marker, no failure
  markers, no evidence file, no artifact spec, or no status path;
* no attempt-specific grant prose in executable source.

## What was built, and why it was not landed

On 2026-08-18 the design above was implemented far enough to prove it, then
reverted. What it demonstrated:

* `session.py` (332 lines) and `session_runner.py` (696 lines) — the runner was
  produced by **transforming** the proven `Preflight` flow rather than rewriting
  it, so the hardware-verified detached start, watchdog, relay, artifact gate and
  teardown survive unchanged in substance.
* The micro-preflight and Phase-A specs both `validate()`.
* **Phase-A pricing reproduced exactly**: `$17.8933 / $22.7183 / $23.0483`, with
  both soft-stop reserves carrying their derived minutes
  (147.7683 and 36.2158) — the refactor is behaviour-preserving where it
  matters most.
* The runner's argument contract shrank from 21 attributes to 18 operational
  knobs, because `teacher_revision`, `uv_max_s` and `tests_max_s` became manifest
  fields. The attribute that killed canary attempt 1 stops being an argument at
  all.

**Why it was reverted:** converting the two remaining launchers, making the
setup script manifest-driven, splitting the authorization schema from grant
provenance, and rebinding **30 failing tests** was more than could be completed
and verified in one session. Those 30 tests each encode a paid failure class, and
rebinding them carelessly would silently stop them testing what they were written
for — the one outcome worse than not refactoring. A half-landed refactor cannot
reach a clean, fully verified tree, so the launchers were restored to the proven
implementation and the design was written down here instead.

### The order to do it in

1. `session.py`, then `session_runner.py` by transforming `Preflight` (do not
   rewrite it).
2. Micro-preflight and Phase A as spec providers. **Check the Phase-A price
   reproduces `$17.8933 / $22.7183 / $23.0483` exactly before going further** —
   it is the cheapest possible proof that the transformation preserved
   behaviour.
3. Continuation and the terminated canary.
4. `autoinit_preflight_setup.sh` → `SESSION_ASSETS` / `SESSION_TEST_IGNORES`.
5. Rebind the 30 tests, one at a time, preserving each one's stated intent.
   `tests/pod/test_phase_a_rehearsal.py` has 11 of them.
6. Split `PhaseAAuthorization`'s schema and frozen plan from grant provenance;
   the issuer takes an explicit one-use grant input and derives timestamp,
   committed base, harness digest and plan identities.
7. The structural checks listed above.

Doing 1–2 and stopping is a valid increment. Doing 1–5 and stopping is not: the
tests and the launchers are coupled, so the tree is red in between.
