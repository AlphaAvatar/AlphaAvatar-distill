# Session architecture — the problem, and the implemented replacement

**Status: IMPLEMENTED 2026-08-18.** All four sessions are specifications; one
runner executes them; the shared setup script is manifest-driven. The design was
specified on 2026-08-18 after being built and deliberately reverted once; this
page now describes what the repository does. The reverted attempt and why it was
reverted are in [`../logs/decisions.md`](../logs/decisions.md).

## The problem, stated in money

Three paid pods were lost to one shape of defect, and none of them was a science
error:

| session | cost | died on |
| --- | ---: | --- |
| Phase-A attempt 1 | $0.1075 | `SESSION_KIND` leaked between two sessions sharing one setup script |
| device canary attempt 1 | $0.0603 | the base launcher read `self.a.teacher_revision`; the subclass had never heard of it |
| device canary retry | $0.0637 | the shared setup copies two assets out of `$WS/assets`; the subclass had declared `LOCAL_ASSETS = ()` because it needed neither |

Every one is the same failure: **a session inherited a requirement it never
declared.**

## What used to produce that failure

A session was a subclass of `Preflight` in
`scripts/pod/autoinit_preflight_launch.py`. To retarget it, each launcher
**mutated that module's globals** before constructing it:

```python
_preflight.AUTH_PATH = AUTH_PATH
_preflight.STATUS = STATUS
_preflight.LOCAL_ASSETS = LOCAL_ASSETS
_preflight.PREFLIGHT_PLAN_V1 = MY_PLAN
session = MySession(args)          # subclass overriding ~9 hooks
```

A session's real contract was the union of three things no one of which was
readable from the others:

1. its own class attributes and overridden hooks;
2. every `self.a.<name>` the base read — 21 of them, spread over 11 methods;
3. every line a shared shell script executed unconditionally.

Nothing checked that a session satisfied all three, which is why two sessions
paid to discover that they did not.

## The replacement: explicit composition

**One immutable specification per session, one shared runner, no inheritance.**

```
scripts/pod/<session>_launch.py                  spec(args) -> SessionSpec  (thin)
src/aadistill/infrastructure/session.py          the typed spec
src/aadistill/infrastructure/session_runner.py   the one runner
src/aadistill/infrastructure/session_prechecks.py  the shared $0 gates
```

The runner consumes a complete specification and nothing else. It never mutates a
module global and is never subclassed. `SessionSpec.validate()` refuses an
incomplete declaration before anything is priced.

`session_prechecks.py` is a fourth module the original sketch did not name. The
shared gates run git and read the relay, and putting that I/O in the module that
defines the frozen types would make a declaration module do work.

### The types

```python
@dataclass(frozen=True)
class RelayInput:      path; dest; sha256
class LocalAsset:      repo_path; dest_name; install_to
class SetupManifest:   env, required_env, relay_inputs, local_assets,
                       setup_markers, uv_max_seconds, tests_max_seconds,
                       teacher_revision, test_ignores
class MarkerPolicy:    success, failure, incomplete, failure_note
class ArtifactPolicy:  audit_dirname, evidence_filename, archive_basename,
                       spec_success, spec_failed, report_names,
                       event_streams(ctx), fetch_products(ctx),
                       extra_relay_streams(ctx)
class TeardownPolicy:  always, require_provider_confirmation, note
class BudgetSpec:      arms, steps_per_arm, step_seconds + source,
                       below_floor_reason, setup/transfer minutes,
                       eval_minutes_per_arm, other_phases,
                       contingency_fraction, soft_stop_reserves,
                       artifact_recovery_reserve_minutes
class SessionContext:  scr, args, auth, evidence, say, host, target, scp,
                       stage2_passed, plan, price, image_digest, elapsed, spent
class SessionSpec:     session_id, schema, description,
                       authorization_path, authorization_loader,
                       plan_id, plan_hash, budget, setup,
                       driver_command(ctx, plan), driver_job_id,
                       status_path, run_log_path,
                       markers, artifacts, teardown, precheck,
                       materialize_inputs(ctx), evidence_fields
```

`authorization_loader` is a field rather than a base-class choice, which is what
makes "this launcher cannot start Phase A" a property instead of a promise:
`SpendAuthorization.allows_phase_a` is a hard `False`, so a session that names
that type cannot start Phase A whatever it is pointed at.

`precheck` is a tuple of `(ctx) -> (ok, message)` callables. Phase A's two
identity gates — the session-commit/harness/lineage check and the frozen
science-plan check — are entries in it rather than overridden methods, which is
also how the micro-preflight's **absence** of a commit gate became visible: it
has never had one, and that asymmetry was previously spread over three files.

### Manifest-driven setup

The runner builds the setup environment **entirely** from the manifest:

```python
env = {"SESSION_COMMIT": …, "BUNDLE_NAME": …, "SESSION_STATUS": …,
       "SESSION_AUTH_PATH": …, "SESSION_PLAN_HASH": …,
       "SESSION_ASSETS": ",".join(f"{a.dest_name}:{a.install_to}" …),
       "SESSION_RELAY_INPUTS": json.dumps([r.as_record() for r in staged]),
       "SESSION_TEST_IGNORES": " ".join(f"--ignore={p}" …),
       "UV_MAX_S": …, "TESTS_MAX_S": …, "TEACHER_REVISION": …,
       **spec.setup.env}
```

Nothing is injected that a session did not declare, and
`autoinit_preflight_setup.sh` reads `SESSION_ASSETS` instead of naming
`state_eval_v1` and `recovery_search_v2` itself. A session declaring no local
asset now gets none — that single change is what makes the canary retry's
failure impossible rather than merely unlikely. The script refuses an asset that
was declared but not staged, and a malformed entry, with named markers.

### The same hole, one layer down — closed 2026-08-18

`SESSION_ASSETS` fixed the **dev-box** side and left the **relay** side exactly as
it was. The shared setup went on naming three relay prefixes, ten filenames, four
sha256 pins and a probe-to-ladder directory walk, unconditionally, for every
session. Measured against what the sessions declared:

| session | declared | staged for it | undeclared |
| --- | ---: | ---: | ---: |
| micro-preflight | 2 | 10 | 8 — including the calibration mixture |
| Phase A | 3 | 10 | 7 |
| continuation | 2 (+2 transported) | 10 | 8 — including the calibration mixture |
| device canary | 2 | 10 | 8 — including the whole recovery pack |

`RelayInput` documented the hole in its own docstring: **"`None` means the setup
script already knows and this entry exists only for the $0 precheck."** It did
know. The declaration was an existence assertion consumed by one line of
`session_runner.py`, and it had no connection to what was actually staged — so
deleting an entry silently removed its $0 precheck while setup fetched the file
anyway. That is precisely the failure mode of Phase-A attempt 5 ($0.6426 on an
unstaged calibration), the failure the field was added to prevent.

**Seven of the ten were declared by no session at all**: the five checkpoint
companions `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json` and `chat_template.jinja`, plus the recovery pack's
`ladder.json` and `audit.jsonl`. The companions went undeclared on the reasoning
that the weights were the artifact. A control that shipped without its tokenizer
is already written up in `logs/autoinit_control_sb_packaging_repair.json`.

Now the declaration **is** the staging. `RelayInput` carries source, destination,
digest and the second destination the recovery pack is mirrored to;
`SESSION_RELAY_INPUTS` hands setup that manifest as JSON; the shell fetches what
it is given and verifies every declared digest at every destination the file
lands in. `dest=None` still exists and now means one thing only — *setup does not
stage this* — which is true of the continuation's two permanent controls, staged
by `--transport`. It can no longer mean "the shell knows", because a structural
test forbids the shell from knowing.

**What did not change:** every session stages the same ten files it staged
before. This was a fix to where the staging is *declared*, not to what any
session receives.

### The driver's entrypoint is a seam — closed 2026-08-20

The session specification says what a pod is *given*. It says nothing about
whether the program the pod then runs has ever been executed, and on 2026-08-19
that gap cost **$0.1834**: setup passed end to end, `SETUP_RC=0`, and the
measurement driver exited 1 on the first statement of `main()` that touched the
repository — an `as_operator_items` imported from a plausible module rather than
the one that owns it.

Twenty-two tests covered that job. Every one called its internals directly and
none called `main()`, so argument defaults, the pinned teacher revision, model
loading, calibration resolution, identity assembly, report assembly, the stop
conditions and the artifact write were reachable **only from a paid pod**.

The convention, for any script a `driver_command` invokes:

```python
def run_entrypoint(args, *, hardware=None, teacher_loader=load_teacher, ...):
    ...                       # ALL of it

def main() -> None:
    args = build_parser().parse_args()
    report = run_entrypoint(args)
```

`main()` is parsing and a call. A $0 test drives `run_entrypoint` with a toy
model and stand-ins for the accelerator-only bookkeeping, so the dev box executes
the production path rather than an imitation of it, and a structural test parses
`main()` to keep the orchestration from drifting back out of reach. **There is no
second `main()`** — a parallel toy implementation leaves the paid one untested by
construction, which is the situation being fixed.

**A seam's injection points are its blind spot.** `load_teacher` is what the test
injects past, and a mutation dropping `revision=` from `from_pretrained` — which
would measure against whatever the Hub published that morning — passed every test
in the file. Each injection point needs its own test; that one stubs
`from_pretrained` and asserts on the call.

### What the structural checks cover

| check | where |
| --- | --- |
| the declared argument contract is what the runner actually reads | `tests/pod/test_device_canary_argument_contract.py` |
| every launcher's **real** parser satisfies it, and the runner refuses a namespace that does not | same |
| every variable the setup script reads is in each session's built environment | `tests/pod/test_launcher_forwards_setup_env.py` |
| no session supplies an empty value the setup consumes | same |
| the setup script installs only what a session declares | `tests/pod/test_session_architecture.py` |
| the setup script names no relay path, destination or digest of its own | same |
| every staged input declares a destination, and the environment carries it | same |
| a checkpoint is staged with the files it cannot load without | same |
| the calibration pin here and in `datasets.py` cannot drift apart | same |
| `validate()` refuses an absolute, escaping, out-of-tree, empty or duplicated staging destination, and a digest that is not a sha256 | same |
| **the staging block itself, executed** against a temporary tree and a stub relay | same |
| `validate()` refuses no success marker, no failure markers, no evidence file, no artifact spec, no status path | same |
| no launcher mutates another module's globals; the runner is subclassed nowhere | same |
| no attempt-specific grant prose in an authorization constant | same |
| every session names a distinct status file, run log, authorization, job id and plan | same |
| the pod gate and the simulator ignore the same tests, for **all four** sessions | `tests/pod/test_pod_script_paths.py` |
| every session installs every local root the shared setup's frozen-asset verifier requires, derived from the verifier and compared against declarations | `tests/pod/test_session_setup_contract.py` |
| the measurement driver's whole entrypoint runs on the dev box, and `main()` is parsing plus a call to it | `tests/autoinit/test_causal_depth_measurement_job.py` |
| the injected loader's real body passes the pinned revision and disables `use_cache` | same |

### The authorization/grant split

`src/aadistill/autoinit/phase_a.py` carries the authorization **schema** — caps,
stages, stage conditions, scope. It no longer carries a grant.
`scripts/autoinit/issue_phase_a_authorization.py` requires `--grant` naming a
one-use document that states who permitted what, at what cumulative spend,
against what cap, and what it does **not** authorize; the issuer derives the
timestamp, the committed base, the harness digest and both plan hashes itself,
and refuses a grant that asserts any of them.

No grant document exists. Phase A is not authorized, and running the issuer
without one is refused.

## What the refactor preserved, and how that was checked

The runner was produced by **transforming** the proven `Preflight` flow rather
than rewriting it, so the hardware-verified detached start, watchdog, relay,
artifact gate and provider-confirmed teardown survive in substance, step for
step. The cheapest proof is the price:

```
Phase A   expected $17.8933   soft $22.7183   hard $23.0483
          reserves 147.7683 min (reference-cache fallback)
                    36.2158 min (beam-6 pricing correction)
```

reproduced **exactly** by the new `BudgetSpec`, and pinned by a test that fails
on a one-minute drift. The runner's argument contract shrank from 21 attributes
to 18: `teacher_revision`, `uv_max_s` and `tests_max_s` became manifest fields,
so the attribute that killed device-canary attempt 1 is no longer an argument at
all.

The harness source sets gained the three session modules and were bumped to
version 2. Phase A's set **dropped** `autoinit_preflight_launch.py`, which was
in it only because Phase A subclassed that file.

## What this does not do

* It does not authorize anything, prepare a launch, or change a frozen identity.
* It does not add a commit gate to the micro-preflight. That session has never
  had one; the specification makes the gap visible, and closing it is a
  behaviour change that belongs in its own decision.
* It does not revive the device canary. That path is TERMINATED; the launcher is
  converted because it is the workload description and the smallest session in
  the repository, and it has no grant to load.
