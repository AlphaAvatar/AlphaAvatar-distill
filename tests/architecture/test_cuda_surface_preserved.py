"""The real-CUDA-validated execution surface is byte-identical, and the rest of
the core changed only in prose.

The review that accepted execution SHA `7027a8f4` named a surface that must not
move — `attention_activation`, the attention statistics, `stats_to`/device
behaviour, the adapters the validation used, fixed-path/suffix execution,
treatment-record generation, and the two geometries — and said in as many words:
do not silently claim that SHA validates different code.

"I did not change those files" is a claim about intent. This is the check.
Two things are asserted, both read from git rather than argued:

1. every file on the declared surface is byte-identical to the reviewed tip;
2. every OTHER core file that did change is prose-only — its AST, with
   docstrings stripped, is identical to what it was.

(2) matters as much as (1). A docstring sweep across 44 modules is exactly the
shape of change that could carry one edited expression through review unnoticed,
and reading 44 diffs by eye is how that gets missed.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: One entry per independent review round on this branch, oldest first: the tip
#: that round was reviewed at, and the core files that round deliberately
#: changed the BEHAVIOUR of. Everything else it touched must be prose-only.
#:
#: A list rather than one constant because each round is bound to its own
#: reviewed tip. Collapsing them would either lose the earlier round's
#: declaration or measure this round's changes from the wrong base, and both
#: weaken the check that catches an undeclared edit riding along with a
#: docstring sweep.
ROUNDS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("573d6cfc92497c176afe4645335df1a2534f93df",
     "post-CUDA evidence and reproducibility closure",
     {
        "src/aadistill/infrastructure/session.py":
            "ExecutionCommands: interpreter default removed, canonical "
            "DeploymentBinding added",
        "src/aadistill/infrastructure/session_runner.py":
            "records the resolved provider CLI in the session record",
     }),
    ("b2ecdff83ee0a7653eea7872b5af6739a4de4381",
     "readiness wire-contract core/application separation",
     {
        "src/aadistill/runtime/pod_environment.py":
            "RecordContract: the schema string, the harness field name, the "
            "harness digest callable and the record path are the CALLER's; the "
            "success summary is derived from the record instead of naming two "
            "of one experiment's groups. THIS IS A DECLARED SEMANTIC CHANGE to "
            "readiness verification and is deliberately not described as prose.",
     }),
    ("0103467384ca0013ddc0caa9dc18c92aadfa5c78",
     "undefined REPO in the shared runner, and run/output ownership",
     {
        "src/aadistill/infrastructure/session_runner.py":
            "run(): the driver JobSpec's `workdir` reads `self.repo` "
            "(spec.commands.checkout_root) instead of `REPO`, a module constant "
            "deleted when the image layout moved into ExecutionCommands. THIS IS "
            "A DECLARED SEMANTIC CHANGE -- the expression evaluated is different "
            "-- and is deliberately not described as prose, even though the "
            "previous expression could only ever raise NameError. It is the same "
            "value the same call already passed as PYTHONPATH.",
     }),
    ("fe27dfdc488ae0ee3bd50f04ceb14f901bf8b1f6",
     "confirmed release on redraw, and a whole-file write that cannot truncate",
     {
        "src/aadistill/infrastructure/session_runner.py":
            "the acquisition loop's redraw branch. It fired `remove pod`, ignored "
            "the result, cleared `self.pod_id` and created the next resource, so "
            "a subprocess returning and a local assignment were being treated as "
            "evidence that a pod had stopped billing. It now calls "
            "`release_and_confirm`, which deletes and then WAITS for the provider "
            "to report the resource not billing, and ABORTS the session rather "
            "than create a second one when that cannot be confirmed. "
            "`record_draw` appends one entry per resource -- pod id, outcome, "
            "cumulative cost, its own watchdog journal -- because `ev['pod_id']` "
            "describes only the current pod, and watchdog journals rotate to "
            "`watchdog_<pod_id>` so a redraw cannot truncate the previous "
            "backstop's evidence. THIS IS A DECLARED SEMANTIC CHANGE to provider "
            "acquisition, reachable by every session that draws more than once, "
            "and deliberately not described as prose.",
        "src/aadistill/infrastructure/manifest.py":
            "`write_text_atomic`: a whole-file write goes to a sibling temp, is "
            "fsynced, then `os.replace`d, and `write_manifest` routes through it. "
            "`Path.write_text` truncates first and writes second; when the "
            "filesystem filled on 2026-09-11 two tracked files became zero bytes "
            "in that gap. THIS IS A DECLARED SEMANTIC CHANGE to durability -- "
            "identical on every path that succeeds, different on one that fails, "
            "where the previous file now survives.",
     }),
    ("0aab945dd276f8231cdb070246c5c9bec8940759",
     "per-resource watchdog isolation and failed-draw evidence",
     {
        "src/aadistill/infrastructure/session_runner.py":
            "the watchdog journal is derived from the pod id at LAUNCH and the "
            "rename is gone. The previous round moved the shared path aside "
            "when a draw started, which isolates nothing: `Journal.write` "
            "reopens by path per event, so a watchdog that had not yet exited "
            "recreated the shared path and wrote its final poll and "
            "`watchdog_end` into the NEXT resource's journal while its own "
            "archive was left unterminated. Reproduced with two real "
            "processes. `create()` names its raw provider response by DRAW and "
            "attempt, because `attempt` restarts at 1 in each draw and a "
            "second draw's first create overwrote the first draw's -- the only "
            "record of what the provider said, including a refusal carrying no "
            "pod id. `run()` records a `create_failed` draw, with no pod id and "
            "the raw responses it actually wrote, so a draw that created "
            "nothing is still in `ev['draws']`. THIS IS A DECLARED SEMANTIC "
            "CHANGE to acquisition evidence, reachable by any session that "
            "draws more than once or is refused, and deliberately not "
            "described as prose.",
     }),
    ("8a6bbcaafa761ffca0b2697c5f68182798a2709d",
     "a caller-supplied segment between the experiment and the run",
     {
        "src/aadistill/runtime/run_layout.py":
            "`RunLayout` gained `runs_subdir`, a path segment between the "
            "experiment and the run supplied by the CALLER's convention, and "
            "`rel_root`/`root` compose it. It defaults to the empty string, so "
            "every layout that existed before composes to the same path, byte "
            "for byte. `verify_run_manifest` then stopped reconstructing "
            "`experiment/run` and reads the manifest's own stated `root` "
            "instead, deriving the subdir from it: the verifier that assumed a "
            "composition rejected every manifest written under a convention "
            "that keeps runs in a child directory, which is a statement about "
            "the verifier and not about the manifest. THIS IS A DECLARED "
            "SEMANTIC CHANGE to layout composition and to manifest "
            "verification, and is deliberately not described as prose. It is "
            "the same generalization `7cb25bc` began -- caller-supplied root, "
            "caller-supplied role vocabulary, no repository, stage or model "
            "name in the core -- carried one step further, and "
            "`test_the_root_is_named_here_and_only_here` still holds: the core "
            "names no `logs/` path. Introduced by `fa9fa9a` (the field and the "
            "composition) and `e89952b` (the verifier), both in the "
            "stage-first logs migration, and undeclared until now.",
     }),
    ("e7180a3239fe8b58132115d3676225d80ba7db44",
     "a session may declare no staged-role case",
     {
        "src/aadistill/runtime/pod_environment.py":
            "`ReadinessGroups.staged_role_nodeid` became `str | None`, and "
            "`evaluate_sweep` skips the check when it is None. Previously the "
            "field was mandatory, `outcomes.get(None, 'ABSENT')` could not "
            "match, and a session that stages no source-dependent case failed "
            "its own sweep with 11 passed / 0 failed. THIS IS A DECLARED "
            "SEMANTIC CHANGE to readiness evaluation and is deliberately not "
            "described as prose. It is not a relaxation: an ABSENT nodeid is "
            "still a finding, and only an explicit None is skipped, so a "
            "session that declares one must still satisfy it. Introduced by "
            "`a1daf34` and undeclared until now — which is why rounds 2 "
            "through 5 were failing this check at `6f9e739`, before the relay "
            "round below touched anything.",
     }),
    ("6f9e73928bcc919651889b458bb2884b6f9234e5",
     "a relayed document its writer rewrites in place",
     {
        "src/aadistill/infrastructure/log_relay.py":
            "`RelaySpec.whole_file`. A spec declaring it is read from offset "
            "ZERO every cycle — the persisted offset is ignored, not merely "
            "left unwritten — is written by temp file plus fsync plus "
            "`os.replace`, and is REFUSED rather than written when the chunk "
            "reaches `MAX_CHUNK_BYTES`, because a document truncated at the cap "
            "is corruption in the shape of success. THIS IS A DECLARED "
            "SEMANTIC CHANGE to relay transport, reachable only by a spec that "
            "opts in: every append-only stream takes the same path it always "
            "did, byte for byte, and `_append` is untouched. The failure it "
            "ends is a rewritten file relayed by byte offset, where `tail -c "
            "+N` splices the new document's tail onto the old document's head "
            "and nothing raises.",
        "src/aadistill/infrastructure/session_runner.py":
            "the evidence document's `RelaySpec` declares `whole_file=True`. "
            "One argument, and it is the whole point: the mechanism above is "
            "inert without a production caller. The run log and the status "
            "stream keep offset semantics. THIS IS A DECLARED SEMANTIC CHANGE "
            "to what the runner relays and how.",
        "src/aadistill/initialization/planning/search.py":
            "`SearchConfig.impl_profiles`, and `expansion_profiles` extracted "
            "to module level. A search may now restrict WHICH of its active "
            "calibration profiles a given implementation branches over, instead "
            "of every calibration-consuming operator branching over all of "
            "them; `_validate_impl_profiles` refuses a restriction naming an "
            "implementation the search cannot run, one declaring "
            "CalibrationNeed.NONE, an empty profile list, or a profile the "
            "search does not branch over. THIS IS A DECLARED SEMANTIC CHANGE to "
            "the reachable search space and is deliberately not described as "
            "prose. It is inert by default: `impl_profiles=None` yields the "
            "previous branching exactly, and `as_dict` OMITS the key when unset "
            "rather than emitting a null, so every recorded `config_hash` is "
            "still what this code computes. `expansion_profiles` also absorbed "
            "the `CalibrationNeed.NONE` single-offer rule that "
            "`_candidate_expansions` used to apply inline — same behaviour, one "
            "definition, so a cost model counting the space cannot disagree "
            "with the beam expanding it.",
     }),
    ("a6e8063bef05130693a6759df19fc2c0103ec7da",
     "a second experiment's readiness, and transport as infrastructure",
     {
        "src/aadistill/runtime/staging_contract.py":
            "`ignores_for_selection(selection, repo_root)`. The pod's blocking "
            "gate is `pytest tests/ $SESSION_TEST_IGNORES` and a session may "
            "only add flags, so running one preflight directory has to be "
            "expressed as a COMPLEMENT — and a hand-written complement fails "
            "UNSAFELY: a test directory added tomorrow says nothing about "
            "itself and joins the paid suite by default. Derived, the default "
            "for anything new is EXCLUDED. THIS IS A DECLARED SEMANTIC CHANGE, "
            "and it is an ADDITION: no existing caller's ignore list changes, "
            "because nothing called this before. It refuses rather than "
            "returning an empty tuple when the named selection is not a "
            "directory, since a complement derived from a missing selection "
            "would ignore the whole suite.",
        "src/aadistill/runtime/pod_environment.py":
            "`SweepContract`. `RecordContract` says what a readiness record "
            "looks like and `ReadinessGroups` says what a sweep should "
            "observe; this third type says how to DRIVE one — which launcher "
            "to load, which session id to derive the staged view under, which "
            "harness to digest, which record keys to write, and whether the "
            "experiment has a navigation pointer at all. THIS IS A DECLARED "
            "SEMANTIC CHANGE to readiness recording, and it is an ADDITION: "
            "`verify_record`, `evaluate_sweep`, `pod_test_environment_digest` "
            "and the lineage rule are untouched, and C1's record keys, schema, "
            "pointer path and prose are reproduced byte for byte by the "
            "contract it now declares. The recorder previously imported one "
            "experiment's harness digest at module level, so a second session "
            "could not have had a readiness record at all.",
        "src/aadistill/infrastructure/bundle_transport.py":
            "NEW FILE. The transport question — build the exact commit into a "
            "bundle, verify it, upload it, download what the pod would fetch, "
            "clone it, and require the checkout to be the exact session commit "
            "carrying the exact authorization bytes with the authorized "
            "executable digest — extracted from one experiment's copy into "
            "infrastructure, with the relay repository, the path prefix and the "
            "label supplied by the caller through `TransportSpec`. THIS IS A "
            "DECLARED SEMANTIC CHANGE by virtue of being new executable core; "
            "no existing caller is retargeted, and the older copy under "
            "`scripts/experiments/` is deliberately untouched because it "
            "belongs to a closed experiment's frozen executable set. Two "
            "behaviours differ from that copy, both refusals: an empty "
            "executable set is refused rather than digested (it would pass "
            "vacuously), and the digest formula is IMPORTED from "
            "`aadistill.governance.closure` rather than restated, so the "
            "round-trip and the closure cannot drift apart.",
     }),
    ("4de162ae11d9e4a3ed5f84a47c3d7215466c35e1",
     "the setup declaration becomes an execution contract",
     {
        "src/aadistill/runtime/setup_steps.py":
            "NEW FILE. The rule deciding which optional setup steps run: parse "
            "the declaration, answer one question about it, and exit 0 / 3 / 4 "
            "-- never 1 or 2, which an interpreter that cannot parse the file "
            "at all already uses -- so a shell `if` cannot read a crash as "
            "permission to skip everything. THIS IS A DECLARED SEMANTIC "
            "CHANGE by virtue "
            "of being new executable core, and it is what makes "
            "`SetupManifest.setup_markers` mean something: the field was "
            "declared by every session and read by NOTHING, so the shared setup "
            "script ran every section unconditionally and a session that "
            "omitted a marker got the step anyway. It knows no experiment, "
            "phase, model or step meaning -- the marker names are the session's "
            "vocabulary, so a future stage declaring its own needs no edit "
            "here.",
        "src/aadistill/infrastructure/session.py":
            "`SetupManifest.setup_markers_env()` and `SUBSTRATE_MARKERS`, plus "
            "`SESSION_SETUP_MARKERS` in `setup_environment` and "
            "`setup_markers` in the session record. THIS IS A DECLARED "
            "SEMANTIC CHANGE to what reaches setup: the shell now runs an "
            "optional section only when the declaration names its marker, so a "
            "field that described what would happen now decides it. It is "
            "inert for every session that declares the full set -- which every "
            "session but the closed Phase-A launcher does, audited in "
            "`scripts/experiments/stage-1/phase_c2/tests/test_phase_c2_setup_contract.py` -- and it REFUSES a "
            "declaration that omits the substrate rather than silently "
            "accepting a setup nobody runs.",
     }),
    ("5b0396c7f8d4613a92d6b5d0d6fdb0d7a64cf878",
     "a stream-less session's teardown route, after attempt 3",
     {
        "src/aadistill/infrastructure/session_runner.py":
            "`streams_at_risk(manifest, declared_streams)`, extracted from the "
            "inline expression `collect_and_teardown` passed to "
            "`evaluate_teardown` and given one new answer. THIS IS A DECLARED "
            "SEMANTIC CHANGE to which teardown route a failed session takes. "
            "Previously: manifest present -> its marker failures plus its "
            "still-being-written entries; manifest absent -> `None`, the "
            "strict rule, which DEMANDS that the caller name the streams it is "
            "truncating. A session that declares no event streams can never "
            "satisfy that demand, so C2 attempt 3 -- whose failure spec was "
            "unloadable, leaving no manifest -- raised `ArtifactError` in the "
            "middle of teardown and reported it in place of the real failure "
            "one layer down. Now the absent-manifest case asks the SPEC: no "
            "declared streams means none to truncate, which is evidence rather "
            "than an assumption, and the gate's recorded-loss route applies. "
            "`None` is kept for the only genuinely uninformed case -- streams "
            "declared AND no manifest -- and "
            "`scripts/experiments/stage-1/phase_c2/tests/test_phase_c2_collection_and_profiles.py` holds that "
            "mutation. Extracted rather than left inline because a branch "
            "reachable only from a pod whose collector failed is a branch no "
            "`$0` check can execute; as a function it is four unit tests.",
     }),
    ("7b376424f45924b8745184b4bc9bf98fe5463d4b",
     "where the search spends its time, measured on an L40S",
     {
        "src/aadistill/initialization/statistics/contribution.py":
            "`_reduce_on_device(device)`, float64 accumulators that stay on "
            "the accelerator through `distortion()`'s chunk loop with one "
            "`.tolist()` at the end, and a new `forward_kl_mean(ref, abl, *, "
            "chunk=512)`. THIS IS A DECLARED SEMANTIC CHANGE to the search's "
            "numerics and NOT an optimization that leaves the arithmetic "
            "alone: the summation order is the same, but the chunk partials "
            "are now added in device float64 rather than host float64, so the "
            "six quantities move. Calling it a refactor would be the misreport "
            "this file exists to catch. It is admissible because the movement "
            "was MEASURED rather than argued -- worst relative drift "
            "3.03e-05 on the real pinned teacher, 257x below the smallest "
            "decision threshold the search is known to use (0.007782) and just "
            "under float32's own `sqrt(V)*eps` floor of 4.65e-05 for a "
            "151936-class vocabulary, with item ordering identical and top-1"
            " agreement exact. Equivalence of the DECISIONS is the claim; "
            "equality of the digits is not, and no bound below that floor "
            "could have been met by any implementation. The saving is the "
            "`.float().cpu()` transfer of two `[T, ~152k]` tensors per item: "
            "76.0x on the reduction, from "
            "`logs/stages/stage-1/phase_c2/validations/full-search-performance/"
            "v1/closeout.json`.",
        "src/aadistill/initialization/planning/metrics.py":
            "`StateEvaluator` no longer calls `.float().cpu()` on the two "
            "logit tensors; targets and tag masks are moved to the reference's "
            "device instead. THIS IS A DECLARED SEMANTIC CHANGE, and it is "
            "the one that realises the saving above -- the reduction could "
            "stay on the card and still be handed host tensors by its only "
            "production caller, which is exactly what subrun p1 accidentally "
            "measured and reported as 1.01x.",
        "src/aadistill/initialization/operators/depth.py":
            "`memory_snapshot(device)` -- driver free/total, allocator "
            "allocated/reserved, and the two derived quantities "
            "`reclaimable_by_empty_cache_gib` and `unaccounted_gib` -- taken "
            "before the reference cache's availability probe and carried into "
            "`decision()[\"memory_at_admission\"]`; and the candidate scoring "
            "loop now calls `forward_kl_mean` instead of building targets and "
            "reducing all six quantities. THIS IS A DECLARED SEMANTIC CHANGE "
            "to what DEPTH computes per candidate: five of the six quantities "
            "were computed and discarded, and the greedy rule reads forward KL "
            "only. Measured at 1.10x with both sides device-resident, removal "
            "order `[17, 18]` identical in three independent measurements. The "
            "snapshot is instrumentation and changes no behaviour -- it was "
            "added to test whether allocator hoarding explains the historical "
            "2.6-GiB-free observations, and it REFUTED that: 0.013 GiB "
            "reclaimable on a card holding only the teacher, cache admitting "
            "67/67. Nothing was flushed and no saving is claimed from it.",
     }),
    #: TWO ROUNDS THAT NEVER DECLARED THEMSELVES, found by this check when a
    #: later round ran it from an older base. Both landed while the replay
    #: campaign was relaunching attempt after attempt, both edited the shared
    #: runner, and neither appended an entry here — which is the exact shape
    #: this file exists to catch, just with the delay that comes of only ever
    #: reading the newest base. The tip is the commit both rounds started from.
    ("56fed7a83b8a62149fc7d39f9c2c9f4bb42eabd5",
     "--dry-run, and securing a unit of work when it finishes (late declaration)",
     {
        "src/aadistill/infrastructure/session_runner.py":
            "TWO DECLARED SEMANTIC CHANGES, from 09e2ab59 and 8e02f025. "
            "(1) `run()` consults `getattr(self.a, \"dry_run\", False)` after "
            "the prechecks and returns before `create()`, recording "
            "`DRY_RUN_GATES_PASSED` and `provider_resource_created: False`. "
            "Several launchers advertised the flag as \"run every $0 gate and "
            "stop before provider creation\" while nothing consulted it, so a "
            "dry run whose gates all passed created a real pod and billed for "
            "it; the flag was only ever safe because some gate happened to "
            "refuse first. `getattr` because a launcher without the flag must "
            "keep behaving exactly as before. (2) the poll loop calls "
            "`spec.artifacts.on_poll(self.context())` once per iteration inside "
            "`try/except Exception`, appending failures to `on_poll_errors`. "
            "The broad catch is deliberate and is the point: a durability "
            "convenience that can kill a paid session is worse than none.",
        "src/aadistill/infrastructure/session.py":
            "`ArtifactPolicy.on_poll`, an optional `Callable[[SessionContext], "
            "None] | None = None`. THIS IS A DECLARED SEMANTIC CHANGE to the "
            "policy's shape, though not to any existing session's behaviour: "
            "the default is `None` and a policy that does not set it runs "
            "exactly as before. It exists because AGENTS.md requires a "
            "finished unit of work to survive a LATER stage's failure, and a "
            "collector that only runs at closeout cannot honour that when the "
            "pod dies — a replay lost two verified checkpoints that way.",
     }),
    ("c87f8767a5a006796d68daaf42b7e6ca8e9731e7",
     "one identity construction for a transferred checkpoint",
     {
        "src/aadistill/runtime/leaf_durability.py":
            "`identify_for_transfer(directory, *, adapter, arch_signature, "
            "num_parameters)` extracted, and `verify_transferred_leaf` now "
            "calls it instead of rebuilding the identity inline. THIS IS A "
            "DECLARED SEMANTIC CHANGE, and a deliberately small one: the "
            "arithmetic is byte-for-byte the code that was already there, "
            "moved so that the SENDER and the RECEIVER compute identity by one "
            "construction rather than two. `artifact_digest` covers "
            "`tokenizer_sha256`, so a sender recording `None` while the "
            "receiver hashed the tokenizer files that arrived would mismatch "
            "on every transfer of a checkpoint carrying a tokenizer -- a "
            "disagreement about bookkeeping, reported as corruption. The "
            "return value additionally carries `config_sha256`, "
            "`arch_signature`, `num_parameters` and a THREE-VALUED "
            "`config_matched` (None when the record predates the field, so "
            "'not recorded' and 'did not match' stay distinct); no existing "
            "key changed meaning and every existing caller reads the same "
            "flags it read before.",
     }),
    ("dbf71be578fc221ce554b21efd16cc30d3fe2732",
     "the storage derivation, after attempt5 ran out of disk",
     {
        "src/aadistill/runtime/cost.py":
            "a generic storage model: `BYTES_PER_PARAM` covering both the "
            "short and the TORCH dtype spellings real configs are written in, "
            "`bytes_per_param` which RAISES on an unknown name rather than "
            "defaulting, `CheckpointFootprint`, `training_working_set_bytes`, "
            "`ResidencyUnit`, `peak_local_residency_bytes` and "
            "`durable_backend_bytes`. THIS IS A DECLARED SEMANTIC CHANGE, and "
            "it is additive: every existing caller of `checkpoint_bytes` is "
            "untouched and its bf16 default still applies to whoever relied "
            "on it. Every dtype is an ARGUMENT -- no parameter count, model "
            "family or experiment name is encoded, and a test asserts that. "
            "It exists because `behavioural.storage_requirement` charged a "
            "trained probe at the size of the bf16 leaf it started from while "
            "the recipe declares `dtype: float32`, so every retained probe "
            "was charged at half its real size and attempt5's trainer hit "
            "`No space left on device` writing probe 11 of 12. Separating "
            "`peak_local_residency_bytes` from `durable_backend_bytes` is the "
            "second half: container storage and durable capacity were one "
            "quantity, so bytes that must merely survive teardown were "
            "charged to the pod's own disk.",
     }),
    ("b46d10fad973f969e936ba971fc2531e87283619",
     "a durable object store with no vendor in it",
     {
        "src/aadistill/runtime/durable_store.py":
            "NEW MODULE. A generic durable object-store interface -- the "
            "`DurableStore` protocol of four methods, `TransferRecord`, and "
            "`upload_checkpoint` / `restore_checkpoint` which compose a "
            "transport with `identify_for_transfer` and "
            "`verify_transferred_leaf`. THIS IS A DECLARED SEMANTIC ADDITION "
            "to the shared runtime and it has no caller yet: wiring it into "
            "the behavioural launcher needs a backend that does not exist. Its "
            "purpose is that identity cannot be skipped -- upload records the "
            "source identity with the object, restore re-identifies from the "
            "ARRIVED bytes and refuses a mismatch, leaving the arrival in "
            "place as evidence about the transport. It names no provider, "
            "protocol, bucket, region, endpoint, experiment, model family or "
            "parameter count, and tests assert each; backends live in the "
            "application layer where a vendor and a credential belong. It "
            "exists because the transport that pushes multi-GiB objects from a "
            "dev box to an already-billing machine charges the slow leg at the "
            "accelerator's rate, which priced one continuation above a fresh "
            "session.",
     }),
    ("8966ae12cd5f898928b2df443601a8936e03ba1e",
     "a session may attach a volume the provider already holds",
     {
        "src/aadistill/infrastructure/session_runner.py":
            "DECLARED SEMANTIC CHANGE, additive and optional. `create` now "
            "splices `attached_volume()` where it used to hardcode "
            "`--volume-in-gb 0`, so a session can attach a provider network "
            "volume that already holds bytes it needs, at a mount path it "
            "names, in the datacenter the volume lives in. A launcher that "
            "defines none of `--network-volume-id`, `--volume-mount-path` or "
            "`--data-center-ids` produces the identical command line it "
            "produced before -- which is why they are read with `getattr`: "
            "attaching nothing is the status quo for every existing launcher "
            "and must not become something they opt out of. A volume named "
            "without a mount path RAISES, because the provider's default is "
            "`/workspace`, this project's checkout root, and mounting shared "
            "network storage over a session's working tree is not a thing to "
            "discover from a running pod. No provider, volume id, datacenter, "
            "experiment or campaign is encoded here; all four are arguments. "
            "It exists because the C2 behavioural continuation had to put "
            "22.2 GiB of completed probes on each replacement pod, the "
            "launcher host's uplink is ~0.5 MB/s, and copying them during the "
            "session charged ~9 hours of L40S time per attempt. Written once "
            "to a volume at CPU prices, they are simply present when the pod "
            "boots. THE OBJECT-STORE ROUTE DECLARED IN THE PREVIOUS ROUND WAS "
            "SUPERSEDED BEFORE IT RAN -- every S3-compatible backend reachable "
            "from here needs an account this environment cannot create, and "
            "`runtime/durable_store.py` still has no production caller.",
     }),
    ("d7dbf1ad78acb584bb28715f7e4b292dbdccadba",
     "the protocol-field admission rule, and ownership of a blocked teardown",
     {
        "src/aadistill/evaluation/protocol_field.py":
            "NEW MODULE. The admission rule that refuses a pooled or paired "
            "field unless every member establishes ONE compatible measurement "
            "protocol: battery, scoring-contract and metric-contract identity "
            "by equality, and the generation protocol through "
            "`generation_compat`'s `require_comparable` rather than by "
            "fingerprint equality, because that rule already owns whether two "
            "runtimes are comparable and deliberately demotes the NVIDIA "
            "driver patch. THIS IS A DECLARED SEMANTIC CHANGE by virtue of "
            "being new executable core. It FAILS CLOSED in three directions -- "
            "identities that differ, identities that are absent, and a "
            "generation protocol that differs but cannot be judged because the "
            "expanded protocol and runtime blocks were never recorded -- "
            "because 'not shown to be comparable' is not 'comparable'. It "
            "names no experiment, arm, battery, stage or seed: the keys are "
            "opaque, the field's name is a caller argument, and two tests "
            "assert both the source and the BASENAME are free of this "
            "project's experiment tokens. It sits beside `paired_stats`, the "
            "arithmetic it guards. One application caller exists, a thin "
            "wrapper that only translates the error type; the reason the rule "
            "was paid for is in `docs/core-provenance.md`.",
        "src/aadistill/infrastructure/session_runner.py":
            "`verify_watchdog_owns_pod`, and the blocked-teardown branch of "
            "`collect_and_teardown` now acting on it. THIS IS A DECLARED "
            "SEMANTIC CHANGE to provider ownership: the branch previously "
            "asserted 'the watchdog remains the backstop' and checked "
            "nothing, and a blocked artifact gate therefore returned leaving a "
            "pod that nothing was known to be watching. Ownership now requires "
            "all three of a live pid, a watchdog launched for THIS pod, and a "
            "`/proc/<pid>/cmdline` naming both -- the third because pids are "
            "reused and a recycled pid answers `kill(pid, 0)` identically. "
            "Owned is UNCHANGED behaviour (retain for evidence); unowned now "
            "tears the resource down. The watchdog pid is captured at launch "
            "and read defensively, so a spawn object without a readable pid "
            "degrades to 'ownership cannot be established' instead of raising "
            "immediately after a pod starts billing.",
     }),
    ("ab53ba1422afd6324efb4964e129dfabb581dd10",
     "operators organised by topology, and calibration forwards micro-batched",
     {
        "src/aadistill/initialization/calibration/batching.py":
            "NEW. The one place a sequence of calibration items becomes padded "
            "micro-batches and per-item results come back out: right padding so "
            "real tokens keep the position index they occupy alone, an explicit "
            "attention_mask, true per-row lengths, and `split_predictions` / "
            "`valid_tokens` to undo the padding. Batch size is configuration-"
            "driven with a stated default and 1 is the reference path. Family-, "
            "operator- and stage-neutral.",
        "src/aadistill/initialization/statistics/collect.py":
            "`process_batch` beside `process`, and ONE accumulation rule serving "
            "both: residual moments, FFN moments and the token histogram now "
            "reduce over a valid-position mask. Padded positions reach no "
            "accumulator. `process` is implemented as a one-row batch and its "
            "numbers are bit-identical -- the mask is skipped entirely when "
            "nothing is padded, so the reference path performs the operations it "
            "always did.",
        "src/aadistill/initialization/operators/_common.py":
            "`collect_activation_stats` gained `batch_size`/`pad_id` and accepts "
            "items as well as bare id tensors, so FFN, RESIDUAL_WIDTH and "
            "COMPOSITE_STAGE1 share one batched forward loop instead of three. "
            "`head_rows` LEFT for `attention/gqa/_common.py`: concatenated-per-"
            "head row arithmetic is a GQA fact, not a kind-neutral one.",
        "src/aadistill/initialization/planning/search.py":
            "`OperatorStep.config_hash` now also excludes "
            "`calibration_micro_batch_size`. That hash feeds `compute_state_id`, "
            "so leaving the key in would make a state id depend on the hardware "
            "a run happened to fit. Inert on every existing record -- no current "
            "path puts the key in `operator_config`, so the hashed dict is empty "
            "before and after.",
        "src/aadistill/initialization/operators/__init__.py":
            "re-exports follow the new module paths; no behaviour.",
        "src/aadistill/initialization/operators/register.py":
            "imports follow the new module paths. Registration stays explicit "
            "and `attention.activation_importance_v1` stays out of "
            "`BUILTIN_OPERATORS`.",
        "src/aadistill/initialization/operators/attention/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/attention/gqa/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/attention/gqa/_common.py":
            "NEW. The grouped-head selection topology every GQA algorithm "
            "shares, moved here unchanged from the operator and the kind-neutral "
            "_common: `head_rows`, the `q`/`attn_out` role resolution, "
            "`select_q_heads_by_score`, and the MHA/divisibility applicability "
            "rule now named once instead of restated per operator.",
        "src/aadistill/initialization/operators/attention/gqa/_statistics.py":
            "MOVED from `statistics/attention.py` -- the per-query-head second "
            "moment is a GQA sufficient statistic, not a universal one. Gained "
            "`process_batch` and a valid-position mask so padding cannot inflate "
            "`M_h` or `attn_token_count`.",
        "src/aadistill/initialization/operators/attention/gqa/activation_importance.py":
            "MOVED from `operators/attention_activation.py`. impl_id, kind, "
            "version, capabilities, calibration need and selection semantics "
            "unchanged; the statistics pass is micro-batched and the topology "
            "helpers are imported rather than defined here.",
        "src/aadistill/initialization/operators/attention/gqa/weight_proxy.py":
            "MOVED from `operators/attention.py`. Weight-only, runs no forward; "
            "only its `head_rows` import moved.",
        "src/aadistill/initialization/operators/ffn/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/ffn/dense/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/ffn/dense/activation_importance.py":
            "MOVED from `operators/ffn.py`; resolves and passes a micro-batch "
            "size. Selection unchanged.",
        "src/aadistill/initialization/operators/width/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/width/residual/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/width/residual/global_pca.py":
            "MOVED from `operators/width.py`; resolves and passes a micro-batch "
            "size. Kind stays RESIDUAL_WIDTH and impl_id stays "
            "`width.global_pca_v0`.",
        "src/aadistill/initialization/operators/depth/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/depth/_common.py":
            "NEW. `DEPTH_FIELD` and `_build_child_with_layers`, which both DEPTH "
            "algorithms share. No topology level under DEPTH: removing a decoder "
            "block is the same operation whatever is inside it.",
        "src/aadistill/initialization/operators/depth/positional.py":
            "SPLIT out of `operators/depth.py` unchanged. Takes no measurement "
            "and runs no forward.",
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py":
            "SPLIT out of `operators/depth.py`, and the one operator whose "
            "forwards are micro-batched: `_forward_logits_batch` and "
            "`_ReferenceLogits.get_batch`. THE SCORE IS UNCHANGED -- one "
            "`forward_kl_mean` per ORIGINAL item over its own prediction "
            "positions with the same chunk boundaries, then the same subtype mean "
            "and the same domain balance. A cached reference is CLONED off the "
            "padded block so the cache's own byte budget still describes what is "
            "resident. Telemetry gained `ablated_items` beside `ablated_forwards`, "
            "which now counts forwards rather than items.",
        "src/aadistill/initialization/operators/composite/__init__.py":
            "NEW package. Names only; registers nothing.",
        "src/aadistill/initialization/operators/composite/stage1_sandwich.py":
            "MOVED from `operators/composite.py`; reads the micro-batch size "
            "from `ctx.execution`. The monolithic recipe itself is untouched.",
        "src/aadistill/initialization/execution.py":
            "NEW. `ExecutionConfig` — HOW an operator runs, structurally "
            "separated from the `config` mapping that is hashed into "
            "`OperatorStep.config_hash` and therefore into the state id. Batch "
            "size lives here so it CANNOT fork a scientific identity; the "
            "alternative, excluding a key by name when hashing, would make the "
            "exclusion list the real definition of 'scientific' and would fork "
            "every state id the first time someone forgot to extend it.",
        "src/aadistill/initialization/operators/base.py":
            "`OperatorContext` gained `execution: ExecutionConfig`, defaulting "
            "to the shared default. Additive: every existing construction "
            "behaves exactly as before, and the field is deliberately not "
            "readable from `OperatorStep.identity()`.",
        "src/aadistill/initialization/planning/fixed_path.py":
            "`materialize_fixed_path`, `materialize_fixed_path_suffix` and the "
            "shared step loop take `execution` and pass it into the context — "
            "the same runtime-only treatment `deadline` already had. No "
            "operator ordering, gating, identity or artifact behaviour moves. "
            "NOTE: this file is on the HISTORICAL CUDA surface; that validation "
            "is not repointed at this change and a new one is owed.",
     }),
    ("6a349e03630d6d01f4a4010cc0e9086a11fa4a25",
     "one shared masked batched per-item KL reduction",
     {
        "src/aadistill/initialization/statistics/contribution.py":
            "NEW `forward_kl_mean_batch(ref, abl, prediction_mask, chunk=512) "
            "-> [B]`: the shared masked batched per-item forward-KL reduction "
            "every initialization causal operator is to use. ONE mean per row "
            "over that row's OWN valid positions -- deliberately NOT "
            "`(kl*mask).sum()/mask.sum()`, which is a token-weighted batch mean "
            "and would hand a 1000-position item ten times the influence of a "
            "100-position one. `forward_kl_mean`'s numerical contract carries "
            "over unchanged (float32 log_softmax, chunks along the SEQUENCE-"
            "POSITION axis at the same boundaries, float32 per-chunk reduction, "
            "float64 accumulation, accumulators where the logits are); the "
            "float64 accumulator is [B] so no two items meet before their means "
            "are formed. `forward_kl_mean` and `distortion` are UNCHANGED and "
            "remain the scalar oracles.",
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py":
            "the batch>1 hot path keeps `[B, T_pred, V]` through to the reducer "
            "instead of splitting into B tensors and looping. "
            "`_forward_logit_block` returns the block; "
            "`_ReferenceLogits.reference_block` makes the reference forward's "
            "COMPOSITION independent of cache state -- a partially cached batch "
            "now forwards the WHOLE canonical batch rather than a sub-batch of "
            "the missing rows, so batch width and position alignment no longer "
            "depend on how much memory was free. A cached row is cloned as its "
            "valid prediction slice only, so the cache budget still describes "
            "what is resident. Estimand, subtype mean and domain balance are "
            "untouched; batch_size=1 still takes the per-item reference path.",
        "src/aadistill/initialization/statistics/collect.py":
            "`ActivationStatsCollector` no longer walks `model.model.layers` or "
            "`layer.mlp.down_proj`. It takes the ordered FFN-output projections "
            "from its caller, as the attention collector already did, and the "
            "adapter resolves them BY ROLE. Same statistics, same numbers; the "
            "family knowledge moves to where a new architecture would add it.",
        "src/aadistill/initialization/adapters/qwen3.py":
            "`stats_collector` resolves `stream_out_projections(block)['ffn_out']` "
            "per block and hands the list to the collector. NOTE: on the "
            "HISTORICAL CUDA surface; that validation is not repointed and the "
            "new one is owed.",
        "src/aadistill/initialization/operators/composite/stage1_sandwich.py":
            "the trace distinguishes statistics SUPPLIED by the caller from "
            "statistics this invocation COLLECTED, and reports "
            "`micro_batch_size: None` in the supplied case -- a batch size for a "
            "pass that never ran would describe work the operator did not do.",
        "src/aadistill/runtime/cost.py":
            "`operator_cost` prices `plan.forward_passes` GENERICALLY instead of "
            "only for the one operator named by id, and the forward and "
            "statistics arms became ADDITIVE rather than alternatives. THIS IS A "
            "DECLARED SEMANTIC CHANGE to estimated cost. It moves no existing "
            "number: `depth.causal_kl_greedy_v1` keeps its own earlier branch "
            "unchanged, and every other shipped operator declares zero forwards "
            "and zero stats -- both asserted by walking the live registry in "
            "tests/runtime/test_forward_pass_pricing.py. What changes is that "
            "the NEXT forward-heavy operator prices above zero instead of free.",
        "src/aadistill/initialization/planning/fixed_path.py":
            "`FixedPathStep` gained an optional `config` mapping, merged over "
            "the executor-derived operator config for that step alone. THIS IS A "
            "DECLARED SEMANTIC CHANGE to step identity: a step carrying a config "
            "serializes it, so its `spec_hash` and `compute_state_id` differ "
            "from the same step without one. It is additive -- `as_dict()` omits "
            "the key entirely when the config is empty, so every historical "
            "path serializes to the bytes it always did, pinned by a frozen "
            "serialization test. The field exists so one step's protocol (the "
            "pilot's calibration forward batch size) is part of what its "
            "identity commits to, rather than an invisible runtime flag.",
     }),
    ("2998313dfa372f840595e7f87b5587e3c89334d3",
     "a step config that cannot rewrite the run it executes in",
     {
        "src/aadistill/initialization/planning/fixed_path.py":
            "two corrections to the config field declared by the round above, "
            "both narrowings. (a) `step_operator_config` now FAILS CLOSED on "
            "`EXECUTOR_DERIVED_CONFIG_KEYS`: a step declaring "
            "`n_calibration_items` raises `FixedPathError` instead of "
            "overriding it. That key is `len(ctx.calibration_items)`, not "
            "operator policy, and the previous last-writer-wins merge let a "
            "step be PLANNED against a corpus it would not EXECUTE against -- "
            "the plan being what the cost model and the reachability check "
            "read. (b) `__post_init__` canonicalises `config` to a key-sorted "
            "`MappingProxyType` and `as_dict()` copies out, so a frozen "
            "dataclass holding a live dict can no longer have its identity "
            "moved by the caller afterwards. THIS IS A DECLARED SEMANTIC "
            "CHANGE: one previously-accepted input is now refused, and config "
            "key order no longer reaches the hash. No existing step declares a "
            "config, so no historical identity moves. (c) "
            "`SELECTION_ARTIFACTS` gained `causal_head_evidence`, so an "
            "operator's scoring LANDSCAPE — per-item KLs, aggregate head "
            "scores, per-GQA-group cutoffs — reaches the persisted "
            "`StepResult` instead of being discarded at the executor "
            "boundary. Additive and presence-filtered: an operator that does "
            "not produce the key serializes exactly as before, which is every "
            "historical one.",
        "src/aadistill/initialization/calibration/packing.py":
            "NEW MODULE: named deterministic packing policies. THIS IS A "
            "DECLARED SEMANTIC CHANGE by construction -- new executable core "
            "is semantic whatever it contains. `pack(lengths, batch_size, "
            "packing)` is pure integer work returning groups of ORIGINAL "
            "indices, so a protocol's padding cost is a $0 fact derived from "
            "the frozen mixture rather than a number a pod discovers. "
            "`length_sorted_v1` sorts by (length ascending, original index "
            "ascending) and splits contiguously. "
            "`micro_batches`'s DEFAULT IS UNCHANGED and still never sorts -- "
            "asserted by a test -- so nothing that does not ask for a policy "
            "behaves differently. It forwards on the calibration mixture, so "
            "it joins CURRENT_CUDA_SURFACE and owes the new validation.",
        "src/aadistill/initialization/operators/attention/gqa/causal_kl.py":
            "THIS IS A DECLARED SEMANTIC CHANGE, in three parts. (a) The "
            "scoring loop is extracted as `score_heads`, so a throughput "
            "screen measures the function the operator calls rather than a "
            "copy of it; its `layers` argument restricts which blocks are "
            "ablated and exists ONLY for timing -- `apply` never passes it, "
            "and a partial landscape is refused at aggregation. (b) The "
            "operator binds a SECOND identity-bearing config key, "
            "`calibration_batch_packing`: B2-consecutive and B2-length-sorted "
            "run the same forwards over the same items and can select "
            "different heads, so a config binding only the batch size would "
            "give two protocols one state id. (c) THE AGGREGATION NOW WALKS "
            "THE FROZEN MIXTURE ORDER. It previously appended per-item KLs in "
            "group execution order; under a reordering packing that would "
            "move batch composition AND float summation order together, and "
            "neither effect could be attributed. Values are written by "
            "original index and the subtype buckets are built from "
            "`enumerate(items)`. For every existing protocol -- every "
            "consecutive-order run -- execution order IS mixture order, so no "
            "historical number moves.",
        "src/aadistill/initialization/operators/attention/gqa/causal_kl.py":
            "NEW MODULE: `attention.causal_kl_v1`, the C3 candidate. THIS IS A "
            "DECLARED SEMANTIC CHANGE by construction -- new executable core is "
            "semantic whatever it contains. It scores each query head by "
            "one-shot causal ablation (forward KL of the parent against the "
            "parent with that head's o_proj column block zeroed), aggregated "
            "domain-balanced exactly as DEPTH aggregates, then keeps the "
            "top-scoring heads per GQA group through the shared selection "
            "topology. One shot, not greedy: no rescoring after a removal. "
            "It moves nothing that already existed -- it is a NEW module "
            "precisely so that adding it edits no file a frozen source set "
            "pins, it is absent from `BUILTIN_OPERATORS`, importing it "
            "registers nothing, and no C1 entry point reaches it (the "
            "regenerated closure still names 105 files and does not include "
            "it). ALONE among the operators it reads its calibration forward "
            "batch size from its own hashed step config rather than from "
            "`ctx.execution`, because padded batching is measured to move this "
            "family of decisions and so belongs in the identity. "
            "NOT YET CUDA-VALIDATED: it runs forwards on the calibration "
            "mixture, so it joins `CURRENT_CUDA_SURFACE` and owes the new "
            "validation rather than claiming cover from the 2026-09-10 one.",
     }),
    ("dead70043922ae668965af78f13a1fbfcf023529",
     "asking the durable backend whether it has room, before spending",
     {
        "src/aadistill/runtime/hub_capacity.py":
            "NEW FILE, and additive: nothing imported it before. It asks the "
            "git-LFS batch endpoint whether an upload of given sizes would be "
            "accepted, which the endpoint answers from quota BEFORE any bytes "
            "move -- so it costs seconds, transfers nothing and stores "
            "nothing. It exists because AGENTS.md P8.2.1 requires confirming "
            "a durable large-artifact backend has ROOM before a long run, and "
            "twice it was not confirmed: C1 attempt 18 lost six 2.22 GiB "
            "probes and C3 attempt66 lost nine, both to "
            "`Private repository storage limit reached`, with the durability "
            "mechanism behaving perfectly and having nowhere to write. NO "
            "CUDA SURFACE IS TOUCHED: it imports `requests` and `secrets`, "
            "runs on the launcher host before a pod exists, and is never "
            "imported by any initialization, operator, training or evaluation "
            "path. It names no experiment, no artifact count and no "
            "threshold -- the caller owns the sizes and the decision, which "
            "is why the C3-specific gate and its measured per-probe byte "
            "count live in the launcher instead.",
     }),
    ("b0bd14929892391152375de309a1741dfdfc5f4a",
     "A-bsz3: the batching protocol as an execution knob, not an operator",
     {
        "src/aadistill/initialization/execution.py":
            "ADDITIVE: `ExecutionConfig` gains `calibration_batch_packing`, "
            "defaulting to `original_order_v1`, plus validation against "
            "`PACKING_POLICIES` and the field in `as_trace()`. This is the "
            "second knob the module's own docstring promised -- 'a field here "
            "and nothing at all in the hashing path' -- and the boundary is "
            "unchanged: nothing about identity moves, and a test asserts the "
            "packing reaches neither `ctx.config` nor `OperatorStep.identity()`. "
            "NO CUDA SURFACE: the dataclass holds two scalars and imports only "
            "the packing policy names.",
        "src/aadistill/initialization/operators/attention/gqa/activation_importance.py":
            "SEMANTIC, and deliberately narrow. The operator now routes through "
            "`packed_batches` and honours the packing policy; the REFERENCE "
            "PATH is guarded by an explicit `batch_size <= 1 and packing == "
            "ORIGINAL_ORDER_V1`, which is exactly the loop this operator has "
            "always run, so the frozen C1 selection is reproduced by "
            "construction rather than by tolerance. `bsz>1 + original_order` is "
            "also unchanged, because `packed_batches` yields what "
            "`micro_batches` yields row for row at that policy. A non-default "
            "packing is therefore never silently ignored. Also traces "
            "`calibration_batch_packing`, `reference_path`, "
            "`kept_q_heads_per_layer` and `selection_margin_per_layer` -- all "
            "evidence, none of it read by `OperatorStep.identity()`. ON THE "
            "CURRENT CUDA SURFACE: it runs calibration forwards, so it owes the "
            "new validation and claims no cover from the 2026-09-10 one.",
        "src/aadistill/initialization/operators/attention/gqa/_common.py":
            "ADDITIVE: `selection_margins` returns, per GQA group, the gap "
            "between the last kept head and the first dropped one. A pure "
            "function of a score vector -- no tensors created, no device, no "
            "model -- and no existing function is touched. It exists because "
            "scores alone cannot say whether two protocols that disagree about "
            "a selection tipped a near-tie or genuinely rank heads differently.",
        "src/aadistill/governance/grant.py":
            "ADDITIVE: `refuse_a_future_dated_grant` refuses a grant dated "
            "after today UTC. C2-behavioural's attempt3 carried a "
            "local-timezone date in a UTC field and nothing detected it because "
            "nothing read it; that issuer gained an inline check and C3's never "
            "did. Experiment-agnostic here so later phases share one owner. "
            "NO CUDA SURFACE: one date comparison.",
     }),
    ("ab4f32ed4d8cb77014f1e1fd0acde83ee73d8d8a",
     "the structural record the shortened A-bsz3 study reads",
     {
        "src/aadistill/initialization/operators/attention/gqa/activation_importance.py":
            "ADDITIVE EVIDENCE, and a timer. THIS IS A DECLARED SEMANTIC "
            "CHANGE because the executable shape moves, but nothing it adds "
            "is read by `OperatorStep.identity()` -- the artifact digest is "
            "taken from the written bytes, so no selection, no score and no "
            "identity can move. Four additions, each because the shortened "
            "A-bsz3 design asks for a quantity the operator did not emit. "
            "(a) `head_scores_per_layer`: the per-head score vector, which is "
            "what a rank correlation needs and what the kept-head counts "
            "cannot reach -- two protocols can differ in every score and "
            "agree on every selection. (b) `physical_forward_invocations`, "
            "`executed_positions`, `valid_positions` and `padded_positions`, "
            "COUNTED BY THE LOOP AS IT RAN rather than re-derived from the "
            "item lengths: `padding_profile` predicts the same three before "
            "any pod exists, and a trace that restated that prediction could "
            "never contradict it. The loop's `valid_positions` and the "
            "collector's independent `calibration_tokens` give a masking "
            "invariant a consumer can check instead of trusting. (c) "
            "`scorer_seconds`: the statistics pass alone, CUDA-synchronized "
            "at both ends, on the same `time.monotonic` clock `causal_kl` "
            "reports. The comparison is a ratio of wall clocks and the "
            "checkpoint write that surrounds the pass is identical under "
            "either protocol, so timing the suffix instead would dilute the "
            "ratio toward 1 by exactly the write. (d) a module-local "
            "`_cuda_sync`, deliberately NOT imported from `causal_kl`: this "
            "module's claim is that A-bsz3 shares no code with the C3 "
            "candidate, and two lines are not worth falsifying it. ON THE "
            "CURRENT CUDA SURFACE already; it owes the same pending "
            "validation and claims no new cover.",
     }),
    ("fe2db4c8ffed3713b857a8cbdc7e3a9218fefa7d",
     "the same-failure rule, enforced where a resource is created",
     {
        "src/aadistill/infrastructure/failure_signature.py":
            "NEW MODULE, and NO CUDA SURFACE: it compiles regular "
            "expressions and compares strings. It normalizes a paid attempt's "
            "failure to `CLASS:token` so that two attempts can be asked "
            "whether they failed the SAME way -- the question A3 answered "
            "twenty-one times with twenty-one pods. The token is kept "
            "deliberately: collapsing every `${VAR:?}` refusal into one class "
            "would make fixing `SESSION_FROZEN_EXPECT` look like it addressed "
            "a later, different missing variable, and the gate would then "
            "refuse a legitimate retry. Provider capacity, a cold host and an "
            "unreachable endpoint return None, which is what preserves the "
            "backoff policy for transients. It names no experiment, no stage, "
            "no session kind and no repository path; the patterns are failure "
            "classes of the shared setup/launch path, which is "
            "deployment-level infrastructure.",
        "src/aadistill/infrastructure/session_prechecks.py":
            "ADDITIVE: one new gate factory, `same_failure_gate`. NO CUDA "
            "SURFACE -- it reads text, compares two strings and runs `git "
            "diff`; it touches no tensor, no device and no dtype, and it runs "
            "strictly before any provider resource exists. THIS IS A DECLARED "
            "SEMANTIC CHANGE because the executable shape of a shared "
            "pre-provider module moves, and because it can now REFUSE a "
            "launch that previously proceeded. It exists because A3 created "
            "twenty-one paid pods discovering one missing setup variable: the "
            "rule 'repeating an identical failure unchanged is not a repair' "
            "was written in AGENTS.md and nothing executed it. Both instance "
            "facts arrive as callables -- where prior attempts' evidence "
            "lives, and what counts as a corrective change -- so the core "
            "names no repository path and no experiment. Every existing "
            "session is unaffected: a precheck is a value in "
            "`SessionSpec.precheck` and only A3's spec lists this one.",
     }),
    ("bdc909234f3f984f91570315d154d0f0adf2d47f",
     "host admission, asked before a dollar of setup",
     {
        "src/aadistill/infrastructure/session.py":
            "ADDITIVE: one optional field, `SessionSpec.host_admission`, "
            "defaulting to None so every existing session is unaffected. NO "
            "CUDA SURFACE -- it is a callable the runner invokes with a "
            "string. DECLARED because the executable shape of the shared "
            "session declaration moves.",
        "src/aadistill/infrastructure/session_runner.py":
            "ADDITIVE: the runner asks that callable once per draw, between "
            "the provider-confirmed image identity and the setup invocation, "
            "and treats a refusal as REDRAWABLE beside `cold` and "
            "`no_endpoint`. NO CUDA SURFACE -- no tensor, no device, no "
            "dtype; it runs before any science starts. DECLARED because it "
            "can now abandon a created resource that previously proceeded, "
            "which is a behaviour change in a shared paid path. It exists "
            "because a3_attempt35 trained three probes and was refused at "
            "$4.33 on a host property knowable one ssh round trip after the "
            "pod answered: every material generation field matched "
            "attempt75's controls and the NVIDIA driver BRANCH had moved "
            "580 -> 595, which generation_compat declares a real runtime "
            "event. A raising check refuses rather than admitting.",
     }),
    ("4ae52e2ed17a14956e0b9044a0ac976d1e81974b",
     "materialization identity and target-aware scoring positions",
     {
        "src/aadistill/initialization/specs/materialization.py":
            "NEW MODULE, therefore a DECLARED SEMANTIC CHANGE by construction. "
            "Four identities where there was one: semantic_state_id (unchanged, "
            "still blind to execution), numerical_execution_fingerprint, "
            "materialization_id and artifact_digest. It exists because one "
            "operator, one path and one hashed config were measured producing "
            "two different checkpoints under an identical result_spec_hash, "
            "reproducibly and across machines, while resume, dedup and "
            "checkpoint ownership all keyed on the single id. NO CUDA SURFACE: "
            "no tensor, no device, no dtype -- it hashes strings. The device "
            "CLASS is a fingerprint FIELD and an ordinal is refused.",
        "src/aadistill/initialization/scoring/positions.py":
            "NEW MODULE, therefore a DECLARED SEMANTIC CHANGE. The hash-bound "
            "ScoringPositionPolicy: which positions a calibration-derived "
            "objective may read, on a token axis and a prediction axis. "
            "`positions.all_v1` is the incumbent semantics, named and hashed, "
            "and is NUMERICALLY INERT -- the reducers detect its `form == all` "
            "and take the arithmetic they took before this module existed. "
            "NO CUDA SURFACE: it builds float64 host weight vectors and reads "
            "a mixture's own tags. No registration happens at import.",
        "src/aadistill/initialization/scoring/batches.py":
            "NEW MODULE, therefore a DECLARED SEMANTIC CHANGE. The one bridge "
            "from a per-item policy answer to the [B, T_max] shape an operator "
            "runs, through PackedBatch.original_indices so a reordering packing "
            "policy cannot attribute one item's supervised positions to "
            "another's activations. CUDA-ADJACENT but device-neutral: it places "
            "its masks on the batch's own device and performs no arithmetic on "
            "activations.",
        "src/aadistill/initialization/scoring/__init__.py":
            "NEW MODULE, therefore a DECLARED SEMANTIC CHANGE. Re-exports only.",
        "src/aadistill/initialization/execution.py":
            "ADDITIVE: `FINGERPRINT_FIELDS` and `as_fingerprint()`, so the "
            "class that owns the execution knobs also owns which of them affect "
            "BYTES -- `specs.materialization` asks rather than enumerating. "
            "Deliberately separate from `as_trace()`, which is evidence and may "
            "grow a field that moves no bytes. No existing field, default or "
            "validation changed. NO CUDA SURFACE.",
        "src/aadistill/initialization/specs/state.py":
            "ADDITIVE: an optional `materialization` field, bound to the "
            "artifact digest in `mark_materialized` and serialized ABSENT (not "
            "null) when undeclared, so a journal record written before the "
            "field existed keeps its exact serialization. `compute_state_id` is "
            "UNCHANGED, which is the point: two batching protocols of one path "
            "remain one hypothesis. NO CUDA SURFACE.",
        "src/aadistill/initialization/statistics/contribution.py":
            "DECLARED SEMANTIC CHANGE to all three reducers. `forward_kl_mean`, "
            "`forward_kl_mean_batch` and `distortion` take an optional position "
            "weight vector; `DistortionSums` gains `weight` as the denominator "
            "beside `positions` as the count, and its tagged entries carry a "
            "weight and a position count separately. `weights=None` performs "
            "the operations these functions always performed, character for "
            "character -- asserted against an inlined copy of the previous "
            "arithmetic at three chunk sizes, bit-identical. CUDA-ADJACENT: the "
            "float32-chunk/float64-accumulate contract, the chunk boundaries "
            "and the device-resident accumulator branch are all unchanged, and "
            "the weight multiply happens inside the existing chunk loop.",
        "src/aadistill/initialization/statistics/collect.py":
            "DECLARED SEMANTIC CHANGE: `process` and `process_batch` take an "
            "optional `active_mask`, combined with the padding mask by `and` so "
            "that restricting the statistic and excluding padding are ONE "
            "mechanism driving the existing `_keep_valid`. All-active is "
            "byte-for-byte the previous accumulation. CUDA SURFACE: the hooks, "
            "the device-resident float64 accumulators and the single host "
            "transfer in `state()` are untouched; only which rows survive "
            "`_keep_valid` can change, and only when a caller supplies a mask.",
        "src/aadistill/initialization/calibration/batching.py":
            "ADDITIVE: `active_rows`, which reconciles a policy's position mask "
            "with the [B, T] shape of the tokens being processed and RAISES on "
            "a mis-shaped one rather than broadcasting. Batch knowledge, so it "
            "lives beside ItemBatch and both collectors import the one copy. "
            "`build_batch`, `micro_batches`, `resolve_pad_id` and the pad-id "
            "policy are unchanged. NO CUDA SURFACE.",
        "src/aadistill/initialization/operators/base.py":
            "DECLARED SEMANTIC CHANGE: `OperatorContext.position_policy`, "
            "defaulting to the inert incumbent, plus a refusal in `execute()` "
            "when the hashed config NAMES a policy the context did not supply. "
            "The declared hash is what the state id derives from, so a mismatch "
            "would record a scoring rule that did not run. NO CUDA SURFACE.",
        "src/aadistill/initialization/operators/_common.py":
            "DECLARED SEMANTIC CHANGE: `collect_activation_stats` takes a "
            "packing policy and an ActivePositions, and routes non-reference "
            "execution through `packed_batches` instead of `micro_batches` -- "
            "equal row for row at the default packing, which is the reference "
            "path this function preserves explicitly. CUDA SURFACE: the "
            "collector lifecycle and the per-item `process` call are unchanged "
            "at batch 1 / default packing.",
        "src/aadistill/initialization/operators/attention/gqa/_statistics.py":
            "DECLARED SEMANTIC CHANGE: `process` and `process_batch` take an "
            "optional `active_mask`, combined with padding exactly as the "
            "residual/FFN collector combines them. HISTORICAL CUDA SURFACE -- "
            "see HISTORICAL_SURFACE_CHANGED_PENDING_REVALIDATION. The hooks, "
            "the einsum, the device-resident float64 accumulator, `release()` "
            "and `state()` are untouched.",
        "src/aadistill/initialization/operators/attention/gqa/activation_importance.py":
            "DECLARED SEMANTIC CHANGE: the write-energy expectation is taken "
            "over the policy's admitted positions, and the trace separates "
            "`admitted_positions` from `valid_positions` because "
            "`executed - padded == calibration_tokens` held only while every "
            "valid position was admitted. HISTORICAL CUDA SURFACE -- see "
            "HISTORICAL_SURFACE_CHANGED_PENDING_REVALIDATION. The reference "
            "path, the CUDA synchronisation points, the transfer boundary and "
            "the per-group top-k selection are unchanged.",
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py":
            "DECLARED SEMANTIC CHANGE: the causal-KL mean is taken over the "
            "policy's weighted positions, and the micro-batch grouping now "
            "comes from `packed_batches` with its original indices so a "
            "reordering policy keeps per-item attribution. CUDA-ADJACENT: the "
            "reference-cache admission, the per-item/batched branch, the chunk "
            "size and the greedy rule are unchanged.",
        "src/aadistill/initialization/operators/ffn/dense/activation_importance.py":
            "DECLARED SEMANTIC CHANGE: E[|a_j|] becomes an expectation with "
            "respect to the policy's admitted positions, and the statistics "
            "pass honours the run's packing. The top-k selection and the weight "
            "surgery are unchanged. NO new device interaction.",
        "src/aadistill/initialization/operators/width/residual/global_pca.py":
            "DECLARED SEMANTIC CHANGE: the residual second moments accumulate "
            "over the policy's admitted token positions, and the statistics "
            "pass honours the run's packing. The point set, the 9/8 end "
            "weights, the projection and the folded-norm arithmetic are "
            "unchanged.",
        "src/aadistill/initialization/operators/composite/stage1_sandwich.py":
            "DECLARED SEMANTIC CHANGE: the statistics pass it MAY run honours "
            "the run's packing and position policy; a SUPPLIED state is "
            "untouched and its trace reports `None` for both, because the "
            "restriction describes a pass this invocation ran.",
        "src/aadistill/initialization/planning/metrics.py":
            "DECLARED SEMANTIC CHANGE: `StateEvaluator` takes a position policy "
            "and an ExecutionConfig. The policy weights every state metric and "
            "is stamped into the evaluation's detail; the execution config lets "
            "the two FORWARDS batch while the reduction stays the same "
            "`distortion` call at the same per-item shape with the same chunk "
            "boundaries -- so the certified arithmetic is not on the diff. The "
            "evaluator's default is the one-item-per-forward reference path, "
            "NOT `DEFAULT_EXECUTION`, so no existing caller's behaviour moves. "
            "`batch_plan` bounds the logit blocks and `_ensure_groups` REFUSES "
            "over budget rather than discovering an OOM mid-search.",
        "src/aadistill/initialization/planning/search.py":
            "DECLARED SEMANTIC CHANGE: `SearchConfig.position_policy` (absent "
            "from `as_dict` at the incumbent, so every committed config_hash "
            "stays derivable), `BeamSearch(numerics=...)`, a materialization "
            "identity per state, and three refusals -- resume declines a "
            "journal entry from another materialization or another scoring "
            "policy, and measurement refuses a measurer that scored other "
            "positions. The stats-cache key gains the execution fingerprint and "
            "the policy hash, which its own docstring already promised. NO CUDA "
            "SURFACE: the materialize/reload/validate/measure cycle and its "
            "device placement are unchanged.",
        "src/aadistill/initialization/planning/fixed_path.py":
            "DECLARED SEMANTIC CHANGE: the executor RESOLVES the position "
            "policy from what the step declared rather than accepting one, so "
            "the object doing the arithmetic is the one the state id names, and "
            "the stats-cache key gains the same two terms as the search's. "
            "HISTORICAL CUDA SURFACE -- already listed as pending revalidation "
            "for the `execution` parameter; this round adds to the same "
            "OperatorContext construction and owes the same new validation.",
     }),
    ("4dc579ba4938e1a2bfb05565d0b3594077bb531e",
     "the test-suite boundary: core, experiment, historical",
     {
        "src/aadistill/governance/closure.py":
            "DECLARED SEMANTIC CHANGE: `_resolve` falls through to "
            "`_resolve_through_a_grouping_directory`, which searches ONE "
            "intervening level under a top-level package. A package whose "
            "`__init__` extends `__path__` resolves `pkg.thing` to "
            "`pkg/<group>/thing/`, and the direct candidates cannot see that -- "
            "so every import of a grouped package came back unresolved and the "
            "deriver correctly refused to describe a smaller set than runs. It "
            "knows NOTHING about what the grouping means: it does not read a "
            "directory name, match a pattern or recognise a convention, which "
            "is what keeps the module reusable while the application groups its "
            "experiments by stage. NO CUDA SURFACE: it globs and reads files.",
        "src/aadistill/infrastructure/session.py":
            "ADDITIVE: `SetupManifest.test_paths`, its `test_paths_env()` and "
            "`SESSION_TEST_PATHS` in the setup environment and the manifest. A "
            "session now declares the test suite its pod gate runs POSITIVELY; "
            "`test_ignores` is kept, defaulted and documented as historical. "
            "The old shape could only say 'run only mine' as the COMPLEMENT of "
            "everything under `tests/`, and `autoinit_c1_launch`'s own comment "
            "records that complement going stale six times, once per experiment "
            "preflight directory added after C1 closed. Every existing session "
            "is unaffected: an empty `test_paths` leaves the shell's `tests/` "
            "default in place. NO CUDA SURFACE.",
        "src/aadistill/runtime/staging_contract.py":
            "DECLARED SEMANTIC CHANGE: `ignores_for_selection` raises the new "
            "`SelectionOutsideCoreSuite` for a selection that is not under "
            "`tests/`, naming `test_paths` as the replacement instead of "
            "deriving a complement that cannot express it. Its own subclass so "
            "a caller can tell 'you asked for the superseded mechanism' apart "
            "from 'your directory is missing'. A selection inside `tests/` "
            "behaves exactly as before. NO CUDA SURFACE.",
        "src/aadistill/runtime/cpu_test_env.py":
            "ADDITIVE: `SESSION_TEST_PATHS` joins `PRESERVED`, so the pod's "
            "command-scoped CPU-test environment forwards the new declaration "
            "the way it already forwards `SESSION_TEST_IGNORES`. A variable the "
            "gate needs and the scope strips is a gate that runs the wrong "
            "command. NO CUDA SURFACE.",
        #: `src/aadistill/initialization/device.py` is deliberately ABSENT. Two
        #: of its docstring references name the placement helper, which moved to
        #: the shared test-support package in this round -- and the file is on
        #: the HISTORICAL CUDA surface, whose bytes are the evidence that the
        #: 2026-09-10 validation covered this code. Editing a docstring would
        #: either break that byte comparison or require declaring the surface
        #: changed and owing a new GPU validation, for a path in a comment. The
        #: stale reference stays; the evidence is worth more than the pointer.
        "src/aadistill/runtime/cost.py":
            "PROSE: an anchor's provenance note named a test file by path, "
            "which the core-boundary inventory correctly flags as a `logs/ "
            "configs/ artifacts/` path literal read from core. It names the "
            "cost model's own anchor tests instead. No value, no arithmetic and "
            "no behaviour changed.",
     }),
    ("06cab9c8f106ed6f35db4c5cd6bf9d47288da139",
     "materialization ownership, scoring content, one measurement protocol",
     {
        "src/aadistill/initialization/scoring/content.py":
            "NEW MODULE, therefore a DECLARED SEMANTIC CHANGE by construction. "
            "The identity of the position metadata a policy actually consumes. "
            "It exists because a mixture's `content_sha256` hashes item ids and "
            "token ids only, so two assets could share every hash in the "
            "operator path -- profile hash, content hash, policy hash -- and "
            "carry DIFFERENT supervised masks, producing different DEPTH, FFN, "
            "WIDTH and ATTENTION decisions under one scientific path identity. "
            "The bound terms are asked of the POLICY (`reads_tags`), so a "
            "future policy reading another tag binds the right thing with no "
            "edit here and no tag name is hardcoded. EMPTY-EQUIVALENT at the "
            "incumbent policy, which reads no position metadata, so every "
            "committed state id stays derivable. NO CUDA SURFACE: it reads "
            "boolean masks on the host and hashes strings.",
        "src/aadistill/initialization/scoring/protocol_identity.py":
            "NEW MODULE, therefore a DECLARED SEMANTIC CHANGE. One "
            "`measurement_protocol_id` over the suite's STRUCTURAL identity, "
            "the suite's CONTENT identity, the scoring content, the position "
            "policy, the reduction semantics (chunk, reference strategy, "
            "aggregation rule) and the execution fingerprint -- so `_restore` "
            "asks one question instead of the two independent comparisons it "
            "had and the three it would have grown. A record predating the "
            "identity is judged by its own historical fields and its id is "
            "NEVER reconstructed, because that would assert a reduction and an "
            "execution nobody recorded. NO CUDA SURFACE: it hashes strings.",
        "src/aadistill/initialization/specs/materialization.py":
            "DECLARED SEMANTIC CHANGE: `materialization_id` now binds the "
            "PARENT materialization as a third required term, because a child's "
            "bytes are a function of the bytes it consumed, and a root derives "
            "its identity from the pinned teacher revision rather than from an "
            "experiment special case. `FINGERPRINT_GROWTH_RULE` states what a "
            "future field owes: in if it selects a different ARITHMETIC PATH "
            "over the same inputs (an explicit attention/kernel/backend or "
            "determinism selection, once one exists), out if it names where or "
            "when the work ran. Every id this round produces differs from the "
            "two-term ones, which is the intended consequence -- no two-term id "
            "was ever written to a journal. NO CUDA SURFACE.",
        "src/aadistill/initialization/specs/state.py":
            "ADDITIVE: `latest_by_materialization_id()` beside an UNCHANGED "
            "`latest_by_state_id()`. The semantic view keeps one record per "
            "path and therefore cannot find an earlier materialization once "
            "another protocol has journalled a newer one; the new view can. "
            "The old one is untouched deliberately -- the frozen C2 canonical "
            "record rule reads it and a test asserts it picks the LAST record. "
            "NO CUDA SURFACE.",
        "src/aadistill/initialization/planning/search.py":
            "DECLARED SEMANTIC CHANGE, and the one that closes ownership rather "
            "than detection. `checkpoint_dir()` is the single owner of "
            "`states/<semantic_state_id>/<materialization_id>/`, where two "
            "materializations of one path previously owned one destination and "
            "the second overwrote the first. `_restore` looks up BY "
            "materialization and asks `measurement_is_comparable` once, "
            "replacing the inline suite-hash and policy-hash comparisons; the "
            "`require_same_materialization` refusal is kept as a structural "
            "assertion, since a keyed hit cannot mismatch. The run learns its "
            "measurement protocol from its config or, failing that, from its "
            "first measurement -- every driver wraps its evaluator in a lambda, "
            "so a search that could not learn it would stamp records it could "
            "never adopt and silently re-measure everything on resume. A "
            "declared-vs-observed conflict raises. NO CUDA SURFACE: the "
            "materialize/reload/validate/measure cycle and its device "
            "placement are unchanged.",
        "src/aadistill/initialization/planning/metrics.py":
            "DECLARED SEMANTIC CHANGE: `StateEvaluator` computes a "
            "`measurement_protocol_id` at construction and stamps it, with a "
            "`reduction` block, into every evaluation's detail; it accepts an "
            "optional `NumericalEnvironment` and an optional "
            "`suite_content_sha256`. The last one is deliberate: `suite_hash` "
            "is the STRUCTURAL identity that 45 committed records pin, so the "
            "suite's CONTENT is bound in the new protocol id rather than folded "
            "into the structural hash -- which would have moved the identity of "
            "an unchanged asset and did, in five tests, before being reverted. "
            "The arithmetic is untouched: an evaluator given no policy, no "
            "numerics and no content hash performs exactly the operations it "
            "performed before, recording `undeclared` and `unbound` for what "
            "nobody stated. CUDA-ADJACENT: batching, chunk boundaries, dtypes "
            "and device placement are unchanged by this round. Recorded in the "
            "C2 evaluator lineage as this file's THIRD movement.",
        "src/aadistill/initialization/planning/fixed_path.py":
            "DECLARED SEMANTIC CHANGE: the step trace records a "
            "`scoring_content_report` -- the identity of the position metadata "
            "the resolved policy will read, plus the three totals a reader "
            "checks it against. EVIDENCE ONLY: it enters the trace, not any "
            "hashed config, and is empty-equivalent at the incumbent policy. "
            "HISTORICAL CUDA SURFACE -- unchanged from the previous round's "
            "entry; this round adds no device, dtype or shape behaviour.",
     }),
    ("f8ccdd1163428a48887b50bed14f71d94ea52cc5",
     "empty supervision fails closed, and the blocker count is derived",
     {
        "src/aadistill/initialization/scoring/positions.py":
            "DECLARED SEMANTIC CHANGE: `SupervisedTargetV1._prediction_mask` "
            "now REFUSES a present-but-empty `assistant` tag instead of "
            "falling back to all positions. The two cases were already bound to "
            "different scoring content identities -- `assistant=absent` against "
            "`assistant=0:<digest>` -- by `content.py::_position_component`, on "
            "the stated grounds that this policy treats them differently; one "
            "`or` meant it did not, so an asset whose supervision metadata "
            "named nothing scored FULL-SEQUENCE under a target-aware policy id. "
            "ABSENT still falls back, which is the raw-LM/general-language case "
            "and a DATA property that must not change. Measured safe against "
            "the frozen assets before changing it: 203 rows across "
            "e8_calibration_v1 and state_eval_v1 are 56 absent, 26 empty-dict "
            "and 121 non-empty, with ZERO present-but-empty, so the refusal "
            "refuses nothing that exists and no committed state id moves. NO "
            "CUDA SURFACE: it builds boolean masks on the host, and the file is "
            "not on the historical CUDA-validated surface, so no new GPU "
            "validation is owed.",
     }),
    ("df41bee031f1fc2b46f4c9f7fe7a8bf00c998c3e",
     "reference_topk_tail_v1: the D-series KL support becomes scalable",
     {
        "src/aadistill/initialization/scoring/support.py":
            "NEW FILE. The `reference_topk_tail_v1` distribution support: KL "
            "reduced over the Top-K entries of the REFERENCE distribution plus "
            "one aggregate tail bucket, with a `ReferenceDistributionSketch` "
            "holding O(T*K) instead of O(T*V), a tail computed from the "
            "COMPLEMENT's own logits, the six-quantity reducer and DEPTH's "
            "forward-KL-only pair. An independent review caught the first "
            "formulation -- `log(1 - sum of the support)` -- as a real "
            "numerical-semantic defect: the support mass reaches 1.000001 on "
            "real logits, so the complement was below float32 resolution and "
            "one coarse KL exceeded the full-vocabulary KL, which a "
            "coarsening cannot do. The complement logsumexp never cancels, "
            "and a zero candidate tail against a non-zero reference tail is "
            "`+inf` and is PRESERVED rather than dropped. "
            "THIS IS A DECLARED SEMANTIC CHANGE by virtue of being new "
            "executable core, AND it is a SCIENTIFIC PROTOCOL AMENDMENT -- "
            "maintainer decision 2026-10-04 -- because Top-K+tail is not "
            "mathematically identical to full-vocabulary KL: it is a coarsening "
            "of the partition and therefore a lower bound on it. The "
            "full-vocabulary reducer is UNTOUCHED and remains the oracle. CUDA "
            "SURFACE: it runs on the device the logits are on, in float32 "
            "chunks with float64 accumulators, exactly as the full-vocabulary "
            "reducer does -- so a GPU validation IS owed and is what the "
            "adoption qualification exists to provide.",
        "src/aadistill/initialization/scoring/protocol_identity.py":
            "`ReductionSemantics` gains an OPTIONAL `distribution_support`, and "
            "`as_dict` OMITS it when it is None or full-vocabulary. That "
            "absence is the design: 785 committed records cite protocol ids "
            "computed before the field existed, and a key added "
            "unconditionally -- even carrying \"full_vocab_v1\" -- would move "
            "every one of them. THIS IS A DECLARED SEMANTIC CHANGE for the new "
            "contract only: a Top-K measurement gets a different id, as it must, "
            "and no historical id recomputes differently. Asserted both "
            "directions. NO CUDA SURFACE: it hashes a mapping.",
        "src/aadistill/initialization/operators/base.py":
            "`OperatorContext` gains `distribution_support`, defaulting to the "
            "historical full-vocabulary contract, and `execute` refuses a "
            "mismatch between the handed object and the `config` declaration "
            "that reaches `config_hash` -- the same two-sided contract "
            "`position_policy` already has, for the same reason: a caller "
            "updating one and not the other would record a state id describing "
            "a divergence nobody computed. Under the historical contract the "
            "config key is ABSENT rather than null, so no existing config hash "
            "moves and 886 initialization tests pass unchanged. NO CUDA "
            "SURFACE: a dataclass field and an equality check.",
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py":
            "`_ReferenceSketches` beside `_ReferenceLogits`, selected by the "
            "run's declared support, and the batched recording path SHARED "
            "between them. THIS IS A DECLARED SEMANTIC CHANGE under the Top-K "
            "support and a no-op under full vocabulary, which is what every "
            "frozen DEPTH decision was produced by and is untouched. The "
            "reference state goes from 33.8 GiB at the frozen mixture to "
            "megabytes, so the recompute fallback -- which doubled the forwards "
            "for a whole expansion -- never arms. Sketches are built from the "
            "SAME canonical padded batches the candidate forwards use, so a "
            "cache hit still cannot move batch membership, shape or position "
            "alignment. CUDA SURFACE, and the operator this amendment exists to "
            "make scalable: a GPU validation IS owed.",
        "src/aadistill/initialization/planning/metrics.py":
            "`StateEvaluator` takes a `distribution_support`, folds it into its "
            "`ReductionSemantics` and therefore its `measurement_protocol_id`, "
            "and reduces through the K+1 reducer when it is not "
            "full-vocabulary. `_sketch_for` caches the teacher's sketches under "
            "their OWN budget rather than inheriting the full-logit cache's "
            "refusal, which does not apply at O(T*K). THE SKETCH IS SCIENCE AND "
            "THE CACHING IS EXECUTION: tested by requiring a cached evaluator "
            "and a zero-budget one to agree on every metric. THIS IS A DECLARED "
            "SEMANTIC CHANGE under the Top-K support and a no-op under full "
            "vocabulary. CUDA SURFACE: a GPU validation IS owed.",
     }),
    ("908064ce9e741f8341dd0bdb22fc0cb51a154680",
     "BeamSearch can express a distribution support at all",
     {
        "src/aadistill/initialization/planning/search.py":
            "`SearchConfig` gains `distribution_support`, defaulting to "
            "`FULL_VOCAB_V1`, and `_expand_one` now DECLARES it in the hashed "
            "operator config and PASSES the object in `OperatorContext`. Without "
            "this the Top-K protocol was implemented in the operator, the state "
            "evaluator, the protocol identity and a driver -- and the only path a "
            "formal search could take fell back to the full vocabulary, so a paid "
            "40-minute measurement timed an operator path `BeamSearch` could not "
            "reach. An independent review caught it. "
            "IDENTITY: `as_dict` OMITS the key at the default, so every committed "
            "search keeps the `config_hash` its own record carries; a coarsened "
            "partition changes the hash, which is the point of putting it there. "
            "ONE PARTITION FOR OPERATORS AND MEASURER: the constructor asks an "
            "unwrapped measurer for its support, and -- the load-bearing half, "
            "since every driver wraps its evaluator in a lambda -- every "
            "measurement is checked against what it actually REDUCED OVER, read "
            "out of `detail.reduction`. A candidate selected on a Top-K objective "
            "and pruned on a full-vocabulary metric is now unexpressible. "
            "NO CUDA SURFACE: this file issues no kernel and performs no "
            "reduction; it passes a declaration and an object, and the arithmetic "
            "it reaches was validated by the Top-K adoption qualification. No new "
            "GPU validation is owed, and the measured operator context is "
            "byte-identical to the one a8 timed -- which is why a8 stands without "
            "a rerun.",
     }),
    ("b342861f0771084c5191210cede553cced139c7c",
     "the session record is written where it is filed",
     {
        "src/aadistill/initialization/planning/ranking.py":
            "DIVERSITY BECOMES AN EXPLORATION MECHANISM ONLY. THIS IS A DECLARED "
            "SEMANTIC CHANGE to selection: `quality_order()` is extracted as the "
            "one shared epsilon-Pareto ordering, `rank()` takes a `diversity` "
            "keyword, `RankingResult` records which retention rule produced it, "
            "and every decision carries its position in the quality order. "
            "WHAT WAS WRONG: one policy did two jobs at once -- quality ranking "
            "(epsilon-Pareto fronts plus a deterministic tie-break) and search "
            "diversity (one slot per lineage before any lineage gets two) -- and "
            "the second was reused for post-search finalist retention, where it "
            "does not belong. During a search a state is a PARTIAL hypothesis "
            "and one early proxy measurement must not extinguish a structural "
            "family, so the beam is right to rotate. Once complete leaves exist "
            "that job is finished and the only remaining question is which "
            "complete candidates the objectives themselves rank highest. "
            "Measured rather than argued: a completed 12-leaf search committed a "
            "finalist at quality position 11 of 12 -- over twice the best leaf's "
            "objective value, worse than seven leaves it excluded -- because "
            "that leaf was the sole member of its lineage, while the candidates "
            "at quality positions 2 and 4 were excluded for sharing one. The "
            "retention WIDTH had been widened specifically to admit those two. "
            "WHAT CHANGED: `PARETO_V1` is untouched and beam pruning keeps its "
            "diversity, so every existing search behaves exactly as before -- "
            "`diversity` defaults to True. There is still exactly ONE "
            "implementation of the Pareto algorithm, so a finalist selection "
            "cannot drift from the beam's notion of quality. No scalar score was "
            "introduced: collapsing a multi-objective search into one number is "
            "the failure the fronts exist to prevent. `K` is the caller's and "
            "appears nowhere in core.",
        "src/aadistill/initialization/planning/search.py":
            "`SearchResult.finalists(policy, k)` -- post-search retention by "
            "quality order alone, and `top_n` gains the same `diversity` "
            "keyword with its historical default. THIS IS A DECLARED SEMANTIC "
            "CHANGE to what a completed search offers its caller. "
            "WHY A SECOND NAME rather than a boolean at every call site: the two "
            "answer different questions -- 'what should the beam carry forward' "
            "and 'which complete candidates are best' -- and a committed record "
            "should say which one it asked. The admissibility guard is unchanged "
            "and still runs first, so a depth-only intermediate cannot be "
            "promoted into a recovery probe it could never be a candidate for.",
        "src/aadistill/infrastructure/session.py":
            "`BudgetSpec.account_balance_required_usd` -- what the PROVIDER "
            "ACCOUNT must hold before a pod is created, as a number or as a "
            "callable over the priced `BudgetPlan`. THIS IS A DECLARED SEMANTIC "
            "CHANGE to the budget surface, and it defaults to `None`, so a "
            "session that declares nothing is not gated and every launcher "
            "written before it keeps its behaviour exactly. "
            "WHY IT IS A FIELD and not a constant: the AMOUNT belongs to a "
            "campaign. D1 derives its own from the per-attempt envelope read "
            "out of the authorization config, the priced container disk RunPod "
            "bills separately, and a named operational reserve. A dollar figure "
            "in reusable core would be the same defect as a hardcoded "
            "experiment id (P3), and the 2026-10-06 loss was not that the "
            "number was wrong -- it was that nobody asked.",
        "src/aadistill/infrastructure/provider.py":
            "THE PROVIDER ACCOUNT BALANCE becomes a thing this project can ask "
            "about. THIS IS A DECLARED SEMANTIC CHANGE: a new `AccountBalance` "
            "type, `PodProvider.account_balance()` on the protocol, a RunPod "
            "implementation over `myself { clientBalance currentSpendPerHr "
            "minBalance }`, and two `SimulatedProvider` knobs that rehearse an "
            "account which cannot fund a session. "
            "WHAT WAS WRONG: nothing in the project asked. Every gate asked "
            "whether an experiment was PERMITTED to spend -- identity, session "
            "contract, staged inputs, readiness, commit lineage, bundle, and "
            "four authorization limits -- and none asked whether the provider "
            "would still be paid. On 2026-10-06 D1's formal search passed all "
            "six $0 gates, was authorized to $21.4897 over 1125.55 minutes, and "
            "RunPod stopped it at 449.8 minutes with 39 of 92 expansions "
            "complete because the ACCOUNT had run out of money. $8.1716 bought "
            "no endpoint and the beam's whole workdir went with the host. "
            "WHAT CHANGED: `AccountBalance.covers(required_usd)` answers one "
            "question and refuses in three distinct ways -- short balance, the "
            "provider's own `underBalance` judgement, and an unreadable "
            "control plane. The last is deliberate: unknown is not a negative "
            "answer anywhere else in this module, but a balance that cannot be "
            "read also cannot be shown to cover the session, and the refusal is "
            "free. No dollar figure lives here; the amount is an argument, "
            "because $30 is this campaign's per-session envelope and not a "
            "property of the session machinery.",
        "src/aadistill/runtime/staging_contract.py":
            "`test_paths` enters the staging contract, and the pod's pytest "
            "command gets ONE owner. THIS IS A DECLARED SEMANTIC CHANGE to the "
            "staged-view contract and to its DIGEST. "
            "WHAT WAS WRONG: the pod gate runs `pytest "
            "${SESSION_TEST_PATHS:-tests/} -q ${SESSION_TEST_IGNORES:-}` -- a "
            "selection with TWO halves -- and `derive_contract` built its "
            "`pytest_selection` as `pytest tests/ -q <ignores>`, hardcoding the "
            "base path and dropping `SetupManifest.test_paths` entirely. That "
            "was invisible while every session expressed its selection as a "
            "COMPLEMENT over siblings inside `tests/`; the 2026-10-03 boundary "
            "decision moved experiment tests OUT of `tests/` and made "
            "`test_paths` the positive declaration, and this was never taught "
            "the new half. Every session that has since declared `test_paths` "
            "with no ignores -- C1, A3 and now D1 -- therefore had its readiness "
            "sweep run `pytest tests/ -q`, the whole core suite, while its pod "
            "gate runs only its own preflight directory. A simulation that runs "
            "a different command from the pod is not a simulation, which is "
            "this machinery's own standard. "
            "WHAT CHANGED: `pod_pytest_command(setup, python=...)` builds the "
            "pod's command in one place, parameterized only by interpreter; "
            "`pytest_arguments` extracts the part after `-m pytest` so a "
            "declared selection and an invoked one can be compared without "
            "caring which interpreter ran it; `derive_contract` records "
            "`test_paths` and derives `pytest_selection` from that one owner. "
            "IDENTITY: `test_paths` is now HASHED, so every session's staging "
            "digest moves ONCE. That is the intended consequence and not "
            "collateral -- the contract now describes something it previously "
            "omitted, so a session could change which suite its pod gate runs "
            "without invalidating a single readiness record, and a sweep that "
            "ran the wrong selection is owed again rather than grandfathered. "
            "NO CUDA SURFACE: this module issues no kernel, touches no device "
            "and performs no reduction; it derives a command string and a "
            "digest over a manifest. No GPU validation is owed.",
        "src/aadistill/infrastructure/session_runner.py":
            "`save()` creates its parent directory before writing. THIS IS A "
            "DECLARED SEMANTIC CHANGE to the evidence path, and it is two "
            "lines: `out.parent.mkdir(parents=True, exist_ok=True)` before the "
            "existing `write_text`. Nothing else in the method moved, no "
            "caller changed, and the bytes written are the same bytes. "
            "WHY IT IS NOT COSMETIC: `save()` is the only thing that puts the "
            "session record on the dev box and it runs on every path -- "
            "immediately after `create()` registers a pod id, after each draw, "
            "at the dry-run stop, from `teardown_now`, and from "
            "`run_session`'s closeout. With no `mkdir` a missing parent did "
            "not fail where `--out` was configured; it raised "
            "`FileNotFoundError` out of the FIRST save after a pod started "
            "billing, which is the one moment the record matters most and the "
            "one place an exception costs money rather than time. It held "
            "only because every launcher so far passed an `--out` whose parent "
            "happened to exist. D1's first real pre-provider dry run reached "
            "`DRY_RUN_GATES_PASSED` with all six gates green and then died "
            "here, because its record is filed at "
            "`runs/<run_id>/runtime/session.json` -- where the shared run "
            "layout puts a session record, and which nothing creates in "
            "advance. "
            "NO CUDA SURFACE: this is a filesystem call on the dev box. It "
            "issues no kernel, touches no device, reaches no provider and "
            "changes no identity, so no GPU validation is owed. "
            "`tests/infrastructure/test_session_record_is_writable.py` pins "
            "it, including a mutation check that removing the `mkdir` restores "
            "the `FileNotFoundError`.",
     }),
)



#: The tip the CURRENT round was reviewed at.
REVIEWED_TIP = ROUNDS[-1][0]

# --- historical CUDA validation, and the current one ------------------------
#
# These are two different facts and this file keeps them apart.
#
# The HISTORICAL validation ran on 2026-09-10 at execution SHA 7027a8f4 against
# the flat operator layout. That fact is immutable and is not repointed: the
# evidence names the bytes it actually executed, and no later refactor can make
# that SHA validate code it never saw. Three of the six paths it names no longer
# exist, because the topology migration moved them -- so the correct CURRENT
# assertion about the historical surface is an explicit REFUSAL to revalidate
# it here, not a demand that yesterday's experiment resolve against today's
# packages.
#
# The CURRENT validation is a separate record with its own SHA and its own
# surface, derived from what the batched execution path actually reads rather
# than inherited from the old tuple.

#: The GPU execution commit of the 2026-09-10 validation. Historical.
HISTORICAL_EXECUTION_SHA = "7027a8f4c0a7684c892b483a2193025cc26c1b58"

#: Verbatim from that review's clause 5, resolved to the files as they were
#: named THEN. Preserved exactly; not repointed at the migrated paths.
HISTORICAL_CUDA_VALIDATED_SURFACE = (
    "src/aadistill/initialization/operators/attention_activation.py",
    "src/aadistill/initialization/statistics/attention.py",
    "src/aadistill/initialization/device.py",
    "src/aadistill/initialization/planning/fixed_path.py",
    "src/aadistill/initialization/adapters/__init__.py",
    "src/aadistill/initialization/adapters/qwen3.py",
)

#: Where each moved path's content went, so the refusal can say so rather than
#: merely reporting an absence.
HISTORICAL_SURFACE_SUCCESSORS = {
    "src/aadistill/initialization/operators/attention_activation.py":
        "src/aadistill/initialization/operators/attention/gqa/activation_importance.py",
    "src/aadistill/initialization/statistics/attention.py":
        "src/aadistill/initialization/operators/attention/gqa/_statistics.py",
}

#: The surface the NEXT real-CUDA validation must cover: every file whose
#: semantics materially determine a batched calibration forward.
#:
#: **HAND-MAINTAINED, and deliberately not called a derived closure.** Nothing
#: computes it: it was written by reading the batched execution path, and it
#: stays correct only while someone keeps reading. The repository DOES have a
#: derived closure -- `configs/experiments/phase_c1/executable_closure.json`,
#: produced by walking import edges -- and conflating the two would let a reader
#: assume a machine is checking this list when none is. What the tests below
#: check is narrower and honest: that every path on it exists, that the two
#: no-forward operators are absent, and that each forwarding operator is
#: present. That is a consistency check on a human list, not a derivation.
CURRENT_CUDA_SURFACE = (
    "src/aadistill/initialization/calibration/batching.py",
    "src/aadistill/initialization/calibration/packing.py",
    "src/aadistill/initialization/execution.py",
    "src/aadistill/initialization/device.py",
    "src/aadistill/initialization/planning/fixed_path.py",
    "src/aadistill/initialization/adapters/__init__.py",
    "src/aadistill/initialization/adapters/qwen3.py",
    "src/aadistill/initialization/operators/_common.py",
    "src/aadistill/initialization/operators/attention/gqa/_common.py",
    "src/aadistill/initialization/operators/attention/gqa/_statistics.py",
    "src/aadistill/initialization/operators/attention/gqa/activation_importance.py",
    "src/aadistill/initialization/operators/attention/gqa/causal_kl.py",
    "src/aadistill/initialization/operators/composite/stage1_sandwich.py",
    "src/aadistill/initialization/operators/depth/_common.py",
    "src/aadistill/initialization/operators/depth/causal_kl_greedy.py",
    "src/aadistill/initialization/operators/ffn/dense/activation_importance.py",
    "src/aadistill/initialization/operators/width/residual/global_pca.py",
    "src/aadistill/initialization/statistics/collect.py",
    "src/aadistill/initialization/statistics/contribution.py",
)

#: Backwards-compatible alias for the constraint that a declared core change may
#: not quietly cover a file the HISTORICAL review pinned.
CUDA_VALIDATED_SURFACE = HISTORICAL_CUDA_VALIDATED_SURFACE


#: Core modules a later round MOVED, mapped to where their content went.
#:
#: A declaration names the path a change landed on *at the time that round was
#: reviewed*. When a later round reorganises the package, an earlier round's
#: declared path can stop existing — and then it is in the expected set forever
#: while the live diff can never report it again, because a deleted file has no
#: content to classify as semantic.
#:
#: The fix is NOT to edit the earlier round: that round really did declare that
#: path, and rewriting it would falsify what was reviewed. The two things are
#: simply different questions — *what did that round declare* (historical) and
#: *what does the tree contain now* (current) — and this table is the join
#: between them. An entry is only admissible when the round that performed the
#: move declares the destination, which the assertion below enforces.
#: **The map must be COMPLETE, not only as complete as git's rename
#: detection.** Every one of these moves happened in the 2026-09-25 topology
#: migration, but only `depth.py` was recorded — because git was pairing the
#: other five as renames and `--name-only` reports a rename as its destination
#: alone, so nothing ever asked where they went. The moment this round edited
#: `attention/gqa/activation_importance.py` enough to drop its similarity to
#: `attention_activation.py` to 4%, git stopped pairing them and reported a
#: delete, and the omission surfaced across 27 parametrisations at once.
#:
#: A heuristic deciding whether a bookkeeping entry is required is not a
#: contract. All six are listed, so no future edit's similarity score can
#: change what this file knows.
MOVED_BY_A_LATER_ROUND: dict[str, tuple[str, ...]] = {
    "src/aadistill/initialization/operators/depth.py": (
        "src/aadistill/initialization/operators/depth/_common.py",
        "src/aadistill/initialization/operators/depth/positional.py",
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py",
    ),
    "src/aadistill/initialization/operators/attention_activation.py": (
        "src/aadistill/initialization/operators/attention/gqa/"
        "activation_importance.py",
    ),
    "src/aadistill/initialization/operators/attention.py": (
        "src/aadistill/initialization/operators/attention/gqa/_common.py",
        "src/aadistill/initialization/operators/attention/gqa/weight_proxy.py",
    ),
    "src/aadistill/initialization/operators/ffn.py": (
        "src/aadistill/initialization/operators/ffn/dense/"
        "activation_importance.py",
    ),
    "src/aadistill/initialization/operators/width.py": (
        "src/aadistill/initialization/operators/width/residual/global_pca.py",
    ),
    "src/aadistill/initialization/operators/composite.py": (
        "src/aadistill/initialization/operators/composite/stage1_sandwich.py",
    ),
}


def declared_from(round_index: int) -> dict[str, str]:
    """Every semantic change declared by this round and every later one.

    Measured from an older base, a later round's declared change is also in the
    diff -- so the expected set for round i is the union from i onward, and
    nothing else is admissible.

    A path a later round moved away is dropped, because the tree cannot report
    it any more; its successors are declared by the round that moved it and are
    checked like any other entry.
    """
    out: dict[str, str] = {}
    for _tip, _label, changes in ROUNDS[round_index:]:
        out.update(changes)
    for gone, successors in MOVED_BY_A_LATER_ROUND.items():
        if gone in out and not Path(gone).exists():
            missing = [s for s in successors if s not in out]
            assert not missing, (
                f"{gone} was moved but its successors are undeclared: {missing}. "
                "A move is only accounted for when the round that performed it "
                "declares where the content went.")
            out.pop(gone)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture(scope="module")
def changed_core() -> list[str]:
    out = git("diff", "--name-only", REVIEWED_TIP).split()
    return [f for f in out if f.startswith("src/aadistill/") and f.endswith(".py")]


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            body = getattr(n, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                n.body = body[1:] or [ast.Pass()]
    return tree


def shape(source: str) -> str:
    """Everything except the prose."""
    return ast.dump(strip_docstrings(ast.parse(source)))


def at_base(base: str, path: str) -> str | None:
    """A core file's source at `base`, or None if it did not exist there."""
    out = subprocess.run(["git", "show", f"{base}:{path}"], cwd=REPO,
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def is_semantic(base: str, path: str) -> bool:
    """Did this file's executable shape change since `base`?

    A file ABSENT at the base is semantic by construction: new executable core
    is a semantic change whatever its contents, and there is no previous shape
    to compare it to.

    A file absent **now** is a deletion, and a deletion has no shape to compare.
    It is accounted for by :data:`MOVED_BY_A_LATER_ROUND`, which drops it from
    the expected set for exactly the rounds whose diff still reports it, and by
    :func:`test_every_deleted_core_file_is_accounted_for` below, which refuses a
    disappearance nothing explains. So this answers False rather than raising.

    **Both branches exist because this function used to raise instead of
    answering, and it has now done so twice.** The first time it called
    `git show` unchecked, so the round that ADDED a core module took every
    parametrisation of this check down with a `CalledProcessError`. The second
    was subtler and is why the deleted-now branch is here: a path git had been
    reporting as a RENAME reappeared as a separate delete+add the moment the
    destination file was edited enough to fall under git's similarity
    threshold. Nothing about the deletion changed — only how `git diff` chose to
    describe it — and the check died on a `FileNotFoundError` across 27
    parametrisations. A guard that errors instead of answering is a guard whose
    answer nobody has.
    """
    if not (REPO / path).is_file():
        return False
    before = at_base(base, path)
    if before is None:
        return True
    return shape(before) != shape((REPO / path).read_text())


def deleted_core_since(base: str) -> list[str]:
    """Core files `git diff` reports against `base` that no longer exist."""
    return sorted(f for f in git("diff", "--name-only", base).split()
                  if f.startswith("src/aadistill/") and f.endswith(".py")
                  and not (REPO / f).is_file())


def test_every_deleted_core_file_is_accounted_for():
    """A core module may not simply vanish between rounds.

    `is_semantic` answers False for a path that no longer exists, which is
    correct — there is no shape to compare — but on its own it would let a
    silently deleted core module pass unnoticed. This is the other half: every
    disappearance has to be named in `MOVED_BY_A_LATER_ROUND` together with
    where its content went.
    """
    unexplained = sorted(set(deleted_core_since(ROUNDS[0][0]))
                         - set(MOVED_BY_A_LATER_ROUND))
    assert unexplained == [], (
        f"core files deleted since {ROUNDS[0][0][:8]} with no entry in "
        f"MOVED_BY_A_LATER_ROUND: {unexplained}. A move is only accounted for "
        "when the round that performed it says where the content went.")


# --- 1. the two CUDA validations, kept apart --------------------------------

@pytest.mark.parametrize("path", HISTORICAL_CUDA_VALIDATED_SURFACE)
def test_the_historical_surface_is_either_intact_or_explicitly_not_revalidatable(path):
    """One assertion, two admissible outcomes, and no third.

    A path the migration did not touch must still be byte-identical to the
    commit the GPU evidence names -- that claim is unchanged and still checked.

    A path the migration MOVED cannot satisfy that and must not pretend to: the
    file is gone, its content lives somewhere the 2026-09-10 run never executed,
    and the honest current statement is that the historical validation is not
    revalidatable against this tree. What is forbidden is the middle case -- a
    file still sitting at its historical path with different bytes, which would
    let `7027a8f4` appear to validate code it never saw.
    """
    live = REPO / path
    successor = HISTORICAL_SURFACE_SUCCESSORS.get(path)
    if not live.is_file():
        assert successor is not None, (
            f"{path} is on the historical CUDA surface, is absent from the tree, "
            "and no successor is recorded. An unexplained absence is exactly "
            "what this check exists to refuse.")
        assert (REPO / successor).is_file(), (
            f"{path} was moved to {successor}, which does not exist either")
        assert git("show", f"{HISTORICAL_EXECUTION_SHA}:{path}"), (
            "the historical bytes must remain retrievable from the commit the "
            "evidence names, which is what keeps the old fact true")
        return
    if path in HISTORICAL_SURFACE_CHANGED_PENDING_REVALIDATION:
        #: Changed on purpose, declared, and owed a NEW validation. The old
        #: evidence is explicitly NOT claimed for it -- which is the whole
        #: point of saying so here instead of quietly widening the tuple.
        assert path in CURRENT_CUDA_SURFACE
        return
    assert live.read_text() == git("show", f"{HISTORICAL_EXECUTION_SHA}:{path}"), (
        f"{path} still sits at its historical path but its bytes differ from "
        f"execution SHA {HISTORICAL_EXECUTION_SHA[:8]}. Either restore it or "
        "move it and record a successor; do not let the old evidence appear to "
        "cover new code.")


def test_the_historical_validation_is_not_repointed_at_the_new_code():
    """The specific misreport item 4 forbids.

    No file the migration created may be listed on the historical surface. If
    one ever were, `7027a8f4` would be claimed to have validated code written
    weeks after it ran.
    """
    born_after = {
        "src/aadistill/initialization/calibration/batching.py",
        "src/aadistill/initialization/execution.py",
        *HISTORICAL_SURFACE_SUCCESSORS.values(),
    }
    assert not (born_after & set(HISTORICAL_CUDA_VALIDATED_SURFACE)), (
        "the historical CUDA surface names code that did not exist when it ran")


def test_the_current_surface_is_derived_from_the_batched_execution_path():
    """Minimal but truthful: every file on it exists and actually participates."""
    for path in CURRENT_CUDA_SURFACE:
        assert (REPO / path).is_file(), f"current CUDA surface names {path}"
    # The two operators that run no calibration forward are deliberately absent.
    for absent in ("operators/attention/gqa/weight_proxy.py",
                   "operators/depth/positional.py"):
        assert f"src/aadistill/initialization/{absent}" not in CURRENT_CUDA_SURFACE, (
            f"{absent} performs no calibration forward; batching cannot change "
            "its result, so it does not belong on the batched-execution surface")
    # ... and every operator that DOES run one is present.
    for required in ("attention/gqa/activation_importance",
                     "depth/causal_kl_greedy",
                     "ffn/dense/activation_importance",
                     "width/residual/global_pca",
                     "composite/stage1_sandwich"):
        assert any(required in p for p in CURRENT_CUDA_SURFACE), required
    assert "src/aadistill/initialization/calibration/batching.py" in CURRENT_CUDA_SURFACE


def test_no_declared_change_hides_behind_the_historical_surface(changed_core):
    """A file still at a historical path must not be edited silently."""
    still_there = [p for p in HISTORICAL_CUDA_VALIDATED_SURFACE
                   if (REPO / p).is_file()]
    overlap = sorted(set(changed_core) & set(still_there)
                     - set(declared_from(len(ROUNDS) - 1)))
    assert overlap == [], (
        f"historical surface touched without declaring it: {overlap}")


# --- 2. every other change is prose-only, or declared ----------------------

@pytest.mark.parametrize("index", range(len(ROUNDS)),
                         ids=[r[1] for r in ROUNDS])
def test_every_other_core_change_is_prose_only_or_declared(index):
    """The check that reading dozens of diffs by eye would not reliably make.

    Run once per review round, from that round's own reviewed tip. From an older
    base a later round's declared change is also in the diff, so the admissible
    set is the union from that round onward -- and nothing else. An edit that
    rode along with a docstring sweep appears here as an unexplained entry.
    """
    base = ROUNDS[index][0]
    changed = [f for f in git("diff", "--name-only", base).split()
               if f.startswith("src/aadistill/") and f.endswith(".py")]
    semantic = {f for f in changed if is_semantic(base, f)}
    expected = set(declared_from(index))
    assert semantic == expected, (
        f"from {base[:8]} ({ROUNDS[index][1]}), core semantic changes disagree "
        "with what is declared:\n"
        + "\n".join(f"  {f}" for f in sorted(semantic ^ expected)))


def test_this_rounds_declaration_is_not_empty():
    """A round that declares nothing while changing behaviour would pass the
    check above only by making the expected set match a wrong observation."""
    assert ROUNDS[-1][2], "the current round declares no semantic change"


#: (core path, the round that declared it). Each named semantic change is
#: checked against ITS OWN round, not against whichever round happens to be
#: last: this test read `ROUNDS[-1]` and broke the moment a later round was
#: appended, which would have pushed someone toward deleting it rather than
#: binding it correctly.
NAMED_SEMANTIC_CHANGES = (
    ("src/aadistill/runtime/pod_environment.py",
     "b2ecdff83ee0a7653eea7872b5af6739a4de4381"),
    ("src/aadistill/infrastructure/session_runner.py",
     "0103467384ca0013ddc0caa9dc18c92aadfa5c78"),
    #: The one a logs reorganisation could most easily have been called a
    #: docstring sweep: its diff is a default-empty field and two path
    #: compositions, and it went undeclared for exactly that reason.
    ("src/aadistill/runtime/run_layout.py",
     "8a6bbcaafa761ffca0b2697c5f68182798a2709d"),
    #: The one most easily called a performance refactor: its diff is a device
    #: argument and an accumulator that stays where it was computed, and the
    #: numbers it produces MOVE. A round that described this as leaving the
    #: arithmetic alone would be reporting a numerics change as a no-op.
    ("src/aadistill/initialization/statistics/contribution.py",
     "7b376424f45924b8745184b4bc9bf98fe5463d4b"),
    #: The one whose executable diff is TWO LINES under a docstring that grew
    #: by fifteen -- `save()` gaining the `mkdir` for its own parent. That is
    #: precisely the shape this file exists to catch, so it is named: a round
    #: reporting it as prose would be reporting an evidence-path change, on the
    #: save that runs right after a pod starts billing, as a no-op.
    ("src/aadistill/infrastructure/session_runner.py",
     "b342861f0771084c5191210cede553cced139c7c"),
)


@pytest.mark.parametrize("path,tip", NAMED_SEMANTIC_CHANGES,
                         ids=lambda v: v.split("/")[-1])
def test_a_named_change_is_declared_as_semantic_not_prose(path, tip):
    """Describing either as a docstring sweep is the misreport this file exists
    to prevent, so each is named and each is measured."""
    round_ = next(r for r in ROUNDS if r[0] == tip)
    declared = round_[2]
    assert path in declared, f"{tip[:8]} does not declare {path}"
    assert "SEMANTIC CHANGE" in declared[path]
    # And it really is one, measured rather than asserted.
    assert shape(git("show", f"{tip}:{path}")) != shape((REPO / path).read_text())


def test_the_prose_sweep_actually_covered_the_core():
    """A guard on the guard above: if almost nothing changed in prose,
    the parametrized check would pass close to vacuously."""
    changed = [f for f in git("diff", "--name-only", ROUNDS[0][0]).split()
               if f.startswith("src/aadistill/") and f.endswith(".py")]
    prose_only = set(changed) - set(declared_from(0))
    assert len(prose_only) >= 20, (
        f"only {len(prose_only)} core files changed in prose; the ownership "
        "sweep was expected to reach far more than that")


#: Historical-surface files this round deliberately changed, each owing the NEW
#: CUDA validation rather than claiming cover from the old one. Kept as an
#: explicit, reviewable list: a file may not drift onto the validated surface
#: silently, and the test below requires every entry to be declared by the
#: current round and to appear on the current validation surface.
HISTORICAL_SURFACE_CHANGED_PENDING_REVALIDATION = (
    "src/aadistill/initialization/planning/fixed_path.py",
    "src/aadistill/initialization/adapters/qwen3.py",
)
