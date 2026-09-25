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
            "`tests/pod/test_phase_c2_setup_contract.py` -- and it REFUSES a "
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
            "`tests/pod/test_phase_c2_collection_and_profiles.py` holds that "
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
#: semantics materially determine a batched calibration forward. Derived from
#: the execution path, not inherited -- `batching.py` and the topology-local
#: statistics module are here because the batched result depends on them, and
#: `weight_proxy` / `positional` are absent because they run no forward.
CURRENT_CUDA_SURFACE = (
    "src/aadistill/initialization/calibration/batching.py",
    "src/aadistill/initialization/execution.py",
    "src/aadistill/initialization/device.py",
    "src/aadistill/initialization/planning/fixed_path.py",
    "src/aadistill/initialization/adapters/__init__.py",
    "src/aadistill/initialization/adapters/qwen3.py",
    "src/aadistill/initialization/operators/_common.py",
    "src/aadistill/initialization/operators/attention/gqa/_common.py",
    "src/aadistill/initialization/operators/attention/gqa/_statistics.py",
    "src/aadistill/initialization/operators/attention/gqa/activation_importance.py",
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
MOVED_BY_A_LATER_ROUND: dict[str, tuple[str, ...]] = {
    "src/aadistill/initialization/operators/depth.py": (
        "src/aadistill/initialization/operators/depth/_common.py",
        "src/aadistill/initialization/operators/depth/positional.py",
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py",
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

    This used to call `git show` unchecked, so the first round that ADDED a core
    module did not report an undeclared change — it raised
    `CalledProcessError` out of the set comprehension and took every
    parametrisation of this check down with it. A guard that errors instead of
    answering is a guard whose answer nobody has.
    """
    before = at_base(base, path)
    if before is None:
        return True
    return shape(before) != shape((REPO / path).read_text())


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
)


def test_no_declared_change_touches_the_validated_surface():
    """A declared change may not silently cover a historically pinned file.

    The exception is narrow and named. `fixed_path.py` is on the 2026-09-10
    surface AND had to gain the `execution` parameter, because that file is
    where the fixed-path route builds an `OperatorContext`. That does not make
    the old evidence cover it — it makes a NEW validation owed, which is why
    every exception must also be on `CURRENT_CUDA_SURFACE`.
    """
    touched = set(declared_from(0)) & set(HISTORICAL_CUDA_VALIDATED_SURFACE)
    unexplained = sorted(touched - set(HISTORICAL_SURFACE_CHANGED_PENDING_REVALIDATION))
    assert unexplained == [], (
        "declared core changes touch the historical CUDA surface without being "
        f"listed as pending revalidation: {unexplained}")
    for path in HISTORICAL_SURFACE_CHANGED_PENDING_REVALIDATION:
        assert path in declared_from(len(ROUNDS) - 1), (
            f"{path} claims a revalidation exception but the current round does "
            "not declare it")
        assert path in CURRENT_CUDA_SURFACE, (
            f"{path} changed under a historical CUDA pin and is not on the "
            "current validation surface, so nothing would ever re-validate it")


# --- the geometries the validation ran are unchanged too --------------------

def test_the_two_validation_geometries_are_unchanged():
    """`suffix_narrow` and `suffix_mid`, from the config the check reads."""
    cfg = "configs/validation/cuda_engineering.json"
    assert (REPO / cfg).read_text() == git("show", f"{HISTORICAL_EXECUTION_SHA}:{cfg}"), (
        "the validation workload changed; the accepted evidence describes a "
        "different configuration")
