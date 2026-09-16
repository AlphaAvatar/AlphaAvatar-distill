"""The full-joint-search driver, EXECUTED end to end at toy scale.

Zero cost, CPU only. Four paid pods in this project have died inside lines no
test had executed, so this drives the real driver — its real stages, its real
`afford` check, its real space derivation, its real selection parsing, its real
`run`/`finish` — rather than asserting things about its source.

Only the model and the calibration are scaled down. `run_phase_a_search` is
wrapped, not stubbed: the wrapper substitutes the toy teacher, geometry and
suite that function already accepts for exactly this purpose, asserts the SPACE
arguments the driver passed, and then calls the real search. Real operators,
real checkpoints on disk, real `from_pretrained` reloads, real hashing, real
measurement.

**What this cannot check.** `--device cuda` is not exercised, because this box
has no GPU. That is stated rather than papered over: a CPU run is the one
substitution that has previously hidden a device-placement bug here, so device
placement remains owed to a real GPU engineering validation before any paid
launch. What a CPU run *can* prove is everything else, which is most of it.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """A toy teacher, target, suite and two materialized profiles."""
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.initialization.specs.metrics import StateEvalSuite, SuiteItem
    from experiments.phase_c2 import full_search_space as FS
    from transformers import Qwen3Config, Qwen3ForCausalLM

    FS.register_c2_operators()

    teacher_geometry = dict(hidden_size=32, num_hidden_layers=6,
                            intermediate_size=48, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, vocab_size=128,
                            tie_word_embeddings=True)
    target_geometry = dict(hidden_size=16, num_hidden_layers=4,
                           intermediate_size=24, num_attention_heads=2,
                           num_key_value_heads=2, head_dim=8, vocab_size=128,
                           tie_word_embeddings=True)

    torch.manual_seed(4242)
    teacher = Qwen3ForCausalLM(Qwen3Config(
        max_position_embeddings=256, rope_theta=5_000_000,
        **teacher_geometry)).float().eval()
    with torch.no_grad():
        for module in teacher.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                module.weight.uniform_(0.5, 1.5)

    def items(seed, n=2, seq_len=24):
        torch.manual_seed(seed)
        out = []
        for k in range(n):
            ids = torch.randint(0, 128, (1, seq_len))
            targets = ids[0, 1:]
            #: The tags must MATCH something, or every state has no value for
            #: `state.critical_token_kl`, the policy drops them all as
            #: ineligible, and the search ends with no leaf — which reads
            #: exactly like a broken operator.
            out.append({"item_id": f"text-{k}", "domain": "general",
                        "subtype": "text", "input_ids": ids,
                        "tags": {"eos_like": targets == 0,
                                 "answer_like": targets % 17 == 0}})
        return out

    #: The REAL profiles are used, not toy substitutes registered under their
    #: ids. Overriding them would mutate the process-global profile registry for
    #: every sibling test in the session — this project has already paid for
    #: that class of coupling — and it is unnecessary: both real mixtures are
    #: materialized here, so the driver's own `get_profile`, materialization
    #: check and `resolve` all execute for real. Only the ITEMS handed to the
    #: search are scaled down, because a 32-wide toy teacher cannot index real
    #: token ids.
    from aadistill.initialization.calibration.profiles import get_profile
    real_profiles = [get_profile(q) for q in FS.PROFILE_IDS]

    suite = StateEvalSuite(
        suite_id="toy.state_eval", version=1, domains=("general",),
        subtypes={"general": ("text",)},
        critical_tags=("eos_like", "answer_like"), description="toy")
    suite_items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                             domain=i["domain"], subtype=i["subtype"],
                             tags=i["tags"]) for i in items(99)]

    adapter = get_adapter("qwen3")
    control = tmp_path / "control"
    adapter.save(adapter.build_model(
        adapter.build_config(teacher.config,
                             ArchSpec.of("qwen3", target_geometry)),
        torch.float32, 4242), str(control))

    return {"teacher": teacher, "target_geometry": target_geometry,
            "suite_bundle": (suite, suite_items, {"teacher_sha256": "0" * 64}),
            "items": items, "profiles": real_profiles, "tmp": tmp_path}


@pytest.fixture
def driven(toy, tmp_path, monkeypatch):
    """The driver, pointed at a toy workspace, with the search scaled down."""
    import autoinit_phase_c2_full_search_driver as D
    import phase_a_search

    monkeypatch.setattr(D, "AUDIT", tmp_path / "audit")
    monkeypatch.setattr(D, "SEARCH_WORKDIR", tmp_path / "search")
    monkeypatch.setattr(D, "STATUS", tmp_path / "status.txt")
    monkeypatch.setattr(D, "STATE_EVAL", tmp_path / "unused")

    seen: dict = {}
    real = phase_a_search.run_phase_a_search

    def wrapped(**kwargs):
        #: The SPACE the driver passed is the thing under test here.
        seen.update(kwargs)
        toy_items = {p.qualified_id: [
            {k: v for k, v in item.items() if k != "tags"}
            for item in toy["items"](abs(hash(p.qualified_id)) % 10_000)]
            for p in toy["profiles"]}
        return real(**{**kwargs,
                       "repo_root": toy["tmp"],
                       "teacher_id": "toy",
                       "canonical_init": "control",
                       "canonical_sha256": None,
                       "teacher_loader": lambda: toy["teacher"],
                       "target_geometry": toy["target_geometry"],
                       "suite_bundle": toy["suite_bundle"],
                       "calibration_items": toy_items,
                       "device": "cpu"})

    monkeypatch.setattr(phase_a_search, "run_phase_a_search", wrapped)
    return D, seen


def _args(**over):
    import autoinit_phase_c2_full_search_driver as D

    base = ["--authorization-path", "unused", "--rate", "1.09",
            "--authorized-usd", "100", "--soft-stop-usd", "90",
            "--search-minutes", "5", "--search-deadline-minutes", "5",
            "--top-n", "3", "--device", "cpu"]
    for key, value in over.items():
        flag = "--" + key.replace("_", "-")
        if flag in base:
            base[base.index(flag) + 1] = str(value)
    return D.build_parser().parse_args(base)


def test_the_driver_runs_all_three_stages_for_real(driven):
    """Search, commit, stop — every line executed, nothing stubbed."""
    D, seen = driven
    driver = D.FullSearchDriver(_args())
    assert driver.run() == 0

    assert driver.completed == ["bind_identities", "full_joint_search",
                               "commit_top_k"]
    assert driver.ev["successful"] is True
    assert driver.ev["outcome"] == "ALL_DONE"

    #: The terminus, asserted on the record rather than on the docstring.
    assert driver.ev["followon_reachable_from_this_driver"] is False
    for flag in ("trains_anything", "measures_behaviour",
                 "runs_recovery_probes", "names_an_incumbent"):
        assert driver.ev[flag] is False, flag

    #: The success marker was emitted, and the failure marker was not.
    status = (D.STATUS).read_text()
    assert f"MARKER:{D.SUCCESS_MARKER}" in status
    assert f"MARKER:{D.FAILURE_MARKER}" not in status
    #: And the evidence file is on disk and parses.
    written = json.loads(
        (D.AUDIT / "c2_full_search_evidence.json").read_text())
    assert written["stages_completed"] == driver.completed


def test_the_driver_searches_the_JOINT_space(driven):
    """`impl_profiles=None` and the derived allowed set, not Search-1's."""
    from experiments.phase_c2 import full_search_space as FS
    from experiments.phase_c2 import search_space as SS

    D, seen = driven
    assert D.FullSearchDriver(_args()).run() == 0

    assert seen["impl_profiles"] is None, (
        "the joint search must pin no calibration; that is the restriction it "
        "exists to reopen")
    assert tuple(seen["allowed_impls"]) == FS.full_joint_space(REPO).allowed_impls
    #: Strictly wider than Search-1's, and not Search-1's run id.
    assert set(seen["allowed_impls"]) > set(SS.C2_ALLOWED_IMPLS)
    assert seen["run_id"] == D.RUN_ID != "autoinit.v1.phase_c2.search1"
    #: No baseline fallback: this session rebuilds and compares nothing.
    assert seen["conditional_candidates"] is None


def test_stage_a_records_the_derived_space_and_no_literals(driven):
    """Stage A binds what a reviewer needs and derives it from the registry."""
    from experiments.phase_c2 import full_search_space as FS

    D, seen = driven
    driver = D.FullSearchDriver(_args())
    assert driver.bind_identities() is True

    detail = driver.ev["stages"]["bind_identities"]["detail"]
    size = FS.size_report(REPO)
    assert detail["space"]["total_leaves"] == size["full_joint"]["total_leaves"]
    assert detail["space"]["impl_profiles"] is None
    assert detail["space"]["exclusions"] == FS.EXCLUSIONS
    assert detail["unmeasured_cost_inputs"] == []
    #: The driver must contain no operator list, profile list or leaf count.
    source = Path(D.__file__).read_text()
    for literal in ("attention.activation_importance_v1",
                    "calib.domain_balanced@v1",
                    str(size["full_joint"]["total_leaves"])):
        assert literal not in source, (
            f"{literal} is written into the driver; the space must be derived")


def test_the_soft_stop_refuses_a_beam_it_cannot_fund(driven):
    """`afford` runs for real, before the expensive stage starts."""
    D, seen = driven
    #: A soft stop below what the beam's envelope would cost.
    driver = D.FullSearchDriver(_args(soft_stop_usd="0.01"))
    assert driver.bind_identities() is True
    assert driver.run() == 1
    failed = driver.ev["stages"]["full_joint_search"]
    assert failed["passed"] is False
    assert "AuthorizationError" in failed["detail"]["error"]
    assert driver.ev["outcome"] == "FAILED"
    assert f"MARKER:{D.FAILURE_MARKER}" in (D.STATUS).read_text()
    #: And the search was never invoked.
    assert "allowed_impls" not in seen


def test_commit_refuses_a_candidate_set_of_the_wrong_size(driven):
    """A set of the wrong size is not the preregistered set."""
    D, seen = driven
    driver = D.FullSearchDriver(_args())
    assert driver.bind_identities() is True
    assert driver.full_joint_search() is True

    #: Rewrite the committed selection to hold one fewer candidate.
    selection = D.SEARCH_WORKDIR / "stage1_selection.json"
    doc = json.loads(selection.read_text())
    doc["selected"] = doc["selected"][:-1]
    selection.write_text(json.dumps(doc))
    with pytest.raises(Exception, match="not the"):
        driver.commit_top_k()


def test_commit_refuses_when_the_search_committed_nothing(driven, tmp_path):
    """No selection artifact means no candidate set to freeze."""
    D, seen = driven
    driver = D.FullSearchDriver(_args())
    (D.SEARCH_WORKDIR).mkdir(parents=True, exist_ok=True)
    with pytest.raises(Exception, match="no candidate set"):
        driver.commit_top_k()


def test_stage_a_refuses_an_unmaterialized_branch_profile(driven, monkeypatch):
    """A declared-but-unbuilt mixture is a space this session cannot search."""
    from aadistill.initialization.calibration import profiles as P
    from experiments.phase_c2 import full_search_space as FS

    D, seen = driven
    real = P.get_profile

    def unbuilt(qualified_id):
        profile = real(qualified_id)
        if qualified_id == FS.PROFILE_IDS[-1]:
            return type(profile).__new__(type(profile)) if False else \
                _Unmaterialized(profile)
        return profile

    class _Unmaterialized:
        def __init__(self, inner):
            self._inner = inner
        def __getattr__(self, name):
            return getattr(self._inner, name)
        @property
        def materialized(self):
            return False

    monkeypatch.setattr(P, "get_profile", unbuilt)
    driver = D.FullSearchDriver(_args())
    with pytest.raises(Exception, match="materialized"):
        driver.bind_identities()


def test_the_driver_has_no_path_into_a_behavioural_stage():
    """One authorization must not be able to buy a search AND a promotion."""
    import autoinit_phase_c2_full_search_driver as D

    source = Path(D.__file__).read_text()
    for forbidden in ("recovery", "probe", "correct_overall", "screening",
                      "confirmation", "usable_rollout"):
        #: Prose may discuss them; no call may reach them. Checked on the parsed
        #: code rather than the text, so the docstrings that EXPLAIN the
        #: boundary do not trip the guard that enforces it.
        import ast
        tree = ast.parse(source)
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)}
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in n.names}
        assert not any(forbidden in name.lower()
                       for name in called | imported), forbidden
