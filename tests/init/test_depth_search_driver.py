"""End-to-end CPU rehearsal of the E8 depth search, before it costs anything.

The expensive path — 260 subset evaluations of a 4B teacher — is exercised here
in miniature: a tiny random Qwen3, a two-domain calibration set, 6 -> 4 layers.
That is 6 + 5 = 11 evaluations and it runs in seconds, which is enough to prove
the parts that can silently go wrong:

* the reference-logit cache is an optimization, **not** a numerical change;
* the artifact carries the whole per-round table, the positional-map comparison,
  and hashes that bind it to the calibration set it was measured on;
* a smoke-test teacher is stamped `revision: "local"` so its artifact can never
  be mistaken for a real search;
* the search resumes from `rounds.jsonl` instead of restarting;
* a non-deterministic objective stops the search rather than ranking noise.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.init.contribution import distortion  # noqa: E402

DRIVER = REPO / "scripts/training/search_depth_map.py"


def tiny_local_teacher(path: Path, layers: int = 6, seed: int = 17):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=layers,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for m in model.modules():
            if m.__class__.__name__ == "Qwen3RMSNorm":
                m.weight.uniform_(0.5, 1.5)
    model.save_pretrained(path)
    return model


def tiny_calibration(path: Path, seed: int = 23):
    """Two domains, three sub-types, so domain balancing is actually exercised."""
    torch.manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    domains = {"general": ["general"], "task": ["alpha", "beta"]}
    items = []
    for subtype in ("general", "alpha", "beta"):
        for k in range(2):
            ids = torch.randint(0, 64, (24,)).tolist()
            n_pred = len(ids) - 1
            tags = {} if subtype == "general" else {
                "assistant": list(range(n_pred // 2, n_pred)),
                "eos": [n_pred - 1],
            }
            items.append({
                "item_id": f"{subtype}/{k}", "domain":
                    "general" if subtype == "general" else "task",
                "subtype": subtype, "source": "synthetic",
                "n_tokens": len(ids), "n_prediction_positions": n_pred,
                "templated": subtype != "general", "ids": ids, "tags": tags,
            })
    (path / "items.jsonl").write_text(
        "".join(json.dumps(i) + "\n" for i in items))
    (path / "manifest.json").write_text(json.dumps({
        "artifact": "synthetic_test_calibration",
        "design": {"domains": domains, "aggregation": "test"},
        "totals": {"items": len(items)},
        "manifest_sha256": "test-manifest", "content_sha256": "test-content",
    }, indent=2))
    return items, domains


def run_driver(teacher: Path, calib: Path, out: Path, *extra: str):
    cmd = [sys.executable, str(DRIVER), "--teacher", str(teacher),
           "--calibration", str(calib), "--out", str(out),
           "--student-layers", "4", "--dtype", "float32", "--device", "cpu",
           *extra]
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
           "HOME": str(Path.home())}
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)


@pytest.fixture(scope="module")
def searched(tmp_path_factory):
    root = tmp_path_factory.mktemp("e8search")
    teacher, calib, out = root / "teacher", root / "calib", root / "out"
    tiny_local_teacher(teacher)
    tiny_calibration(calib)
    proc = run_driver(teacher, calib, out)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads((out / "depth_search.json").read_text()), root


def test_the_search_completes_and_reports_the_expected_evaluation_count(searched):
    report, _ = searched
    assert report["objective"]["subset_evaluations"] == 6 + 5 == 11
    assert report["evaluations"] == 11
    assert len(report["rounds"]) == 2
    assert len(report["result"]["kept_teacher_layers"]) == 4
    assert len(report["result"]["removed_teacher_layers"]) == 2
    assert report["result"]["kept_teacher_layers"] == sorted(
        report["result"]["kept_teacher_layers"])


def test_every_round_table_is_complete_and_carries_the_detail(searched):
    report, _ = searched
    first, second = report["rounds"]
    assert [e["candidate"] for e in first["table"]] == [0, 1, 2, 3, 4, 5]
    assert len(second["table"]) == 5
    for entry in first["table"] + second["table"]:
        assert set(entry["per_domain_kl"]) == {"general", "task"}
        assert set(entry["per_subtype_kl"]) == {"general", "alpha", "beta"}
        assert "eos" in entry["diagnostics"]        # recorded, never selecting
    # The primary is the mean of the two domain means, not a token-weighted mean.
    e = first["table"][0]
    assert e["score"] == pytest.approx(
        sum(e["per_domain_kl"].values()) / 2, rel=1e-9)
    assert e["per_domain_kl"]["task"] == pytest.approx(
        (e["per_subtype_kl"]["alpha"] + e["per_subtype_kl"]["beta"]) / 2, rel=1e-9)


def test_the_greedy_choice_is_the_argmin_of_the_primary_score(searched):
    report, _ = searched
    for record in report["rounds"]:
        best = min(record["table"], key=lambda r: (r["score"], r["candidate"]))
        assert record["chosen"] == best["candidate"]
        assert record["chosen_score"] == pytest.approx(best["score"])


def test_the_positional_map_is_scored_by_the_same_objective(searched):
    report, _ = searched
    base = report["positional_baseline"]
    assert len(base["kept_teacher_layers"]) == 4
    assert base["primary_kl"] > 0
    assert report["comparison"]["contribution_map_is_lower_kl"] == (
        report["result"]["primary_kl"] < base["primary_kl"])
    # The greedy search minimizes this objective, so it cannot lose to a map it
    # was free to choose... unless it chose that map, which is also informative.
    if not report["comparison"]["maps_identical"]:
        assert report["result"]["primary_kl"] <= base["primary_kl"]


def test_the_instrument_reports_its_own_noise_floor_as_zero(searched):
    report, _ = searched
    assert report["self_consistency"]["deterministic"] is True
    assert report["self_consistency"]["max_item_kl"] <= 1e-6
    assert report["intact_reference"]["per_subtype"]["general"]["ref_ce"] > 0


def test_a_local_smoke_teacher_is_stamped_and_cannot_pass_as_the_real_one(searched):
    report, _ = searched
    assert report["teacher"]["revision"] == "local"
    assert report["calibration"]["manifest_sha256"] == "test-manifest"
    assert report["calibration"]["content_sha256"] == "test-content"
    assert "report_sha256" in report


def test_the_reference_cache_is_an_optimization_not_a_numerical_change(tmp_path):
    teacher, calib = tmp_path / "teacher", tmp_path / "calib"
    tiny_local_teacher(teacher)
    tiny_calibration(calib)
    cached = run_driver(teacher, calib, tmp_path / "a")
    assert cached.returncode == 0, cached.stderr[-2000:]
    plain = run_driver(teacher, calib, tmp_path / "b", "--no-reference-cache")
    assert plain.returncode == 0, plain.stderr[-2000:]
    a = json.loads((tmp_path / "a" / "depth_search.json").read_text())
    b = json.loads((tmp_path / "b" / "depth_search.json").read_text())
    assert a["result"]["removed_teacher_layers"] == b["result"]["removed_teacher_layers"]
    assert a["result"]["primary_kl"] == pytest.approx(b["result"]["primary_kl"],
                                                     rel=0, abs=0)
    assert b["forward_passes"] > a["forward_passes"]
    assert b["reference_cached"] is False


def test_the_search_resumes_from_the_round_log_instead_of_restarting(tmp_path):
    teacher, calib, out = tmp_path / "teacher", tmp_path / "calib", tmp_path / "out"
    tiny_local_teacher(teacher)
    tiny_calibration(calib)
    full = run_driver(teacher, calib, out)
    assert full.returncode == 0, full.stderr[-2000:]
    first_round = (out / "rounds.jsonl").read_text().splitlines()[0]
    reference = json.loads((out / "depth_search.json").read_text())

    resumed_out = tmp_path / "resumed"
    resumed_out.mkdir()
    (resumed_out / "rounds.jsonl").write_text(first_round + "\n")
    proc = run_driver(teacher, calib, resumed_out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    again = json.loads((resumed_out / "depth_search.json").read_text())
    assert again["result"]["removed_teacher_layers"] == \
        reference["result"]["removed_teacher_layers"]
    assert again["rounds"][0]["resumed"] is True
    assert again["evaluations"] == 5           # round 0 was replayed, not re-scored
    assert "resuming" in proc.stdout


def test_a_nondeterministic_objective_stops_the_search(tmp_path):
    """The self-consistency gate is the thing standing between a ranking and noise."""
    import types

    from aadistill.init.contribution import DistortionSums

    class Jittery:
        def __init__(self):
            self.calls = 0

        def __call__(self, ids):
            self.calls += 1
            torch.manual_seed(self.calls)
            return types.SimpleNamespace(
                logits=torch.randn(1, ids.shape[1], 8))

    sys.path.insert(0, str(REPO / "scripts/training"))
    import search_depth_map as sdm

    prepared = [{"item_id": "x", "subtype": "s",
                 "ids": torch.zeros(1, 5, dtype=torch.long),
                 "targets": torch.zeros(4, dtype=torch.long), "tags": {}}]
    searcher = sdm.Searcher(Jittery(), prepared, {"d": ["s"]})
    noise = sdm.self_consistency(searcher)
    assert noise["deterministic"] is False
    assert noise["max_item_kl"] > sdm.SELF_KL_TOLERANCE
    # And the sums type the driver relies on is the one under test.
    assert isinstance(DistortionSums(), DistortionSums)
    assert distortion(torch.zeros(2, 3), torch.zeros(2, 3),
                      torch.zeros(2, dtype=torch.long)).positions == 2
