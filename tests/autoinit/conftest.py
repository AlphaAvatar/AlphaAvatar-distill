"""Tiny real models, not mocks.

Every fixture here is an actual Qwen3 model with an actual config, saved and
reloaded through the actual ``from_pretrained`` path. A 32-wide, 6-layer teacher
runs a whole beam search in seconds on CPU, so there is no reason to fake the
part of the pipeline that has to be trusted at 4B — the materialize -> reload ->
hash -> validate -> measure cycle is exercised for real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.autoinit.arch import ArchSpec  # noqa: E402
from aadistill.autoinit.calibration import CalibrationProfile, CalibrationSource  # noqa: E402
from aadistill.autoinit.datasets import DatasetRole  # noqa: E402
from aadistill.autoinit.metrics import StateEvalSuite, SuiteItem  # noqa: E402

TEACHER_GEOMETRY = dict(
    hidden_size=32, num_hidden_layers=6, intermediate_size=48,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    vocab_size=128, tie_word_embeddings=True,
)
TARGET_GEOMETRY = dict(
    hidden_size=16, num_hidden_layers=4, intermediate_size=24,
    num_attention_heads=2, num_key_value_heads=2, head_dim=8,
    vocab_size=128, tie_word_embeddings=True,
)


def build_tiny_model(geometry: dict, seed: int = 7, randomize_norms: bool = True):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(max_position_embeddings=256, rope_theta=5_000_000, **geometry)
    model = Qwen3ForCausalLM(cfg).float().eval()
    if randomize_norms:
        # Fresh models initialize every RMSNorm weight to 1.0, which would let a
        # norm-folding bug pass unnoticed.
        with torch.no_grad():
            for m in model.modules():
                if m.__class__.__name__ == "Qwen3RMSNorm":
                    m.weight.uniform_(0.5, 1.5)
    return model


@pytest.fixture
def teacher():
    return build_tiny_model(TEACHER_GEOMETRY)


@pytest.fixture
def teacher_spec():
    return ArchSpec.of("qwen3", TEACHER_GEOMETRY)


@pytest.fixture
def target_spec():
    return ArchSpec.of("qwen3", TARGET_GEOMETRY)


def make_items(n_per_subtype: int = 2, seq_len: int = 24, vocab: int = 128,
               seed: int = 101):
    """Calibration/eval items with domain and sub-type labels and critical tags."""
    torch.manual_seed(seed)
    items = []
    for domain, subtype in (("general", "text"), ("math", "arith")):
        for k in range(n_per_subtype):
            ids = torch.randint(0, vocab, (1, seq_len))
            targets = ids[0, 1:]
            items.append({
                "item_id": f"{subtype}-{k}",
                "input_ids": ids,
                "domain": domain,
                "subtype": subtype,
                # A stand-in for the real critical-token classes (think_close, eos,
                # final_answer, tool_close): a sparse, structurally meaningful mask.
                "tags": {"eos_like": targets == 0, "answer_like": targets % 17 == 0},
            })
    return items


@pytest.fixture
def calibration_items():
    return make_items()


@pytest.fixture
def eval_suite():
    return StateEvalSuite(
        suite_id="test.state_eval", version=1,
        domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"),
        n_items=4, description="tiny held-out suite for the CPU dry run",
    )


@pytest.fixture
def suite_items():
    return [
        SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"], domain=i["domain"],
                  subtype=i["subtype"], tags=i["tags"])
        for i in make_items(seed=202)
    ]


def make_profile(name: str, seed: int = 1) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=f"test.{name}", version=1, description=f"test mixture {name}",
        sources=(CalibrationSource("test/mixture", "local", "general", 2),
                 CalibrationSource("test/mixture", "local", "math", 2)),
        domain_weights={"general": 0.5, "math": 0.5},
        token_budget=96, sample_rule="fixed", seed=seed,
        role=DatasetRole.OPERATOR_CALIBRATION,
    )


@pytest.fixture
def profile():
    return make_profile("balanced")


@pytest.fixture
def two_profiles():
    return (make_profile("balanced", seed=1), make_profile("reasoning", seed=2))
