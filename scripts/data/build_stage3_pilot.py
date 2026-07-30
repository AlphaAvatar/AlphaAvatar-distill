"""Build the two arms of the Stage 3 teacher-target 2x2 from a generated corpus.

Pre-registration: logs/PROPOSAL.md

The experiment asks whether teacher-native targets beat public ones at a fixed
compute budget, so the two arms must differ in *exactly one thing*: the
assistant turn. This script is what enforces that.

  * **Prompt set is the accepted subset, shared.** Only prompts whose teacher
    target passed `aadistill.verify` enter either arm. Prompts that fell back to
    a public target during generation (`target_source == "v1_public"`) are
    dropped from *both* arms — keeping them would put control data in the
    treatment arm and quietly turn the comparison into a mixture ablation.
  * **train/val assignment is shared**, computed from the prompt id alone, so
    the two arms' validation sets hold the same prompts and their val CE is
    comparable. The abort rule (R4) compares an arm against its own step 0, but
    a val split that differed between arms would also make the guard rail
    incomparable across arms.
  * **Only the assistant turn differs.** The user turn is copied from the
    generated corpus for both arms, so prompt rendering is byte-identical.

Emits `<out>/control/` and `<out>/treatment/`, each with `train/` and `val/`
splits laid out as `<split>/<group>.jsonl` the way `load_split` expects, plus a
manifest carrying per-slice counts, rendered-token statistics and sha256 of
every output file.

Nothing here decides the step count: the accepted count is an output of
generation and the run is sized from it (proposal 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.dataset import (  # noqa: E402
    load_jsonl,
    load_split,
    validate_sample,
)
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

ARMS = ("control", "treatment")

# Groups the in-scope generation slices map to (rag_evidence, multihop_qa, gsm8k,
# openmath -> the first two plus code_math). The capability scope was frozen on
# 2026-07-30 and `refusal_uncertainty` is evaluation-only, so it must not reach a
# training mixture (AGENTS.md P3/P10.1). The 2026-07-29 pilot corpus predates
# that decision and does contain refusal rows, which is exactly the case this
# guard exists to catch.
IN_SCOPE_GROUPS = ("rag_evidence", "multihop_qa", "code_math")

# Fields the generator attaches for provenance. They are not part of the
# training schema (the loader ignores unknown top-level keys) but are carried
# through so any sample can be traced back to its candidate (P4).
PROVENANCE = ("target_source", "candidate_index", "think_tokens")


def val_bucket(sample_id: str, val_frac: float) -> bool:
    """Assign a prompt to val by hashing its id — arm-independent by design.

    A shuffle would need a seed, and a seed shared between arms is one more
    thing that can silently drift apart. Hashing the id makes the split a
    function of the prompt set alone.
    """
    digest = hashlib.sha256(sample_id.encode()).digest()
    return (int.from_bytes(digest[:8], "big") % 10_000) < int(val_frac * 10_000)


def build_arms(
    targets: list[dict],
    public: dict[str, dict],
    allowed_groups: tuple[str, ...] = IN_SCOPE_GROUPS,
    drop_out_of_scope: bool = False,
) -> tuple[dict, list[str], dict]:
    """Return ({arm: [samples]}, shared prompt ids, scope report)."""
    accepted = [r for r in targets if r.get("target_source") == "teacher_verified"]
    if not accepted:
        raise SystemExit(
            "no accepted teacher targets in the corpus — nothing to build. "
            "Check the generation manifest's per-slice accept@n."
        )

    out_of_scope = [r for r in accepted if r["group"] not in allowed_groups]
    if out_of_scope:
        counts = Counter(r["group"] for r in out_of_scope)
        if not drop_out_of_scope:
            raise SystemExit(
                f"corpus carries {len(out_of_scope)} accepted prompt(s) in "
                f"out-of-scope group(s) {dict(counts)}; allowed: "
                f"{list(allowed_groups)}. Training on these would widen the "
                "declared capability scope silently (P3/P10.1). Re-generate "
                "with the in-scope slices, or pass --drop-out-of-scope to "
                "exclude them and have the exclusion recorded."
            )
        accepted = [r for r in accepted if r["group"] in allowed_groups]
        if not accepted:
            raise SystemExit("every accepted prompt was out of scope")
    scope = {
        "allowed_groups": list(allowed_groups),
        "dropped_out_of_scope": dict(Counter(r["group"] for r in out_of_scope)),
    }

    missing = [r["id"] for r in accepted if r["id"] not in public]
    if missing:
        raise SystemExit(
            f"{len(missing)} accepted prompts have no public target in the "
            f"public dir (first few: {missing[:5]}). The control arm cannot be "
            "built over the same prompt set, so the comparison is impossible."
        )

    arms: dict[str, list[dict]] = {a: [] for a in ARMS}
    seen: set[str] = set()
    for row in accepted:
        sid = row["id"]
        if sid in seen:  # a duplicate prompt would weight one arm's loss twice
            raise SystemExit(f"duplicate accepted prompt id {sid!r} in corpus")
        seen.add(sid)

        user_turns = [m for m in row["messages"] if m["role"] != "assistant"]
        teacher_turn = [m for m in row["messages"] if m["role"] == "assistant"]
        if len(teacher_turn) != 1:
            raise SystemExit(
                f"{sid}: expected exactly one assistant turn, got "
                f"{len(teacher_turn)}"
            )

        pub = public[sid]
        pub_turn = [m for m in pub["messages"] if m["role"] == "assistant"]
        if len(pub_turn) != 1:
            raise SystemExit(
                f"{sid}: public target has {len(pub_turn)} assistant turns"
            )
        if any(m.get("reasoning_content") for m in pub_turn):
            raise SystemExit(
                f"{sid}: public target already carries a reasoning trace — the "
                "control arm would not be a public-target control"
            )

        base = {k: row[k] for k in ("id", "group", "source", "format")}
        common = {k: row[k] for k in PROVENANCE if k in row}
        arms["treatment"].append({**base, **common, "messages": user_turns + teacher_turn})
        # Same user turns as the treatment arm, so the two renders share a
        # byte-identical prompt prefix and differ only after the assistant header.
        arms["control"].append({
            **base, "target_source": "v1_public",
            "messages": user_turns + pub_turn,
        })

    return arms, sorted(seen), scope


def write_arm(out_dir: Path, samples: list[dict], val_frac: float) -> dict:
    """Write train/val group files for one arm; return per-split counts."""
    splits: dict[str, dict[str, list[dict]]] = {
        "train": defaultdict(list), "val": defaultdict(list)
    }
    for s in samples:
        validate_sample(s)  # fail here, on the dev box, not on a paid pod
        split = "val" if val_bucket(s["id"], val_frac) else "train"
        splits[split][s["group"]].append(s)

    if not splits["train"]:
        raise SystemExit(f"{out_dir}: empty train split")
    if not splits["val"]:
        raise SystemExit(
            f"{out_dir}: empty val split — the abort rule (R4) compares val CE "
            "against step 0 and cannot run without one"
        )

    counts: dict[str, dict] = {}
    for split, groups in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        counts[split] = {"total": 0, "by_group": {}, "by_slice": {}}
        for group, rows in sorted(groups.items()):
            path = split_dir / f"{group}.jsonl"
            with open(path, "w") as f:
                for r in rows:  # sorted by id: file order is reproducible
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            counts[split]["by_group"][group] = len(rows)
            counts[split]["total"] += len(rows)
        counts[split]["by_slice"] = dict(
            Counter(r["source"] for rows in groups.values() for r in rows)
        )
    return counts


def token_stats(arm_dir: Path, tokenizer, block_len: int, seed: int) -> dict:
    """Token distribution plus the packing the *trainer* will actually build.

    Packing goes through `build_blocks`, not `best_fit_blocks` directly, because
    the trainer packs **per group** — a block never straddles groups. That
    changes the padding bill: a group with few samples cannot fill a block, so
    measuring over the pooled corpus would understate padding and overstate how
    much real signal a step carries.
    """
    from aadistill.data.dataset import encode_sample
    from aadistill.training.train import build_blocks

    totals, supervised = [], []
    for split in ("train", "val"):
        for rows in load_split(arm_dir, split).values():
            for s in rows:
                ids, mask = encode_sample(tokenizer, s)
                totals.append(len(ids))
                supervised.append(int(sum(mask)))

    def q(v):
        v = sorted(v)
        return {"n": len(v), "min": v[0], "p50": int(statistics.median(v)),
                "p90": v[int(0.9 * len(v)) - 1], "max": v[-1],
                "mean": round(sum(v) / len(v), 1)}

    packed = {}
    for split in ("train", "val"):
        ids_t, mask_t, _, per_group = build_blocks(
            tokenizer, arm_dir, split, block_len, None,
            packing="best_fit", seed=seed,
        )[:4]
        real = sum(g["real_tokens"] for g in per_group.values())
        blocks = int(ids_t.shape[0])
        packed[split] = {
            "blocks": blocks,
            "block_tokens": blocks * block_len,
            "real_tokens": real,
            "padding_tokens": blocks * block_len - real,
            "efficiency": round(real / (blocks * block_len), 4) if blocks else 0.0,
            "supervised_kept": int(mask_t.sum()),
            "truncated_samples": sum(
                g["truncated_samples"] for g in per_group.values()
            ),
            "by_group": per_group,
        }

    kept = sum(p["supervised_kept"] for p in packed.values())
    return {
        "rendered_tokens": q(totals),
        "supervised_tokens": q(supervised),
        "supervised_total": sum(supervised),
        "supervised_fraction": round(sum(supervised) / sum(totals), 4),
        f"packed_best_fit_{block_len}": packed,
        "lossless": all(p["truncated_samples"] == 0 for p in packed.values())
        and kept == sum(supervised),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="artifacts/stage2_v2/pilot/targets.jsonl",
                    help="targets.jsonl from generate_teacher_answers.py")
    ap.add_argument("--public-dir", default="data/stage2_v1",
                    help="data dir supplying the control arm's public targets")
    ap.add_argument("--out", default="data/stage3_pilot")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--block-len", type=int, default=8192,
                    help="block length to report best_fit packing against")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--seed", type=int, default=20260726,
                    help="best_fit placement seed used for the reported packing")
    ap.add_argument("--allowed-groups", default=",".join(IN_SCOPE_GROUPS),
                    help="groups permitted in the training mixture")
    ap.add_argument("--drop-out-of-scope", action="store_true",
                    help="exclude out-of-scope groups instead of failing")
    ap.add_argument("--no-token-stats", action="store_true",
                    help="skip tokenizer-dependent stats (no model download)")
    args = ap.parse_args()

    targets_path = REPO_ROOT / args.targets
    targets = load_jsonl(targets_path, validate=False)

    public: dict[str, dict] = {}
    for split in ("train", "val"):
        try:
            groups = load_split(REPO_ROOT / args.public_dir, split)
        except FileNotFoundError:
            continue
        for rows in groups.values():
            for s in rows:
                public[s["id"]] = s

    allowed = tuple(g.strip() for g in args.allowed_groups.split(",") if g.strip())
    arms, prompt_ids, scope = build_arms(
        targets, public, allowed, args.drop_out_of_scope
    )
    out_root = REPO_ROOT / args.out

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": "Stage 3 teacher-target 2x2: control vs treatment over one "
                   "shared accepted prompt subset",
        "targets_corpus": {
            "path": args.targets, "sha256": sha256_file(targets_path),
            "rows": len(targets),
        },
        "public_dir": args.public_dir,
        "accepted_prompts": len(prompt_ids),
        "dropped_public_fallback": sum(
            1 for r in targets if r.get("target_source") != "teacher_verified"
        ),
        "capability_scope": scope,
        "packing": {"mode": "best_fit", "block_len": args.block_len,
                    "seed": args.seed},
        "val_frac": args.val_frac,
        "val_assignment": "sha256(id) % 10000 < val_frac*10000 — shared by both "
                          "arms, seed-free",
        "arms": {},
    }

    for arm in ARMS:
        counts = write_arm(out_root / arm, arms[arm], args.val_frac)
        manifest["arms"][arm] = {"counts": counts}

    # Both arms must hold the same prompts in the same split, or the comparison
    # is confounded by which prompts each arm never trained on.
    for split in ("train", "val"):
        ids = {}
        for arm in ARMS:
            ids[arm] = {
                s["id"] for rows in load_split(out_root / arm, split).values()
                for s in rows
            }
        if ids["control"] != ids["treatment"]:
            raise SystemExit(
                f"{split}: arms hold different prompt sets "
                f"(symmetric difference {len(ids['control'] ^ ids['treatment'])})"
            )
    manifest["prompt_sets_identical"] = True

    if not args.no_token_stats:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        for arm in ARMS:
            manifest["arms"][arm]["tokens"] = token_stats(
                out_root / arm, tok, args.block_len, args.seed
            )

    for arm in ARMS:
        manifest["arms"][arm]["files"] = {
            str(p.relative_to(out_root)): sha256_file(p)
            for p in sorted((out_root / arm).rglob("*.jsonl"))
        }

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"accepted prompts: {manifest['accepted_prompts']} "
          f"(dropped {manifest['dropped_public_fallback']} public fallbacks)")
    if scope["dropped_out_of_scope"]:
        print(f"  EXCLUDED out-of-scope groups: {scope['dropped_out_of_scope']}")
    for arm in ARMS:
        info = manifest["arms"][arm]
        c = info["counts"]
        print(f"\n=== {arm} ===")
        print(f"  train {c['train']['total']}  val {c['val']['total']}")
        print(f"  by slice (train): {c['train']['by_slice']}")
        if "tokens" in info:
            t = info["tokens"]
            tr = t[f"packed_best_fit_{args.block_len}"]["train"]
            print(f"  rendered tokens : {t['rendered_tokens']}")
            print(f"  supervised      : {t['supervised_total']} "
                  f"(fraction {t['supervised_fraction']})")
            print(f"  train packed    : {tr['blocks']} blocks, real "
                  f"{tr['real_tokens']}, padding {tr['padding_tokens']}, "
                  f"efficiency {tr['efficiency']}")
            print(f"  lossless        : {t['lossless']} "
                  f"(truncated {tr['truncated_samples']})")
    print(f"\nwrote {out_root}/manifest.json")


if __name__ == "__main__":
    main()
