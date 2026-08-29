"""The one canonical way this project turns a file set into a source identity.

Every session type digests its declared executable so a grant can pin the code it
was issued against, and `session_commit_gate` **independently re-derives** that
digest from the blobs at the launch commit. Producer and consumer must therefore
agree exactly — and the formula is currently inlined in seven places
(`authorization.py`, `phase_a.py`, `phase_b.py`, `recovery.py` x2,
`generation.py`, and `session_prechecks.py`, the consumer).

The behavioural continuation became the eighth and got it wrong: it used
`sha256_json(entries)`. Same field name, same type, a value the gate can never
reproduce. Every `$0` check passed anyway, because they all compared the digest
against *itself* — the producer and the continuation's own verifier shared one
implementation, so they agreed under either formula. Only `session_commit_gate`,
which re-derives independently, could see it, and it refused every launch.

So this module exists to be the thing a new session imports instead of writing an
eighth copy. It deliberately does **not** refactor the existing seven: each lives
inside a frozen source set, and rewriting them would move Phase-A and Phase-B
digests for a reason unrelated to their own science. They are byte-identical to
this implementation, and a test asserts that.

The real guarantee is not shared code, which can still drift. It is the
regression that runs the actual `session_commit_gate` against the actual
authorization and requires acceptance.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

#: What v3 of a source-set identity means. Recorded in artifacts so a reader can
#: tell a format change from a content change: the same files under a different
#: algorithm produce a different digest and would otherwise look like an edit.
CANONICAL_DIGEST_ALGORITHM = "sha256-over-sorted-path-colon-sha256-lines/v1"


def canonical_source_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    """SHA256 over ``path:sha256\\n`` lines, in sorted path order.

    Byte-for-byte the formula `session_commit_gate` recomputes. `entries` may
    carry extra keys (`bytes`, and anything else a caller records) — only `path`
    and `sha256` participate, because the consumer only has those two: it reads
    blobs out of git and never sees a stat.

    Sorting happens here rather than being assumed of the caller. The consumer
    iterates `sorted(auth.harness_source_files)`, so a producer that preserved
    declaration order would agree only by luck.
    """
    ordered = sorted(entries, key=lambda e: e["path"])
    return hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in ordered).encode()
    ).hexdigest()
