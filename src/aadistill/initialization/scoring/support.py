"""Which vocabulary entries a divergence is reduced over.

Two supports, and the choice is part of what a measurement MEANS rather than how
fast it runs:

``full_vocab_v1``
    Every vocabulary entry. :func:`aadistill.initialization.statistics.contribution.distortion`
    implements it and remains the ORACLE — it is not superseded and not deleted.

``reference_topk_tail_v1(top_k=N)``
    The Top-N entries of the **reference** distribution at each prediction
    position, plus one aggregate ``K+1`` bucket holding every entry outside that
    support.

For reference ``p`` and compared distribution ``q``, with ``S = TopK(p, K)``::

    p_tail = 1 - sum_{i in S} p_i
    q_tail = 1 - sum_{i in S} q_i

    KL = sum_{i in S} p_i * (log p_i - log q_i)
         + p_tail * (log p_tail - log q_tail)

**The support is REFERENCE-defined, and that is a semantic commitment, not an
implementation detail.** The candidate never chooses its own Top-K. Forward KL
``KL(p || q)`` weights each term by ``p_i``, so the entries that matter are the
ones ``p`` puts mass on; letting ``q`` pick the support would weight terms by
mass ``p`` may not have and would make two candidates incomparable, because each
would be scored on its own partition. The consequence to accept is that a
candidate which moves mass onto entries outside ``S`` has that movement counted
only through the single tail bucket.

**This is not mathematically identical to full-vocabulary KL.** It is a coarsening
of the partition, so by the data-processing inequality it is a LOWER bound on the
full KL for the same pair of distributions. Anything that needs to tell the two
apart therefore has to carry the support in its identity — see
:class:`aadistill.initialization.scoring.protocol_identity.ReductionSemantics`,
whose ``as_dict`` omits the field entirely under the historical full-vocab
contract so that no existing ``measurement_protocol_id`` recomputes differently.

``top_k`` is always supplied by the caller. It is an experiment-policy parameter,
not a framework default, and this module contains no value for it.

**The tail is computed from the complement's own logits, never by subtracting
the support's mass from one.** An independent review caught the subtraction as a
real numerical-semantic defect, and the evidence was already in our own report: on
real logits the measured Top-200 support mass reached ``1.000001``, a probability
above one, so the complement was below float32 resolution and one per-item score
had the coarse KL EXCEED the full-vocabulary KL by 2.2e-03 relative. A coarsening
cannot do that. Masking the support and reducing the rest has no cancellation, and
the partition's two masses then sum to one by construction.

**A zero candidate tail against a non-zero reference tail is ``+inf``, and stays
``+inf``.** Only zero REFERENCE mass makes a forward-KL term vanish. The earlier
code dropped the term whenever either mass rounded to zero, which silently turned
an infinite divergence into a finite and favourable score.

Implementation notes taken from ``lasgroup/SDPO`` @
``7c457fc1b1f636ae794eb0362ba37d4743b06fbc``: normalized top-k log
probabilities, the optional aggregate tail bucket, and gathering the compared
distribution on one fixed support. Its support policy — the *student's* top-k —
is deliberately NOT adopted, for the reason above, and its stable-complement
formulation is superseded here by the complement logsumexp, which needs no
stabilisation because it never cancels.

``reference_student_union_topk_tail_v1(top_k=N)``
    A THIRD support, declared separately and never a silent replacement of the
    second. With ``S_T = TopK(p, N)`` (the reference's) and ``S_C = TopK(q, N)``
    (the compared distribution's own), the partition is every entry of
    ``U = S_T ∪ S_C`` -- between ``N`` and ``2N`` entries, per position -- plus
    ONE tail bucket holding everything outside ``U``::

        KL_U = sum_{i in U} p_i (log p_i - log q_i)
               + p(not U) (log p(not U) - log q(not U))

    The compared distribution DOES move the partition here, and that is the
    declared difference: an entry a candidate promotes into its own Top-N is
    scored individually instead of disappearing into the tail. ``U`` refines
    the reference ``K+1`` partition (``S_T`` is inside it, and its tail is a
    subset of the reference's tail), and the full vocabulary refines ``U``, so
    by the data-processing inequality, for every pair of distributions,

        KL_{S_T + tail}  <=  KL_U  <=  KL_full

    It is still NOT the full-vocabulary KL, and nothing may report it as one.
    Both Top-N selections are DETERMINISTIC (:func:`deterministic_topk`): value
    descending, then vocabulary index ascending, so a tie at the N-th value --
    frequent in bf16 -- always admits the same entries.

    The reference's probability of a candidate-only entry (``S_C \\ S_T``) is
    not in a ``K+1`` sketch. It is recovered EXACTLY, never approximated from
    the tail, by a caller-supplied :class:`TeacherRetrieval` (the reference's
    final hidden state and full-vocabulary log-normalizer, projected through
    its output head on just those entries); see
    :mod:`aadistill.initialization.scoring.output_head`. The reference's mass
    outside ``U`` is its own Top-N tail -- computed from its complement logits
    when the sketch was built -- minus the candidate-only entries, which lie
    inside that tail; the candidate's is reduced directly from its own
    complement logits with ``U`` masked, never by subtraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from aadistill.initialization.statistics.contribution import (
    DistortionSums, _position_weights, _reduce_on_device,
)

SUPPORT_FULL_VOCAB_V1 = "full_vocab_v1"
SUPPORT_REFERENCE_TOPK_TAIL_V1 = "reference_topk_tail_v1"
SUPPORT_REFERENCE_STUDENT_UNION_TOPK_TAIL_V1 = (
    "reference_student_union_topk_tail_v1")
#: The supports that take a caller-supplied ``top_k``.
TOPK_SUPPORTS = (SUPPORT_REFERENCE_TOPK_TAIL_V1,
                 SUPPORT_REFERENCE_STUDENT_UNION_TOPK_TAIL_V1)
#: How both Top-K selections of the union support break ties. Part of the
#: union support's identity: bf16 logits tie at the K-th value often enough
#: that an unstated rule would make the partition depend on the kernel.
TOPK_TIE_BREAK = "value_desc_then_index_asc/v1"


@dataclass(frozen=True)
class DistributionSupport:
    """The entries a divergence is reduced over, as an identity-bearing value.

    ``as_dict`` is what goes into a hash, so it is deliberately minimal and it
    OMITS ``top_k`` under the full-vocabulary contract: a key present with a
    ``None`` value would still change the serialization, and the whole point of
    the optional field is that historical identities are byte-identical.
    """

    support_id: str
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.support_id == SUPPORT_FULL_VOCAB_V1:
            if self.top_k is not None:
                raise ValueError(
                    f"{SUPPORT_FULL_VOCAB_V1} reduces over every entry and takes "
                    f"no top_k; got {self.top_k!r}")
            return
        if self.support_id not in TOPK_SUPPORTS:
            raise ValueError(
                f"unknown distribution support {self.support_id!r}; known: "
                f"{SUPPORT_FULL_VOCAB_V1!r}, "
                + ", ".join(repr(s) for s in TOPK_SUPPORTS))
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise ValueError(
                f"{self.support_id} needs an integer top_k, got "
                f"{self.top_k!r}. It is an experiment-policy parameter and this "
                "module has no default for it.")
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")

    @property
    def is_full_vocab(self) -> bool:
        return self.support_id == SUPPORT_FULL_VOCAB_V1

    @property
    def is_union(self) -> bool:
        """The reference ∪ candidate Top-K support, which needs retrieval."""
        return self.support_id == SUPPORT_REFERENCE_STUDENT_UNION_TOPK_TAIL_V1

    def as_dict(self) -> dict[str, Any]:
        if self.is_full_vocab:
            return {"support_id": self.support_id}
        if self.is_union:
            #: The tie-break is serialized for the union support ONLY, so the
            #: reference-Top-K identity every historical record cites is
            #: unchanged.
            return {"support_id": self.support_id, "top_k": int(self.top_k),
                    "topk_tie_break": TOPK_TIE_BREAK}
        return {"support_id": self.support_id, "top_k": int(self.top_k)}

    def __str__(self) -> str:
        return (self.support_id if self.is_full_vocab
                else f"{self.support_id}(top_k={self.top_k})")


#: The historical contract. Named so callers can be explicit rather than passing
#: `None` and relying on a default that may change meaning later.
FULL_VOCAB_V1 = DistributionSupport(SUPPORT_FULL_VOCAB_V1)


def reference_topk_tail(top_k: int) -> DistributionSupport:
    """``reference_topk_tail_v1`` at a caller-supplied ``top_k``."""
    return DistributionSupport(SUPPORT_REFERENCE_TOPK_TAIL_V1, top_k=int(top_k))


def reference_student_union_topk_tail(top_k: int) -> DistributionSupport:
    """``reference_student_union_topk_tail_v1`` at a caller-supplied ``top_k``."""
    return DistributionSupport(SUPPORT_REFERENCE_STUDENT_UNION_TOPK_TAIL_V1,
                               top_k=int(top_k))


# --- numerics ---------------------------------------------------------------

def owned_float32(logits: torch.Tensor) -> torch.Tensor:
    """An fp32 copy this module may destroy.

    `.float()` RETURNS THE SAME TENSOR when the input is already fp32, so masking
    the support in place would mutate the caller's logits -- the reference block a
    search has cached, or the candidate block the operator is still holding. The
    copy is explicit for that reason, not defensively.
    """
    return logits.to(torch.float32, copy=True)


def complement_log_mass(z_owned: torch.Tensor, support: torch.Tensor,
                        log_z: torch.Tensor) -> torch.Tensor:
    """``log P(outside support)``, from the complement's own logits.

    DESTROYS ``z_owned``: the support entries are set to ``-inf`` and the
    remaining row is reduced. The caller must own it.

    This replaces ``log(1 - sum of the support's probabilities)``, which is the
    defect an independent review caught. Reconstructing the tail by subtraction
    crosses float32 resolution whenever the support holds nearly all the mass:
    the measured Top-200 support mass reached **1.000001** on real logits, a
    probability above one, and from there the complement is noise. One per-item
    score consequently had the coarse KL EXCEED the full-vocabulary KL by
    2.2e-03 relative, which a coarsening cannot do.

    The complement logits already exist in the row. Reducing them directly has no
    cancellation at all, and it is exact in the only sense that matters here: the
    K+1 partition's two masses are computed by the same ``logsumexp`` against the
    same normalizer, so they sum to one by construction rather than by luck.
    """
    z_owned.scatter_(-1, support, float("-inf"))
    return torch.logsumexp(z_owned, dim=-1) - log_z


#: Rows per full-row tie-resolution block. Only rows whose ties at the K-th
#: value reach past the 2K candidates take that path, so this bounds a rare
#: transient (a few ``[rows, V]`` masks), not the common one.
_TIE_ROWS = 64


@torch.no_grad()
def deterministic_topk(values: torch.Tensor, k: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(top values, indices)`` of the last dimension, ``[..., k]``, by a
    STATED order: value descending, then vocabulary index ascending.

    ``torch.topk`` returns *a* valid Top-K, and when several entries tie at the
    K-th value which of them it admits is unspecified -- it can differ between
    devices, kernels and batch shapes. bf16 logits tie there often (measured
    on real bf16 logits of a six-figure vocabulary at a Top-K of a few
    hundred: most rows tie at the K-th value, and ``torch.topk``'s set differs
    from this one on more than half of them), so a support
    defined by ``torch.topk`` alone would let the kernel choose the partition.
    Here the SET is fixed -- every entry strictly above the K-th value, then
    the lowest-index entries equal to it -- and so is the ORDER.

    How, cheaply: one ``torch.topk`` of ``2K`` candidates in the input's own
    dtype (ordering and equality survive exact widening, so bf16 selects what
    a float32 copy would). When the smallest candidate is strictly below the
    K-th value, every entry at or above it is a candidate and the answer is the
    first K candidates in (value desc, index asc) order -- an ``[rows, 2K]``
    sort. Only rows whose ties reach past ``2K`` candidates fall back to a
    full-row mask-and-rank pass. NaN ranks as ``+inf`` for selection only; the
    returned values are re-read from ``values``, NaN included.
    """
    vocab = int(values.shape[-1])
    k = int(k)
    if not 1 <= k <= vocab:
        raise ValueError(f"top-k of {k} over a last dimension of {vocab}")
    lead = values.shape[:-1]
    original = values.reshape(-1, vocab)
    flat = original
    nan = torch.isnan(flat)
    if bool(nan.any()):
        #: SELECTION ONLY: a NaN ranks as +inf so the set is defined.
        flat = torch.where(nan, torch.full_like(flat, float("inf")), flat)
    k2 = min(vocab, 2 * k)
    top = torch.topk(flat, k2, dim=-1, largest=True, sorted=True)
    kth = top.values[:, k - 1:k]
    #: Canonical order among the candidates: index ascending, then a STABLE
    #: sort by value, which keeps index-ascending order among equal values.
    cand = top.indices.sort(dim=-1).values
    order = torch.sort(flat.gather(-1, cand), dim=-1, descending=True,
                       stable=True).indices
    idx = cand.gather(-1, order)[:, :k]
    if k2 < vocab:
        #: Ties at the K-th value that reach past the 2K candidates: only
        #: there can an entry outside the candidates belong to the set.
        spill = (top.values[:, -1] >= kth.squeeze(-1)).nonzero().squeeze(-1)
        for start in range(0, int(spill.numel()), _TIE_ROWS):
            rows = spill[start:start + _TIE_ROWS]
            x = flat[rows]
            cut = kth[rows]
            above = x > cut
            level = x == cut
            need = k - above.sum(dim=-1, keepdim=True)
            rank = level.cumsum(dim=-1, dtype=torch.int32)
            keep = above | (level & (rank <= need))
            #: `nonzero` is row-major: each row's k columns, index ascending.
            sel = keep.nonzero()[:, 1].view(-1, k)
            reorder = torch.sort(x.gather(-1, sel), dim=-1, descending=True,
                                 stable=True).indices
            idx[rows] = sel.gather(-1, reorder)
    idx = idx.contiguous()
    return original.gather(-1, idx).reshape(*lead, k), idx.reshape(*lead, k)


def log1mexp(d: torch.Tensor) -> torch.Tensor:
    """``log(1 - exp(d))`` for ``d <= 0``, stable at both ends.

    ``-inf`` at ``d = 0`` (nothing left) and ``0`` at ``d = -inf`` (nothing
    removed). The two branches are the standard split at ``-log 2``.
    """
    return torch.where(d > -0.6931471805599453,
                       torch.log(-torch.expm1(d)),
                       torch.log1p(-torch.exp(d)))


def topk_tail_entropy(support_log_probs: torch.Tensor,
                      tail_log_prob: torch.Tensor, *,
                      has_tail: bool) -> torch.Tensor:
    """Shannon entropy of the reference's ``K+1``-category distribution.

    The categories are its own Top-K entries and, when ``K < |V|``, one tail
    bucket; their probabilities are the ABSOLUTE full-softmax ones, never
    renormalized inside the Top-K::

        H^{K+1} = - sum_{i in S} p_i log p_i - p_tail log p_tail

    with ``0 log 0 = 0``. Reduced in float64 from the stored float32 log
    probabilities ([..., K] is small, so this costs nothing) and returned as
    float32. The number of categories is the caller's to derive from the same
    ``has_tail`` (:func:`topk_tail_categories`); nothing here assumes ``K``.
    """
    lp = support_log_probs.double()
    h = -(torch.where(torch.isneginf(lp), torch.zeros_like(lp), lp.exp() * lp)
          ).sum(dim=-1)
    if has_tail:
        t = tail_log_prob.double()
        h = h - torch.where(torch.isneginf(t), torch.zeros_like(t), t.exp() * t)
    return h.float()


def topk_tail_categories(top_k: int, vocab_size: int) -> int:
    """How many categories :func:`topk_tail_entropy` ranges over: ``K`` plus
    one tail bucket when ``K < |V|``, else ``|V|``."""
    k, v = int(top_k), int(vocab_size)
    return k + 1 if k < v else v


# --- teacher retrieval and the union support ---------------------------------

@dataclass(frozen=True)
class TeacherRetrieval:
    """The reference's log probability of ANY vocabulary entry, at fixed rows.

    ``hidden`` ``[N, H]`` is the reference's final (normalized) hidden state at
    each row and ``log_z`` ``[N]`` its full-vocabulary log-normalizer under the
    same logit definition; ``head`` is a
    :class:`~aadistill.initialization.scoring.output_head.LinearOutputHead`
    (anything with ``token_logits(hidden, token_ids)`` returning float32). A
    retrieved log probability is ``head.token_logits(h, ids) - log_z`` -- the
    same arithmetic the sketch's own log probabilities were formed with, so a
    retrieved entry and a sketched one are values of one distribution.
    """

    hidden: torch.Tensor
    log_z: torch.Tensor
    head: Any

    def __post_init__(self) -> None:
        if self.hidden.ndim != 2 or self.log_z.ndim != 1:
            raise ValueError(
                f"retrieval needs [N, H] hidden and [N] log_z, got "
                f"{tuple(self.hidden.shape)} and {tuple(self.log_z.shape)}")
        if int(self.hidden.shape[0]) != int(self.log_z.shape[0]):
            raise ValueError("hidden and log_z disagree on the row count")

    @property
    def rows(self) -> int:
        return int(self.hidden.shape[0])

    def log_probs(self, token_ids: torch.Tensor, start: int = 0,
                  stop: int | None = None) -> torch.Tensor:
        """``[n, M]`` float32 log probabilities of ``token_ids`` ``[n, M]`` at
        rows ``start:stop``."""
        stop = self.rows if stop is None else int(stop)
        h = self.hidden[start:stop]
        z = self.log_z[start:stop].to(h.device).float()
        return (self.head.token_logits(h, token_ids.to(h.device))
                - z[:, None])

    def select(self, rows: torch.Tensor) -> TeacherRetrieval:
        """The rows a boolean ``[N]`` mask keeps, in order."""
        keep = rows.to(self.hidden.device).bool()
        return TeacherRetrieval(self.hidden[keep], self.log_z[keep], self.head)


@dataclass
class UnionKLTerms:
    """Per-position results of the union-support reduction, ``[N]`` each.

    ``kl`` is the forward KL on ``S_T ∪ S_C`` + one tail -- the quantity a
    union-support objective weights. Everything else is evidence: the
    reference-Top-K + tail KL of the SAME pair (``kl_reference_topk``, the
    lower bound), the candidate's log-normalizer, the union's size and
    overlap, both tail masses, and -- when asked for -- the reverse KL on the
    same partition.
    """

    kl: torch.Tensor
    log_z: torch.Tensor
    kl_reference_topk: torch.Tensor
    union_size: torch.Tensor
    overlap: torch.Tensor
    reference_union_tail_log: torch.Tensor
    candidate_union_tail_log: torch.Tensor
    reverse_kl: torch.Tensor | None = None

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """The per-position evidence a record carries beside the KL."""
        return {"kl_reference_topk": self.kl_reference_topk,
                "union_size": self.union_size,
                "union_overlap": self.overlap,
                "reference_union_tail_mass":
                    self.reference_union_tail_log.exp(),
                "candidate_union_tail_mass":
                    self.candidate_union_tail_log.exp()}


@torch.no_grad()
def union_forward_kl_terms(support_indices: torch.Tensor,
                           support_log_probs: torch.Tensor,
                           tail_log_prob: torch.Tensor,
                           cand_logits: torch.Tensor,
                           retrieval: TeacherRetrieval, *,
                           top_k: int, has_tail: bool,
                           reverse: bool = False,
                           chunk: int = 512) -> UnionKLTerms:
    """Forward KL on ``S_T ∪ S_C`` plus one tail, per position.

    ``support_indices`` / ``support_log_probs`` ``[..., K]`` and
    ``tail_log_prob`` ``[...]`` are the reference's sketch (built with
    :func:`deterministic_topk`); ``cand_logits`` is ``[..., V]``; ``retrieval``
    covers the same positions flattened row-major (``[N, H]``). Generic over
    leading dimensions like :func:`sketch_forward_kl_terms`, and reduced in
    blocks of ``chunk`` rows so the transient is ``O(chunk * V)``.

    Per row:

    * ``S_C`` = the candidate's deterministic Top-``top_k`` of its own logits;
      an entry of ``S_C`` already in ``S_T`` is the SAME bucket (dedup by
      sorted search), the rest are candidate-only ``E``;
    * reference log probabilities: stored on ``S_T``, retrieved on ``E``;
    * candidate log probabilities: gathered from its logits against its own
      full-vocabulary normalizer;
    * the candidate's tail outside ``U`` from its complement logits with
      ``S_T`` and ``S_C`` masked (no subtraction);
    * the reference's tail outside ``U``: ``log(exp(tail_T) - sum_E p)`` in
      log space, relative to its own Top-K tail -- the only subtraction, and
      it is relative to the tail, not to one. Rounding that would make it
      negative makes it zero.

    Zero reference mass contributes exactly zero; reference mass where the
    candidate has none is ``+inf`` and stays ``+inf`` (:func:`_term`). NaN
    candidate logits propagate to NaN, never to a finite score.
    """
    vocab = int(cand_logits.shape[-1])
    lead = tuple(cand_logits.shape[:-1])
    k_t = int(support_indices.shape[-1])
    idx_t = support_indices.reshape(-1, k_t)
    lp_t = support_log_probs.reshape(-1, k_t)
    tail_t = tail_log_prob.reshape(-1)
    cand_all = cand_logits.reshape(-1, vocab)
    n = int(cand_all.shape[0])
    if retrieval.rows != n:
        raise ValueError(
            f"the retrieval covers {retrieval.rows} rows and the candidate "
            f"{n}; they do not describe the same positions")
    if int(idx_t.shape[0]) != n:
        raise ValueError("the reference sketch and the candidate disagree on "
                         "the position count")
    k_c = max(1, min(int(top_k), vocab))
    dev = cand_all.device
    outs = {name: torch.empty(n, dtype=torch.float32, device=dev)
            for name in ("kl", "log_z", "kl_ref", "size", "overlap",
                         "p_tail_u", "q_tail_u", "rkl")}
    step = max(1, int(chunk))
    for start in range(0, n, step):
        stop = min(start + step, n)
        #: The candidate's Top-K selected in ITS OWN dtype (exact widening
        #: preserves order and ties), before the float32 copy exists.
        _, s_c = deterministic_topk(cand_all[start:stop], k_c)
        #: OWNED: the tail is computed by masking this tensor.
        cand = owned_float32(cand_all[start:stop])
        log_z = torch.logsumexp(cand, dim=-1, keepdim=True)
        st = idx_t[start:stop].to(dev)
        p_t_log = lp_t[start:stop].to(dev).float()
        q_t_log = cand.gather(-1, st) - log_z
        q_c_log = cand.gather(-1, s_c) - log_z
        #: DEDUP: an S_C entry inside S_T is one bucket, scored once, on the
        #: reference's stored value.
        st_sorted = st.sort(dim=-1).values
        pos = torch.searchsorted(st_sorted, s_c).clamp(max=k_t - 1)
        in_t = st_sorted.gather(-1, pos) == s_c
        extra = ~in_t
        #: RETRIEVAL ONLY FOR E: candidate-only entries first (stable), the
        #: widest E in the block, and a row's columns past its own |E| filled
        #: with its first E entry -- a duplicate, so the head reads no extra
        #: column for it -- and masked below.
        order = in_t.to(torch.int8).argsort(dim=-1, stable=True)
        width = int(extra.sum(dim=-1).max()) if bool(extra.any()) else 0
        p_c_log = torch.full_like(q_c_log, float("-inf"))
        if width:
            cols = order[:, :width]
            ids_e = s_c.gather(-1, cols)
            valid = extra.gather(-1, cols)
            ids_e = torch.where(valid, ids_e, ids_e[:, :1])
            got = retrieval.log_probs(ids_e, start, stop).to(dev).float()
            p_c_log.scatter_(-1, cols, torch.where(
                valid, got, torch.full_like(got, float("-inf"))))
        kl_t = _term(p_t_log.exp(), p_t_log, q_t_log).sum(dim=-1)
        #: `where` rather than a multiply: a non-E column must contribute an
        #: exact zero whatever `_term` produced there.
        kl_e = torch.where(extra, _term(p_c_log.exp(), p_c_log, q_c_log),
                           torch.zeros_like(q_c_log)).sum(dim=-1)
        size = k_t + extra.sum(dim=-1)
        overlap = in_t.sum(dim=-1)
        if has_tail:
            tail = tail_t[start:stop].to(dev).float()
            #: The candidate outside U, from its own complement logits.
            cand.scatter_(-1, st, float("-inf"))
            cand.scatter_(-1, s_c, float("-inf"))
            q_tail_u = torch.logsumexp(cand, dim=-1) - log_z.squeeze(-1)
            #: The reference outside U: its Top-K tail minus E's mass.
            lse_e = torch.logsumexp(p_c_log, dim=-1)
            d = (lse_e - tail).clamp(max=0.0)
            p_tail_u = torch.where(torch.isneginf(tail), tail,
                                   tail + log1mexp(d))
            kl = kl_t + kl_e + _term(p_tail_u.exp(), p_tail_u, q_tail_u)
            #: The reference-Top-K + tail KL of the same pair: the candidate's
            #: mass outside S_T is its mass outside U plus its mass on E, an
            #: ADDITION in log space -- no second vocabulary pass.
            q_e = torch.where(extra, q_c_log,
                              torch.full_like(q_c_log, float("-inf")))
            q_tail_t = torch.logaddexp(q_tail_u, torch.logsumexp(q_e, dim=-1))
            kl_ref = kl_t + _term(tail.exp(), tail, q_tail_t)
        else:
            #: K >= V: S_T is the vocabulary, E is empty, there is no tail.
            q_tail_u = torch.full_like(kl_t, float("-inf"))
            p_tail_u = torch.full_like(kl_t, float("-inf"))
            kl = kl_t + kl_e
            kl_ref = kl_t
        if reverse:
            #: Reverse KL on the SAME partition, roles swapped.
            rkl = (_term(q_t_log.exp(), q_t_log, p_t_log).sum(dim=-1)
                   + torch.where(extra, _term(q_c_log.exp(), q_c_log, p_c_log),
                                 torch.zeros_like(q_c_log)).sum(dim=-1))
            if has_tail:
                rkl = rkl + _term(q_tail_u.exp(), q_tail_u, p_tail_u)
            outs["rkl"][start:stop] = rkl
        outs["kl"][start:stop] = kl
        outs["log_z"][start:stop] = log_z.squeeze(-1)
        outs["kl_ref"][start:stop] = kl_ref
        outs["size"][start:stop] = size.float()
        outs["overlap"][start:stop] = overlap.float()
        outs["p_tail_u"][start:stop] = p_tail_u
        outs["q_tail_u"][start:stop] = q_tail_u
        del cand, log_z, s_c, q_t_log, q_c_log, p_c_log

    def shaped(name: str) -> torch.Tensor:
        return outs[name].reshape(lead)
    return UnionKLTerms(
        kl=shaped("kl"), log_z=shaped("log_z"),
        kl_reference_topk=shaped("kl_ref"), union_size=shaped("size"),
        overlap=shaped("overlap"),
        reference_union_tail_log=shaped("p_tail_u"),
        candidate_union_tail_log=shaped("q_tail_u"),
        reverse_kl=shaped("rkl") if reverse else None)


# --- the reference sketch ---------------------------------------------------

@dataclass(frozen=True)
class ReferenceDistributionSketch:
    """Everything a ``K+1`` reduction needs from the reference, and nothing more.

    The persistent reference state is ``O(T*K)`` rather than ``O(T*V)``. Shapes,
    with ``T`` prediction positions:

    ======================  ==========  ==================================
    ``support_indices``     ``[T, K]``  reference Top-K entries, descending
    ``support_log_probs``   ``[T, K]``  their normalized log probabilities
    ``tail_log_prob``       ``[T]``     ``log(1 - sum of the above)``
    ``top1_token``          ``[T]``     reference argmax, for exact top-1
    ``target_log_prob``     ``[T]``     exact reference CE at the gold token
    ======================  ==========  ==================================

    ``top1_token`` and ``target_log_prob`` are carried so that top-1 agreement and
    CE stay EXACT under the coarsened partition: a gold token does not have to be
    inside the Top-K, and a metric that silently became approximate because the
    KL partition changed would be a second, unannounced protocol change.
    """

    support_indices: torch.Tensor
    support_log_probs: torch.Tensor
    tail_log_prob: torch.Tensor
    top1_token: torch.Tensor
    target_log_prob: torch.Tensor
    top_k: int
    vocab_size: int

    def __post_init__(self) -> None:
        t, k = self.support_indices.shape
        if k != self.top_k:
            raise ValueError(
                f"support_indices has width {k} but top_k is {self.top_k}")
        for name, want in (("support_log_probs", (t, k)),
                           ("tail_log_prob", (t,)),
                           ("top1_token", (t,)),
                           ("target_log_prob", (t,))):
            got = tuple(getattr(self, name).shape)
            if got != want:
                raise ValueError(f"{name} is {got}, expected {want}")

    @property
    def positions(self) -> int:
        return int(self.support_indices.shape[0])

    def bytes_held(self) -> int:
        return sum(t.numel() * t.element_size() for t in (
            self.support_indices, self.support_log_probs, self.tail_log_prob,
            self.top1_token, self.target_log_prob))


def sketch_reference(logits: torch.Tensor, targets: torch.Tensor, *,
                     top_k: int, chunk: int = 512,
                     ) -> ReferenceDistributionSketch:
    """Reduce ``[T, V]`` reference logits to an ``O(T*K)`` sketch.

    The ``reference_topk_tail_v1`` sketch, selected by ``torch.topk``: every
    recorded decision on that support was made on exactly this arithmetic, and
    the union support's
    sketch (:func:`sketch_reference_deterministic`) is a separate entry point so
    that nothing here moves.

    ``logits`` is ``[T_pred, V]`` already aligned to the positions being scored
    and ``targets`` is ``[T_pred]``. The vocabulary comes from the tensor, never
    from a constant: this must work for any model this framework loads.

    Chunked in float32 for the same reason the full reducer is — a full-sequence
    float32 softmax at a six-figure vocabulary is a needless spike on the one
    device that also holds the model.

    ``top_k`` larger than the vocabulary is clamped to the vocabulary, which makes
    the sketch lossless and the resulting KL equal to the full-vocabulary KL. That
    is the identity the correctness tests pin, so it is a supported input rather
    than an error.
    """
    return _sketch(logits, targets, top_k=top_k, chunk=chunk,
                   deterministic=False)[0]


def sketch_reference_deterministic(
        logits: torch.Tensor, targets: torch.Tensor, *, top_k: int,
        chunk: int = 512) -> tuple[ReferenceDistributionSketch, torch.Tensor]:
    """``(sketch, log_z)``: the union support's reference sketch.

    The same reduction as :func:`sketch_reference` with the Top-K chosen by
    :func:`deterministic_topk`, and the full-vocabulary log-normalizer
    ``log_z`` ``[T]`` returned beside it -- a retrieved entry's log
    probability is its logit minus exactly this value.
    """
    return _sketch(logits, targets, top_k=top_k, chunk=chunk,
                   deterministic=True)


def _sketch(logits: torch.Tensor, targets: torch.Tensor, *, top_k: int,
            chunk: int, deterministic: bool
            ) -> tuple[ReferenceDistributionSketch, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"reference logits must be [T, V], got "
                         f"{tuple(logits.shape)}")
    t_pred, vocab = int(logits.shape[0]), int(logits.shape[1])
    if int(targets.shape[0]) != t_pred:
        raise ValueError("logits and targets disagree on the position count")
    if t_pred == 0:
        raise ValueError("a sketch over zero positions measures nothing")
    k = max(1, min(int(top_k), vocab))
    norm = torch.empty((t_pred,), dtype=torch.float32, device=logits.device)

    idx = torch.empty((t_pred, k), dtype=torch.long, device=logits.device)
    lp = torch.empty((t_pred, k), dtype=torch.float32, device=logits.device)
    tail = torch.empty((t_pred,), dtype=torch.float32, device=logits.device)
    top1 = torch.empty((t_pred,), dtype=torch.long, device=logits.device)
    tgt = torch.empty((t_pred,), dtype=torch.float32, device=logits.device)

    has_tail = k < vocab
    for start in range(0, t_pred, max(1, int(chunk))):
        stop = min(start + max(1, int(chunk)), t_pred)
        #: OWNED, because the tail is computed by masking this tensor below.
        block = owned_float32(logits[start:stop])
        #: ONE log-normalizer over the FULL vocabulary. Every log probability
        #: below is normalized against it, including the gold token's -- so CE
        #: stays exact whether or not the gold token is in the Top-K.
        log_z = torch.logsumexp(block, dim=-1, keepdim=True)
        norm[start:stop] = log_z.squeeze(-1)
        if deterministic:
            top_values, top_indices = deterministic_topk(block, k)
        else:
            top = torch.topk(block, k, dim=-1, largest=True, sorted=True)
            top_values, top_indices = top.values, top.indices
        idx[start:stop] = top_indices
        lp[start:stop] = top_values - log_z
        top1[start:stop] = top_indices[:, 0]
        #: EVERYTHING THAT NEEDS THE UNMODIFIED ROW IS CAPTURED FIRST. The gold
        #: token may be inside the support, so its logit must be read before the
        #: support is masked out.
        tgt[start:stop] = (
            block.gather(1, targets[start:stop].view(-1, 1).to(block.device))
            .squeeze(1) - log_z.squeeze(-1))
        #: Only now: the tail from the complement's own logits. `block` is
        #: destroyed by this and is not read again.
        if has_tail:
            tail[start:stop] = complement_log_mass(
                block, top_indices, log_z.squeeze(-1))
        else:
            #: K >= V: the support IS the vocabulary and there is structurally no
            #: tail bucket. `-inf` records an empty one; no consumer reads it,
            #: because `has_tail` is false everywhere downstream.
            tail[start:stop] = float("-inf")
        del block, log_z, top_values, top_indices

    return ReferenceDistributionSketch(
        support_indices=idx, support_log_probs=lp, tail_log_prob=tail,
        top1_token=top1, target_log_prob=tgt, top_k=k,
        vocab_size=vocab), norm


# --- the K+1 reduction ------------------------------------------------------

def distortion_on_reference_support(
    sketch: ReferenceDistributionSketch,
    cand_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tags: Mapping[str, torch.Tensor] | None = None,
    weights: torch.Tensor | None = None,
    chunk: int = 512,
    position_sink: Any = None,
) -> DistortionSums:
    """The six quantities, reduced on the reference's ``K+1`` partition.

    Same output contract as the full-vocabulary
    :func:`~aadistill.initialization.statistics.contribution.distortion`, so every
    consumer — ``greedy_removal``, the state evaluator, the domain-balanced
    aggregation — works unchanged. Which quantities change meaning and which do
    not is the whole point:

    ``kl``, ``reverse_kl``, ``tagged``
        on the ``K+1`` partition. **Both directions on the SAME partition**: one
        state evaluation carrying a reference-defined forward KL and a
        candidate-defined reverse KL would be two incompatible partitions in one
        number.
    ``ref_ce``, ``abl_ce``, ``top1_agree``
        EXACT. CE uses the exact target logit against the full log-normalizer,
        and top-1 compares true argmaxes. A gold token outside the Top-K is
        ordinary, not a special case.

    ``weights`` and ``tags`` keep the semantics they have in the full reducer: a
    tag selects a diagnostic subset of positions, the weights say how much each
    position counts, and a tagged accumulator takes the product.

    ``position_sink(start, stop, kl, cand_log_z)``, when given, receives each
    block's per-position forward KL and candidate log-normalizer as soon as
    they exist, so a per-position objective can be formed without a second
    vocabulary-wide pass. It reads; it cannot change anything accumulated
    here, and when absent nothing in this function's arithmetic differs.
    """
    if cand_logits.ndim != 2:
        raise ValueError(f"candidate logits must be [T, V], got "
                         f"{tuple(cand_logits.shape)}")
    t_pred = sketch.positions
    if int(cand_logits.shape[0]) != t_pred:
        raise ValueError(
            f"the sketch covers {t_pred} positions and the candidate logits "
            f"{int(cand_logits.shape[0])}; they do not describe the same "
            "positions, so no divergence between them exists")
    if int(cand_logits.shape[1]) != sketch.vocab_size:
        raise ValueError(
            f"the sketch was built at vocabulary {sketch.vocab_size} and the "
            f"candidate has {int(cand_logits.shape[1])}; the two models are not "
            "logit-comparable")
    if int(targets.shape[0]) != t_pred:
        raise ValueError("targets and the sketch disagree on the position count")

    tags = dict(tags or {})
    for name, mask in tags.items():
        if int(mask.shape[0]) != t_pred:
            raise ValueError(f"tag {name!r} mask has the wrong length")

    device = cand_logits.device
    out = DistortionSums()
    w = _position_weights(weights, t_pred, device, "distortion_topk")
    out.weight = float(t_pred) if w is None else float(w.sum())
    resident = _reduce_on_device(device)
    acc = torch.zeros(5, dtype=torch.float64, device=device)
    tag_acc = {name: torch.zeros(3, dtype=torch.float64, device=device)
               for name in tags}

    has_tail = sketch.top_k < sketch.vocab_size
    step = max(1, int(chunk))
    for start in range(0, t_pred, step):
        stop = min(start + step, t_pred)
        #: OWNED: the candidate's tail is computed by masking this tensor.
        cand = owned_float32(cand_logits[start:stop])
        #: The candidate's OWN full-vocabulary normalizer. Not optional: its
        #: probabilities on the reference support only mean anything relative to
        #: its whole distribution.
        cand_log_z = torch.logsumexp(cand, dim=-1, keepdim=True)

        p_log = sketch.support_log_probs[start:stop].to(device).float()
        p_tail_log = sketch.tail_log_prob[start:stop].to(device).float()
        support = sketch.support_indices[start:stop].to(device)
        #: GATHERED ON THE REFERENCE SUPPORT. The candidate does not choose it.
        q_log = cand.gather(1, support) - cand_log_z
        tgt = targets[start:stop].to(device)
        #: Captured before the mask, since the gold token may be in the support.
        abl_ce = -(cand.gather(1, tgt.view(-1, 1)).squeeze(1)
                   - cand_log_z.squeeze(-1))
        cand_argmax = cand.argmax(dim=-1)
        #: The candidate's tail from ITS OWN complement logits, against the
        #: reference support. `cand` is destroyed here and not read again.
        q_tail_log = (complement_log_mass(cand, support,
                                          cand_log_z.squeeze(-1))
                      if has_tail else None)

        p = p_log.exp()
        q = q_log.exp()

        #: Forward KL on the K+1 partition. `_term` is the convention that makes
        #: the empty-bucket limits right: a bucket with zero reference mass
        #: contributes zero however small the candidate's mass is, which is
        #: `0 * log(0/q) = 0`; a bucket with reference mass and ZERO candidate
        #: mass is a genuine infinity and is reported as one rather than hidden
        #: behind a floor, because silently clamping it would make an impossible
        #: candidate look merely bad.
        kl_support = _term(p, p_log, q_log).sum(dim=-1)
        rkl_support = _term(q, q_log, p_log).sum(dim=-1)
        if not has_tail:
            #: K >= V: the partition is the vocabulary and has no tail bucket.
            kl, rkl = kl_support, rkl_support
        else:
            #: THE TAIL TERM IS NOT CONDITIONAL. For finite logits and K < V a
            #: mathematical complement exists, and the only thing that makes the
            #: forward term vanish is zero REFERENCE mass -- which `_term`
            #: handles. If the candidate has zero complement mass while the
            #: reference does not, forward KL is `+inf` and is PRESERVED: that is
            #: a candidate assigning zero probability to something the reference
            #: does, and reporting it as finite would be a false measurement.
            p_tail, q_tail = p_tail_log.exp(), q_tail_log.exp()
            kl = kl_support + _term(p_tail, p_tail_log, q_tail_log)
            #: Reverse KL on the SAME partition, roles swapped.
            rkl = rkl_support + _term(q_tail, q_tail_log, p_tail_log)
        if position_sink is not None:
            position_sink(start, stop, kl, cand_log_z.squeeze(-1))

        ref_ce = -sketch.target_log_prob[start:stop].to(device).float()
        agree = (cand_argmax
                 == sketch.top1_token[start:stop].to(device)).float()

        block_w = None if w is None else w[start:stop]
        if block_w is None:
            acc[0] += kl.sum().double()
            acc[1] += rkl.sum().double()
            acc[2] += ref_ce.sum().double()
            acc[3] += abl_ce.sum().double()
            acc[4] += agree.sum().double()
        else:
            acc[0] += (kl * block_w).sum().double()
            acc[1] += (rkl * block_w).sum().double()
            acc[2] += (ref_ce * block_w).sum().double()
            acc[3] += (abl_ce * block_w).sum().double()
            acc[4] += (agree * block_w).sum().double()

        for name, mask in tags.items():
            m = mask[start:stop].to(device)
            if block_w is None:
                tag_acc[name][0] += (kl * m).sum().double()
                tag_acc[name][1] += m.sum().double()
            else:
                tag_acc[name][0] += (kl * m * block_w).sum().double()
                tag_acc[name][1] += (m * block_w).sum().double()
            tag_acc[name][2] += m.sum().double()

        del cand, cand_log_z, p_log, p_tail_log, support, q_log, kl, rkl

    host = acc.cpu() if resident else acc
    out.positions = t_pred
    out.kl = float(host[0])
    out.reverse_kl = float(host[1])
    out.ref_ce = float(host[2])
    out.abl_ce = float(host[3])
    out.top1_agree = float(host[4])
    for name, a in tag_acc.items():
        h = a.cpu() if resident else a
        out.tagged[name] = [float(h[0]), float(h[1]), float(h[2])]
    return out


def distortion_on_union_support(
    sketch: ReferenceDistributionSketch,
    retrieval: TeacherRetrieval,
    cand_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tags: Mapping[str, torch.Tensor] | None = None,
    weights: torch.Tensor | None = None,
    chunk: int = 512,
    position_sink: Any = None,
) -> DistortionSums:
    """The six quantities, with both KL directions on ``S_T ∪ S_C`` + tail.

    The union-support twin of :func:`distortion_on_reference_support`, with the
    same output contract and the same split: ``kl`` / ``reverse_kl`` /
    ``tagged`` on the union partition (both directions on the SAME partition),
    ``ref_ce`` / ``abl_ce`` / ``top1_agree`` EXACT -- the reference's from the
    sketch (built against its full normalizer), the candidate's from its own
    logits. ``position_sink(start, stop, kl, cand_log_z, union=terms)``
    receives each block's per-position values and the block's
    :class:`UnionKLTerms`, read-only.
    """
    if cand_logits.ndim != 2:
        raise ValueError(f"candidate logits must be [T, V], got "
                         f"{tuple(cand_logits.shape)}")
    t_pred = sketch.positions
    if int(cand_logits.shape[0]) != t_pred or retrieval.rows != t_pred:
        raise ValueError(
            f"the sketch covers {t_pred} positions, the candidate "
            f"{int(cand_logits.shape[0])} and the retrieval {retrieval.rows}; "
            "they do not describe the same positions")
    if int(cand_logits.shape[1]) != sketch.vocab_size:
        raise ValueError(
            f"the sketch was built at vocabulary {sketch.vocab_size} and the "
            f"candidate has {int(cand_logits.shape[1])}; the two models are not "
            "logit-comparable")
    if int(targets.shape[0]) != t_pred:
        raise ValueError("targets and the sketch disagree on the position count")
    tags = dict(tags or {})
    for name, mask in tags.items():
        if int(mask.shape[0]) != t_pred:
            raise ValueError(f"tag {name!r} mask has the wrong length")

    device = cand_logits.device
    out = DistortionSums()
    w = _position_weights(weights, t_pred, device, "distortion_union")
    out.weight = float(t_pred) if w is None else float(w.sum())
    resident = _reduce_on_device(device)
    acc = torch.zeros(5, dtype=torch.float64, device=device)
    tag_acc = {name: torch.zeros(3, dtype=torch.float64, device=device)
               for name in tags}
    has_tail = sketch.top_k < sketch.vocab_size
    step = max(1, int(chunk))
    for start in range(0, t_pred, step):
        stop = min(start + step, t_pred)
        block = cand_logits[start:stop]
        part = TeacherRetrieval(retrieval.hidden[start:stop],
                                retrieval.log_z[start:stop], retrieval.head)
        terms = union_forward_kl_terms(
            sketch.support_indices[start:stop],
            sketch.support_log_probs[start:stop],
            sketch.tail_log_prob[start:stop], block, part,
            top_k=sketch.top_k, has_tail=has_tail, reverse=True, chunk=step)
        kl, rkl = terms.kl, terms.reverse_kl
        if position_sink is not None:
            position_sink(start, stop, kl, terms.log_z, union=terms)
        tgt = targets[start:stop].to(device)
        cand32 = block.float()
        abl_ce = -(cand32.gather(1, tgt.view(-1, 1)).squeeze(1) - terms.log_z)
        cand_argmax = cand32.argmax(dim=-1)
        del cand32
        ref_ce = -sketch.target_log_prob[start:stop].to(device).float()
        agree = (cand_argmax
                 == sketch.top1_token[start:stop].to(device)).float()
        block_w = None if w is None else w[start:stop]
        for slot, value in enumerate((kl, rkl, ref_ce, abl_ce, agree)):
            acc[slot] += (value.sum() if block_w is None
                          else (value * block_w).sum()).double()
        for name, mask in tags.items():
            m = mask[start:stop].to(device)
            if block_w is None:
                tag_acc[name][0] += (kl * m).sum().double()
                tag_acc[name][1] += m.sum().double()
            else:
                tag_acc[name][0] += (kl * m * block_w).sum().double()
                tag_acc[name][1] += (m * block_w).sum().double()
            tag_acc[name][2] += m.sum().double()
        del terms, kl, rkl

    host = acc.cpu() if resident else acc
    out.positions = t_pred
    out.kl = float(host[0])
    out.reverse_kl = float(host[1])
    out.ref_ce = float(host[2])
    out.abl_ce = float(host[3])
    out.top1_agree = float(host[4])
    for name, a in tag_acc.items():
        h = a.cpu() if resident else a
        out.tagged[name] = [float(h[0]), float(h[1]), float(h[2])]
    return out


def _term(p: torch.Tensor, p_log: torch.Tensor,
          q_log: torch.Tensor) -> torch.Tensor:
    """``p * (log p - log q)``, with ``p == 0`` contributing exactly zero.

    Needed because an empty bucket gives ``0 * (-inf - x)``, which is NaN in
    floating point and zero in the limit. Masking the multiply rather than
    flooring ``p`` keeps a bucket with real reference mass and no candidate mass
    as ``+inf``, which is the truthful answer for a candidate that assigns zero
    probability to something the reference does.
    """
    empty = p <= 0
    diff = torch.where(empty, torch.zeros_like(p_log), p_log - q_log)
    return torch.where(empty, torch.zeros_like(p), p * diff)


# --- DEPTH's reducer: forward KL only, on the reference support --------------

def sketch_forward_kl(support_indices: torch.Tensor,
                      support_log_probs: torch.Tensor,
                      tail_log_prob: torch.Tensor,
                      cand_logits: torch.Tensor,
                      *, has_tail: bool) -> torch.Tensor:
    """Per-position forward KL on the reference's ``K+1`` partition.

    The KL half of :func:`sketch_forward_kl_terms`, which also returns the
    candidate's log-normalizer; the arithmetic is the same operations in the
    same order either way.
    """
    return sketch_forward_kl_terms(support_indices, support_log_probs,
                                   tail_log_prob, cand_logits,
                                   has_tail=has_tail)[0]


def sketch_forward_kl_terms(support_indices: torch.Tensor,
                            support_log_probs: torch.Tensor,
                            tail_log_prob: torch.Tensor,
                            cand_logits: torch.Tensor,
                            *, has_tail: bool
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(kl, log_z)``: the forward KL and the candidate's log-normalizer.

    ``log_z`` ``[...]`` is returned because a student-confidence objective
    needs it for the Top-K mass, and recomputing a vocabulary-wide reduction
    the KL has already paid for would double this reducer's cost.

    Generic over leading dimensions: ``support_indices`` and
    ``support_log_probs`` are ``[..., K]``, ``tail_log_prob`` is ``[...]`` and
    ``cand_logits`` is ``[..., V]``; the result is ``[...]``. ONE implementation
    serves the per-item and the batched DEPTH paths, so the two cannot drift.

    ``has_tail`` is the caller's and is not inferred from the sketch. A tail value
    exists in the sketch either way, and only the caller knows whether
    ``top_k >= vocab_size`` — trusting the stored value is how a partition that
    should have no tail acquires a spurious one.

    Forward KL ONLY. DEPTH reads exactly this quantity and discards the other
    five, 260 candidate subsets x 67 items per expansion, which is why it does
    not come through the six-quantity reducer.
    """
    #: OWNED: the tail is computed by masking this tensor, and `.float()` would
    #: alias a caller-owned fp32 block -- the reference a search has cached or the
    #: candidate the operator still holds.
    cand = owned_float32(cand_logits)
    log_z = torch.logsumexp(cand, dim=-1, keepdim=True)
    q_log = cand.gather(-1, support_indices) - log_z
    p_log = support_log_probs.float()

    kl = _term(p_log.exp(), p_log, q_log).sum(dim=-1)
    if not has_tail:
        #: K >= V: no tail bucket exists.
        return kl, log_z.squeeze(-1)
    p_tail_log = tail_log_prob.float()
    #: The candidate's tail from ITS OWN complement logits on the reference
    #: support. NOT `1 - sum(support)`, which crosses float32 resolution once the
    #: support holds nearly all the mass and is what let a coarse KL exceed the
    #: full one. `cand` is destroyed here.
    q_tail_log = complement_log_mass(cand, support_indices, log_z.squeeze(-1))
    #: UNCONDITIONAL. Only zero REFERENCE mass makes the forward term vanish, and
    #: `_term` is what decides that. A candidate with zero complement mass against
    #: a non-zero reference tail is `+inf`, and that is preserved.
    return (kl + _term(p_tail_log.exp(), p_tail_log, q_tail_log),
            log_z.squeeze(-1))


def sketch_forward_kl_mean(sketch: ReferenceDistributionSketch,
                           cand_logits: torch.Tensor,
                           *, weights: torch.Tensor | None = None) -> float:
    """``forward_kl_mean``'s quantity, on the reference support. One item."""
    kl = sketch_forward_kl(
        sketch.support_indices.to(cand_logits.device),
        sketch.support_log_probs.to(cand_logits.device),
        sketch.tail_log_prob.to(cand_logits.device),
        cand_logits, has_tail=sketch.top_k < sketch.vocab_size)
    if weights is None:
        return float(kl.mean())
    w = weights.to(kl.device).float()
    total = float(w.sum())
    if total <= 0:
        raise ValueError(
            "every scoring weight is zero, so the weighted mean has no "
            "denominator; a zero-weight item cannot contribute a score")
    return float((kl * w).sum() / total)


def sketch_forward_kl_mean_batch(support_indices: torch.Tensor,
                                 support_log_probs: torch.Tensor,
                                 tail_log_prob: torch.Tensor,
                                 cand_logits: torch.Tensor,
                                 prediction_mask: torch.Tensor,
                                 *, has_tail: bool,
                                 weights: torch.Tensor | None = None,
                                 ) -> torch.Tensor:
    """``forward_kl_mean_batch``'s quantity, on the reference support.

    ``[B]`` out, where ``out[i]`` is item ``i``'s mean over **item i's own** valid
    positions — a PER-ITEM mean, never a pooled one. Pooling would hand a
    1000-position item ten times the influence of a 100-position one and change
    the objective, which is the same reason the full-vocabulary batched reducer
    reduces per row.

    ``weights`` multiplies INTO the validity mask rather than replacing it: the
    mask says which positions exist, the weights say which the objective cares
    about, and a padded position must be zero under both.
    """
    kl = sketch_forward_kl(support_indices, support_log_probs, tail_log_prob,
                           cand_logits, has_tail=has_tail)
    return masked_item_means(kl, prediction_mask, weights=weights)


def masked_item_means(kl: torch.Tensor, prediction_mask: torch.Tensor,
                      *, weights: torch.Tensor | None = None) -> torch.Tensor:
    """``[B]`` per-item weighted means of per-position values ``kl`` ``[B, T]``.

    The reduction half of :func:`sketch_forward_kl_mean_batch`, separated so the
    pre-vocabulary row path (:mod:`aadistill.initialization.scoring.rows`) forms
    its per-item means with the SAME operations: it scatters its selected-row
    values into the dense layout and calls this. A value at a position this
    masks out is never read, which is what lets the row path leave zeros there.
    """
    m = prediction_mask.to(kl.device).float()
    if weights is not None:
        m = m * weights.to(kl.device).float()
    denom = m.sum(dim=-1)
    if bool((denom <= 0).any()):
        raise ValueError(
            "a row has no weighted valid position, so its mean has no "
            "denominator; an item that contributes nothing must not be scored")
    #: MASK BEFORE MULTIPLY, and never `nan_to_num`. A padded or zero-weight
    #: position holds whatever the pad token produced and must contribute zero; a
    #: VALID position holding `+inf` must stay `+inf`, because that is a candidate
    #: assigning zero probability to something the reference does. The previous
    #: `nan_to_num(posinf=0.0)` erased exactly that distinction and could turn a
    #: genuine infinite divergence into a finite -- and favourable -- score.
    effective = m
    safe = torch.where(effective > 0, kl, torch.zeros_like(kl))
    #: float64 to match `forward_kl_mean_batch`, which this stands in for; the
    #: dtype is part of the contract its caller consumes.
    return (safe * effective).double().sum(dim=-1) / denom.double()
