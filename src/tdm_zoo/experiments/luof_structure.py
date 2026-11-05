"""Does the Druk-Ishai rank bottleneck give a linear distinguisher on ``M``?

Three exact measurements on the ``luof`` candidate, smallest claim first:

* Structure (:func:`structure_point`): shape and rank of ``M`` and of every public layer, and the
  shape and parameter count of every secret family. In the split view the middle publics
  ``Enc Pi Ext`` are ``N x N`` of rank ``k``: the bottleneck, public by construction.
* Subspace (:func:`subspace_point`): ``V = {Ext H Enc : H block-diagonal}``, the linear span of
  one secret layer. ``codim V > 0`` means every layer satisfies public linear relations.
* Span (:func:`span_point`): with the publics fixed, the dimension of the linear span of sampled
  ``M``. A proper span is a membership distinguisher, and its orthogonal complement is the set
  of public relations ``M`` satisfies; a full span rules out any linear distinguisher.

Scaling beyond the sizes where sampling real secrets is affordable:

* ``dim V`` (:func:`subspace_point`) always obeys ``dim V <= 4N = O(k)``, so ``dim V / k^2 -> 0``.
  ``method="gram"`` measures the rank of ``G (D_1 x D_2) G^T``, a lower bound on ``dim V`` that
  fits in memory at ``k >= 192``; :func:`block_bound` is the matching upper bound.
* Fast span (:func:`fast_span_point`): as Span, but each layer is drawn block by block, as
  ``sum_b Ext_b H_b Enc_b``, without the dense ``N x N`` matrix ``H``. A linear image of uniform
  ``H``, so uniform over ``V``: the same distribution as the real secrets, not an approximation.
* Block pattern (:func:`pattern_point`): the span with no sampling and no ``k^2``-sized rank.
  Write ``M = Ext H_1 T_2 H_2 ... T_l H_l Enc`` with ``T_i = Enc Pi_i Ext``. A nonzero block
  ``T`` generates every ``Delta x Delta`` matrix as ``X T Y``, and each ``H_i`` can be supported
  on one block, so the span of ``H_1 T_2 ... H_l`` is exactly the set of block matrices supported
  on ``P``, the boolean product of the nonzero-block patterns of the ``T_i``. Therefore

      span{M} = sum_{(b, c) in P} colspan(Ext_b) x rowspan(Enc_c).

  ``certified_full`` is a proof of ``span = k^2`` for that draw of the publics: some block rows
  ``b`` have ``{Enc_c : (b, c) in P}`` spanning ``F^k`` and their ``Ext_b`` span ``F^k`` too.

``dim V`` depends on the sampled codes (the graphs are random), so it varies with the seed.
The codes come first out of the generator in both :func:`subspace_point` and :func:`span_point`,
so the two share codes at equal ``seed``: the ``ell = 1`` span then equals ``dim V`` exactly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..arith.backends.flint_backend import rows_to_flint
from ..fields import GF
from ..ops import linalg
from ..tdm import make_candidate
from ..tdm.di14 import DI14Params, LUOFCodes, di14_dimension, sample_luof_codes
from ..tdm.product import LayeredProduct

PARAMS = DI14Params(8, 6, 4)
DIRECT_MAX_K = 128  # beyond this the k^2-wide generator matrix no longer fits; use the Gram rank
FULL_MARGIN_MAX_K = 96  # beyond this the 25% sample margin costs gigabytes; see fast_sample_count


@dataclass
class StructureRow:
    k: int
    view: str
    ell: int
    prime: int
    seed: int
    N: int
    M_shape: tuple
    M_rank: int
    publics: list  # [layer, shape, rank]
    secrets: list  # [layer, family, shape, nparams]
    seconds: float
    experiment: str = "structure"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubspaceRow:
    k: int
    prime: int
    seed: int
    N: int
    k2: int
    dim_V: int
    codim: int
    fill_percent: float
    seconds: float
    method: str = "direct"  # "gram" is a lower bound on dim V
    params: int | None = None  # dim V <= params: the 4N entries of H
    block_bound: int | None = None  # dim V <= sum_b rank(Ext_b) rank(Enc_b)
    experiment: str = "subspace"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpanRow:
    k: int
    ell: int
    prime: int
    seed: int
    samples: int
    k2: int
    span: int
    codim: int
    distinguisher: bool  # span < k^2: every M lies in a known proper subspace
    relations_hold_on_fresh: bool | None  # a new M satisfies the relations
    relations_reject_uniform: bool | None  # a uniform matrix does not
    seconds: float
    relations: list = field(default_factory=list)  # codim x k^2 basis of the orthogonal complement
    options: dict = field(default_factory=dict)
    experiment: str = "span"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatternRow:
    k: int
    ell: int
    prime: int
    seed: int
    N: int
    blocks: int
    live_blocks: int  # Ext_b != 0 and Enc_b != 0; the others never reach M
    dead_pairs: int  # live (b, c) outside the pattern P
    certified_full: bool
    seconds: float
    options: dict = field(default_factory=dict)
    experiment: str = "pattern"

    def to_dict(self) -> dict:
        return asdict(self)


def sample_count(k: int) -> int:
    """More samples than ``k^2``, so that the span is capped by structure and not by the count."""
    return k * k + k * k // 4 + 20


def fast_sample_count(k: int) -> int:
    """As :func:`sample_count` up to ``FULL_MARGIN_MAX_K``, then ``k^2 + 64``.

    ``M`` is polynomial in the secrets, so ``d`` random samples of a ``d``-dimensional span already
    span it except with probability about ``d deg / p``; the margin only guards that bound. A
    full rank is conclusive for any count.
    """
    return sample_count(k) if k <= FULL_MARGIN_MAX_K else k * k + 64


def structure_point(k: int, view: str, prime: int, ell: int = 3, seed: int = 0) -> StructureRow:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    field_ = GF(prime)
    product = make_candidate("luof", field_, k=k, ell=ell, view=view).sample_product(rng)
    matrix = product.sample(rng).matrix
    public_view = product.public_view()
    publics = [
        [f"A{i + 1}", public.shape, linalg.rank(field_, public_view.public_matrix(i))]
        for i, public in enumerate(public_view.publics)
    ]
    secrets = [
        [f"S{i + 1}", family.name, family.shape, family.nparams]
        for i, family in enumerate(public_view.secret_families)
    ]
    return StructureRow(
        k=k,
        view=view,
        ell=ell,
        prime=prime,
        seed=seed,
        N=di14_dimension(k, PARAMS),
        M_shape=matrix.shape,
        M_rank=linalg.rank(field_, matrix),
        publics=publics,
        secrets=secrets,
        seconds=time.perf_counter() - start,
    )


def _generators(codes: LUOFCodes) -> tuple[np.ndarray, np.ndarray]:
    """Pairs ``(a, c)`` in the same block, with both ``Ext[:, a]`` and ``Enc[c, :]`` nonzero."""
    enc, ext = codes.encoder.to_dense(), codes.extractor.to_dense()
    block = codes.block_size
    a, b = np.divmod(np.arange(codes.inner_dimension * block), block)
    c = a // block * block + b
    keep = ext[:, a].any(axis=0) & enc[c].any(axis=1)
    return a[keep], c[keep]


def direct_dimension(field_, codes: LUOFCodes) -> int:
    """``dim V`` as the rank of the outer products ``Ext[:, a] Enc[c, :]``, streamed row by row."""
    enc, ext = codes.encoder.to_dense(), codes.extractor.to_dense()
    a, c = _generators(codes)
    rows = (field_.mul(ext[:, i][:, None], enc[j][None, :]).ravel() for i, j in zip(a, c, strict=True))
    return int(rows_to_flint(rows, (a.size, ext.shape[0] ** 2), field_.p).rank())


def gram_dimension(field_, codes: LUOFCodes, rng: np.random.Generator) -> int:
    """Rank of ``G (D_1 x D_2) G^T``: a lower bound on ``dim V``, never wider than ``G`` is tall.

    The entry at ``(a, c), (a', c')`` is ``(Ext^T D_1 Ext)[a, a'] (Enc D_2 Enc^T)[c, c']``.
    """
    enc, ext = codes.encoder.to_dense(), codes.extractor.to_dense()
    a, c = _generators(codes)
    d1, d2 = field_.random(rng, ext.shape[0]), field_.random(rng, ext.shape[0])
    x = field_.matmul(field_.mul(ext.T, d1[None, :]), ext)
    y = field_.matmul(field_.mul(enc, d2[None, :]), enc.T)
    rows = (field_.mul(x[i, a], y[j, c]) for i, j in zip(a, c, strict=True))
    return int(rows_to_flint(rows, (a.size, a.size), field_.p).rank())


def block_bound(field_, codes: LUOFCodes) -> int:
    """``sum_b rank(Ext_b) rank(Enc_b)``, an upper bound on ``dim V``."""
    enc, ext = codes.encoder.to_dense(), codes.extractor.to_dense()
    size = codes.block_size
    return sum(
        linalg.rank(field_, ext[:, b : b + size]) * linalg.rank(field_, enc[b : b + size])
        for b in range(0, codes.inner_dimension, size)
    )


def subspace_point(k: int, prime: int, seed: int = 0, method: str | None = None) -> SubspaceRow:
    start = time.perf_counter()
    field_ = GF(prime)
    rng = np.random.default_rng(seed)
    codes = sample_luof_codes(field_, k, rng, PARAMS, "di14")
    method = method or ("direct" if k <= DIRECT_MAX_K else "gram")
    dim_v = direct_dimension(field_, codes) if method == "direct" else gram_dimension(field_, codes, rng)
    return SubspaceRow(
        k=k,
        prime=prime,
        seed=seed,
        N=codes.inner_dimension,
        k2=k * k,
        dim_V=dim_v,
        codim=k * k - dim_v,
        fill_percent=100 * dim_v / (k * k),
        seconds=time.perf_counter() - start,
        method=method,
        params=codes.inner_dimension * codes.block_size,
        block_bound=block_bound(field_, codes),
    )


def span_point(k: int, ell: int, prime: int, seed: int = 0) -> SpanRow:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    field_ = GF(prime)
    product = make_candidate("luof", field_, k=k, ell=ell, view="composite").sample_product(rng)
    samples = sample_count(k)
    stacked = np.empty((samples, k * k), dtype=np.int64)
    for index in range(samples):
        stacked[index] = product.sample(rng).matrix.ravel()
    span = linalg.rank(field_, stacked)
    relations, hold, reject = [], None, None
    if span < k * k:
        basis = linalg.nullspace(field_, stacked).T  # r @ vec(M) = 0 for every sampled M
        hold = not field_.matmul(basis, product.sample(rng).matrix.reshape(-1, 1)).any()
        reject = bool(field_.matmul(basis, product.sample_uniform(rng).reshape(-1, 1)).any())
        relations = basis.tolist()
    return SpanRow(
        k=k,
        ell=ell,
        prime=prime,
        seed=seed,
        samples=samples,
        k2=k * k,
        span=span,
        codim=k * k - span,
        distinguisher=span < k * k,
        relations_hold_on_fresh=hold,
        relations_reject_uniform=reject,
        seconds=time.perf_counter() - start,
        relations=relations,
    )


# --- fast path ----------------------------------------------------------------------------------
def _layer_codes(product: LayeredProduct) -> tuple[np.ndarray, np.ndarray, int]:
    """Dense ``Ext``, ``Enc`` and the block size of a shared-codes composite product."""
    family = product.secrets[0]
    if any(secret.left is not family.left for secret in product.secrets):
        raise ValueError("the fast path needs shared codes")
    return family.left.to_dense(), family.right.to_dense(), family.inner.block


def fast_sampler(product: LayeredProduct):
    """A function ``rng -> M`` with the distribution of ``product.sample(rng).matrix``."""
    field_ = product.field
    ext, enc, block = _layer_codes(product)
    k, n = ext.shape
    blocks = n // block
    ext_blocks = ext.reshape(k, blocks, block).transpose(1, 0, 2)  # (blocks, k, block)
    publics = [public.to_dense() for public in product.publics]

    def layer(rng: np.random.Generator) -> np.ndarray:
        h = field_.random(rng, (blocks, block, block))
        left = linalg.batched_matmul(field_, ext_blocks, h)  # Ext_b H_b
        return field_.matmul(left.transpose(1, 0, 2).reshape(k, n), enc)

    def sample(rng: np.random.Generator) -> np.ndarray:
        current = publics[-1]
        for index in reversed(range(product.ell)):
            current = field_.matmul(publics[index], field_.matmul(layer(rng), current))
        return current

    return sample


def fast_span_point(
    k: int, ell: int, prime: int, seed: int = 0, samples: int | None = None, **options
) -> SpanRow:
    """The span of :func:`span_point` through :func:`fast_sampler`; ``options`` go to ``luof``."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    field_ = GF(prime)
    product = make_candidate("luof", field_, k=k, ell=ell, view="composite", **options).sample_product(rng)
    sample = fast_sampler(product)
    samples = samples or fast_sample_count(k)
    matrix = rows_to_flint((sample(rng).ravel() for _ in range(samples)), (samples, k * k), prime)
    span = int(matrix.rank())
    relations, hold, reject = [], None, None
    if span < k * k:
        kernel, nullity = matrix.nullspace()
        basis = np.array([[int(kernel[i, j]) for i in range(k * k)] for j in range(nullity)], dtype=np.int64)
        hold = not field_.matmul(basis, product.sample(rng).matrix.reshape(-1, 1)).any()
        reject = bool(field_.matmul(basis, product.sample_uniform(rng).reshape(-1, 1)).any())
        relations = basis.tolist()
    return SpanRow(
        k=k,
        ell=ell,
        prime=prime,
        seed=seed,
        samples=samples,
        k2=k * k,
        span=span,
        codim=k * k - span,
        distinguisher=span < k * k,
        relations_hold_on_fresh=hold,
        relations_reject_uniform=reject,
        seconds=time.perf_counter() - start,
        relations=relations,
        options=options,
        experiment="span_fast",
    )


# --- block pattern ------------------------------------------------------------------------------
def block_pattern(product: LayeredProduct) -> np.ndarray:
    """Boolean ``P``: block ``(b, c)`` of ``H_1 T_2 H_2 ... H_l`` can be nonzero. Needs ``ell >= 2``."""
    field_ = product.field
    ext, enc, block = _layer_codes(product)
    blocks = enc.shape[0] // block
    pattern = None
    for public in product.publics[1:-1]:
        t = field_.matmul(field_.matmul(enc, public.to_dense()), ext)
        step = t.reshape(blocks, block, blocks, block).any(axis=(1, 3))
        pattern = step if pattern is None else (pattern.astype(np.float32) @ step.astype(np.float32)) > 0
    return pattern


def pattern_span(product: LayeredProduct) -> int:
    """``dim span{M}`` from the formula in the module docstring. Small ``k`` only."""
    field_ = product.field
    ext, enc, block = _layer_codes(product)
    offsets = np.arange(block)
    rows = [
        field_.mul(ext[:, block * b + i][:, None], enc[block * c + j][None, :]).ravel()
        for b, c in zip(*np.nonzero(block_pattern(product)), strict=True)
        for i in offsets
        for j in offsets
    ]
    return linalg.rank(field_, np.stack(rows))


def certify_full(product: LayeredProduct, pattern: np.ndarray) -> bool:
    """Proof of ``span = k^2``, see the module docstring. ``False`` proves nothing either way."""
    field_ = product.field
    ext, enc, block = _layer_codes(product)
    k = ext.shape[0]
    blocks = pattern.shape[0]
    enc_blocks = enc.reshape(blocks, block, k)
    passing: dict[bytes, bool] = {}
    good = []
    for b in range(blocks):
        key = pattern[b].tobytes()
        if key not in passing:
            passing[key] = linalg.rank(field_, enc_blocks[pattern[b]].reshape(-1, k)) == k
        if passing[key]:
            good.append(b)
    columns = ext.reshape(k, blocks, block)[:, good].reshape(k, -1)
    return linalg.rank(field_, columns) == k


def pattern_point(k: int, ell: int, prime: int, seed: int = 0, **options) -> PatternRow:
    start = time.perf_counter()
    field_ = GF(prime)
    rng = np.random.default_rng(seed)
    product = make_candidate("luof", field_, k=k, ell=ell, view="composite", **options).sample_product(rng)
    ext, enc, block = _layer_codes(product)
    blocks = enc.shape[0] // block
    live = ext.reshape(k, blocks, block).any(axis=(0, 2)) & enc.reshape(blocks, block, k).any(axis=(1, 2))
    pattern = block_pattern(product)
    return PatternRow(
        k=k,
        ell=ell,
        prime=prime,
        seed=seed,
        N=enc.shape[0],
        blocks=blocks,
        live_blocks=int(live.sum()),
        dead_pairs=int((~pattern[np.ix_(live, live)]).sum()),
        certified_full=certify_full(product, pattern),
        seconds=time.perf_counter() - start,
        options=options,
    )
