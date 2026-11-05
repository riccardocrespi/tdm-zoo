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

``dim V`` depends on the sampled codes (the graphs are random), so it varies with the seed.
The codes come first out of the generator in both :func:`subspace_point` and :func:`span_point`,
so the two share codes at equal ``seed``: the ``ell = 1`` span then equals ``dim V`` exactly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..fields import GF
from ..ops import linalg
from ..tdm import make_candidate
from ..tdm.di14 import DI14Params, di14_dimension, sample_luof_codes
from ..tdm.publics import luof_span_dimension

PARAMS = DI14Params(8, 6, 4)


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
    experiment: str = "span"

    def to_dict(self) -> dict:
        return asdict(self)


def sample_count(k: int) -> int:
    """More samples than ``k^2``, so that the span is capped by structure and not by the count."""
    return k * k + k * k // 4 + 20


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


def subspace_point(k: int, prime: int, seed: int = 0) -> SubspaceRow:
    start = time.perf_counter()
    field_ = GF(prime)
    codes = sample_luof_codes(field_, k, np.random.default_rng(seed), PARAMS, "di14")
    dim_v = luof_span_dimension(field_, codes)
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
