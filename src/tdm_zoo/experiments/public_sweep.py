"""Public-layer sweep: does the peeling rank see what the public matrices ``A_i`` are?

For one (family, n, ell, publics) point, over several independent public samples, measure the
same quantities as :mod:`.family_sweep` (``K``, ``C``, ``rank T(M)``, ``rank T(U)``, ``gamma``)
with ``A_i`` uniform invertible, Druk-Ishai ``A_i = G H_i P E`` (``publics="luof"``), or
permutations (control: publics that are visibly not generic). ``zeros > 0`` forces that many
random entries of every ``A_i`` to zero (control: how little non-genericity the rank reacts to).

``span`` is ``dim span{G H P E}``. For uniform ``H``, ``A`` is linear in ``H``, so at
``span = n^2`` it is exactly uniform and the ``luof`` row is the uniform row in distribution;
only ``span < n^2`` is a real change of public distribution.

:func:`composite_relations` covers the other reading of the Druk-Ishai layer, with ``H`` secret:
``S_i = G H_i P E`` as the secret family and permutation publics. There the question is whether
any base relation exists at all.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..attacks.peeling import PeelingSystem
from ..attacks.peeling.relations import base_relations
from ..fields import GF
from ..ops import linalg
from ..ops.operators import DenseOperator
from ..tdm import make_candidate
from ..tdm.product import LayeredProduct


@dataclass
class PublicRow:
    family: str
    n: int
    ell: int
    prime: int
    publics: str
    options: dict
    seed: int
    status: str  # "ok" or why the point could not be measured
    zeros: int = 0  # entries forced to zero in every A_i
    span: int | None = None
    zero_entries: int | None = None  # zeros of A_2, the public inside the base relations
    K: int | None = None
    C: int | None = None
    R: int | None = None
    rank_M: int | None = None
    rank_U: int | None = None
    delta_M: int | None = None
    delta_U: int | None = None
    gamma: int | None = None
    witness_in_kernel: bool | None = None  # T(M) v* = 0
    witness_not_in_uniform_kernel: bool | None = None  # T(U) v* != 0
    seconds: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def with_zeros(product: LayeredProduct, zeros: int, rng: np.random.Generator) -> LayeredProduct:
    """The same product with ``zeros`` random entries of every public cleared (kept invertible)."""
    f = product.field
    publics = []
    for layer in product.publics:
        dense = layer.to_dense()
        while True:
            matrix = dense.copy()
            matrix.ravel()[rng.choice(matrix.size, zeros, replace=False)] = 0
            if linalg.is_invertible(f, matrix):
                break
        publics.append(DenseOperator(f, matrix))
    return LayeredProduct(f, publics, product.secrets, metadata=product.metadata)


def composite_relations(n: int, prime: int, seed: int = 0, d: int = 3, **options) -> dict:
    """``K`` for ``M = S_1 Pi_2 S_2 ...`` with secret ``S_i = G H_i P E``: the view of ``luof``."""
    rng = np.random.default_rng(seed)
    field_ = GF(prime)
    candidate = make_candidate("luof", field_, k=n, view="composite", **options)
    view = candidate.sample_product(rng).public_view()
    maps = [family.param_map for family in view.secret_families[:2]]
    publics = [view.public_matrix(i) for i in range(3)]
    relations = base_relations(field_, publics, maps[0], maps[1], d, rng)
    images = field_.random(rng, (2 * n * n, maps[0].nparams))
    image = linalg.rank(field_, np.stack([maps[0].evaluate(w).ravel() for w in images]))
    return {
        "n": n,
        "seed": seed,
        "family_image": image,
        "K": relations.count,
        "monomials": relations.basis.size,
    }


def public_point(
    family: str,
    n: int,
    prime: int,
    publics: str = "random_invertible",
    ell: int = 4,
    d: int = 3,
    multipliers: int = 0,
    seed: int = 0,
    zeros: int = 0,
    **options,
) -> PublicRow:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    field_ = GF(prime)
    row = PublicRow(family, n, ell, prime, publics, options, seed, status="ok", zeros=zeros)
    candidate = make_candidate(
        "layered_tdm", field_, k=n, ell=ell, families=[family] * ell, publics=publics, **options
    )
    try:
        product = candidate.sample_product(rng)
    except ValueError as error:
        row.status = f"no invertible publics: {error}"
        row.seconds = time.perf_counter() - start
        return row
    row.span = product.metadata.get("luof_span")
    if zeros:
        product = with_zeros(product, zeros, rng)
    system = PeelingSystem(product.public_view(), d, rng)
    row.zero_entries = int(np.count_nonzero(product.public_view().public_matrix(1) == 0))
    shape = system.shape(multipliers)
    row.K, row.C, row.R = system.relation_count, shape.cols, shape.rows
    if row.K == 0:
        row.status = "no base relations (K = 0)"
        row.seconds = time.perf_counter() - start
        return row

    sample = product.sample(rng)
    t_m = system.macaulay(sample.matrix, multipliers)
    t_u = system.macaulay(product.sample_uniform(rng), multipliers)
    row.rank_M, row.rank_U = linalg.rank(field_, t_m), linalg.rank(field_, t_u)
    row.delta_M, row.delta_U = row.C - row.rank_M, row.C - row.rank_U
    row.gamma = row.rank_U - row.rank_M
    witness = system.witness_vector(system.witness(sample), multipliers)
    row.witness_in_kernel = not np.any(field_.matmul(t_m, witness[:, None]))
    row.witness_not_in_uniform_kernel = bool(np.any(field_.matmul(t_u, witness[:, None])))
    row.seconds = time.perf_counter() - start
    return row
