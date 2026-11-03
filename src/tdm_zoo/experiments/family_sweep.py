"""Secret-family sweep: does the secret's shape create a *sparse* structured-only relation?

For one (family, n, ell, a, prime) point:

* build ``T(M)`` for a structured ``M`` (every secret layer from ``family``) and ``T(U)`` for
  uniform ``U`` (dense, exact elimination);
* ``gamma = rank T(U) - rank T(M)``;
* ``s_witness`` = number of nonzeros of the witness vector ``v*``;
* ``s`` = size of a column set ``S`` on which the restricted matrices still distinguish,
  ``rank T(M)|_S < rank T(U')|_S`` for independent references ``U'`` (spec Sec. 4.2: this holds
  iff ``ker T(M)`` has a structured-only vector supported in ``S``). ``S`` is found by greedy
  column dropping down to a local minimum, so ``s`` upper-bounds the true minimum.

``ker T(M)`` is not the witness alone: it also contains a generic part (present for random
``U`` too), so ``s`` can be smaller than ``s_witness`` even when ``v*`` has full support.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from math import prod

import numpy as np

from ..attacks.peeling import PeelingSystem
from ..fields import GF
from ..ops import linalg
from ..ops.multihomogeneous import num_monomials
from ..tdm import make_candidate


@dataclass
class SweepRow:
    family: str
    n: int
    ell: int
    multipliers: int
    prime: int
    status: str  # "ok" or a reason the point could not be measured
    K: int | None = None
    C: int | None = None
    R: int | None = None
    rank_M: int | None = None
    rank_U: int | None = None
    delta_M: int | None = None
    delta_U: int | None = None
    gamma: int | None = None
    shared_kernel_dim: int | None = None  # dim(ker T(M) ∩ ker T(U))
    s_witness: int | None = None
    s: int | None = None
    s_over_C: float | None = None
    s_minus_rank_U: int | None = None  # 0: only the generic slack delta(U) was droppable
    candidates_tested: int | None = None  # gap tests used by the column-dropping search
    minimal: bool | None = None  # no single column of the final support can be removed
    lightest_kernel_vector: int | None = None  # lightest vector of ker T(M) found (any kind)
    witness_in_kernel: bool | None = None  # T(M) v* = 0
    witness_not_in_uniform_kernel: bool | None = None  # T(U) v* != 0
    sparse_vector: dict | None = None  # support and labels, recorded when s < C
    seconds: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _intersection_dim(field_, a: np.ndarray, b: np.ndarray) -> int:
    return a.shape[1] + b.shape[1] - linalg.rank(field_, np.hstack([a, b]))


def _gap_on_support(field_, t_m: np.ndarray, references: list[np.ndarray], support: np.ndarray) -> bool:
    """Direct check on the restricted matrices (used once, to confirm the final support)."""
    rank_m = linalg.rank(field_, t_m[:, support])
    return all(linalg.rank(field_, t_u[:, support]) > rank_m for t_u in references)


def _deficiency_after_drop(field_, kernel: np.ndarray, dropped: np.ndarray) -> int:
    """``dim ker(A|_S)`` for ``S`` = complement of ``dropped``, from a kernel basis of ``A``.

    ``ker(A|_S)`` = kernel vectors vanishing on ``dropped`` = ``dim ker A - rank kernel[dropped]``.
    """
    if dropped.size == 0:
        return kernel.shape[1]
    return kernel.shape[1] - linalg.rank(field_, kernel[dropped])


def _gap_after_drop(field_, kernel_m: np.ndarray, kernels_ref: list[np.ndarray], dropped: np.ndarray) -> bool:
    deficiency_m = _deficiency_after_drop(field_, kernel_m, dropped)
    return all(deficiency_m > _deficiency_after_drop(field_, k, dropped) for k in kernels_ref)


def _column_labels(system: PeelingSystem, columns, support: np.ndarray) -> list[list[list[int]]]:
    """Per-block monomials (variable index tuples) of each support column."""
    digits = np.unravel_index(support, columns.shape)
    return [
        [list(columns.blocks[b].monomials[int(digits[b][i])]) for b in range(len(columns.blocks))]
        for i in range(support.size)
    ]


def minimal_gap_support(
    field_,
    kernel_m: np.ndarray,
    kernels_ref: list[np.ndarray],
    rng: np.random.Generator,
    max_tests: int = 20000,
) -> tuple[np.ndarray, int, bool]:
    """Shrink the column set while the restricted matrices still distinguish.

    The gap on ``S`` is ``dim ker T(M)|_S > dim ker T(U')|_S`` for every reference, computed from
    kernel bases (cheap ranks of ``|dropped| x dim ker`` matrices). Random chunks are dropped
    with halving sizes, then every remaining column is tried singly until none can go.
    Returns ``(S, tests, minimal)``; ``|S|`` upper-bounds the true minimum.
    """
    columns = kernel_m.shape[0]
    keep = np.ones(columns, dtype=bool)
    tests = 0

    def gap_without(extra: np.ndarray) -> bool:
        dropped = np.concatenate([np.flatnonzero(~keep), extra])
        return _gap_after_drop(field_, kernel_m, kernels_ref, dropped)

    chunk, failures = max(1, columns // 2), 0
    while chunk > 1 and tests < max_tests:
        alive = np.flatnonzero(keep)
        extra = rng.choice(alive, size=min(chunk, alive.size - 1), replace=False)
        tests += 1
        if gap_without(extra):
            keep[extra], failures = False, 0
        else:
            failures += 1
            if failures >= 3:
                chunk, failures = chunk // 2, 0
    changed = True
    while changed and tests < max_tests:
        changed = False
        for column in rng.permutation(np.flatnonzero(keep)):
            if tests >= max_tests:
                break
            tests += 1
            if gap_without(np.array([column])):
                keep[column], changed = False, True
    return np.flatnonzero(keep), tests, tests < max_tests


def lightest_kernel_vector(
    field_, kernel: np.ndarray, rng: np.random.Generator, iterations: int
) -> np.ndarray:
    """Information-set sampling: lightest RREF row of the kernel basis over random column orders."""
    columns, dim = kernel.shape
    best = None
    for _ in range(iterations):
        perm = rng.permutation(columns)
        reduced, _ = linalg.rref(field_, kernel.T[:, perm])
        for row in reduced[:dim]:
            if best is None or np.count_nonzero(row) < np.count_nonzero(best):
                best = np.empty(columns, dtype=np.int64)
                best[perm] = row
    return best


def sweep_point(
    family: str,
    n: int,
    prime: int,
    ell: int = 4,
    multipliers: int = 0,
    d: int = 3,
    seed: int = 0,
    references: int = 2,
    iterations: int = 50,
    max_tests: int = 20000,
    max_dense_cells: int = 5 * 10**7,
) -> SweepRow:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    field_ = GF(prime)
    row = SweepRow(family, n, ell, multipliers, prime, status="ok")
    candidate = make_candidate("layered_tdm", field_, k=n, ell=ell, families=[family] * ell)
    product = candidate.sample_product(rng)
    # Size check before any preprocessing (grids/Vandermonde inverses scale with C).
    peeled = [product.secrets[j - 1].homogeneous_inverse_map for j in range(ell, 2, -1)]
    columns_a_priori = prod(num_monomials(g.nparams, d * g.degree + multipliers) for g in peeled)
    if columns_a_priori * num_monomials(n * n, d) > max_dense_cells:
        row.C = columns_a_priori
        row.status = (
            f"no regime: C = {columns_a_priori:,} columns vs at most {num_monomials(n * n, d):,} rows "
            f"(peeled g: {peeled[0].nparams} vars, degree {peeled[0].degree})"
        )
        row.seconds = time.perf_counter() - start
        return row
    system = PeelingSystem(product.public_view(), d, rng)
    shape = system.shape(multipliers)
    row.K, row.C, row.R = system.relation_count, shape.cols, shape.rows
    if shape.rows < shape.cols:
        row.notes.append("fewer rows than columns: T(U) cannot reach full column rank")
    if shape.entries > max_dense_cells:
        row.status = f"no regime: {shape.rows} x {shape.cols} is beyond dense elimination"
        row.seconds = time.perf_counter() - start
        return row
    if row.K == 0:
        row.status = "no base relations (K = 0)"
        row.seconds = time.perf_counter() - start
        return row

    sample = product.sample(rng)
    t_m = system.macaulay(sample.matrix, multipliers)
    uniforms = [system.macaulay(product.sample_uniform(rng), multipliers) for _ in range(1 + references)]
    t_u, reference_mats = uniforms[0], uniforms[1:]
    row.rank_M, row.rank_U = linalg.rank(field_, t_m), linalg.rank(field_, t_u)
    row.delta_M, row.delta_U = row.C - row.rank_M, row.C - row.rank_U
    row.gamma = row.rank_U - row.rank_M

    witness = system.witness_vector(system.witness(sample), multipliers)
    row.witness_in_kernel = not np.any(field_.matmul(t_m, witness[:, None]))
    row.witness_not_in_uniform_kernel = bool(np.any(field_.matmul(t_u, witness[:, None])))
    row.s_witness = int(np.count_nonzero(witness))

    kernel_m, kernel_u = linalg.nullspace(field_, t_m), linalg.nullspace(field_, t_u)
    row.shared_kernel_dim = _intersection_dim(field_, kernel_m, kernel_u)

    if row.gamma < 1:
        row.status = "no gap at this point"
        row.seconds = time.perf_counter() - start
        return row

    lightest = lightest_kernel_vector(field_, kernel_m, rng, iterations)
    row.lightest_kernel_vector = int(np.count_nonzero(lightest))
    if not _gap_on_support(field_, t_m, reference_mats, np.flatnonzero(witness)):
        row.notes.append("gap test failed on supp(v*)")
    kernels_ref = [linalg.nullspace(field_, t) for t in reference_mats]
    support, tests, minimal = minimal_gap_support(field_, kernel_m, kernels_ref, rng, max_tests)
    if not _gap_on_support(field_, t_m, reference_mats, support):
        row.notes.append("final support failed the direct rank check")
    row.candidates_tested, row.minimal = tests, minimal
    row.s = int(support.size)
    row.s_over_C = row.s / row.C
    row.s_minus_rank_U = row.s - row.rank_U
    if row.s < row.C:
        restricted_kernel = linalg.nullspace(field_, t_m[:, support])
        vector = np.zeros(row.C, dtype=np.int64)
        vector[support] = lightest_kernel_vector(field_, restricted_kernel, rng, 20)
        nz = np.flatnonzero(vector)
        missing = np.setdiff1d(np.arange(row.C), support)
        row.sparse_vector = {
            "support": support.tolist(),
            "dropped_columns": missing.tolist(),
            "dropped_monomials": _column_labels(system, shape.columns, missing),
            "kernel_dim_on_support": int(restricted_kernel.shape[1]),
            "vector_support": nz.tolist(),
            "vector_values": vector[nz].tolist(),
        }
    row.seconds = time.perf_counter() - start
    return row
