"""Does secret structure give fewer witness unknowns?

Test A (witness cloud). Sample many secrets of one family, compute each true witness block
``W*`` (the parameters of ``g`` with ``g(W*) = S^{-1}``), and find every polynomial relation of
degree <= ``max_degree`` vanishing on all of them (kernel of the monomial evaluation matrix,
with more samples than monomials). Relations implied by lower-degree ones are not counted as
new. The dimension of the witness set is bounded by ``v - rank(Jacobian of the relations)`` at
a sample point (exact formal derivatives).

Test B (rigidity). For one structured ``M``, the Jacobian of ``{p_i}`` at ``W*`` over all
witness unknowns; its kernel is the space of floating directions. Each block always floats by
one direction (the ``p_i`` are homogeneous in each block, so ``W_b -> lambda W_b`` stays a
root), so ``excess = floating - (#blocks)`` is what would indicate redundant unknowns.
Derivatives are exact: ``p(W* + t e_k)`` is a polynomial in ``t`` of known degree, and its
linear coefficient is recovered by interpolation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..attacks.peeling import PeelingSystem
from ..fields import GF
from ..ops import linalg
from ..ops.poly import monomials_up_to
from ..tdm import make_candidate
from ..tdm.families import SECRET_FAMILIES


@dataclass
class WitnessRow:
    family: str
    n: int
    prime: int
    ell: int
    base_family: str
    vars_per_block: int  # v: unknowns per witness block as used by the attack
    family_params: int  # free parameters of the secret family
    samples: int
    relations_by_degree: dict  # degree -> number of NEW relations at that degree
    linear_relations: int
    dimension_bound: int  # v - rank(Jacobian of all relations found) at a sample witness
    effective_unknowns: int  # min(dimension_bound, family params, rank dg)
    blocks: int
    total_unknowns: int
    jacobian_rank: int
    floating: int
    floating_per_block: list
    excess_floating: int  # floating - blocks (per-block scaling removed)
    parametrisation_redundancy: int  # sum over blocks of (v - rank dg_b at W*): directions g ignores
    genuine_floating: int  # excess_floating - parametrisation_redundancy
    rank_dg_per_block: list
    bar_l5: float
    bar_l6: float
    clears_bar: bool
    relations_written: list = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --- polynomial helpers on coefficient vectors over ``monomials_up_to(v, d)`` -------------------
def _evaluation_matrix(field_, points: np.ndarray, monomials) -> np.ndarray:
    out = field_.ones((points.shape[0], len(monomials)))
    for j, monomial in enumerate(monomials):
        for variable in monomial:
            out[:, j] = field_.mul(out[:, j], points[:, variable])
    return out


def _multiply_by_variable(vector, monomials, index_of, variable, size) -> np.ndarray:
    out = np.zeros(size, dtype=np.int64)
    for j in np.flatnonzero(vector):
        out[index_of[tuple(sorted(monomials[j] + (variable,)))]] = vector[j]
    return out


def _gradient(field_, vector, monomials, point, nvars) -> np.ndarray:
    grad = field_.zeros(nvars)
    for j in np.flatnonzero(vector):
        monomial = monomials[j]
        for variable in set(monomial):
            rest = list(monomial)
            rest.remove(variable)
            value = field_.mul(vector[j], field_.from_int(monomial.count(variable)))
            for other in rest:
                value = field_.mul(value, point[other])
            grad[variable] = field_.add(grad[variable], value)
    return grad


def _format(vector, monomials, names, p) -> str:
    terms = []
    for j in np.flatnonzero(vector):
        coefficient = int(vector[j])
        coefficient = coefficient - p if coefficient > p // 2 else coefficient
        name = "*".join(names[v] for v in monomials[j]) or "1"
        terms.append(f"{coefficient}*{name}" if name != "1" else str(coefficient))
    return " + ".join(terms) + " = 0"


def _variable_names(family: str, n: int, v: int) -> list[str]:
    if family in ("toeplitz", "hankel") and v == 2 * n + 1:
        return [f"x{i}" for i in range(n)] + [f"z{i}" for i in range(n)] + ["t"]
    return [f"w{i}" for i in range(v)]


# --- Test A -------------------------------------------------------------------------------------
def witness_relations(field_, family, rng, max_degree: int = 3, margin: int = 30, write: int = 8):
    v = family.homogeneous_inverse_map.nparams
    monomials = monomials_up_to(v, max_degree)
    samples = len(monomials) + margin
    points = np.stack([family.homogeneous_inverse_params(family.sample_params(rng)) for _ in range(samples)])
    by_degree, kernels = {}, {}
    previous = None
    for degree in range(1, max_degree + 1):
        basis = monomials_up_to(v, degree)
        index_of = {m: i for i, m in enumerate(basis)}
        kernel = linalg.nullspace(field_, _evaluation_matrix(field_, points, basis))  # (|basis|, k)
        generated = 0
        if previous is not None and previous[1].shape[1]:
            prev_basis, prev_kernel = previous
            vectors = []
            for col in prev_kernel.T:
                lifted = np.zeros(len(basis), dtype=np.int64)
                for j in np.flatnonzero(col):
                    lifted[index_of[prev_basis[j]]] = col[j]
                vectors.append(lifted)
                vectors += [
                    _multiply_by_variable(lifted, basis, index_of, var, len(basis)) for var in range(v)
                ]
            generated = linalg.rank(field_, np.stack(vectors)) if vectors else 0
        by_degree[degree] = int(kernel.shape[1] - generated)
        kernels[degree] = (basis, kernel)
        previous = (basis, kernel)
    basis, kernel = kernels[max_degree]
    point = points[0]
    if kernel.shape[1]:
        jac = np.stack([_gradient(field_, col, basis, point, v) for col in kernel.T])
        jac_rank = linalg.rank(field_, jac)
    else:
        jac_rank = 0
    names = _variable_names(family.name, family.shape[0], v)
    written = []
    for degree in range(1, max_degree + 1):
        if by_degree[degree] and len(written) < write:
            deg_basis, deg_kernel = kernels[degree]
            reduced, _ = linalg.rref(field_, deg_kernel.T)
            for row in reduced[: deg_kernel.shape[1]]:
                if len(written) >= write:
                    break
                written.append(f"[deg<= {degree}] " + _format(row, deg_basis, names, field_.p))
    return {
        "vars": v,
        "samples": samples,
        "by_degree": by_degree,
        "dimension_bound": v - jac_rank,
        "written": written,
    }


# --- Test B -------------------------------------------------------------------------------------
def _linear_coefficient(field_, values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Coefficient of ``t`` of the polynomial interpolating ``values[j]`` (vectors) at ``nodes``."""
    vandermonde = np.stack([field_.power(nodes, e) for e in range(len(nodes))], axis=1)
    coefficients = field_.matmul(linalg.inv(field_, vandermonde), values)
    return coefficients[1]


def map_differential(field_, g, point: np.ndarray) -> np.ndarray:
    """Exact differential of the matrix map ``g`` at ``point``: ``(rows*cols, nparams)``."""
    nodes = np.arange(g.degree + 1, dtype=np.int64)
    columns = []
    for k in range(g.nparams):
        values = []
        for t in nodes:
            shifted = point.copy()
            shifted[k] = field_.add(shifted[k], t)
            values.append(g.evaluate(shifted).ravel())
        columns.append(_linear_coefficient(field_, np.stack(values), nodes))
    return np.stack(columns, axis=1)


def system_jacobian(system: PeelingSystem, matrix: np.ndarray, witness: list[np.ndarray]) -> np.ndarray:
    field_ = system.field
    columns = []
    for b, block in enumerate(system.blocks):
        degree = system.degree * block.degree
        nodes = np.arange(degree + 1, dtype=np.int64)
        for k in range(block.nvars):
            values = []
            for t in nodes:
                point = [w.copy() for w in witness]
                point[b][k] = field_.add(point[b][k], t)
                values.append(system.relations.evaluate(field_, system.residual(matrix, point)[None])[0])
            columns.append(_linear_coefficient(field_, np.stack(values), nodes))
    return np.stack(columns, axis=1)  # (K, total unknowns)


# --- one point ----------------------------------------------------------------------------------
def witness_point(
    family: str,
    n: int,
    prime: int,
    ell: int = 4,
    d: int = 3,
    seed: int = 0,
    family_kwargs: dict | None = None,
) -> WitnessRow:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    field_ = GF(prime)
    kwargs = dict(family_kwargs or {})
    fam = SECRET_FAMILIES.create(family, field_, n, **kwargs)
    test_a = witness_relations(field_, fam, rng)

    # Base layers: same family when it yields base relations, otherwise diagonal.
    base = family if family in ("diagonal", "circulant", "weighted_accumulator") else "diagonal"
    spec = (family, kwargs)
    candidate = make_candidate("layered_tdm", field_, k=n, ell=ell, families=[base] * 2 + [spec] * (ell - 2))
    product = candidate.sample_product(rng)
    system = PeelingSystem(product.public_view(), d, rng)
    sample = product.sample(rng)
    witness = system.witness(sample)
    jacobian = system_jacobian(system, sample.matrix, witness)
    rank = linalg.rank(field_, jacobian)
    kernel = linalg.nullspace(field_, jacobian)
    offsets = np.cumsum([0] + [b.nvars for b in system.blocks])
    per_block = [
        int(linalg.rank(field_, kernel[offsets[b] : offsets[b + 1]])) if kernel.shape[1] else 0
        for b in range(len(system.blocks))
    ]
    total = int(offsets[-1])
    rank_dg = [
        int(linalg.rank(field_, map_differential(field_, block.g, w)))
        for block, w in zip(system.blocks, witness, strict=True)
    ]
    redundancy = sum(block.nvars - r for block, r in zip(system.blocks, rank_dg, strict=True))
    floating = total - int(rank)
    # True number of unknowns a block carries: bounded by the relations found (Test A), by the
    # family's own parameter count, and by the rank of g's differential.
    effective = min(test_a["dimension_bound"], fam.nparams, max(rank_dg))
    return WitnessRow(
        family=family if not kwargs else f"{family}{kwargs}",
        n=n,
        prime=prime,
        ell=ell,
        base_family=base,
        vars_per_block=test_a["vars"],
        family_params=fam.nparams,
        samples=test_a["samples"],
        relations_by_degree=test_a["by_degree"],
        linear_relations=test_a["by_degree"][1],
        dimension_bound=effective,
        effective_unknowns=effective,
        blocks=len(system.blocks),
        total_unknowns=total,
        jacobian_rank=int(rank),
        floating=floating,
        floating_per_block=per_block,
        excess_floating=floating - len(system.blocks),
        parametrisation_redundancy=redundancy,
        genuine_floating=floating - len(system.blocks) - redundancy,
        rank_dg_per_block=rank_dg,
        bar_l5=n ** (2 / 3),
        bar_l6=n ** (1 / 2),
        clears_bar=effective <= n ** (2 / 3),
        relations_written=test_a["written"],
        seconds=time.perf_counter() - start,
    )
