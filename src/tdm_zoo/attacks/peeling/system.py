"""The peeling polynomial system ``p_i(W) = Q_i(x(W))``.

For ``M = A_1 S_1 A_2 S_2 ... A_l S_l A_{l+1}`` the peeled layers are ``j = l, ..., 3``.
With candidate layers ``S_j(W_j) := A_{j+1}^{-1} g_j(W_j)`` (``g_j`` the family's
homogeneous inverse map, of degree ``e_j``) the residual

    x(W) = vec(M . S_l(W_l) . S_{l-1}(W_{l-1}) ... S_3(W_3))

is multi-homogeneous of multidegree ``(e_l, ..., e_3)``, and for the true witness
``W*`` (``g_j(W_j*) = S_j^{-1}``) it equals the core ``A_1 S_1 A_2 S_2 A_3``, so every
``p_i = Q_i o x`` vanishes at ``W*``.

Blocks are ordered ``b = 0, ..., l-3`` with ``b = 0`` <-> ``j = l`` (the layer adjacent
to ``M``). Coefficients of the ``p_i`` are obtained by evaluation on a tensor grid
(one unisolvent point set per block) followed by interpolation with the inverse
per-block Vandermonde matrices, so no symbolic expansion is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...fields import Field
from ...ops import linalg
from ...ops.matrix_maps import MatrixMap
from ...ops.multihomogeneous import ProductBasis
from ...tdm.product import PublicView, TDMSample
from .macaulay import DEFAULT_MAX_ENTRIES, MacaulayShape, macaulay_matrix, macaulay_shape
from .relations import BaseRelations, base_relations

Array = np.ndarray

_MAX_GRID_TRIES = 100


@dataclass
class PeelingBlock:
    layer: int  # j
    g: MatrixMap  # homogeneous inverse map g_j
    public_inverse: Array  # A_{j+1}^{-1}
    grid: Array = field(repr=False)  # interpolation points (N_b, nvars)
    vandermonde_inverse: Array = field(repr=False)  # (N_b, N_b)
    grid_layers: Array = field(repr=False)  # S_j(W) at the grid points: (N_b, n, n)

    @property
    def nvars(self) -> int:
        return self.g.nparams

    @property
    def degree(self) -> int:
        return self.g.degree

    def layer_matrix(self, field_: Field, params) -> Array:
        return field_.matmul(self.public_inverse, self.g.evaluate(params))


class PeelingSystem:
    """Everything that depends only on the public view: ``Q``, the layer maps and the grids."""

    def __init__(self, view: PublicView, degree: int, rng: np.random.Generator) -> None:
        if view.ell < 3:
            raise ValueError("peeling needs at least three secret layers")
        if any(p is None for p in view.publics):
            raise ValueError("peeling needs every public layer to be revealed")
        n = view.shape[0]
        publics = [view.public_matrix(i) for i in range(view.ell + 1)]
        if any(a.shape != (n, n) for a in publics) or any(s.shape != (n, n) for s in view.secret_families):
            raise ValueError("peeling is implemented for square n x n layers")
        self.field = view.field
        self.n = n
        self.ell = view.ell
        self.degree = degree
        self.relations: BaseRelations = base_relations(
            self.field,
            publics[:3],
            view.secret_families[0].param_map,
            view.secret_families[1].param_map,
            degree,
            rng,
        )
        self.blocks: list[PeelingBlock] = []
        for j in range(self.ell, 2, -1):
            g = view.secret_families[j - 1].homogeneous_inverse_map
            if g is None:
                raise ValueError(f"layer S_{j} has no homogeneous inverse map")
            self.blocks.append(self._block(j, g, linalg.inv(self.field, publics[j]), rng))
        self.base = ProductBasis(
            tuple(b.nvars for b in self.blocks), tuple(degree * b.degree for b in self.blocks)
        )

    def _block(self, layer: int, g: MatrixMap, public_inverse: Array, rng) -> PeelingBlock:
        basis = ProductBasis((g.nparams,), (self.degree * g.degree,)).blocks[0]
        for _ in range(_MAX_GRID_TRIES):
            grid = self.field.random(rng, (basis.size, g.nparams))
            vandermonde = basis.evaluate(self.field, grid)
            if linalg.is_invertible(self.field, vandermonde):
                break
        else:
            raise RuntimeError("could not find a unisolvent interpolation grid")
        layers = np.stack([self.field.matmul(public_inverse, g.evaluate(w)) for w in grid])
        return PeelingBlock(layer, g, public_inverse, grid, linalg.inv(self.field, vandermonde), layers)

    # -- dimensions -----------------------------------------------------------------------
    @property
    def relation_count(self) -> int:
        return self.relations.count

    def shape(self, multipliers) -> MacaulayShape:
        multipliers = (multipliers,) * len(self.blocks) if np.isscalar(multipliers) else tuple(multipliers)
        return macaulay_shape(self.relation_count, self.base, multipliers)

    def describe(self) -> dict:
        return {
            "n": self.n,
            "ell": self.ell,
            "d": self.degree,
            "K": self.relation_count,
            "blocks": [{"layer": b.layer, "nvars": b.nvars, "degree": b.degree} for b in self.blocks],
            "base_columns": self.base.size,
        }

    # -- composition --------------------------------------------------------------------
    def residual(self, matrix: Array, block_params: list[Array]) -> Array:
        """``M . S_l(W_l) ... S_3(W_3)`` at one point."""
        current = np.asarray(matrix, dtype=np.int64)
        for block, params in zip(self.blocks, block_params, strict=True):
            current = self.field.matmul(current, block.layer_matrix(self.field, params))
        return current

    def compose(self, matrix: Array) -> Array:
        """Coefficients of ``p_1, ..., p_K`` in the basis ``self.base``: shape ``(K, |base|)``."""
        f, n = self.field, self.n
        residuals = np.asarray(matrix, dtype=np.int64)[None]
        for block in self.blocks:
            residuals = linalg.batched_matmul(f, residuals[:, None], block.grid_layers[None])
            residuals = residuals.reshape(-1, n, n)
        values = self.relations.evaluate(f, residuals)  # (|grid|, K), grid in mixed-radix order
        tensor = values.reshape(*self.base.shape, self.relation_count)
        for axis, block in enumerate(self.blocks):
            moved = np.moveaxis(tensor, axis, 0)
            flat = f.matmul(block.vandermonde_inverse, moved.reshape(moved.shape[0], -1))
            tensor = np.moveaxis(flat.reshape(moved.shape), 0, axis)
        return np.moveaxis(tensor, -1, 0).reshape(self.relation_count, -1)

    def macaulay(self, matrix: Array, multipliers=0, max_entries: int = DEFAULT_MAX_ENTRIES) -> Array:
        """``T(M)`` at multiplier multidegree ``multipliers`` (int = same for every block)."""
        return macaulay_matrix(self.field, self.compose(matrix), self.shape(multipliers), max_entries)

    # -- witnesses (need the trapdoor; for tests and analysis only) ---------------------
    def witness(self, sample: TDMSample) -> list[Array]:
        """``W_j*`` for ``j = l, ..., 3`` (block order)."""
        return [
            sample.secrets[block.layer - 1].family.homogeneous_inverse_params(
                sample.secrets[block.layer - 1].params
            )
            for block in self.blocks
        ]

    def witness_vector(self, witness: list[Array], multipliers=0) -> Array:
        """Column monomials evaluated at the witness: ``T(M) v* = 0`` for structured ``M``."""
        return self.shape(multipliers).columns.evaluate(self.field, witness)
