"""Multi-homogeneous monomial bases (variables split into blocks).

A *block basis* lists the degree-``e`` monomials in ``nvars`` variables, in the
order of ``itertools.combinations_with_replacement`` (lexicographic on sorted
index tuples). A *product basis* over blocks ``b = 0, ..., B-1`` indexes the
monomials of multidegree ``(e_0, ..., e_{B-1})`` by mixed radix, block 0 most
significant. This single ordering is used for Macaulay columns, coefficient
vectors and witness vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations_with_replacement
from math import comb, prod

import numpy as np

from ..fields import Field

Array = np.ndarray


def num_monomials(nvars: int, degree: int) -> int:
    return comb(nvars + degree - 1, degree)


@dataclass(frozen=True)
class BlockBasis:
    nvars: int
    degree: int

    @cached_property
    def monomials(self) -> list[tuple[int, ...]]:
        return list(combinations_with_replacement(range(self.nvars), self.degree))

    @cached_property
    def index_array(self) -> Array:
        """``(size, degree)`` variable indices of each monomial."""
        return np.array(self.monomials, dtype=np.int64).reshape(len(self.monomials), self.degree)

    @cached_property
    def position(self) -> dict[tuple[int, ...], int]:
        return {m: i for i, m in enumerate(self.monomials)}

    @property
    def size(self) -> int:
        return num_monomials(self.nvars, self.degree)

    def evaluate(self, field: Field, points: Array) -> Array:
        """Monomial values at each point: ``(P, nvars) -> (P, size)``."""
        points = np.atleast_2d(np.asarray(points, dtype=np.int64))
        values = field.ones((points.shape[0], self.size))
        for t in range(self.degree):
            values = field.mul(values, points[:, self.index_array[:, t]])
        return values

    def multiplication_table(self, other: BlockBasis) -> Array:
        """``table[i, j]``: index of ``self[i] * other[j]`` in the product-degree basis."""
        if self.nvars != other.nvars:
            raise ValueError("blocks must have the same variables")
        target = BlockBasis(self.nvars, self.degree + other.degree).position
        table = np.empty((self.size, other.size), dtype=np.int64)
        for i, a in enumerate(self.monomials):
            for j, b in enumerate(other.monomials):
                table[i, j] = target[tuple(sorted(a + b))]
        return table


@dataclass(frozen=True)
class ProductBasis:
    """Monomials of multidegree ``degrees`` in blocks of ``nvars`` variables."""

    nvars: tuple[int, ...]
    degrees: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.nvars) != len(self.degrees):
            raise ValueError("one degree per block")

    @cached_property
    def blocks(self) -> list[BlockBasis]:
        return [BlockBasis(v, e) for v, e in zip(self.nvars, self.degrees, strict=True)]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(block.size for block in self.blocks)

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def strides(self) -> Array:
        shape = self.shape
        return np.array([prod(shape[b + 1 :]) for b in range(len(shape))], dtype=np.int64)

    def evaluate(self, field: Field, block_points: list[Array]) -> Array:
        """Kronecker product of per-block monomial vectors at one point (one vector per block)."""
        vector = field.ones(1)
        for block, point in zip(self.blocks, block_points, strict=True):
            vector = field.mul(vector[:, None], block.evaluate(field, point)[0][None, :]).reshape(-1)
        return vector

    def times(self, other: ProductBasis) -> list[Array]:
        """Per-block multiplication tables between ``self`` and ``other``."""
        return [a.multiplication_table(b) for a, b in zip(self.blocks, other.blocks, strict=True)]

    def describe(self) -> dict:
        return {"nvars": list(self.nvars), "degrees": list(self.degrees), "size": self.size}
