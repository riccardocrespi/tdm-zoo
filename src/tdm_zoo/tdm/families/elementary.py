"""Diagonal, block-diagonal and weighted-accumulator secret families."""

from __future__ import annotations

import numpy as np

from ...fields import Field
from ...ops import linalg
from ...ops.matrix_maps import LinearMatrixMap, MatrixMap
from ...ops.operators import DiagonalOperator, LinearOperator
from ...ops.structured import AccumulatorOperator, BlockDiagonalOperator
from .base import SECRET_FAMILIES, SecretFamily

Array = np.ndarray


@SECRET_FAMILIES.register("diagonal")
class DiagonalFamily(SecretFamily):
    """``diag(w)``; with ``nonzero=True`` the weights are sampled from F*."""

    def __init__(self, field: Field, n: int, nonzero: bool = True) -> None:
        super().__init__(field, n, n)
        self.nonzero = nonzero

    @property
    def nparams(self) -> int:
        return self.shape[0]

    def options(self) -> dict:
        return {"n": self.shape[0], "nonzero": self.nonzero}

    def sample_params(self, rng):
        sampler = self.field.random_nonzero if self.nonzero else self.field.random
        return sampler(rng, self.nparams)

    def operator(self, params) -> LinearOperator:
        return DiagonalOperator(self.field, params)

    def _param_map(self) -> MatrixMap:
        idx = np.arange(self.nparams)
        return LinearMatrixMap(self.field, self.shape, self.nparams, idx, idx, idx, 1)

    def _inverse_map(self) -> MatrixMap | None:
        return self.param_map if self.nonzero else None

    def inverse_params(self, params) -> Array:
        return self.field.inv(params)


@SECRET_FAMILIES.register("block_diagonal")
class BlockDiagonalFamily(SecretFamily):
    """``diag(B_1, ..., B_{n/b})`` with ``b x b`` blocks; parameters are the blocks, row-major.

    ``invertible=True`` samples each block from GL_b(F); otherwise blocks are uniform.
    """

    def __init__(self, field: Field, n: int, block: int, invertible: bool = True) -> None:
        if block <= 0 or n % block:
            raise ValueError("block size must divide n")
        super().__init__(field, n, n)
        self.block = block
        self.invertible = invertible

    @property
    def count(self) -> int:
        return self.shape[0] // self.block

    @property
    def nparams(self) -> int:
        return self.shape[0] * self.block

    def options(self) -> dict:
        return {"n": self.shape[0], "block": self.block, "invertible": self.invertible}

    def _blocks(self, params) -> Array:
        return np.asarray(params, dtype=np.int64).reshape(self.count, self.block, self.block)

    def sample_params(self, rng):
        blocks = self.field.random(rng, (self.count, self.block, self.block))
        if self.invertible:
            for index in range(self.count):
                while not linalg.is_invertible(self.field, blocks[index]):
                    blocks[index] = self.field.random(rng, (self.block, self.block))
        return blocks.reshape(-1)

    def operator(self, params) -> LinearOperator:
        return BlockDiagonalOperator(self.field, self._blocks(params))

    def _param_map(self) -> MatrixMap:
        b = self.block
        blk, r, c = np.meshgrid(np.arange(self.count), np.arange(b), np.arange(b), indexing="ij")
        param = (blk * b + r) * b + c
        return LinearMatrixMap(
            self.field,
            self.shape,
            self.nparams,
            param.ravel(),
            (blk * b + r).ravel(),
            (blk * b + c).ravel(),
            1,
        )

    def _inverse_map(self) -> MatrixMap | None:
        return self.param_map if self.invertible else None

    def inverse_params(self, params) -> Array:
        return np.stack([linalg.inv(self.field, block) for block in self._blocks(params)]).reshape(-1)


@SECRET_FAMILIES.register("weighted_accumulator")
class WeightedAccumulatorFamily(SecretFamily):
    """``S(w) = C_n diag(w)``: ``(S x)_a = sum_{b <= a} w_b x_b`` (RAA-style secret).

    Its inverse is ``diag(u) (I - Z)`` with ``u = 1/w``, again linear in ``u``.
    """

    def __init__(self, field: Field, n: int, nonzero: bool = True) -> None:
        super().__init__(field, n, n)
        self.nonzero = nonzero

    @property
    def nparams(self) -> int:
        return self.shape[0]

    def options(self) -> dict:
        return {"n": self.shape[0], "nonzero": self.nonzero}

    def sample_params(self, rng):
        sampler = self.field.random_nonzero if self.nonzero else self.field.random
        return sampler(rng, self.nparams)

    def operator(self, params) -> LinearOperator:
        return AccumulatorOperator(self.field, params)

    def _param_map(self) -> MatrixMap:
        a, b = np.tril_indices(self.shape[0])
        return LinearMatrixMap(self.field, self.shape, self.nparams, b, a, b, 1)

    def _inverse_map(self) -> MatrixMap | None:
        if not self.nonzero:
            return None
        n = self.shape[0]
        diag = np.arange(n)
        sub = np.arange(1, n)
        minus_one = int(self.field.neg(np.int64(1)))
        return LinearMatrixMap(
            self.field,
            self.shape,
            n,
            np.concatenate([diag, sub]),
            np.concatenate([diag, sub]),
            np.concatenate([diag, sub - 1]),
            np.concatenate([np.ones(n, dtype=np.int64), np.full(n - 1, minus_one)]),
        )

    def inverse_params(self, params) -> Array:
        return self.field.inv(params)
