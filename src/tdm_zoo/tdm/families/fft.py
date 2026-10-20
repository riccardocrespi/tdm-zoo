"""FFT-style secret families: Toeplitz, Hankel, circulant and quasi-cyclic.

Inverse parametrisations:

* circulant - the inverse of a circulant is circulant, so ``g = f``;
* Toeplitz - the Gohberg-Semencul-type formula of the paper, with ``2k``
  parameters ``(x, z)`` and degree 2::

      g_T(x, z) = L(z) U(Z J x) - L(x) U(-e_0 + Z J z)
                = L(z) U(Z J x) - L(x) U(Z J z) + L(x),

  where ``L(v)`` is lower-triangular Toeplitz with first column ``v`` and
  ``U(v)`` upper-triangular Toeplitz with first row ``v``. For invertible ``T``
  the witness is ``x = T^{-1} e_0`` and ``z = T^{-1} Z J r`` with ``r`` the first
  row of ``T`` (verified exhaustively for small parameters, including the cases
  ``x_0 = 0`` where the textbook formula does not apply);
* Hankel - ``H = T J`` with ``T`` Toeplitz on the same parameters, so
  ``g_H = J g_T``.
"""

from __future__ import annotations

import numpy as np

from ...fields import Field
from ...ops import linalg
from ...ops.matrix_maps import ConstantMatrixMap, LinearMatrixMap, MatrixMap, ProductSumMatrixMap
from ...ops.operators import (
    DiagonalOperator,
    HStackOperator,
    LinearOperator,
    VStackOperator,
)
from ...ops.structured import CirculantOperator, ToeplitzOperator, hankel_operator
from .base import SECRET_FAMILIES, SecretFamily

Array = np.ndarray


def _check_square_invertible(rows: int, cols: int, invertible: bool | None) -> bool:
    if invertible is None:
        return rows == cols
    if invertible and rows != cols:
        raise ValueError("only square secrets can be invertible")
    return invertible


# --- Gohberg-Semencul building blocks ------------------------------------------------------
def _lower_toeplitz_map(field: Field, k: int, offset: int, nparams: int) -> LinearMatrixMap:
    """``L(v)`` with ``v = params[offset : offset + k]``."""
    i, j = np.tril_indices(k)
    return LinearMatrixMap(field, (k, k), nparams, offset + (i - j), i, j, 1)


def _upper_toeplitz_shift_reverse_map(field: Field, k: int, offset: int, nparams: int) -> LinearMatrixMap:
    """``U(Z J v)`` with ``v = params[offset : offset + k]``.

    Entry ``(i, j)`` is ``v_{k-(j-i)}`` for ``j > i`` and zero otherwise.
    """
    i, j = np.triu_indices(k, 1)
    return LinearMatrixMap(field, (k, k), nparams, offset + k - (j - i), i, j, 1)


def gohberg_semencul_map(field: Field, k: int, left: Array | None = None) -> ProductSumMatrixMap:
    """``g_T`` (optionally premultiplied by a constant matrix ``left``) on parameters ``(x, z)``."""
    n = 2 * k
    lx, lz = _lower_toeplitz_map(field, k, 0, n), _lower_toeplitz_map(field, k, k, n)
    ux, uz = (
        _upper_toeplitz_shift_reverse_map(field, k, 0, n),
        _upper_toeplitz_shift_reverse_map(field, k, k, n),
    )
    prefix = [] if left is None else [ConstantMatrixMap(field, left, n)]
    return ProductSumMatrixMap(field, [(1, prefix + [lz, ux]), (-1, prefix + [lx, uz]), (1, prefix + [lx])])


def gohberg_semencul_witness(field: Field, toeplitz: Array) -> Array:
    """Parameters ``(x, z)`` with ``g_T(x, z) = toeplitz^{-1}``."""
    k = toeplitz.shape[0]
    first_row = toeplitz[0]
    shifted_reversed = field.zeros(k)
    shifted_reversed[1:] = first_row[::-1][: k - 1]
    e0 = field.zeros(k)
    e0[0] = 1
    x = linalg.solve(field, toeplitz, e0)
    z = linalg.solve(field, toeplitz, shifted_reversed)
    return np.concatenate([x, z])


# --- families --------------------------------------------------------------------------------
@SECRET_FAMILIES.register("toeplitz")
class ToeplitzFamily(SecretFamily):
    """``T[i, j] = a[i - j + cols - 1]`` with ``rows + cols - 1`` parameters.

    ``invertible`` defaults to ``True`` for square shapes (rejection sampling).
    """

    def __init__(
        self, field: Field, rows: int, cols: int | None = None, invertible: bool | None = None
    ) -> None:
        cols = rows if cols is None else cols
        super().__init__(field, rows, cols)
        self.invertible = _check_square_invertible(rows, cols, invertible)

    @property
    def nparams(self) -> int:
        return self.shape[0] + self.shape[1] - 1

    def options(self) -> dict:
        return {"rows": self.shape[0], "cols": self.shape[1], "invertible": self.invertible}

    def sample_params(self, rng):
        draw = lambda g: self.field.random(g, self.nparams)  # noqa: E731
        return self._rejection_sample(rng, draw, self._is_invertible) if self.invertible else draw(rng)

    def operator(self, params) -> LinearOperator:
        return ToeplitzOperator(self.field, params, *self.shape)

    def _param_map(self) -> MatrixMap:
        rows, cols = self.shape
        i, j = np.indices(self.shape)
        return LinearMatrixMap(
            self.field, self.shape, self.nparams, (i - j + cols - 1).ravel(), i.ravel(), j.ravel(), 1
        )

    def _inverse_map(self) -> MatrixMap | None:
        return gohberg_semencul_map(self.field, self.shape[0]) if self.invertible else None

    def inverse_params(self, params) -> Array:
        return gohberg_semencul_witness(self.field, self.matrix(params))


@SECRET_FAMILIES.register("hankel")
class HankelFamily(SecretFamily):
    """``H[i, j] = h[i + j]`` with ``rows + cols - 1`` parameters; ``H = Toeplitz(h) J``."""

    def __init__(
        self, field: Field, rows: int, cols: int | None = None, invertible: bool | None = None
    ) -> None:
        cols = rows if cols is None else cols
        super().__init__(field, rows, cols)
        self.invertible = _check_square_invertible(rows, cols, invertible)

    @property
    def nparams(self) -> int:
        return self.shape[0] + self.shape[1] - 1

    def options(self) -> dict:
        return {"rows": self.shape[0], "cols": self.shape[1], "invertible": self.invertible}

    def sample_params(self, rng):
        draw = lambda g: self.field.random(g, self.nparams)  # noqa: E731
        return self._rejection_sample(rng, draw, self._is_invertible) if self.invertible else draw(rng)

    def operator(self, params) -> LinearOperator:
        return hankel_operator(self.field, params, *self.shape)

    def _param_map(self) -> MatrixMap:
        i, j = np.indices(self.shape)
        return LinearMatrixMap(self.field, self.shape, self.nparams, (i + j).ravel(), i.ravel(), j.ravel(), 1)

    def _inverse_map(self) -> MatrixMap | None:
        if not self.invertible:
            return None
        k = self.shape[0]
        return gohberg_semencul_map(self.field, k, left=self.field.eye(k)[::-1])

    def inverse_params(self, params) -> Array:
        toeplitz = self.matrix(params)[:, ::-1]  # H J
        return gohberg_semencul_witness(self.field, toeplitz)


@SECRET_FAMILIES.register("circulant")
class CirculantFamily(SecretFamily):
    """``C[i, j] = c[(i - j) mod n]``; parameters are the first column."""

    def __init__(self, field: Field, n: int, invertible: bool = True) -> None:
        super().__init__(field, n, n)
        self.invertible = invertible

    @property
    def nparams(self) -> int:
        return self.shape[0]

    def options(self) -> dict:
        return {"n": self.shape[0], "invertible": self.invertible}

    def sample_params(self, rng):
        draw = lambda g: self.field.random(g, self.nparams)  # noqa: E731
        return self._rejection_sample(rng, draw, self._is_invertible) if self.invertible else draw(rng)

    def operator(self, params) -> LinearOperator:
        return CirculantOperator(self.field, params)

    def _param_map(self) -> MatrixMap:
        n = self.shape[0]
        i, j = np.indices(self.shape)
        return LinearMatrixMap(self.field, self.shape, n, ((i - j) % n).ravel(), i.ravel(), j.ravel(), 1)

    def _inverse_map(self) -> MatrixMap | None:
        return self.param_map if self.invertible else None

    def inverse_params(self, params) -> Array:
        e0 = self.field.zeros(self.shape[0])
        e0[0] = 1
        return linalg.solve(self.field, self.matrix(params), e0)


@SECRET_FAMILIES.register("quasi_cyclic")
class QuasiCyclicFamily(SecretFamily):
    """``[I | C]`` (``k x 2k``, ``orientation="horizontal"``) or ``[I // C]`` (``2k x k``).

    ``C`` is a ``k x k`` circulant. The family is affine in ``C``; to give the
    attack a *linear* parametrisation we add a homogenising parameter ``w_0`` for
    the identity block, i.e. ``f(w_0, c) = [w_0 I | C(c)]``. Samples always
    have ``w_0 = 1``.
    """

    def __init__(self, field: Field, k: int, orientation: str = "horizontal") -> None:
        if orientation not in ("horizontal", "vertical"):
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        shape = (k, 2 * k) if orientation == "horizontal" else (2 * k, k)
        super().__init__(field, *shape)
        self.k = k
        self.orientation = orientation

    @property
    def nparams(self) -> int:
        return self.k + 1

    def options(self) -> dict:
        return {"k": self.k, "orientation": self.orientation}

    def sample_params(self, rng):
        return np.concatenate([[1], self.field.random(rng, self.k)]).astype(np.int64)

    def operator(self, params) -> LinearOperator:
        params = np.asarray(params, dtype=np.int64)
        blocks = [
            DiagonalOperator(self.field, np.full(self.k, params[0], dtype=np.int64)),
            CirculantOperator(self.field, params[1:]),
        ]
        return HStackOperator(blocks) if self.orientation == "horizontal" else VStackOperator(blocks)

    def _param_map(self) -> MatrixMap:
        k = self.k
        diag = np.arange(k)
        i, j = (a.ravel() for a in np.indices((k, k)))
        circ_param = 1 + (i - j) % k
        if self.orientation == "horizontal":
            rows = np.concatenate([diag, i])
            cols = np.concatenate([diag, k + j])
        else:
            rows = np.concatenate([diag, k + i])
            cols = np.concatenate([diag, j])
        params = np.concatenate([np.zeros(k, dtype=np.int64), circ_param])
        return LinearMatrixMap(self.field, self.shape, self.nparams, params, rows, cols, 1)
