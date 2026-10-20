"""Linear uniform output family of Druk-Ishai [DI14], over any finite field.

A member of the family is the linear map (applied right to left)

    A_r = Ext o H_r o Enc :  F^k  ->  F^N  ->  F^N  ->  F^k

* ``Enc`` (Theorem 2): a Spielman-style linear-time code (Lemma 4) followed by
  distance amplification along a ``Delta``-regular bipartite graph (Lemma 5),
  whose output is parsed into ``N / Delta`` blocks of size ``Delta``;
* ``H_r``: block-diagonal with uniformly random ``Delta x Delta`` blocks - the
  only secret randomness;
* ``Ext`` (Theorem 3): a linear extractor for bit-fixing sources, i.e. the
  transpose of a Spielman-style encoder, truncated to ``k`` outputs.

Deviations from the paper, all standard in implementations:

* the explicit lossless expanders of Capalbo et al. (Lemma 3) and the graphs of
  Lemma 5 are replaced by random graphs of the same degrees (lossless expanders
  with high probability);
* the base code ``C_0`` of Lemma 4 is a random rate-1/4 code of length ``n0``;
* message lengths are padded with zeros, and the extractor's code is punctured,
  to reach the lengths ``n0 * 2**h`` of the recursion.

With constant degrees the guarantees of [DI14] are asymptotic; for small
dimensions these objects reproduce the *structure* of the construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fields import Field
from ..ops import linalg
from ..ops.operators import (
    ComposedOperator,
    DenseOperator,
    LinearOperator,
    SelectionOperator,
    SparseOperator,
)


@dataclass(frozen=True)
class DI14Params:
    base_length: int = 8  # n0 in Lemma 4; divisible by 4
    expander_degree: int = 6  # left degree of the Lemma-3 graphs
    block_size: int = 4  # Delta (= beta) in Lemma 5 / Construction 1

    def __post_init__(self) -> None:
        if self.base_length <= 0 or self.base_length % 4:
            raise ValueError("base_length must be a positive multiple of 4")
        if self.expander_degree <= 0 or self.block_size <= 0:
            raise ValueError("degrees must be positive")


# --- graphs ----------------------------------------------------------------------------------
def random_left_regular_graph(n_left: int, n_right: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    """Neighbour lists ``(n_left, d)`` with ``d = min(degree, n_right)`` distinct right vertices."""
    d = min(degree, n_right)
    return np.argsort(rng.random((n_left, n_right)), axis=1)[:, :d]


def random_regular_bipartite(n: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    """``gamma[i, j]``: j-th neighbour of right vertex ``i`` in a union of ``degree`` random matchings."""
    return np.stack([rng.permutation(n) for _ in range(degree)], axis=1)


def compression_operator(field: Field, graph: np.ndarray, n_right: int) -> SparseOperator:
    """``(m1)_j = sum_{i in Gamma(j)} m_i`` for the bipartite graph given by left neighbour lists."""
    n_left, d = graph.shape
    rows = graph.ravel()
    cols = np.repeat(np.arange(n_left), d)
    return SparseOperator(field, rows, cols, field.ones(rows.size), (n_right, n_left))


# --- Lemma 4: Spielman-style recursive code ----------------------------------------------------
def message_length(level: int, base_length: int) -> int:
    return base_length * 2**level // 4


def level_for(length: int, base_length: int) -> int:
    level = 0
    while message_length(level, base_length) < length:
        level += 1
    return level


class SpielmanCode(LinearOperator):
    """``E_h : F^{n0 2^{h-2}} -> F^{n0 2^h}``, ``E_h(m) = m o E_{h-1}(G_{h-2} m) o G_{h-1}(...)``."""

    def __init__(self, field: Field, level: int, params: DI14Params, rng: np.random.Generator) -> None:
        msg = message_length(level, params.base_length)
        super().__init__(field, (4 * msg, msg))
        self.level = level
        if level == 0:
            generator = field.random(rng, (4 * msg, msg))
            while linalg.rank(field, generator) < msg:
                generator = field.random(rng, (4 * msg, msg))
            self.base = DenseOperator(field, generator)
            return
        self.base = None
        half = msg // 2
        self.compress_in = compression_operator(
            field, random_left_regular_graph(msg, half, params.expander_degree, rng), half
        )
        self.inner = SpielmanCode(field, level - 1, params, rng)
        self.compress_out = compression_operator(
            field, random_left_regular_graph(2 * msg, msg, params.expander_degree, rng), msg
        )

    def _apply(self, x: np.ndarray) -> np.ndarray:
        if self.base is not None:
            return self.base._apply(x)
        c1 = self.inner._apply(self.compress_in._apply(x))
        c2 = self.compress_out._apply(c1)
        return np.concatenate([x, c1, c2], axis=0)

    def _apply_T(self, y: np.ndarray) -> np.ndarray:
        if self.base is not None:
            return self.base._apply_T(y)
        msg = self.cols
        vm, vc1, vc2 = y[:msg], y[msg : 3 * msg], y[3 * msg :]
        t = self.field.add(vc1, self.compress_out._apply_T(vc2))
        return self.field.add(vm, self.compress_in._apply_T(self.inner._apply_T(t)))

    @property
    def cost(self) -> int:
        if self.base is not None:
            return self.base.cost
        return self.compress_in.cost + self.inner.cost + self.compress_out.cost


# --- Theorems 2 and 3 --------------------------------------------------------------------------
def di14_encoder(field: Field, k: int, params: DI14Params, rng: np.random.Generator) -> LinearOperator:
    """``Enc : F^k -> F^N`` with ``N = 4 L Delta`` (``L >= k`` the padded message length)."""
    code = SpielmanCode(field, level_for(k, params.base_length), params, rng)
    pad = SelectionOperator(field, np.arange(k), code.cols).T
    graph = random_regular_bipartite(code.rows, params.block_size, rng)
    redistribute = SelectionOperator(field, graph.ravel(), code.rows)
    return ComposedOperator([redistribute, code, pad])


def di14_extractor(
    field: Field, n: int, k: int, params: DI14Params, rng: np.random.Generator
) -> LinearOperator:
    """``Ext : F^N -> F^k``: transpose of a (punctured) Spielman code, truncated to ``k`` outputs."""
    code = SpielmanCode(field, level_for(-(-n // 4), params.base_length), params, rng)
    if code.cols < k:
        raise ValueError("extractor code is too short for the requested output length")
    punctured = ComposedOperator([SelectionOperator(field, np.arange(n), code.rows), code])
    truncate = SelectionOperator(field, np.arange(k), code.cols)
    return ComposedOperator([truncate, punctured.T])


def di14_dimension(k: int, params: DI14Params) -> int:
    """The inner dimension ``N`` used by :func:`di14_encoder`."""
    return 4 * message_length(level_for(k, params.base_length), params.base_length) * params.block_size


@dataclass(frozen=True)
class LUOFCodes:
    """Public encoder/extractor pair of the linear uniform output family."""

    encoder: LinearOperator  # F^k -> F^N
    extractor: LinearOperator  # F^N -> F^k
    block_size: int
    construction: str

    @property
    def inner_dimension(self) -> int:
        return self.encoder.rows


def sample_luof_codes(
    field: Field,
    k: int,
    rng: np.random.Generator,
    params: DI14Params | None = None,
    construction: str = "di14",
) -> LUOFCodes:
    """``construction="di14"`` (faithful structure) or ``"random"`` (dense random ``Enc``, ``Ext``
    with the same dimensions)."""
    params = params or DI14Params()
    n = di14_dimension(k, params)
    if construction == "di14":
        encoder = di14_encoder(field, k, params, rng)
        extractor = di14_extractor(field, n, k, params, rng)
    elif construction == "random":
        encoder = DenseOperator(field, field.random(rng, (n, k)))
        extractor = DenseOperator(field, field.random(rng, (k, n)))
    else:
        raise ValueError("construction must be 'di14' or 'random'")
    return LUOFCodes(encoder, extractor, params.block_size, construction)
