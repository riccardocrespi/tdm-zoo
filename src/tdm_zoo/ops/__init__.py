"""Layer 2: operations over a field.

* :mod:`.linalg` - dense linear algebra (rank, rref, kernels, inverses, ...);
* :mod:`.operators`, :mod:`.structured` - fast linear operators (evaluation circuits);
* :mod:`.matrix_maps` - symbolic matrix-valued parametrisations of secret families;
* :mod:`.poly` - sparse multivariate polynomials and monomial bookkeeping.
"""

from . import linalg, matrix_maps, operators, poly, structured
from .matrix_maps import (
    ConstantMatrixMap,
    LinearMatrixMap,
    MatrixMap,
    ProductSumMatrixMap,
    SandwichMatrixMap,
)
from .operators import (
    ComposedOperator,
    DenseOperator,
    DiagonalOperator,
    HStackOperator,
    IdentityOperator,
    LinearOperator,
    PermutationOperator,
    SelectionOperator,
    SparseOperator,
    TransposedOperator,
    VStackOperator,
)
from .structured import (
    AccumulatorOperator,
    BlockDiagonalOperator,
    CirculantOperator,
    ToeplitzOperator,
    hankel_operator,
)

__all__ = [
    "AccumulatorOperator",
    "BlockDiagonalOperator",
    "CirculantOperator",
    "ComposedOperator",
    "ConstantMatrixMap",
    "DenseOperator",
    "DiagonalOperator",
    "HStackOperator",
    "IdentityOperator",
    "LinearMatrixMap",
    "LinearOperator",
    "MatrixMap",
    "PermutationOperator",
    "ProductSumMatrixMap",
    "SandwichMatrixMap",
    "SelectionOperator",
    "SparseOperator",
    "ToeplitzOperator",
    "TransposedOperator",
    "VStackOperator",
    "hankel_operator",
    "linalg",
    "matrix_maps",
    "operators",
    "poly",
    "structured",
]
