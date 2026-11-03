"""Peeling attack on layered TDMs, in Macaulay form.

* :mod:`.relations` - base relations ``Q`` annihilating ``A_1 f(X_1) A_2 f(X_2) A_3``;
* :mod:`.system` - the composed system ``p_i(W) = Q_i(x(W))`` and the witness;
* :mod:`.macaulay` - multi-homogeneous Macaulay matrices;
* :mod:`.distinguisher` - the rank test, registered as ``peeling_macaulay``.
"""

from .distinguisher import PeelingMacaulay, PeelingState
from .macaulay import MacaulayShape, macaulay_matrix, macaulay_shape
from .relations import BaseRelations, base_relations, core_map_values
from .system import PeelingBlock, PeelingSystem

__all__ = [
    "BaseRelations",
    "MacaulayShape",
    "PeelingBlock",
    "PeelingMacaulay",
    "PeelingState",
    "PeelingSystem",
    "base_relations",
    "core_map_values",
    "macaulay_matrix",
    "macaulay_shape",
]
