"""Layer 4: distinguishers against layered matrix products.

Only the interface lives here for now. Planned implementations, following the
paper draft:

* generic rank distinguisher (three-layer base case + recursion), for square
  invertible layers;
* its rectangular extension (equation / inverse-substitution choice per level).

Both consume :attr:`SecretFamily.param_map` (``f``), :attr:`SecretFamily.inverse_map`
(``g``) and the revealed public matrices from :class:`~tdm_zoo.tdm.PublicView`.
Add an attack as one module that subclasses :class:`Distinguisher` and registers
itself with ``@DISTINGUISHERS.register("<name>")``.
"""

from . import peeling
from .base import DISTINGUISHERS, Decision, Distinguisher
from .peeling import PeelingMacaulay

__all__ = ["DISTINGUISHERS", "Decision", "Distinguisher", "PeelingMacaulay", "peeling"]
