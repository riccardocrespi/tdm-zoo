"""Layer 3: trapdoored-matrix candidates.

* :mod:`.families` - secret matrix families (sampling, fast operator, symbolic ``f``/``g``);
* :mod:`.publics` - public layers (permutations, repetition, accumulators, ...);
* :mod:`.di14` - Druk-Ishai linear-time encoder/extractor;
* :mod:`.product` - layered products ``A_1 S_1 ... S_l A_{l+1}`` and their samples;
* :mod:`.candidates` - named constructions (``fft_style``, ``raa``, ``luof``, ``layered``).
"""

from . import candidates, di14, families, publics
from .candidates import CANDIDATES, Candidate, make_candidate
from .families import SECRET_FAMILIES, SecretFamily, make_family
from .product import LayeredProduct, PublicView, TDMSample

__all__ = [
    "CANDIDATES",
    "SECRET_FAMILIES",
    "Candidate",
    "LayeredProduct",
    "PublicView",
    "SecretFamily",
    "TDMSample",
    "candidates",
    "di14",
    "families",
    "make_candidate",
    "make_family",
    "publics",
]
