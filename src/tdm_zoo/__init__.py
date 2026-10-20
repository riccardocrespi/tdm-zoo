"""tdm-zoo: trapdoored-matrix candidates over finite fields and their cryptanalysis.

Layers, bottom to top (each depends only on the ones below it):

0. :mod:`tdm_zoo.arith` - modular arithmetic kernels and prime-field linear-algebra backends;
1. :mod:`tdm_zoo.fields` - field-agnostic interface, F_p and F_{p^m};
2. :mod:`tdm_zoo.ops` - linear algebra, fast operators, symbolic parametrisations, polynomials;
3. :mod:`tdm_zoo.tdm` - secret families, public layers, layered products, named candidates;
4. :mod:`tdm_zoo.attacks` - distinguishers;

plus :mod:`tdm_zoo.experiments` to run distinguishers against candidates.
"""

from .fields import GF, ExtensionField, Field, PrimeField
from .tdm import make_candidate, make_family

__version__ = "0.1.0"

__all__ = ["GF", "ExtensionField", "Field", "PrimeField", "make_candidate", "make_family"]
