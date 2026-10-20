"""FFT-style candidates [EMVP Sec. 5.2]: Toeplitz, Hankel, circulant or quasi-cyclic secrets.

Square variants (``secret`` in ``toeplitz``, ``hankel``, ``circulant``, or a list
mixing them, one per layer)::

    M = S_1 Pi_2 S_2 Pi_3 ... Pi_l S_l        (k x k secrets)

The quasi-cyclic variant generalises EMVP's Eq. (1) with expansion ``c = 2``::

    M = [I | C_1] Pi_2 C_2 Pi_3 ... Pi_{l-1} C_{l-1} Pi_l [I // C_l]

with ``C_2, ..., C_{l-1}`` circulant ``2k x 2k``. For ``l = 3`` this is
``S_L Pi_L S Pi_R S_R``. (EMVP states ``S_L`` in ``F^{ck x k}``; the shapes compose
only with ``S_L`` being ``k x ck``, which is what is implemented.)

All permutations are public. With ``outer_permutations=True`` the products are
additionally wrapped by public permutations ``A_1 = Pi_1`` and ``A_{l+1} = Pi_{l+1}``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...fields import Field
from ..families import CirculantFamily, HankelFamily, QuasiCyclicFamily, SecretFamily, ToeplitzFamily
from ..product import LayeredProduct
from ..publics import identity, random_permutation
from .base import CANDIDATES, Candidate

SQUARE_FAMILIES = {"toeplitz": ToeplitzFamily, "hankel": HankelFamily, "circulant": CirculantFamily}


@CANDIDATES.register("fft_style")
class FFTStyle(Candidate):
    def __init__(
        self,
        field: Field,
        k: int,
        ell: int = 3,
        secret: str | Sequence[str] = "toeplitz",
        invertible: bool = True,
        outer_permutations: bool = False,
    ) -> None:
        super().__init__(field, k, ell)
        if secret == "quasi_cyclic":
            if ell < 2:
                raise ValueError("the quasi-cyclic variant needs ell >= 2")
            self.secret: str | tuple[str, ...] = "quasi_cyclic"
        else:
            names = (secret,) * ell if isinstance(secret, str) else tuple(secret)
            if len(names) != ell or any(name not in SQUARE_FAMILIES for name in names):
                raise ValueError(
                    f"secret must be 'quasi_cyclic' or name(s) among {sorted(SQUARE_FAMILIES)}, one per layer"
                )
            self.secret = names[0] if len(set(names)) == 1 else names
        self.invertible = invertible
        self.outer_permutations = outer_permutations

    def options(self) -> dict:
        return {
            "secret": self.secret,
            "invertible": self.invertible,
            "outer_permutations": self.outer_permutations,
        }

    def _families(self) -> list[SecretFamily]:
        f, k = self.field, self.k
        if self.secret == "quasi_cyclic":
            middle = [CirculantFamily(f, 2 * k, invertible=self.invertible) for _ in range(self.ell - 2)]
            return [QuasiCyclicFamily(f, k, "horizontal"), *middle, QuasiCyclicFamily(f, k, "vertical")]
        names = (self.secret,) * self.ell if isinstance(self.secret, str) else self.secret
        return [SQUARE_FAMILIES[name](f, k, invertible=self.invertible) for name in names]

    def sample_product(self, rng) -> LayeredProduct:
        families = self._families()
        inner = [random_permutation(self.field, s.shape[1], rng) for s in families[:-1]]
        outer = (
            [random_permutation(self.field, self.k, rng) for _ in range(2)]
            if self.outer_permutations
            else [identity(self.field, self.k)] * 2
        )
        return LayeredProduct(self.field, [outer[0], *inner, outer[1]], families, metadata=self.describe())
