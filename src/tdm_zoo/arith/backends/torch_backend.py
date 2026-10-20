"""PyTorch backend for CUDA GPUs (or CPU), exact modulo p.

Products are computed in float64 on chunks small enough to be exact; the
modulus is split into 16-bit limbs when ``(p-1)**2`` exceeds 2**53. CUDA has no
int64 GEMM, which is why the float path is used.
"""

from __future__ import annotations

import numpy as np

from .base import PrimeBackend

_FLOAT_EXACT = 2**53
_LIMB_BITS = 16


class TorchBackend(PrimeBackend):
    def __init__(self, device: str | None = None) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on environment
            raise RuntimeError("PyTorch is required for the torch backend") from error
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available")
        self._torch = torch
        self.device = torch.device(device)
        self.name = f"torch:{self.device}"

    # -- conversion ---------------------------------------------------------------------
    def _to(self, matrix: np.ndarray):
        return self._torch.as_tensor(np.ascontiguousarray(matrix, dtype=np.int64), device=self.device)

    @staticmethod
    def _back(tensor) -> np.ndarray:
        return tensor.to("cpu").numpy().astype(np.int64, copy=False)

    # -- products -----------------------------------------------------------------------
    def matmul(self, left: np.ndarray, right: np.ndarray, p: int) -> np.ndarray:
        if left.shape[1] != right.shape[0]:
            raise ValueError(f"incompatible shapes {left.shape} and {right.shape}")
        return self._back(self.matmul_device(self._to(left), self._to(right), p))

    def matmul_device(self, left, right, p: int):
        torch = self._torch
        left, right = torch.remainder(left, p), torch.remainder(right, p)
        if (p - 1) ** 2 < _FLOAT_EXACT:
            return self._float_mod(left, right, p, (p - 1) ** 2)
        mask = (1 << _LIMB_BITS) - 1
        l0, l1 = left & mask, left >> _LIMB_BITS
        r0, r1 = right & mask, right >> _LIMB_BITS
        bound = mask * mask
        low = self._float_mod(l0, r0, p, bound)
        mid = torch.remainder(self._float_mod(l0, r1, p, bound) + self._float_mod(l1, r0, p, bound), p)
        high = self._float_mod(l1, r1, p, bound)
        high = torch.remainder(high * pow(2, 2 * _LIMB_BITS, p), p)
        mid = torch.remainder(mid * pow(2, _LIMB_BITS, p), p)
        return torch.remainder(high + mid + low, p)

    def _float_mod(self, left, right, p: int, entry_bound: int):
        torch = self._torch
        inner = left.shape[1]
        chunk = max(1, (_FLOAT_EXACT - 1) // max(entry_bound, 1))
        result = torch.zeros((left.shape[0], right.shape[1]), dtype=torch.int64, device=self.device)
        for start in range(0, inner, chunk):
            stop = min(start + chunk, inner)
            block = left[:, start:stop].to(torch.float64) @ right[start:stop].to(torch.float64)
            result = torch.remainder(result + torch.remainder(block.to(torch.int64), p), p)
        return result

    # -- elimination --------------------------------------------------------------------
    def rref(self, matrix: np.ndarray, p: int) -> tuple[np.ndarray, tuple[int, ...]]:
        reduced, pivots = self.rref_device(self._to(matrix), p)
        return self._back(reduced), pivots

    def rref_device(self, matrix, p: int):
        torch = self._torch
        reduced = torch.remainder(matrix, p).clone()
        rows, columns = map(int, reduced.shape)
        nonzero_any = (reduced != 0).any(dim=0).to("cpu").numpy()
        pivots: list[int] = []
        row = 0
        for column in range(columns):
            if row == rows:
                break
            if not nonzero_any[column]:
                continue
            candidates = torch.nonzero(reduced[row:, column], as_tuple=False)
            if candidates.numel() == 0:
                continue
            pivot = row + int(candidates[0, 0])
            if pivot != row:
                reduced[[row, pivot]] = reduced[[pivot, row]]
            scale = pow(int(reduced[row, column]), -1, p)
            reduced[row, column:] = torch.remainder(reduced[row, column:] * scale, p)
            factors = reduced[:, column].clone()
            factors[row] = 0
            reduced[:, column:] = torch.remainder(
                reduced[:, column:] - factors[:, None] * reduced[row, column:][None, :], p
            )
            pivots.append(column)
            row += 1
        return reduced, tuple(pivots)
