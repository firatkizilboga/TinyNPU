from __future__ import annotations

import numpy as np

from .ir import DType


class GoldenModel:
    def matmul(
        self,
        lhs,
        rhs,
        bias=None,
        multiplier: int = 1,
        shift: int = 0,
        activation: str = "none",
        out_dtype: DType = DType.INT16,
    ) -> np.ndarray:
        lhs_arr = np.array(lhs, dtype=np.int64)
        rhs_arr = np.array(rhs, dtype=np.int64)

        if bias is None:
            bias_arr = np.zeros(rhs_arr.shape[1], dtype=np.int64)
        else:
            bias_arr = np.array(bias, dtype=np.int64).flatten()

        acc = np.matmul(lhs_arr, rhs_arr)
        out = np.zeros(acc.shape, dtype=np.int32)
        for row in range(acc.shape[0]):
            for col in range(acc.shape[1]):
                out[row, col] = self._ppu(
                    acc[row, col],
                    bias_arr[col],
                    multiplier=multiplier,
                    shift=shift,
                    activation=activation,
                    out_dtype=out_dtype,
                )
        return out

    def _ppu(self, acc: int, bias: int, multiplier: int, shift: int, activation: str, out_dtype: DType) -> int:
        value = np.int64(acc) + np.int64(np.int32(bias))
        value *= np.int64(multiplier & 0xFFFF)
        if shift > 0:
            value = (value + (np.int64(1) << (shift - 1))) >> shift
        if activation == "relu":
            value = max(0, value)
        if out_dtype == DType.INT4:
            return int(np.clip(value, -8, 7))
        if out_dtype == DType.INT8:
            return int(np.clip(value, -128, 127))
        return int(np.clip(value, -32768, 32767))
