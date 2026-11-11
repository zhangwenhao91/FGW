"""Current unified MHA: v7, Q resident, two-stage pipeline by default.

Import this on the configured Ascend development machine.
"""
from operators.mha.kernels.mha_paged_decode_ascend_v7 import build_kernel, mha_paged_decode

__all__ = ["build_kernel", "mha_paged_decode"]
