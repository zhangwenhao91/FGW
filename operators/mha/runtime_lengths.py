"""Opt-in runtime-shape families sharing restricted v7's computation.

B/H, split and merge row tile remain compile-time tuning parameters.
Q, physical pool rows and page-table width are runtime shape symbols.
Logical KV lengths remain device data in cache_seqlens, as in static v7.
"""
from functools import lru_cache


@lru_cache(maxsize=64)
def build_runtime_kernel(batch, heads, num_splits, merge_rows=8, device_index=0):
    if min(batch, heads) <= 0 or num_splits not in (1, 2, 4, 8, 16):
        raise ValueError("Positive B/H and split1/2/4/8/16 required")
    if merge_rows not in (1, 2, 4, 8):
        raise ValueError("merge_rows must be 1/2/4/8")
    from operators.mha.kernels.mha_paged_decode_ascend_v7 import build_kernel
    # Positive specialization anchors are replaced by symbols before device IR.
    width = max(4, num_splits)
    program = build_kernel(batch, merge_rows, heads, 128, width*128, width, 128,
                           False, "bfloat16", device_index, num_splits=num_splits,
                           split_policy="legacy", runtime_lengths=True)

    def call(q, k, v, table, lengths):
        import torch
        if (q.ndim != 4 or q.shape[0] != batch or q.shape[2:] != (heads, 128)
                or q.shape[1] <= 0 or q.dtype != torch.bfloat16):
            raise ValueError("Expected BF16 Q[B,Q,H,128] with positive Q")
        if (k.ndim != 3 or k.shape[1:] != (heads, 128) or k.shape[0] <= 0
                or k.shape[0] % 128 or v.shape != k.shape
                or k.dtype != q.dtype or v.dtype != q.dtype):
            raise ValueError("Expected matching BF16 paged K/V[pool_rows,H,128]")
        if (table.ndim != 2 or table.shape[0] != batch or table.shape[1] < num_splits
                or table.dtype != torch.int32 or lengths.shape != (batch,)
                or lengths.dtype != torch.int32):
            raise ValueError("Expected int32 table/lengths with sufficient split capacity")
        if any(t.device != q.device or t.device.type != "npu" or not t.is_contiguous()
               for t in (q, k, v, table, lengths)):
            raise ValueError("All inputs must be contiguous on the same NPU")
        # No host read of lengths/page IDs; caller retains positive, aligned KV.
        return program(q, k, v, table, lengths)

    call.family_key = (batch, heads, num_splits, merge_rows, device_index)
    call.program = program
    return call
