"""Blocked-KV SIMT prototype for TileLang Ascend a7cb753d.

One program per (request, query, head), one VF/online-softmax update per
16 or 32 keys. Still no Cube GEMM or split-KV. NPU validation required.
"""
from functools import lru_cache

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


@lru_cache(maxsize=128)
def build_kernel(batch, sq, heads, dim, pool_rows, width, page_size, causal, dtype,
                 device_index=None):
    if dim < 16 or dim > 256 or dim % 16:
        raise ValueError("v1 requires D divisible by 16, in [16, 256]")
    if page_size <= 0 or page_size % 16 or pool_rows % page_size:
        raise ValueError("v1 requires whole pages and page_size divisible by 16")
    tile = 32 if page_size % 32 == 0 else 16
    padded = tilelang.next_power_of_2(dim)
    threads = min(128, padded)
    scale = dim ** -0.5

    @T.prim_func
    def main(
        Q: T.Tensor((batch, sq, heads, dim), dtype),
        K: T.Tensor((pool_rows, heads, dim), dtype),
        V: T.Tensor((pool_rows, heads, dim), dtype),
        Table: T.Tensor((batch, width), "int32"),
        Lengths: T.Tensor((batch,), "int32"),
        O: T.Tensor((batch, sq, heads, dim), dtype),
    ):
        with T.Kernel(batch * sq * heads) as program:
            b = program // (sq * heads)
            qpos = program // heads % sq
            h = program % heads
            q_ub = T.alloc_shared((dim,), dtype)
            k_ub = T.alloc_shared((tile, dim), dtype)
            v_ub = T.alloc_shared((tile, dim), dtype)
            acc_ub = T.alloc_shared((dim,), "float32")
            out_ub = T.alloc_shared((dim,), dtype)
            state = T.alloc_shared((8,), "float32")
            T.copy(Q[b, qpos, h, :], q_ub)
            with T.SimtVF(threads=threads):
                for d in T.Parallel(padded):
                    if d < dim:
                        acc_ub[d] = 0
                for i in T.Parallel(1):
                    state[0] = -1.0e30
                    state[1] = 0
            end = Lengths[b]
            if causal:
                end = T.max(end - sq + qpos + 1, 0)
            for block in T.serial(T.ceildiv(end, tile)):
                row0 = block * tile
                physical = Table[b, row0 // page_size] * page_size + row0 % page_size
                # tile divides page_size: the full physical tile remains in one
                # allocated page even when the logical cache ends partway through it.
                T.copy(K[physical:physical + tile, h, :], k_ub)
                T.copy(V[physical:physical + tile, h, :], v_ub)
                with T.SimtVF(threads=threads):
                    dots = T.alloc_reducer((tile,), "float32", op="sum")
                    T.reducer_init(dots)
                    for j, d in T.Parallel(tile, padded, coalesced_width=T.int32(1)):
                        if row0 + j < end and d < dim:
                            T.reducer_update(dots[j], T.cast(q_ub[d], "float32") * T.cast(k_ub[j, d], "float32"))
                    scores = T.alloc_fragment((tile,), "float32")
                    T.finalize_reducer(dots, scores)
                    maximum = T.alloc_reducer((1,), "float32", op="max")
                    T.reducer_init(maximum)
                    for j in T.Parallel(tile):
                        scores[j] = T.if_then_else(row0 + j < end, scores[j] * scale, -1.0e30)
                        T.reducer_update(maximum[0], scores[j])
                    tile_max = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(maximum, tile_max)
                    probabilities = T.alloc_fragment((tile,), "float32")
                    total = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(total)
                    for j in T.Parallel(tile):
                        probabilities[j] = T.if_then_else(
                            row0 + j < end, T.exp(scores[j] - T.max(state[0], tile_max[0])), 0)
                        T.reducer_update(total[0], probabilities[j])
                    tile_sum = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(total, tile_sum)
                    weighted_v = T.alloc_reducer((padded,), "float32", op="sum")
                    T.reducer_init(weighted_v)
                    for j, d in T.Parallel(tile, padded, coalesced_width=T.int32(1)):
                        # Branch before reading V: 0*NaN is still NaN.
                        if row0 + j < end and d < dim:
                            T.reducer_update(weighted_v[d], probabilities[j] * T.cast(v_ub[j, d], "float32"))
                    tile_output = T.alloc_fragment((padded,), "float32")
                    T.finalize_reducer(weighted_v, tile_output)
                    for d in T.Parallel(padded):
                        if d < dim:
                            alpha = T.exp(state[0] - T.max(state[0], tile_max[0]))
                            acc_ub[d] = acc_ub[d] * alpha + tile_output[d]
                    for i in T.Parallel(1):
                        new_max = T.max(state[0], tile_max[0])
                        state[1] = state[1] * T.exp(state[0] - new_max) + tile_sum[0]
                        state[0] = new_max
            with T.SimtVF(threads=threads):
                for d in T.Parallel(padded):
                    if d < dim:
                        out_ub[d] = T.cast(acc_ub[d] / T.max(state[1], 1.0e-30), dtype)
            T.copy(out_ub, O[b, qpos, h, :])

    return tilelang.compile(main, out_idx=-1, target="ascend", execution_backend="auto")


def mha_paged_decode(q, k_cache, v_cache, block_table, cache_seqlens,
                     page_size, is_causal=False):
    """Same public tensor contract as v0; metadata values must be valid."""
    if q.ndim != 4 or k_cache.ndim != 3:
        raise ValueError("expected Q[B,Sq,H,D], K/V[N,H,D]")
    b, sq, h, d = q.shape
    if min(b, sq, h, d) <= 0 or block_table.ndim != 2:
        raise ValueError("positive Q extents and rank-2 table required")
    if k_cache.shape[1:] != (h, d) or v_cache.shape != k_cache.shape:
        raise ValueError("K/V shape mismatch")
    if block_table.shape[0] != b or block_table.shape[1] <= 0 or cache_seqlens.shape != (b,):
        raise ValueError("metadata shape mismatch")
    if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("matching FP16/BF16 Q/K/V required")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise ValueError("int32 metadata required")
    inputs = (q, k_cache, v_cache, block_table, cache_seqlens)
    if q.device.type != "npu" or any(x.device != q.device or not x.is_contiguous() for x in inputs):
        raise ValueError("contiguous inputs on the same NPU required")
    with torch.npu.device(q.device):
        kernel = build_kernel(b, sq, h, d, k_cache.shape[0], block_table.shape[1],
                              page_size, is_causal,
                              "float16" if q.dtype == torch.float16 else "bfloat16", q.device.index)
        return kernel(*inputs)
