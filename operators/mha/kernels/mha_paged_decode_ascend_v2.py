"""Split-KV blocked SIMT prototype for TileLang Ascend a7cb753d.

Multiple programs per output process disjoint live KV tiles; a second kernel
merges FP32 unnormalized sums. No Cube GEMM. NPU validation required.
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
        raise ValueError("v2 requires D divisible by 16, in [16, 256]")
    if page_size <= 0 or page_size % 16 or pool_rows % page_size:
        raise ValueError("v2 requires whole pages and page_size divisible by 16")
    tile = 32 if page_size % 32 == 0 else 16
    padded = tilelang.next_power_of_2(dim)
    threads = min(128, padded)
    scale = dim ** -0.5
    tasks = batch * sq * heads
    capacity_tiles = width * page_size // tile
    splits = 1
    while splits < 16 and splits * 2 <= capacity_tiles and tasks * splits < 128:
        splits *= 2
    if splits == 1:
        from operators.mha.kernels.mha_paged_decode_ascend_v1 import build_kernel as unsplit
        return unsplit(batch, sq, heads, dim, pool_rows, width, page_size, causal, dtype, device_index)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, sq, heads, dim), dtype),
        K: T.Tensor((pool_rows, heads, dim), dtype),
        V: T.Tensor((pool_rows, heads, dim), dtype),
        Table: T.Tensor((batch, width), "int32"),
        Lengths: T.Tensor((batch,), "int32"),
        Partial: T.Tensor((tasks, splits, padded), "float32"),
        Stats: T.Tensor((tasks, splits, 8), "float32"),
    ):
        with T.Kernel(tasks * splits) as block_id:
            program = block_id // splits
            split = block_id % splits
            b = program // (sq * heads)
            qpos = program // heads % sq
            h = program % heads
            q_ub = T.alloc_shared((dim,), dtype)
            k_ub = T.alloc_shared((tile, dim), dtype)
            v_ub = T.alloc_shared((tile, dim), dtype)
            acc_ub = T.alloc_shared((dim,), "float32")
            partial_ub = T.alloc_shared((padded,), "float32")
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
            # Partition live tiles, not physical pool rows. Empty splits still
            # publish zero sums and accumulators, so no uninitialized reads.
            tiles = T.ceildiv(end, tile)
            per_split = T.ceildiv(tiles, splits)
            begin = split * per_split
            count = T.max(T.min(begin + per_split, tiles) - begin, 0)
            for block in T.serial(count):
                row0 = (begin + block) * tile
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
                    partial_ub[d] = 0
                    if d < dim:
                        partial_ub[d] = acc_ub[d]
                for i in T.Parallel(8):
                    if i >= 2:
                        state[i] = 0
            T.copy(partial_ub, Partial[program, split, :])
            T.copy(state, Stats[program, split, :])

    @T.prim_func
    def merge(
        Partial: T.Tensor((tasks, splits, padded), "float32"),
        Stats: T.Tensor((tasks, splits, 8), "float32"),
        O: T.Tensor((batch, sq, heads, dim), dtype),
    ):
        with T.Kernel(tasks) as program:
            b = program // (sq * heads)
            qpos = program // heads % sq
            h = program % heads
            acc = T.alloc_shared((padded,), "float32")
            part = T.alloc_shared((padded,), "float32")
            incoming = T.alloc_shared((8,), "float32")
            running = T.alloc_shared((8,), "float32")
            out = T.alloc_shared((dim,), dtype)
            with T.SimtVF(threads=threads):
                for d in T.Parallel(padded):
                    acc[d] = 0
                for i in T.Parallel(1):
                    running[0] = -1.0e30
                    running[1] = 0
            for split in T.serial(splits):
                T.copy(Partial[program, split, :], part)
                T.copy(Stats[program, split, :], incoming)
                with T.SimtVF(threads=threads):
                    for d in T.Parallel(padded):
                        peak = T.max(running[0], incoming[0])
                        acc[d] = acc[d] * T.exp(running[0] - peak) + part[d] * T.exp(incoming[0] - peak)
                    for i in T.Parallel(1):
                        peak = T.max(running[0], incoming[0])
                        running[1] = running[1] * T.exp(running[0] - peak) + incoming[1] * T.exp(incoming[0] - peak)
                        running[0] = peak
            with T.SimtVF(threads=threads):
                for d in T.Parallel(padded):
                    if d < dim:
                        out[d] = T.cast(acc[d] / T.max(running[1], 1.0e-30), dtype)
            T.copy(out, O[b, qpos, h, :])

    partial_kernel = tilelang.compile(main, out_idx=[5, 6], target="ascend", execution_backend="auto")
    merge_kernel = tilelang.compile(merge, out_idx=-1, target="ascend", execution_backend="auto")

    def launch(q, k, v, table, lengths):
        # Both launches use the current stream; stream ordering publishes partials
        # before merge. Temporary buffers are per invocation, not shared state.
        partial, stats = partial_kernel(q, k, v, table, lengths)
        return merge_kernel(partial, stats)

    return launch


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
