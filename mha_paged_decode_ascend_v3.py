"""Experimental BF16/D128 Cube QK/PV path, with v2 fallback elsewhere.

SIMD softmax/NZ packing macros adapted from tile-ai/tilelang core.py at
a7cb753dafc6b2514cbbd7c8431260c9d05e7df0; see TILELANG_LICENSE.
This first Cube bring-up duplicates one query across 128 rows. This wastes
Cube work and is NOT an optimized decode layout. NPU validation required.
"""
from functools import lru_cache
import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout
from mha_paged_decode_ascend_v2 import build_kernel as build_v2


def route(dtype, dim, page_size, sq=1, causal=False):
    return "cube_bf16_d128" if str(dtype) in ("bfloat16", "torch.bfloat16") and dim == 128 and (not causal or sq == 1) and page_size > 0 and page_size % 128 == 0 else "fallback_v2"


@lru_cache(maxsize=128)
def build_kernel(batch, sq, heads, dim, pool_rows, width, page_size, causal, dtype, device_index=None):
    if page_size <= 0 or pool_rows <= 0 or pool_rows % page_size or width <= 0:
        raise ValueError("positive page size, whole pool pages and positive table width required")
    if route(dtype, dim, page_size, sq, causal) != "cube_bf16_d128":
        return build_v2(batch, sq, heads, dim, pool_rows, width, page_size, causal, dtype, device_index)
    BR, BC, ROWS, VL, D = 128, 128, 64, 64, 128
    dtype, accum_dtype = "bfloat16", "float32"
    scale = D ** -0.5
    tasks = batch * sq * heads
    splits = 1
    while splits < 16 and splits * 2 <= width * page_size // BC and tasks * splits < 128:
        splits *= 2
    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC),
                "bfloat16",
            )

            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), softmax_full, "NORM_B32")

            for r in range(0, ROWS, 2):
                s00 = S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full)
                s01 = S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full)
                s10 = S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full)
                s11 = S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full)
                S.vsts(S_ub[r, 0], s00, softmax_full, "NORM_B32")
                S.vsts(S_ub[r, VL], s01, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, 0], s10, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, VL], s11, softmax_full, "NORM_B32")

                max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)
                max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)
                S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")

            S.mem_bar("VST_VLD")
            merged_max = S.vmax(S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), softmax_full)
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")

            for r in range(0, ROWS, 2):
                max0 = S.vld(m_ub[r], "BRC_B32")
                max1 = S.vld(m_ub[r + 1], "BRC_B32")
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                e00 = S.vexpdif(p00, max0, softmax_full)
                e10 = S.vexpdif(p10, max1, softmax_full)
                e01 = S.vexpdif(p01, max0, softmax_full)
                e11 = S.vexpdif(p11, max1, softmax_full)
                row_sum0 = S.vcadd(S.vadd(e00, e01, softmax_full), softmax_full)
                row_sum1 = S.vcadd(S.vadd(e10, e11, softmax_full), softmax_full)
                S.vsts(row_sum_ub[r], row_sum0, softmax_one, "ONEPT_B32")
                S.vsts(row_sum_ub[r + 1], row_sum1, softmax_one, "ONEPT_B32")

                even_bf16_0 = S.vcvt(e00, "bfloat16", softmax_full, sat=False, part=0)
                even_bf16_1 = S.vcvt(e10, "bfloat16", softmax_full, sat=False, part=0)
                odd_bf16_0 = S.vcvt(e01, "bfloat16", softmax_full, sat=False, part=1)
                odd_bf16_1 = S.vcvt(e11, "bfloat16", softmax_full, sat=False, part=1)
                merged_bits0 = S.vor(
                    T.reinterpret(even_bf16_0, "uint16x128"),
                    T.reinterpret(odd_bf16_0, "uint16x128"),
                    softmax_b16,
                )
                merged_bits1 = S.vor(
                    T.reinterpret(even_bf16_1, "uint16x128"),
                    T.reinterpret(odd_bf16_1, "uint16x128"),
                    softmax_b16,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits0, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )

            S.mem_bar("VST_VLD")
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), S.vld(m_ub[0], "NORM"), softmax_full)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            l_scaled = S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, softmax_full)
            l_updated = S.vadd(l_scaled, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def clear_O_ub(O_ub):
        with T.SimdVF(latency=142):
            clear_mask = S.pset(32, "PAT_ALL")
            output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
            for i in range(ROWS):
                for c in range(0, D, VL):
                    S.vsts(O_ub[i, c], output_zero, clear_mask, "NORM_B32")


    @T.prim_func
    def partial(
        Q: T.Tensor((tasks, D), dtype),
        K: T.Tensor((pool_rows, heads, D), dtype),
        V: T.Tensor((pool_rows, heads, D), dtype),
        Table: T.Tensor((batch, width), "int32"),
        Lengths: T.Tensor((batch,), "int32"),
        Parts: T.Tensor((tasks, splits, BR, D), accum_dtype),
        LSE: T.Tensor((tasks * splits * BR,), accum_dtype),
    ):
        with T.Kernel(tasks * splits) as bx:
            task = bx // splits
            split = bx % splits
            b = task // (sq * heads)
            qi = task // heads % sq
            h = task % heads
            Q_l1 = T.alloc_l1((BR, D), dtype)
            K_l1 = T.alloc_l1((BC, D), dtype)
            V_l1 = T.alloc_l1((D, BC), dtype)
            P_l1 = T.alloc_l1((BR, BC), dtype)
            qa = T.alloc_l0a((BR, D), dtype)
            kb = T.alloc_l0b((BC, D), dtype)
            pa = T.alloc_l0a((BR, BC), dtype)
            vb = T.alloc_l0b((D, BC), dtype)
            qk = T.alloc_l0c((BR, BC), accum_dtype)
            pv = T.alloc_l0c((BR, D), accum_dtype)
            scores = T.alloc_shared((ROWS, BC), accum_dtype)
            probs = T.alloc_shared((ROWS + 1, BC), dtype)
            T.annotate_layout({probs: make_ascend_compact_nz_layout(probs)})
            current = T.alloc_shared((ROWS, D), accum_dtype)
            output = T.alloc_shared((ROWS, D), accum_dtype)
            maximum = T.alloc_shared((ROWS,), accum_dtype)
            total = T.alloc_shared((ROWS,), accum_dtype)
            alpha = T.alloc_shared((ROWS,), accum_dtype)
            old_max = T.alloc_shared((ROWS,), accum_dtype)
            row_sum = T.alloc_shared((ROWS,), accum_dtype)
            lse = T.alloc_shared((ROWS,), accum_dtype)
            # Duplicate rows make both AIV halves compute the same query.
            # This is deliberately explicit for initial Cube-path validation.
            for row in T.serial(BR):
                T.copy(Q[task, :], Q_l1[row, :])
            init_m_l(maximum, total)
            clear_O_ub(output)
            end = Lengths[b]
            if causal:
                end = T.max(end - sq + qi + 1, 0)
            tiles = end // BC
            per_split = T.ceildiv(tiles, splits)
            begin = split * per_split
            count = T.max(T.min(begin + per_split, tiles) - begin, 0)
            # Keep automatic stage assignment. Multi-version only the streamed
            # operands/results; online row statistics remain loop-carried state.
            T.annotate_buffer_versions({K_l1: 2, V_l1: 2, qk: 2, pv: 2,
                                       scores: 2, probs: 2, current: 2, alpha: 2})
            for tile in T.Pipelined(count, num_stages=2):
                logical = (begin + tile) * BC
                physical = Table[b, logical // page_size] * page_size + logical % page_size
                T.copy(K[physical:physical + BC, h, :], K_l1)
                T.copy(V[physical:physical + BC, h, :], V_l1, transpose=True)
                T.copy(Q_l1, qa)
                T.copy(K_l1, kb)
                T.gemm(qa, kb, qk, transpose_B=True, clear_accum=True)
                T.dual_copy(qk, scores)
                softmax(scores, probs, maximum, total, alpha, old_max, row_sum)
                T.dual_copy(probs[0:ROWS, :], P_l1)
                T.copy(P_l1, pa)
                T.copy(V_l1, vb)
                T.gemm(pa, vb, pv, transpose_B=True, clear_accum=True)
                T.dual_copy(pv, current)
                accumulate_output(current, output, alpha)
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(ROWS, D):
                    output[r, d] = output[r, d] / T.max(total[r], 1.0e-30)
                for r in T.Parallel(ROWS):
                    if total[r] > 0:
                        lse[r] = T.log2(T.max(total[r], 1.0e-30)) * 0.6931471805599453 + maximum[r]
                    else:
                        lse[r] = -1.0e30
            T.dual_copy(output, Parts[task, split, :, :])
            # Integer indices in a multidimensional BufferRegion retain rank.
            # Flatten LSE so UB[ROWS] -> GM[BR] is an actual 1-D 1:2 copy.
            lse_offset = (task * splits + split) * BR
            T.dual_copy(lse, LSE[lse_offset:lse_offset + BR])

    @T.prim_func
    def merge(
        Parts: T.Tensor((tasks, splits, BR, D), accum_dtype),
        LSE: T.Tensor((tasks * splits * BR,), accum_dtype),
        O: T.Tensor((batch, sq, heads, D), dtype),
    ):
        with T.Kernel(tasks) as task:
            part = T.alloc_shared((D,), accum_dtype)
            score = T.alloc_shared((8,), accum_dtype)
            state = T.alloc_shared((8,), accum_dtype)
            acc = T.alloc_shared((D,), accum_dtype)
            out = T.alloc_shared((D,), dtype)
            with T.SimtVF(threads=128):
                for d in T.Parallel(D):
                    acc[d] = 0
                for i in T.Parallel(1):
                    state[0] = -1.0e30
                    state[1] = 0
            for split in T.serial(splits):
                T.copy(Parts[task, split, 0, :], part)
                lse_offset = (task * splits + split) * BR
                T.copy(LSE[lse_offset:lse_offset + 8], score)
                with T.SimtVF(threads=128):
                    for d in T.Parallel(D):
                        peak = T.max(state[0], score[0])
                        weight = T.if_then_else(score[0] > -1.0e29, T.exp(score[0] - peak), 0)
                        acc[d] = acc[d] * T.exp(state[0] - peak) + part[d] * weight
                    for i in T.Parallel(1):
                        peak = T.max(state[0], score[0])
                        weight = T.if_then_else(score[0] > -1.0e29, T.exp(score[0] - peak), 0)
                        state[1] = state[1] * T.exp(state[0] - peak) + weight
                        state[0] = peak
            with T.SimtVF(threads=128):
                for d in T.Parallel(D):
                    out[d] = T.cast(acc[d] / T.max(state[1], 1.0e-30), dtype)
            T.copy(out, O[task // (sq * heads), task // heads % sq, task % heads, :])

    partial_kernel = tilelang.compile(partial, out_idx=[5, 6], target="ascend", execution_backend="auto")
    merge_kernel = tilelang.compile(merge, out_idx=-1, target="ascend", execution_backend="auto")

    def launch(q, k, v, table, lengths):
        parts, lse = partial_kernel(q.view(tasks, D), k, v, table, lengths)
        return merge_kernel(parts, lse)
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
    if route(q.dtype, d, page_size, sq, is_causal) == "cube_bf16_d128":
        lengths = cache_seqlens.cpu().tolist()
        if any(n <= 0 or n % 128 or n > block_table.shape[1] * page_size for n in lengths):
            raise ValueError("Full-block Cube path requires positive 128-aligned cache lengths within table capacity")
    with torch.npu.device(q.device):
        kernel = build_kernel(b, sq, h, d, k_cache.shape[0], block_table.shape[1],
                              page_size, is_causal,
                              "float16" if q.dtype == torch.float16 else "bfloat16", q.device.index)
        return kernel(*inputs)
