"""Experimental BF16/D128 Cube QK/PV path, with v2 fallback elsewhere.

SIMD softmax/NZ packing macros adapted from tile-ai/tilelang core.py at
a7cb753dafc6b2514cbbd7c8431260c9d05e7df0; see third_party/licenses/TILELANG_LICENSE.
Single-split direct output; multi-split batched merge with per-row weights; Q tiles contain up to 128 distinct query rows; partial Q tiles are zero-padded.
KV lengths must remain positive multiples of 128. NPU validation required.
"""
from functools import lru_cache
import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout
from operators.mha.kernels.mha_paged_decode_ascend_v2 import build_kernel as build_v2


def route(dtype, dim, page_size, sq=1, causal=False):
    return "cube_bf16_d128" if str(dtype) in ("bfloat16", "torch.bfloat16") and dim == 128 and (not causal or sq == 1) and page_size > 0 and page_size % 128 == 0 else "fallback_v2"


@lru_cache(maxsize=128)
def build_kernel(batch, sq, heads, dim, pool_rows, width, page_size, causal, dtype, device_index=None,
                 pipeline_stages=2, buffer_versions=2, q_resident=True, kv_versions=None):
    kv_versions = buffer_versions if kv_versions is None else kv_versions
    if kv_versions not in (1, 2, 3):
        raise ValueError("K/V L1 versions must be 1, 2 or 3")
    if pipeline_stages not in (1, 2, 3) or buffer_versions not in (1, 2, 3):
        raise ValueError("Experimental pipeline stages and buffer versions must be 1, 2 or 3")
    if page_size <= 0 or pool_rows <= 0 or pool_rows % page_size or width <= 0:
        raise ValueError("positive page size, whole pool pages and positive table width required")
    if route(dtype, dim, page_size, sq, causal) != "cube_bf16_d128":
        return build_v2(batch, sq, heads, dim, pool_rows, width, page_size, causal, dtype, device_index)
    BR, BC, ROWS, VL, D = 128, 128, 64, 64, 128
    dtype, accum_dtype = "bfloat16", "float32"
    scale = D ** -0.5
    q_tiles = (sq + BR - 1) // BR
    tasks = batch * q_tiles * heads
    outputs = batch * sq * heads
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
        Q: T.Tensor((batch, heads, sq, D), dtype),
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
            b = task // (q_tiles * heads)
            qi = (task // heads % q_tiles) * BR
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
            # As in the reference PR, GM->L1 clamps a partial query tile
            # and zero-pads its missing rows. Only valid rows are merged out.
            T.copy(Q[b, h, qi:qi + BR, 0:D], Q_l1)
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
            T.annotate_buffer_versions({K_l1: kv_versions, V_l1: kv_versions,
                                       qk: buffer_versions, pv: buffer_versions,
                                       scores: buffer_versions, probs: buffer_versions,
                                       current: buffer_versions, alpha: buffer_versions})
            if q_resident:
                T.copy(Q_l1, qa)
            for tile in T.Pipelined(count, num_stages=pipeline_stages):
                logical = (begin + tile) * BC
                physical = Table[b, logical // page_size] * page_size + logical % page_size
                T.copy(K[physical:physical + BC, h, :], K_l1)
                T.copy(V[physical:physical + BC, h, :], V_l1, transpose=True)
                if not q_resident:
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

    merge_rows = min(8, tilelang.next_power_of_2(sq))
    merge_groups = (sq + merge_rows - 1) // merge_rows
    merge_work = batch * heads * merge_groups
    # Keep the grid bounded; large Q uses several work items per program.
    # Workloads below this cap retain their original grid and one iteration.
    merge_grid = min(merge_work, 32768)

    @T.prim_func
    def merge(
        Parts: T.Tensor((tasks, splits, BR, D), accum_dtype),
        LSE: T.Tensor((tasks * splits * BR,), accum_dtype),
        O: T.Tensor((batch, sq, heads, D), dtype),
    ):
        with T.Kernel(merge_grid) as core:
            for work in T.serial(T.ceildiv(merge_work, merge_grid)):
                program = work * merge_grid + core
                if program < merge_work:
                    b = program // (heads * merge_groups)
                    h = program // merge_groups % heads
                    qi = program % merge_groups * merge_rows
                    group_task = (b * q_tiles + qi // BR) * heads + h
                    row = qi % BR
                    part = T.alloc_shared((merge_rows, D), accum_dtype)
                    score = T.alloc_shared((merge_rows,), accum_dtype)
                    maximum = T.alloc_shared((merge_rows,), accum_dtype)
                    total = T.alloc_shared((merge_rows,), accum_dtype)
                    peak_rows = T.alloc_shared((merge_rows,), accum_dtype)
                    old_weights = T.alloc_shared((merge_rows,), accum_dtype)
                    new_weights = T.alloc_shared((merge_rows,), accum_dtype)
                    acc = T.alloc_shared((merge_rows, D), accum_dtype)
                    out = T.alloc_shared((merge_rows, D), dtype)
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(merge_rows, D):
                            acc[r, d] = 0
                        for r in T.Parallel(merge_rows):
                            maximum[r] = -1.0e30
                            total[r] = 0
                    for split in T.serial(splits):
                        # merge_rows divides BR, so no chunk crosses a partial Q tile.
                        T.copy(Parts[group_task, split, row:row + merge_rows, :], part)
                        offset = (group_task * splits + split) * BR + row
                        T.copy(LSE[offset:offset + merge_rows], score)
                        with T.SimtVF(threads=128):
                            # Compute transcendental expressions per row before their
                            # broadcast across D; keep old maximum unchanged until used.
                            for r in T.Parallel(merge_rows):
                                peak_rows[r] = T.max(maximum[r], score[r])
                                old_weights[r] = T.exp(maximum[r] - peak_rows[r])
                                new_weights[r] = T.if_then_else(
                                    score[r] > -1.0e29, T.exp(score[r] - peak_rows[r]), 0)
                            for r, d in T.Parallel(merge_rows, D):
                                acc[r, d] = acc[r, d] * old_weights[r] + part[r, d] * new_weights[r]
                            for r in T.Parallel(merge_rows):
                                total[r] = total[r] * old_weights[r] + new_weights[r]
                                maximum[r] = peak_rows[r]
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(merge_rows, D):
                            out[r, d] = T.cast(acc[r, d] / T.max(total[r], 1.0e-30), dtype)
                    # UB->GM uses the same tail-clamping convention as the reference.
                    T.copy(out, O[b, qi:qi + merge_rows, h, :])

    if splits == 1:
        @T.prim_func
        def direct(
            Q: T.Tensor((batch, heads, sq, D), dtype),
            K: T.Tensor((pool_rows, heads, D), dtype),
            V: T.Tensor((pool_rows, heads, D), dtype),
            Table: T.Tensor((batch, width), "int32"),
            Lengths: T.Tensor((batch,), "int32"),
            O: T.Tensor((batch, sq, heads * D), dtype),
        ):
            with T.Kernel(tasks * splits) as bx:
                task = bx // splits
                split = bx % splits
                b = task // (q_tiles * heads)
                qi = (task // heads % q_tiles) * BR
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
                # As in the reference PR, GM->L1 clamps a partial query tile
                # and zero-pads its missing rows. Only valid rows are merged out.
                T.copy(Q[b, h, qi:qi + BR, 0:D], Q_l1)
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
                T.annotate_buffer_versions({K_l1: kv_versions, V_l1: kv_versions,
                                           qk: buffer_versions, pv: buffer_versions,
                                           scores: buffer_versions, probs: buffer_versions,
                                           current: buffer_versions, alpha: buffer_versions})
                if q_resident:
                    T.copy(Q_l1, qa)
                for tile in T.Pipelined(count, num_stages=pipeline_stages):
                    logical = (begin + tile) * BC
                    physical = Table[b, logical // page_size] * page_size + logical % page_size
                    T.copy(K[physical:physical + BC, h, :], K_l1)
                    T.copy(V[physical:physical + BC, h, :], V_l1, transpose=True)
                    if not q_resident:
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
                # Reuse FP32 output storage for BF16 conversion, as in upstream.
                # Sequential SIMD rows ensure a cast store cannot overwrite a future
                # FP32 source row before it is read.
                cast_view = T.view(output, (ROWS * 2, D), dtype=dtype)
                with T.SimdVF(latency=193):
                    mask = S.pset(32, "PAT_ALL")
                    for r in range(ROWS):
                        for c in range(0, D, VL):
                            value = S.vcvt(S.vld(output[r, c], "NORM"), dtype, mask)
                            S.vsts(cast_view[r, c], value, mask, "PK_B32")
                # Keep Sq as its own bounded axis for Q-tail clamping, and expose
                # query/D as the trailing two region axes for dual-copy inference.
                T.dual_copy(cast_view[0:ROWS, 0:D], O[b, qi:qi + BR, h * D:(h + 1) * D])

        direct_kernel = tilelang.compile(direct, out_idx=-1, target="ascend", execution_backend="auto")

        def launch_direct(q, k, v, table, lengths):
            prepared = q.permute(0, 2, 1, 3).contiguous()
            return direct_kernel(prepared, k, v, table, lengths).view(batch, sq, heads, D)

        launch_direct.execution_mode = "direct_single_split"
        launch_direct.direct_kernel = direct_kernel
        launch_direct.prepare_q = lambda q: q.permute(0, 2, 1, 3).contiguous()
        return launch_direct

    partial_kernel = tilelang.compile(partial, out_idx=[5, 6], target="ascend", execution_backend="auto")
    merge_kernel = tilelang.compile(merge, out_idx=-1, target="ascend", execution_backend="auto")

    def launch(q, k, v, table, lengths):
        # Put each head's query rows on the last two axes for the same
        # Q-tail DMA geometry as the reference. Include this conversion in timing.
        q_head_major = q.permute(0, 2, 1, 3).contiguous()
        parts, lse = partial_kernel(q_head_major, k, v, table, lengths)
        return merge_kernel(parts, lse)
    # Expose the exact compiled callables for isolated profiling; launch unchanged.
    launch.execution_mode = "split_and_merge"
    launch.partial_kernel = partial_kernel
    launch.merge_kernel = merge_kernel
    launch.prepare_q = lambda q: q.permute(0, 2, 1, 3).contiguous()
    return launch

def mha_paged_decode(q, k_cache, v_cache, block_table, cache_seqlens,
                     page_size, is_causal=False, *, pipeline_stages=2, buffer_versions=2,
                     q_resident=True, kv_versions=None):
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
                              "float16" if q.dtype == torch.float16 else "bfloat16", q.device.index, pipeline_stages=pipeline_stages, buffer_versions=buffer_versions,
                              q_resident=q_resident, kv_versions=kv_versions)
        return kernel(*inputs)
