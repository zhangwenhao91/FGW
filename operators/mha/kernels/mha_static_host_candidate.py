"""Static host KV-block plan prototype: B1H1, BF16/D128, full page128 KV.

QK/PV and SIMD macros are copied from default v7. Only work assignment and
output destinations change. No device atomics or dynamic work queue.
SIMD macros adapted from tile-ai/tilelang at a7cb753dafc6b2514cbbd7c8431260c9d05e7df0;
see third_party/licenses/TILELANG_LICENSE.
"""

from functools import lru_cache
import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout
from operators.mha.static_schedule import make_plan, choose_plan


@lru_cache(maxsize=64)
def build_kernel(sq, kv, pool_rows, plan_policy="balanced"):
    if plan_policy not in ("balanced", "cost"):
        raise ValueError("plan_policy must be balanced or cost")
    plan = choose_plan(sq, kv) if plan_policy == "cost" else make_plan(sq, kv)
    if pool_rows <= 0 or pool_rows % 128 or pool_rows < kv:
        raise ValueError("whole physical pool pages, at least KV capacity required")
    BR, BC, ROWS, VL, D, STATS_ROWS = 128, 128, 64, 64, 128, 64
    dtype, accum_dtype = "bfloat16", "float32"
    scale = D**-0.5
    page_size = 128
    width = kv // 128
    grid = len(plan.programs)
    max_segments = plan.max_segments
    slots = max(1, plan.partial_count)
    merge_count = len(plan.merges)
    max_parts = plan.max_parts
    has_merge = merge_count > 0
    merge_rows = min(8, tilelang.next_power_of_2(sq))
    groups_per_tile = BR // merge_rows
    pipeline_annotations = {"enable_offset": True}

    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        # Online softmax: update the row maximum and denominator, pack P into
        # Cube NZ layout, and return the factor that rescales the previous PV.
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC), "bfloat16")
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
                    T.int32(ROWS + 1 << 16 | 1),
                    softmax_b16,
                    update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(ROWS + 1 << 16 | 1),
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
        # Preserve the original online-softmax accumulation order.
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

    @T.macro
    def local_body(Q, K, V, Table, Counts, Tasks, Parts, LSE, O):
        with T.Kernel(grid) as bx:
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
            maximum = T.alloc_shared((STATS_ROWS,), accum_dtype)
            total = T.alloc_shared((STATS_ROWS,), accum_dtype)
            alpha = T.alloc_shared((STATS_ROWS,), accum_dtype)
            old_max = T.alloc_shared((STATS_ROWS,), accum_dtype)
            row_sum = T.alloc_shared((STATS_ROWS,), accum_dtype)
            lse = T.alloc_shared((STATS_ROWS,), accum_dtype)
            T.annotate_buffer_versions(
                {K_l1: 3, V_l1: 3, qk: 2, pv: 2, scores: 3, probs: 2, current: 1, alpha: 3}
            )
            for work in T.serial(max_segments):
                if work < (Counts[bx] if has_merge else ((sq + BR - 1) // BR // grid + T.if_then_else(bx < ((sq + BR - 1) // BR) % grid, 1, 0))):
                    if has_merge:
                        task = Tasks[bx, work, 0]
                        begin = Tasks[bx, work, 1]
                        count = Tasks[bx, work, 2]
                        slot = Tasks[bx, work, 3]
                    else:
                        task = bx * (((sq + BR - 1) // BR) // grid) + T.min(bx, ((sq + BR - 1) // BR) % grid) + work
                        begin = 0
                        count = width
                        slot = -1
                    qi = task * BR
                    b, h = 0, 0
                    T.copy(Q[b, h, qi : qi + BR, :], Q_l1)
                    init_m_l(maximum, total)
                    clear_O_ub(output)
                    T.copy(Q_l1, qa)
                    for tile in T.Pipelined(count, num_stages=2, annotations=pipeline_annotations):
                        logical = (begin + tile) * BC
                        physical = Table[b, logical // page_size] * page_size + logical % page_size
                        with T.Stage(0):
                            T.copy(K[physical : physical + BC, h, :], K_l1)
                        with T.Stage(1):
                            T.copy(K_l1, kb)
                            T.gemm(qa, kb, qk, transpose_B=True, clear_accum=True)
                        with T.Stage(3):
                            T.dual_copy(qk, scores)
                        with T.Stage(5):
                            softmax(scores, probs, maximum, total, alpha, old_max, row_sum)
                        with T.Stage(6):
                            T.dual_copy(probs[0:ROWS, :], P_l1)
                        with T.Stage(3):
                            T.copy(V[physical : physical + BC, h, :], V_l1, transpose=True)
                        with T.Stage(6):
                            T.copy(P_l1, pa)
                            T.copy(V_l1, vb, transpose=False)
                            T.gemm(pa, vb, pv, transpose_B=True, clear_accum=True)
                        with T.Stage(7):
                            T.dual_copy(pv, current)
                            accumulate_output(current, output, alpha)
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(ROWS, D):
                            output[r, d] = output[r, d] / T.max(total[r], 1e-30)
                    if has_merge:
                        if slot >= 0:
                            with T.SimtVF(threads=128):
                                for r in T.Parallel(ROWS):
                                    lse[r] = T.log2(T.max(total[r], 1e-30)) * 0.6931471805599453 + maximum[r]
                            T.dual_copy(output, Parts[slot, :, :])
                            T.dual_copy(lse, LSE[slot * BR : (slot + 1) * BR])
                        else:
                            cast_view = T.view(output, (ROWS * 2, D), dtype=dtype)
                            with T.SimdVF(latency=193):
                                mask = S.pset(32, "PAT_ALL")
                                for r in range(ROWS):
                                    for c in range(0, D, VL):
                                        value = S.vcvt(S.vld(output[r, c], "NORM"), dtype, mask)
                                        S.vsts(cast_view[r, c], value, mask, "PK_B32")
                            T.dual_copy(cast_view[0:ROWS, 0:D], O[b, qi : qi + BR, h * D : (h + 1) * D])
                    else:
                        cast_view = T.view(output, (ROWS * 2, D), dtype=dtype)
                        with T.SimdVF(latency=193):
                            mask = S.pset(32, "PAT_ALL")
                            for r in range(ROWS):
                                for c in range(0, D, VL):
                                    value = S.vcvt(S.vld(output[r, c], "NORM"), dtype, mask)
                                    S.vsts(cast_view[r, c], value, mask, "PK_B32")
                        T.dual_copy(cast_view[0:ROWS, 0:D], O[b, qi : qi + BR, h * D : (h + 1) * D])

    # Both entry signatures inline the same Cube/Vector computation macro.
    # With no fragments, unused workspace and metadata never enter the ABI.
    if has_merge:
        @T.prim_func
        def local_split(
            Q: T.Tensor((1, 1, sq, D), dtype),
            K: T.Tensor((pool_rows, 1, D), dtype),
            V: T.Tensor((pool_rows, 1, D), dtype),
            Table: T.Tensor((1, width), "int32"),
            Counts: T.Tensor((grid,), "int32"),
            Tasks: T.Tensor((grid, max_segments, 4), "int32"),
            Parts: T.Tensor((slots, BR, D), accum_dtype),
            LSE: T.Tensor((slots * BR,), accum_dtype),
            O: T.Tensor((1, sq, D), dtype),
        ):
            local_body(Q, K, V, Table, Counts, Tasks, Parts, LSE, O)

        local_entry = local_split
    else:
        @T.prim_func
        def local_direct(
            Q: T.Tensor((1, 1, sq, D), dtype),
            K: T.Tensor((pool_rows, 1, D), dtype),
            V: T.Tensor((pool_rows, 1, D), dtype),
            Table: T.Tensor((1, width), "int32"),
            O: T.Tensor((1, sq, D), dtype),
        ):
            # Aliases are placeholders only: compile-time has_merge=False removes
            # every Counts/Tasks/Parts/LSE reference in the shared macro.
            local_body(Q, K, V, Table, Table, Table, O, O, O)

        local_entry = local_direct

    merge_grid = max(1, merge_count * groups_per_tile)

    @T.prim_func
    def merge(
        Parts: T.Tensor((slots, BR, D), accum_dtype),
        LSE: T.Tensor((slots * BR,), accum_dtype),
        Merges: T.Tensor((max(1, merge_count), 3), "int32"),
        O: T.Tensor((1, sq, D), dtype),
    ):
        with T.Kernel(merge_grid) as core:
            group = core // groups_per_tile
            row = core % groups_per_tile * merge_rows
            tile = Merges[group, 0]
            first = Merges[group, 1]
            pieces = Merges[group, 2]
            qi = tile * BR + row
            if qi < sq:
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
                        maximum[r] = -1e30
                        total[r] = 0
                for piece in T.serial(max_parts):
                    if piece < pieces:
                        slot = first + piece
                        T.copy(Parts[slot, row : row + merge_rows, :], part)
                        offset = slot * BR + row
                        T.copy(LSE[offset : offset + merge_rows], score)
                        with T.SimtVF(threads=128):
                            for r in T.Parallel(merge_rows):
                                peak_rows[r] = T.max(maximum[r], score[r])
                                old_weights[r] = T.exp(maximum[r] - peak_rows[r])
                                new_weights[r] = T.if_then_else(
                                    score[r] > -1e29, T.exp(score[r] - peak_rows[r]), 0
                                )
                            for r, d in T.Parallel(merge_rows, D):
                                acc[r, d] = acc[r, d] * old_weights[r] + part[r, d] * new_weights[r]
                            for r in T.Parallel(merge_rows):
                                total[r] = total[r] * old_weights[r] + new_weights[r]
                                maximum[r] = peak_rows[r]
                with T.SimtVF(threads=128):
                    for r, d in T.Parallel(merge_rows, D):
                        out[r, d] = T.cast(acc[r, d] / T.max(total[r], 1e-30), dtype)
                T.copy(out, O[0, qi : qi + merge_rows, :])

    local_kernel = tilelang.compile(
        local_entry,
        out_idx=[6, 7, 8] if has_merge else [4],
        target="ascend", execution_backend="auto", pass_configs={}
    )
    merge_kernel = (
        tilelang.compile(merge, out_idx=[], target="ascend", execution_backend="auto", pass_configs={})
        if has_merge
        else None
    )
    uploaded = {}

    def prepare_plan(device):
        if not has_merge:
            return None, None, None
        if device not in uploaded:
            counts = [len(p) for p in plan.programs]
            tasks = [list(p) + [(-1, -1, -1, -1)] * (max_segments - len(p)) for p in plan.programs]
            merges = list(plan.merges) or [(-1, -1, -1)]
            uploaded[device] = tuple(
                torch.tensor(x, dtype=torch.int32, device=device) for x in (counts, tasks, merges)
            )
        return uploaded[device]

    def validate_inputs(q, k, v, table, lengths):
        assert tuple(q.shape) == (1, sq, 1, D) and tuple(k.shape) == (pool_rows, 1, D) and v.shape == k.shape
        assert tuple(table.shape) == (1, width) and lengths.cpu().tolist() == [kv]
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert table.dtype == lengths.dtype == torch.int32
        assert q.device.type == "npu" and all(
            x.device == q.device and x.is_contiguous() for x in (q, k, v, table, lengths)
        )

    def run_local(prepared, k, v, table, counts=None, tasks=None):
        if has_merge:
            return local_kernel(prepared, k, v, table, counts, tasks)
        return None, None, local_kernel(prepared, k, v, table)

    def compute(q, k, v, table, lengths):
        counts, tasks, merges = prepare_plan(q.device)
        prepared = q.permute(0, 2, 1, 3).contiguous()
        return run_local(prepared, k, v, table, counts, tasks)

    def finish(parts, lse, output):
        if merge_kernel is not None:
            merges = prepare_plan(output.device)[2]
            merge_kernel(parts, lse, merges, output)
        return output.view(1, sq, 1, D)

    def launch(q, k, v, table, lengths):
        return finish(*compute(q, k, v, table, lengths))

    launch.plan = plan
    launch.prepare_plan = prepare_plan
    launch.validate_inputs = validate_inputs
    launch.run_local = run_local
    launch.local_kernel = local_kernel
    launch.merge_kernel = merge_kernel
    launch.compute = compute
    launch.finish = finish
    return launch
