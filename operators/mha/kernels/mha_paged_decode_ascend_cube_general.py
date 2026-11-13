"""General Cube QK/PV candidate for the v7 tensor contract.

Precision-first implementation: FP16/BF16, D=16..256 (step 16), page sizes
divisible by 16, arbitrary runtime lengths, Q tails and right-aligned causal
masking. Shared QK/PV body for direct and split-KV modes. NPU validation required.
"""
from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout
from operators.mha.general_split_policy import select_splits


@lru_cache(maxsize=128)
def build_kernel(batch, sq, heads, dim, pool_rows, width, page_size, causal,
                 dtype, device_index=None, num_splits=None, split_policy="legacy", cache_lengths=None, model_version=1):
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("General Cube requires FP16 or BF16")
    if dim < 16 or dim > 256 or dim % 16:
        raise ValueError("General Cube requires D in [16, 256], divisible by 16")
    if min(batch, sq, heads, pool_rows, width, page_size) <= 0:
        raise ValueError("Positive tensor extents required")
    if page_size % 16 or pool_rows % page_size:
        raise ValueError("General Cube requires whole pages of a multiple of 16 tokens")
    # ND->NZ scatter requires >=16 rows per AIV, including the K tile.
    # Splitting BC=16 between two AIVs would produce an unsupported 8-row copy.
    BR, BC, ROWS = 32, 32, 16
    # Keep the transposed V half-tile at least 16 rows wide. Zero feature
    # padding preserves QK and PV; attention scaling uses the original D.
    D = max(32, tilelang.next_power_of_2(dim))
    scale = dim ** -0.5
    q_tiles = (sq + BR - 1) // BR
    work = batch * heads * q_tiles

    capacity_tiles = (width * page_size + BC - 1) // BC
    if split_policy not in ("auto", "legacy", "cost_model"):
        raise ValueError("General split_policy must be auto, legacy or cost_model")
    if num_splits is None and split_policy == "cost_model":
        import torch
        props = torch.npu.get_device_properties(device_index if device_index is not None else torch.npu.current_device())
        if getattr(props, "cube_core_num", None) != 28:
            raise ValueError("General split model is calibrated for28 Cube cores")
        from operators.mha.general_split_cost_model import select_model_splits
        splits = select_model_splits(batch, sq, heads, dim, page_size, width, cache_lengths, version=model_version)
    else:
        splits = select_splits(work, capacity_tiles, num_splits)

    @T.macro
    def compute_tile(Q, K, V, Table, Lengths, b, qi, h, sid, split, output, maximum, total):
        q_l1 = T.alloc_l1((BR, D), dtype)
        k_l1 = T.alloc_l1((BC, D), dtype)
        v_l1 = T.alloc_l1((D, BC), dtype)
        p_l1 = T.alloc_l1((BR, BC), dtype)
        qa = T.alloc_l0a((BR, D), dtype)
        kb = T.alloc_l0b((BC, D), dtype)
        pa = T.alloc_l0a((BR, BC), dtype)
        vb = T.alloc_l0b((D, BC), dtype)
        qk = T.alloc_l0c((BR, BC), "float32")
        pv = T.alloc_l0c((BR, D), "float32")
        q_ub = T.alloc_shared((ROWS, D), dtype)
        k_ub = T.alloc_shared((BC // 2, D), dtype)
        v_ub = T.alloc_shared((D // 2, BC), dtype)
        p_ub = T.alloc_shared((ROWS, BC), dtype)
        q_nz = T.alloc_shared((ROWS + 1, D), dtype)
        k_nz = T.alloc_shared((BC // 2 + 1, D), dtype)
        v_nz = T.alloc_shared((D // 2 + 1, BC), dtype)
        p_nz = T.alloc_shared((ROWS + 1, BC), dtype)
        T.annotate_layout({q_nz: make_ascend_compact_nz_layout(q_nz),
                           k_nz: make_ascend_compact_nz_layout(k_nz),
                           v_nz: make_ascend_compact_nz_layout(v_nz),
                           p_nz: make_ascend_compact_nz_layout(p_nz)})
        scores = T.alloc_shared((ROWS, BC), "float32")
        current = T.alloc_shared((ROWS, D), "float32")
        alpha = T.alloc_shared((ROWS,), "float32")
        with T.SimtVF(threads=128):
            for r, d in T.Parallel(ROWS, D):
                q_ub[r, d] = 0
                if qi + sid * ROWS + r < sq and d < dim:
                    q_ub[r, d] = Q[b, qi + sid * ROWS + r, h, d]
                output[r, d] = 0
            for r in T.Parallel(ROWS):
                maximum[r] = -1.0e30
                total[r] = 0
        T.copy(q_ub, q_nz[:ROWS, :])
        T.dual_copy(q_nz[:ROWS, :], q_l1)
        T.copy(q_l1, qa)
        live_tiles = T.ceildiv(Lengths[b], BC)
        begin = live_tiles * split // splits
        stop = live_tiles * (split + 1) // splits
        for block in T.serial(stop - begin):
            row0 = (begin + block) * BC
            # BC can cross a page boundary (e.g. page_size=16 or 48).
            # Resolve each logical token through the page table, inside
            # the validity guard, before reading either K or V.
            # Unread cache rows can be NaN, and 0*NaN would corrupt PV.
            with T.SimtVF(threads=128):
                for j, d in T.Parallel(BC // 2, D):
                    k_ub[j, d] = 0
                    kpos = row0 + sid * (BC // 2) + j
                    if kpos < Lengths[b] and d < dim:
                        kphysical = Table[b, kpos // page_size] * page_size + kpos % page_size
                        k_ub[j, d] = K[kphysical, h, d]
                for d, j in T.Parallel(D // 2, BC):
                    v_ub[d, j] = 0
                    vpos = row0 + j
                    if vpos < Lengths[b] and sid * (D // 2) + d < dim:
                        vphysical = Table[b, vpos // page_size] * page_size + vpos % page_size
                        v_ub[d, j] = V[vphysical, h, sid * (D // 2) + d]
            T.copy(k_ub, k_nz[:BC // 2, :])
            T.copy(v_ub, v_nz[:D // 2, :])
            T.dual_copy(k_nz[:BC // 2, :], k_l1)
            T.dual_copy(v_nz[:D // 2, :], v_l1)
            T.copy(k_l1, kb)
            T.gemm(qa, kb, qk, transpose_B=True, clear_accum=True)
            T.dual_copy(qk, scores)
            with T.SimtVF(threads=128):
                peaks = T.alloc_reducer((ROWS,), "float32", op="max")
                T.reducer_init(peaks)
                for r, j in T.Parallel(ROWS, BC):
                    visible = Lengths[b]
                    if causal:
                        visible = T.max(Lengths[b] - sq + qi + sid * ROWS + r + 1, 0)
                    scores[r, j] = T.if_then_else(
                        qi + sid * ROWS + r < sq and row0 + j < visible,
                        scores[r, j] * scale, -1.0e30)
                    T.reducer_update(peaks[r], scores[r, j])
                tile_max = T.alloc_fragment((ROWS,), "float32")
                T.finalize_reducer(peaks, tile_max)
                for r in T.Parallel(ROWS):
                    alpha[r] = T.exp(maximum[r] - T.max(maximum[r], tile_max[r]))
                    maximum[r] = T.max(maximum[r], tile_max[r])
                sums = T.alloc_reducer((ROWS,), "float32", op="sum")
                T.reducer_init(sums)
                probabilities = T.alloc_fragment((ROWS, BC), "float32")
                for r, j in T.Parallel(ROWS, BC):
                    visible = Lengths[b]
                    if causal:
                        visible = T.max(Lengths[b] - sq + qi + sid * ROWS + r + 1, 0)
                    probabilities[r, j] = T.if_then_else(
                        qi + sid * ROWS + r < sq and row0 + j < visible,
                        T.exp(scores[r, j] - maximum[r]), 0)
                    T.reducer_update(sums[r], probabilities[r, j])
                    p_ub[r, j] = T.cast(probabilities[r, j], dtype)
                tile_sum = T.alloc_fragment((ROWS,), "float32")
                T.finalize_reducer(sums, tile_sum)
                for r in T.Parallel(ROWS):
                    total[r] = total[r] * alpha[r] + tile_sum[r]
            T.copy(p_ub, p_nz[:ROWS, :])
            T.dual_copy(p_nz[:ROWS, :], p_l1)
            T.copy(p_l1, pa)
            T.copy(v_l1, vb)
            T.gemm(pa, vb, pv, transpose_B=True, clear_accum=True)
            T.dual_copy(pv, current)
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(ROWS, D):
                    output[r, d] = output[r, d] * alpha[r] + current[r, d]

    # One shared computation above; only the output epilogue differs.
    if splits == 1:
        @T.prim_func
        def direct(
            Q: T.Tensor((batch, sq, heads, dim), dtype),
            K: T.Tensor((pool_rows, heads, dim), dtype),
            V: T.Tensor((pool_rows, heads, dim), dtype),
            Table: T.Tensor((batch, width), "int32"),
            Lengths: T.Tensor((batch,), "int32"),
            O: T.Tensor((batch, sq, heads * dim), dtype),
        ):
            with T.MixedKernel(work, sids=2) as (bx, sid):
                b = bx // (heads * q_tiles)
                h = bx // q_tiles % heads
                qi = bx % q_tiles * BR
                output = T.alloc_shared((ROWS, D), "float32")
                maximum = T.alloc_shared((ROWS,), "float32")
                total = T.alloc_shared((ROWS,), "float32")
                out = T.alloc_shared((ROWS, D), dtype)
                compute_tile(Q, K, V, Table, Lengths, b, qi, h, sid, 0, output, maximum, total)
                with T.SimtVF(threads=128):
                    for r, d in T.Parallel(ROWS, D):
                        out[r, d] = T.cast(output[r, d] / T.max(total[r], 1.0e-30), dtype)
                T.dual_copy(out[:, :dim], O[b, qi:qi + BR, h * dim:(h + 1) * dim])

        direct_kernel = tilelang.compile(direct, out_idx=-1, target="ascend", execution_backend="auto")

        def launch(q, k, v, table, lengths):
            return direct_kernel(q, k, v, table, lengths).view(batch, sq, heads, dim)

        launch.execution_mode = "direct_single_split"
        launch.direct_kernel = direct_kernel
    else:
        @T.prim_func
        def partial(
            Q: T.Tensor((batch, sq, heads, dim), dtype),
            K: T.Tensor((pool_rows, heads, dim), dtype),
            V: T.Tensor((pool_rows, heads, dim), dtype),
            Table: T.Tensor((batch, width), "int32"),
            Lengths: T.Tensor((batch,), "int32"),
            Parts: T.Tensor((work, splits, BR, D), "float32"),
            LSE: T.Tensor((work * splits * BR,), "float32"),
        ):
            with T.MixedKernel(work * splits, sids=2) as (bx, sid):
                task = bx // splits
                split = bx % splits
                b = task // (heads * q_tiles)
                h = task // q_tiles % heads
                qi = task % q_tiles * BR
                output = T.alloc_shared((ROWS, D), "float32")
                maximum = T.alloc_shared((ROWS,), "float32")
                total = T.alloc_shared((ROWS,), "float32")
                lse = T.alloc_shared((ROWS,), "float32")
                compute_tile(Q, K, V, Table, Lengths, b, qi, h, sid, split, output, maximum, total)
                with T.SimtVF(threads=128):
                    for r, d in T.Parallel(ROWS, D):
                        output[r, d] = output[r, d] / T.max(total[r], 1.0e-30)
                    for r in T.Parallel(ROWS):
                        lse[r] = -1.0e30
                        if total[r] > 0:
                            lse[r] = maximum[r] + T.log2(T.max(total[r], 1.0e-30)) * 0.6931471805599453
                T.dual_copy(output, Parts[task, split, :, :])
                offset = (task * splits + split) * BR
                T.dual_copy(lse, LSE[offset:offset + BR])

        merge_rows = min(8, tilelang.next_power_of_2(sq))
        merge_groups = (sq + merge_rows - 1) // merge_rows
        merge_work = batch * heads * merge_groups
        merge_grid = min(merge_work, 32768)

        @T.prim_func
        def merge(
            Parts: T.Tensor((work, splits, BR, D), "float32"),
            LSE: T.Tensor((work * splits * BR,), "float32"),
            O: T.Tensor((batch, sq, heads, dim), dtype),
        ):
            with T.Kernel(merge_grid) as core:
                part = T.alloc_shared((merge_rows, D), "float32")
                score = T.alloc_shared((merge_rows,), "float32")
                maximum = T.alloc_shared((merge_rows,), "float32")
                total = T.alloc_shared((merge_rows,), "float32")
                peak = T.alloc_shared((merge_rows,), "float32")
                old_weight = T.alloc_shared((merge_rows,), "float32")
                new_weight = T.alloc_shared((merge_rows,), "float32")
                acc = T.alloc_shared((merge_rows, D), "float32")
                out = T.alloc_shared((merge_rows, dim), dtype)
                for item in T.serial(T.ceildiv(merge_work, merge_grid)):
                    program = core + item * merge_grid
                    if program < merge_work:
                        b = program // (heads * merge_groups)
                        h = program // merge_groups % heads
                        qi = program % merge_groups * merge_rows
                        task = (b * heads + h) * q_tiles + qi // BR
                        row = qi % BR
                        with T.SimtVF(threads=128):
                            for r, d in T.Parallel(merge_rows, D):
                                acc[r, d] = 0
                            for r in T.Parallel(merge_rows):
                                maximum[r] = -1.0e30
                                total[r] = 0
                        for split in T.serial(splits):
                            T.copy(Parts[task, split, row:row + merge_rows, :], part)
                            offset = (task * splits + split) * BR + row
                            T.copy(LSE[offset:offset + merge_rows], score)
                            with T.SimtVF(threads=128):
                                for r in T.Parallel(merge_rows):
                                    peak[r] = T.max(maximum[r], score[r])
                                    old_weight[r] = T.exp(maximum[r] - peak[r])
                                    # Empty partitions have LSE=-1e30 and zero
                                    # weight, including when every split is empty.
                                    new_weight[r] = T.if_then_else(
                                        score[r] > -1.0e29, T.exp(score[r] - peak[r]), 0)
                                for r, d in T.Parallel(merge_rows, D):
                                    acc[r, d] = acc[r, d] * old_weight[r] + part[r, d] * new_weight[r]
                                for r in T.Parallel(merge_rows):
                                    total[r] = total[r] * old_weight[r] + new_weight[r]
                                    maximum[r] = peak[r]
                        with T.SimtVF(threads=128):
                            for r, d in T.Parallel(merge_rows, dim):
                                out[r, d] = T.cast(acc[r, d] / T.max(total[r], 1.0e-30), dtype)
                        T.copy(out, O[b, qi:qi + merge_rows, h, :])

        partial_kernel = tilelang.compile(partial, out_idx=[5, 6], target="ascend", execution_backend="auto")
        merge_kernel = tilelang.compile(merge, out_idx=-1, target="ascend", execution_backend="auto")

        def launch(q, k, v, table, lengths):
            # Per-call FP32 temporaries, published to merge by current-stream
            # ordering. No persistent shared scratch or cross-call counters.
            parts, lse = partial_kernel(q, k, v, table, lengths)
            return merge_kernel(parts, lse)

        launch.execution_mode = "split_and_merge"
        launch.partial_kernel = partial_kernel
        launch.merge_kernel = merge_kernel

    launch.execution_path = "cube_general"
    launch.num_splits = splits
    launch.requested_num_splits = num_splits
    launch.split_policy = ("explicit" if num_splits is not None else
                           "cost_model" if split_policy == "cost_model" else "fill_28")
    launch.requested_split_policy = split_policy
    launch.general_model_version = model_version if launch.split_policy == "cost_model" else None
    launch.compute_grid = work * splits
    launch.compute_work = work * splits
    launch.persistent_programs = None
    launch.query_tile_tasks = work
    return launch
