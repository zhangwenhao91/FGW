"""First TileLang Ascend DSA kernel: B1, H128, Hkv1, BF16, D512+64."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


@lru_cache(maxsize=8)
def build_kernel(
    sq,
    sk,
    topk,
    *,
    prepacked=False,
    pv_width=128,
    reuse_kv=False,
    dma_gather=False,
    pipeline=False,
    head_tile=32,
    kv_tile=32,
    softmax_nz=False,
    staged=False,
    selective_zero=False,
    simd_softmax=False,
    simd_pv=False,
    batch_pv=False,
):
    """Run on unique indices, stride one, optionally with KV packed in index order."""
    if min(sq, sk, topk) < 1 or sq > sk:
        raise ValueError("expected positive sizes and right-aligned causal queries")
    if batch_pv and not simd_pv:
        raise ValueError("batched PV handoff requires SIMD PV")
    if simd_pv and not simd_softmax:
        raise ValueError("SIMD PV requires the SIMD softmax baseline")
    if simd_softmax and not (staged and kv_tile == 32 and not selective_zero):
        raise ValueError("SIMD softmax requires staged NZ, KV32, and full zero fill")
    if selective_zero and not dma_gather:
        raise ValueError("selective zero requires DMA gather")
    if staged and not (pipeline and softmax_nz):
        raise ValueError("staged attention requires DMA pipeline and direct NZ softmax")
    if pipeline and not dma_gather:
        raise ValueError("pipeline requires DMA gather")
    if dma_gather and (not reuse_kv or prepacked):
        raise ValueError("DMA gather requires indexed L1 reuse")
    heads, width, dim = 128, 576, 512
    if pv_width not in (128, 256, 512):
        raise ValueError("pv_width must be 128, 256 or 512")
    if head_tile not in (32, 64) or (head_tile == 64 and not pipeline):
        raise ValueError("64-head tiles require the DMA prefetch path")
    if pv_width == 512 and not (head_tile == 64 and kv_tile == 32 and batch_pv):
        raise ValueError("512-column PV requires the batched 64-head KV32 path")
    if kv_tile not in (32, 64) or (kv_tile == 64 and head_tile != 64):
        raise ValueError("64-row KV tiles require 64-head tiles")
    br, bc, rows, part = head_tile, kv_tile, head_tile // 2, pv_width
    gather_rows = 16
    qk_width = width if br == 32 else 192
    dtype, fp32 = "bfloat16", "float32"
    scale = width**-0.5
    kv_rows = sq * topk if prepacked else sk

    @T.macro
    def gather_block(block, qi, sid, KV, Indices, valid, k, kn, k_l1):
        with T.SimtVF(threads=128):
            for j in T.Parallel(bc):
                valid[j] = -1
                if block * bc + j < topk:
                    index = Indices[qi, block * bc + j]
                    if index >= 0 and index <= sk - sq + qi:
                        valid[j] = index
        if dma_gather:
            # Reuse a 16-row UB staging area for each AIV half-tile.
            for segment in T.serial(bc // (2 * gather_rows)):
                with T.SimtVF(threads=128):
                    for j, d in T.Parallel(gather_rows, width):
                        if (
                            not selective_zero
                            or valid[
                                segment * (2 * gather_rows) + sid * gather_rows + j
                            ]
                            < 0
                        ):
                            k[j, d] = 0
                # Fixed pairs give DMA static extents. Disjoint or masked
                # rows fall back to one DMA per valid row.
                for pair in T.serial(gather_rows // 2):
                    j = pair * 2
                    first = valid[segment * (2 * gather_rows) + sid * gather_rows + j]
                    second = valid[
                        segment * (2 * gather_rows) + sid * gather_rows + j + 1
                    ]
                    if first >= 0 and second == first + 1:
                        T.copy(KV[first : first + 2, :], k[j : j + 2, :])
                    else:
                        if first >= 0:
                            T.copy(KV[first, :], k[j, :])
                        if second >= 0:
                            T.copy(KV[second, :], k[j + 1, :])
                T.copy(k, kn[:gather_rows, :])
                first_row = segment * (2 * gather_rows)
                T.dual_copy(
                    kn[:gather_rows, :],
                    k_l1[first_row : first_row + 2 * gather_rows, :],
                )
        else:
            with T.SimtVF(threads=128):
                for j, d in T.Parallel(bc // 2, width):
                    index = valid[sid * (bc // 2) + j]
                    k[j, d] = 0
                    if index >= 0:
                        source = (
                            qi * topk + block * bc + sid * (bc // 2) + j
                            if prepacked
                            else index
                        )
                        k[j, d] = KV[source, d]
            T.copy(k, kn[:gather_rows, :])
            T.dual_copy(kn[:gather_rows, :], k_l1)

    @T.macro
    def compute_qk(k_l1, q_l1, kb, qa, qk, scores):
        if br == 32:
            T.copy(k_l1, kb)
            T.gemm(qa, kb, qk, transpose_B=True, clear_accum=True)
        else:
            # Full-width Q exceeds L0A with 64 heads. Accumulate three K slices
            # in FP32 before applying scale and softmax.
            for section in T.serial(width // qk_width):
                lo = section * qk_width
                T.copy(q_l1[:, lo : lo + qk_width], qa)
                T.copy(k_l1[:, lo : lo + qk_width], kb)
                T.gemm(qa, kb, qk, transpose_B=True, clear_accum=(section == 0))
        T.dual_copy(qk, scores[:rows, :])

    @T.macro
    def compute_softmax(valid, scores, maximum, alpha, total, p, pn, p_l1):
        if simd_softmax:
            with T.SimdVF():
                active = S.pset(32, "PAT_VL32")
                one = S.pset(32, "PAT_VL1")
                half = S.pset(32, "PAT_VL16")
                full = S.pset(32, "PAT_ALL")
                live = S.vcmps(S.vld(valid[0]), T.int32(0), active, "ge")
                neg_inf = S.vdup(-T.infinity(fp32), fp32, full)
                rotate = S.vand(
                    T.reinterpret(S.vci(16, "int32"), "uint32x64"),
                    S.vdup(T.uint32(63), "uint32", full),
                    full,
                )
                for r in T.serial(rows):
                    scaled = S.vmuls(S.vld(scores[r, 0]), T.float32(scale), active)
                    masked = S.vsel(scaled, neg_inf, live)
                    previous = S.vld(maximum[r], "BRC_B32")
                    block_max = S.vdupv(S.vcmax(masked, active), active)
                    peak = S.vmax(previous, block_max, active)
                    weight = S.vexpdif(previous, peak, active)
                    prob = S.vexpdif(masked, peak, active)
                    previous_sum = S.vld(total[r], "BRC_B32")
                    updated = S.vadd(
                        S.vmul(previous_sum, weight, active),
                        S.vcadd(prob, active),
                        active,
                    )
                    S.vsts(maximum[r], peak, one, "ONEPT_B32")
                    S.vsts(total[r], updated, one, "ONEPT_B32")
                    S.vsts(alpha[r], weight, one, "ONEPT_B32")
                    low = S.vcvt(
                        prob, dtype, active, round="ROUND_R", sat=False, part=0
                    )
                    high = S.vcvt(
                        S.vselr(prob, rotate),
                        dtype,
                        half,
                        round="ROUND_R",
                        sat=False,
                        part=0,
                    )
                    S.vsts(pn[r, 0], low, half, "PK_B32", extent=16)
                    S.vsts(pn[r, 16], high, half, "PK_B32", extent=16)
        else:
            with T.SimtVF(threads=128):
                peaks = T.alloc_reducer((rows,), fp32, op="max")
                T.reducer_init(peaks)
                for r, j in T.Parallel(rows, bc):
                    scores[r, j] = T.if_then_else(
                        valid[j] >= 0, scores[r, j] * scale, -1.0e30
                    )
                    T.reducer_update(peaks[r], scores[r, j])
                tile_max = T.alloc_fragment((rows,), fp32)
                T.finalize_reducer(peaks, tile_max)
                for r in T.Parallel(rows):
                    peak = T.max(maximum[r], tile_max[r])
                    alpha[r] = T.exp(maximum[r] - peak)
                    maximum[r] = peak
                sums = T.alloc_reducer((rows,), fp32, op="sum")
                T.reducer_init(sums)
                for r, j in T.Parallel(rows, bc):
                    prob = T.if_then_else(
                        valid[j] >= 0, T.exp(scores[r, j] - maximum[r]), 0
                    )
                    T.reducer_update(sums[r], prob)
                    if softmax_nz:
                        pn[r, j] = T.cast(prob, dtype)
                    else:
                        p[r, j] = T.cast(prob, dtype)
                tile_sum = T.alloc_fragment((rows,), fp32)
                T.finalize_reducer(sums, tile_sum)
                for r in T.Parallel(rows):
                    total[r] = total[r] * alpha[r] + tile_sum[r]
        if not softmax_nz:
            T.copy(p, pn[:rows, :])
        T.dual_copy(pn[:rows, :], p_l1)

    @T.macro
    def compute_pv(
        block,
        qi,
        sid,
        KV,
        valid,
        k_l1,
        p_l1,
        pa,
        v,
        vn,
        v_l1,
        vb,
        pv,
        current,
        output,
        alpha,
    ):
        T.copy(p_l1, pa)
        for chunk in T.serial(dim // part):
            if reuse_kv:
                # The latent prefix already in L1 is also the value tile.
                T.copy(
                    k_l1[:, chunk * part : (chunk + 1) * part],
                    vb,
                    transpose=False,
                )
                T.gemm(pa, vb, pv, transpose_B=False, clear_accum=True)
            else:
                with T.SimtVF(threads=128):
                    for d, j in T.Parallel(part // 2, bc):
                        index = valid[j]
                        v[d, j] = 0
                        if index >= 0:
                            source = qi * topk + block * bc + j if prepacked else index
                            v[d, j] = KV[source, chunk * part + sid * (part // 2) + d]
                T.copy(v, vn[: part // 2, :])
                T.dual_copy(vn[: part // 2, :], v_l1)
                T.copy(v_l1, vb)
                T.gemm(pa, vb, pv, transpose_B=True, clear_accum=True)
            if batch_pv:
                T.dual_copy(pv, current[:, chunk * part : (chunk + 1) * part])
            else:
                T.dual_copy(pv, current)
                if simd_pv:
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        for r in T.serial(rows):
                            factor = S.vld(alpha[r], "BRC_B32")
                            for c in T.serial(part // 64):
                                col = chunk * part + c * 64
                                previous = S.vld(output[r, col])
                                contribution = S.vld(current[r, c * 64])
                                updated = S.vadd(
                                    S.vmul(previous, factor, full), contribution, full
                                )
                                S.vsts(output[r, col], updated, full, "NORM_B32")
                else:
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(rows, part):
                            col = chunk * part + d
                            output[r, col] = output[r, col] * alpha[r] + current[r, d]
        if batch_pv:
            with T.SimdVF():
                mask = S.pset(32, "PAT_ALL")
                for r in T.serial(rows):
                    factor = S.vld(alpha[r], "BRC_B32")
                    for col in T.serial(dim // 64):
                        previous = S.vld(output[r, col * 64])
                        contribution = S.vld(current[r, col * 64])
                        accumulated = S.vadd(
                            S.vmul(previous, factor, mask), contribution, mask
                        )
                        S.vsts(output[r, col * 64], accumulated, mask, "NORM_B32")

    @T.macro
    def attend_block(
        block,
        qi,
        sid,
        KV,
        valid,
        k_l1,
        q_l1,
        kb,
        qa,
        qk,
        scores,
        maximum,
        alpha,
        total,
        p,
        pn,
        p_l1,
        pa,
        v,
        vn,
        v_l1,
        vb,
        pv,
        current,
        output,
    ):
        compute_qk(k_l1, q_l1, kb, qa, qk, scores)
        compute_softmax(valid, scores, maximum, alpha, total, p, pn, p_l1)
        compute_pv(
            block,
            qi,
            sid,
            KV,
            valid,
            k_l1,
            p_l1,
            pa,
            v,
            vn,
            v_l1,
            vb,
            pv,
            current,
            output,
            alpha,
        )

    @T.prim_func
    def forward(
        Q: T.Tensor((sq, heads, width), dtype),
        KV: T.Tensor((kv_rows, width), dtype),
        Indices: T.Tensor((sq, topk), "int32"),
        Out: T.Tensor((sq, heads, dim), dtype),
    ):
        with T.MixedKernel(sq * (heads // br), sids=2) as (task, sid):
            qi = task // (heads // br)
            hi = task % (heads // br) * br
            q_l1 = T.alloc_l1((br, width), dtype)
            k_l1 = T.alloc_l1((bc, width), dtype)
            v_l1 = T.alloc_l1((part, bc), dtype)
            p_l1 = T.alloc_l1((br, bc), dtype)
            qa = T.alloc_l0a((br, qk_width), dtype)
            kb = T.alloc_l0b((bc, qk_width), dtype)
            pa = T.alloc_l0a((br, bc), dtype)
            vb = T.alloc_l0b((bc, part) if reuse_kv else (part, bc), dtype)
            qk = T.alloc_l0c((br, bc), fp32)
            pv = T.alloc_l0c((br, part), fp32)
            k = T.alloc_shared((gather_rows, width), dtype)
            v = T.alloc_shared((part // 2, bc), dtype)
            p = T.alloc_shared((rows, bc), dtype)
            kn = T.alloc_shared((gather_rows + 1, width), dtype)
            vn = T.alloc_shared((part // 2 + 1, bc), dtype)
            pn = T.alloc_shared((rows + 1, bc), dtype)
            T.annotate_layout(
                {
                    kn: make_ascend_compact_nz_layout(kn),
                    vn: make_ascend_compact_nz_layout(vn),
                    pn: make_ascend_compact_nz_layout(pn),
                }
            )
            scores = T.alloc_shared((rows + (1 if simd_softmax else 0), bc), fp32)
            current = T.alloc_shared((rows, dim if batch_pv else part), fp32)
            output = T.alloc_shared((rows, dim), fp32)
            maximum = T.alloc_shared((rows,), fp32)
            total = T.alloc_shared((rows,), fp32)
            alpha = T.alloc_shared((rows,), fp32)
            valid = T.alloc_shared((64 if simd_softmax else bc,), "int32")
            T.copy(Q[qi, hi : hi + br, :], q_l1)
            if br == 32:
                T.copy(q_l1, qa)
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(rows, dim):
                    output[r, d] = 0
                for r in T.Parallel(rows):
                    maximum[r] = -1.0e30
                    total[r] = 0
            if staged:
                for block in T.Pipelined(
                    T.ceildiv(topk, bc),
                    num_stages=3,
                    annotations={"enable_offset": True},
                ):
                    with T.Stage(0):
                        gather_block(block, qi, sid, KV, Indices, valid, k, kn, k_l1)
                    with T.Stage(0):
                        compute_qk(k_l1, q_l1, kb, qa, qk, scores)
                    with T.Stage(1):
                        compute_softmax(
                            valid, scores, maximum, alpha, total, p, pn, p_l1
                        )
                    with T.Stage(2):
                        compute_pv(
                            block,
                            qi,
                            sid,
                            KV,
                            valid,
                            k_l1,
                            p_l1,
                            pa,
                            v,
                            vn,
                            v_l1,
                            vb,
                            pv,
                            current,
                            output,
                            alpha,
                        )
            elif pipeline:
                T.annotate_buffer_versions({k: 2, kn: 2, k_l1: 2, valid: 2})
                for block in T.Pipelined(
                    T.ceildiv(topk, bc),
                    num_stages=2,
                    annotations={"enable_offset": True},
                ):
                    with T.Stage(0):
                        gather_block(block, qi, sid, KV, Indices, valid, k, kn, k_l1)
                    with T.Stage(1):
                        attend_block(
                            block,
                            qi,
                            sid,
                            KV,
                            valid,
                            k_l1,
                            q_l1,
                            kb,
                            qa,
                            qk,
                            scores,
                            maximum,
                            alpha,
                            total,
                            p,
                            pn,
                            p_l1,
                            pa,
                            v,
                            vn,
                            v_l1,
                            vb,
                            pv,
                            current,
                            output,
                        )
            else:
                for block in T.serial(T.ceildiv(topk, bc)):
                    gather_block(block, qi, sid, KV, Indices, valid, k, kn, k_l1)
                    attend_block(
                        block,
                        qi,
                        sid,
                        KV,
                        valid,
                        k_l1,
                        q_l1,
                        kb,
                        qa,
                        qk,
                        scores,
                        maximum,
                        alpha,
                        total,
                        p,
                        pn,
                        p_l1,
                        pa,
                        v,
                        vn,
                        v_l1,
                        vb,
                        pv,
                        current,
                        output,
                    )
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(rows, dim):
                    Out[qi, hi + sid * rows + r, d] = T.cast(
                        output[r, d] / T.max(total[r], 1e-30), dtype
                    )

    return tilelang.compile(
        forward, out_idx=[3], target="ascend", execution_backend="auto"
    )
