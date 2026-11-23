"""First TileLang Ascend DSA kernel: B1, H128, Hkv1, BF16, D512+64."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout


@lru_cache(maxsize=8)
def build_kernel(sq, sk, topk, *, prepacked=False, pv_width=128, reuse_kv=False):
    """Run on unique indices, stride one, optionally with KV packed in index order."""
    if min(sq, sk, topk) < 1 or sq > sk:
        raise ValueError("expected positive sizes and right-aligned causal queries")
    heads, width, dim = 128, 576, 512
    if pv_width not in (128, 256):
        raise ValueError("pv_width must be 128 or 256")
    br, bc, rows, part = 32, 32, 16, pv_width
    dtype, fp32 = "bfloat16", "float32"
    scale = width**-0.5
    kv_rows = sq * topk if prepacked else sk

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
            qa = T.alloc_l0a((br, width), dtype)
            kb = T.alloc_l0b((bc, width), dtype)
            pa = T.alloc_l0a((br, bc), dtype)
            vb = T.alloc_l0b((bc, part) if reuse_kv else (part, bc), dtype)
            qk = T.alloc_l0c((br, bc), fp32)
            pv = T.alloc_l0c((br, part), fp32)
            k = T.alloc_shared((bc // 2, width), dtype)
            v = T.alloc_shared((part // 2, bc), dtype)
            p = T.alloc_shared((rows, bc), dtype)
            kn = T.alloc_shared((bc // 2 + 1, width), dtype)
            vn = T.alloc_shared((part // 2 + 1, bc), dtype)
            pn = T.alloc_shared((rows + 1, bc), dtype)
            T.annotate_layout(
                {
                    kn: make_ascend_compact_nz_layout(kn),
                    vn: make_ascend_compact_nz_layout(vn),
                    pn: make_ascend_compact_nz_layout(pn),
                }
            )
            scores = T.alloc_shared((rows, bc), fp32)
            current = T.alloc_shared((rows, part), fp32)
            output = T.alloc_shared((rows, dim), fp32)
            maximum = T.alloc_shared((rows,), fp32)
            total = T.alloc_shared((rows,), fp32)
            alpha = T.alloc_shared((rows,), fp32)
            valid = T.alloc_shared((bc,), "int32")
            T.copy(Q[qi, hi : hi + br, :], q_l1)
            T.copy(q_l1, qa)
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(rows, dim):
                    output[r, d] = 0
                for r in T.Parallel(rows):
                    maximum[r] = -1.0e30
                    total[r] = 0
            for block in T.serial(T.ceildiv(topk, bc)):
                with T.SimtVF(threads=128):
                    for j in T.Parallel(bc):
                        valid[j] = -1
                        if block * bc + j < topk:
                            index = Indices[qi, block * bc + j]
                            if index >= 0 and index <= sk - sq + qi:
                                valid[j] = index
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
                T.copy(k, kn[: bc // 2, :])
                T.dual_copy(kn[: bc // 2, :], k_l1)
                T.copy(k_l1, kb)
                T.gemm(qa, kb, qk, transpose_B=True, clear_accum=True)
                T.dual_copy(qk, scores)
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
                        p[r, j] = T.cast(prob, dtype)
                    tile_sum = T.alloc_fragment((rows,), fp32)
                    T.finalize_reducer(sums, tile_sum)
                    for r in T.Parallel(rows):
                        total[r] = total[r] * alpha[r] + tile_sum[r]
                T.copy(p, pn[:rows, :])
                T.dual_copy(pn[:rows, :], p_l1)
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
                                    source = (
                                        qi * topk + block * bc + j
                                        if prepacked
                                        else index
                                    )
                                    v[d, j] = KV[
                                        source, chunk * part + sid * (part // 2) + d
                                    ]
                        T.copy(v, vn[: part // 2, :])
                        T.dual_copy(vn[: part // 2, :], v_l1)
                        T.copy(v_l1, vb)
                        T.gemm(pa, vb, pv, transpose_B=True, clear_accum=True)
                    T.dual_copy(pv, current)
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(rows, part):
                            col = chunk * part + d
                            output[r, col] = output[r, col] * alpha[r] + current[r, d]
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(rows, dim):
                    Out[qi, hi + sid * rows + r, d] = T.cast(
                        output[r, d] / T.max(total[r], 1e-30), dtype
                    )

    return tilelang.compile(
        forward, out_idx=[3], target="ascend", execution_backend="auto"
    )
