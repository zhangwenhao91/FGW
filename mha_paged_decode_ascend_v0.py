"""Review prototype: paged MHA decode in TileLang's Ascend SIMT dialect.

Targets tilelang 0.1.15+cpu.gita7cb753d. Not hardware-validated.
One program per (batch, query, head), one KV token per update. Deliberately
simple and slow: no Cube GEMM, split-KV, autotuning or TileOPs registration.
Cache contents, lengths and page table are read-only. No CPU fallback.
"""
from functools import lru_cache

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


@lru_cache(maxsize=32)
def build_kernel(batch, sq, heads, dim, pool_rows, width, page_size, causal, dtype):
    if dim < 16 or dim > 256 or dim % 16:
        raise ValueError("v0 requires D divisible by 16, in [16, 256]")
    if page_size <= 0 or pool_rows % page_size:
        raise ValueError("pool rows must be divisible by positive page_size")
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
            k_ub = T.alloc_shared((dim,), dtype)
            v_ub = T.alloc_shared((dim,), dtype)
            out_ub = T.alloc_shared((dim,), dtype)
            acc_ub = T.alloc_shared((dim,), "float32")
            # Shared state survives successive VF calls: running maximum and sum.
            state = T.alloc_shared((8,), "float32")
            T.copy(Q[b, qpos, h, :], q_ub)
            with T.SimtVF(threads=threads):
                for d in T.Parallel(dim):
                    acc_ub[d] = 0
                for i in T.Parallel(1):
                    state[0] = -1.0e30
                    state[1] = 0

            end = Lengths[b]
            if causal:
                end = T.max(end - sq + qpos + 1, 0)
            # No load at all for masked tokens: unused/NaN cache rows cannot
            # enter the accumulation, including fully masked queries.
            for token in T.serial(end):
                physical = Table[b, token // page_size] * page_size + token % page_size
                T.copy(K[physical, h, :], k_ub)
                T.copy(V[physical, h, :], v_ub)
                with T.SimtVF(threads=threads):
                    dot = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(dot)
                    for d in T.Parallel(padded):
                        if d < dim:
                            T.reducer_update(dot[0], T.cast(q_ub[d], "float32") * T.cast(k_ub[d], "float32"))
                    score = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(dot, score)
                    # Keep old state until every output dimension has consumed it.
                    for d in T.Parallel(dim):
                        s = score[0] * scale
                        new_max = T.max(state[0], s)
                        alpha = T.exp(state[0] - new_max)
                        probability = T.exp(s - new_max)
                        acc_ub[d] = acc_ub[d] * alpha + probability * T.cast(v_ub[d], "float32")
                    for i in T.Parallel(1):
                        s = score[0] * scale
                        new_max = T.max(state[0], s)
                        state[1] = state[1] * T.exp(state[0] - new_max) + T.exp(s - new_max)
                        state[0] = new_max

            with T.SimtVF(threads=threads):
                for d in T.Parallel(dim):
                    denominator = T.max(state[1], 1.0e-30)
                    out_ub[d] = T.cast(acc_ub[d] / denominator, dtype)
            T.copy(out_ub, O[b, qpos, h, :])

    return tilelang.compile(main, out_idx=-1, target="ascend", execution_backend="auto")


def mha_paged_decode(q, k_cache, v_cache, block_table, cache_seqlens,
                     page_size, is_causal=False):
    """Public order matches MHAPagedFwdOp; no cache updates.

    Preconditions: valid page IDs and 0 <= lengths <= table_width * page_size.
    Metadata values are not copied to the host to validate them on every call.
    """
    if q.ndim != 4 or k_cache.ndim != 3:
        raise ValueError("expected Q[B,Sq,H,D], K/V[N,H,D]")
    b, sq, h, d = q.shape
    if min(b, sq, h, d) <= 0 or block_table.ndim != 2:
        raise ValueError("positive Q extents and a rank-2 page table required")
    if k_cache.shape[1:] != (h, d) or v_cache.shape != k_cache.shape:
        raise ValueError("K/V shapes must agree with Q heads and dimension")
    if block_table.shape[0] != b or block_table.shape[1] <= 0 or cache_seqlens.shape != (b,):
        raise ValueError("metadata shapes do not match Q")
    if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("Q/K/V must have matching float16 or bfloat16 dtype")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise ValueError("metadata must be int32")
    inputs = (q, k_cache, v_cache, block_table, cache_seqlens)
    if q.device.type != "npu" or any(x.device != q.device or not x.is_contiguous() for x in inputs):
        raise ValueError("all inputs must be contiguous and on the same NPU")
    dtype = "float16" if q.dtype == torch.float16 else "bfloat16"
    # Compile under the same device context as the input.
    with torch.npu.device(q.device):
        kernel = build_kernel(b, sq, h, d, k_cache.shape[0], block_table.shape[1],
                              page_size, is_causal, dtype)
        return kernel(q, k_cache, v_cache, block_table, cache_seqlens)


def reference(q, k, v, table, lengths, page_size, causal):
    """CPU FP32 reference, reconstructing only each request's live cache."""
    q, k, v, table, lengths = [x.cpu() for x in (q, k, v, table, lengths)]
    b, sq, h, d = q.shape
    out = torch.zeros_like(q)
    for bi in range(b):
        length = int(lengths[bi])
        pos = torch.arange(length)
        physical = table[bi, pos // page_size].long() * page_size + pos % page_size
        kk, vv = k[physical].float(), v[physical].float()
        for qi in range(sq):
            end = max(length - sq + qi + 1, 0) if causal else length
            if end:
                scores = torch.einsum("hd,thd->ht", q[bi, qi].float(), kk[:end]) * d ** -0.5
                out[bi, qi] = torch.einsum("ht,thd->hd", scores.softmax(-1), vv[:end]).to(q.dtype)
    return out


if __name__ == "__main__":
    torch.manual_seed(7)
    device = "npu:0"  # Logical device after ASCEND_RT_VISIBLE_DEVICES filtering.
    # Reordered pages, shared pages, partly filled page, zero-length cache,
    # and causal query rows with no visible key. Unread pool rows contain NaN.
    b, sq, h, d, p, pages = 3, 4, 2, 64, 16, 8
    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(b, sq, h, d, dtype=dtype)
        k = torch.full((pages * p, h, d), float("nan"), dtype=dtype)
        v = k.clone()
        table = torch.tensor([[5, 1, 6], [1, 7, 2], [0, 3, 4]], dtype=torch.int32)
        lengths = torch.tensor([37, 2, 0], dtype=torch.int32)
        for bi in range(b):
            for token in range(int(lengths[bi])):
                row = int(table[bi, token // p]) * p + token % p
                k[row] = torch.randn(h, d, dtype=dtype)
                v[row] = torch.randn(h, d, dtype=dtype)
        inputs = [x.to(device) for x in (q, k, v, table, lengths)]
        for causal in (False, True):
            got = mha_paged_decode(*inputs, page_size=p, is_causal=causal).cpu()
            expected = reference(q, k, v, table, lengths, p, causal)
            torch.testing.assert_close(got, expected, rtol=1e-2 if dtype == torch.bfloat16 else 2e-3,
                                       atol=4e-3 if dtype == torch.bfloat16 else 2e-3)
            assert torch.isfinite(got).all()
            print(dtype, "causal=", causal, "PASS", "max_error=", (got.float() - expected.float()).abs().max().item())
