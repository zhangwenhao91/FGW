"""Call pinned upstream Gather factories without importing its unrelated CUDA registry.

The vendored file is unmodified. Only its imports and three top-level factory
functions are selected; no kernel AST is rewritten. This tests direct low-level
kernel reuse, not installation of the complete TileOPs public Op package.
"""
import ast
import hashlib
from pathlib import Path

COMMIT = "2ffc69b29d1e456f28ade01d50fc3a6f715d1392"


def build_gather_attention(case, dtype_name, inputs, cube, splits, device_index):
    import torch
    path = Path(__file__).resolve().parents[2] / "third_party/tileops_yyttt6/paged_cache_gather.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_vector_cores", "_compile", "_compile_generic"}
    tree.body = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in names
        or isinstance(n, ast.Import) and all(a.name in ("functools", "math", "torch", "tilelang", "tilelang.ascend.language") for a in n.names)
    )]
    namespace = {"__name__": "fgw_pinned_external_gather"}
    exec(compile(tree, str(path), "exec"), namespace)
    name, b, sq, h, d, page, width, lengths, causal = case
    q, k, v, table, device_lengths = inputs
    # Round each live region to whole original pages, so the existing attention
    # kernel can consume the packed cache through an identity page table.
    # Only the last live page is padded, never the full cache capacity.
    pages_per_req = [(length + page - 1) // page for length in lengths]
    offsets = [0]
    for count in pages_per_req:
        offsets.append(offsets[-1] + count * page)
    rows = max(page, offsets[-1])
    cu = torch.tensor(offsets, dtype=torch.int32, device=q.device)
    dense_table = torch.tensor([[offsets[i] // page + j for j in range(width)] for i in range(b)],
                               dtype=torch.int32, device=q.device)
    fast = page in (16, 64) and h*d in (576, 1024) and b <= 256
    factory = namespace["_compile" if fast else "_compile_generic"]
    gather = factory(b, k.shape[0] // page, page, width, h*d, dtype_name, dtype_name,
                     False, namespace["_vector_cores"](device_index))
    attention = cube.build_kernel(b, sq, h, d, rows, width, page, causal, dtype_name,
                                  device_index, num_splits=splits)
    flat_k, flat_v = k.view(-1, h*d), v.view(-1, h*d)

    def gather_pair():
        dense_k = torch.empty((rows, h, d), dtype=k.dtype, device=k.device)
        dense_v = torch.empty_like(dense_k)
        if offsets[-1]:
            # Factory writes a caller-owned destination; no implicit output index.
            gather(dense_k[:offsets[-1]].view(-1, h*d), flat_k, table, cu, cu, flat_k)
            gather(dense_v[:offsets[-1]].view(-1, h*d), flat_v, table, cu, cu, flat_v)
        return dense_k, dense_v

    # Exact cache-row validation is separate from final attention tolerances.
    dense_k, dense_v = gather_pair()
    for original, packed in ((k, dense_k), (v, dense_v)):
        original_cpu, packed_cpu, table_cpu = original.cpu(), packed.cpu(), table.cpu()
        for i, count in enumerate(pages_per_req):
            if count:
                ids = table_cpu[i, :count].long()
                expected = original_cpu.view(-1, page, h, d)[ids].reshape(-1, h, d)
                torch.testing.assert_close(packed_cpu[offsets[i]:offsets[i+1]], expected,
                                           rtol=0, atol=0, equal_nan=True)

    def combined():
        dk, dv = gather_pair()
        return attention(q, dk, dv, dense_table, device_lengths)

    return combined, {
        "upstream_commit": COMMIT, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "factory": "specialized" if fast else "generic", "gather_accuracy": "exact_passed",
        "gather_calls_per_attention": 2 if offsets[-1] else 0,
        "packed_rows": offsets[-1], "live_rows": sum(lengths), "num_splits": splits,
        "scope": "Unmodified upstream low-level Gather + existing general Cube attention; not public Op integration",
        "timed": "K/V allocation, two Gather launches, all attention/split/merge kernels; metadata prepared outside timing",
    }
