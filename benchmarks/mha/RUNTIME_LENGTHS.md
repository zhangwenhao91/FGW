# Runtime Q/KV-shape experiment

## Dynamic DMA descriptor repair

The first NPU attempt failed during `AscendLowerTileOp`, before correctness or
timing: the compiler could not normalize the dynamic rank4 Q slice to one 2D MTE
descriptor (`all axes outside one row/inner pair must have extent 1`).
The B1H1 candidate now uses zero-copy rank2 views for Q, K/V, output and partial
payloads, matching the reference's Q/K/V rank. Q-tail clamping remains on the
bounded Q row axis; flattened partial storage has the same padded tile extent.
Dynamic merge receives rank2 Q as its exact-shape anchor. No Gather, extra
payload copies, synchronization, arithmetic or split-policy change is added.
The runtime factory explicitly requires B1H1 for this experiment. Static routes
retain their original tensor shapes and DMA regions. Rerun the same script;
the split1 and split4 gates both must pass before the full comparison.

Reference: [PR3896's runtime length example](https://github.com/tile-ai/tilelang/blob/49a51226eb4068d5c0bf205cee886d2a7dcf0b05/examples/ascend/flash_attention/example_mha.py).

Run `python fgw.py mha run run_mha_runtime_lengths`. Return
`reports/mha/latest/mha_runtime_lengths_paste.txt`; the current run is also retained
in a fresh mha_runtime_lengths.* directory with raw JSON, logs and generated C++.
This opt-in experiment does not replace the current static-shape public default.

Logical per-request KV length was already runtime device data (`cache_seqlens`).
This change introduces `T.dynamic` symbols for Q length, physical K/V pool rows
and page-table width; no independent logical-KV shape scalar is needed. It keeps
current contiguous Q assignment, manual two-stage pipeline, cap28, tile128,
BF16/D128/page128/noncausal computation and Fast Math's unspecified default.

`operators.mha.runtime_lengths.build_runtime_kernel(B,H,split,merge_rows,device)`
caches families independent of Q/KV extents. Split remains compile-time tuning,
selected by the unchanged frozen host model outside timing. Multi-split merge-row
tiles (1/2/4/8) match the static version; direct split1 needs no merge tile and
normalizes its family key to8. This is multiple reusable split families, not one
program with a runtime-varying split strategy. Each family's direct kernel, or
partial and merge kernels, are compiled once through the Python builder cache.

The merge's partial tensor encodes ceil(Q/128), which cannot recover the exact
Q tail. Dynamic merge therefore takes the already-prepared Q as a shape anchor;
it does not read its payload or copy it again. The static merge body is copied
unchanged into the dynamic wrapper with the extra shape argument. All arithmetic
and on-chip tile storage remain static-size. Tensor shapes choose runtime launch
grids and output sizes; the wrapper does no `.cpu()`, `.item()` or live-KV sync.
Callers retain the restricted v7 contract of positive 128-aligned live KV lengths
within valid page-table capacity; no host device-data validation is inserted.

Two smoke gates prove reuse over four Q/KV pairs (129/512,257/2048,4096/8192,
8191/4096). One forces split1; the other forces split4 and checks partial/merge
and exact output Q-tail recovery. Each gate asserts one cached family is reused
for all four shapes. Full coverage repeats the prior 56-case B1H1 Q/KV grid and
uses exactly the same selected split and merge row tile for static/dynamic.
Seeds7/19 test full-live KV and KV-128 respectively, including shuffled/repeated
physical pages and unreferenced pool pages, against the chunked CPU reference.

Each shape passes all three variants (static, dynamic, native AscendC FIA v2)
before six rotating/reversing rounds of full-call Event timing. Host family/split
selection is outside timing, runtime wrapper validation and all device-call work
are inside. Lookup/compile wall time is separately recorded; disk/JIT cache state
is not reset, so those values are not cold-build measurements. Family object
identity, cache hit/miss counts and reused shape lists are reported. Local syntax
and structural tests cannot establish NPU support for symbolic launch/output shapes
or precision/performance. No automatic default promotion occurs.
