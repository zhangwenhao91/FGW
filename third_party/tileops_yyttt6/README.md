# Pinned external Gather reference

Source: https://github.com/yyttt6/TileOPs/blob/2ffc69b29d1e456f28ade01d50fc3a6f715d1392/src/tileops/_kernels/ascend/attention/paged_cache_gather.py

MIT license; copyright Tile-AI. The source file and license are copied without
modification. `benchmarks/mha/external_cache_gather.py` loads the original
low-level factory functions, bypassing unrelated TileOPs registry dependencies.
This is a direct kernel reuse experiment, not validation of the complete public
TileOPs Op/package installation. No framework stub or replacement kernel is used.

Run on the configured NPU machine:

```bash
python fgw.py mha run run_mha_external_gather
cat reports/mha/latest/mha_external_gather_paste.txt
```

The smoke gate covers generic/specialized Gather, FP16/BF16, zero-length requests,
causal masking and Q/KV tails. Exact gathered cache rows are checked before final
Attention tolerance checks. The full matrix contains 56 dtype/shape cases;
48 have documented compatible original page layouts for native AscendC FIA.

Comparison: original general Cube attention with automatic fill-core splits,
upstream Gather followed by the same general Cube with identical split count,
and installed native FIA v2. This does not compare the optimized restricted v7
path or promote the experimental split-cost model. Original cached K/V is input;
no KV write is needed. Gather rounds only live request lengths to original page
boundaries; the final Attention receives the original lengths. Metadata is
prepared outside warm-cache Event timing. Both gathered K/V allocations, both
Gather kernels and all Attention/split/merge work are inside timing.

Speedups are direct_us/gather_us and AscendC_us/gather_us: above 1 means the
Gather pipeline wins. MFU uses the same useful unpadded QK+PV FLOPs as the
existing benchmark. Compilation/precision failures stop before performance.
The EXIT handler writes a compact paste report even on errors. NPU execution is
required; local checks cover syntax, matrix generation, source identity and
speedup aggregation only.
