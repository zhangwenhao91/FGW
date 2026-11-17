# Default v7 only

`operators/mha/current.py` continues to export the cleaned
`operators/mha/kernels/mha_paged_decode_ascend_v7.py`.

The default implementation retains Q128, balanced contiguous tasks capped at 28
programs, resident Q, manual stage sequence 0/1/3/5/6/3/6/7, three K/V L1
versions, original V GM-to-L1 transpose, SIMD online softmax, SIMT output
normalization, and the unchanged frozen split selector. Split1 writes final
output directly; multiple splits use the existing state merge. Q tails, causal
Q1, validation and the default Vector fallback are retained.

The builder exposes shape/device arguments plus `num_splits`, `split_policy`
and `full_kv_cache`. The public tensor API exposes `num_splits` and
`split_policy`. Scheduling, Q tile, prefetch, Fast Math, runtime-shape,
normalization, transpose, general Cube and buffer tuning options are removed
from the default API. Passing removed keywords raises TypeError instead of
silently ignoring an experiment.

The exact previous implementation is retained in
`mha_paged_decode_ascend_v7_experimental.py`. Historical tuning/profile harnesses
and the opt-in runtime-length factory explicitly import that module, so old
experiments remain reproducible. The current original-TL-vs-AscendC quick
runner (`run_mha_kv_prefetch_recompile`) imports the clean default when using
`--baseline-only`; it retains the two shapes, two builds and equal immediate
warmup. Prefetch experiments use the archive.

Offline verification, on a TileLang installation supporting Ascend:

```bash
python tools/mha/check_default_equivalence.py
```

The checker captures real PrimFuncs before constructing the NPU launch adapter,
lowers clean and archived defaults, and compares generated C++ byte for byte.
Coverage includes direct, split/merge, Q tails, multi-batch/head and causal Q1.
It also checks fallback dispatch. It does not claim new hardware accuracy or
performance measurements; the NPU runner can be rerun against the cleaned file.

Validation completed locally with TileLang e84ce3de: all six generated-code
comparisons passed, along with three fallback routes. Twelve host-side automatic
selection configurations (six Q/KV pairs, full-cache true/false, device-property
stub reporting 28 Cube cores) also matched split policy, split count, grid,
Q-tile tasks and execution mode. This host selector check does not execute an NPU.
