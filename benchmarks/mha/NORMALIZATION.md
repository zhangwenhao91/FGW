# Row-wise SIMD output normalization candidate

`build_kernel(..., output_normalization="simd")` opts into SIMD FP32 output
normalization in the restricted v7 Cube route. `"simt"` remains the default.
Uses the row-broadcast division organization from [PR3896](https://github.com/tile-ai/tilelang/blob/49a51226eb4068d5c0bf205cee886d2a7dcf0b05/examples/ascend/flash_attention/core.py).
The denominator remains max(total,1e-30). Division is retained (no reciprocal
approximation); the BF16 cast remains separate. QK/PV, online softmax, output
accumulation, merge weights, manual schedule, contiguous assignment and frozen
split selection remain unchanged. Partial output normalization is also changed
so split1 and split>1 use the same method; partial LSE retains its original SIMT
arithmetic in a separate block. General Cube and Vector paths are untouched.

Run `python fgw.py mha run run_mha_normalization` on the NPU machine and return
`reports/mha/latest/mha_normalization_paste.txt`. Smoke checks Q129/KV2048
(selected split4 with Q tail) and Q4096/KV8192 (direct). Full coverage repeats
the B1H1 BF16 D128 page128 noncausal 56-case Q/KV grid. Both seeds7/19 must
pass the chunked CPU reference before that shape's timing. Six rotating/reversing
three-way full-call Event rounds compare SIMT, SIMD and native AscendC FIA v2.
The SIMD candidate receives the exact SIMT-selected split, same persistent grid
and all other options. No automatic default promotion.

Local official CAModel simulation for B1H1/Q128/KV512/D128/BF16/split1/grid1
completed and passed numerical checks, with the same random seed7 and physical
page table as the baseline. The local TileLang build differs from the NPU
machine; this is evidence for a candidate, not a hardware speedup claim.
Sources, original official traces and Perfetto-normalized copies are retained in
the task workspace outside Git. The normalized copy only remaps string PID/TID
to integer identifiers, leaving instruction timestamps and durations intact.
