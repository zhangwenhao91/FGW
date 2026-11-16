# Fused SIMD normalization and BF16 conversion

`build_kernel(..., output_normalization="fused")` opts into this candidate.
Original SIMT remains the default; `"simd"` retains the previous separate-SIMD
control. All three use the same contiguous manual pipeline and split model.

For split1, a SIMD row loop broadcasts max(total,1e-30), divides FP32 accumulator
vectors, converts them and stores BF16 directly into an independent UB buffer.
This avoids normalized FP32 writeback and reload. It avoids overlapping
FP32/BF16 aliases rather than relying on cross-row compiler ordering. The BF16
logical tile is16 KiB per AIV (64x128x2); actual memory reuse is determined by
the compiler. No output intermediate in GM, reciprocal approximation or layout
conversion is added.

For split>1, FP32 partial output normalization and LSE remain exactly the default
SIMT path. The final merge's divide/cast is performed by SIMD in one VF. Its
default SIMT already combines division and casting in source, so reduced
intermediate FP32 traffic is a split1 hypothesis, not a split>1 claim. The partial
precision, merge weights, output dtype and denominator clamp remain unchanged.
The same finalization principle is used for all Q lengths, with no Q threshold.

Run `python fgw.py mha run run_mha_normalization_fused`, then return
`reports/mha/latest/mha_normalization_fused_paste.txt`. Smoke covers split4
Q129/KV2048 and direct Q4096/KV8192. Full coverage repeats56 B1H1 BF16 D128
page128 noncausal configurations, two seeds and six rounds. Four variants
(SIMT/separate SIMD/fused/native FIA v2) run in rotating/reversing Event order.
The fused candidate receives the SIMT-selected split; no promotion or refit.

The local official CAModel B1H1/Q128/KV512/split1 run completed and passed
accuracy with the original seeded data and page table. Local compiler/runtime
versions differ from the NPU machine; simulator timings are not hardware
speedup evidence. Raw/Perfetto traces and local replay scripts are retained
outside Git in the workspace output directory.
