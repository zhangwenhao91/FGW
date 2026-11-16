# Q tile128 versus64

Opt-in `build_kernel(..., block_q=64, num_splits=<explicit>)`. Default remains128.
The frozen split model was fitted with Q tiles128: baseline selects its split,
then the candidate reuses that exact number. The candidate does not rerun the
model with an altered task formula or silently select a different split. This is
a tile-size comparison, not a claim to find the best Q64 split policy.

Q64 uses 32 rows per AIV; whole-register SIMD statistics still load/store64
lanes. Statistics buffers are padded to64 lanes to avoid UB overrun. Partial LSE
copy selects the valid32-row source region for its two-AIV 64-row GM tile.
P NZ layout/packing strides follow the existing ROWS+1 formula. The Q/P/output
tile dimensions follow BR64; K/V tiles remain128. The metadata reflects these
necessary geometric changes; arithmetic, denominator clamp, manual stage IDs,
buffer version counts, GM-to-L1 V transpose, original SIMT normalization and
default Fast Math remain unchanged. Persistent cap28 and contiguous balanced
assignment remain the same. Runtime-shape experiments and other routes are not
combined with this candidate.

Run `python fgw.py mha run run_mha_q_tile64`. Return
`reports/mha/latest/mha_q_tile64_paste.txt`. Smoke checks Q1/KV2048 (small merge),
Q65/KV512 (Q64 tail), Q129/KV2048 (split and tail) and Q4096/KV8192 (direct).
Then the existing56 B1H1 BF16 D128 page128 noncausal cases and8 KV8192 boundary
shapes (Q63/65/1792/1793/3584/3585/7168/7169) are tested, seeds7/19. Six paired
rounds compare Q128/Q64/native AscendC FIA v2 full calls, with rotating/reversing
order. Both versions' query task counts, work, grid, split and maximum tasks per
program are reported. Because each Q tile64 handles half as many rows, grids
and task counts may differ; forcing equal grids would hide the intended
parallelism change. K/V reuse may worsen as the number of query tiles grows.

Local official CAModel B1H1/Q128/KV512/D128/split1 precision passed: Q128 uses
grid1, Q64 uses grid2 (two Cube plus four Vector subcores). Thus any timing gain
in that probe includes increased parallelism, not improved single-core
efficiency. The local compiler differs from the NPU machine; no hardware
performance or default promotion is inferred from this probe. Traces and
generated sources remain in workspace outputs, not Git.

An additional local CAModel split4 run passed the CPU numerical reference for
the same Q128/KV512 input: partial-grid8, merge-grid16, max absolute error about
0.001006. This exercises the padded row-statistics/LSE copy and final merge
paths; Q-tail and broader shapes remain gated by the NPU smoke/full tests.
