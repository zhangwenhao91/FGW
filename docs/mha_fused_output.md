# Fuse direct-output normalization and BF16 conversion

On main:

```bash
git pull --ff-only
python fgw.py mha run run_mha_fused_output
cat reports/mha/latest/mha_fused_output_paste.txt
```

Only Q32768, KV512/1024/2048/4096/8192, B1H1 BF16 D128 noncausal page128.
The first three cases focus on long-Q/short-KV output overhead; the latter
two are comparison cases with the existing mixed direct/split host plans.

The candidate is opt-in (`fused_output=True`, `--fused-output`). Original
raw+SIMD defaults and the abandoned row-softmax and overlap paths are unchanged
and are not combined with this experiment. Only the direct-output epilogue
changes: load the FP32 accumulator, SIMD divide by max(sum,1e-30), convert to
BF16 and pack into the existing aliased output UB. This removes the separate
SIMT normalization VF and its intermediate FP32 UB store/reload. It uses
ordinary SIMD division, not a reciprocal approximation. Each row denominator
is broadcast once and reused for its two 64-lane halves.

Packing retains the existing forward row/column order. The BF16 destination
occupies half of the FP32 accumulator storage; writes target already-consumed
or current source bytes and do not overwrite future unread values. Direct
Q tails retain the original bounded copy. Split fragments continue to write
raw FP32 A and max/sum planes without normalization. Online softmax, QK/PV,
task ownership, slot layout, merge and scheduling buffers remain unchanged.

Controls are full calls for candidate (`static`), original raw+SIMD (`raw_ref`)
and installed native FIA (`ascendc`). Candidate and reference use the same
compiled merge object when a merge is needed. Each local variant builds
once per shape. Six repeats reuse those builds; each repeat performs 100
initial warmups per control, primes Events, then nine rotated triples with
ten immediate warmups before every sample. Each control occupies each order
position three times per repeat. Seeds7/19 pass all three controls against
the CPU reference before timing; outputs are checked again after every repeat
outside Event samples. No profiling or multi-stream execution is used.

Warm-cache Event timing includes Q preparation, ordinary allocations and all
kernels. Compilation and initial task-table upload are excluded. All 54
samples/control remain in the per-shape JSON. Compact paste reports pooled
medians, per-repeat ratios/ranges/wins and per-repeat medians, without trimming
any samples. Generated sources, hashes, host logs and resource guard logs are
retained in the private run directory. Current host resource limits remain.

Local lowering compares original-default generated local source byte for byte
with the pre-change baseline, checks unchanged merge source, and verifies the
candidate uses SIMD division without a normalization SIMT VF. Official
Ascend950PR_9579 CAModel validates mixed direct/split plus Q tail and standalone
direct plus Q tail. Local compiler differs from server a7cb753d; actual NPU
precision and speedup still require the above run. No default promotion is made.

Official CAModel output checks passed: Q129/KV512 standalone direct with two
programs (max absolute output error 0.001131), and Q257/KV512 mixed direct/raw
partial/selective merge with two programs (0.001788). The latter also checks
the individual max/sum planes and normalized raw partial values. Four host
report/order tests, all five CLI configurations and Python/Bash syntax checks
passed. These checks establish local correctness, not a performance gain.
