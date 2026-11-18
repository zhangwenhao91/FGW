# Current best raw+SIMD division path: component timing

```bash
git pull --ff-only
python fgw.py mha run run_mha_raw_breakdown
cat reports/mha/latest/mha_raw_breakdown_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
No kernel, numerical algorithm, metadata plan or workspace changes. Compile
current raw+SIMD division local/merge once; no normalized-SIMD, reciprocal or
SIMT comparison variants are compiled. Native AscendC FIA is retained.

Five controls:
- static/full: ordinary Q preparation, cached metadata lookup, allocating
  local JIT, selective merge and final output view.
- static/local: prepared Q and cached metadata, local JIT including its normal
  raw Partial/max-sum/final-output allocations. Returns a tuple, not final O.
- static/merge: same compiled merge on a fixed valid local-result fixture;
  workspace allocated/prepared before timing. Selective merged rows are
  updated in place; direct-output rows must remain valid and unchanged.
- static/chain: prepared Q, allocating local JIT followed by selective merge.
- ascendc: ordinary installed native FIA call.

All final-output controls are checked against independent chunked CPU
attention for two correctness seeds. The local fixture's correctness is
checked through the same merge/output contract. Seed7 fixtures are timed.
Record whether Q preparation aliases the input; for B1H1 permute/contiguous
can be only a view, so do not assume a device layout-conversion kernel exists.

One compilation round, six measurement repeats. Before each repeat warm every
control 100 times, rotate their order, synchronize and prime Event objects.
Ten rotated five-control groups follow; before each sample immediately warm
that same call ten times. Every control occupies every order position twice
per repeat. Sixty raw samples/control/shape; all retained. Round medians and
aggregate medians are reported. Print/save progress and each repeat as before.

These are separate warm-cache Event experiments with different buffer and
allocation contexts; their medians are not additive. local+merge need not
match chain, and full-chain/chain-local cannot isolate pure Q conversion,
launch overhead or inter-kernel device gaps. Such gaps require a hardware
trace/profiler if still material after inspecting these measurements. This
run locates likely cost concentration without making that subtraction claim.

Local checks cover argument policy, all five ordering positions, complete/
incomplete summaries and shell syntax. No kernel changes require another
compiler/simulator sweep; actual component timings require the NPU run above.
