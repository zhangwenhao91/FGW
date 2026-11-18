# Fused-output candidate: original 80-shape grid

```bash
git pull --ff-only
python fgw.py mha run run_mha_fused_output_grid
cat reports/mha/latest/mha_fused_output_grid_paste.txt
```

This retains the tested `simd_div_cast_v1` candidate unchanged and compares
it with the original raw+SIMD baseline and installed native AscendC FIA in
each process. Defaults remain unchanged. No row-softmax fusion or multi-stream
overlap experiment is enabled. The exact computation/packing change is
documented in [mha_fused_output.md](mha_fused_output.md).

All 80 shapes are B1H1 BF16 D128, noncausal, page128, full live KV:

* Q: 64,128,256,512,1024,2048,4096,8192,16384,32768.
* KV: 512,1024,2048,4096,8192,16384,24576,32768.

Each shape compiles each TL variant once and shares the compiled merge when
needed. Seeds7/19 check all three controls against the CPU reference. Six
measurement repeats reuse compilation. Each repeat warms all controls 100
times, primes Events, then runs nine rotated triples; each control occupies
each position three times. Every sample has ten immediate warmups. Accuracy
is checked again after each repeat outside Event timing. All 54 samples per
control are retained. No profiler runs during the test.

Full-call warm-cache Event timing includes Q preparation, normal allocations
and all kernels. Compilation and initial host task construction/upload are
excluded. Task plans, softmax, QK/PV, raw partial state and merge are the same
between the two TL controls. This run changes only benchmark/report tools,
not the candidate kernel. Existing host resource guards remain active.

The terminal prints the private run directory immediately, shape progress out
of 80, stage progress and each sample. Every shape is saved separately. To
continue an interrupted run from the same repository and NPU environment:

```bash
bash scripts/mha/run_mha_fused_output_grid.sh --resume reports/mha/latest/mha_fused_output_grid.XXXXXXXX
```

Only complete, validated reports with matching current benchmark/operator
source hashes, model hash and measurement settings are reused. Incomplete
cases restart compilation and all repeats; individual partial repeats are
not reused. Resume assumes the same hardware/software environment; the final
summary rejects mixed recorded device/version identities. If the environment
changed, start a fresh run. A kernel/benchmark source change also invalidates
prior case reports. Documentation-only changes do not invalidate them.

The paste reports two matrices (original/fused and AscendC/fused), geometric
means for all three pairwise comparisons, per-case latencies and repeat
ratio ranges/wins. Cases where fused latency exceeds original by more than
2% are listed separately. This threshold is for reporting, not a promotion
rule. Interpret small gains together with the six repeat ratios.

The private directory also contains raw case JSON, source dumps, resource logs,
`summary.json` (including per-repeat medians) and `summary.csv`. No slow sample
is removed. Missing or invalid cases keep the overall status incomplete.
The five-shape results do not establish full-grid gains; NPU testing is pending.
