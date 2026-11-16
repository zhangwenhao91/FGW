# Isolated K/V prefetch experiment

The default remains Q128/manual, with the existing split selector. Both direct
and partial kernels accept `kv_prefetch`; the merge implementation is unchanged.
QK, softmax, PV and accumulation retain their original order and relative stages.

| Candidate | K load | QK | score copy | softmax | P copy | V load | PV | accumulate | V L1 versions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 1 | 3 | 5 | 6 | 3 | 6 | 7 | 3 |
| k_early | 0 | 2 | 4 | 6 | 7 | 4 | 7 | 8 | 3 |
| v_buffer4 | 0 | 1 | 3 | 5 | 6 | 3 | 6 | 7 | 4 |
| v_early | 0 | 1 | 3 | 5 | 6 | 2 | 6 | 7 | 4 |

Stage offsets describe overlapping iterations, not changes to the attention
formula. Advancing K by one stage is represented by shifting every other stage
forward, since K already occupies stage zero. Advancing V extends its live range
from three to four iterations; a fourth 32 KiB L1 slot avoids reusing a live slot.
`v_buffer4` isolates this capacity change: compare `v_early/v_buffer4` for timing
alone, and `v_early/baseline` for the complete candidate.

## Validation and measurement

Local official CANN CAModel, Ascend950PR_9579, B1H1 Q128/KV512/D128 BF16,
shuffled physical pages, seed7: direct output passed CPU FP32 reference
(rtol .01, atol .004), maximum absolute error .00100562 for both early-load
candidates. Default generated device C++ remains byte-identical to the previous
baseline. Local TileLang e84ce3de differs from the NPU machine's a7cb753d.

Longest simulated core spans: baseline 8.50121 us, k_early 8.55394 us,
v_early 8.41636 us. These are instruction trace spans, not device Event latency
or evidence of hardware speedup.

Both early-load partial kernels also completed official simulation at split4
(one KV block per split). Exported Partial/LSE followed by CPU state merge
passed the same reference, maximum absolute error .000462307. The local
two-launch harness stalls during merge profiling; no device merge completion
is claimed locally. Generated merge source hashes match baseline for both
candidates. The NPU smoke gate validates full partial+merge calls before the sweep.

Run on the NPU machine:

```bash
git pull --ff-only
python fgw.py mha run run_mha_kv_prefetch
cat reports/mha/latest/mha_kv_prefetch_paste.txt
```

The smoke gate precedes 56 B1H1 shapes (Q1..8192, KV512/2048/4096/8192).
All candidates reuse the baseline's frozen split, compute grid, Q tile,
normalization, transpose and math settings. Every shape passes seeds7/19
against the chunked CPU reference before timing. Five-way rotating/reversing
warm-cache Event measurement includes AscendC FIA, six rounds of 51 samples.
Full calls include Q preparation, allocations and all kernels. Source hashes,
generated C++, raw samples and failure phase are retained in a fresh run folder.
Speedup fields use control latency / candidate latency; values above one mean
the candidate is faster. No candidate is promoted automatically.
